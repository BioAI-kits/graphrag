import os
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse,HTMLResponse
from pydantic import BaseModel, Field
from graphrag.query.context_builder.conversation_history import ConversationHistory
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.basic_search.search import BasicSearch
from graphrag.query.structured_search.drift_search.search import DRIFTSearch
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.local_search.search import LocalSearch
from jinja2 import Template
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from pathlib import Path
import gtypes
from utils import consts
import utils
import logging
from serve import search
from serve.configs import settings
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid
import os
from dotenv import load_dotenv
# 读取环境变量
load_dotenv(dotenv_path=settings.ENV_PATH)

current_chat_model = os.getenv("CHAT_MODEL", "未选择")
if current_chat_model == "未选择":
    print("请先在 .env 文件中设置 CHAT_MODEL 变量。或者使用切换对话模型辅助+一键启动选择模型")
    exit(1)
else:
    print(f"当前选择的对话模型为: {current_chat_model}")

cors_allowed_origins = settings.cors_allowed_origins

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="GraphRAG‑OpenAI‑Proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 你的 GraphRAG 项目根目录 / Project root dir
GRAPH_ROOT = Path("../graphrag_zh").resolve()
OUTPUT_DIR = GRAPH_ROOT / "output"

basic_search: BasicSearch
local_search: LocalSearch
global_search: GlobalSearch
drift_search: DRIFTSearch
question_gen: LocalQuestionGen


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = Field(..., description="模型标识，如 graphrag-local-search:latest")
    messages: list[Message]
    stream: bool = False
    temperature: float | None = None
    # 允许透传自定义查询选项
    query_options: dict | None = None


@app.on_event("startup")
async def startup_event():
    global local_search
    global global_search
    global question_gen
    global drift_search
    global basic_search
    root = Path(settings.root).resolve()
    data_dir = Path(settings.data).resolve()
    config, data = await search.load_context(root, data_dir)
    local_search = await search.load_local_search_engine(config, data)
    global_search = await search.load_global_search_engine(config, data)
    drift_search = await search.load_drift_search_engine(config, data)
    basic_search = await search.load_basic_search_engine(config, data)


class ReferenceRequest(BaseModel):
    content: str = Field(..., description="需要提取引用信息的文本")
    model: str | None = Field(None, description="模型ID，可选，用于生成引用链接")


class ReferenceResponse(BaseModel):
    references: list[str] = Field(..., description="提取出的引用列表")
    links: str | None = Field(None, description="引用生成的HTML链接字符串")


# 列出模型 /v1/models
@app.get("/v1/models", response_model=gtypes.ModelList)
async def list_models():
    models: list[gtypes.Model] = [
        gtypes.Model(id=consts.INDEX_LOCAL, object="model", created=1644752340, owned_by="graphrag"),
        gtypes.Model(id=consts.INDEX_GLOBAL, object="model", created=1644752340, owned_by="graphrag"),
        gtypes.Model(id=consts.INDEX_DRIFT, object="model", created=1644752340, owned_by="graphrag"),
        gtypes.Model(id=consts.INDEX_BASIC, object="model", created=1644752340, owned_by="graphrag")
    ]
    return gtypes.ModelList(data=models)

@app.get("/")
async def index():
    html_file_path = os.path.join("templates", "index.html")
    with open(html_file_path, "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)


async def handle_sync_response(request, search, conversation_history):
    result = await search.search(request.messages[-1].content, conversation_history=conversation_history)
    if isinstance(search, DRIFTSearch):
        response = result.response
        response = response["nodes"][0]["answer"]
    else:
        response = result.response

    reference = utils.get_reference(response)
    if reference:
        response += f"\n{utils.generate_ref_links(reference, request.model)}"
    from openai.types.chat.chat_completion import Choice
    completion = ChatCompletion(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=response
                )
            )
        ],
        usage=CompletionUsage(
            completion_tokens=-1,
            prompt_tokens=result.prompt_tokens,
            total_tokens=-1
        )
    )
    return JSONResponse(content=jsonable_encoder(completion))

async def handle_stream_response(request, search, conversation_history):
    """
    将 search.stream_search 生成的 token 流封装成符合 OpenAI ChatCompletion
    SSE 协议的流式响应。

    Wrap search.stream_search tokens into SSE chunks fully compatible with
    OpenAI's /v1/chat/completions?stream=true format.

    Parameters
    ----------
    request : RequestLike
        包含模型名 / messages 等属性的对象。
    search : SearchEngine
        必须实现 async generator `stream_search(prompt, history) -> str token`.
    conversation_history : list
        对话上下文。
    """

    async def event_stream():
        """真正执行流写入的内嵌协程 / inner coroutine that yields SSE lines."""

        chat_id = f"chatcmpl-{uuid.uuid4().hex}"  # 与官方格式对齐 / same pattern as OpenAI
        full_response = ""  # 累积全部内容 / store full assistant text
        model_name = request.model

        # -------- 1) 发送首包 / send the very first chunk (role only) --------
        initial_chunk = ChatCompletionChunk(
            id=chat_id,
            created=int(time.time()),
            model=model_name,
            object="chat.completion.chunk",
            choices=[
                Choice(
                    index=0,  # 单路复用始终为 0 / always 0 for single choice
                    finish_reason=None,
                    delta=ChoiceDelta(role="assistant")  # 仅包含 role / role only
                )
            ]
        )
        yield f"data: {initial_chunk.model_dump_json()}\n\n"

        # -------- 2) 逐 token 输出 / emit each token as its own chunk --------
        async for token in search.stream_search(
                request.messages[-1].content, conversation_history
        ):
            content_chunk = ChatCompletionChunk(
                id=chat_id,
                created=int(time.time()),
                model=model_name,
                object="chat.completion.chunk",
                choices=[
                    Choice(
                        index=0,
                        finish_reason=None,
                        delta=ChoiceDelta(content=token)  # 仅 content / content only
                    )
                ]
            )
            yield f"data: {content_chunk.model_dump_json()}\n\n"
            full_response += token

        # -------- 3) (可选) 发送引用信息 / optional reference info --------
        reference = utils.get_reference(full_response)
        if reference:
            ref_text = utils.generate_ref_links(reference, model_name)
            for line in ref_text.splitlines():
                chunk = ChatCompletionChunk(
                    id=chat_id,
                    created=int(time.time()),
                    model=model_name,
                    object="chat.completion.chunk",
                    choices=[Choice(
                        index=0,
                        finish_reason=None,
                        delta=ChoiceDelta(content=line + "\n")  # 保留换行
                    )]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        # -------- 4) 终止包 / final stop chunk (empty delta) --------
        stop_chunk = ChatCompletionChunk(
            id=chat_id,
            created=int(time.time()),
            model=model_name,
            object="chat.completion.chunk",
            choices=[
                Choice(
                    index=0,
                    finish_reason="stop",
                    delta=ChoiceDelta()  # 空对象 / empty delta {}
                )
            ]
        )
        yield f"data: {stop_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"  # OpenAI 明确要求的结束标记 / mandatory terminator

    # 返回 FastAPI StreamingResponse / return FastAPI StreamingResponse
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: gtypes.ChatCompletionRequest):
    if not local_search or not global_search or not drift_search or not basic_search:
        logger.error("search engines is not initialized")
        raise HTTPException(status_code=500, detail="search engines is not initialized")

    try:
        history = request.messages[:-1]
        conversation_history = ConversationHistory.from_list([message.model_dump() for message in history])

        if request.model == consts.INDEX_GLOBAL:
            search_engine = global_search
        elif request.model == consts.INDEX_LOCAL:
            search_engine = local_search
        elif request.model == consts.INDEX_DRIFT:
            search_engine = drift_search
        else:
            search_engine = basic_search

        if not request.stream:
            return await handle_sync_response(request, search_engine, conversation_history)
        else:
            return await handle_stream_response(request, search_engine, conversation_history)
    except Exception as e:
        html_file_path = os.path.join("./templates", f"error.html")
        with open(html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        template = Template(html_content)
        html_content = template.render()
        return HTMLResponse(
            content=html_content,
            status_code=404
        )

@app.get("/v1/references/{index_id}/{datatype}/{id}", response_class=HTMLResponse)
async def get_reference(index_id: str, datatype: str, id: int):
    if not os.path.exists(settings.data):
        raise HTTPException(status_code=404, detail=f"{index_id} not found")
    if datatype not in ["entities", "claims", "sources", "reports", "relationships"]:
        raise HTTPException(status_code=404, detail=f"{datatype} not found")
    try:
        data = await search.get_index_data(settings.data, datatype, id)
        html_file_path = os.path.join("./templates", f"{datatype}_template.html")
        with open(html_file_path, 'r',encoding='utf-8') as file:
            html_content = file.read()
        template = Template(html_content)
        html_content = template.render(data=data)
        return HTMLResponse(content=html_content)

    except Exception as e:
        html_file_path = os.path.join("./templates", f"error.html")
        with open(html_file_path, 'r',encoding='utf-8') as file:
            html_content = file.read()
        template = Template(html_content)
        html_content = template.render()
        return HTMLResponse(
            content=html_content,
            status_code=404
        )

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
