import glob
import os
import sys
import threading
import time
from openai import AsyncOpenAI
import asyncio
import pandas as pd
import json
from collections import deque
import openai
from tenacity import retry, wait_random_exponential, retry_if_exception_type
import tiktoken
from pathlib import Path
from dotenv import load_dotenv
env_path = Path("../graphrag_zh/.env")
load_dotenv(dotenv_path=env_path)
chat_model = os.getenv("CHAT_MODEL", "未选择")  # 默认模型

if chat_model == "未选择":
    print("未选择模型，请在 .env 文件中设置 CHAT_MODEL")
    sys.exit(1)
chat_model = chat_model.replace("/", "@")  # 替换斜杠为下划线，避免文件名问题
save_dir = f'{chat_model}_label_jsons'
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
# 1. 读取模型额度 / Read limits from env
MAX_TPM = int(os.getenv("OPENAI_TPM", 60000))  # 单位：tokens per minute
MAX_RPM = int(os.getenv("OPENAI_RPM", 1200))  # 单位：requests per minute
api_key = os.getenv("OPENAI_API_KEY", 'xxxx')

# 2. 创建限流器 / Build a simple sliding‑window limiter
class OpenAiRateLimiter:
    """滑动窗口限流 / Sliding‑window rate limiter"""

    def __init__(self, max_tpm: int, max_rpm: int):
        self.max_tpm = max_tpm
        self.max_rpm = max_rpm
        self.req_ts = deque()  # 每次请求的时间戳
        self.token_ts = deque()  # (tokens, 时间戳)
        self.lock = threading.Lock()

    def acquire(self, tokens: int):
        """阻塞直到可以发送 / Block until the request is allowed"""
        while True:
            with self.lock:
                now = time.time()
                # 清理 60s之外的记录 / Drop records older than 60s
                while self.req_ts and now - self.req_ts[0] > 60:
                    self.req_ts.popleft()
                while self.token_ts and now - self.token_ts[0][1] > 60:
                    self.token_ts.popleft()

                # 计算当前用量 / Current usage
                cur_rpm = len(self.req_ts)
                cur_tpm = sum(t for t, _ in self.token_ts)

                # 判断是否超限 / Check quota left
                if cur_rpm < self.max_rpm and (cur_tpm + tokens) < self.max_tpm:
                    # 记录本次消耗 / Record this request
                    self.req_ts.append(now)
                    self.token_ts.append((tokens, now))
                    return  # 通过
            # 若超限，等待 0.5s 再试 / Wait and retry
            time.sleep(0.5)


# 3. 初始化 / Init
limiter = OpenAiRateLimiter(MAX_TPM, MAX_RPM)

# 4. 简易 token 估算器 / Rough token counter
enc = tiktoken.get_encoding("o200k_base")


def count_tokens(msgs):
    return sum(len(enc.encode(m["content"])) for m in msgs)

@retry(wait=wait_random_exponential(min=1, max=60),
       retry=retry_if_exception_type(openai.RateLimitError))
async def async_query_openai(query, semaphore):
    async with semaphore:
        client = AsyncOpenAI(
            base_url='http://localhost:8000/v1',
            api_key=api_key,
        )
        question = query["question"]
        messages = [{"role": "user", "content": question},]
        tokens = count_tokens(messages)
        limiter.acquire(tokens)  # 主动节流
        res = await client.chat.completions.create(
            model="graphrag-local-search",
            messages=messages
        )
        try:
            response = res.choices[0].message.content
            dic = {}
            dic['id'] = query["id"]
            dic['answer'] = response
            dic['question'] = question
            return dic
        except Exception as e:
            print(e)
            return {}


async def async_process_queries(queries, concurrency_limit=100):
    semaphore = asyncio.Semaphore(concurrency_limit)
    results = await asyncio.gather(*(async_query_openai(query, semaphore) for query in queries))
    return results


async def main():
    xlsx_path = glob.glob("Questions/*.xlsx")
    all_queries = []
    for xlsx in xlsx_path:
        print(f"Processing {xlsx}")
        df = pd.read_excel(xlsx)
        for index, row in df.iterrows():
            all_queries.append(row.to_dict())
    if os.path.exists(f'{save_dir}/all_results.jsonl'):
        ids = pd.read_json(f'{save_dir}/all_results.jsonl', lines=True)['id'].tolist()
    else:
        ids = []
    all_queries = [q for q in all_queries if q['id'] not in ids]
    print(f"Total queries to process: {len(all_queries)}")
    # 翻页
    page = 1
    # 每页数量
    page_size = 20
    while True:
        queries = all_queries[(page - 1) * page_size:page * page_size]
        print(f"Page {page}, {len(queries)} queries")
        if not queries:
            break
        results = await async_process_queries(queries)
        for result in results:
            if not result:
                continue
            print(result)
            # 保存jsonl
            with open(f'{save_dir}/all_results.jsonl', 'a', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        page += 1


if __name__ == '__main__':

    # 运行主函数
    asyncio.run(main())
