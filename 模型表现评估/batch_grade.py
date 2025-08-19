import glob
import os
import re
import sys
import threading
import time
from openai import AsyncOpenAI
from format_to_json import parse_openai_arguments
import asyncio
import pandas as pd
import json
from prompt import PROMPT, Tools
from collections import deque
import openai
from tenacity import retry, wait_random_exponential, retry_if_exception_type
import tiktoken
from pathlib import Path
from dotenv import load_dotenv

save_dir = f'grade_label'
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
# 1. 读取模型额度 / Read limits from env
api_key = 'sk-3fTk1RRVs2uyBJR2fuFikScsnJ4vEAV0QsHeXOsaLONG8ZXX'


def build_user_prompt(comment):
    user_prompt = f"""
    现在要分析的是:\n
    问题:{comment['question']}\n
    模型给出的答案是:\n
    {comment['answer']}\n"""

    return user_prompt


@retry(wait=wait_random_exponential(min=1, max=60),
       retry=retry_if_exception_type(openai.RateLimitError))
async def async_query_openai(query, semaphore):
    async with semaphore:
        client = AsyncOpenAI(
            base_url='https://api.key77qiqi.com/v1',
            api_key=api_key,
        )
        content = query["content"]
        messages = [
            {"role": "system",
             "content": PROMPT},
            {"role": "user", "content": content},
        ]
        print(messages)
        res = await client.chat.completions.create(
            model="gemini-2.5-pro",
            messages=messages,
            tools=Tools,
        )
        try:
            response = parse_openai_arguments(res.choices[0].message.tool_calls[0].function.arguments)[-1]
            query['response'] = response
            return query
        except Exception as e:
            print(e)
            return {}


async def async_process_queries(queries, concurrency_limit=100):
    semaphore = asyncio.Semaphore(concurrency_limit)
    results = await asyncio.gather(*(async_query_openai(query, semaphore) for query in queries))
    return results


async def main():
    all_queries = []
    jsonl_paths = glob.glob('Answers/*/all_results.jsonl')
    if os.path.exists(f'{save_dir}/all_results.jsonl'):
        df = pd.read_json(f'{save_dir}/all_results.jsonl', lines=True)
        id_models = df[['id', 'model_name']].values.tolist()
        id_models = [(int(id_), model_name) for id_, model_name in id_models]
        print(f"Loaded {len(id_models)} existing queries from {save_dir}/all_results.jsonl")
    else:
        id_models = []
    for jsonl_path in jsonl_paths:
        model_name = os.path.basename(os.path.dirname(jsonl_path))
        df = pd.read_json(jsonl_path, lines=True)
        df['model_name'] = model_name
        for index, row in df.iterrows():
            content_id = row['id']
            if (content_id, model_name) in id_models:
                print(f"Skipping existing query for id {content_id} with model {model_name}")
                continue
            # id在1-100是简单 题，101-200是中等，201-300是困难
            if content_id <= 100:
                difficulty = 'easy'
            elif content_id <= 200:
                difficulty = 'medium'
            else:
                difficulty = 'hard'
            content = build_user_prompt(row.to_dict())
            all_queries.append({"id": content_id, "content": content,"difficulty": difficulty, "model_name": model_name})
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
            with open(f'{save_dir}/all_results.jsonl', 'a', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        page += 1


if __name__ == '__main__':
    # 运行主函数
    asyncio.run(main())
