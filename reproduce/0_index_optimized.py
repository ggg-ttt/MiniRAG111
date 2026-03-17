# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import sys
import os
import logging
import json

# 控制 OpenAI 客户端的日志级别
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 将上级目录加入sys.path，方便导入minirag包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入MiniRAG相关模块和函数
from minirag import MiniRAG
from minirag.llm import hf_embed
from minirag.utils import EmbeddingFunc, compute_mdhash_id
from transformers import AutoModel, AutoTokenizer
from minirag.llm import openai_complete_if_cache

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import torch

# 指定用于文本嵌入的模型
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# OpenAI API 配置
OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENAI_API_KEY = "sk-5f75d189476d4055aae52c1553e74dae"


# ==================== Token Bucket 速率限制器 ====================
@dataclass
class TokenBucketLimiter:
    """
    Token Bucket 算法实现的速率限制器
    同时控制 RPM (Requests Per Minute) 和 TPM (Tokens Per Minute)
    """
    rpm: int = 600          # 每分钟最大请求数
    tpm: int = 1_000_000    # 每分钟最大token数

    # 内部状态
    _req_tokens: float = field(default=0, repr=False)
    _token_tokens: float = field(default=0, repr=False)
    _last_update: float = field(default_factory=time.time, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self):
        self._req_rate = self.rpm / 60.0      # 每秒请求数
        self._token_rate = self.tpm / 60.0    # 每秒token数
        self._req_tokens = self._req_rate     # 初始满桶
        self._token_tokens = self._token_rate

    async def acquire(self, estimated_tokens: int = 2000) -> None:
        """
        获取执行权限

        Args:
            estimated_tokens: 估计本次请求需要的token数（输入+输出）
        """
        async with self._lock:
            while True:
                now = time.time()
                elapsed = now - self._last_update
                self._last_update = now

                # 补充token
                self._req_tokens = min(self.rpm, self._req_tokens + self._req_rate * elapsed)
                self._token_tokens = min(self.tpm, self._token_tokens + self._token_rate * elapsed)

                # 检查是否满足条件
                if self._req_tokens >= 1 and self._token_tokens >= estimated_tokens:
                    self._req_tokens -= 1
                    self._token_tokens -= estimated_tokens
                    return

                # 计算需要等待的时间
                wait_time = 0
                if self._req_tokens < 1:
                    wait_time = max(wait_time, (1 - self._req_tokens) / self._req_rate)
                if self._token_tokens < estimated_tokens:
                    wait_time = max(wait_time, (estimated_tokens - self._token_tokens) / self._token_rate)

                # 释放锁，等待，然后重试
                await asyncio.sleep(max(0.001, wait_time))


# 全局速率限制器实例
rate_limiter = TokenBucketLimiter(rpm=600, tpm=1_000_000)


async def openai_server_complete(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
    """
    通过 OpenAI API 调用模型的包装函数
    使用 Token Bucket 算法精确控制速率
    """
    # 从 kwargs 中获取模型名称
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]

    # 估算token数（粗略估计：输入长度 / 4 + 输出预留）
    prompt_tokens = len(prompt) // 4 if prompt else 0
    system_tokens = len(system_prompt) // 4 if system_prompt else 0
    history_tokens = sum(len(m.get("content", "")) // 4 for m in history_messages)
    estimated_input = prompt_tokens + system_tokens + history_tokens
    estimated_total = estimated_input + 4000  # 预留4000输出token

    # 使用 Token Bucket 等待许可
    await rate_limiter.acquire(estimated_tokens=estimated_total)

    # 调用 openai_complete_if_cache（带重试机制）
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = await openai_complete_if_cache(
                model=model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=OPENAI_API_BASE,
                api_key=OPENAI_API_KEY,
                timeout=300.0,
                **kwargs
            )
            break
        except Exception as e:
            error_msg = str(e).lower()
            is_rate_limit = "rate limit" in error_msg or "too many requests" in error_msg
            is_timeout = "timeout" in error_msg or "connect" in error_msg

            if (is_rate_limit or is_timeout) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                if is_timeout:
                    print(f"Connection timeout, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                else:
                    print(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                await asyncio.sleep(wait_time)
            else:
                raise

    if keyword_extraction:
        from minirag.utils import locate_json_string_body_from_string
        return locate_json_string_body_from_string(result)

    return result


# ==================== 参数解析 ====================
def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG - Optimized Version")
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--outputpath", type=str, default="./tests/qwen06b/Default_output.csv")
    parser.add_argument("--workingdir", type=str, default="./tests/qwen06b")
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/LiHua-World")
    parser.add_argument("--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv")
    # 新增优化参数
    parser.add_argument("--max_workers", type=int, default=4, help="文件读取并发数 (默认: 4)")
    parser.add_argument("--batch_size", type=int, default=16, help="每批文档数 (默认: 16)")
    parser.add_argument("--llm_max_async", type=int, default=10, help="LLM最大并发数 (默认: 10)")
    args = parser.parse_args()
    return args


args = get_args()

# 根据参数选择不同的LLM模型
if args.model == "PHI":
    LLM_MODEL = "microsoft/Phi-3.5-mini-instruct"
elif args.model == "dpsk":
    LLM_MODEL = "Pro/deepseek-ai/DeepSeek-V3.2"
elif args.model == "glm":
    LLM_MODEL = "glm-4.5-air"
elif args.model == "qwen":
    LLM_MODEL = "qwen3-0.6b"
else:
    print("Invalid model name")
    exit(1)

WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath

print("=" * 60)
print("MiniRAG 优化版本")
print("=" * 60)
print(f"USING LLM: {LLM_MODEL}")
print(f"USING WORKING DIR: {WORKING_DIR}")
print(f"文件读取并发: {args.max_workers}")
print(f"每批文档数: {args.batch_size}")
print(f"LLM最大并发: {args.llm_max_async}")
print(f"API速率限制: RPM={rate_limiter.rpm}, TPM={rate_limiter.tpm}")
print("=" * 60)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# 预先加载分词器与嵌入模型
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, device_map="auto")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
embed_model = AutoModel.from_pretrained(
    EMBEDDING_MODEL,
    device_map="auto",
    dtype=torch.float16,
)

# 初始化MiniRAG对象（优化配置）
rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=openai_server_complete,
    llm_model_max_token_size=8192,
    llm_model_name=LLM_MODEL,
    llm_model_max_async=args.llm_max_async,  # 使用配置的最大并发
    embedding_batch_num=32,  # 增大embedding批次
    embedding_func_max_async=16,  # 增大embedding并发
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=tokenizer,
            embed_model=embed_model,
        ),
    ),
)


# ==================== 文档处理 ====================
def load_processed_doc_ids(work_dir: str):
    """加载已处理的文档ID集合"""
    kv_path = os.path.join(work_dir, "kv_store_full_docs.json")
    if not os.path.exists(kv_path):
        return set()
    try:
        with open(kv_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.keys())
    except Exception as e:
        print(f"Warning: failed to load processed doc ids from {kv_path}: {e}")
        return set()


def find_txt_files(root_path):
    """查找指定目录下所有txt文件"""
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files


def load_txt_file(path: str):
    """加载文本文件"""
    with open(path, "r", encoding="utf-8") as f:
        return path, f.read()


processed_doc_ids = load_processed_doc_ids(WORKING_DIR)
WEEK_LIST = find_txt_files(DATA_PATH)
print(f"共找到 {len(WEEK_LIST)} 个txt文件")

# 过滤已处理的文件
remaining_files = []
for path in WEEK_LIST:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        doc_id = compute_mdhash_id(content, prefix="doc-")
        if doc_id not in processed_doc_ids:
            remaining_files.append(path)

print(f"其中 {len(remaining_files)} 个待处理，{len(WEEK_LIST) - len(remaining_files)} 个已跳过")

if not remaining_files:
    print("所有文件已处理完成！")
    exit(0)

# 使用线程池并行读取文件
max_workers = args.max_workers
BATCH_SIZE = args.batch_size
buffer = []

start_time = time.time()
processed_count = 0

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(load_txt_file, path) for path in remaining_files]

    for future in tqdm(
        as_completed(futures),
        total=len(remaining_files),
        desc="读取并插入",
        unit="file",
        mininterval=2.0,
    ):
        file_path, content = future.result()
        buffer.append(content)
        processed_count += 1

        if len(buffer) >= BATCH_SIZE:
            rag.insert(buffer)
            buffer.clear()

            # 显示进度统计
            elapsed = time.time() - start_time
            speed = processed_count / elapsed if elapsed > 0 else 0
            print(f"  [进度] 已处理 {processed_count}/{len(remaining_files)} 文件, "
                  f"速度: {speed:.2f} 文件/秒")

# 插入剩余不足一批的内容
if buffer:
    rag.insert(buffer)

# 最终统计
elapsed = time.time() - start_time
print("\n" + "=" * 60)
print("处理完成！")
print(f"总文件数: {len(remaining_files)}")
print(f"总耗时: {elapsed:.2f} 秒")
print(f"平均速度: {len(remaining_files) / elapsed:.2f} 文件/秒")
print("=" * 60)
