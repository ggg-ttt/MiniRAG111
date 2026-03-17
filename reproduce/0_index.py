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
from minirag.utils import EmbeddingFunc, compute_mdhash_id, clean_text
from transformers import AutoModel, AutoTokenizer
from minirag.llm import openai_complete_if_cache

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

# 指定用于文本嵌入的模型
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# OpenAI API 配置
OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPENAI_API_KEY = "sk-a078bffba20e4153bc287ef815678981"
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "未检测到 API Key，请先设置 DASHSCOPE_API_KEY（或 OPENAI_API_KEY）。"
    )

# API调用稳定性参数（可由命令行覆盖）
API_TIMEOUT = 600.0
API_MAX_RETRIES = 5
API_RETRY_BASE_WAIT = 5


# ==================== Token Bucket 速率限制器 ====================
@dataclass
class TokenBucketLimiter:
    """
    Token Bucket 算法实现的速率限制器
    同时控制 RPM (Requests Per Minute) 和 TPM (Tokens Per Minute)
    """

    rpm: int = 600  # 每分钟最大请求数
    tpm: int = 1_000_000  # 每分钟最大token数

    # 内部状态
    _req_tokens: float = field(default=0, repr=False)
    _token_tokens: float = field(default=0, repr=False)
    _last_update: float = field(default_factory=time.time, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self):
        self._req_rate = self.rpm / 60.0  # 每秒请求数 (10 req/s)
        self._token_rate = self.tpm / 60.0  # 每秒token数 (~16666 tokens/s)
        self._req_tokens = self._req_rate  # 初始满桶
        self._token_tokens = self._token_rate

    async def acquire(self, estimated_tokens: int = 2000) -> None:
        """获取执行权限"""
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


# 全局速率限制器实例 (600 RPM, 1,000,000 TPM)
rate_limiter = TokenBucketLimiter(rpm=600, tpm=1_000_000)


async def openai_server_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
):
    """通过 OpenAI API 调用模型的包装函数（使用 Token Bucket 速率限制）"""
    global llm_stats

    if history_messages is None:
        history_messages = []

    # 从 kwargs 中获取模型名称
    keyword_extraction_flag = kwargs.pop("keyword_extraction", keyword_extraction)
    if "hashing_kv" in kwargs:
        model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    else:
        model_name = kwargs.pop("model", None)
        if not model_name:
            raise ValueError("缺少模型名：请通过 hashing_kv 或 model 参数提供")

    # 估算token数（粗略估计：输入长度 / 4 + 输出预留）
    prompt_tokens = len(prompt) // 4 if prompt else 0
    system_tokens = len(system_prompt) // 4 if system_prompt else 0
    history_tokens = sum(len(m.get("content", "")) // 4 for m in history_messages)
    estimated_input = prompt_tokens + system_tokens + history_tokens
    estimated_total = estimated_input + 4000  # 预留4000输出token

    # 记录输入 token 统计
    llm_stats["total_calls"] += 1
    llm_stats["total_input_tokens"] += estimated_input

    # 使用 Token Bucket 等待许可
    await rate_limiter.acquire(estimated_tokens=estimated_total)

    # 调用 openai_complete_if_cache（带重试机制）
    max_retries = API_MAX_RETRIES
    for attempt in range(max_retries):
        try:
            result = await openai_complete_if_cache(
                model=model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=OPENAI_API_BASE,
                api_key=OPENAI_API_KEY,
                timeout=API_TIMEOUT,
                **kwargs,
            )
            break
        except Exception as e:
            error_msg = str(e).lower()

            # 优先使用异常类型判断，可避免文案差异导致漏判（如 timed out）
            is_rate_limit = False
            if RateLimitError is not None and isinstance(e, RateLimitError):
                is_rate_limit = True

            timeout_types = [asyncio.TimeoutError, TimeoutError]
            if APITimeoutError is not None:
                timeout_types.append(APITimeoutError)
            if APIConnectionError is not None:
                timeout_types.append(APIConnectionError)

            is_timeout = isinstance(e, tuple(timeout_types))

            # 字符串兜底：覆盖不同SDK/网关的错误文案
            if not is_rate_limit:
                is_rate_limit = (
                    "rate limit" in error_msg
                    or "too many requests" in error_msg
                    or "429" in error_msg
                )
            if not is_timeout:
                is_timeout = any(
                    k in error_msg
                    for k in [
                        "timeout",
                        "timed out",
                        "time out",
                        "connection",
                        "connect",
                        "read timed out",
                    ]
                )

            if (is_rate_limit or is_timeout) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * API_RETRY_BASE_WAIT
                if is_timeout:
                    print(
                        f"Connection timeout, waiting {wait_time}s before retry {attempt + 1}/{max_retries}..."
                    )
                else:
                    print(
                        f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}..."
                    )
                await asyncio.sleep(wait_time)
            else:
                raise

    # 记录输出 token 统计（粗略估计：输出长度 / 4）
    output_tokens = len(result) // 4 if result else 0
    llm_stats["total_output_tokens"] += output_tokens

    if keyword_extraction_flag:
        from minirag.utils import locate_json_string_body_from_string

        return locate_json_string_body_from_string(result)

    return result


import argparse
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime

# ==================== 统计信息全局变量 ====================
llm_stats = {
    "total_calls": 0,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "start_time": None,
    "end_time": None,
}

try:
    from openai import APITimeoutError, APIConnectionError, RateLimitError
except Exception:
    APITimeoutError = None
    APIConnectionError = None
    RateLimitError = None


# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


# 解析命令行参数
def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG - Optimized")
    parser.add_argument("--model", type=str, default="qwen", help="指定LLM模型 (PHI/dpsk/glm/qwen)")
    parser.add_argument("--outputpath", type=str, default="./tests/qwen06b/Default_output.csv", help="输出路径")
    parser.add_argument("--workingdir", type=str, default="./tests/qwen06b", help="工作目录")
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/LiHua-World", help="数据目录")
    parser.add_argument("--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv", help="查询集路径")
    # 优化参数
    parser.add_argument("--max_workers", type=int, default=4, help="文件读取并发数 (默认: 4)")
    parser.add_argument("--batch_size", type=int, default=4, help="每批文档数 (默认: 8)")
    parser.add_argument("--llm_max_async", type=int, default=10, help="LLM最大并发数 (默认: 10)")
    parser.add_argument("--rpm", type=int, default=600, help="API RPM限制 (默认: 600)")
    parser.add_argument("--tpm", type=int, default=1000000, help="API TPM限制 (默认: 1000000)")
    parser.add_argument("--api_timeout", type=float, default=600.0, help="单次API请求超时秒数 (默认: 300)")
    parser.add_argument("--api_max_retries", type=int, default=5, help="可重试错误的最大重试次数 (默认: 5)")
    parser.add_argument("--api_retry_base_wait", type=int, default=5, help="重试基础等待秒数，按 attempt 线性递增 (默认: 5)")
    parser.add_argument("--checkpoint_every", type=int, default=20, help="每处理多少个文档保存一次checkpoint (默认: 20)")
    parser.add_argument("--continue_on_error", action="store_true", help="批次失败后降级为单文档重试并记录失败项")
    args = parser.parse_args()
    return args


# 获取命令行参数
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

# 设定各路径参数
WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath

# 更新速率限制器配置
rate_limiter.rpm = args.rpm
rate_limiter.tpm = args.tpm
rate_limiter._req_rate = args.rpm / 60.0
rate_limiter._token_rate = args.tpm / 60.0

# 更新 API 调用稳定性配置
API_TIMEOUT = args.api_timeout
API_MAX_RETRIES = args.api_max_retries
API_RETRY_BASE_WAIT = args.api_retry_base_wait

print("=" * 60)
print("MiniRAG 优化版本")
print("=" * 60)
print(f"USING LLM: {LLM_MODEL}")
print(f"USING WORKING DIR: {WORKING_DIR}")
print(f"文件读取并发: {args.max_workers}")
print(f"每批文档数: {args.batch_size}")
print(f"LLM最大并发: {args.llm_max_async}")
print(f"API速率限制: RPM={args.rpm}, TPM={args.tpm}")
print(f"API稳定性: timeout={API_TIMEOUT}s, retries={API_MAX_RETRIES}, retry_base_wait={API_RETRY_BASE_WAIT}s")
print(f"Checkpoint间隔: 每 {args.checkpoint_every} 个文档")
print(f"失败继续模式: {'开启' if args.continue_on_error else '关闭'}")
print("=" * 60)

# 如果工作目录不存在则创建
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# 预先加载分词器与嵌入模型，避免在循环中重复加载（显著提速）默认auto是自动第一个
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, device_map="auto")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
embed_model = AutoModel.from_pretrained(
    EMBEDDING_MODEL,
    device_map="auto",
    dtype=torch.float16,  # 降低显存占用
)

# 初始化MiniRAG对象（优化配置）
# 注意：enable_thinking 只支持流式调用，MiniRAG使用非流式调用，必须显式设为 False
rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=openai_server_complete,
    llm_model_max_token_size=8192,
    llm_model_name=LLM_MODEL,
    llm_model_max_async=args.llm_max_async,  # LLM最大并发数
    llm_model_kwargs={
        "extra_body": {"enable_thinking": False},  # 非流式调用必须显式禁用
    },
    embedding_batch_num=32,  # 增大embedding批次
    embedding_func_max_async=16,  # embedding并发
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


# 载入已处理文档的 doc_id 集合（来源：working_dir/kv_store_full_docs.json + kv_store_doc_status.json）
def load_processed_doc_ids(work_dir: str):
    """从 full_docs + doc_status 加载已完成文档ID集合，增强断点续跑一致性。"""
    processed_ids = set()

    full_docs_path = os.path.join(work_dir, "kv_store_full_docs.json")
    if os.path.exists(full_docs_path):
        try:
            with open(full_docs_path, "r", encoding="utf-8") as f:
                full_docs_data = json.load(f)
            if isinstance(full_docs_data, dict):
                processed_ids.update(full_docs_data.keys())
        except Exception as e:
            print(f"Warning: failed to load processed doc ids from {full_docs_path}: {e}")

    doc_status_path = os.path.join(work_dir, "kv_store_doc_status.json")
    if os.path.exists(doc_status_path):
        try:
            with open(doc_status_path, "r", encoding="utf-8") as f:
                doc_status_data = json.load(f)
            success_status = {"processed", "completed", "done", "success"}
            if isinstance(doc_status_data, dict):
                for doc_id, info in doc_status_data.items():
                    if isinstance(info, dict):
                        status = str(info.get("status", "")).lower()
                        if status in success_status:
                            processed_ids.add(doc_id)
        except Exception as e:
            print(f"Warning: failed to load processed doc ids from {doc_status_path}: {e}")

    return processed_ids


def save_checkpoint(work_dir: str, processed_count: int, total_count: int, failed_count: int) -> None:
    """保存轻量 checkpoint 统计，便于中断后快速排查进度。"""
    checkpoint_path = os.path.join(work_dir, "index_checkpoint.json")
    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processed_in_this_run": processed_count,
        "total_to_process": total_count,
        "failed_in_this_run": failed_count,
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_failed_record(work_dir: str, path: str, doc_id: str, error: str) -> None:
    """追加写入失败文档记录（JSONL），便于后续重试。"""
    failed_path = os.path.join(work_dir, "failed_docs.jsonl")
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "path": path,
        "doc_id": doc_id,
        "error": error,
    }
    with open(failed_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _normalize_path_for_resume(path: str) -> str:
    """统一路径格式，避免相对/绝对路径差异导致断点匹配失败。"""
    return os.path.abspath(path)


def load_processed_paths(work_dir: str) -> set[str]:
    """加载路径级断点状态，优先保证同一数据源可稳定续跑。"""
    resume_path = os.path.join(work_dir, "index_processed_paths.json")
    if not os.path.exists(resume_path):
        return set()

    try:
        with open(resume_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {_normalize_path_for_resume(str(p)) for p in data}
    except Exception as e:
        print(f"Warning: failed to load processed paths from {resume_path}: {e}")
    return set()


def save_processed_paths(work_dir: str, processed_paths: set[str]) -> None:
    """保存路径级断点状态。"""
    resume_path = os.path.join(work_dir, "index_processed_paths.json")
    with open(resume_path, "w", encoding="utf-8") as f:
        json.dump(sorted(processed_paths), f, ensure_ascii=False, indent=2)


processed_doc_ids = load_processed_doc_ids(WORKING_DIR)
processed_paths = load_processed_paths(WORKING_DIR)
print(f"已加载历史已处理文档ID: {len(processed_doc_ids)}")
print(f"已加载路径级断点记录: {len(processed_paths)}")


# 查找指定目录下所有txt文件
def find_txt_files(root_path):
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files


# 查找所有txt文件
WEEK_LIST = find_txt_files(DATA_PATH)

# 预过滤：只保留未处理的文件
remaining_files = []
for path in WEEK_LIST:
    normalized_path = _normalize_path_for_resume(path)
    if normalized_path in processed_paths:
        continue

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        # 与 MiniRAG 内部一致：先 clean_text 再计算 doc_id
        cleaned_content = clean_text(content)
        doc_id = compute_mdhash_id(cleaned_content, prefix="doc-")
        if doc_id not in processed_doc_ids:
            remaining_files.append({
                "path": path,
                "normalized_path": normalized_path,
                "content": cleaned_content,
                "doc_id": doc_id,
            })

print(f"共找到 {len(WEEK_LIST)} 个txt文件")
print(f"其中 {len(remaining_files)} 个待处理，{len(WEEK_LIST) - len(remaining_files)} 个已跳过")

if not remaining_files:
    print("所有文件已处理完成！")
    exit(0)


# 使用线程池并行处理文件
def process_file(item):
    """处理单个文件：返回结构化内容"""
    return item


def insert_batch_with_fallback(batch_items, continue_on_error=False):
    """先按批次插入；失败时可降级为单文档插入，返回失败列表。"""
    failed_items = []
    if not batch_items:
        return failed_items

    try:
        rag.insert([x["content"] for x in batch_items])
        success_items = list(batch_items)
        return failed_items, success_items
    except Exception as batch_error:
        if not continue_on_error:
            raise

        print(f"[WARN] 批次插入失败，降级单文档重试。batch={len(batch_items)} error={batch_error}")
        success_items = []
        for item in batch_items:
            try:
                rag.insert([item["content"]])
                success_items.append(item)
            except Exception as single_error:
                item["error"] = str(single_error)
                failed_items.append(item)
        return failed_items, success_items


buffer = []
start_time = time.time()
processed_count = 0
failed_count = 0

with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    futures = [executor.submit(process_file, item) for item in remaining_files]

    for future in tqdm(
        as_completed(futures),
        total=len(remaining_files),
        desc="处理文件",
        unit="file",
        mininterval=2.0,
    ):
        item = future.result()
        buffer.append(item)
        processed_count += 1

        if len(buffer) >= args.batch_size:
            failed_items, success_items = insert_batch_with_fallback(
                buffer,
                continue_on_error=args.continue_on_error,
            )
            for success_item in success_items:
                processed_paths.add(success_item["normalized_path"])
            save_processed_paths(WORKING_DIR, processed_paths)

            for failed_item in failed_items:
                failed_count += 1
                append_failed_record(
                    WORKING_DIR,
                    failed_item["path"],
                    failed_item["doc_id"],
                    failed_item.get("error", "unknown error"),
                )
            buffer.clear()

            # 显示进度统计
            elapsed = time.time() - start_time
            speed = processed_count / elapsed if elapsed > 0 else 0
            print(f"  [进度] 已处理 {processed_count}/{len(remaining_files)} 文件, 速度: {speed:.2f} 文件/秒")

        if args.checkpoint_every > 0 and processed_count % args.checkpoint_every == 0:
            save_checkpoint(WORKING_DIR, processed_count, len(remaining_files), failed_count)
            print(f"  [checkpoint] 已保存: {processed_count}/{len(remaining_files)}, failed={failed_count}")

# 插入剩余不足一批的内容
if buffer:
    failed_items, success_items = insert_batch_with_fallback(
        buffer,
        continue_on_error=args.continue_on_error,
    )
    for success_item in success_items:
        processed_paths.add(success_item["normalized_path"])
    save_processed_paths(WORKING_DIR, processed_paths)

    for failed_item in failed_items:
        failed_count += 1
        append_failed_record(
            WORKING_DIR,
            failed_item["path"],
            failed_item["doc_id"],
            failed_item.get("error", "unknown error"),
        )

save_checkpoint(WORKING_DIR, processed_count, len(remaining_files), failed_count)

# 最终统计
elapsed = time.time() - start_time
llm_stats["end_time"] = datetime.now()
total_tokens = llm_stats["total_input_tokens"] + llm_stats["total_output_tokens"]

# 构建统计信息字典
stats_data = {
    "timestamp": llm_stats["end_time"].strftime("%Y-%m-%d %H:%M:%S"),
    "model": LLM_MODEL,
    "total_files": len(remaining_files),
    "elapsed_seconds": round(elapsed, 2),
    "elapsed_minutes": round(elapsed / 60, 2),
    "files_per_second": round(len(remaining_files) / elapsed, 2) if elapsed > 0 else 0,
    "llm_stats": {
        "total_calls": llm_stats["total_calls"],
        "input_tokens": llm_stats["total_input_tokens"],
        "output_tokens": llm_stats["total_output_tokens"],
        "total_tokens": total_tokens,
        "avg_tokens_per_call": total_tokens // max(llm_stats["total_calls"], 1),
    },
}

# 保存统计信息到工作目录
stats_file = os.path.join(WORKING_DIR, "index_stats.json")
with open(stats_file, "w", encoding="utf-8") as f:
    json.dump(stats_data, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("处理完成！")
print(f"总文件数: {len(remaining_files)}")
print(f"总耗时: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分钟)")
print(f"平均速度: {len(remaining_files) / elapsed:.2f} 文件/秒" if elapsed > 0 else "平均速度: 0.00 文件/秒")
print(f"失败文件数: {failed_count}")
print("-" * 60)
print("LLM 调用统计:")
print(f"  总调用次数: {llm_stats['total_calls']}")
print(f"  输入 tokens: {llm_stats['total_input_tokens']:,}")
print(f"  输出 tokens: {llm_stats['total_output_tokens']:,}")
print(f"  总 tokens: {total_tokens:,}")
print(f"  平均每次调用 tokens: {total_tokens // max(llm_stats['total_calls'], 1):,}")
print("-" * 60)
print(f"统计信息已保存至: {stats_file}")
print(f"checkpoint已保存至: {os.path.join(WORKING_DIR, 'index_checkpoint.json')}")
if failed_count > 0:
    print(f"失败记录已保存至: {os.path.join(WORKING_DIR, 'failed_docs.jsonl')}")
print("=" * 60)
