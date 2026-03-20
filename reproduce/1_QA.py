# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import asyncio
import csv
import os
import sys
import warnings
import logging
import argparse
import time

from tqdm import tqdm

# 抑制transformers警告
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

# 将上级目录加入sys.path，方便导入minirag包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 增加CSV字段大小限制，防止大字段写入时出现"field larger than field limit"错误
max_field_size = sys.maxsize if sys.maxsize > 0 else 100 * 1024 * 1024
csv.field_size_limit(max_field_size)

from minirag import MiniRAG, QueryParam
from minirag.llm import hf_embed, openai_complete_if_cache
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

# 指定用于文本嵌入的模型
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="dpsk")
    parser.add_argument("--outputpath", type=str, default="./tests/dpsk_mini/dpsk_light_output.csv")
    parser.add_argument("--workingdir", type=str, default="./tests/dpsk_mini")
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/LiHua-World/")
    parser.add_argument("--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv")
    parser.add_argument("--mode", type=str, default="mini", choices=["naive", "light", "mini"])

    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="断点续跑：跳过输出文件里已经有结果的行，仅补全空结果行",
    )
    # parser.add_argument(
    #     "--no-resume",
    #     action="store_false",
    #     dest="resume",
    #     help="关闭断点续跑：即使已有结果也重新计算",
    # )

    # 并发与稳定性参数
    parser.add_argument("--query_concurrency", type=int, default=4, help="外层问题并发数")
    parser.add_argument("--llm_max_async", type=int, default=8, help="MiniRAG内部LLM最大并发")
    parser.add_argument("--embedding_max_async", type=int, default=16, help="MiniRAG内部embedding最大并发")
    parser.add_argument("--save_every", type=int, default=50, help="每处理多少条就落盘一次")
    parser.add_argument("--api_timeout", type=float, default=180.0, help="单次LLM请求超时秒数")

    # 实时吞吐统计参数
    parser.add_argument("--stats_interval_sec", type=int, default=60, help="实时统计打印间隔（秒）")
    parser.add_argument("--stall_warn_sec", type=int, default=300, help="超过该秒数无完成请求时打印告警")
    parser.add_argument("--question_timeout_sec", type=float, default=300.0, help="单个问题整体超时秒数，0 表示不限制")

    # 自动升档参数（仅调整外层query并发）
    parser.set_defaults(auto_scale_query=True)
    parser.add_argument("--auto_scale_query", action="store_true", dest="auto_scale_query", help="开启外层query并发自动升档")
    parser.add_argument("--no_auto_scale_query", action="store_false", dest="auto_scale_query", help="关闭外层query并发自动升档")
    parser.add_argument("--auto_scale_start_query", type=int, default=2, help="自动升档起始query并发")
    parser.add_argument("--auto_scale_step_query", type=int, default=1, help="每次升档增加的query并发")
    parser.add_argument("--auto_scale_stable_windows", type=int, default=2, help="连续稳定窗口数达到后触发升档")
    parser.add_argument("--auto_scale_min_completed", type=int, default=1, help="判定稳定窗口所需最少完成请求数")
    parser.add_argument("--auto_scale_min_success_rate", type=float, default=95.0, help="判定稳定窗口所需最小成功率(%)")

    return parser.parse_args()


args = get_args()

if args.model == "PHI":
    LLM_MODEL = "microsoft/Phi-4-mini-instruct"
elif args.model == "dpsk":
    LLM_MODEL = "deepseek-v3.2"
elif args.model == "MiniCPM":
    LLM_MODEL = "openbmb/MiniCPM3-4B"
elif args.model == "qwen":
    LLM_MODEL = "qwen3-1.7b"
else:
    print("Invalid model name")
    raise SystemExit(1)

WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath

print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)
print("MODE:", args.mode)
print("并发配置: query_concurrency=", args.query_concurrency,
      "llm_max_async=", args.llm_max_async,
      "embedding_max_async=", args.embedding_max_async)
print("实时统计: interval=", args.stats_interval_sec, "sec")
if args.auto_scale_query:
    print(
        "自动升档: start_query=", args.auto_scale_start_query,
        "step=", args.auto_scale_step_query,
        "stable_windows=", args.auto_scale_stable_windows,
        "max_query=", args.query_concurrency,
    )

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# vLLM Server 配置
VLLM_SERVER_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VLLM_API_KEY = "sk-4146362dee444d49832d571b3a22cac7"


class ThroughputStats:
    """线程安全的请求吞吐统计器（异步环境）。"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._global_count = 0
        self._global_success = 0
        self._global_fail = 0
        self._global_latency_sum = 0.0

        self._window_count = 0
        self._window_success = 0
        self._window_fail = 0
        self._window_latency_sum = 0.0

    async def record(self, latency_sec: float, success: bool):
        async with self._lock:
            self._global_count += 1
            self._global_latency_sum += latency_sec
            self._window_count += 1
            self._window_latency_sum += latency_sec
            if success:
                self._global_success += 1
                self._window_success += 1
            else:
                self._global_fail += 1
                self._window_fail += 1

    async def take_window(self):
        async with self._lock:
            snapshot = {
                "count": self._window_count,
                "success": self._window_success,
                "fail": self._window_fail,
                "latency_sum": self._window_latency_sum,
            }
            self._window_count = 0
            self._window_success = 0
            self._window_fail = 0
            self._window_latency_sum = 0.0
            return snapshot

    async def summary(self):
        async with self._lock:
            return {
                "count": self._global_count,
                "success": self._global_success,
                "fail": self._global_fail,
                "latency_sum": self._global_latency_sum,
            }


class TokenStats:
    """线程安全的Token使用统计器（异步环境）。"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0

    async def record(self, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0):
        async with self._lock:
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            if total_tokens > 0:
                self._total_tokens += total_tokens
            else:
                self._total_tokens += prompt_tokens + completion_tokens

    async def summary(self):
        async with self._lock:
            return {
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._total_tokens,
            }


class ErrorStats:
    """线程安全的失败类型统计器（异步环境）。"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._global = {}
        self._window = {}

    async def record(self, reason: str):
        key = str(reason or "unknown").lower()
        async with self._lock:
            self._global[key] = self._global.get(key, 0) + 1
            self._window[key] = self._window.get(key, 0) + 1

    async def take_window(self):
        async with self._lock:
            snap = dict(self._window)
            self._window = {}
            return snap

    async def summary(self):
        async with self._lock:
            return dict(self._global)


class QueryProgress:
    """跟踪任务进展，用于长时间无进展告警。"""

    def __init__(self, total_tasks: int):
        self._lock = asyncio.Lock()
        self._total_tasks = total_tasks
        self._completed_tasks = 0
        self._inflight_tasks = 0
        self._started_at = time.perf_counter()
        self._last_completed_at = self._started_at

    async def mark_start(self):
        async with self._lock:
            self._inflight_tasks += 1

    async def mark_done(self):
        async with self._lock:
            self._inflight_tasks = max(0, self._inflight_tasks - 1)
            self._completed_tasks += 1
            self._last_completed_at = time.perf_counter()

    async def snapshot(self):
        async with self._lock:
            now = time.perf_counter()
            return {
                "total": self._total_tasks,
                "completed": self._completed_tasks,
                "inflight": self._inflight_tasks,
                "pending": max(0, self._total_tasks - self._completed_tasks),
                "idle_sec": max(0.0, now - self._last_completed_at),
                "elapsed_sec": max(0.0, now - self._started_at),
            }


class AdaptiveLimiter:
    """可动态调整上限的异步并发限流器。"""

    def __init__(self, initial_limit: int):
        self._limit = max(1, int(initial_limit))
        self._inflight = 0
        self._cond = asyncio.Condition()

    async def acquire(self):
        async with self._cond:
            while self._inflight >= self._limit:
                await self._cond.wait()
            self._inflight += 1

    async def release(self):
        async with self._cond:
            if self._inflight > 0:
                self._inflight -= 1
            self._cond.notify_all()

    async def resize(self, new_limit: int):
        async with self._cond:
            self._limit = max(1, int(new_limit))
            self._cond.notify_all()

    async def snapshot(self):
        async with self._cond:
            return {
                "limit": self._limit,
                "inflight": self._inflight,
            }


# 全局Token统计器
GLOBAL_TOKEN_STATS = TokenStats()
# 全局失败原因统计器
GLOBAL_ERROR_STATS = ErrorStats()


async def stats_reporter(
    stats: ThroughputStats,
    progress: QueryProgress,
    limiter: AdaptiveLimiter,
    stop_event: asyncio.Event,
    interval_sec: int,
    stall_warn_sec: int,
    enable_auto_scale: bool,
    max_query_concurrency: int,
    scale_step: int,
    stable_windows_needed: int,
    min_completed_per_window: int,
    min_success_rate: float,
):
    alerted_at_completed = -1
    stable_windows = 0

    while True:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
            break
        except asyncio.TimeoutError:
            window = await stats.take_window()
            error_window = await GLOBAL_ERROR_STATS.take_window()
            progress_snapshot = await progress.snapshot()
            limiter_snapshot = await limiter.snapshot()
            warned_this_window = False

            # 若长时间没有任何任务完成，打印一次告警；任务完成后会自动解除本轮告警状态
            if (
                progress_snapshot["pending"] > 0
                and progress_snapshot["idle_sec"] >= stall_warn_sec
                and progress_snapshot["completed"] != alerted_at_completed
            ):
                print(
                    f"[warn] {progress_snapshot['idle_sec']:.1f}s 无新完成请求，"
                    f"inflight={progress_snapshot['inflight']}, "
                    f"completed={progress_snapshot['completed']}/{progress_snapshot['total']}, "
                    f"limit={limiter_snapshot['limit']}"
                )
                alerted_at_completed = progress_snapshot["completed"]
                warned_this_window = True
                stable_windows = 0

            if window["count"] == 0:
                print(f"[stats] 最近窗口无完成请求 (limit={limiter_snapshot['limit']})")
                stable_windows = 0
                continue

            if progress_snapshot["completed"] != alerted_at_completed:
                alerted_at_completed = -1

            rpm = window["count"] * 60.0 / interval_sec
            avg_latency = window["latency_sum"] / window["count"]
            success_rate = window["success"] * 100.0 / window["count"]
            fail_detail = ""
            if error_window:
                detail = ", ".join(f"{k}:{v}" for k, v in sorted(error_window.items()))
                fail_detail = f", fail_detail={{ {detail} }}"

            print(
                f"[stats] 最近{interval_sec}s: "
                f"RPM={rpm:.2f}, "
                f"AvgLatency={avg_latency:.2f}s, "
                f"SuccessRate={success_rate:.2f}% "
                f"(ok={window['success']}, fail={window['fail']}), "
                f"limit={limiter_snapshot['limit']}"
                f"{fail_detail}"
            )

            if not enable_auto_scale:
                continue
            if progress_snapshot["pending"] <= 0:
                continue
            if warned_this_window:
                continue

            is_stable = (
                window["count"] >= min_completed_per_window
                and success_rate >= min_success_rate
                and window["fail"] == 0
            )

            if is_stable:
                stable_windows += 1
            else:
                stable_windows = 0

            if stable_windows >= stable_windows_needed and limiter_snapshot["limit"] < max_query_concurrency:
                new_limit = min(max_query_concurrency, limiter_snapshot["limit"] + max(1, scale_step))
                await limiter.resize(new_limit)
                stable_windows = 0
                print(
                    f"[autoscale] 升档 query_concurrency: "
                    f"{limiter_snapshot['limit']} -> {new_limit}"
                )


async def vllm_server_complete(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs):
    """通过 vLLM server 调用模型的包装函数"""
    import openai
    if history_messages is None:
        history_messages = []

    keyword_extraction_flag = kwargs.pop("keyword_extraction", keyword_extraction)
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    api_key = VLLM_API_KEY if VLLM_API_KEY else "dummy"

    default_params = {
        "max_tokens": 200,
        "temperature": 0.3,
        "top_p": 0.8,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None,
        "timeout": args.api_timeout,
    }

    merged_params = {**default_params, **kwargs}
    supported_params = {
        "max_tokens", "temperature", "top_p", "frequency_penalty",
        "presence_penalty", "stop", "stream", "logprobs", "top_logprobs", "timeout"
    }
    filtered_params = {k: v for k, v in merged_params.items() if k in supported_params}

    # 创建OpenAI客户端直接调用，以获取token使用信息
    client_api_key = api_key or os.environ.get("OPENAI_API_KEY", "dummy")
    timeout = filtered_params.get("timeout", 60.0)
    openai_async_client = openai.AsyncOpenAI(
        base_url=VLLM_SERVER_BASE_URL,
        api_key=client_api_key,
        timeout=timeout
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = await openai_async_client.chat.completions.create(
        model=model_name,
        messages=messages,
        **filtered_params,
    )

    # 提取内容
    content = ""
    if response and hasattr(response, "choices") and response.choices:
        content = response.choices[0].message.content or ""

    # 记录token使用情况
    if response and hasattr(response, "usage") and response.usage:
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0
        await GLOBAL_TOKEN_STATS.record(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )

    if keyword_extraction_flag:
        from minirag.utils import locate_json_string_body_from_string
        return locate_json_string_body_from_string(content)

    return content


# 关键优化：embedding模型只加载一次，避免每题重复初始化
EMBED_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
EMBED_MODEL = AutoModel.from_pretrained(EMBEDDING_MODEL)

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=vllm_server_complete,
    llm_model_max_token_size=8192,
    llm_model_name=LLM_MODEL,
    llm_model_max_async=args.llm_max_async,
    embedding_func_max_async=args.embedding_max_async,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=EMBED_TOKENIZER,
            embed_model=EMBED_MODEL,
        ),
    ),
)

QUESTION_LIST = []
GA_LIST = []
with open(QUERY_PATH, mode="r", encoding="utf-8") as question_file:
    reader = csv.DictReader(question_file)
    for row in reader:
        QUESTION_LIST.append(row["Question"])
        GA_LIST.append(row["Gold Answer"])


def build_query_param(mode: str) -> QueryParam:
    return QueryParam(
        mode=mode,
        max_token_for_text_unit=2000,
        max_token_for_global_context=2000,
        max_token_for_local_context=2000,
        max_token_for_node_context=500,
    )


def persist_rows(output_path, fieldnames, rows):
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(output_path, result_column):
    fieldnames = ["Question", "Gold Answer", result_column]
    existing_rows = []

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        with open(output_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or fieldnames)
            if result_column not in fieldnames:
                fieldnames.append(result_column)
            for row in reader:
                existing_rows.append(row)

    by_question = {row.get("Question", ""): row for row in existing_rows if row.get("Question")}
    merged_rows = []

    for i, question in enumerate(QUESTION_LIST):
        row = by_question.get(question, {
            "Question": question,
            "Gold Answer": GA_LIST[i],
            result_column: "",
        })
        row["Question"] = question
        row["Gold Answer"] = GA_LIST[i]
        if result_column not in row:
            row[result_column] = ""
        merged_rows.append(row)

    return fieldnames, merged_rows


async def aquery_one(question: str, mode: str):
    started = time.perf_counter()
    timeout = args.question_timeout_sec if args.question_timeout_sec > 0 else None
    try:
        coro = rag.aquery(question, param=build_query_param(mode))
        answer = await asyncio.wait_for(coro, timeout=timeout)
        latency = time.perf_counter() - started
        cleaned = str(answer).replace("\n", "").replace("\r", "")
        return cleaned, True, latency
    except asyncio.TimeoutError:
        latency = time.perf_counter() - started
        q_prefix = question[:50].replace("\n", " ")
        await GLOBAL_ERROR_STATS.record("timeout")
        print(f"[timeout] {latency:.1f}s question={q_prefix!r}")
        return "Timeout", False, latency
    except Exception as e:
        latency = time.perf_counter() - started
        reason = f"error:{type(e).__name__}"
        await GLOBAL_ERROR_STATS.record(reason)
        print(f"Error in minirag_answer ({type(e).__name__}): {e}")
        return "Error", False, latency


async def fill_answers_concurrently(rows, result_column, mode, resume, output_path, fieldnames):
    pending_indices = []
    skipped_count = 0

    for idx, row in enumerate(rows):
        existing = str(row.get(result_column, "") or "").strip()
        if resume and existing:
            skipped_count += 1
            continue
        pending_indices.append(idx)

    if not pending_indices:
        print("没有需要新计算的问题。")
        if resume:
            print(f"断点续跑已跳过 {skipped_count} 条已有结果")
        return 0, skipped_count, 0.0

    initial_query_limit = args.query_concurrency
    if args.auto_scale_query:
        initial_query_limit = max(1, min(args.auto_scale_start_query, args.query_concurrency))

    limiter = AdaptiveLimiter(initial_query_limit)
    stats = ThroughputStats()
    progress = QueryProgress(total_tasks=len(pending_indices))
    stop_event = asyncio.Event()
    reporter_task = asyncio.create_task(
        stats_reporter(
            stats=stats,
            progress=progress,
            limiter=limiter,
            stop_event=stop_event,
            interval_sec=args.stats_interval_sec,
            stall_warn_sec=args.stall_warn_sec,
            enable_auto_scale=args.auto_scale_query,
            max_query_concurrency=args.query_concurrency,
            scale_step=args.auto_scale_step_query,
            stable_windows_needed=args.auto_scale_stable_windows,
            min_completed_per_window=args.auto_scale_min_completed,
            min_success_rate=args.auto_scale_min_success_rate,
        )
    )

    async def worker(idx: int):
        await limiter.acquire()
        await progress.mark_start()
        question = rows[idx]["Question"]
        try:
            answer, success, latency = await aquery_one(question, mode)
            return idx, answer, success, latency
        finally:
            await progress.mark_done()
            await limiter.release()

    tasks = [asyncio.create_task(worker(i)) for i in pending_indices]

    answered_count = 0
    started_all = time.perf_counter()

    for done in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="并发处理问题"):
        idx, answer, success, latency = await done
        rows[idx][result_column] = answer
        answered_count += 1

        await stats.record(latency_sec=latency, success=success)

        if answered_count % args.save_every == 0:
            persist_rows(output_path, fieldnames, rows)
            print(f"已累计保存 {answered_count} 条新结果到文件: {output_path}")

    persist_rows(output_path, fieldnames, rows)

    stop_event.set()
    await reporter_task

    total_elapsed = max(1e-9, time.perf_counter() - started_all)
    summary = await stats.summary()

    if summary["count"] > 0:
        avg_latency = summary["latency_sum"] / summary["count"]
        success_rate = summary["success"] * 100.0 / summary["count"]
        overall_rpm = summary["count"] * 60.0 / total_elapsed
        print(
            f"[stats-final] 总计: Requests={summary['count']}, "
            f"RPM={overall_rpm:.2f}, "
            f"AvgLatency={avg_latency:.2f}s, "
            f"SuccessRate={success_rate:.2f}% "
            f"(ok={summary['success']}, fail={summary['fail']})"
        )

        fail_summary = await GLOBAL_ERROR_STATS.summary()
        if fail_summary:
            detail = ", ".join(f"{k}:{v}" for k, v in sorted(fail_summary.items()))
            print(f"[stats-final] 失败类型统计: {{ {detail} }}")

    if resume:
        print(f"断点续跑已跳过 {skipped_count} 条已有结果")

    return answered_count, skipped_count, total_elapsed


def save_stats_to_file(stats_path: str, mode: str, total_time: float, token_stats: dict, total_questions: int, answered_count: int):
    """保存统计信息到新文件"""
    import json
    from datetime import datetime

    stats_data = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "model": LLM_MODEL,
        "total_questions": total_questions,
        "answered_count": answered_count,
        "total_time_seconds": round(total_time, 2),
        "total_time_minutes": round(total_time / 60, 2),
        "token_usage": {
            "prompt_tokens": token_stats.get("prompt_tokens", 0),
            "completion_tokens": token_stats.get("completion_tokens", 0),
            "total_tokens": token_stats.get("total_tokens", 0),
        },
        "avg_time_per_question_seconds": round(total_time / max(answered_count, 1), 2),
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)

    print(f"\n统计信息已保存到: {stats_path}")
    print(f"总时间: {stats_data['total_time_minutes']:.2f} 分钟 ({total_time:.2f} 秒)")
    print(f"总Token数: {stats_data['token_usage']['total_tokens']}")
    print(f"  - Prompt tokens: {stats_data['token_usage']['prompt_tokens']}")
    print(f"  - Completion tokens: {stats_data['token_usage']['completion_tokens']}")


def run_experiment(output_path, mode: str, resume: bool = True):
    if mode == "naive":
        result_column = "naiveRAG"
    elif mode == "light":
        result_column = "lightRAG"
    elif mode == "mini":
        result_column = "miniRAG"
    else:
        print("Invalid mode")
        raise SystemExit(1)

    fieldnames, rows = load_rows(output_path, result_column)
    print(f"总问题数: {len(rows)}")

    # 记录整个实验开始时间
    experiment_start = time.perf_counter()

    answered_count, _, _ = asyncio.run(
        fill_answers_concurrently(
            rows=rows,
            result_column=result_column,
            mode=mode,
            resume=resume,
            output_path=output_path,
            fieldnames=fieldnames,
        )
    )

    # 计算总耗时
    total_elapsed = time.perf_counter() - experiment_start

    # 注意：Token统计需要在异步上下文中获取，已在fill_answers_concurrently中获取
    # 这里我们重新创建获取token统计的任务
    async def get_token_stats():
        return await GLOBAL_TOKEN_STATS.summary()

    token_stats = asyncio.run(get_token_stats())

    # 生成统计文件路径（与输出文件同目录，但加上_stats后缀）
    output_dir = os.path.dirname(output_path)
    output_name = os.path.splitext(os.path.basename(output_path))[0]
    stats_path = os.path.join(output_dir, f"{output_name}_stats.json")

    # 保存统计信息
    save_stats_to_file(
        stats_path=stats_path,
        mode=mode,
        total_time=total_elapsed,
        token_stats=token_stats,
        total_questions=len(rows),
        answered_count=answered_count,
    )

    print(f"\n已将结果写入文件: {output_path}")
    print(f"本次新回答条数: {answered_count}")


if __name__ == "__main__":
    run_experiment(OUTPUT_PATH, mode=args.mode, resume=args.resume)

    """
    python3 reproduce/1_QA.py \
  --mode mini \
  --resume \
  --query_concurrency 8 \
  --llm_max_async 12 \
  --embedding_max_async 16 \
  --api_timeout 180 \
  --question_timeout_sec 180 \
  --stats_interval_sec 60 \
  --stall_warn_sec 180 \
  --auto_scale_query \
  --auto_scale_start_query 2 \
  --auto_scale_step_query 1 \
  --auto_scale_stable_windows 2 \
  --auto_scale_min_completed 1 \
  --auto_scale_min_success_rate 95 \
  --save_every 50
    """
