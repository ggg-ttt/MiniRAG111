# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import sys
import os
import warnings
import logging
import time
import traceback
import asyncio  # 添加asyncio导入

# 抑制transformers警告
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

# 将上级目录加入sys.path，方便导入minirag包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from tqdm import trange
from minirag import MiniRAG, QueryParam
from minirag.llm import (
    hf_model_complete,
    hf_embed,
    openai_complete_if_cache
)
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

# 指定用于文本嵌入的模型
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import argparse

# 解析命令行参数
def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="qwen")  # 指定LLM模型
    parser.add_argument("--outputpath", type=str, default="./tests/Qwen3-4B-Instruct-2507_vllm_debug/Default_output_debug.csv")  # 输出文件的路径，追加本次回答
    parser.add_argument("--workingdir", type=str, default="./tests/Qwen3-4B-Instruct-2507_vllm_debug")  # 工作目录
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/LiHua-World/")  # 数据目录
    parser.add_argument(
        "--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv"
    )  # 查询集路径
    parser.add_argument("--delay", type=float, default=2.0)  # API调用之间的延时（秒）
    parser.add_argument(
        "--mode",
        type=str,
        default="mini",
        choices=["naive", "light", "mini"],
        help="RAG 模式：naive / light / mini（默认: mini）",
    )
    args = parser.parse_args()
    return args

# 获取命令行参数
args = get_args()

# 根据参数选择不同的LLM模型
if args.model == "PHI":
    LLM_MODEL = "microsoft/Phi-4-mini-instruct"
elif args.model == "GLM":
    LLM_MODEL = "THUDM/glm-edge-1.5b-chat"
elif args.model == "MiniCPM":
    LLM_MODEL = "openbmb/MiniCPM3-4B"
elif args.model == "qwen":
    LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
else:
    print("Invalid model name")
    exit(1)

# 设定各路径参数
WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath
API_DELAY = args.delay  # API调用延时

print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)
print("API DELAY:", API_DELAY, "seconds")

# 如果工作目录不存在则创建
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# vLLM Server 配置
VLLM_SERVER_BASE_URL = "http://0.0.0.0:8000/v1"  # vLLM server 地址
VLLM_API_KEY = None  # 如果 vLLM server 设置了 API key，在这里填写

# 创建包装函数，连接到 vLLM server，添加日志和错误处理
async def vllm_server_complete(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
    """通过 vLLM server 调用模型的包装函数，添加详细日志"""
    # 从 kwargs 中获取模型名称（MiniRAG 会通过 hashing_kv 传递）
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]

    # vLLM server 不需要真实的 API key，但 OpenAI 客户端要求必须设置
    # 如果未设置，使用 dummy key
    api_key = VLLM_API_KEY if VLLM_API_KEY else "dummy"

    # 设置默认的生成参数（可以通过 kwargs 覆盖）
    default_params = {
        "max_tokens": 200,        # 最大输出长度（tokens）
        "temperature": 0.3,        # 温度参数（0.0-2.0，越高越随机）
        "top_p": 0.8,              # top-p 采样（0.0-1.0）
        "frequency_penalty": 0.0,   # 频率惩罚（-2.0 到 2.0）
        "presence_penalty": 0.0,   # 存在惩罚（-2.0 到 2.0）
        "stop": None,              # 停止序列（列表或None）
    }

    # 合并默认参数和用户传入的参数（用户参数优先）
    merged_params = {**default_params, **kwargs}

    # 过滤掉 OpenAI API 不支持的参数
    # OpenAI API 支持的参数: max_tokens, temperature, top_p, frequency_penalty, presence_penalty, stop
    supported_params = {
        "max_tokens", "temperature", "top_p", "frequency_penalty",
        "presence_penalty", "stop", "stream", "logprobs", "top_logprobs"
    }
    filtered_params = {k: v for k, v in merged_params.items() if k in supported_params}

    # 记录API调用开始
    print(f"\n[API Call] Starting request to vLLM server...")
    print(f"[API Call] Model: {model_name}")
    print(f"[API Call] Prompt length: {len(prompt)} chars")
    print(f"[API Call] Params: {filtered_params}")

    start_time = time.time()

    try:
        # 调用 openai_complete_if_cache，指定 base_url 连接到 vLLM server
        result = await openai_complete_if_cache(
            model=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=VLLM_SERVER_BASE_URL,  # 指定 vLLM server 地址
            api_key=api_key,  # API key（vLLM server 不需要真实 key，但客户端要求必须设置）
            **filtered_params  # 传递过滤后的参数
        )

        # 计算响应时间
        response_time = time.time() - start_time
        print(f"[API Call] Success! Response time: {response_time:.2f}s")
        print(f"[API Call] Response length: {len(result)} chars")
        print(f"[API Call] Response preview: {result[:100]}...")

    except Exception as e:
        # 记录详细错误信息
        response_time = time.time() - start_time
        print(f"\n[API ERROR] Error occurred after {response_time:.2f}s")
        print(f"[API ERROR] Error type: {type(e).__name__}")
        print(f"[API ERROR] Error message: {str(e)}")
        print(f"[API ERROR] Traceback:")
        traceback.print_exc()
        raise  # 重新抛出异常，让上层处理

    # 如果需要关键词提取，处理 JSON 响应
    if keyword_extraction:
        from minirag.utils import locate_json_string_body_from_string
        return locate_json_string_body_from_string(result)

    return result


# 初始化MiniRAG对象
rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=vllm_server_complete,  # 指定LLM推理函数
    llm_model_max_token_size=8192,      # LLM最大token数
    llm_model_name=LLM_MODEL,          # LLM模型名称
    embedding_func=EmbeddingFunc(
        embedding_dim=384,             # 嵌入维度
        max_token_size=1000,           # 嵌入最大token数
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),  # 加载分词器
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),    # 加载嵌入模型
        ),
    ),
)

# 读取问题和标准答案
QUESTION_LIST = []
GA_LIST = []
with open(QUERY_PATH, mode="r", encoding="utf-8") as question_file:
    reader = csv.DictReader(question_file)
    for row in reader:
        QUESTION_LIST.append(row["Question"])
        GA_LIST.append(row["Gold Answer"])

# 运行实验并记录结果
def run_experiment(output_path, mode: str):
    if mode == "naive":
        result_column = "naiveRAG"  # 结果列名
        mode_suffix = "naive"
    elif mode == "light":
        result_column = "lightRAG"  # 结果列名
        mode_suffix = "light"
    elif mode == "mini":
        result_column = "miniRAG"  # 结果列名
        mode_suffix = "mini"
    else:
        print("Invalid mode")
        exit(1)

    # 为不同模式创建独立的输出文件
    base_name = os.path.splitext(output_path)[0]  # 文件名不带扩展名
    extension = os.path.splitext(output_path)[1]  # 文件扩展名（如.csv）
    mode_output_path = f"{args.model}_{mode_suffix}{extension}"

    print(f"使用 {mode} 模式，结果将保存到: {mode_output_path}")

    # 检查输出文件是否已存在
    existing_data = []
    file_exists = os.path.exists(mode_output_path) and os.path.getsize(mode_output_path) > 0

    if file_exists:
        # 读取现有数据
        print(f"检测到现有文件: {mode_output_path}")
        with open(mode_output_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)
        print(f"读取到 {len(existing_data)} 行现有数据")

    # 准备所有问题的数据
    headers = ["Question", "Gold Answer", result_column, "Error_Info"]
    all_rows = []

    # 遍历所有问题
    for QUESTIONid in trange(len(QUESTION_LIST), desc="处理问题"):
        QUESTION = QUESTION_LIST[QUESTIONid]
        Gold_Answer = GA_LIST[QUESTIONid]

        # 检查是否已有该问题的数据
        existing_row = None
        if existing_data:
            for row in existing_data:
                if row["Question"] == QUESTION and row["Gold Answer"] == Gold_Answer:
                    existing_row = row
                    break

        # 判断是否需要重新生成答案
        need_regenerate = False
        if existing_row:
            if result_column in existing_row:
                existing_answer = existing_row[result_column].strip()
                if existing_answer == "Error" or not existing_answer:
                    need_regenerate = True
                    print(f"\n问题 {QUESTIONid + 1} 现有答案为 '{existing_answer}'，需要重新生成")
                else:
                    print(f"\n问题 {QUESTIONid + 1} 已有有效答案，跳过")
                    all_rows.append(existing_row)
                    continue
            else:
                need_regenerate = True
                print(f"\n问题 {QUESTIONid + 1} 缺少 {result_column} 列，需要生成答案")
        else:
            need_regenerate = True
            print(f"\n问题 {QUESTIONid + 1} 无现有数据，需要生成答案")

        print()
        print(f"{'='*60}")
        print(f"问题 {QUESTIONid + 1}/{len(QUESTION_LIST)}: {QUESTION}")
        print(f"标准答案: {Gold_Answer}")
        print(f"{'='*60}")

        if need_regenerate:
            # 直接调用 rag.query，不再进行超时控制
            try:
                minirag_answer = rag.query(QUESTION, param=QueryParam(mode=mode))
                minirag_answer = minirag_answer.replace("\n", "").replace("\r", "")
                error_info = ""
                print(f"\n[SUCCESS] 成功生成答案")
            except Exception as e:
                print(f"\n[ERROR] Error in minirag_answer: {e}")
                print(f"[ERROR] Error type: {type(e).__name__}")
                traceback.print_exc()
                minirag_answer = "Error"
                error_info = f"{type(e).__name__}: {str(e)}"

            # API调用延时（如果成功）
            if minirag_answer != "Error" and API_DELAY > 0:
                print(f"\n[DELAY] Waiting {API_DELAY} seconds before next request...")
                time.sleep(API_DELAY)
        else:
            # 使用现有数据
            minirag_answer = existing_row[result_column] if existing_row else "Error"
            error_info = existing_row.get("Error_Info", "")

        # 创建或更新行数据
        if existing_row:
            existing_row[result_column] = minirag_answer
            existing_row["Error_Info"] = error_info if minirag_answer == "Error" else ""
            all_rows.append(existing_row)
        else:
            new_row = {
                "Question": QUESTION,
                "Gold Answer": Gold_Answer,
                result_column: minirag_answer,
                "Error_Info": error_info if minirag_answer == "Error" else ""
            }
            all_rows.append(new_row)

    # 写入所有数据到文件（覆盖模式）
    with open(mode_output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{mode} 模式的实验数据已记录到文件: {mode_output_path}")

    # 统计错误数量
    error_count = sum(1 for row in all_rows if row[result_column] == "Error")
    print(f"错误数量: {error_count}/{len(all_rows)}")

    return mode_output_path


def merge_answer(output_path, mode: str):
    # 合并不同模式的答案到一个文件
    base_name = os.path.splitext(output_path)[0]  # 文件名不带扩展名
    extension = os.path.splitext(output_path)[1]  # 文件扩展名（如.csv）
    merged_output_path = f"{args.model}_merged{extension}"

    print(f"\n正在合并答案到文件: {merged_output_path}")

    # 读取所有模式的答案
    mode_suffixes = ["naive", "light", "mini"]
    all_rows = []

    for mode_suffix in mode_suffixes:
        mode_output_path = f"{base_name}_{mode_suffix}{extension}"
        if not os.path.exists(mode_output_path):
            print(f"警告: 未找到文件 {mode_output_path}，跳过")
            continue

        with open(mode_output_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 查找是否已有该问题的数据
                existing_row = next((r for r in all_rows if r["Question"] == row["Question"]), None)
                if existing_row:
                    # 更新现有行
                    existing_row.update(row)
                else:
                    # 添加新行
                    all_rows.append(row)

    # 写入合并后的数据到文件
    if all_rows:
        headers = all_rows[0].keys()
        with open(merged_output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"合并完成，结果保存到: {merged_output_path}")
    else:
        print("没有数据可供合并")

# 主流程，只跑一个模式（通过命令行 --mode 指定）
if __name__ == "__main__":
    mode = args.mode

    # 统计错误信息
    error_count = 0
    total_count = len(QUESTION_LIST)

    print(f"\n当前模式: {mode}")
    print(f"共 {total_count} 个问题")
    print(f"API调用延时设置为: {API_DELAY} 秒")

    # 运行实验并获取模式特定的输出路径
    actual_output_path = run_experiment(OUTPUT_PATH, mode=mode)

    # 读取结果并统计错误率
    if os.path.exists(actual_output_path):
        with open(actual_output_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 根据当前模式检查对应的错误列
                result_column = f"{mode}RAG"
                if result_column in row and row.get(result_column, "") == "Error":
                    error_count += 1

        error_rate = (error_count / total_count) * 100
        print(f"\n{'='*60}")
        print(f"实验完成！（模式: {mode}）")
        print(f"总问题数: {total_count}")
        print(f"错误数: {error_count}")
        print(f"错误率: {error_rate:.2f}%")
        print(f"结果文件: {actual_output_path}")
        print(f"{'='*60}")