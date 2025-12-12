# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import sys
import os
import warnings
import logging
import time
import traceback

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
    parser.add_argument("--outputpath", type=str, default="./tests/Qwen3-4B-Instruct-2507_vllm/Default_output_debug.csv")  # 输出文件的路径，追加本次回答
    parser.add_argument("--workingdir", type=str, default="./tests/Qwen3-4B-Instruct-2507_vllm_debug")  # 工作目录
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/LiHua-World/")  # 数据目录
    parser.add_argument(
        "--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv"
    )  # 查询集路径
    parser.add_argument("--delay", type=float, default=2.0)  # API调用之间的延时（秒）
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
    llm_model_max_token_size=10240,      # LLM最大token数
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
    elif mode == "light":
        result_column = "lightRAG"  # 结果列名
    elif mode == "mini":
        result_column = "miniRAG"  # 结果列名
    else:
        print("Invalid mode")
        exit(1)

    # 检查输出文件是否已存在且有数据
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        # 读取现有文件的所有行
        existing_rows = []
        with open(output_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])

            # 如果结果列不存在，添加到表头
            if result_column not in fieldnames:
                fieldnames.append(result_column)

            for row in reader:
                existing_rows.append(row)

        # 如果文件存在但没有数据行，按新文件处理
        if len(existing_rows) == 0:
            print("输出文件存在但无数据，按新文件处理")
            # 创建新文件
            headers = ["Question", "Gold Answer", result_column, "Error_Info"]

            with open(output_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)  # 写入表头

                # 遍历所有问题
                for QUESTIONid in trange(len(QUESTION_LIST), desc="处理问题"):
                    QUESTION = QUESTION_LIST[QUESTIONid]
                    Gold_Answer = GA_LIST[QUESTIONid]
                    print()
                    print(f"\n{'='*60}")
                    print(f"问题 {QUESTIONid + 1}/{len(QUESTION_LIST)}: {QUESTION}")
                    print(f"标准答案: {Gold_Answer}")
                    print(f"{'='*60}")

                    try:
                        # 使用MiniRAG进行问答
                        minirag_answer = (
                            rag.query(QUESTION, param=QueryParam(mode=mode))
                            .replace("\n", "")
                            .replace("\r", "")
                        )
                        error_info = ""
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

                    # 写入一行结果
                    writer.writerow([QUESTION, Gold_Answer, minirag_answer, error_info])

            print(f"\n实验数据已记录到文件: {output_path}")
        else:
            print(f"读取到 {len(existing_rows)} 行已存在的数据")

            # 对每行的 Question 使用 MiniRAG 进行问答
            for idx in trange(len(existing_rows), desc="处理问题"):
                row = existing_rows[idx]
                question = row["Question"]

                # 如果该问题已有结果且不为空且不是Error，可以跳过
                # 否则（不存在、为空、为Error），则重新计算
                if (result_column in row and row[result_column] and 
                    row[result_column].strip() and row[result_column] != "Error"):
                    print(f"问题 {idx + 1}/{len(existing_rows)} 已有答案，跳过")
                    continue

                print()
                print(f"\n{'='*60}")
                print(f"问题 {idx + 1}/{len(existing_rows)}: {question}")
                print(f"{'='*60}")

                try:
                    # 使用MiniRAG进行问答
                    minirag_answer = (
                        rag.query(question, param=QueryParam(mode=mode))
                        .replace("\n", "")
                        .replace("\r", "")
                    )
                    error_info = ""
                except Exception as e:
                    print(f"\n[ERROR] Error in minirag_answer: {e}")
                    print(f"[ERROR] Error type: {type(e).__name__}")
                    traceback.print_exc()
                    minirag_answer = "Error"
                    error_info = f"{type(e).__name__}: {str(e)}"

                # 更新该行的结果列
                row[result_column] = minirag_answer
                if "Error_Info" not in row:
                    row["Error_Info"] = ""
                if minirag_answer == "Error":
                    row["Error_Info"] = error_info

                # API调用延时（如果成功）
                if minirag_answer != "Error" and API_DELAY > 0:
                    print(f"\n[DELAY] Waiting {API_DELAY} seconds before next request...")
                    time.sleep(API_DELAY)

            # 写回文件（覆盖模式）
            with open(output_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames + ["Error_Info"])
                writer.writeheader()
                writer.writerows(existing_rows)

            print(f"\n已将结果追加到文件: {output_path}")
    else:
        # 文件不存在，创建新文件
        headers = ["Question", "Gold Answer", result_column, "Error_Info"]

        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)  # 写入表头

            # 遍历所有问题
            for QUESTIONid in trange(len(QUESTION_LIST), desc="处理问题"):
                QUESTION = QUESTION_LIST[QUESTIONid]
                Gold_Answer = GA_LIST[QUESTIONid]
                print()
                print(f"\n{'='*60}")
                print(f"问题 {QUESTIONid + 1}/{len(QUESTION_LIST)}: {QUESTION}")
                print(f"标准答案: {Gold_Answer}")
                print(f"{'='*60}")

                try:
                    # 使用MiniRAG进行问答
                    minirag_answer = (
                        rag.query(QUESTION, param=QueryParam(mode=mode))
                        .replace("\n", "")
                        .replace("\r", "")
                    )
                    error_info = ""
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

                # 写入一行结果
                writer.writerow([QUESTION, Gold_Answer, minirag_answer, error_info])

        print(f"\n实验数据已记录到文件: {output_path}")

# 主流程，直接运行实验
if __name__ == "__main__":
    mode = "light"

    # 统计错误信息
    error_count = 0
    total_count = len(QUESTION_LIST)

    print(f"\n开始运行实验，共 {total_count} 个问题")
    print(f"API调用延时设置为: {API_DELAY} 秒")

    run_experiment(OUTPUT_PATH, mode=mode)

    # 读取结果并统计错误率
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "lightRAG" in row and row["lightRAG"] == "Error":
                    error_count += 1

        error_rate = (error_count / total_count) * 100
        print(f"\n{'='*60}")
        print(f"实验完成！")
        print(f"总问题数: {total_count}")
        print(f"错误数: {error_count}")
        print(f"错误率: {error_rate:.2f}%")
        print(f"结果文件: {OUTPUT_PATH}")
        print(f"{'='*60}")