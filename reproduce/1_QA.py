# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import sys
import os

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
    parser.add_argument("--outputpath", type=str, default="./tests/Qwen3-4B-Instruct-2507_vllm/Default_output.csv")  # 输出文件的路径，追加本次回答
    parser.add_argument("--workingdir", type=str, default="./tests/Qwen3-4B-Instruct-2507_vllm")  # 工作目录
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/LiHua-World/")  # 数据目录
    parser.add_argument(
        "--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv"
    )  # 查询集路径
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
print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)

# 如果工作目录不存在则创建
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# vLLM Server 配置
VLLM_SERVER_BASE_URL = "http://0.0.0.0:8000/v1"  # vLLM server 地址
VLLM_API_KEY = None  # 如果 vLLM server 设置了 API key，在这里填写

# 创建包装函数，连接到 vLLM server
async def vllm_server_complete(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
    """通过 vLLM server 调用模型的包装函数"""
    # 从 kwargs 中获取模型名称（MiniRAG 会通过 hashing_kv 传递）
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    
    # vLLM server 不需要真实的 API key，但 OpenAI 客户端要求必须设置
    # 如果未设置，使用 dummy key
    api_key = VLLM_API_KEY if VLLM_API_KEY else "dummy"
    
    # 设置默认的生成参数（可以通过 kwargs 覆盖）
    default_params = {
        "max_tokens": 2048,        # 最大输出长度（tokens）
        "temperature": 0.3,        # 温度参数（0.0-2.0，越高越随机）
        "top_p": 0.8,              # top-p 采样（0.0-1.0）
        "frequency_penalty": 0.0,   # 频率惩罚（-2.0 到 2.0）
        "presence_penalty": 0.0,   # 存在惩罚（-2.0 到 2.0）
        "stop": None,              # 停止序列（列表或 None）
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

    # 检查输出文件是否已存在
    if os.path.exists(output_path):
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
        
        print(f"读取到 {len(existing_rows)} 行已存在的数据")
        
        # 对每行的 Question 使用 MiniRAG 进行问答
        for idx in trange(len(existing_rows), desc="处理问题"):
            row = existing_rows[idx]
            question = row["Question"]
            
            # 如果该问题已有结果且不为空，可以选择跳过或重新计算
            # 这里选择重新计算（如果需要跳过，可以取消下面的注释）
            # if result_column in row and row[result_column] and row[result_column].strip():
            #     continue
            
            print()
            print(f"问题 {idx + 1}/{len(existing_rows)}: {question}")
            
            try:
                # 使用MiniRAG进行问答
                minirag_answer = (
                    rag.query(question, param=QueryParam(mode=mode))
                    .replace("\n", "")
                    .replace("\r", "")
                )
            except Exception as e:
                print(f"Error in minirag_answer: {e}")
                minirag_answer = "Error"
            
            # 更新该行的结果列
            row[result_column] = minirag_answer
        
        # 写回文件（覆盖模式）
        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
        
        print(f"已将结果追加到文件: {output_path}")
    else:
        # 文件不存在，创建新文件
        headers = ["Question", "Gold Answer", result_column]
        
        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)  # 写入表头
            
            # 遍历所有问题
            for QUESTIONid in trange(len(QUESTION_LIST), desc="处理问题"):
                QUESTION = QUESTION_LIST[QUESTIONid]
                Gold_Answer = GA_LIST[QUESTIONid]
                print()
                print(f"问题 {QUESTIONid + 1}/{len(QUESTION_LIST)}: {QUESTION}")
                print(f"标准答案: {Gold_Answer}")
                
                try:
                    # 使用MiniRAG进行问答
                    minirag_answer = (
                        rag.query(QUESTION, param=QueryParam(mode=mode))
                        .replace("\n", "")
                        .replace("\r", "")
                    )
                except Exception as e:
                    print(f"Error in minirag_answer: {e}")
                    minirag_answer = "Error"
                
                # 写入一行结果
                writer.writerow([QUESTION, Gold_Answer, minirag_answer])
        
        print(f"实验数据已记录到文件: {output_path}")

# 主流程，直接运行实验
if __name__ == "__main__":
    import sys
    mode = "light"
    run_experiment(OUTPUT_PATH, mode=mode)
