# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import sys
import os
import logging

# 设置 CUDA 可见设备
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 控制 vLLM 和 OpenAI 客户端的日志级别
# 可选值: DEBUG, INFO, WARNING, ERROR, CRITICAL
# 设置为 WARNING 或 ERROR 可以减少日志输出
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"  # 控制 vLLM server 的日志级别
logging.getLogger("openai").setLevel(logging.WARNING)  # 控制 OpenAI 客户端的日志级别
logging.getLogger("httpx").setLevel(logging.WARNING)  # 控制 HTTP 请求的日志级别

# 将上级目录加入sys.path，方便导入minirag包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入MiniRAG相关模块和函数
from minirag import MiniRAG
from minirag.llm import (
    hf_embed,  # Embedding 使用 transformers，vLLM 不提供 embedding 功能
    openai_complete_if_cache,  # 直接使用底层函数，可以传递 base_url
)
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

# 指定用于文本嵌入的模型【】
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import argparse
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
# 解析命令行参数
def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="qwen")  # 指定LLM模型
    parser.add_argument("--outputpath", type=str, default="./tests/Qwen/Default_output.csv")  # 输出路径
    parser.add_argument("--workingdir", type=str, default="./tests/Qwen3-4B-Instruct-2507_vllm")  # 工作目录
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/LiHua-World")  # 数据目录
    parser.add_argument(
        "--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv"
    )  # 查询集路径
    args = parser.parse_args()
    return args

# 获取命令行参数
args = get_args()

# 根据参数选择不同的LLM模型
if args.model == "PHI":
    LLM_MODEL = "microsoft/Phi-3.5-mini-instruct"
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
# BATCH_TEST_DATAPATH = "./dataset/LiHua-World/data/LiHua-World/week2"


print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)

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
    
    # 调用 openai_complete_if_cache，指定 base_url 连接到 vLLM server
    result = await openai_complete_if_cache(
        model=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=VLLM_SERVER_BASE_URL,  # 指定 vLLM server 地址
        api_key=api_key,  # API key（vLLM server 不需要真实 key，但客户端要求必须设置）
        **kwargs
    )
    
    # 如果需要关键词提取，处理 JSON 响应
    if keyword_extraction:
        from minirag.utils import locate_json_string_body_from_string
        return locate_json_string_body_from_string(result)
    
    return result

# 初始化MiniRAG对象
rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=vllm_server_complete,  # 使用 vLLM server 包装函数
    llm_model_max_token_size=8192,         # LLM最大token数
    llm_model_name=LLM_MODEL,              # LLM模型名称
    embedding_batch_num=16,                # 减小embedding批次大小，降低显存占用（默认32）
    embedding_func=EmbeddingFunc(
        embedding_dim=384,                 # 嵌入维度
        max_token_size=1000,               # 嵌入最大token数
        func=lambda texts: hf_embed(      # 使用 hf_embed，vLLM 不提供 embedding 功能
            texts,
            tokenizer=tokenizer,           # 复用分词器
            embed_model=embed_model,       # 复用模型
        )
    ),
)

# 查找指定目录下所有txt文件
def find_txt_files(root_path):
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files
from tqdm import tqdm

# WEEK_LIST = find_txt_files(BATCH_TEST_DATAPATH)
#测试batch大小
WEEK_LIST = find_txt_files(DATA_PATH)
# 获取所有txt文件路径
print(f"共找到 {len(WEEK_LIST)} 个txt文件，开始处理...")

# 使用线程池并行读取文件，按批次边读边插入，避免一次性占用大量内存
def load_txt_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
# print("CPU核心数:", os.cpu_count())
max_workers = 4
BATCH_SIZE = 2  # 减小单批文档数，降低单次显存压力
buffer = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(load_txt_file, path) for path in WEEK_LIST]
    for future in tqdm(
        as_completed(futures),
        total=len(WEEK_LIST),
        desc="读取并插入",
        unit="file",
        mininterval=1.0,  # 控制进度条刷新间隔（秒），可按需调整
    ):
        buffer.append(future.result())
        if len(buffer) >= BATCH_SIZE:
            # 保持文档粒度，直接传列表，避免跨文件合并导致实体/关系混淆
            rag.insert(buffer)
            buffer.clear()

# 插入剩余不足一批的内容
if buffer:
    rag.insert(buffer)
