# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import sys
import os
import logging
import json

# 控制 OpenAI 客户端的日志级别
# 可选值: DEBUG, INFO, WARNING, ERROR, CRITICAL
# 设置为 WARNING 或 ERROR 可以减少日志输出
logging.getLogger("openai").setLevel(logging.WARNING)  # 控制 OpenAI 客户端的日志级别
logging.getLogger("httpx").setLevel(logging.WARNING)  # 控制 HTTP 请求的日志级别

# 将上级目录加入sys.path，方便导入minirag包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入MiniRAG相关模块和函数
from minirag import MiniRAG
from minirag.llm import hf_embed
from minirag.utils import EmbeddingFunc, compute_mdhash_id
from transformers import AutoModel, AutoTokenizer
from minirag.llm.zhipu import zhipu_complete_if_cache
# 指定用于文本嵌入的模型
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 智谱 API 配置
ZHIPU_API_KEY = "fc9d4f54470c4222b3131e54af0295ec.JozSQr3hvOPzZKSS"  # 请替换为你的智谱 API 密钥
import argparse
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# 解析命令行参数
def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="glm")  # 指定LLM模型
    parser.add_argument("--outputpath", type=str, default="./tests/glm/Default_output.csv")  # 输出路径
    parser.add_argument("--workingdir", type=str, default="./tests/glm")  # 工作目录
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
elif args.model == "dpsk":
    LLM_MODEL = "deepseek-ai/DeepSeek-V3.2"
elif args.model == "glm":
    LLM_MODEL = "glm-4.5-air"
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

# 预先加载分词器与嵌入模型，避免在循环中重复加载（显著提速）默认auto是自动第一个
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, device_map="auto")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
embed_model = AutoModel.from_pretrained(
    EMBEDDING_MODEL,
    device_map="auto",
    dtype=torch.float16,  # 降低显存占用
)

# 初始化MiniRAG对象
rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=lambda prompt, **kwargs: zhipu_complete_if_cache(
        prompt=prompt,
        model=LLM_MODEL,
        api_key=ZHIPU_API_KEY,
        **kwargs
    ),                                       # 使用智谱 API
    llm_model_max_token_size=8192,          # LLM最大token数（输入+输出总和）
    llm_model_name=LLM_MODEL,            # 模型名称
    embedding_batch_num=16,                 # 减小embedding批次大小，降低显存占用（默认32）
    embedding_func=EmbeddingFunc(
        embedding_dim=384,                  # 嵌入维度
        max_token_size=1000,                # 嵌入最大token数
        func=lambda texts: hf_embed(
            texts,
            tokenizer=tokenizer,            # 复用分词器
            embed_model=embed_model,        # 复用模型
        ),
    ),
)

# 载入已处理文档的 doc_id 集合（来源：working_dir/kv_store_full_docs.json）
def load_processed_doc_ids(work_dir: str):
    """加载已处理的文档ID集合，避免重复处理"""
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

processed_doc_ids = load_processed_doc_ids(WORKING_DIR)

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
    """加载文本文件，返回文件路径和内容"""
    with open(path, "r", encoding="utf-8") as f:
        return path, f.read()

# print("CPU核心数:", os.cpu_count())
max_workers = 2  # 降低并发数，避免API请求过于集中
BATCH_SIZE = 2   # 减小单批文档数，降低单次显存压力
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
        file_path, content = future.result()

        # 基于内容计算 doc_id（与 MiniRAG 内部一致的 MD5 前缀）
        doc_id = compute_mdhash_id(content, prefix="doc-")
        if doc_id in processed_doc_ids:
            # 已处理文档，跳过
            continue

        buffer.append(content)
        if len(buffer) >= BATCH_SIZE:
            # 保持文档粒度，直接传列表，避免跨文件合并导致实体/关系混淆
            rag.insert(buffer)
            buffer.clear()

# 插入剩余不足一批的内容
if buffer:
    rag.insert(buffer)
