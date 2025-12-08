# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import sys
import os

# 将上级目录加入sys.path，方便导入minirag包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入MiniRAG相关模块和函数
from minirag import MiniRAG
from minirag.llm import (
    hf_model_complete,
    hf_embed,
)
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

# 指定用于文本嵌入的模型
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import argparse
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# 解析命令行参数
def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="qwen")  # 指定LLM模型
    parser.add_argument("--outputpath", type=str, default="./tests/Qwen/Default_output.csv")  # 输出路径
    parser.add_argument("--workingdir", type=str, default="./tests/Qwen3-4B-Instruct-2507_batch4")  # 工作目录
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

# 初始化MiniRAG对象d
rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,      # 指定LLM推理函数
    llm_model_max_token_size=8192,         # LLM最大token数
    llm_model_name=LLM_MODEL,              # LLM模型名称
    embedding_func=EmbeddingFunc(
        embedding_dim=384,                 # 嵌入维度
        max_token_size=1000,               # 嵌入最大token数
        func=lambda texts: hf_embed(
            texts,
            tokenizer=tokenizer,           # 复用分词器
            embed_model=embed_model,       # 复用模型
        ),
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
max_workers = 8
BATCH_SIZE = 4  # 减小单批文档数，降低单次显存压力
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
