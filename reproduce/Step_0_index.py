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
    gpt_4o_mini_complete,
    hf_embed,
)
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

# 指定用于文本嵌入的模型
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import argparse

# 解析命令行参数
def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="PHI")  # 指定LLM模型
    parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv")  # 输出路径
    parser.add_argument("--workingdir", type=str, default="./LiHua-World")  # 工作目录
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/")  # 数据目录
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
    LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
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

# 初始化MiniRAG对象
rag = MiniRAG(
    working_dir=WORKING_DIR,
    # llm_model_func=hf_model_complete,
    llm_model_func=gpt_4o_mini_complete,  # 指定LLM推理函数
    llm_model_max_token_size=200,         # LLM最大token数
    llm_model_name=LLM_MODEL,             # LLM模型名称
    embedding_func=EmbeddingFunc(
        embedding_dim=384,                # 嵌入维度
        max_token_size=1000,              # 嵌入最大token数
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),  # 加载分词器
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),    # 加载嵌入模型
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

# 获取所有txt文件路径
WEEK_LIST = find_txt_files(DATA_PATH)
for WEEK in WEEK_LIST:
    id = WEEK_LIST.index(WEEK)
    print(f"{id}/{len(WEEK_LIST)}")
    # 读取每个txt文件内容并插入到RAG索引中
    with open(WEEK) as f:
        rag.insert(f.read())
