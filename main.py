# 如果需要从Hugging Face Hub下载私有模型，需要取消注释并填入你的token
# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import os
from minirag import MiniRAG, QueryParam
from minirag.llm.hf import (
    hf_model_complete,  # 从Hugging Face加载模型进行文本补全的函数
    hf_embed,           # 从Hugging Face加载模型进行文本嵌入的函数
)
from minirag.utils import EmbeddingFunc  # 嵌入函数的封装类
from transformers import AutoModel, AutoTokenizer  # Hugging Face的transformers库

# 定义默认的文本嵌入模型
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import argparse  # 用于解析命令行参数

# 定义一个函数来获取和解析命令行参数
def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    # --model: 指定使用的大语言模型（LLM）
    parser.add_argument("--model", type=str, default="PHI")
    # --outputpath: 指定输出结果（如日志、CSV文件）的路径
    parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv")
    # --workingdir: 指定MiniRAG的工作目录，用于存放索引、缓存等文件
    parser.add_argument("--workingdir", type=str, default="./LiHua-World")
    # --datapath: 指定待索引的数据源路径
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/")
    # --querypath: 指定包含查询问题的CSV文件路径
    parser.add_argument(
        "--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv"
    )
    args = parser.parse_args()
    return args

# 获取命令行参数
args = get_args()

# 根据命令行参数选择使用的大语言模型（LLM）
if args.model == "PHI":
    LLM_MODEL = "microsoft/Phi-3.5-mini-instruct"
elif args.model == "GLM":
    LLM_MODEL = "THUDM/glm-edge-1.5b-chat"
elif args.model == "MiniCPM":
    LLM_MODEL = "openbmb/MiniCPM3-4B"
elif args.model == "qwen":
    LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
else:
    print("无效的模型名称 (Invalid model name)")
    exit(1)

# 设置工作目录、数据路径、查询路径和输出路径
WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath
print("正在使用的大语言模型 (USING LLM):", LLM_MODEL)
print("正在使用的工作目录 (USING WORKING DIR):", WORKING_DIR)

# 如果工作目录不存在，则创建它
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# 初始化MiniRAG系统
rag = MiniRAG(
    working_dir=WORKING_DIR,  # 设置工作目录
    llm_model_func=hf_model_complete,  # 指定用于生成回答的LLM函数
    llm_model_max_token_size=200,      # LLM生成的最大token数
    llm_model_name=LLM_MODEL,          # 指定LLM的模型名称
    embedding_func=EmbeddingFunc(      # 配置嵌入函数
        embedding_dim=384,             # 嵌入向量的维度
        max_token_size=1000,           # 嵌入模型处理的最大token数
        func=lambda texts: hf_embed(   # 使用lambda定义具体的嵌入实现
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),  # 加载分词器
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),    # 加载嵌入模型
        ),
    ),
)


# --- 数据索引阶段 ---
# 定义一个函数来查找指定根目录下的所有.txt文件
def find_txt_files(root_path):
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files

# 查找数据路径下的所有.txt文件
WEEK_LIST = find_txt_files(DATA_PATH)
# 遍历文件列表，并将每个文件的内容插入到RAG系统中进行索引
for WEEK in WEEK_LIST:
    id = WEEK_LIST.index(WEEK)
    print(f"{id}/{len(WEEK_LIST)}")  # 打印处理进度
    with open(WEEK) as f:
        rag.insert(f.read())  # 读取文件内容并插入

# --- 查询阶段 ---
# 一个示例查询
query = 'What does LiHua predict will happen in "The Rings of Power"?'
# 调用RAG系统的query方法进行查询
# QueryParam(mode="mini") 指定使用MiniRAG的特定查询模式
# .replace("\n", "").replace("\r", "") 清理响应中的换行符
answer = (
    rag.query(query, param=QueryParam(mode="mini")).replace("\n", "").replace("\r", "")
)
# 打印最终的答案
print(answer)
