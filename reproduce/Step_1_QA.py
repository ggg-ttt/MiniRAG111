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
    llm_model_func=hf_model_complete,  # 指定LLM推理函数
    # llm_model_func=gpt_4o_mini_complete,
    llm_model_max_token_size=200,      # LLM最大token数
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
def run_experiment(output_path):
    headers = ["Question", "Gold Answer", "minirag"]  # CSV表头

    q_already = []
    # 检查输出文件是否已存在，避免重复写入
    if os.path.exists(output_path):
        with open(output_path, mode="r", encoding="utf-8") as question_file:
            reader = csv.DictReader(question_file)
            for row in reader:
                q_already.append(row["Question"])

    row_count = len(q_already)
    print("row_count", row_count)

    # 以追加模式写入实验结果
    with open(output_path, mode="a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if row_count == 0:
            writer.writerow(headers)  # 首次写入表头

        # 遍历所有未处理的问题
        for QUESTIONid in trange(row_count, len(QUESTION_LIST)):  #
            QUESTION = QUESTION_LIST[QUESTIONid]
            Gold_Answer = GA_LIST[QUESTIONid]
            print()
            print("QUESTION", QUESTION)
            print("Gold_Answer", Gold_Answer)

            try:
                # 使用MiniRAG进行问答
                minirag_answer = (
                    rag.query(QUESTION, param=QueryParam(mode="mini"))
                    .replace("\n", "")
                    .replace("\r", "")
                )
            except Exception as e:
                print("Error in minirag_answer", e)
                minirag_answer = "Error"

            # 写入一行结果
            writer.writerow([QUESTION, Gold_Answer, minirag_answer])

    print(f"Experiment data has been recorded in the file: {output_path}")

# 主流程，直接运行实验
# if __name__ == "__main__":

run_experiment(OUTPUT_PATH)
