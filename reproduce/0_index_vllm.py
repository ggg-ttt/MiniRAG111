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
    hf_embed,  # Embedding 使用 transformers
    openai_complete_if_cache,  # 直接使用底层函数，可以传递 base_url
)
from minirag.utils import EmbeddingFunc, compute_mdhash_id
from transformers import AutoModel, AutoTokenizer

# 指定用于文本嵌入的模型【】
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import argparse
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

"""
python 0_index_vllm.py --api_type openai --model deepseek-chat \
    --openai_api_key sk-xxx --openai_base_url https://api.deepseek.com/v1

"""


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="qwen")  # 指定LLM模型
    parser.add_argument("--outputpath", type=str, default="./tests/Qwen/Default_output.csv")  # 输出路径
    parser.add_argument("--workingdir", type=str, default="./tests/dpsk")  # 工作目录
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/LiHua-World")  # 数据目录
    parser.add_argument(
        "--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv"
    )  # 查询集路径
    # OpenAI API 相关参数
    parser.add_argument(
        "--api_type", type=str, default="vllm", choices=["vllm", "openai"],
        help="API 类型：vllm（本地vLLM server）或 openai（OpenAI兼容接口）"
    )
    parser.add_argument(
        "--openai_base_url", type=str, default="https://api.siliconflow.cn/v1",
        help="OpenAI API base URL，也可以是兼容 OpenAI 格式的第三方 API 地址"
    )
    parser.add_argument(
        "--openai_api_key", type=str, default=None,
        help="OpenAI API Key"
    )
    args = parser.parse_args()
    return args

# 获取命令行参数
args = get_args()

# 根据参数选择不同的LLM模型
if args.api_type == "openai":
    # OpenAI 模式下，--model 直接作为模型名称（如 gpt-4o、gpt-4-turbo、deepseek-chat 等）
    LLM_MODEL = "deepseek-ai/DeepSeek-V3.2"
else:
    # vLLM 模式下，根据参数选择本地模型
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

# OpenAI API 配置（由命令行参数指定）
OPENAI_BASE_URL = args.openai_base_url
OPENAI_API_KEY = args.openai_api_key

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

# OpenAI API 包装函数（与 vllm_server_complete 接口完全一致）
async def openai_complete(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
    """通过 OpenAI 兼容 API 调用模型的包装函数"""
    keyword_extraction = kwargs.pop("keyword_extraction", None)

    if not OPENAI_API_KEY:
        raise ValueError("使用 OpenAI API 时必须通过 --openai_api_key 参数提供 API Key")

    result = await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=OPENAI_BASE_URL,   # OpenAI API 地址
        api_key=OPENAI_API_KEY,     # OpenAI API Key
        **kwargs
    )

    # 如果需要关键词提取，处理 JSON 响应
    if keyword_extraction:
        from minirag.utils import locate_json_string_body_from_string
        return locate_json_string_body_from_string(result)

    return result

# 根据 api_type 选择对应的 LLM 函数
if args.api_type == "openai":
    llm_func = openai_complete
    print("API 类型: OpenAI")
    print("OPENAI BASE URL:", OPENAI_BASE_URL)
else:
    llm_func = vllm_server_complete
    print("API 类型: vLLM Server")
    print("VLLM SERVER URL:", VLLM_SERVER_BASE_URL)

# 初始化MiniRAG对象
rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_func,  # 根据 api_type 选择 LLM 函数
    llm_model_max_token_size=8192,         # 输入（Prompt）与输出（Completion）的总和长度，即模型的最大上下文窗口（Context Window）
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

# 载入已处理文档的 doc_id 集合（来源：working_dir/kv_store_full_docs.json）
def load_processed_doc_ids(work_dir: str):
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
    with open(path, "r", encoding="utf-8") as f:
        return path, f.read()
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
