# 导入所需库
import asyncio
import html
import io
import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union, List
import xml.etree.ElementTree as ET
import copy
import numpy as np
import tiktoken  # OpenAI的token计算库
from nltk.metrics import edit_distance
from rouge import Rouge  # ROUGE评分库
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # BLEU评分库
from sentence_transformers import SentenceTransformer  # 句子嵌入模型
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

# 全局tiktoken编码器实例，延迟初始化
ENCODER = None

# 获取名为"minirag"的日志记录器
logger = logging.getLogger("minirag")


def set_logger(log_file: str):
    """
    配置日志记录器，将日志输出到指定文件。
    Args:
        log_file (str): 日志文件的路径。
    """
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # 创建并设置日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # 如果记录器没有处理器，则添加新的文件处理器
    if not logger.handlers:
        logger.addHandler(file_handler)


@dataclass
class EmbeddingFunc:
    """
    一个数据类，用于封装嵌入函数及其元数据（维度、最大token数）。
    """
    embedding_dim: int  # 嵌入向量的维度
    max_token_size: int  # 函数能处理的最大token数
    func: callable  # 实际的嵌入函数

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        """使类的实例可以像函数一样被调用。"""
        return await self.func(*args, **kwargs)


def compute_mdhash_id(content, prefix: str = ""):
    """
    计算内容的MD5哈希值，并可选择添加前缀，作为唯一标识符。
    Args:
        content (any): 要计算哈希的内容。
        prefix (str): 哈希值的前缀。
    Returns:
        str: 带前缀的MD5哈希字符串。
    """
    return prefix + md5(str(content).encode()).hexdigest()


def compute_args_hash(*args, cache_type: str | None = None) -> str:
    """
    计算函数参数的哈希值，用于缓存键的生成。
    Args:
        *args: 函数的参数。
        cache_type (str, optional): 缓存类型，会加入哈希计算。
    Returns:
        str: 参数的MD5哈希字符串。
    """
    args_str = "".join([str(arg) for arg in args])
    if cache_type:
        args_str = f"{cache_type}:{args_str}"
    return md5(args_str.encode()).hexdigest()


def clean_text(text: str) -> str:
    """
    通过移除空字节(0x00)和首尾空白符来清理文本。
    """
    return text.strip().replace("\x00", "")


def get_content_summary(content: str, max_length: int = 100) -> str:
    """
    获取文档内容的摘要，如果内容过长则进行截断。
    """
    content = content.strip()
    return content if len(content) <= max_length else content[:max_length] + "..."


def locate_json_string_body_from_string(content: str) -> Union[str, None]:
    """
    从一个可能包含非JSON内容的字符串中，定位并提取出JSON格式的字符串主体。
    """
    # 使用正则表达式查找被{}包裹的部分
    maybe_json_str = re.search(r"({.*})", content, re.DOTALL)
    if maybe_json_str is not None:
        return maybe_json_str.group(0)
    else:
        # 再次尝试查找从 { 开始到 } 结束的部分
        maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
        if maybe_json_str is not None:
            return maybe_json_str.group(0)
        else:
            return None


def convert_response_to_json(response: str) -> dict:
    """
    将包含JSON的字符串响应转换为字典。
    """
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"无法从响应中解析JSON: {response}"
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"解析JSON失败: {json_str}")
        raise e from None


def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """
    一个装饰器，用于限制异步函数的并发调用数量。
    Args:
        max_size (int): 最大并发数。
        waitting_time (float): 当达到并发上限时，每次等待的时间（秒）。
    """
    def final_decro(func):
        """未使用async.Semaphore是为了避免使用nest-asyncio可能导致的问题。"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """
    一个装饰器，将一个普通函数包装成带有属性的EmbeddingFunc实例。
    """
    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


def load_json(file_name):
    """从文件加载JSON对象。"""
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    """将JSON对象写入文件。"""
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    """使用tiktoken将字符串编码为token列表。"""
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    """使用tiktoken将token列表解码为字符串。"""
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def pack_user_ass_to_openai_messages(*args: str):
    """
    将一系列字符串打包成OpenAI API要求的用户和助手交替的消息格式。
    """
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """
    使用多个标记（分隔符）来分割一个字符串。
    """
    if not markers:
        return [content]
    # 构建正则表达式，例如：'marker1|marker2|marker3'
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    # 移除分割后产生的空字符串和首尾空白
    return [r.strip() for r in results if r.strip()]


def clean_str(input: Any) -> str:
    """
    清理输入字符串，移除HTML转义字符、控制字符和其他不需要的字符。
    参考自官方GraphRAG实现：https://github.com/microsoft/graphrag
    """
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # 移除控制字符
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def is_float_regex(value):
    """使用正则表达式判断一个字符串是否可以表示为浮点数。"""
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", str(value)))


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """
    根据累计的token数量来截断一个数据列表。
    Args:
        list_data (list): 待截断的列表。
        key (callable): 一个函数，用于从列表项中获取要计算token的文本。
        max_token_size (int): 最大允许的token总数。
    Returns:
        list: 截断后的列表。
    """
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def list_of_list_to_csv(data: List[List[str]]) -> str:
    """将二维列表转换为CSV格式的字符串。"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(data)
    return output.getvalue()


def csv_string_to_list(csv_string: str) -> List[List[str]]:
    """将CSV格式的字符串转换为二维列表。"""
    output = io.StringIO(csv_string)
    reader = csv.reader(output)
    return [row for row in reader]


def save_data_to_file(data, file_name):
    """将数据以JSON格式保存到文件。"""
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def xml_to_json(xml_file):
    """
    解析GraphML格式的XML文件，并将其中的节点和边信息转换为JSON格式。
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        data = {"nodes": [], "edges": []}

        # GraphML的命名空间
        namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

        # 提取节点信息
        for node in root.findall(".//node", namespace):
            node_data = {
                "id": node.get("id").strip('"'),
                "entity_type": (
                    node.find("./data[@key='d0']", namespace).text.strip('"')
                    if node.find("./data[@key='d0']", namespace) is not None
                    else ""
                ),
                # ... 其他节点属性
            }
            data["nodes"].append(node_data)

        # 提取边信息
        for edge in root.findall(".//edge", namespace):
            edge_data = {
                "source": edge.get("source").strip('"'),
                "target": edge.get("target").strip('"'),
                "weight": (
                    float(edge.find("./data[@key='d3']", namespace).text)
                    if edge.find("./data[@key='d3']", namespace) is not None
                    else 0.0
                ),
                # ... 其他边属性
            }
            data["edges"].append(edge_data)
        return data
    except ET.ParseError as e:
        print(f"解析XML文件出错: {e}")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None


def safe_unicode_decode(content):
    """
    安全地将字符串中的Unicode转义序列（如 \\uXXXX）解码为实际字符。
    """
    # 匹配 \\uXXXX 格式的正则表达式
    unicode_escape_pattern = re.compile(r"\\u([0-9a-fA-F]{4})")

    def replace_unicode_escape(match):
        # 将匹配到的十六进制值转换为Unicode字符
        return chr(int(match.group(1), 16))

    # 执行替换
    decoded_content = unicode_escape_pattern.sub(
        replace_unicode_escape, content.decode("utf-8")
    )
    return decoded_content


def process_combine_contexts(hl, ll):
    """
    合并高级(hl)和低级(ll)上下文（均为CSV字符串），去重并重新格式化。
    """
    header = None
    list_hl = csv_string_to_list(hl.strip())
    list_ll = csv_string_to_list(ll.strip())

    if list_hl:
        header = list_hl[0]
        list_hl = list_hl[1:]
    if list_ll:
        if header is None:
            header = list_ll[0]
        list_ll = list_ll[1:]
    if header is None:
        return ""

    # 提取内容行（忽略ID列）
    if list_hl:
        list_hl = [",".join(item[1:]) for item in list_hl if item]
    if list_ll:
        list_ll = [",".join(item[1:]) for item in list_ll if item]

    # 合并并去重
    combined_sources_set = set(filter(None, list_hl + list_ll))

    # 重新构建带ID和表头的CSV字符串
    combined_sources = [",\t".join(header)]
    for i, item in enumerate(combined_sources_set, start=1):
        combined_sources.append(f"{i},\t{item}")

    return "\n".join(combined_sources)


def is_continuous_subsequence(subseq, seq):
    """
    检查一个小元组(subseq)是否是另一个大元组(seq)中的连续子序列。
    """
    def find_all_indexes(tup, value):
        # 查找元组中某个值的所有出现位置
        indexes = []
        start = 0
        while True:
            try:
                index = tup.index(value, start)
                indexes.append(index)
                start = index + 1
            except ValueError:
                break
        return indexes

    # 查找子序列第一个元素在大元组中的所有位置
    index_list = find_all_indexes(seq, subseq[0])
    for idx in index_list:
        # 检查是否是连续的
        if idx < len(seq) - 1 and seq[idx + 1] == subseq[-1]:
            return True
    return False


def merge_tuples(list1, list2):
    """
    合并两个元组列表，用于扩展图中的路径。
    """
    result = []
    for tup in list1:
        last_element = tup[-1]
        # 如果最后一个元素在元组前面已经出现过，说明可能形成环路，直接添加
        if last_element in tup[:-1]:
            result.append(tup)
        else:
            # 查找list2中以tup最后一个元素开头的元组，进行路径扩展
            matching_tuples = [t for t in list2 if t[0] == last_element]
            
            already_match_flag = 0
            for match in matching_tuples:
                matchh = (match[1], match[0])
                # 如果匹配的元组已经是当前路径的子序列，则跳过
                if is_continuous_subsequence(match, tup) or is_continuous_subsequence(
                    matchh, tup
                ):
                    continue
                
                already_match_flag = 1
                merged_tuple = tup + match[1:] # 拼接路径
                result.append(merged_tuple)
            
            # 如果没有找到可以拼接的路径，则保留原路径
            if not already_match_flag:
                result.append(tup)
    return result


def count_elements_in_tuple(tuple_elements, list_elements):
    """
    计算元组中有多少元素也存在于列表中。
    """
    sorted_list = sorted(list_elements)
    tuple_elements = sorted(tuple_elements)
    count = 0
    list_index = 0

    for elem in tuple_elements:
        while list_index < len(sorted_list) and sorted_list[list_index] < elem:
            list_index += 1
        if list_index < len(sorted_list) and sorted_list[list_index] == elem:
            count += 1
            list_index += 1
    return count


def cal_path_score_list(candidate_reasoning_path, maybe_answer_list):
    """
    为候选推理路径打分。分数基于路径中包含“可能答案”节点的数量。
    """
    scored_reasoning_path = {}
    for k, v in candidate_reasoning_path.items():
        score = v["Score"]
        paths = v["Path"]
        scores = {}
        for p in paths:
            # 计算路径p中与答案列表重合的节点数
            scores[p] = [count_elements_in_tuple(p, maybe_answer_list)]
        scored_reasoning_path[k] = {"Score": score, "Path": scores}
    return scored_reasoning_path


def edge_vote_path(path_dict, edge_list):
    """
    使用给定的边列表（edge_list）对路径进行投票，从而调整路径分数。
    """
    return_dict = copy.deepcopy(path_dict)
    EDGELIST = []
    pairs_append = {}
    for i in edge_list:
        EDGELIST.append((i["src_id"], i["tgt_id"]))
    
    for i in return_dict.values():
        for path_tuple, path_scores in i["Path"].items():
            if path_scores:
                count = 0
                for pairs in EDGELIST:
                    # 如果给定的边是路径的一部分，则投票数加一
                    if is_continuous_subsequence(pairs, path_tuple):
                        count += 1
                        if path_tuple not in pairs_append:
                            pairs_append[path_tuple] = [pairs]
                        else:
                            pairs_append[path_tuple].append(pairs)
                # 将投票数作为新的分数追加
                path_scores.append(count)
    return return_dict, pairs_append


def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度。"""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    # 防止除以零
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def quantize_embedding(embedding: np.ndarray | list[float], bits: int = 8):
    """
    量化嵌入向量，将浮点数向量压缩为整数向量以节省空间。
    Args:
        embedding: 原始嵌入向量。
        bits: 量化位数，默认为8位。
    Returns:
        tuple: (量化后的向量, 最小值, 最大值)
    """
    embedding = np.array(embedding)
    min_val = embedding.min()
    max_val = embedding.max()
    # 防止除以零
    if max_val == min_val:
        return np.zeros_like(embedding, dtype=np.uint8), min_val, max_val
    scale = (2**bits - 1) / (max_val - min_val)
    quantized = np.round((embedding - min_val) * scale).astype(np.uint8)
    return quantized, min_val, max_val


def dequantize_embedding(
    quantized: np.ndarray, min_val: float, max_val: float, bits=8
) -> np.ndarray:
    """
    反量化嵌入向量，将压缩后的整数向量恢复为浮点数向量。
    """
    scale = (max_val - min_val) / (2**bits - 1)
    return (quantized * scale + min_val).astype(np.float32)


def calculate_similarity(sentences, target, method="levenshtein", n=1, k=1):
    """
    计算目标字符串与一组句子之间的相似度，并返回最相似的k个句子的索引。
    支持多种相似度计算方法。
    Args:
        sentences (list[str]): 待比较的句子列表。
        target (str): 目标字符串。
        method (str): 计算方法，可选 'jaccard', 'levenshtein', 'rouge', 'bert', 'overlap', 'bleu'。
        n (int): 用于ROUGE-N中的N值。
        k (int): 返回最相似的句子数量。
    Returns:
        list[int]: 最相似的k个句子的索引列表。
    """
    target_tokens = target.lower().split()
    similarities_with_index = []
    
    if method == "jaccard":
        # Jaccard相似度：交集/并集
        for i, sentence in enumerate(sentences):
            sentence_tokens = sentence.lower().split()
            intersection = set(sentence_tokens).intersection(set(target_tokens))
            union = set(sentence_tokens).union(set(target_tokens))
            jaccard_score = len(intersection) / len(union) if union else 0
            similarities_with_index.append((i, jaccard_score))
    elif method == "levenshtein":
        # Levenshtein距离（编辑距离）
        for i, sentence in enumerate(sentences):
            distance = edit_distance(target_tokens, sentence.lower().split())
            similarities_with_index.append(
                (i, 1 - (distance / max(len(target_tokens), len(sentence.split()))))
            )
    elif method == "rouge":
        # ROUGE-N F1分数
        rouge = Rouge()
        for i, sentence in enumerate(sentences):
            scores = rouge.get_scores(sentence, target)
            rouge_score = scores[0].get(f"rouge-{n}", {}).get("f", 0)
            similarities_with_index.append((i, rouge_score))
    elif method == "bert":
        # 基于BERT的语义相似度
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(sentences + [target])
        target_vec = embeddings[-1]
        similarities_with_index = [(i, cosine_similarity(embeddings[i], target_vec))
                                   for i in range(len(sentences))]
    elif method == "overlap":
        # 重叠系数
        for i, sentence in enumerate(sentences):
            sentence_tokens = set(sentence.lower().split())
            overlap = sentence_tokens.intersection(set(target_tokens))
            score = len(overlap) / min(len(sentence_tokens), len(target_tokens)) if sentence_tokens else 0
            similarities_with_index.append((i, score))
    elif method == "bleu":
        # BLEU分数
        smooth_fn = SmoothingFunction().method1
        target_tokens_bleu = word_tokenize(target.lower())
        for i, sentence in enumerate(sentences):
            sentence_tokens_bleu = word_tokenize(sentence.lower())
            score = sentence_bleu([target_tokens_bleu], sentence_tokens_bleu, smoothing_function=smooth_fn)
            similarities_with_index.append((i, score))
    else:
        raise ValueError("不支持的方法。请从 'jaccard', 'levenshtein', 'rouge', 'bert', 'overlap', 'bleu' 中选择。")

    # 按相似度得分降序排序
    similarities_with_index.sort(key=lambda x: x[1], reverse=True)
    # 返回前k个最相似句子的索引
    return [index for index, score in similarities_with_index[:k]]