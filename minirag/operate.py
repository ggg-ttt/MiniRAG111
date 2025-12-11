# 本文件为MiniRAG核心操作模块，包含分块、实体关系抽取、查询等主要流程实现。
# 主要涉及知识图谱、向量数据库、KV存储等多种数据结构的协同操作。
# 适用于RAG（Retrieval-Augmented Generation）场景下的本地与全局检索、混合推理等。

import asyncio
import json
import re
from typing import Union
from collections import Counter, defaultdict
import warnings
import json_repair

from .utils import (
    list_of_list_to_csv,  # 列表转CSV字符串
    truncate_list_by_token_size,  # 按token数截断列表
    split_string_by_multi_markers,  # 按多种分隔符切分字符串
    logger,  # 日志工具
    locate_json_string_body_from_string,  # 从字符串中定位JSON体
    process_combine_contexts,  # 合并上下文内容
    clean_str,  # 清洗字符串
    edge_vote_path,  # 边路径投票
    encode_string_by_tiktoken,  # tiktoken编码
    decode_tokens_by_tiktoken,  # tiktoken解码
    is_float_regex,  # 判断字符串是否为浮点数
    pack_user_ass_to_openai_messages,  # 打包对话历史
    compute_mdhash_id,  # 计算哈希ID
    calculate_similarity,  # 计算相似度
    cal_path_score_list,  # 路径打分
)
from .base import (
    BaseGraphStorage,  # 图存储基类
    BaseKVStorage,     # KV存储基类
    BaseVectorStorage, # 向量存储基类
    TextChunkSchema,   # 文本块结构
    QueryParam,        # 查询参数结构
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS  # 分隔符与提示词


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    """
    按token数对长文本进行智能分块，支持重叠窗口机制
    
    该函数是MiniRAG文档处理的核心组件之一，负责将长文本按照指定的token限制进行切分。
    通过重叠窗口机制，确保相邻文本块之间的语义连续性，避免重要信息被截断。
    分块结果包含每个块的基本信息，为后续的实体抽取和向量化提供基础。
    
    Args:
        content (str): 待分块的原始文本内容，支持任意长度的中文或英文文本
        
        overlap_token_size (int, optional): 文本块之间的重叠token数量，默认128
            - 作用：确保相邻块之间的语义连续性，避免关键信息被分割
            - 建议：对于需要高召回率的场景，可适当增加重叠token数量
            - 注意：重叠越多，存储和计算成本越高
            
        max_token_size (int, optional): 每个文本块的最大token数量，默认1024
            - 作用：限制单次处理的文本长度，避免超出LLM输入限制
            - 建议：根据下游任务的token限制和性能要求调整
            - 注意：过小的分块可能导致语义不完整
            
        tiktoken_model (str, optional): 用于token计算的模型名称，默认"gpt-4o"
            - 作用：确保token计算的准确性和一致性
            - 支持：gpt-4、gpt-3.5-turbo等OpenAI模型
            - 注意：不同模型的tokenization方式可能略有差异
    
    Returns:
        list[dict]: 分块结果列表，每个元素包含以下字段：
            - tokens (int): 当前块的token数量
            - content (str): 分块后的文本内容（已去除首尾空格）
            - chunk_order_index (int): 当前块在原文本中的顺序索引（从0开始）
    
    算法逻辑:
        1. 使用指定的tiktoken模型对输入文本进行tokenization
        2. 按照 max_token_size - overlap_token_size 的步长遍历tokens
        3. 对每个窗口内的tokens进行截取和decode
        4. 计算实际的token数量并构建结果字典
    
    使用示例:
        >>> # 基本用法
        >>> content = "这是一个很长的文档内容..."
        >>> chunks = chunking_by_token_size(content)
        >>> print(f"共分块 {len(chunks)} 个")
        
        >>> # 自定义参数
        >>> chunks = chunking_by_token_size(
        ...     content, 
        ...     overlap_token_size=256, 
        ...     max_token_size=2048,
        ...     tiktoken_model="gpt-3.5-turbo"
        ... )
        
        >>> # 处理分块结果
        >>> for chunk in chunks:
        ...     print(f"块 {chunk['chunk_order_index']}: {chunk['tokens']} tokens")
        ...     print(f"内容: {chunk['content'][:100]}...")
    
    注意事项:
        - 函数会自动处理文本的边界条件，最后一块可能不足max_token_size
        - 重叠区域可能导致重复计算，但对保持语义连续性很重要
        - token计算基于指定的模型，不同模型可能有不同的计算结果
        - 对于极短文本，可能只返回一个包含全部内容的块
    """
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """
    对实体或关系描述进行智能摘要处理（当描述长度超限时）
    
    该函数是MiniRAG实体关系抽取流程中的重要组件，负责对过长描述进行摘要压缩。
    通过智能压缩确保实体和关系信息能够适配存储和处理的token限制，
    同时保持核心语义信息的完整性。
    
    Args:
        entity_or_relation_name (str): 待摘要的实体或关系的名称
            - 用于标识当前处理的实体或关系
            - 帮助在日志和调试中追踪处理过程
            - 可能包含中文、英文或混合命名
            
        description (str): 需要摘要的原始描述文本
            - 包含实体或关系的详细描述信息
            - 可能包含冗余、重复或过于详细的信息
            - 需要根据token限制进行智能压缩
            
        global_config (dict): 全局配置参数，包含摘要相关的设置
            - tiktoken_model_name: 用于token计算的模型名称
            - entity_summary_to_max_tokens: 摘要后的最大token限制
            - 其他可能影响摘要处理的配置项
    
    Returns:
        str: 摘要处理后的描述文本
            - 如果原描述不超过token限制，返回原始描述
            - 如果超限，返回智能摘要压缩后的版本
            - 保证核心语义信息得到保留
    
    算法逻辑:
        1. 从全局配置中获取token计算模型和最大限制
        2. 使用tiktoken对描述文本进行tokenization
        3. 检查token数量是否超过预设限制
        4. 如果未超限，直接返回原始描述
        5. 如果超限，调用LLM进行智能摘要（当前实现中此部分被注释）
        
    当前状态:
        - 函数已实现token超限检查逻辑
        - 实际的LLM摘要调用部分暂时被注释
        - 目前当token未超限时直接返回原始描述
        
    使用场景:
        - 实体抽取后对长描述进行压缩存储
        - 关系抽取中对关系描述进行长度控制
        - 知识图谱构建中的信息压缩优化
        - 减少存储空间和提升检索效率
        
    注意事项:
        - 摘要过程需要权衡信息完整性和长度限制
        - 不同类型的实体可能需要不同的摘要策略
        - 摘要质量直接影响后续检索和推理的效果
        - 建议在实际使用时启用LLM摘要功能以获得更好效果
    """
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # 不超限则直接返回
        return description


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    """
    解析并标准化单条实体抽取结果，转换为统一的实体数据结构
    
    该函数是实体抽取管道中的关键解析组件，负责将LLM抽取的原始实体记录
    转换为标准化的实体数据结构，确保后续处理的一致性和可靠性。
    通过严格的数据验证和清洗流程，保证实体信息的质量和格式统一性。
    
    Args:
        record_attributes (list[str]): LLM抽取的实体属性列表
            - 预期格式：['"entity"', entity_name, entity_type, description, ...]
            - 包含实体类型标识、名称、类型和描述等核心字段
            - 可能包含其他可选属性或元数据
            
        chunk_key (str): 实体来源文本块的唯一标识键
            - 用于追溯实体信息的原始文本位置
            - 支持实体到源文档的溯源追踪
            - 格式通常为文档ID_chunk_index的组合
    
    Returns:
        dict[str, str] | None: 标准化实体信息字典，包含以下字段：
            - entity_name (str): 清洗后的实体名称（已转大写）
            - entity_type (str): 清洗后的实体类型（已转大写）
            - description (str): 清洗后的实体描述
            - source_id (str): 来源文本块的标识键
            - 如果解析失败或数据无效，返回None
    
    解析逻辑:
        1. 验证记录格式：检查属性列表长度和第一个元素是否为'"entity"'
        2. 实体名称提取：提取第二个元素作为实体名称并进行清洗
        3. 名称有效性检查：确保实体名称不为空或只包含空白字符
        4. 实体类型提取：提取第三个元素作为实体类型并进行清洗
        5. 描述信息提取：提取第四个元素作为实体描述
        6. 数据标准化：所有文本字段都经过clean_str函数清洗
        7. 结构构建：组装成标准化的实体字典格式
    
    数据清洗规则:
        - 去除首尾空白字符
        - 移除不必要的引号和特殊字符
        - 统一大小写处理（名称和类型转大写）
        - 处理编码和格式问题
        - 过滤无效或危险的字符
    
    错误处理:
        - 属性列表长度不足4个元素：返回None
        - 非实体类型记录（不是'"entity"'开头）：返回None
        - 空实体名称或无效名称：返回None
        - 解析过程中的任何异常都会被捕获并返回None
    
    使用场景:
        - 从LLM输出中解析实体信息
        - 实体抽取管道的标准化处理环节
        - 批量实体处理的并行处理单元
        - 实体信息的质量控制和验证
    
    注意事项:
        - 函数是异步的，支持并发处理
        - 输入的record_attributes必须按预期格式组织
        - 实体名称和类型的清洗会影响后续匹配和合并
        - source_id的完整性对溯源功能至关重要
        - 返回的None值需要在上层调用中妥善处理
    """
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # 组装实体节点
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    """
    解析并标准化单条关系抽取结果，转换为统一的关系数据结构
    
    该函数是关系抽取管道中的核心解析组件，负责将LLM抽取的原始关系记录
    转换为标准化的关系数据结构。关系作为连接实体的桥梁，其数据质量直接影响
    知识图谱的完整性和查询效果。此函数通过严格的数据验证和标准化流程，
    确保关系信息的准确性和一致性。
    
    Args:
        record_attributes (list[str]): LLM抽取的关系属性列表
            - 预期格式：['"relationship"', source_entity, target_entity, description, keywords, weight]
            - 包含关系类型标识、源实体、目标实体、描述和关键词等字段
            - 最后一个元素应为数值权重，可能包含浮点数
            - 至少需要5个元素才能构成有效的关系记录
            
        chunk_key (str): 关系来源文本块的唯一标识键
            - 用于追溯关系信息的原始文本位置
            - 支持关系到源文档的溯源追踪
            - 格式通常为文档ID_chunk_index的组合
    
    Returns:
        dict[str, Union[str, float]] | None: 标准化关系信息字典，包含以下字段：
            - src_id (str): 清洗后的源实体ID（已转大写）
            - tgt_id (str): 清洗后的目标实体ID（已转大写）
            - weight (float): 关系权重，支持浮点数，默认1.0
            - description (str): 清洗后的关系描述
            - keywords (str): 清洗后的关系关键词
            - source_id (str): 来源文本块的标识键
            - 如果解析失败或数据无效，返回None
    
    解析逻辑:
        1. 验证记录格式：检查属性列表长度和第一个元素是否为'"relationship"'
        2. 源实体提取：提取第二个元素作为源实体名称并清洗转大写
        3. 目标实体提取：提取第三个元素作为目标实体名称并清洗转大写
        4. 关系描述提取：提取第四个元素作为关系描述
        5. 关键词提取：提取第五个元素作为关系关键词
        6. 权重处理：提取最后一个元素，尝试转换为浮点数，失败则使用1.0
        7. 数据标准化：所有文本字段都经过clean_str函数清洗
        8. 结构构建：组装成标准化的关系字典格式
    
    权重处理:
        - 最后一个元素被视为关系权重
        - 使用is_float_regex函数验证是否为有效浮点数
        - 有效权重保持原值，无效权重使用默认值1.0
        - 权重用于后续的关系重要性评估和排序
    
    数据清洗规则:
        - 去除首尾空白字符
        - 移除不必要的引号和特殊字符
        - 实体名称统一转大写（便于匹配和合并）
        - 处理编码和格式问题
        - 过滤无效或危险的字符
    
    实体ID标准化:
        - 源实体和目标实体ID都转换为大写
        - 确保实体ID的一致性和匹配性
        - 避免因大小写差异导致的实体分离
    
    错误处理:
        - 属性列表长度不足5个元素：返回None
        - 非关系类型记录（不是'"relationship"'开头）：返回None
        - 权重转换失败：使用默认值1.0（不返回None）
        - 解析过程中的其他异常会被捕获并返回None
    
    使用场景:
        - 从LLM输出中解析关系信息
        - 关系抽取管道的标准化处理环节
        - 批量关系处理的并行处理单元
        - 知识图谱构建中的关系数据处理
        - 关系质量控制和验证
    
    注意事项:
        - 函数是异步的，支持并发处理
        - 输入的record_attributes必须按预期格式组织
        - 实体ID的大小写一致性对后续处理很重要
        - source_id的完整性对关系溯源功能至关重要
        - 权重信息的准确性影响关系排序和筛选
        - 返回的None值需要在上层调用中妥善处理
        - 关系描述和关键词是后续相似度计算的重要依据
    """
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # 组装边信息
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    合并同名实体节点的所有属性信息并更新到知识图谱存储中
    
    该函数是知识图谱构建过程中的核心合并组件，负责处理同名实体的属性融合。
    在实体抽取过程中，相同的实体可能在多个文本块中被识别和描述，
    此函数通过智能合并策略，将所有相关属性整合为一个统一的实体节点，
    确保知识图谱的一致性和完整性。
    
    Args:
        entity_name (str): 待处理的实体名称（已标准化）
            - 作为实体的唯一标识符
            - 通常为清洗后的大写格式
            - 用于在知识图谱中定位和更新对应节点
            
        nodes_data (list[dict]): 包含同名实体的数据列表
            - 每个元素包含该实体在不同文本块中的信息
            - 可能包含不同的类型、描述和来源ID组合
            - 需要通过合并算法整合所有相关信息
            - 数据格式：{'entity_type': str, 'description': str, 'source_id': str}
            
        knowledge_graph_inst (BaseGraphStorage): 知识图谱存储实例
            - 提供节点查询、插入和更新功能
            - 支持并发操作和数据持久化
            - 必须实现upsert_node方法
            
        global_config (dict): 全局配置参数
            - 包含摘要处理、token限制等配置信息
            - 可能影响数据合并和存储策略
            - 为future功能预留配置接口
    
    Returns:
        dict[str, str]: 合并后的实体数据字典，包含以下字段：
            - entity_name (str): 实体名称
            - entity_type (str): 合并后的实体类型（出现频次最高）
            - description (str): 合并后的实体描述
            - source_id (str): 合并后的来源ID集合
    
    合并策略:
        1. 实体类型合并：
           - 统计所有nodes_data中实体类型的出现频次
           - 选择出现频次最高的类型作为主要类型
           - 处理边界情况：当列表为空时的默认值
        
        2. 描述信息合并：
           - 收集所有nodes_data中的描述信息
           - 使用GRAPH_FIELD_SEP作为分隔符连接
           - 去重处理，保留唯一描述
           - 按字母序排序以确保一致性
        
        3. 来源ID合并：
           - 收集所有nodes_data中的source_id
           - 使用split_string_by_multi_markers解析已有source_id
           - 去重处理，避免重复计算
           - 合并所有相关来源
        
        4. 已存在节点处理：
           - 查询知识图谱中是否已存在该实体
           - 如果存在，合并已存在节点的属性信息
           - 确保历史数据和新数据的完整融合

    
    图存储操作:
        1. 检查已存在节点：调用get_node方法获取历史数据
        2. 提取历史属性：解析已存在节点的类型、描述和来源
        3. 数据合并：将历史数据与新数据按规则合并
        4. 节点更新：调用upsert_node方法保存合并结果
        5. 结果返回：返回最终合并的节点数据
    采用了以下策略来保证数据质量：
    抗噪：通过“少数服从多数”的投票机制，过滤掉 LLM 偶尔产生的错误实体分类。
    不遗忘：通过拼接描述，确保新旧知识都被保留，而不是直接覆盖。
    幂等性：通过 set 去重和 sorted 排序，确保多次运行相同数据不会导致数据无限膨胀或顺序混乱。
    
    """
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    # 统计出现最多的实体类型
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    # 合并描述和来源ID
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )

    # description = await _handle_entity_relation_summary(
    #     entity_name, description, global_config
    # )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    合并同一对实体间的所有关系属性并更新到知识图谱存储中
    
    该函数是知识图谱构建过程中的关系合并核心组件，负责处理两个实体间的所有关系信息融合。
    在关系抽取过程中，同一对实体间可能存在多条不同描述、不同权重或不同来源的关系记录。
    此函数通过智能合并策略，将所有相关属性整合为一条统一的关系边，
    确保知识图谱的关系一致性和完整性，同时优化存储效率。
    
    Args:
        src_id (str): 关系源实体的标准化ID
            - 关系的起始实体标识符
            - 通常为清洗后的大写格式
            - 用于在知识图谱中定位源节点
            
        tgt_id (str): 关系目标实体的标准化ID
            - 关系的终止实体标识符
            - 通常为清洗后的大写格式
            - 用于在知识图谱中定位目标节点
            
        edges_data (list[dict]): 包含同一对实体间所有关系的列表
            - 每个元素包含该关系在不同文本块中的信息
            - 可能包含不同的权重、描述、关键词和来源ID组合
            - 需要通过合并算法整合所有相关信息
            - 数据格式：{'weight': float, 'description': str, 'keywords': str, 'source_id': str}
            
        knowledge_graph_inst (BaseGraphStorage): 知识图谱存储实例
            - 提供边查询、插入和更新功能
            - 支持并发操作和数据持久化
            - 必须实现has_edge、get_edge、upsert_edge、has_node、upsert_node等方法
            
        global_config (dict): 全局配置参数
            - 包含摘要处理、token限制等配置信息
            - 可能影响数据合并和存储策略
            - 为future功能预留配置接口
    
    Returns:
        dict[str, str]: 合并后的关系数据字典，包含以下字段：
            - src_id (str): 关系源实体ID
            - tgt_id (str): 关系目标实体ID
            - description (str): 合并后的关系描述
            - keywords (str): 合并后的关系关键词
    
    合并策略:
        1. 权重合并：
           - 计算所有关系权重的总和
           - 如果存在历史权重，累加到历史权重上
           - 权重数值可能为浮点数
           - 总权重反映关系的整体重要性和强度
        
        2. 描述信息合并：
           - 收集所有edges_data中的描述信息
           - 使用GRAPH_FIELD_SEP作为分隔符连接
           - 去重处理，保留唯一描述
           - 按字母序排序以确保一致性
        
        3. 关键词合并：
           - 收集所有edges_data中的关键词
           - 使用GRAPH_FIELD_SEP作为分隔符连接
           - 去重处理，避免重复关键词
           - 按字母序排序以确保一致性
        
        4. 来源ID合并：
           - 收集所有edges_data中的source_id
           - 使用split_string_by_multi_markers解析已有source_id
           - 去重处理，避免重复计算
           - 合并所有相关来源以保持完整的溯源信息
        
        5. 已存在边处理：
           - 查询知识图谱中是否已存在该关系
           - 如果存在，合并已存在边的属性信息
           - 确保历史数据和新数据的完整融合
    
    节点处理逻辑:
        在更新边之前，函数会确保源节点和目标节点都存在于知识图谱中：
        1. 检查源节点是否存在：调用has_node(src_id)
        2. 检查目标节点是否存在：调用has_node(tgt_id)
        3. 如果任一节点不存在，创建默认节点：
           - entity_type: '"UNKNOWN"'
           - source_id: 当前合并后的source_id
           - description: 当前合并后的description
        4. 使用upsert_node方法插入或更新节点
    
    数据结构处理:
        - 权重：数值累加，反映关系统计强度
        - 描述信息：去重排序后用分隔符连接
        - 关键词：去重排序后用分隔符连接
        - 来源ID：解析已有ID，合并新ID，去重存储
        - 异常处理：为各种边界情况提供默认值
    
    图存储操作:
        1. 检查已存在边：调用has_edge方法查询历史关系
        2. 提取历史属性：解析已存在边的权重、描述、关键词和来源
        3. 数据合并：将历史数据与新数据按规则合并
        4. 节点保障：确保源节点和目标节点存在，必要时创建默认节点
        5. 边更新：调用upsert_edge方法保存合并结果
        6. 结果返回：返回最终合并的边数据
    
    性能优化:
        - 权重累加计算，时间复杂度O(n)
        - 描述和关键词合并使用集合去重，效率较高
        - 批量处理多个关系的合并操作
        - 并发执行：支持与并发关系处理结合
        - 内存管理：及时释放临时数据结构
    
    使用场景:
        - 关系抽取后的批量合并处理
        - 知识图谱构建中的关系融合
        - 多文档关系信息的统一管理
        - 增量更新和去重操作
        - 关系权重累积和强度计算
    
    注意事项:
        - 函数是异步的，支持并发处理多个关系
        - 实体ID必须标准化，否则可能导致关系创建失败
        - 合并策略的选择会影响最终的知识图谱质量
        - source_id的完整性对关系追溯功能很重要
        - 大量关系的合并可能消耗较多内存
        - 权重累加可能导致权重值过大，需要考虑范围限制
        - 建议在批量处理时监控内存使用情况
        - 默认节点创建可能会影响知识图谱的质量评估
    """
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    # description = await _handle_entity_relation_summary(
    #     (src_id, tgt_id), description, global_config
    # )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    """
    从文本块中智能提取实体和关系，构建完整的知识图谱和向量索引
    
    这是MiniRAG系统的核心函数之一，负责将原始文本块转换为结构化的知识表示。
    通过大语言模型的强大能力，从非结构化文本中抽取实体、关系和属性信息，
    并构建知识图谱和多层次向量索引，为后续的智能检索和问答提供基础。
    
    函数采用并发处理架构，支持大规模文档的高效处理。通过多轮对话和迭代优化，
    确保实体和关系抽取的准确性和完整性。同时集成了去重、合并和标准化机制，
    保证最终构建的知识图谱质量。
    
    Args:
        chunks (dict[str, TextChunkSchema]): 待处理的文本块字典
            - 键：文本块的唯一标识符（chunk_key）
            - 值：TextChunkSchema对象，包含content等字段
            - 通常来源于chunking_by_token_size函数的输出
            - 每个块代表文档的一个语义片段
            
        knowledge_graph_inst (BaseGraphStorage): 知识图谱存储实例
            - 提供图结构的存储、查询和更新功能
            - 支持节点和边的增加、删除、修改操作
            - 必须实现upsert_node、upsert_edge、get_node等方法
            - 可以是内存图、Neo4j、网络图等多种实现
            
        entity_vdb (BaseVectorStorage): 实体内容向量数据库
            - 存储实体的文本描述信息用于相似度检索
            - 向量化内容：实体名称 + 实体描述
            - 用于基于内容相似度的实体搜索
            - 可选参数，为None时跳过实体内容索引
            
        entity_name_vdb (BaseVectorStorage): 实体名称向量数据库
            - 专门存储实体名称用于精确匹配
            - 向量化内容：仅包含实体名称
            - 用于基于名称的实体检索和匹配
            - 可选参数，为None时跳过实体名称索引
            
        relationships_vdb (BaseVectorStorage): 关系向量数据库
            - 存储关系的文本描述和关键词信息
            - 向量化内容：关键词 + 源实体 + 目标实体 + 关系描述
            - 用于基于语义相似度的关系搜索
            - 可选参数，为None时跳过关系索引
            
        global_config (dict): 全局配置参数字典
            - llm_model_func：用于实体关系抽取的大语言模型函数
            - entity_extract_max_gleaning：迭代抽取的最大轮数
            - tiktoken_model_name：token计算模型名称（用于长度限制）
            - entity_summary_to_max_tokens：实体摘要的最大token数
            - 其他影响处理流程的配置参数
    
    Returns:
        BaseGraphStorage | None: 更新后的知识图谱实例或None
            - 成功处理：返回更新后的knowledge_graph_inst
            - 失败处理：返回None，可能的原因：
              * 未提取到任何实体（LLM可能未正常工作）
              * 未提取到任何关系（LLM可能未正常工作）
              * 知识图谱实例异常或不可用
    
    核心处理流程:
        1. 初始化阶段：
           - 从全局配置中提取LLM模型和抽取参数
           - 构建抽取提示词和上下文信息
           - 初始化处理进度统计变量
            
        2. 并发处理阶段（_process_single_content内部函数）：
           - 对每个文本块独立进行LLM抽取
           - 多轮迭代优化（gleaning机制）
           - 解析LLM输出，提取实体和关系
           - 数据清洗和标准化处理
           
        3. 合并整合阶段：
           - 收集所有文本块的抽取结果
           - 合并同名列体（同名列体合并）
           - 合并同一对实体间的关系（关系去重）
           - 批量更新知识图谱
            
        4. 向量索引阶段：
           - 构建实体内容向量索引
           - 构建实体名称向量索引
           - 构建关系向量索引
           - 支持后续的相似度检索
            
        5. 质量验证阶段：
           - 检查是否成功提取实体和关系
           - 记录警告信息用于问题诊断
           - 返回最终的知识图谱实例
    
    LLM抽取机制:
        1. 初始抽取：
           - 使用预定义的entity_extraction提示词
           - 注入上下文信息和文本内容
           - 获取初始的实体关系列表
            
        2. 迭代优化（gleaning）：
           - 根据entity_extract_max_gleaning参数进行多轮抽取
           - 每轮使用continue_prompt继续抽取
           - 维护对话历史以获得上下文信息
           - 使用if_loop_prompt判断是否继续下一轮
            
        3. 智能终止：
           - 基于LLM判断是否需要继续抽取
           - 避免无效的重复抽取
           - 优化处理效率和资源使用
    
    数据解析流程:
        1. 记录分割：
           - 使用record_delimiter和completion_delimiter分割LLM输出
           - 提取结构化的实体和关系记录
            
        2. 格式验证：
           - 使用正则表达式提取记录内容
           - 验证记录格式的完整性和有效性
            
        3. 属性提取：
           - 使用tuple_delimiter分割记录属性
           - 分别调用实体和关系专用解析函数
           - 执行数据清洗和标准化处理
            
        4. 分类存储：
           - 将实体按名称分组到maybe_nodes字典
           - 将关系按实体对分组到maybe_edges字典
           - 为后续的合并操作做准备
    
    并发处理优化:
        - 使用asyncio.gather实现真正的并发处理
        - 每个文本块独立处理，互不依赖
        - LLM调用受信号量控制，避免过载
        - 实时进度反馈，支持大规模文档处理
    
    合并算法:
        1. 实体合并：
           - 按实体名称分组处理
           - 合并实体类型（选择最高频类型）
           - 合并描述信息（去重排序）
           - 合并来源ID（去重合并）
           - 批量更新知识图谱节点
            
        2. 关系合并：
           - 按实体对分组处理
           - 累加关系权重
           - 合并关系描述和关键词
           - 合并来源ID（去重合并）
           - 批量更新知识图谱边
    
    向量索引构建:
        1. 实体内容索引：
           - 文档格式：{"content": name + description, "entity_name": name}
           - ID前缀："ent-"
           - 支持基于内容相似度的实体搜索
            
        2. 实体名称索引：
           - 文档格式：{"content": name, "entity_name": name}
           - ID前缀："Ename-"
           - 支持精确的实体名称匹配
            
        3. 关系索引：
           - 文档格式：{"src_id": src, "tgt_id": tgt, "content": keywords + src + tgt + description}
           - ID前缀："rel-"
           - 支持基于语义的关系搜索
    
    性能特性:
        - 并发处理：充分利用异步IO和并发能力
        - 内存优化：及时释放临时数据结构
        - 进度跟踪：实时显示处理进度和统计信息
        - 错误容错：单个文本块失败不影响整体处理
        - 可扩展性：支持大规模文档批量处理
    
    使用场景:
        - 文档知识图谱构建
        - 非结构化文本的结构化转换
        - RAG系统的知识索引构建
        - 实体关系抽取和知识挖掘
        - 多模态数据的信息抽取
    
    注意事项:
        - LLM模型的选择直接影响抽取质量
        - 大规模文档处理时注意内存和计算资源
        - 向量数据库的构建是异步的，需要等待完成
        - 建议在生产环境中添加更详细的重试和错误处理
        - 进度显示使用print，可能需要在生产环境中改为日志
        - 当前实现中有重复的entity_vdb更新代码，需要注意
        - 大量并发LLM调用可能需要考虑API限制和成本控制
        - 建议监控处理时间和资源使用情况
    """
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
    # if global_config['RAGmode'] == 'minirag':
    #     # entity_extract_prompt = PROMPTS["entity_extraction_noDes"]
    #     entity_extract_prompt = PROMPTS["entity_extraction"]
    # else:
    entity_extract_prompt = PROMPTS["entity_extraction"]

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]

    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """
        异步处理单个文本块，进行实体和关系的多轮迭代抽取
        
        该内部函数是extract_entities函数的核心处理单元，负责对单个文本块进行
        完整的实体和关系抽取流程。函数采用多轮迭代优化机制，通过LLM的智能分析
        从非结构化文本中提取结构化的知识表示，体现了MiniRAG系统的核心技术能力。
        
        多轮迭代机制（Gleaning）通过智能终止判断，逐步完善抽取结果，确保
        实体和关系的完整性和准确性。这种设计显著提高了抽取质量，特别适合
        处理复杂文本和长文档。
        
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]): 文本块元组
                - 第一个元素：chunk_key，文本块的唯一标识符
                - 第二个元素：chunk_dp，TextChunkSchema对象，包含content等字段
                - chunk_key格式通常为"文档路径::起始位置::结束位置"
                - chunk_dp包含原始文本内容和元数据信息
                
        Returns:
            tuple[dict, dict]: 抽取结果元组
                - 第一个元素：实体字典，键为实体名称，值为实体数据列表
                  * 可能包含多个同名实体的不同描述和来源
                  * 用于后续的实体合并和去重处理
                - 第二个元素：关系字典，键为(src_id, tgt_id)元组，值为关系数据列表
                  * 可能包含同一对实体间的多个关系记录
                  * 用于后续的关系合并和去重处理
        
        多轮迭代抽取流程:
            1. 初始抽取阶段：
               - 使用entity_extract_prompt构建初始提示词
               - 将context_base和input_text注入提示词模板
               - 调用use_llm_func进行第一轮实体关系抽取
               - 获得初步的抽取结果final_result
               
            2. 对话历史维护：
               - 使用pack_user_ass_to_openai_messages打包对话历史
               - 构建包含上下文的完整对话记录
               - 为后续轮次提供抽取上下文信息
               - 确保多轮抽取的连贯性和完整性
               
            3. 迭代优化循环：
               - 根据entity_extract_max_gleaning参数进行多轮抽取
               - 每轮使用continue_prompt继续抽取剩余内容
               - 累加抽取结果到final_result
               - 智能判断是否需要继续下一轮抽取
               
            4. 智能终止判断：
               - 在非最后一轮使用if_loop_prompt询问是否继续
               - 基于LLM判断结果决定是否退出循环
               - 避免无效的重复抽取和资源浪费
               - 提高处理效率和成本效益
        
        记录解析和数据提取:
            1. 结果分割：
               - 使用split_string_by_multi_markers解析LLM输出
               - 按record_delimiter和completion_delimiter分割记录
               - 支持多种可能的输出格式和分隔符
               
            2. 正则表达式提取：
               - 使用re.search提取记录中的"(...)"部分
               - 匹配并提取括号内的属性信息
               - 过滤掉格式不正确的记录
               
            3. 属性列表解析：
               - 使用tuple_delimiter分割record_attributes
               - 将字符串转换为结构化的属性列表
               - 为后续的类型判断和解析做准备
               
            4. 实体和关系分类：
               - 调用_handle_single_entity_extraction尝试解析为实体
               - 调用_handle_single_relationship_extraction尝试解析为关系
               - 根据解析结果进行正确的分类和存储
               - 支持混合类型的记录处理
        
        数据结构构建:
            1. 实体数据结构（maybe_nodes）：
               - 使用defaultdict(list)存储同名实体的多个记录
               - 键：实体名称（字符串）
               - 值：该实体的所有记录列表
               - 支持实体合并和去重的后续处理
               
            2. 关系数据结构（maybe_edges）：
               - 使用defaultdict(list)存储同一对实体间的多个关系
               - 键：(src_id, tgt_id)元组
               - 值：该关系的所有记录列表
               - 支持关系合并和去重的后续处理
        
        进度统计和输出:
            1. 处理计数器更新：
               - already_processed：已处理的文本块数量
               - already_entities：已发现的实体数量（去重前）
               - already_relations：已发现的关系数量（去重前）
               
            2. 实时进度显示：
               - 使用process_tickers提供动态进度指示器
               - 循环显示不同的处理符号
               - 使用\r实现行内更新，不产生新行
               - flush=True确保输出立即显示
        
        错误处理和容错:
            1. 记录格式验证：
               - 检查正则表达式匹配结果
               - 跳过格式不正确的记录
               - 不因个别记录失败影响整体处理
               
            2. 解析结果容错：
               - 实体或关系解析失败不影响其他记录
               - 使用条件判断确保数据有效性
               - 返回空列表而不是抛出异常
        
        性能优化特性:
            - 异步并发处理能力
            - 智能终止机制减少无效计算
            - 内存友好的数据结构设计
            - 高效的字符串处理和正则匹配
            - 实时进度反馈提升用户体验
        
        使用场景:
            - 批量文档的实体关系抽取
            - 知识图谱的自动化构建
            - 非结构化文本的结构化处理
            - RAG系统的知识库构建
            - 大规模文本挖掘和知识提取
        
        """
        # 使用nonlocal关键字访问闭包变量，这些变量用于跟踪处理进度
        nonlocal already_processed, already_entities, already_relations
        # 解包chunk_key_dp元组，获取文本块的唯一标识符
        chunk_key = chunk_key_dp[0]  # 文本块的唯一标识符，用于追踪源文本位置
        # 获取文本块的详细数据字典
        chunk_dp = chunk_key_dp[1]   # 包含文本内容和其他元数据的字典
        # 提取文本块的实际内容
        content = chunk_dp["content"]
        # 根据上下文基础信息和输入文本格式化实体提取提示词
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        # 调用LLM函数执行实体提取，获取初始提取结果
        final_result = await use_llm_func(hint_prompt)

        # 打包对话历史，用于后续的补充提取和上下文理解
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        # 多次补充提取循环，最多执行entity_extract_max_gleaning次
        for now_glean_index in range(entity_extract_max_gleaning):
            # 调用LLM进行补充提取，基于已有的对话历史
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            # 更新对话历史和最终结果
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            # 检查是否已达到最大提取次数
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            # 向LLM询问是否还有更多实体或关系需要提取
            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            # 清理响应结果，移除可能的引号并转为小写，便于比较
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            # 如果不需要继续提取，提前结束循环
            if if_loop_result != "yes":
                break

        # 根据配置的分隔符分割LLM返回的结构化结果
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        # 初始化字典用于存储可能的实体和关系
        maybe_nodes = defaultdict(list)  # 键为实体名称，值为实体信息列表
        maybe_edges = defaultdict(list)  # 键为(源节点ID,目标节点ID)元组，值为关系信息列表
        # 遍历每条记录，解析实体和关系信息
        for record in records:
            # 使用正则表达式提取括号中的内容，匹配结构化数据
            record = re.search(r"\((.*)\)", record)
            # 如果没有找到匹配项，跳过当前记录
            if record is None:
                continue
            # 获取括号中的内容
            record = record.group(1)
            # 根据元组分隔符分割记录属性
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            # 尝试将记录处理为实体
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            # 如果成功处理为实体，添加到maybe_nodes字典
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            # 如果不是实体，则尝试处理为关系
            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            # 如果成功处理为关系，添加到maybe_edges字典
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        # 更新处理计数
        already_processed += 1  # 已处理的文本块数量加1
        already_entities += len(maybe_nodes)  # 累加提取的实体数量（可能有重复）
        already_relations += len(maybe_edges)  # 累加提取的关系数量（可能有重复）
        # 选择当前进度指示器图标
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        # 打印进度信息，使用\r覆盖当前行以保持进度条在同一行
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",  # 不换行
            flush=True,  # 立即刷新输出
        )
        # 返回提取的实体和关系字典
        return dict(maybe_nodes), dict(maybe_edges)

    # 注意：use_llm_func被asyncio.Semaphore包装，限制了最大并发调用数量，防止API过载
    # 并发处理所有文本块，提高处理效率
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # 输出空行，清除进度条
    # 初始化全局的实体和关系字典，用于合并所有文本块的结果
    maybe_nodes = defaultdict(list)  # 全局实体字典，键为实体名称
    maybe_edges = defaultdict(list)  # 全局关系字典，键为节点ID对
    # 合并所有文本块的处理结果
    for m_nodes, m_edges in results:
        # 合并实体信息
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        # 合并关系信息，对节点ID进行排序以确保一致性（避免(a,b)和(b,a)被视为不同关系）
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    # 并发合并相同实体并将其插入到知识图谱中
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    # 并发合并相同关系并将其插入到知识图谱中
    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    # 验证是否成功提取了实体
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    # 验证是否成功提取了关系
    if not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relationships, maybe your LLM is not working"
        )
        return None

    # 如果提供了实体向量数据库，将实体信息插入向量数据库以支持相似性检索
    if entity_vdb is not None:
        # 构建实体向量数据库的插入数据，计算实体ID并准备向量内容
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],  # 实体名称+描述作为向量内容
                "entity_name": dp["entity_name"],  # 保存原始实体名称
            }
            for dp in all_entities_data
        }
        # 执行向量数据插入
        await entity_vdb.upsert(data_for_vdb)
    # 再次插入实体向量数据，但这次使用空格分隔名称和描述
    # 注：这里可能是代码冗余，也可能是为了提高不同检索场景下的匹配效果
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + " " + dp["description"],  # 实体名称+空格+描述
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    # 如果提供了实体名称向量数据库，仅将实体名称插入向量数据库以支持精确名称匹配
    if entity_name_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="Ename-"): {
                "content": dp["entity_name"],  # 仅使用实体名称作为向量内容
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_name_vdb.upsert(data_for_vdb)

    # 如果提供了关系向量数据库，将关系信息插入向量数据库
    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],  # 源实体ID
                "tgt_id": dp["tgt_id"],  # 目标实体ID
                "content": dp["keywords"]  # 构建关系内容，包含关键词、源ID、目标ID和描述
                + " " + dp["src_id"]
                + " " + dp["tgt_id"]
                + " " + dp["description"],
            }
            for dp in all_relationships_data
        }

        await relationships_vdb.upsert(data_for_vdb)

    # 返回更新后的知识图谱实例
    return knowledge_graph_inst


async def local_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """
    执行本地查询流程，结合知识图谱、向量数据库和LLM生成智能回答
    
    该函数是MiniRAG系统的核心查询功能之一，采用本地查询模式（local query），
    通过提取查询关键词，在实体向量数据库中搜索相似实体，结合知识图谱获取
    相关实体、关系和文本块，最终生成基于知识库的智能回答。
    
    本地查询模式侧重于基于相似实体的检索和推理，适合查找具体的实体信息、
    实体间关系和相关文本片段。通过多层次的检索和整合，为LLM提供丰富的
    上下文信息，生成准确、相关的回答。
    
    Args:
        query (str): 用户查询文本
            - 自然语言形式的问题或查询请求
            - 可以是中文或英文
            - 支持各种类型的查询（事实查询、关系查询、描述查询等）
            - 例如："苹果公司的CEO是谁？"或"机器学习的应用领域"
            
        knowledge_graph_inst (BaseGraphStorage): 知识图谱存储实例
            - 提供图结构数据的存储、查询和遍历功能
            - 支持节点查询、边查询、节点度计算等操作
            - 必须实现get_node、get_edge、node_degree等方法
            - 可以是内存图、Neo4j、网络图等多种实现
            
        entities_vdb (BaseVectorStorage): 实体向量数据库
            - 存储实体名称和描述的向量化表示
            - 支持基于余弦相似度的实体检索
            - 返回top-k最相似的实体及其相似度分数
            - 用于找到与查询最相关的实体
            
        relationships_vdb (BaseVectorStorage): 关系向量数据库
            - 存储关系关键词、实体和描述的向量化表示
            - 用于全局查询模式中的关系检索
            - 在本地查询中主要提供接口兼容性
            - 暂未在本地查询中使用到
            
        text_chunks_db (BaseKVStorage[TextChunkSchema]): 文本块键值存储
            - 存储原始文档分块后的文本内容
            - 提供基于chunk_key的快速内容检索
            - 支持异步批量获取操作
            - 为LLM回答提供原始文本证据
            
        query_param (QueryParam): 查询参数配置对象
            - top_k: 检索返回的最大结果数量
            - only_need_context: 是否只返回上下文而不生成回答
            - response_type: 生成回答的类型（如"JSON"、"TEXT"等）
            - max_token_for_text_unit: 文本块的最大token限制
            - 其他影响检索和回答质量的参数
            
        global_config (dict): 全局配置参数字典
            - llm_model_func: 用于关键词提取和回答生成的大语言模型函数
            - tiktoken_model_name: token计算模型名称
            - 其他影响LLM调用的配置参数
    
    Returns:
        str: LLM生成的回答或上下文信息
            - 成功情况：返回基于知识库的智能回答
            - 仅上下文模式：返回结构化的上下文信息（entities、relationships、sources）
            - 失败情况：返回预定义的失败回答（如"抱歉，我无法回答"）
    
    查询处理流程:
        1. 关键词提取阶段：
           - 使用LLM从用户查询中提取低层关键词
           - 生成结构化的JSON格式关键词列表
           - 处理多种可能的JSON解析格式
           
        2. 本地查询构建阶段：
           - 基于提取的关键词在实体向量数据库中检索相似实体
           - 获取实体的详细信息、度分数和相关数据
           - 查找与实体相关的文本块和关系
           
        3. 上下文整合阶段：
           - 整合实体、关系和文本块信息
           - 生成结构化的CSV格式上下文数据
           - 为LLM回答提供丰富的背景信息
           
        4. 回答生成阶段：
           - 构建包含上下文的系统提示词
           - 调用LLM生成基于知识库的回答
           - 清理回答中的提示词和多余内容
    
    关键词提取机制:
        1. 提示词构建：
           - 使用预定义的keywords_extraction提示词模板
           - 将用户查询注入到提示词模板中
           - 引导LLM提取低层关键词（low_level_keywords）
           
        2. JSON解析处理：
           - 优先尝试标准的JSON解析方式
           - 失败时尝试非标准格式的解析（去除prompt标记）
           - 支持多种可能的JSON输出格式
           
        3. 错误处理：
           - JSON解析失败时返回预定义的失败回答
           - 记录解析错误信息用于调试
           - 提供默认的降级处理机制
    
    本地查询上下文构建:
        1. 实体检索：
           - 在entities_vdb中基于关键词进行相似度检索
           - 获取top-k个最相似的实体及其基本信息
           - 查询知识图谱中对应节点的详细信息
           
        2. 数据整合：
           - 并发查询节点的度和相关信息
           - 整合实体信息、度分数和相关性数据
           - 过滤掉无效或缺失的节点数据
           
        3. 相关内容查找：
           - 调用_find_most_related_text_unit_from_entities获取相关文本块
           - 调用_find_most_related_edges_from_entities获取相关关系
           - 确保相关内容的完整性和质量
            
        4. 上下文格式化：
           - 将实体信息格式化为CSV表格形式
           - 将关系信息格式化为CSV表格形式
           - 将文本块信息格式化为CSV表格形式
           - 添加结构化的标记和分隔符
    
    LLM回答生成机制:
        1. 系统提示词构建：
           - 使用预定义的rag_response提示词模板
           - 注入上下文数据和回答类型
           - 指导LLM基于知识库信息生成准确回答
           
        2. 对话调用：
           - 传入用户查询和系统提示词
           - 调用LLM生成上下文相关的回答
           - 支持多种LLM模型的调用接口
           
        3. 回答后处理：
           - 去除系统提示词和重复的用户查询
           - 清理markdown标记和特殊字符
           - 提取纯净的回答内容
    
    性能优化策略:
        - 并发处理：使用asyncio.gather并发查询多个组件
        - 缓存机制：可能利用已缓存的节点和关系信息
        - 批量操作：批量查询减少数据库调用次数
        - 错误容错：单个组件失败不影响整体查询
    
    使用场景:
        - 基于知识库的问答系统
        - 实体关系查询和分析
        - 文档内容检索和引用
        - 知识图谱的探索式查询
        - RAG系统的本地检索模式
    
    注意事项:
        - 关键词提取的质量直接影响检索效果
        - 实体向量数据库的建设质量是关键因素
        - 需要平衡检索的精确度和召回率
        - 大量并发查询时注意资源限制
        - LLM回答的质量依赖于知识库的完整性和准确性
        - 建议在生产环境中添加更详细的错误日志
        - 文本块的内容长度可能影响LLM的token消耗
        - 不同的query_param配置会影响检索策略和结果
    """
    context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)

    try:
        keywords_data = json.loads(json_text)
        keywords = keywords_data.get("low_level_keywords", [])
        keywords = ", ".join(keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"

            keywords_data = json.loads(result)
            keywords = keywords_data.get("low_level_keywords", [])
            keywords = ", ".join(keywords)
        # Handle parsing error
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if keywords:
        context = await _build_local_query_context(
            keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response


async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    """
    构建本地查询的上下文数据，整合实体、关系和文本块信息
    
    该函数负责从知识图谱和向量数据库中检索与查询相关的实体、
    关系和文本块，并将其整合为结构化的CSV格式，为LLM回答
    提供丰富的上下文信息。这是local_query函数的核心辅助函数，
    负责构建查询所需的知识库上下文。
    
    函数采用并发处理策略，同时从多个存储组件检索数据，
    提高查询效率。通过实体检索的权重排序和相关度过滤，
    确保上下文信息的质量和相关性。最终生成标准化的CSV格式
    上下文数据，便于LLM理解和处理。
    
    Args:
        query (str): 用户查询的关键词字符串
            - 通常来自关键词提取的结果，用逗号分隔
            - 用作实体向量数据库的检索基准
            - 例如："苹果公司, CEO, 史蒂夫·乔布斯"
            
        knowledge_graph_inst (BaseGraphStorage): 知识图谱存储实例
            - 提供图结构数据的查询和遍历功能
            - 实现get_node、node_degree等方法
            - 用于获取实体节点的详细信息和度分数
            
        entities_vdb (BaseVectorStorage): 实体向量数据库
            - 基于向量化表示的实体相似度检索
            - 返回与查询相关的top-k个实体
            - 提供实体的基本信息和相似度分数
            
        text_chunks_db (BaseKVStorage[TextChunkSchema]): 文本块键值存储
            - 存储原始文档分块后的文本内容
            - 支持基于chunk_key的批量查询
            - 为上下文提供原始文本证据
            
        query_param (QueryParam): 查询参数配置
            - top_k: 控制检索返回的最大实体数量
            - max_token_for_text_unit: 限制文本块的最大长度
            - max_token_for_global_context: 限制全局上下文的token大小
            - 其他影响检索质量和性能的参数
    
    Returns:
        str: 格式化的CSV上下文数据字符串
            - 包含entities、relationships、sources三个CSV表格
            - 每个表格都有header和具体数据行
            - 以分隔符和标记字符组织结构化数据
    
    处理流程:
        1. 实体检索阶段：
           - 在entities_vdb中使用向量相似度检索top-k个实体
           - 获取每个实体的相似度分数和基本ID信息
           - 过滤掉低相似度的实体，提高相关性
            
        2. 实体信息增强阶段：
           - 从知识图谱中获取每个实体的详细信息
           - 查询节点的度和相关属性
           - 并发处理多个实体的信息查询
            
        3. 相关内容发现阶段：
           - 基于检索到的实体查找相关文本块
           - 基于检索到的实体查找相关关系边
           - 确保找到的内容与查询的强相关性
            
        4. 上下文格式化阶段：
           - 将实体信息格式化为标准CSV格式
           - 将关系信息格式化为标准CSV格式  
           - 将文本块信息格式化为标准CSV格式
           - 添加结构化标记和分隔符
    
    实体检索策略:
        1. 向量相似度搜索：
           - 使用查询关键词的向量化表示
           - 在实体向量数据库中进行相似度计算
           - 返回相似度分数最高的top-k个实体
            
        2. 实体权重计算：
           - 基于相似度分数和节点度进行权重计算
           - 综合考虑实体相关性和重要性
           - 用于后续排序和过滤
            
        3. 相关实体发现：
           - 从检索到的实体扩展到相关实体
           - 查找通过关系边连接的邻居实体
           - 扩大查询的覆盖范围
    
    数据整合逻辑:
        1. 并发查询优化：
           - 使用asyncio.gather并发执行多个查询
           - 减少总体查询时间，提高响应速度
           - 平衡并发度和资源消耗
            
        2. 数据验证和清洗：
           - 检查实体节点的存在性和有效性
           - 过滤掉数据不完整的记录
           - 确保数据的一致性和准确性
            
        3. 去重和去噪：
           - 避免重复的实体和关系数据
           - 过滤掉低质量或无关的内容
           - 保持数据的简洁性和相关性
    
    相关文本块查找:
        1. 实体关联查找：
           - 从实体节点获取相关的source_id列表
           - 查找与实体直接关联的文本块
            
        2. 关系扩展查找：
           - 获取实体的一跳邻居节点
           - 查找与邻居节点关联的文本块
           - 计算文本块与实体的关系强度
            
        3. 相关性排序：
           - 基于实体原始排序和关系计数排序
           - 确保最相关的内容排在前面
           - 考虑检索质量和相关性平衡
    
    相关关系查找:
        1. 实体边检索：
           - 获取与实体节点直接相连的所有边
           - 并发查询多条边的详细信息
            
        2. 边信息获取：
           - 查询边的源节点、目标节点和描述信息
           - 计算边的权重和度数
           - 对边进行排序和过滤
            
        3. 边度计算：
           - 计算每条边的度数和权重分数
           - 用于评估关系的重要性和相关性
           - 支持基于重要性的关系排序
    
    CSV格式规范:
        1. Entities表格：
           - 列：id, entity, type, description, rank
           - id: 实体的唯一数字标识
           - entity: 实体名称
           - type: 实体类型（如人名、地名、组织等）
           - description: 实体的详细描述
           - rank: 基于权重和度的综合排名分数
            
        2. Relationships表格：
           - 列：id, source, target, description, keywords, weight, rank
           - id: 关系的唯一数字标识
           - source: 源实体名称
           - target: 目标实体名称
           - description: 关系的详细描述
           - keywords: 关系的关键词标签
           - weight: 关系的权重或置信度
           - rank: 基于权重和度的综合排名分数
            
        3. Sources表格：
           - 列：id, content
           - id: 文本块的唯一数字标识
           - content: 文本块的实际内容

    
    注意事项:
        - 检索质量依赖于实体向量数据库的建设
        - 相似度阈值设置需要平衡精确度和召回率
        - 大批量查询时注意内存和CPU资源使用
        - CSV格式的兼容性影响后续LLM处理效果
        - 建议根据实际应用场景调整top_k参数
        - 文本块内容可能需要进一步的摘要或截断
        - 实体和关系的命名规范需要保持一致性
        - 实体节点数据的完整性检查是必要的
        - 一跳邻居的扩展可能会显著增加检索范围
        - token限制需要根据具体应用场景调整
    """
    results = await entities_vdb.query(query, top_k=query_param.top_k)

    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    """
    从实体节点中智能查找最相关的文本块，构建高质量的上下文内容
    
    该函数是本地查询构建的核心辅助函数，负责从给定的实体列表中
    查找与其最相关的文本块。它采用多层次的查找策略：先查找与
    实体直接关联的文本块，再扩展到与实体邻居节点相关的文本块，
    最终基于实体排序和相关度进行综合排序，为LLM回答提供精准、
    高质量的文本证据。
    
    函数通过实体节点扩展算法，扩大查找范围，同时通过相关性
    计数和排序策略，确保找到的文本块既相关又有序。这是MiniRAG
    系统中文本内容检索的关键组件，直接影响问答系统的质量。
    
    Args:
        node_datas (list[dict]): 实体数据列表
            - 每个字典包含实体的基本信息
            - 典型字段：entity_name, source_id, entity_type等
            - 来源于向量检索和知识图谱查询的结果
            - 实体已按相关性和权重进行排序
            
        query_param (QueryParam): 查询参数配置对象
            - max_token_for_text_unit: 控制返回文本块的最大token数
            - 控制文本块的数量和质量平衡
            - 影响最终上下文的详细程度
            
        text_chunks_db (BaseKVStorage[TextChunkSchema]): 文本块键值存储
            - 存储所有文本块的内容和元数据
            - 支持基于chunk_key的快速检索
            - 提供批量和异步查询接口
            
        knowledge_graph_inst (BaseGraphStorage): 知识图谱存储实例
            - 提供图结构数据的查询和遍历功能
            - 用于获取实体的邻居节点和相关边
            - 支持多层次的实体关系扩展
    
    Returns:
        list: 排序后的最相关文本块列表
            - 每个元素都是包含content字段的字典
            - 文本块按相关性和重要性排序
            - 数量受max_token_for_text_unit限制
    
    核心处理流程:
        1. 直接关联查找阶段：
           - 从每个实体节点获取source_id列表
           - 这些source_id直接指向相关的文本块
           - 建立实体与文本块的直接映射关系
            
        2. 邻居节点扩展阶段：
           - 获取每个实体的一跳邻居节点
           - 查找邻居节点关联的文本块
           - 通过关系传播扩大查找范围
            
        3. 相关性计算阶段：
           - 计算每个文本块与查询实体的关联强度
           - 基于实体排序和关系计数进行综合评分
           - 考虑直接关联和间接关联的影响
            
        4. 排序和过滤阶段：
           - 按相关性对文本块进行排序
           - 根据token限制进行截断
           - 确保内容质量和数量平衡
    
    实体关联映射:
        1. 直接source_id获取：
           - 从node_datas中提取每个实体的source_id
           - source_id通常以GRAPH_FIELD_SEP分隔
           - 建立实体到文本块ID的直接映射
            
        2. source_id解析：
           - 使用split_string_by_multi_markers解析source_id
           - 处理可能的多个source_id分隔情况
           - 提取所有相关的文本块标识符
            
        3. 映射关系验证：
           - 检查source_id在文本块存储中的存在性
           - 验证文本块内容的完整性和有效性
           - 过滤掉无效或损坏的文本块
    
    邻居节点扩展算法:
        1. 一跳邻居获取：
           - 使用knowledge_graph_inst.get_node_edges
           - 获取与每个实体直接相连的所有边
           - 提取边的目标节点作为邻居实体
            
        2. 邻居数据检索：
           - 并发查询所有邻居节点的信息
           - 批量获取邻居节点的source_id
           - 建立邻居节点与文本块的关联映射
            
        3. 扩展查找优化：
           - 避免重复查找相同的文本块
           - 只考虑存在且有效的邻居节点
           - 平衡查找范围和计算效率
    
    相关性计算策略:
        1. 实体排序优先级：
           - 主排序键：原始实体在node_datas中的位置
           - 保证高相关性实体对应的文本块优先
            
        2. 关系计数权重：
           - 计算文本块与查询实体的关系强度
           - 通过邻居节点的共同source_id计算关系数
           - 关系数越多表示文本块越相关
            
        3. 复合排序算法：
           - 排序键：(order, -relation_counts)
           - order确保高相关性实体优先
           - -relation_counts确保高关系强度优先
    
    数据验证和清洗:
        1. 节点数据验证：
           - 检查node_datas中每个实体的有效性
           - 确保source_id字段存在且格式正确
           - 过滤掉数据不完整的实体
            
        2. 边数据验证：
           - 检查get_node_edges返回的结果
           - 验证边数据的完整性和有效性
           - 处理可能的None或空边列表
            
        3. 文本块数据验证：
           - 检查chunk_data的存在性和content字段
           - 确保文本内容不为空且格式正确
           - 过滤掉内容缺失或损坏的文本块
    
    文本块处理和优化:
        1. 查找范围管理：
           - 避免对同一文本块的重复处理
           - 使用all_text_units_lookup字典去重
           - 提高查找效率和数据一致性
            
        2. 内容截断控制：
           - 使用truncate_list_by_token_size函数
           - 基于文本内容长度进行智能截断
           - 平衡内容完整性和token限制
            
        3. 数据结构标准化：
           - 统一文本块的输出格式
           - 确保返回数据的一致性和可用性
           - 为后续LLM处理做准备
    
    错误处理和容错:
        1. 缺失数据处理：
           - 检查并处理node_datas中的缺失字段
           - 对边数据为None的情况进行容错
           - 确保整体处理流程的稳定性
            
        2. 空结果处理：
           - 当没有找到有效文本块时返回空列表
           - 记录警告信息用于问题诊断
           - 提供有意义的错误反馈
            
        3. 部分失败容错：
           - 单个文本块的获取失败不影响整体处理
           - 通过条件检查和数据过滤避免崩溃
           - 保证系统的鲁棒性和可靠性
    
    性能优化策略:
        - 并发查询：使用asyncio.gather并发处理多个查询
        - 批量操作：批量查询减少数据库调用次数
        - 内存优化：及时释放不需要的中间数据结构
        - 查找优化：避免重复查找和无效操作
    
    使用场景:
        - RAG系统的本地查询模式
        - 知识图谱的文本内容检索
        - 实体关系查询的证据收集
        - 文档内容的相关性分析
        - 智能问答系统的上下文构建
    
    注意事项:
        - 查找范围过大可能影响性能，建议合理控制top_k
        - 邻居节点扩展可能显著增加计算量
        - token限制设置需要平衡内容完整性和成本
        - 文本块的内容质量直接影响最终回答质量
        - source_id的格式一致性对查找效果有重要影响
        - 建议在生产环境中添加详细的性能监控
        - 大量实体查询时注意内存和CPU资源使用
        - 文本块的排序策略需要根据具体应用调整
    """
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            if this_edges:  # Add check for None edges
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        relation_counts += 1

            chunk_data = await text_chunks_db.get_by_id(c_id)
            if chunk_data is not None and "content" in chunk_data:  # Add content check
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                    "relation_counts": relation_counts,
                }

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    """
    从实体节点中智能查找最相关的关系边，构建高质量的关系上下文
    
    该函数是全局查询构建的重要辅助函数，负责从给定的实体列表中
    查找与它们最相关的关系边。它采用实体扩展策略，从多个实体节点
    同时获取相关的边信息，然后对边进行去重、排序和过滤，确保
    找到的关系边既相关又有代表性。最终返回按重要性和相关性
    排序的关系边列表，为全局查询提供丰富的关系上下文。
    
    函数通过并发查询和高效的数据处理算法，在保证查找质量的
    同时优化性能。它是MiniRAG系统关系检索的核心组件，特别
    适用于全局查询模式的复杂关系分析场景。
    
    Args:
        node_datas (list[dict]): 实体数据列表
            - 每个字典包含实体的基本信息（entity_name等）
            - 实体来源于向量检索和知识图谱查询
            - 已按相关性和权重进行排序
            - 主要用于获取实体名称以查询相关边
            
        query_param (QueryParam): 查询参数配置对象
            - max_token_for_global_context: 控制返回关系的最大token数
            - top_k: 限制查询返回的边数量
            - 控制关系质量和数量的平衡
            
        knowledge_graph_inst (BaseGraphStorage): 知识图谱存储实例
            - 提供图结构数据的查询和遍历功能
            - 实现get_node_edges、get_edge、edge_degree等方法
            - 支持边的批量查询和度数计算
    
    Returns:
        list: 排序后的最相关关系边列表
            - 每个元素包含src_tgt、rank、description等字段
            - 按度数和权重综合排序
            - 数量受token限制控制
    
    核心处理流程:
        1. 实体边获取阶段：
           - 并发查询每个实体的直接相连边
           - 使用knowledge_graph_inst.get_node_edges
           - 收集所有实体的关联边信息
            
        2. 边去重和标准化阶段：
           - 将边转换为元组形式并进行排序
           - 使用set数据结构实现自动去重
           - 标准化边表示以避免重复计算
            
        3. 边信息增强阶段：
           - 并发查询边的详细信息（get_edge）
           - 并发计算边的度数（edge_degree）
           - 整合边信息、度数和相关数据
            
        4. 排序和过滤阶段：
           - 按度数和权重进行综合排序
           - 根据token限制进行智能截断
           - 确保返回最相关和最重要的关系
    
    实体边检索策略:
        1. 并发边查询：
           - 使用asyncio.gather批量查询多个实体的边
           - 每个实体的get_node_edges调用独立并发
           - 显著提高大规模实体查询的效率
            
        2. 边数据收集：
           - 收集所有实体的直接相连边
           - 边的格式通常为(源实体, 目标实体)
           - 为后续的去重和合并做准备
            
        3. 查询结果验证：
           - 检查get_node_edges返回的结果有效性
           - 处理可能的空结果或异常情况
           - 确保边数据的完整性和准确性
    
    边去重和标准化算法:
        1. 边元组化转换：
           - 将每条边转换为tuple(sorted(edge))形式
           - sorted函数确保边表示的一致性
           - 无向图：A-B和B-A都表示为(A, B)
           
        2. 集合去重：
           - 使用Python的set数据类型
           - 自动去除重复的边表示
           - 提高数据处理效率和内存使用
            
        3. 列表转换：
           - 将去重后的set转换回list
           - 保持边的列表格式便于后续处理
           - 为批量查询做准备
    
    边信息增强机制:
        1. 边详情查询：
           - 使用get_edge方法获取每条边的完整信息
           - 包括边描述、关键词、权重等属性
           - 为边的重要性和相关性评估提供数据基础
            
        2. 边度数计算：
           - 使用edge_degree方法计算边的度数
           - 度数反映边在图中的重要性和影响力
           - 用于后续的排序和筛选
            
        3. 并发查询优化：
           - 使用asyncio.gather并发执行多个查询
           - 减少网络调用和I/O等待时间
           - 提高整体查询性能
    
    边数据整合算法:
        1. 数据合并策略：
           - 使用zip将边、边详情和度数信息合并
           - 构建包含所有相关信息的统一数据结构
           - 便于后续的排序和过滤操作
            
        2. 数据验证过滤：
           - 过滤掉边详情为None的无效记录
           - 确保返回数据的完整性和有效性
           - 处理可能的数据损坏或缺失情况
            
        3. 结构化输出：
           - 为每条边添加src_tgt和rank字段
           - 保持边数据的结构化和标准化
           - 便于后续的CSV格式化和显示
    
    排序和筛选策略:
        1. 复合排序算法：
           - 主要排序键：rank（边度数）
           - 次要排序键：weight（边权重）
           - 降序排列确保最重要的边优先
            
        2. 智能截断控制：
           - 使用truncate_list_by_token_size函数
           - 基于description字段长度进行截断
           - 平衡关系完整性和token预算
            
        3. 相关性保证：
           - 排序优先考虑度数而非仅权重
           - 度数反映边在知识图谱中的实际重要性
           - 确保返回的关系具有实际意义
    
    数据结构设计:
        1. 输出边数据结构：
           - src_tgt: (源实体, 目标实体)的元组
           - rank: 边在图中的度数
           - description: 边的详细描述
           - keywords: 边的关键词标签
           - weight: 边的权重或置信度
            
        2. 字段完整性保证：
           - 确保所有必要字段都存在且有效
           - 处理可能的字段缺失或异常值
           - 提供标准化的数据输出格式
    
    性能优化特点:
        - 高效的并发查询机制
        - 智能的边去重算法
        - 自适应的token截断控制
        - 批量数据处理优化
    
    使用场景:
        - 全局查询模式的关系检索
        - 知识图谱的关系分析
        - 实体关系的可视化展示
        - 复杂查询路径的构建
        - 关系网络的深度分析
    
    注意事项:
        - 大量实体查询时注意内存和CPU资源消耗
        - 边度数的计算可能涉及图遍历，性能需要考虑
        - 排序策略的选择影响最终结果的相关性
        - token限制的设置需要平衡完整性和成本
        - 边的去重可能丢失一些细微的区别信息
        - 建议在生产环境中监控查询性能
        - 不同知识图谱实现的度计算可能有所差异
        - 边的权重解释依赖于具体应用场景
    """
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data


async def global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """
    执行全局查询流程，结合知识图谱、向量数据库和LLM生成智能回答
    
    全局查询是MiniRAG系统的核心查询功能之一，采用全局查询模式（global query）。
    与本地查询不同，全局查询更侧重于基于关系和概念层次的检索，能够发现
    实体间复杂的关联模式和语义关系。它首先提取查询的高级关键词，然后在
    关系向量数据库中搜索相关关系，再通过关系扩展到实体和文本块。
    
    全局查询模式适合处理需要综合分析、推理和多层次关联的复杂查询场景。
    通过从关系出发的检索策略，能够更好地理解查询的深层意图，发现隐藏的
    知识连接，为LLM提供更加丰富和立体的上下文信息。
    
    Args:
        query (str): 用户查询文本
            - 自然语言形式的问题或查询请求
            - 可以是中文或英文
            - 支持复杂查询、推理查询和关系分析查询
            - 例如："人工智能在医疗领域的应用及发展"或"苹果公司与供应商的关系"
            
        knowledge_graph_inst (BaseGraphStorage): 知识图谱存储实例
            - 提供图结构数据的存储、查询和遍历功能
            - 支持节点查询、边查询、路径查找等操作
            - 用于获取关系相关的实体和文本块
            - 可以是内存图、Neo4j、网络图等多种实现
            
        entities_vdb (BaseVectorStorage): 实体向量数据库
            - 存储实体名称和描述的向量化表示
            - 支持基于余弦相似度的实体检索
            - 在全局查询中主要用于实体相关性分析
            - 为最终结果提供实体层面的验证和补充
            
        relationships_vdb (BaseVectorStorage): 关系向量数据库
            - 存储关系关键词、实体和描述的向量化表示
            - 全局查询的核心检索组件
            - 基于高级关键词检索最相关的关系
            - 提供关系的语义相似度和详细信息
            
        text_chunks_db (BaseKVStorage[TextChunkSchema]): 文本块键值存储
            - 存储原始文档分块后的文本内容
            - 提供基于chunk_key的快速内容检索
            - 为全局查询提供原始文本证据
            - 支持异步批量获取操作
            
        query_param (QueryParam): 查询参数配置对象
            - top_k: 检索返回的最大结果数量
            - only_need_context: 是否只返回上下文而不生成回答
            - response_type: 生成回答的类型（如"JSON"、"TEXT"等）
            - max_token_for_text_unit: 文本块的最大token限制
            - max_token_for_global_context: 全局上下文的token限制
            - 其他影响检索和回答质量的参数
            
        global_config (dict): 全局配置参数字典
            - llm_model_func: 用于关键词提取和回答生成的大语言模型函数
            - tiktoken_model_name: token计算模型名称
            - 其他影响LLM调用和查询处理的配置参数
    
    Returns:
        str: LLM生成的回答或上下文信息
            - 成功情况：返回基于知识库的智能回答
            - 仅上下文模式：返回结构化的全局上下文信息
            - 失败情况：返回预定义的失败回答（如"抱歉，我无法回答"）
    
    全局查询处理流程:
        1. 高级关键词提取阶段：
           - 使用LLM从用户查询中提取高级关键词
           - 生成结构化的JSON格式关键词列表
           - 重点提取概念性、关系性关键词
            
        2. 关系检索阶段：
           - 基于高级关键词在relationships_vdb中检索相似关系
           - 获取关系的详细信息和相似度分数
           - 建立关系到实体和文本块的映射
            
        3. 全局上下文构建阶段：
           - 基于检索到的关系构建完整的全局上下文
           - 整合实体、关系和文本块信息
           - 生成结构化的CSV格式上下文数据
            
        4. 回答生成阶段：
           - 构建包含全局上下文的系统提示词
           - 调用LLM生成基于全局分析的智能回答
           - 清理回答中的提示词和多余内容
    
    全局查询与本地查询的区别:
        1. 关键词类型：
           - 全局查询：high_level_keywords（高级概念关键词）
           - 本地查询：low_level_keywords（低层实体关键词）
            
        2. 检索策略：
           - 全局查询：以关系为中心的检索模式
           - 本地查询：以实体为中心的检索模式
            
        3. 查询范围：
           - 全局查询：关注关系网络和概念层次
           - 本地查询：关注直接实体关联和文本证据
            
        4. 应用场景：
           - 全局查询：概念分析、关系推理、复杂查询
           - 本地查询：事实查询、实体信息查找、具体问题解答
    
    关键词提取机制:
        1. 高级关键词提取：
           - 使用相同的keywords_extraction提示词模板
           - 但专门提取high_level_keywords
           - 重点关注概念、关系和抽象层面的词汇
            
        2. JSON解析处理：
           - 与本地查询相同的解析逻辑
           - 支持多种可能的JSON输出格式
           - 包含完整的错误处理和降级机制
            
        3. 关键词使用：
           - 提取的关键词用于relationships_vdb.query
           - 重点搜索关系层面的相关性
           - 为全局分析提供语义基础
    
    全局上下文构建特色:
        1. 关系中心策略：
           - 优先检索和分析关系信息
           - 基于关系发现相关实体
           - 通过关系扩展文本块查找范围
            
        2. 多层次整合：
           - 关系→实体→文本块的层次化构建
           - 确保上下文信息的完整性和连贯性
           - 支持复杂关系的深度分析
            
        3. 相关性增强：
           - 通过关系度数和权重计算重要性
           - 过滤低质量和无关的关系信息
           - 确保上下文的相关性和可靠性
    
    LLM回答生成特点:
        1. 全局视角回答：
           - 基于关系网络生成更全面的回答
           - 支持概念层面的分析和推理
           - 提供更深层次的洞察和结论
            
        2. 上下文丰富性：
           - 包含更多的关系和概念信息
           - 支持跨领域的知识关联
           - 为LLM提供更完整的推理基础
            
        3. 回答质量：
           - 适合复杂查询和推理型问题
           - 能够发现隐含的关联和模式
           - 提供更加深入和全面的解答
    
    性能特性:
        - 关系检索的高效性
        - 多层次数据的智能整合
        - 全局视角的深度分析能力
        - 灵活的参数配置支持
        - 强大的错误处理和容错机制
    
    使用场景:
        - 概念分析和关系推理
        - 复杂查询的多层次解答
        - 知识图谱的全局探索
        - 跨领域关联分析
        - 深度推理和问题分析
        - RAG系统的全局检索模式
    
    注意事项:
        - 全局查询比本地查询更复杂，性能开销可能更大
        - 关系检索的质量直接影响整体查询效果
        - 高级关键词的提取质量是关键因素
        - 全局上下文的token消耗可能较高
        - 建议根据具体应用场景选择查询模式
        - 复杂查询时注意LLM的上下文长度限制
        - 建议监控查询时间和资源使用情况
        - 全局查询的召回率较高但精确度需要优化
        - 关系网络的构建质量影响查询效果
    """
    context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)

    try:
        keywords_data = json.loads(json_text)
        keywords = keywords_data.get("high_level_keywords", [])
        keywords = ", ".join(keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"

            keywords_data = json.loads(result)
            keywords = keywords_data.get("high_level_keywords", [])
            keywords = ", ".join(keywords)

        except json.JSONDecodeError as e:
            # Handle parsing error
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if keywords:
        context = await _build_global_query_context(
            keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response


async def _build_global_query_context(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    """
    构建全局查询的上下文，包括实体、关系和文本块。
    Args:
        keywords: 查询关键词
        knowledge_graph_inst: 知识图谱实例
        entities_vdb: 实体向量数据库
        relationships_vdb: 关系向量数据库
        text_chunks_db: 文本块KV存储
        query_param: 查询参数
    Returns:
        str: 上下文字符串
    """
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return None

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
    )
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst
    )
    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )
    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    """
    从关系中查找最相关的实体。
    Args:
        edge_datas: 关系数据列表
        query_param: 查询参数
        knowledge_graph_inst: 知识图谱实例
    Returns:
        list: 最相关的实体列表
    """
    entity_names = set()
    for e in edge_datas:
        entity_names.add(e["src_id"])
        entity_names.add(e["tgt_id"])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    """
    从关系中查找相关的文本块。
    Args:
        edge_datas: 关系数据列表
        query_param: 查询参数
        text_chunks_db: 文本块KV存储
        knowledge_graph_inst: 知识图谱实例
    Returns:
        list: 相关的文本块列表
    """
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]

    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                }

    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]

    return all_text_units


async def hybrid_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """
    lightRAG的混合查询流程，结合低级和高级查询结果。
    
    混合查询是lightRAG的核心特性之一，通过分离高级和低级关键词，
    同时执行全局查询和本地查询，然后智能合并结果，提供更全面的回答。
    
    Args:
        query: 查询文本，用户的自然语言问题
        knowledge_graph_inst: 知识图谱实例，提供图结构存储和查询能力
        entities_vdb: 实体向量数据库，存储实体嵌入向量用于语义检索
        relationships_vdb: 关系向量数据库，存储关系嵌入向量用于语义检索
        text_chunks_db: 文本块KV存储，提供原始文本块的键值存储访问
        query_param: 查询参数配置，包括top_k、响应类型等设置
        global_config: 全局配置，包含LLM模型函数等系统配置
        
    Returns:
        str: LLM生成的智能回答，格式化的文本响应
        
    查询流程:
        1. 关键词提取：使用LLM将查询分离为高级和低级关键词
        2. 并行执行：同时进行本地查询和全局查询
        3. 上下文合并：智能合并两个查询的结果，避免重复和冲突
        4. LLM回答：基于合并的上下文生成最终回答
        
    Note:
        - 高级关键词通常涉及抽象概念、主题分类、理论框架
        - 低级关键词通常涉及具体实体、细节信息、具体事件
        - 混合策略能够平衡深度和广度，提供更全面的知识覆盖
    """
    low_level_context = None
    high_level_context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)

    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)
    try:
        keywords_data = json.loads(json_text)
        hl_keywords = keywords_data.get("high_level_keywords", [])
        ll_keywords = keywords_data.get("low_level_keywords", [])
        hl_keywords = ", ".join(hl_keywords)
        ll_keywords = ", ".join(ll_keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            keywords_data = json.loads(result)
            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
            hl_keywords = ", ".join(hl_keywords)
            ll_keywords = ", ".join(ll_keywords)
        # Handle parsing error
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if ll_keywords:
        low_level_context = await _build_local_query_context(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )

    if hl_keywords:
        high_level_context = await _build_global_query_context(
            hl_keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )

    tiktoken_model_name = global_config.get("tiktoken_model_name", "gpt-4o")

    context = combine_contexts(
        high_level_context,
        low_level_context,
        query_param,
        tiktoken_model_name,
    )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return response


def truncate_str_by_tokens(text: str, max_tokens: int, model_name: str) -> str:
    """
    按token数截断字符串，返回截断后的字符串
    """
    if max_tokens <= 0:
        return ""
    toks = encode_string_by_tiktoken(text, model_name=model_name)
    if len(toks) <= max_tokens:
        return text
    return decode_tokens_by_tiktoken(toks[:max_tokens], model_name=model_name)


def combine_contexts(
    high_level_context,
    low_level_context,
    query_param: QueryParam,
    tiktoken_model_name: str,
):
    """
    合并高级别和低级别查询的上下文。
    
    该函数是混合查询的关键组件，负责智能合并全局查询和本地查询的结果。
    通过分别提取、合并和格式化实体、关系、来源三个维度，避免信息冗余，
    并通过token限制确保上下文在模型输入限制内。
    
    Args:
        high_level_context: 高级别查询的上下文，来源于全局查询结果
        low_level_context: 低级别查询的上下文，来源于本地查询结果
        query_param: 查询参数，包含各类上下文的token上限
        tiktoken_model_name: 用于token计算的模型名称
        
    Returns:
        str: 合并后的标准化上下文字符串，包含实体、关系、来源三个部分
    """

    def extract_sections(context):
        """
        从上下文字符串中提取实体、关系和来源部分。
        """
        entities_match = re.search(
            r"-----Entities-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        relationships_match = re.search(
            r"-----Relationships-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        sources_match = re.search(
            r"-----Sources-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )

        entities = entities_match.group(1) if entities_match else ""
        relationships = relationships_match.group(1) if relationships_match else ""
        sources = sources_match.group(1) if sources_match else ""

        return entities, relationships, sources

    # 从两个上下文中提取结构化数据部分
    if high_level_context is None:
        warnings.warn(
            "High Level context is None. Return empty High entity/relationship/source"
        )
        hl_entities, hl_relationships, hl_sources = "", "", ""
    else:
        hl_entities, hl_relationships, hl_sources = extract_sections(high_level_context)

    if low_level_context is None:
        warnings.warn(
            "Low Level context is None. Return empty Low entity/relationship/source"
        )
        ll_entities, ll_relationships, ll_sources = "", "", ""
    else:
        ll_entities, ll_relationships, ll_sources = extract_sections(low_level_context)

    combined_entities = truncate_str_by_tokens(
        process_combine_contexts(hl_entities, ll_entities),
        2000,
        tiktoken_model_name,
    )
    
    # 合并和去重关系数据  
    # 关系数据合并时保持三元组结构完整性 (source, target, relationship)
    # 去重逻辑基于关系类型和参与实体，避免信息冗余
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )
    combined_relationships = truncate_str_by_tokens(
        combined_relationships,
        2000,
        tiktoken_model_name,
    )
    
    # 合并和去重来源数据
    # 来源合并确保所有引用的文本块都被保留，避免信息丢失
    # 去重基于文本块ID，保证每个来源只出现一次
    combined_sources = process_combine_contexts(hl_sources, ll_sources)
    combined_sources = truncate_str_by_tokens(
        combined_sources,
        2000,
        tiktoken_model_name,
    )
    
    # 格式化合并后的最终上下文
    # 采用标准的三部分结构，便于LLM理解和处理
    return f"""
-----Entities-----
```csv
{combined_entities}
```
-----Relationships-----
```csv
{combined_relationships}
```
-----Sources-----
```csv
{combined_sources}
```
"""


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    """
    简单查询流程，直接从向量数据库检索文本块。
    
    朴素RAG（Retrieval-Augmented Generation）查询是最基本的增强生成流程。
    该函数绕过知识图谱，直接使用向量相似度检索最相关的文本块，
    然后将检索结果作为上下文输入LLM进行回答生成。
    
    Args:
        query: 查询文本，用户要提问的自然语言问题
        chunks_vdb: 文本块向量数据库，存储所有文本块的嵌入向量
        text_chunks_db: 文本块KV存储，提供原始文本内容的键值存储
        query_param: 查询参数配置，控制检索数量和输出格式
        global_config: 全局配置，包含LLM模型函数等系统参数
        
    Returns:
        str: 基于检索文本块生成的LLM回答
        
    查询流程:
        1. 向量检索：将查询文本转换为向量，在文本块向量数据库中检索top_k个最相似块
        2. 内容提取：根据检索结果ID从KV存储中获取对应的原始文本内容
        3. 智能截断：按token数量限制对检索结果进行排序和截断，避免输入过长
        4. 上下文构建：将截断后的文本块按格式组织为上下文
        5. LLM生成：基于上下文和查询调用LLM生成最终回答
        
    适用场景:
        - 简单的文档问答系统
        - 不需要复杂知识推理的场景
        - 快速原型的RAG应用
        - 文本相似度基础的信息检索
        
    Note:
        - 该方法不利用知识图谱的结构化信息
        - 依赖向量嵌入的质量和检索算法的效果
        - 适合处理基于内容的直接问题，不适合复杂的逻辑推理
    """
    use_model_func = global_config["llm_model_func"]
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]

    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    if query_param.only_need_context:
        return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    return response


async def path2chunk(
    scored_edged_reasoning_path, knowledge_graph_inst, pairs_append, query, max_chunks=5
):
    """
    根据路径得分和投票结果，将文本块聚合为最终答案。
    
    这是MiniRAG系统中核心的路径推理组件，负责将知识图谱中的路径推理结果
    转换为具体的文本块集合，为后续的LLM回答提供内容支撑。
    该函数通过多层次的聚合和评分机制，确保选择最相关和信息量最大的文本块。
    
    Args:
        scored_edged_reasoning_path: 带分数的路径推理结果字典
            - 格式：{实体名: {"Score": 分数, "Path": [(实体A, 实体B), ...]}}
            - Score: 基于路径长度和查询相关性的综合评分
            - Path: 从该实体到其他相关实体的路径列表
        knowledge_graph_inst: 知识图谱实例，提供节点和边的数据访问接口
        pairs_append: 边路径投票结果，存储经过投票筛选的高质量边集合
        query: 原始查询文本，用于计算节点描述的查询相关度
        max_chunks: 每个路径最多选择的文本块数量上限
        
    Returns:
        dict: 更新后的带分数的路径字典，格式为 {实体名: {"Score": 新分数, "Path": [文本块ID列表]}}
        
    聚合流程:
        1. 路径解析：遍历每个实体的路径，提取对应的边和节点信息
        2. 文本块提取：从边和节点中提取source_id，转换为文本块ID列表
        3. 相关度计算：使用查询文本计算节点描述的相似度，进行智能筛选
        4. 投票整合：结合边投票结果，增强高质量路径的权重
        5. 频次统计：统计所有文本块ID的出现频次，乘以路径得分计算最终权重
        6. 选择策略：按最终权重排序，选择top-K个文本块作为结果
        
    算法特点:
        - 多源融合：同时考虑边路径和节点内容，提高召回率
        - 权重传递：通过频次统计和得分计算，实现权重合理传递
        - 相关度筛选：基于查询内容进行节点描述的相似度计算
        - 投票优化：利用边投票机制过滤低质量连接，提升精度
    """
    # 初始化已处理节点字典，用于缓存已处理过的节点文本块信息，避免重复计算
    already_node = {}
    
    # 遍历带分数的路径字典中的每个实体及其路径信息
    for k, v in scored_edged_reasoning_path.items():
        # 初始化当前实体的文本块计数字典为None
        node_chunk_id = None

        # 遍历当前实体的每个路径元组及其得分列表
        for pathtuple, scorelist in v["Path"].items():
            # 检查当前路径元组是否在投票结果中
            if pathtuple in pairs_append:
                # 获取投票后的边列表
                use_edge = pairs_append[pathtuple]
                edge_datas = []
                # 并发获取每条边的数据
                edge_datas = await asyncio.gather(
                    *[knowledge_graph_inst.get_edge(r[0], r[1]) for r in use_edge]
                )
                # 从边数据中提取文本块ID，使用分隔符分割source_id
                text_units = [
                    split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
                    for dp in edge_datas  # chunk ID
                ][0]  # 取第一个边的数据作为文本单元
            else:
                # 如果路径未经过投票，初始化空的边和文本单元列表
                use_edge = []
                text_units = []

            # 并发获取路径第一个节点的数据
            node_datas = await asyncio.gather(
                *[knowledge_graph_inst.get_node(pathtuple[0])]
            )
            # 处理第一个节点的文本块ID
            for dp in node_datas:
                # 从节点的source_id中提取文本块ID
                text_units_node = split_string_by_multi_markers(
                    dp["source_id"], [GRAPH_FIELD_SEP]
                )
                # 将节点文本块ID添加到总体文本单元列表
                text_units = text_units + text_units_node

            # 并发获取路径中剩余所有节点的数据
            node_datas = await asyncio.gather(
                *[knowledge_graph_inst.get_node(ents) for ents in pathtuple[1:]]
            )
            # 当查询不为空时，基于查询相关性进行文本块筛选
            if query is not None:
                for dp in node_datas:
                    # 提取节点的文本块ID
                    text_units_node = split_string_by_multi_markers(
                        dp["source_id"], [GRAPH_FIELD_SEP]
                    )
                    # 提取节点的描述信息
                    descriptionlist_node = split_string_by_multi_markers(
                        dp["description"], [GRAPH_FIELD_SEP]
                    )
                    
                    # 检查节点描述是否已处理过，避免重复计算
                    if descriptionlist_node[0] not in already_node.keys():
                        # 标记当前节点已处理
                        already_node[descriptionlist_node[0]] = None

                        # 当文本块ID数量与描述数量匹配时进行相似度筛选
                        if len(text_units_node) == len(descriptionlist_node):
                            # 当文本块数量超过5个时进行智能筛选
                            if len(text_units_node) > 5:
                                # 计算需要考虑的最大ID数量，至少5个或总数量的一半
                                max_ids = int(max(5, len(text_units_node) / 2))
                                # 计算描述与查询的相似度，选择最相关的前max_ids个描述
                                should_consider_idx = calculate_similarity(
                                    descriptionlist_node, query, k=max_ids
                                )
                                # 根据相似度结果筛选文本块ID
                                text_units_node = [
                                    text_units_node[i] for i in should_consider_idx
                                ]
                                # 缓存筛选后的文本块ID
                                already_node[descriptionlist_node[0]] = text_units_node
                    else:
                        # 如果节点已处理，直接使用缓存的文本块ID
                        text_units_node = already_node[descriptionlist_node[0]]
                    
                    # 将节点文本块ID添加到总体文本单元列表
                    if text_units_node is not None:
                        text_units = text_units + text_units_node

            # 统计当前路径中各文本块ID的出现频次
            count_dict = Counter(text_units)
            # 计算当前路径的总得分（路径得分+节点得分+1，+1避免得分为0）
            total_score = scorelist[0] + scorelist[1] + 1
            # 将每个文本块ID的频次乘以路径总得分，实现权重传递
            for key, value in count_dict.items():
                count_dict[key] = value * total_score
            
            # 合并当前路径的文本块计数到节点总体计数
            if node_chunk_id is None:
                # 如果是第一个路径，直接赋值
                node_chunk_id = count_dict
            else:
                # 否则累加计数
                node_chunk_id = node_chunk_id + count_dict
        
        # 清空原路径数据，准备存储文本块ID列表
        v["Path"] = []
        
        # 如果没有成功获取任何文本块计数（处理失败情况）
        if node_chunk_id is None:
            # 直接获取当前实体的节点数据
            node_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(k)])
            for dp in node_datas:
                # 提取节点的文本块ID
                text_units_node = split_string_by_multi_markers(
                    dp["source_id"], [GRAPH_FIELD_SEP]
                )
                # 统计文本块ID频次
                count_dict = Counter(text_units_node)

            # 选择频次最高的前max_chunks个文本块ID
            for id in count_dict.most_common(max_chunks):
                v["Path"].append(id[0])
            # 注释掉的备选实现，直接赋值结果列表
            # v['Path'] = count_dict.most_common(max_chunks)#[]
        else:
            # 选择权重最高的前max_chunks个文本块ID
            for id in count_dict.most_common(max_chunks):
                v["Path"].append(id[0])
            # 注释掉的备选实现，直接赋值结果列表
            # v['Path'] = node_chunk_id.most_common(max_chunks)
    
    # 返回更新后的路径字典，其中Path字段已替换为文本块ID列表
    return scored_edged_reasoning_path


def scorednode2chunk(input_dict, values_dict):
    """
    将实体和关系数据映射到文本块ID。
    
    该函数是MiniRAG系统的核心数据映射组件，负责将查询得到的实体或关系数据
    映射到对应的文本块ID，实现从抽象实体到具体文档内容的转换。
    
    Args:
        input_dict (dict): 实体或关系字典，键为实体/关系名称，值为对应的文本块ID列表
        values_dict (dict): 文本块ID到文本块数据的完整映射字典
    
    功能流程:
        1. 遍历输入字典中的每个实体/关系项
        2. 根据文本块ID在values_dict中查找对应的数据
        3. 进行数据有效性验证和过滤
        4. 更新输入字典，将ID列表替换为实际数据列表
    
    数据处理策略:
        - 使用get方法安全访问，避免KeyError异常
        - 双重过滤：先检查ID存在性，再过滤None值
        - 原地修改输入字典，提高内存效率
    
    返回值:
        无返回值，直接修改input_dict的内容
    
    使用场景:
        - 在MiniRAG查询流程中，将实体查询结果转换为文本块数据
        - 关系数据的后处理和数据清洗
        - 上下文构建前的数据标准化
    
    性能特点:
        - 时间复杂度：O(n*m)，其中n是input_dict中的键数，m是每个值列表的平均长度
        - 空间复杂度：O(k)，k是最终保留的有效数据项数量
        - 就地修改，避免额外内存开销
    
    注意事项:
        - 输入字典会被原地修改，原有ID列表会被替换为数据对象列表
        - 确保values_dict包含所有需要的文本块ID，否则对应项会被过滤掉
        - 适用于已经过预处理的实体/关系数据，输入数据应保证基本格式正确性
    """
    for key, value_list in input_dict.items():
        # 第一步：使用get方法安全获取数据，将不存在的ID映射为None
        input_dict[key] = [
            values_dict.get(val, None) for val in value_list if val in values_dict
        ]
        # 第二步：过滤掉None值，确保最终结果只包含有效数据
        input_dict[key] = [val for val in input_dict[key] if val is not None]


def kwd2chunk(ent_from_query_dict, chunks_ids, chunk_nums):
    """
    根据实体和关系查询结果，智能聚合文本块ID。
    
    该函数是MiniRAG系统的文本块聚合核心，基于实体和关系查询结果，
    通过多层次评分机制和权重计算，实现智能的文本块选择和排序。
    
    Args:
        ent_from_query_dict (dict): 实体查询结果字典，包含每个实体的查询结果列表
        chunks_ids (list): 通过向量相似度检索得到的候选文本块ID列表
        chunk_nums (int): 最终返回的聚合文本块数量限制
    
    功能流程:
        1. 遍历每个实体的查询结果列表
        2. 为每个实体计算文本块的综合得分
        3. 应用多层权重策略（首项奖励、路径匹配奖励）
        4. 跨实体聚合所有文本块得分
        5. 按得分排序并返回top-k文本块
    
    评分机制:
        - 首项奖励：每个实体的首个结果获得2倍权重
        - 路径匹配：路径中的第一个ID且在候选列表中获得10倍权重
        - 累积计分：同一文本块的多次出现会累积得分
        - 最终排序：使用Counter的most_common方法按得分降序排序
    
    权重策略详解:
        - d == list_of_dicts[0]: 给予首个结果更高权重，鼓励直接匹配
        - id == path[0] and id in chunks_ids: 优先选择向量检索确认的相关文本块
        - 累积得分机制：确保在多个查询中出现的重要文本块获得更高排名
    
    返回值:
        list: 聚合后的文本块ID列表，按综合得分降序排列
    
    使用场景:
        - MiniRAG查询流程中的文本块筛选和排序
        - 多实体查询结果的融合和去重
        - 基于语义相似度和结构相关性的混合排序
    
    性能特点:
        - 时间复杂度：O(n*m*p)，其中n是实体数，m是每个实体的结果数，p是路径平均长度
        - 空间复杂度：O(u)，u是所有唯一文本块ID的数量
        - 使用Counter进行高效的计数和排序操作
    
    注意事项:
        - 输入字典的每个值应该是一个包含Score和Path字段的字典列表
        - chunk_nums参数控制最终返回结果的数量，避免输出过多无关文本块
        - 评分机制假设查询结果已经过基本的相似度排序
        - 函数会改变Counter对象的状态，如需保留原始数据请提前备份
    """
    final_chunk = Counter()  # 最终的文本块计数器，累积所有实体的得分
    final_chunk_id = []      # 最终返回的文本块ID列表
    
    # 遍历每个实体的查询结果
    for key, list_of_dicts in ent_from_query_dict.items():
        total_id_scores = Counter()    # 当前实体的文本块得分统计
        id_scores_list = []            # 得分字典列表
        id_scores = {}                 # 单个实体的文本块得分映射
        
        # 处理当前实体的每个查询结果
        for d in list_of_dicts:
            # 应用首项奖励机制：首个结果获得2倍得分
            if d == list_of_dicts[0]:
                score = d["Score"] * 2
            else:
                score = d["Score"]
            path = d["Path"]  # 获取关联的文本块路径

            # 处理路径中的每个文本块ID
            for id in path:
                # 路径匹配奖励：路径首ID且在候选列表中时给予10倍权重
                if id == path[0] and id in chunks_ids:
                    score = score * 10
                # 累积计分机制：同一文本块多次出现时累加得分
                if id in id_scores:
                    id_scores[id] += score
                else:
                    id_scores[id] = score
        id_scores_list.append(id_scores)  # 保存当前实体的得分结果

        # 跨所有得分字典进行聚合
        for scores in id_scores_list:
            total_id_scores.update(scores)
        # 将当前实体的得分合并到全局计数器
        final_chunk = final_chunk + total_id_scores  # .most_common(3)

    # 按得分排序并选择top-k结果
    for i in final_chunk.most_common(chunk_nums):
        final_chunk_id.append(i[0])
    return final_chunk_id


async def _build_mini_query_context(
    ent_from_query,  # 从查询中识别的实体列表
    type_keywords,  # 查询中识别的类型关键词
    originalquery,  # 原始查询文本
    knowledge_graph_inst: BaseGraphStorage,  # 知识图谱实例
    entities_vdb: BaseVectorStorage,  # 实体向量数据库
    entity_name_vdb: BaseVectorStorage,  # 实体名称向量数据库
    relationships_vdb: BaseVectorStorage,  # 关系向量数据库
    chunks_vdb: BaseVectorStorage,  # 文本块向量数据库
    text_chunks_db: BaseKVStorage[TextChunkSchema],  # 文本块KV存储
    embedder,  # 嵌入模型
    query_param: QueryParam,  # 查询参数配置
):
    """
    构建MiniRAG查询的上下文，包括实体、关系和文本块检索。
    Args:
        ent_from_query: 实体查询结果
        type_keywords: 类型关键词
        originalquery: 原始查询文本
        knowledge_graph_inst: 知识图谱实例
        entities_vdb: 实体向量数据库
        entity_name_vdb: 实体名称向量数据库
        relationships_vdb: 关系向量数据库
        chunks_vdb: 文本块向量数据库
        text_chunks_db: 文本块KV存储
        embedder: 嵌入模型
        query_param: 查询参数
    Returns:
        str: 上下文字符串
    """
    # 1初始化重要实体列表，用于存储后续处理中的关键实体
    imp_ents = []
    # 初始化查询节点列表，存储从实体名称向量数据库查询的结果
    nodes_from_query_list = []
    # 初始化实体查询结果字典，键为原始实体，值为对应的匹配实体列表
    ent_from_query_dict = {}

    # 遍历查询中识别的每个实体
    for ent in ent_from_query:
        # 初始化实体对应的匹配结果列表
        ent_from_query_dict[ent] = []
        # 在实体名称向量数据库中查询匹配的实体
        results_node = await entity_name_vdb.query(ent, top_k=query_param.top_k)
        #nanovector:         results = [
        #     {
        #         **dp,
        #         "id": dp["__id__"],
        #         "distance": dp["__metrics__"],  # 字段表示查询文本与数据库中存储的实体名称之间的相似度距离
        #         "created_at": dp.get("__created_at__"),
        #     }
        #     for dp in results
        # ]
        # 
        # 将查询结果添加到节点列表中
        nodes_from_query_list.append(results_node)
        # 提取实体名称并保存到字典中
        ent_from_query_dict[ent] = [e["entity_name"] for e in results_node]

    # 初始化候选推理路径字典
    candidate_reasoning_path = {}

    # 遍历每个实体的查询结果列表
    for results_node_list in nodes_from_query_list:
        # 创建新的候选推理路径字典，键为实体名称，值为包含分数和路径的字典
        candidate_reasoning_path_new = {
            key["entity_name"]: {"Score": key["distance"], "Path": []}
            for key in results_node_list
        }
        #值为包含两个字段的字典：
        # - Score : 存储实体与查询的相似度距离（用于后续排序）
        # - Path : 初始为空列表，准备用于存储从该实体出发的推理路径
        # 合并新路径到候选推理路径字典
        
        # "实体名称1": {
        #     "Score": 距离分数值,  # 来自results_node_list中对应元素的distance字段
        #     "Path": []           # 初始为空列表，后续会填充路径信息
        # },
        # "实体名称2": {
        #     "Score": 距离分数值,
        #     "Path": []
        # },
        # # 更多实体...
        candidate_reasoning_path = {
            **candidate_reasoning_path,
            **candidate_reasoning_path_new,
        }
    
    # 2为每个候选实体查找k跳邻居路径
    for key in candidate_reasoning_path.keys():
        # 获取实体的2跳邻居路径信息（candidate_reasoning_path返回从该节点出发的所有 2 跳路径（元组形式，如 (A, B, C)））
        candidate_reasoning_path[key][
            "Path"
        ] = await knowledge_graph_inst.get_neighbors_within_k_hops(key, 2)
        # 将实体添加到重要实体列表
        imp_ents.append(key)
    #3. 路径筛选与优化
    # 过滤出路径长度小于1的短路径条目（没有邻居节点的实体）
    short_path_entries = {
        name: entry
        for name, entry in candidate_reasoning_path.items()
        if len(entry["Path"]) < 1
    }
    # 按分数降序排序短路径条目
    sorted_short_path_entries = sorted(
        short_path_entries.items(), key=lambda x: x[1]["Score"], reverse=True
    )
    # 计算要保留的短路径数量，至少保留1个，最多保留20%
    save_p = max(1, int(len(sorted_short_path_entries) * 0.2))
    # 获取分数最高的短路径条目
    top_short_path_entries = sorted_short_path_entries[:save_p]
    # 转换为字典格式
    top_short_path_dict = {name: entry for name, entry in top_short_path_entries}
    
    # 过滤出路径长度大于等于1的长路径条目
    long_path_entries = {
        name: entry
        for name, entry in candidate_reasoning_path.items()
        if len(entry["Path"]) >= 1
    }
    # 合并长路径和高分短路径，形成新的候选推理路径
    candidate_reasoning_path = {**long_path_entries, **top_short_path_dict}

    # 4. 获取可能的答案节点并进行路径评分
    # 根据类型关键词获取相关节点
    node_datas_from_type = await knowledge_graph_inst.get_node_from_types(
        type_keywords
    )  # entity_type, description,...

    # 提取类型相关实体的名称
    maybe_answer_list = [n["entity_name"] for n in node_datas_from_type]
    # 将类型相关实体添加到重要实体列表
    imp_ents = imp_ents + maybe_answer_list
    # 计算推理路径的分数
    scored_reasoning_path = cal_path_score_list(
        candidate_reasoning_path, maybe_answer_list
    )

    # 5. 获取相关关系并进行边投票优化（关键步骤2）
    # 在关系向量数据库中查询与原始查询相关的关系
    results_edge = await relationships_vdb.query(
        originalquery, top_k=len(ent_from_query) * query_param.top_k
    )
    # 初始化好边和坏边列表
    goodedge = []
    badedge = []
    # 过滤关系，区分与重要实体相关的边和不相关的边
    for item in results_edge:
        if item["src_id"] in imp_ents or item["tgt_id"] in imp_ents:
            goodedge.append(item)  # 至少有一个端点是重要实体的边为好边
        else:
            badedge.append(item)  # 两个端点都不是重要实体的边为坏边
    
    # 使用边投票机制对推理路径进行优化，获取更新后的路径和新增的实体对
    scored_edged_reasoning_path, pairs_append = edge_vote_path(
        scored_reasoning_path, goodedge
    )
    # 6. 路径转文本块
    # 将推理路径转换为相关文本块，添加更多上下文信息
    scored_edged_reasoning_path = await path2chunk(
        scored_edged_reasoning_path,  # 优化后的推理路径
        knowledge_graph_inst,  # 知识图谱实例
        pairs_append,  # 新增的实体对
        originalquery,  # 原始查询
        max_chunks=3,  # 每个路径最多关联3个文本块
    )

    #7. 构建最终上下文
    # 初始化实体部分列表，用于构建最终的实体上下文
    entites_section_list = []
    # 并发获取所有实体的详细信息
    node_datas = await asyncio.gather(
        *[
            knowledge_graph_inst.get_node(entity_name)
            for entity_name in scored_edged_reasoning_path.keys()
        ]
    )
    # 合并实体信息和分数，构建完整的实体数据
    node_datas = [
        {**n, "entity_name": k, "Score": scored_edged_reasoning_path[k]["Score"]}
        for k, n in zip(scored_edged_reasoning_path.keys(), node_datas)
    ]
    
    # 构建实体部分列表，每个实体包含名称、分数和描述
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                n["entity_name"],  # 实体名称
                n["Score"],  # 实体分数
                n.get("description", "UNKNOWN"),  # 实体描述，默认为UNKNOWN
            ]
        )
    
    # 按分数降序排序实体列表
    entites_section_list = sorted(
        entites_section_list, key=lambda x: x[1], reverse=True
    )
    
    # 根据token数量限制截断实体列表，避免上下文过长
    entites_section_list = truncate_list_by_token_size(
        entites_section_list,
        key=lambda x: x[2],  # 使用描述文本计算token数
        max_token_size=query_param.max_token_for_node_context,
    )

    # 添加CSV表头
    entites_section_list.insert(0, ["entity", "score", "description"])
    # 转换实体列表为CSV格式字符串
    entities_context = list_of_list_to_csv(entites_section_list)

    # 将实体与文本块关联
    scorednode2chunk(ent_from_query_dict, scored_edged_reasoning_path)

    # 在文本块向量数据库中查询与原始查询相关的文本块
    results = await chunks_vdb.query(originalquery, top_k=int(query_param.top_k / 2))
    # 提取文本块ID列表
    chunks_ids = [r["id"] for r in results]
    # 根据关键词选择最相关的文本块
    final_chunk_id = kwd2chunk(
        ent_from_query_dict,  # 实体查询结果字典
        chunks_ids,  # 候选文本块ID列表
        chunk_nums=int(query_param.top_k / 2)  # 最终返回的文本块数量
    )

    # 检查是否有查询结果，如果没有节点或边结果，返回None
    if not len(results_node):
        return None

    if not len(results_edge):
        return None

    # 并发获取所有选中文本块的详细信息
    use_text_units = await asyncio.gather(
        *[text_chunks_db.get_by_id(id) for id in final_chunk_id]
    )
    # 初始化文本块部分列表，添加表头
    text_units_section_list = [["id", "content"]]

    # 构建文本块部分列表
    for i, t in enumerate(use_text_units):
        if t is not None:  # 只添加有效的文本块
            text_units_section_list.append([i, t["content"]])  # 添加ID和内容
    
    # 转换文本块列表为CSV格式字符串
    text_units_context = list_of_list_to_csv(text_units_section_list)

    # 构建并返回最终的上下文字符串，包含实体和文本块两部分
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def minirag_query(  # MiniRAG
    query,  # 用户提出的自然语言问题文本
    knowledge_graph_inst: BaseGraphStorage,  # 知识图谱实例，提供图结构存储和查询能力
    entities_vdb: BaseVectorStorage,  # 实体向量数据库，存储实体内容的嵌入向量
    entity_name_vdb: BaseVectorStorage,  # 实体名称向量数据库，专门存储实体名称的嵌入向量
    relationships_vdb: BaseVectorStorage,  # 关系向量数据库，存储关系描述的嵌入向量
    chunks_vdb: BaseVectorStorage,  # 文本块向量数据库，存储文本块的嵌入向量
    text_chunks_db: BaseKVStorage[TextChunkSchema],  # 文本块KV存储，提供原始文本内容的快速访问
    embedder,  # 嵌入模型，用于文本向量化和相似度计算
    query_param: QueryParam,  # 查询参数配置，控制检索深度、响应格式等
    global_config: dict,  # 全局配置，包含LLM模型函数等系统设置
) -> str:
    """
    MiniRAG主查询流程，调用MiniRAG专用提示词和模型。
    
    MiniRAG是本系统的核心查询模式，集成了知识图谱、向量检索和路径推理
    的完整能力。该函数使用专门的MiniRAG提示词模板，支持类型关键词提取、
    多层次实体查询、图谱路径推理和智能文本块聚合，最终生成高质量的回答。
    
    Args:
        query: 查询文本，用户提出的自然语言问题
        knowledge_graph_inst: 知识图谱实例，提供图结构存储和查询能力
        entities_vdb: 实体向量数据库，存储实体内容的嵌入向量
        entity_name_vdb: 实体名称向量数据库，专门存储实体名称的嵌入向量
        relationships_vdb: 关系向量数据库，存储关系描述的嵌入向量
        chunks_vdb: 文本块向量数据库，存储文本块的嵌入向量
        text_chunks_db: 文本块KV存储，提供原始文本内容的快速访问
        embedder: 嵌入模型，用于文本向量化和相似度计算
        query_param: 查询参数配置，控制检索深度、响应格式等
        global_config: 全局配置，包含LLM模型函数等系统设置
        
    Returns:
        str: 基于MiniRAG路径推理生成的LLM智能回答
        
    查询流程:
        1. 关键词提取：使用MiniRAG专用提示词提取类型关键词和实体关键词
        2. 类型匹配：在知识图谱中检索匹配类型的实体作为候选集合
        3. 多路检索：并行执行实体名称检索和关系检索，构建候选推理路径
        4. 路径推理：基于图谱结构进行k跳邻居搜索，识别关联实体
        5. 边投票机制：对检索到的边进行投票筛选，提升路径质量
        6. 文本块聚合：使用路径推理结果聚合最相关的文本块
        7. 上下文构建：组织实体、关系、文本块为标准格式上下文
        8. LLM生成：调用专用LLM模型生成基于推理的回答
        
    """
    # 从全局配置中获取LLM模型调用函数
    use_model_func = global_config["llm_model_func"]
    # 获取MiniRAG专用的关键词提取提示词模板
    kw_prompt_temp = PROMPTS["minirag_query2kwd"]
    # 从知识图谱实例中获取所有实体类型（用于类型关键词提取）
    TYPE_POOL, TYPE_POOL_w_CASE = await knowledge_graph_inst.get_types()
    # 将查询文本和类型池格式化到提示词模板中
    kw_prompt = kw_prompt_temp.format(query=query, TYPE_POOL=TYPE_POOL)
    # 调用LLM模型执行关键词提取
    result = await use_model_func(kw_prompt)

    # 尝试解析LLM返回的JSON格式结果
    try:
        # 使用json_repair库处理可能有格式问题的JSON
        keywords_data = json_repair.loads(result)
        
        # 提取回答类型关键词列表
        type_keywords = keywords_data.get("answer_type_keywords", [])
        # 提取从查询中识别的实体列表，最多取前5个
        entities_from_query = keywords_data.get("entities_from_query", [])[:5]

    except json.JSONDecodeError:
        # 首次解析失败，尝试清理和重新格式化结果
        try:
            # 清理结果文本，移除提示词和可能的用户/模型标识
            result = (
                result.replace(kw_prompt[:-1], "")  # 移除提示词部分
                .replace("user", "")  # 移除"user"字符串
                .replace("model", "")  # 移除"model"字符串
                .strip()  # 去除首尾空白字符
            )
            # 尝试提取结果中的JSON对象部分并重新包装
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            # 再次尝试解析处理后的JSON
            keywords_data = json_repair.loads(result)
            type_keywords = keywords_data.get("answer_type_keywords", [])
            entities_from_query = keywords_data.get("entities_from_query", [])[:5]

        # 处理解析错误，提供故障响应
        except Exception as e:
            print(f"JSON parsing error: {e}")  # 记录错误信息到控制台
            return PROMPTS["fail_response"]  # 返回预定义的失败响应

    # 构建查询上下文，这是MiniRAG的核心处理步骤
    context = await _build_mini_query_context(
        entities_from_query,  # 从查询中识别的实体
        type_keywords,  # 识别的类型关键词
        query,  # 原始查询
        knowledge_graph_inst,  # 知识图谱实例
        entities_vdb,  # 实体向量数据库
        entity_name_vdb,  # 实体名称向量数据库
        relationships_vdb,  # 关系向量数据库
        chunks_vdb,  # 文本块向量数据库
        text_chunks_db,  # 文本块KV存储
        embedder,  # 嵌入模型
        query_param,  # 查询参数
    )

    # 如果只需要上下文而不需要生成回答
    if query_param.only_need_context:
        return context  # 直接返回构建的上下文
    # 如果上下文为空
    if context is None:
        return PROMPTS["fail_response"]  # 返回预定义的失败响应

    # 获取RAG响应提示词模板
    sys_prompt_temp = PROMPTS["rag_response"]
    # 格式化系统提示词，填入上下文和响应类型
    sys_prompt = sys_prompt_temp.format(
        context_data=context,  # 构建的查询上下文
        response_type=query_param.response_type  # 期望的响应类型
    )
    # 调用LLM模型生成最终回答，使用原始查询作为用户提示，上下文作为系统提示
    response = await use_model_func(
        query,  # 用户原始查询
        system_prompt=sys_prompt,  # 包含上下文的系统提示
    )

    # 返回生成的最终回答
    return response
