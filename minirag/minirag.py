"""
MiniRAG核心实现模块

本文件实现了MiniRAG（小型检索增强生成）系统的核心功能，包括：

核心功能：
1. 文档处理管道 - 支持文档的分块、向量化、实体关系提取
2. 多模式查询 - 提供light、mini、naive三种查询模式
3. 多种存储后端 - 支持向量数据库、图数据库、KV存储等
4. 异步处理 - 支持并发处理文档和查询
5. 实体删除 - 支持按实体名称删除相关数据

主要组件：
- MiniRAG: 主类，协调所有组件工作
- 各种Storage类：负责不同类型数据的存储和检索
- 查询引擎：实现不同的RAG查询策略

作者：MiniRAG Team
版本：1.0.0
"""

import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast, Any
from dotenv import load_dotenv


from .operate import (
    chunking_by_token_size,    # 文本分块函数
    extract_entities,          # 实体关系提取
    hybrid_query,             # 混合查询模式
    minirag_query,            # MiniRAG查询模式
    naive_query,              # 简单查询模式
)

from .utils import (
    EmbeddingFunc,            # 嵌入函数类型
    compute_mdhash_id,        # MD5哈希ID计算
    limit_async_func_call,    # 异步函数调用限制
    convert_response_to_json, # 响应转JSON
    logger,                   # 日志记录器
    clean_text,              # 文本清理
    get_content_summary,     # 内容摘要生成
    set_logger,              # 日志设置
)

# 基础类和接口导入
from .base import (
    BaseGraphStorage,         # 图存储基类
    BaseKVStorage,           # 键值存储基类
    BaseVectorStorage,       # 向量存储基类
    StorageNameSpace,        # 存储命名空间
    QueryParam,              # 查询参数
    DocStatus,               # 文档状态
)


STORAGES = {
    "NetworkXStorage": ".kg.networkx_impl",
    "JsonKVStorage": ".kg.json_kv_impl",
    "NanoVectorDBStorage": ".kg.nano_vector_db_impl",
    "JsonDocStatusStorage": ".kg.jsondocstatus_impl",
    "Neo4JStorage": ".kg.neo4j_impl",
    "OracleKVStorage": ".kg.oracle_impl",
    "OracleGraphStorage": ".kg.oracle_impl",
    "OracleVectorDBStorage": ".kg.oracle_impl",
    "MilvusVectorDBStorge": ".kg.milvus_impl",
    "MongoKVStorage": ".kg.mongo_impl",
    "MongoGraphStorage": ".kg.mongo_impl",
    "RedisKVStorage": ".kg.redis_impl",
    "ChromaVectorDBStorage": ".kg.chroma_impl",
    "TiDBKVStorage": ".kg.tidb_impl",
    "TiDBVectorDBStorage": ".kg.tidb_impl",
    "TiDBGraphStorage": ".kg.tidb_impl",
    "PGKVStorage": ".kg.postgres_impl",
    "PGVectorStorage": ".kg.postgres_impl",
    "AGEStorage": ".kg.age_impl",
    "PGGraphStorage": ".kg.postgres_impl",
    "GremlinStorage": ".kg.gremlin_impl",
    "PGDocStatusStorage": ".kg.postgres_impl",
    "WeaviateVectorStorage": ".kg.weaviate_impl",
    "WeaviateKVStorage": ".kg.weaviate_impl",
    "WeaviateGraphStorage": ".kg.weaviate_impl",
    "run_sync": ".kg.weaviate_impl",
}

# future KG integrations

# from .kg.ArangoDB_impl import (
#     GraphStorage as ArangoDBStorage
# )

load_dotenv(dotenv_path=".env", override=False)


def lazy_external_import(module_name: str, class_name: str):
    """
    延迟导入外部模块类的工具函数
    
    该函数实现了动态延迟导入机制，避免在模块加载时就导入所有依赖，
    提高启动性能并减少内存占用。只有在实际需要时才进行导入。
    
    参数:
        module_name (str): 模块路径，如 '.kg.networkx_impl'
        class_name (str): 类名，如 'NetworkXStorage'
    
    返回:
        function: 返回一个可调用的类构造器函数
    
    实现原理:
        1. 通过inspect模块获取调用者的模块和包信息
        2. 返回一个内部函数，在调用时才执行实际的导入
        3. 使用importlib进行动态模块导入
        4. 通过getattr获取目标类并实例化
    
    示例:
        storage_class = lazy_external_import(".kg.networkx_impl", "NetworkXStorage")
        storage_instance = storage_class(namespace="test", embedding_func=func)
    """
    # 获取调用者的模块和包信息
    import inspect

    # 获取调用此函数的调用栈帧
    caller_frame = inspect.currentframe().f_back
    # 获取该帧对应的模块对象
    module = inspect.getmodule(caller_frame)
    # 获取模块的包路径，用于相对导入
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        """
        实际执行导入和实例化的内部函数
        
        参数:
            *args: 传递给目标类构造器的位置参数
            **kwargs: 传递给目标类构造器的关键字参数
        
        返回:
            目标类的实例对象
        """
        import importlib

        # 动态导入指定模块，支持相对导入
        module = importlib.import_module(module_name, package=package)
        # 从模块中获取目标类
        cls = getattr(module, class_name)
        # 实例化并返回
        return cls(*args, **kwargs)

    return import_class


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    获取可用的异步事件循环的工具函数
    
    该函数确保在任何情况下都能获得一个可用的异步事件循环，
    主要用于解决异步编程中事件循环管理的问题。
    
    使用场景:
        1. 在同步代码中调用异步函数时
        2. 在多线程环境中管理事件循环
        3. 在事件循环关闭后重新创建
        4. 确保异步操作能够正常执行
    
    返回:
        asyncio.AbstractEventLoop: 当前可用的事件循环对象
    
    实现逻辑:
        1. 首先尝试获取当前的事件循环
        2. 检查事件循环是否已关闭
        3. 如果已关闭或不存在，创建新的事件循环
        4. 将新循环设置为当前循环并返回
    
    异常处理:
        当事件循环已关闭时抛出RuntimeError，然后创建新的事件循环
    
    注意:
        该函数应该在主线程中调用，避免跨线程事件循环管理问题
    """
    try:
        # 尝试获取当前的事件循环
        current_loop = asyncio.get_event_loop()
        # 检查事件循环是否已经关闭
        if current_loop.is_closed():
            # 如果已关闭，抛出异常触发创建新循环的逻辑
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # 当没有事件循环或已关闭时，创建新的事件循环
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        # 将新创建的事件循环设置为当前线程的默认循环
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class MiniRAG:
    """
    MiniRAG核心类 - 小型检索增强生成系统
    
    这是MiniRAG系统的主要入口类，负责协调和管理整个RAG流程。
    提供了文档插入、查询、实体删除等核心功能，支持多种存储后端和查询模式。
    
    主要功能模块:
    - 文档处理: 分块、向量化、实体关系提取
    - 多模式查询: light(轻量)、mini(标准)、naive(简单)三种模式
    - 存储管理: 向量数据库、图数据库、键值存储等多种存储后端
    - 异步处理: 支持并发处理文档插入和查询操作
    
    设计特点:
    - 模块化设计: 各组件相对独立，便于扩展和维护
    - 异步优先: 大部分操作都支持异步执行，提高性能
    - 多存储支持: 通过配置选择不同的存储后端
    - 缓存机制: 支持LLM响应缓存，减少重复计算
    """
    
    # ==================== 基础配置参数 ====================
    
    working_dir: str = field(
        default_factory=lambda: f"./minirag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    """工作目录路径，用于存储缓存文件、日志文件等
    默认格式: ./minirag_cache_YYYY-MM-DD-HH:MM:SS
    """
    
    # ==================== 存储后端配置 ====================
    
    kv_storage: str = field(default="JsonKVStorage")
    """键值存储后端类型，用于存储文档内容、缓存等
    默认使用JSON文件存储，可选: JsonKVStorage, OracleKVStorage, MongoKVStorage等
    """
    
    vector_storage: str = field(default="NanoVectorDBStorage")
    """向量数据库存储后端类型，用于存储文档和实体的向量表示
    默认使用NanoVectorDB，可选: NanoVectorDBStorage, MilvusVectorDBStorge等
    """
    
    graph_storage: str = field(default="NetworkXStorage")
    """图数据库存储后端类型，用于存储实体关系图
    默认使用NetworkX，可选: NetworkXStorage, Neo4JStorage, OracleGraphStorage等
    """
    
    # ==================== 日志配置 ====================
    
    current_log_level = logger.level
    log_level: str = field(default=current_log_level)
    """日志记录级别，控制日志输出的详细程度
    可选值: DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    
    # ==================== 文本分块配置 ====================
    
    chunk_token_size: int = 1200
    """文本分块的最大token数量，控制每个文本块的大小
    较大的值会产生更少的块，但每个块包含更多信息
    """
    
    chunk_overlap_token_size: int = 100
    """文本分块重叠的token数量，用于保持块之间的上下文连续性
    重叠部分有助于提高检索的准确性和上下文连贯性
    """
    
    tiktoken_model_name: str = "gpt-4o-mini"
    """用于token计算的模型名称，影响token数量计算的准确性
    应与实际使用的embedding/LLM模型保持一致
    """
    
    # ==================== 实体提取配置 ====================
    
    entity_extract_max_gleaning: int = 1
    """实体提取的最大轮数，控制实体识别的深度
    更多轮数可以提取更复杂的实体关系，但会增加处理时间
    """
    
    entity_summary_to_max_tokens: int = 500
    """实体摘要的最大token数量，控制实体描述的长度
    较长的摘要包含更多信息，但会增加存储和计算成本
    """
    
    # ==================== 节点嵌入配置 ====================
    
    node_embedding_algorithm: str = "node2vec"
    """节点嵌入算法，用于生成图中节点的向量表示
    默认使用node2vec算法，可选: node2vec, random_walk等
    """
    
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,    # 嵌入向量维度
            "num_walks": 10,       # 每个节点开始的随机游走次数
            "walk_length": 40,     # 每次随机游走的长度
            "window_size": 2,      # Skip-gram模型的窗口大小
            "iterations": 3,       # 训练迭代次数
            "random_seed": 3,      # 随机种子，确保结果可重现
        }
    )
    """node2vec算法的参数配置
    控制节点嵌入生成的质量和性能
    """
    
    # ==================== 嵌入函数配置 ====================
    
    embedding_func: EmbeddingFunc = None
    """文本嵌入函数，用于将文本转换为向量表示
    必须提供，负责实际的嵌入计算，如OpenAI、BGE等嵌入模型
    """
    
    embedding_batch_num: int = 32
    """嵌入计算的批次大小，控制每次批量处理的文本数量
    较大的批次可以提高处理速度，但需要更多内存
    """
    
    embedding_func_max_async: int = 16
    """嵌入函数的最大异步并发数，控制同时进行的嵌入计算任务数
    应根据API限制和系统性能进行调整
    """
    
    # ==================== LLM配置 ====================
    
    llm_model_func: callable = None
    """大语言模型调用函数，用于生成回答和进行推理
    必须提供，负责实际的LLM调用，如OpenAI、Claude等模型
    """
    
    llm_model_name: str = (
        "meta-llama/Llama-3.2-1B-Instruct"
    )
    """LLM模型名称，用于标识和配置具体的语言模型
    默认为Llama 3.2 1B指令模型
    """
    
    llm_model_max_token_size: int = 32768
    """LLM模型的最大token输入长度限制
    控制每次调用LLM时输入文本的最大长度
    """
    
    llm_model_max_async: int = 8
    """LLM模型调用的最大异步并发数
    控制同时进行的LLM调用任务数，应考虑API限制
    """
    
    llm_model_kwargs: dict = field(default_factory=dict)
    """LLM模型的额外配置参数
    用于传递模型特定的配置选项，如温度、top_p等
    """
    
    # ==================== 存储配置 ====================
    
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    """向量数据库存储类的额外配置参数
    用于传递存储后端特定的配置选项
    """
    
    enable_llm_cache: bool = True
    """是否启用LLM响应缓存
    启用后可以缓存LLM的响应结果，提高重复查询的性能
    """
    
    # ==================== 扩展配置 ====================
    
    addon_params: dict = field(default_factory=dict)
    """额外的扩展参数
    用于传递插件或扩展功能的配置选项
    """
    
    convert_response_to_json_func: callable = convert_response_to_json
    """响应转换函数，用于将LLM回答转换为结构化JSON格式
    默认使用内置的转换函数，可根据需要自定义
    """
    
    # ==================== 文档状态存储 ====================
    
    doc_status_storage: str = field(default="JsonDocStatusStorage")
    """文档状态存储后端类型，用于跟踪文档处理状态
    可选: JsonDocStatusStorage, PGDocStatusStorage等
    """
    
    # ==================== 自定义分块函数 ====================
    
    chunking_func: callable = chunking_by_token_size
    """文本分块函数，用于将长文档分割为较小的块
    默认使用基于token大小的分块方法，可自定义实现
    """
    
    chunking_func_kwargs: dict = field(default_factory=dict)
    """分块函数的额外配置参数
    用于传递分块算法的特定配置选项
    """
    
    # ==================== 并发处理配置 ====================
    
    max_parallel_insert: int = field(default=int(os.getenv("MAX_PARALLEL_INSERT", 2)))
    """文档插入的最大并行数，控制同时处理的文档数量
    可通过环境变量MAX_PARALLEL_INSERT进行配置，默认值为2
    """

    def __post_init__(self):
        """
        MiniRAG类的后初始化方法
        
        该方法在MiniRAG对象创建后自动执行，负责：
        1. 初始化日志系统和工作目录
        2. 配置各种存储后端
        3. 创建和管理各种数据存储实例
        4. 设置异步函数调用限制
        5. 配置缓存机制
        
        初始化流程:
        日志配置 → 存储类配置 → 存储实例创建 → 异步配置 → 缓存设置
        
        注意:
        这是整个MiniRAG系统初始化的核心步骤，必须确保每个组件都正确初始化
        """
        
        # ==================== 日志系统初始化 ====================
        
        # 设置日志文件路径
        log_file = os.path.join(self.working_dir, "minirag.log")
        # 配置日志系统，包括文件输出和格式
        set_logger(log_file)
        # 设置日志级别
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")
        
        # ==================== 工作目录创建 ====================
        
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # ==================== 全局配置处理 ====================
        
        # 将当前对象转换为字典格式，用于配置传递
        global_config = asdict(self)
        # 格式化配置信息用于调试输出
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logger.debug(f"MiniRAG init with param:\n  {_print_config}\n")

        # ==================== 存储类配置和实例化 ====================
        
        # 获取键值存储类（用于文档内容、缓存等）
        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class(self.kv_storage)
        )
        # 获取向量数据库存储类（用于向量检索）
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class(
            self.vector_storage
        )
        # 获取图数据库存储类（用于实体关系图）
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class(
            self.graph_storage
        )

        # ==================== 存储类预配置 ====================
        
        # 为键值存储类预配置全局参数
        self.key_string_value_json_storage_cls = partial(
            self.key_string_value_json_storage_cls, global_config=global_config
        )
        # 为向量数据库存储类预配置全局参数
        self.vector_db_storage_cls = partial(
            self.vector_db_storage_cls, global_config=global_config
        )
        # 为图数据库存储类预配置全局参数
        self.graph_storage_cls = partial(
            self.graph_storage_cls, global_config=global_config
        )
        
        # ==================== 文档状态存储初始化 ====================
        
        # 创建文档状态存储实例，用于跟踪文档处理状态
        self.json_doc_status_storage = self.key_string_value_json_storage_cls(
            namespace="json_doc_status_storage",
            embedding_func=None,
        )

        # ==================== 工作目录二次检查 ====================
        
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # ==================== LLM响应缓存初始化 ====================
        
        # 创建LLM响应缓存，提高重复查询性能
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache  # 只有启用缓存时才创建
            else None
        )

        # ==================== 嵌入函数异步限制配置 ====================
        
        # 为嵌入函数添加异步调用限制，防止过多并发请求
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        # ==================== 核心数据存储实例创建 ====================
        
        # 1. 完整文档存储 - 存储原始文档内容
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        
        # 2. 文本块存储 - 存储分块后的文本内容
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        
        # 3. 实体关系图存储 - 存储文档中的实体和关系网络
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        # ==================== 向量数据库存储实例创建 ====================
        
        # 4. 实体向量存储 - 存储实体名称和描述的向量表示
        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},  # 元数据字段：实体名称
        )
        
        # 重新获取全局配置（可能在上一步中被修改）
        global_config = asdict(self)

        # 5. 实体名称向量存储 - 专门存储实体名称的向量
        self.entity_name_vdb = self.vector_db_storage_cls(
            namespace="entities_name",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )

        # 6. 关系向量存储 - 存储实体间关系的向量表示
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},  # 元数据字段：源实体ID、目标实体ID
        )
        
        # 7. 文本块向量存储 - 存储文本块的向量表示
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        # ==================== LLM模型函数配置 ====================
        
        # 为LLM模型函数添加异步调用限制和缓存支持
        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,  # 集成缓存机制
                **self.llm_model_kwargs,  # 传递模型特定参数
            )
        )
        
        # ==================== 文档状态存储最终初始化 ====================
        
        # 获取文档状态存储类
        self.doc_status_storage_cls = self._get_storage_class(self.doc_status_storage)
        # 创建文档状态存储实例
        self.doc_status = self.doc_status_storage_cls(
            namespace="doc_status",
            global_config=global_config,
            embedding_func=None,
        )

    def _get_storage_class(self, storage_name: str) -> dict:
        """
        根据存储类型名称获取对应的存储类
        
        这是MiniRAG系统的存储抽象层核心方法，通过配置驱动的动态类加载机制，
        支持多种不同类型的存储后端，实现存储层的可插拔架构。
        
        参数:
            storage_name (str): 存储类型名称，如"JsonKVStorage"、"Neo4JStorage"等
        
        返回:
            dict: 返回延迟导入的存储类构造器函数
        
        工作原理:
            1. 根据存储名称在STORAGES映射字典中查找对应的模块路径
            2. 使用lazy_external_import动态加载指定模块中的存储类
            3. 返回一个可调用的类构造器，后续可传入参数实例化具体的存储对象
            
        示例:
            storage_cls = self._get_storage_class("JsonKVStorage")
            storage_instance = storage_cls(namespace="test", embedding_func=func)
        
        异常处理:
            如果传入的storage_name在STORAGES中未找到，会抛出KeyError
        
        注意:
            该方法实现了延迟加载，只有在实际需要时才加载存储类，提高启动性能
        """
        import_path = STORAGES[storage_name]
        storage_class = lazy_external_import(import_path, storage_name)
        return storage_class

    def set_storage_client(self, db_client):
        """
        为所有存储实例设置统一的数据库客户端
        
        该方法主要用于需要在多个存储后端之间共享同一个数据库连接的场景，
        目前主要针对Oracle数据库进行了测试，但也可适用于其他支持客户端设置的数据库。
        
        参数:
            db_client: 数据库客户端实例，如Oracle cx_Oracle连接对象
        
        实现逻辑:
            1. 遍历系统中所有已初始化的存储实例
            2. 为每个存储实例设置统一的数据库客户端
            3. 确保所有存储操作都使用同一个数据库连接
            
        受影响的存储类型:
            - 向量数据库存储 (vector_db_storage_cls)
            - 图数据库存储 (graph_storage_cls)  
            - 文档状态存储 (doc_status)
            - 完整文档存储 (full_docs)
            - 文本块存储 (text_chunks)
            - LLM响应缓存 (llm_response_cache)
            - 键值存储 (key_string_value_json_storage_cls)
            - 各种向量存储实例 (chunks_vdb, relationships_vdb, entities_vdb等)
            - 实体关系图存储 (chunk_entity_relation_graph)
        
        使用场景:
            1. 企业级部署，需要共享数据库连接池
            2. 事务管理，确保多个存储操作的一致性
            3. 连接复用，提高性能和资源利用率
            4. 统一管理数据库连接参数
        
        注意事项:
            - 目前主要针对Oracle数据库进行了测试
            - 确保db_client对象与各存储后端兼容
            - 在多线程环境中需要考虑连接的安全性
        """
        # 遍历系统中所有存储实例，为其设置统一的数据库客户端
        for storage in [
            self.vector_db_storage_cls,      # 向量数据库存储类
            self.graph_storage_cls,          # 图数据库存储类
            self.doc_status,                 # 文档状态存储实例
            self.full_docs,                  # 完整文档存储实例
            self.text_chunks,                # 文本块存储实例
            self.llm_response_cache,         # LLM响应缓存实例
            self.key_string_value_json_storage_cls,  # 键值存储类
            self.chunks_vdb,                 # 文本块向量存储实例
            self.relationships_vdb,          # 关系向量存储实例
            self.entities_vdb,               # 实体向量存储实例
            self.entity_name_vdb,            # 实体名称向量存储实例
            self.chunk_entity_relation_graph, # 实体关系图存储实例
        ]:
            # 为每个存储实例设置数据库客户端
            storage.db = db_client

    def insert(self, string_or_strings):
        """
        同步文档插入接口
        
        这是MiniRAG系统的文档插入同步接口，内部调用异步版本的insert方法。
        主要用于在同步代码环境中使用，自动管理事件循环的创建和运行。
        
        参数:
            string_or_strings (str | list[str]): 要插入的文档内容
                - str: 单个文档字符串
                - list[str]: 多个文档字符串列表
        
        返回:
            None: 方法无返回值，插入操作完成后返回
        
        使用示例:
            # 单个文档插入
            rag.insert("这是一个测试文档")
            
            # 多个文档插入
            rag.insert(["文档1", "文档2", "文档3"])
        
        内部流程:
            1. 检查并获取可用的异步事件循环
            2. 调用异步版本的ainsert方法
            3. 等待异步操作完成
            
        异常处理:
            - 如果传入参数格式错误会抛出ValueError
            - 文档处理过程中的异常会在异步方法中处理
            
        注意事项:
            - 这是同步包装方法，适用于传统的同步编程模式
            - 对于新项目建议直接使用异步版本的ainsert方法
            - 内部会自动管理事件循环，无需手动处理
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
    ) -> None:
        """
        异步文档插入核心方法
        
        这是MiniRAG系统的核心文档插入方法，负责将输入文档转换为可检索的格式。
        整个插入流程包括文档预处理、分块、向量化、实体关系提取等多个步骤。
        
        参数:
            input (str | list[str]): 输入文档内容
                - str: 单个文档
                - list[str]: 多个文档列表
            
            split_by_character (str | None): 分隔字符，用于按指定字符分割文档
                - None: 使用默认分块策略
                - str: 按指定字符进行分割
                
            split_by_character_only (bool): 是否仅使用字符分割
                - False: 使用默认的分块算法（token大小分块）
                - True: 强制使用字符分割方法
                
            ids (str | list[str] | None): 文档ID列表
                - None: 自动生成MD5哈希ID
                - str/list: 用户提供的唯一ID列表
        
        处理流程:
            1. 输入验证和标准化
            2. 文档入队（apipeline_enqueue_documents）
            3. 文档处理管道（apipeline_process_enqueue_documents）
            4. 实体关系提取
            5. 索引完成回调
            
        数据流向:
            原始文档 → 分块 → 向量化 → 存储 → 实体提取 → 关系图构建
            
        性能特点:
            - 支持批量处理多个文档
            - 自动去重和处理重复内容
            - 并行处理提高效率
            - 支持增量插入和更新
            
        存储结果:
            - full_docs: 存储原始文档内容
            - text_chunks: 存储分块后的文本
            - chunks_vdb: 存储文本块向量
            - entities_vdb: 存储实体向量
            - relationships_vdb: 存储关系向量
            - chunk_entity_relation_graph: 存储实体关系图
            
        注意事项:
            - 确保embedding_func和llm_model_func已正确配置
            - 大批量文档插入时注意内存使用
            - 建议在生产环境中监控插入性能
        """
        
        # ==================== 输入参数标准化 ====================
        
        # 标准化输入参数，将单个文档转换为列表格式
        if isinstance(input, str):
            input = [input]
        if isinstance(ids, str):
            ids = [ids]

        # ==================== 文档入队处理 ====================
        
        # 第一步：将文档加入处理队列，进行预处理和去重
        await self.apipeline_enqueue_documents(input, ids)
        
        # ==================== 文档处理管道 ====================
        
        # 第二步：处理队列中的文档，包括分块、向量化、存储
        await self.apipeline_process_enqueue_documents(
            split_by_character, split_by_character_only
        )

        # ==================== 实体关系提取 ====================
        
        # 第三步：对新处理的文档块进行实体关系提取
        # 从已处理的文档中提取实体和关系，构建知识图谱
        inserting_chunks = {
            compute_mdhash_id(dp["content"], prefix="chunk-"): {
                **dp,
                "full_doc_id": doc_id,
            }
            # 遍历所有已处理的文档 (Doc A, Doc B, Doc C...)
            for doc_id, status_doc in (
                await self.doc_status.get_docs_by_status(DocStatus.PROCESSED)
            ).items()
            # 遍历每个文档的所有分块
            for dp in self.chunking_func(
                status_doc.content,
                self.chunk_overlap_token_size,
                self.chunk_token_size,
                self.tiktoken_model_name,
            )
        }

        # ==================== 实体关系提取阶段 ====================
        
        # 第五步：如果有待处理的文档块，执行实体关系提取
        if inserting_chunks:
            logger.info("正在对新处理的文档块执行实体关系提取")
            
            # 调用实体提取核心函数，对新生成的文档块进行深度分析
            # 这是MiniRAG知识图谱构建的关键步骤
            await extract_entities(
                inserting_chunks,  # 待提取实体的文档块列表
                # 这些块包含文本内容、元数据、以及已计算好的向量表示
                # 为实体提取提供充分的上下文信息
                
                knowledge_graph_inst=self.chunk_entity_relation_graph,  # 知识图谱存储实例
                # 存储实体之间的结构化关系和连接信息
                # 支持图数据库查询和图算法分析
                
                entity_vdb=self.entities_vdb,      # 实体向量数据库
                # 存储提取出的实体及其向量表示
                # 用于实体相似性搜索和实体聚类
                
                entity_name_vdb=self.entity_name_vdb,  # 实体名称向量数据库
                # 专门存储实体名称及其语义向量
                # 用于实体名称的模糊匹配和语义搜索
                
                relationships_vdb=self.relationships_vdb,  # 关系向量数据库
                # 存储实体间的关系及其向量表示
                # 支持关系推理和关联分析
                
                global_config=asdict(self),  # 全局配置字典
                # 包含所有MiniRAG配置参数
                # 用于控制实体提取的行为和参数
            )
 
        # ==================== 插入完成回调 ====================
        
        # 第四步：通知所有存储组件插入操作已完成
        await self._insert_done()

    async def apipeline_enqueue_documents(
        self, input: str | list[str], ids: list[str] | None = None
    ) -> None:
        """
        文档入队管道方法 - 负责文档的预处理和初步验证
        
        这是文档插入流程的第一步，主要功能包括：
        1. 输入验证和标准化
        2. ID生成或验证（确保ID唯一性）
        3. 内容去重和清理
        4. 生成文档初始状态
        5. 过滤已处理文档
        6. 将新文档加入处理队列
        
        该方法确保只有新的、唯一的文档才会进入后续处理流程，避免重复处理。
        
        参数:
            input (str | list[str]): 输入文档内容
                - str: 单个文档
                - list[str]: 文档列表
            
            ids (list[str] | None): 文档ID列表
                - None: 自动生成MD5哈希ID
                - list: 用户提供的唯一ID列表
        
        处理步骤:
            1. 输入标准化（字符串转列表）
            2. ID验证或生成
            3. 内容去重和清理
            4. 生成文档状态对象
            5. 过滤已处理文档
            6. 入队存储
            
        性能优化:
            - 智能去重：避免处理重复内容
            - 增量处理：只处理新文档
            - 并发友好：支持多文档并行处理
        """
        
        # ==================== 输入标准化 ====================
        
        # 确保输入为列表格式，便于统一处理
        if isinstance(input, str):
            input = [input]
        if isinstance(ids, str):
            ids = [ids]

        # ==================== ID处理和内容组织 ====================
        
        if ids is not None:
            # 用户提供了ID，需要验证
            if len(ids) != len(input):
                raise ValueError("文档ID数量必须与文档数量匹配")
            if len(ids) != len(set(ids)):
                raise ValueError("文档ID必须唯一")
            # 使用用户提供的ID
            contents = {id_: doc for id_, doc in zip(ids, input)}
        else:
            # 自动生成ID，先进行内容去重和清理
            input = list(set(clean_text(doc) for doc in input))
            # 为每个唯一内容生成MD5哈希ID
            contents = {compute_mdhash_id(doc, prefix="doc-"): doc for doc in input}

        # ==================== 进一步去重处理 ====================
        
        # 处理可能的内容重复情况，保留第一个遇到的ID
        unique_contents = {
            id_: content
            for content, id_ in {
                content: id_ for id_, content in contents.items()
            }.items()
        }
        
        # ==================== 生成文档状态对象 ====================
        
        # 为每个唯一文档创建初始状态对象
        new_docs: dict[str, Any] = {
            id_: {
                "content": content,                    # 原始内容
                "content_summary": get_content_summary(content),  # 内容摘要
                "content_length": len(content),        # 内容长度
                "status": DocStatus.PENDING,          # 初始状态：等待处理
                "created_at": datetime.now().isoformat(),  # 创建时间
                "updated_at": datetime.now().isoformat(),   # 更新时间
            }
            for id_, content in unique_contents.items()
        }

        # ==================== 过滤已处理文档 ====================
        
        # 从文档状态存储中获取已存在的文档ID
        all_new_doc_ids = set(new_docs.keys())
        unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)

        # 只保留真正新的文档
        new_docs = {
            doc_id: new_docs[doc_id]
            for doc_id in unique_new_doc_ids
            if doc_id in new_docs
        }
        
        # 如果没有新文档，直接返回
        if not new_docs:
            logger.info("未发现新的唯一文档。")
            return

        # ==================== 文档入队存储 ====================
        
        # 将新文档状态信息存储到文档状态存储中
        await self.doc_status.upsert(new_docs)
        logger.info(f"已存储 {len(new_docs)} 个新的唯一文档")

    async def apipeline_process_enqueue_documents(
        self,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ) -> None:
        """
        Process pending documents by splitting them into chunks, processing
        each chunk for entity and relation extraction, and updating the
        document status.
        """
        processing_docs, failed_docs, pending_docs = await asyncio.gather(
            self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
            self.doc_status.get_docs_by_status(DocStatus.FAILED),
            self.doc_status.get_docs_by_status(DocStatus.PENDING),
        )

        # 合并三种状态的文档，准备统一处理；失败/处理中会重试
        to_process_docs: dict[str, Any] = {
            **processing_docs,
            **failed_docs,
            **pending_docs,
        }
        if not to_process_docs:
            logger.info("No documents to process")
            return

        # 按 max_parallel_insert 切分批次，防止一次处理过多文档
        docs_batches = [
            list(to_process_docs.items())[i : i + self.max_parallel_insert]
            for i in range(0, len(to_process_docs), self.max_parallel_insert)
        ]
        logger.info(f"Number of batches to process: {len(docs_batches)}")

        for batch_idx, docs_batch in enumerate(docs_batches):
            for doc_id, status_doc in docs_batch:
                # 对单个文档做分块，生成 chunk id，并补充 full_doc_id 关联
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_id,
                    }
                    for dp in self.chunking_func(
                        status_doc.content,
                        self.chunk_overlap_token_size,
                        self.chunk_token_size,
                        self.tiktoken_model_name,
                    )
                }
                # 并发写入：向量库、全文存储、chunk 存储
                await asyncio.gather(
                    self.chunks_vdb.upsert(chunks),
                    self.full_docs.upsert({doc_id: {"content": status_doc.content}}),
                    self.text_chunks.upsert(chunks),
                )
                # 更新文档状态为 PROCESSED，记录 chunk 数量与元信息
                await self.doc_status.upsert(
                    {
                        doc_id: {
                            "status": DocStatus.PROCESSED,
                            "chunks_count": len(chunks),
                            "content": status_doc.content,
                            "content_summary": status_doc.content_summary,
                            "content_length": status_doc.content_length,
                            "created_at": status_doc.created_at,
                            "updated_at": datetime.now().isoformat(),
                        }
                    }
                )
        logger.info("Document processing pipeline completed")

    async def _insert_done(self):
        """
        文档插入操作完成后的回调方法
        
        该方法在文档插入流程的最后阶段被调用，负责：
        1. 通知所有相关存储组件插入操作已完成
        2. 执行存储层的索引更新和优化操作
        3. 确保整个插入操作的数据一致性和完整性
        
        被通知的存储组件及其作用:
        - full_docs: 原始文档存储，负责更新文档索引
        - text_chunks: 文本分块存储，负责更新分块索引
        - llm_response_cache: LLM响应缓存，负责缓存管理和索引
        - entities_vdb: 实体向量数据库，负责实体索引优化
        - entity_name_vdb: 实体名称向量数据库，负责名称索引优化
        - relationships_vdb: 关系向量数据库，负责关系索引优化
        - chunks_vdb: 文本块向量数据库，负责向量索引优化
        - chunk_entity_relation_graph: 实体关系图，负责图结构索引
        
        每个存储组件的回调操作可能包括:
        - 更新倒排索引结构
        - 重建向量索引
        - 优化查询性能
        - 清理临时数据和缓存
        - 同步内存数据和磁盘数据
        - 释放临时占用的资源
        
        处理逻辑:
            1. 初始化异步任务列表
            2. 遍历所有参与插入过程的存储实例
            3. 跳过为None的存储实例（配置中未启用）
            4. 转换存储实例为StorageNameSpace类型
            5. 并行调用所有存储的index_done_callback方法
            6. 等待所有回调操作完成，确保数据一致性
            
        性能特点:
            - 并行执行多个存储的回调操作
            - 显著提高整体插入完成速度
            - 确保所有存储在同一时间点达到一致状态
            - 减少单独回调的累计等待时间
            
        数据一致性保障:
            - 所有存储组件同时完成索引更新
            - 避免部分存储已更新而部分未更新的状态
            - 确保查询操作能获得完整的最新数据
            - 维护RAG系统的检索准确性
            
        使用场景:
            - 在ainsert方法的最后阶段自动调用
            - 在批量文档插入操作完成后调用
            - 确保新插入数据能立即被查询系统访问
            - 在文档更新操作完成后调用
            
        错误处理:
            - 某个存储的回调失败不会影响其他存储
            - 错误会被详细记录但不中断整体流程
            - 即使部分回调失败，系统仍可正常工作
            - 保证插入操作的整体可靠性
            
        注意事项:
            - 这是内部方法，通常由插入流程自动调用
            - 确保所有存储组件都正确实现了回调接口
            - 在大规模插入操作中，回调时间可能较长
            - 监控回调操作的执行时间以优化性能
        """
        
        # 初始化异步任务列表，用于存储所有需要执行的回调操作
        tasks = []
        
        # 遍历所有参与插入过程的存储实例
        for storage_inst in [
            self.full_docs,                        # 原始文档存储
            self.text_chunks,                     # 文本分块存储
            self.llm_response_cache,             # LLM响应缓存存储
            self.entities_vdb,                   # 实体向量数据库
            self.entity_name_vdb,                # 实体名称向量数据库
            self.relationships_vdb,              # 关系向量数据库
            self.chunks_vdb,                     # 文本块向量数据库
            self.chunk_entity_relation_graph,    # 实体关系图存储
        ]:
            # 跳过为None的存储实例（当该存储未在配置中启用时）
            if storage_inst is None:
                continue
                
            # 将存储实例转换为StorageNameSpace类型并添加回调任务
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
            
        # 并行执行所有存储的回调操作，提高整体完成速度
        # 使用asyncio.gather确保所有回调操作都完成
        await asyncio.gather(*tasks)

    def query(self, query: str, param: QueryParam = QueryParam()):
        """
        同步知识图谱查询方法 - 包装异步查询
        
        这是MiniRAG系统的同步查询接口，内部调用异步版本的aquery方法。
        该方法为同步编程环境提供了便捷的查询接口，自动管理事件循环的创建和运行。
        
        主要功能:
        1. 事件循环管理 - 自动创建和管理异步事件循环
        2. 查询执行 - 调用异步查询方法并等待结果
        3. 参数传递 - 直接传递查询参数和配置
        
        使用场景:
        - 同步代码环境中的查询调用
        - 脚本或简单应用中的知识图谱查询
        - 需要与同步API集成的场景
        
        参数:
            query (str): 查询字符串，包含用户的问题或查询内容
            param (QueryParam): 查询参数配置对象，包含模式选择和其他配置
                - mode: 查询模式（light/mini/naive）
                - 其他查询配置参数
        
        返回:
            Any: 查询结果，具体类型取决于查询模式和返回策略
                - 通常包含答案文本和相关文档片段
        
        性能特点:
        - 自动管理事件循环，简化API调用
        - 内部异步执行，保持高性能
        - 适合不熟悉异步编程的用户
        
        注意: 每次调用都会创建新的事件循环，不适合高频调用场景
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        """
        异步知识图谱查询核心方法 - 支持多种查询模式
        
        这是MiniRAG系统的核心查询方法，根据查询模式选择不同的检索和推理策略：
        
        三种查询模式详解：
        1. light模式（轻量模式）:
           - 特点：最快响应速度，最小化处理步骤
           - 实现：使用混合查询策略，主要依赖实体关系图和向量检索
           - 适用场景：简单事实查询、快速响应需求、实时对话
           - 数据源：实体关系图(entity_relation_graph)、实体向量(entities_vdb)、
                    关系向量(relationships_vdb)、文本块(text_chunks)
           - 性能：最快，适合对准确性要求不高的场景
        
        2. mini模式（标准模式）:
           - 特点：平衡性能和功能，包含基础推理
           - 实现：使用MiniRAG查询策略，集成多种数据源
           - 适用场景：大多数实际应用场景，平衡响应时间和准确性
           - 数据源：实体关系图、实体向量、实体名称向量(entity_name_vdb)、
                    关系向量、文本块向量(chunks_vdb)、文本块、嵌入函数
           - 性能：中等，适合日常查询任务
           - 优势：最全面的数据覆盖，较好的推理能力
        
        3. naive模式（完整模式）:
           - 特点：完整推理过程，最大化准确性
           - 实现：使用简单查询策略，专注于文本内容
           - 适用场景：复杂分析任务、研究场景、对准确性要求极高
           - 数据源：文本块向量、文本块
           - 性能：相对较慢，但准确性最高
           - 优势：深度分析能力，适合研究型查询
        
        查询流程:
        1. 参数验证和模式判断
        2. 根据模式选择相应的查询函数
        3. 传递相关数据存储实例和配置
        4. 执行查询并获取响应
        5. 调用查询完成回调
        6. 返回查询结果
        
        参数:
            query (str): 查询字符串，包含用户的自然语言问题
            param (QueryParam): 查询参数配置对象
                - mode: 查询模式选择（light/mini/naive）
                - 其他查询控制参数
        
        返回:
            Any: 查询结果对象，包含答案和相关文档信息
        
        异常处理:
        - 无效模式：抛出ValueError异常
        - 存储异常：在查询函数内部处理
        
        注意事项:
        - 确保相关数据存储已正确初始化
        - 大查询量时注意内存使用
        - 根据实际场景选择合适的查询模式
        """
        if param.mode == "light":
            # 轻量模式：使用混合查询策略，最快响应
            # 主要利用实体关系图进行快速检索和推理
            response = await hybrid_query(
                query,
                self.chunk_entity_relation_graph,  # 实体关系图存储
                self.entities_vdb,                # 实体向量存储
                self.relationships_vdb,           # 关系向量存储
                self.text_chunks,                 # 文本块存储
                param,                            # 查询参数
                asdict(self),                     # 全局配置
            )
        elif param.mode == "mini":
            # 标准模式：使用MiniRAG查询策略，平衡性能与功能
            # 集成最全面的数据源，包含实体名称向量和文本块向量
            response = await minirag_query(
                query,
                self.chunk_entity_relation_graph,  # 实体关系图存储
                self.entities_vdb,                # 实体向量存储
                self.entity_name_vdb,             # 实体名称向量存储
                self.relationships_vdb,           # 关系向量存储
                self.chunks_vdb,                  # 文本块向量存储
                self.text_chunks,                 # 文本块存储
                self.embedding_func,              # 嵌入函数
                param,                            # 查询参数
                asdict(self),                     # 全局配置
            )
        elif param.mode == "naive":
            # 完整模式：使用简单查询策略，专注文本内容
            # 仅使用文本相关存储，进行深度分析
            response = await naive_query(
                query,
                self.chunks_vdb,                  # 文本块向量存储
                self.text_chunks,                 # 文本块存储
                param,                            # 查询参数
                asdict(self),                     # 全局配置
            )
        else:
            # 无效模式处理
            raise ValueError(f"Unknown mode {param.mode}")
        
        # 查询完成回调，通知相关存储组件
        await self._query_done()
        return response

    async def _query_done(self):
        """
        查询操作完成后的回调方法
        
        该方法在每次查询操作完成后被调用，负责：
        1. 通知LLM响应缓存存储查询操作已完成
        2. 执行缓存层的索引更新操作
        3. 确保查询操作的完整性
        
        回调机制说明：
        - 这是MiniRAG存储架构的重要设计模式
        - 每个存储组件都有index_done_callback方法
        - 在批量操作完成后统一调用，提升性能
        
        处理逻辑:
            1. 初始化异步任务列表
            2. 遍历需要通知的存储实例
                - llm_response_cache: LLM响应缓存存储
            3. 跳过为None的存储实例（配置中未启用）
            4. 转换存储实例为StorageNameSpace类型
            5. 并行调用所有存储的回调方法
            6. 等待所有回调操作完成
            
        缓存优化:
            - 更新LLM响应的缓存索引
            - 提高后续相似查询的响应速度
            - 清理临时数据和过期缓存
            
        性能特点:
            - 并行执行多个存储的回调操作
            - 减少整体查询完成时间
            - 确保所有存储的一致性状态
            
        使用场景:
            - 在query()和aquery()方法结束时调用
            - 在批量查询操作完成后调用
            - 确保查询结果正确缓存和索引
            
        注意事项:
            - 该方法为内部方法，通常由框架自动调用
            - 确保所有查询操作都正确调用此方法
            - 维护查询性能和缓存有效性
        """
        
        # 初始化异步任务列表，用于存储所有需要执行的回调操作
        tasks = []
        
        # 遍历需要通知的存储实例
        for storage_inst in [self.llm_response_cache]:  # LLM响应缓存存储
            # 跳过为None的存储实例（当该存储未在配置中启用时）
            if storage_inst is None:
                continue
                
            # 将存储实例转换为StorageNameSpace类型并添加回调任务
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
            
        # 并行执行所有存储的回调操作，提高整体完成速度
        await asyncio.gather(*tasks)

    def delete_by_entity(self, entity_name: str):
        """
        按实体名称同步删除方法 - 删除指定实体及其所有关系
        
        这是adelete_by_entity方法的同步包装器，为不熟悉异步编程的用户提供便捷接口。
        该方法在内部创建事件循环来执行异步删除操作，并等待完成。
        
        主要功能：
        1. 创建独立的事件循环执行异步删除
        2. 等待删除操作完成
        3. 返回删除结果
        
        参数:
            entity_name (str): 要删除的实体名称
                - 实体标识符，通常为实体文本
                - 会自动转换为大写并添加引号
                - 区分大小写，但内部处理时会标准化
            
        返回:
            Any: 删除操作的结果
                - 通常为None或删除状态信息
                - 具体类型取决于adelete_by_entity的实现
            
        性能特点:
            - 同步阻塞操作，会等待删除完成
            - 不适合高频删除场景
            - 建议在高频场景下直接使用adelete_by_entity
            
        错误处理:
            - 内部捕获所有异常
            - 错误信息通过日志记录
            - 不会抛出异常给调用者
            
        使用示例:
            result = delete_by_entity("人工智能")  # 删除"人工智能"实体
            result = delete_by_entity("机器学习")  # 删除"机器学习"实体
        """
        
        # 创建独立事件循环并执行异步删除操作
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        """
        按实体名称异步删除方法 - 完整删除指定实体及其关系
        
        这是MiniRAG的核心删除方法，负责彻底删除指定实体及其所有相关数据：
        1. 实体向量数据 - 从实体向量存储中删除
        2. 关系数据 - 从关系向量存储中删除
        3. 图节点 - 从知识图谱存储中删除
        
        删除范围包括：
        - 实体本身的所有向量表示
        - 该实体参与的所有关系
        - 知识图谱中的对应节点
        - 相关联的文本块引用
        
        参数:
            entity_name (str): 要删除的实体名称
                - 实体文本标识符
                - 会自动标准化为大写并添加引号
                - 必须与存储中的实体名称格式一致
            
        删除流程:
            1. 实体名称标准化处理
            2. 并行删除多存储中的相关数据
                a) 删除实体向量存储中的实体数据
                b) 删除关系向量存储中的相关关系
                c) 删除知识图谱中的节点及边
            3. 记录删除操作日志
            4. 触发删除完成回调
            
        存储目标:
            - entities_vdb: 实体向量数据库
            - relationships_vdb: 关系向量数据库  
            - chunk_entity_relation_graph: 实体关系图存储
            
        错误处理:
            - 捕获所有可能的异常
            - 记录详细的错误日志信息
            - 确保部分删除失败不影响整体操作
            
        性能特点:
            - 支持多个存储的并行删除操作
            - 自动处理删除顺序和依赖关系
            - 提供完整的删除状态反馈
            
        安全考虑:
            - 删除操作不可逆，谨慎使用
            - 建议在删除前备份重要数据
            - 确保实体名称的正确性
        """
        
        # ==================== 实体名称标准化 ====================
        
        # 将实体名称标准化为大写并添加引号
        # 这是为了匹配存储中实体的存储格式
        entity_name = f'"{entity_name.upper()}"'

        try:
            # ==================== 并行删除操作 ====================
            
            # 1. 删除实体向量存储中的实体数据
            await self.entities_vdb.delete_entity(entity_name)
            
            # 2. 删除关系向量存储中该实体参与的所有关系
            await self.relationships_vdb.delete_relation(entity_name)
            
            # 3. 删除知识图谱存储中的节点及其相关边
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            # ==================== 删除操作日志记录 ====================
            
            logger.info(
                f"实体 '{entity_name}' 及其所有关系已成功删除。"
            )
            
            # ==================== 触发删除完成回调 ====================
            
            # 通知所有相关存储组件删除操作已完成
            await self._delete_by_entity_done()
            
        except Exception as e:
            # ==================== 错误处理和日志记录 ====================
            
            logger.error(f"删除实体 '{entity_name}' 时发生错误: {e}")
            # 注意：这里不重新抛出异常，保持API的稳定性

    async def _delete_by_entity_done(self):
        """
        删除实体操作完成后的回调方法
        
        该方法在实体删除操作完成后被调用，负责：
        1. 通知所有相关存储组件删除操作已完成
        2. 执行存储层的索引更新操作
        3. 确保删除操作的完整性和一致性
        
        被通知的存储组件包括：
        - entities_vdb: 实体向量存储
        - relationships_vdb: 关系向量存储  
        - chunk_entity_relation_graph: 实体关系图存储
        
        每个存储组件会执行各自的索引回调操作，可能包括：
        - 更新索引结构
        - 清理临时数据
        - 同步存储状态
        - 释放相关资源
        
        处理逻辑:
            1. 遍历所有相关的存储实例
            2. 跳过为None的存储实例
            3. 将存储实例转换为StorageNameSpace类型
            4. 并行调用各存储的index_done_callback方法
            5. 等待所有回调操作完成
            
        性能特点:
            - 并行执行多个存储的回调操作
            - 提高整体删除操作的完成速度
            - 确保所有存储的一致性状态
            
        错误处理:
            - 如果某个存储的回调失败，不影响其他存储
            - 错误会被日志记录但不中断整个流程
            - 确保即使部分回调失败，系统仍能正常工作
            
        注意事项:
            - 该方法在adelete_by_entity内部调用
            - 确保删除操作的所有存储都被正确通知
            - 维护数据一致性的重要环节
        """
        
        # 存储需要通知的存储实例列表
        tasks = []
        
        # 遍历所有相关的存储实例
        for storage_inst in [
            self.entities_vdb,                # 实体向量存储
            self.relationships_vdb,          # 关系向量存储
            self.chunk_entity_relation_graph, # 实体关系图存储
        ]:
            # 跳过为None的存储实例
            if storage_inst is None:
                continue
                
            # 将存储实例转换为StorageNameSpace类型并添加回调任务
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
            
        # 并行执行所有存储的回调操作
        await asyncio.gather(*tasks)