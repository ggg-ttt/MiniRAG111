# 导入抽象基类所需模块
from abc import abstractmethod
# 导入数据类、字段、枚举等类型工具
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict, Optional, Union, Literal, Generic, TypeVar
import os
import numpy as np
# 从本地工具模块导入EmbeddingFunc
from .utils import EmbeddingFunc

# 定义一个类型字典，用于规范化文本块的数据结构
TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {
        "tokens": int,  # 块中的token数量
        "content": str,  # 块的文本内容
        "full_doc_id": str,  # 所属完整文档的ID
        "chunk_order_index": int,  # 块在文档中的顺序索引
    },
)

# 定义一个泛型类型变量，用于KV存储
T = TypeVar("T")


@dataclass
class QueryParam:
    """
    一个数据类，用于封装RAG查询时的所有参数。
    """
    # 查询模式: "light"（轻量级）, "naive"（朴素）, "mini"（迷你RAG模式）
    mode: Literal["light", "naive", "mini"] = "mini"
    # 是否只返回构建好的上下文，而不进行最终的问答生成
    only_need_context: bool = False
    # 是否只返回构建好的提示词，而不调用LLM
    only_need_prompt: bool = False
    # 期望的响应类型或格式，例如 "一段话", "一个列表"
    response_type: str = "Multiple Paragraphs"
    # 是否以流式方式返回响应
    stream: bool = False
    # 检索时返回的top-k个项目数；在"local"模式下对应实体，在"global"模式下对应关系
    top_k: int = int(os.getenv("TOP_K", "60"))
    # # 检索的文档块数量（此行已注释掉）
    # top_n: int = 10
    # 文本单元（如原始块）允许的最大token数
    max_token_for_text_unit: int = 4000
    # 全局上下文（如关系描述）允许的最大token数
    max_token_for_global_context: int = 4000
    # 本地上下文（如实体描述）允许的最大token数
    max_token_for_local_context: int = 4000
    # （MiniRAG模式专用）节点上下文允许的最大token数，以防SLM（小语言模型）无法生成响应
    max_token_for_node_context: int = 500

    # 高级（概念性）关键词列表
    hl_keywords: list[str] = field(default_factory=list)
    # 低级（具体）关键词列表
    ll_keywords: list[str] = field(default_factory=list)
    
    # 对话历史支持
    conversation_history: list[dict] = field(
        default_factory=list
    )  # 格式: [{"role": "user/assistant", "content": "message"}]
    # 要考虑的完整对话轮数（用户-助手对）
    history_turns: int = (
        3
    )


@dataclass
class StorageNameSpace:
    """
    所有存储类的基类，提供命名空间和全局配置。
    """
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        """索引操作完成后的回调函数，例如用于提交存储事务。"""
        pass

    async def query_done_callback(self):
        """查询操作完成后的回调函数。"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    """
    向量存储的抽象基类。
    """
    embedding_func: EmbeddingFunc  # 嵌入函数实例
    meta_fields: set = field(default_factory=set)  # 存储的元数据字段集合

    async def query(self, query: str, top_k: int) -> list[dict]:
        """根据查询文本进行向量相似度搜索。"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """
        插入或更新数据。
        - 使用字典中的'content'字段进行嵌入。
        - 使用字典的键作为ID。
        - 如果embedding_func为None，则直接使用值中的'embedding'字段。
        """
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    """
    键值（KV）存储的抽象基类（支持泛型）。
    """
    embedding_func: EmbeddingFunc

    async def all_keys(self) -> list[str]:
        """获取所有键。"""
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        """根据ID获取单个值。"""
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        """根据ID列表批量获取值，可指定返回字段。"""
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """过滤出一个列表中不存在于存储中的键。"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        """插入或更新数据。"""
        raise NotImplementedError

    async def drop(self):
        """清空所有数据。"""
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    """
    图存储的抽象基类。
    """
    embedding_func: EmbeddingFunc = None

    @abstractmethod
    async def get_types(self) -> tuple[list[str], list[str]]:
        """获取图中所有实体和关系的类型。"""
        raise NotImplementedError

    async def has_node(self, node_id: str) -> bool:
        """检查节点是否存在。"""
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """检查边是否存在。"""
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        """获取节点的度（连接的边数）。"""
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """获取边的度（此处可能指权重或其他度量）。"""
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        """获取单个节点的数据。"""
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        """获取单个边的数据。"""
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        """获取一个节点所有的出边和入边。"""
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        """插入或更新一个节点。"""
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        """插入或更新一条边。"""
        raise NotImplementedError

    async def delete_node(self, node_id: str):
        """删除一个节点。"""
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        """对图中的节点进行嵌入（在minirag中未使用）。"""
        raise NotImplementedError("Node embedding is not used in minirag.")


class DocStatus(str, Enum):
    """文档处理状态的枚举类。"""

    PENDING = "pending"  # 待处理
    PROCESSING = "processing"  # 处理中
    PROCESSED = "processed"  # 已处理
    FAILED = "failed"  # 处理失败


@dataclass
class DocProcessingStatus:
    """文档处理状态的数据结构。"""

    content: str
    """文档原始内容"""
    content_summary: str
    """文档内容摘要（前100个字符），用于预览"""
    content_length: int
    """文档总长度"""
    status: DocStatus
    """当前处理状态"""
    created_at: str
    """文档创建时间的ISO格式时间戳"""
    updated_at: str
    """文档最后更新时间的ISO格式时间戳"""
    chunks_count: Optional[int] = None
    """分块后的块数量，用于处理过程"""
    error: Optional[str] = None
    """如果处理失败，记录错误信息"""
    metadata: dict[str, Any] = field(default_factory=dict)
    """附加的元数据"""


class DocStatusStorage(BaseKVStorage):
    """文档状态存储的基类。"""

    async def get_status_counts(self) -> dict[str, int]:
        """获取各种状态下的文档数量统计。"""
        raise NotImplementedError

    async def get_failed_docs(self) -> dict[str, DocProcessingStatus]:
        """获取所有处理失败的文档。"""
        raise NotImplementedError

    async def get_pending_docs(self) -> dict[str, DocProcessingStatus]:
        """获取所有待处理的文档。"""
        raise NotImplementedError
