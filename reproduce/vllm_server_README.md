# vLLM Server 部署指南

## 概述

`vllm_server.py` 用于启动 Qwen3-4B-Instruct-2507 模型的 OpenAI 兼容 API 服务器。

## 快速开始

### 1. 启动服务器

```bash
# 使用默认配置启动（GPU 0，端口 8000）
python reproduce/vllm_server.py

# 指定 GPU 和端口
python reproduce/vllm_server.py --gpu 0 --port 8000

# 使用多个 GPU（张量并行）
python reproduce/vllm_server.py --gpu 0,1 --tensor-parallel-size 2

# 完整参数示例
python reproduce/vllm_server.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu 0 \
    --max-model-len 8192 \
    --dtype auto \
    --trust-remote-code
```

### 2. 测试连接

```bash
# 使用示例脚本测试
python reproduce/vllm_server_example.py

# 自定义提示
python reproduce/vllm_server_example.py --prompt "解释一下什么是 RAG"
```

### 3. 使用 OpenAI Python 客户端

```python
from openai import OpenAI

# 连接到本地 vLLM 服务器
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM 不需要真实的 API key
)

# 调用模型
response = client.chat.completions.create(
    model="Qwen/Qwen3-4B-Instruct-2507",
    messages=[
        {"role": "user", "content": "你好，请介绍一下你自己。"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
```

## 参数说明

### vllm_server.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `Qwen/Qwen3-4B-Instruct-2507` | 模型名称或路径 |
| `--host` | str | `0.0.0.0` | 服务器监听地址 |
| `--port` | int | `8000` | 服务器监听端口 |
| `--gpu` | str | `0` | 使用的 GPU 设备（如 `0` 或 `0,1`） |
| `--max-model-len` | int | `8192` | 最大模型长度（上下文窗口） |
| `--tensor-parallel-size` | int | `1` | 张量并行大小（多 GPU 时使用） |
| `--trust-remote-code` | flag | `False` | 信任远程代码（某些模型需要） |
| `--dtype` | str | `auto` | 模型数据类型（auto/float16/bfloat16/float32） |
| `--api-key` | str | `None` | API 密钥（可选，用于保护 API 访问） |
| `--log-level` | str | `info` | 日志级别（debug/info/warning/error） |

## API 端点

启动服务器后，可以通过以下端点访问：

- **OpenAI 兼容 API**: `http://localhost:8000/v1`
- **API 文档**: `http://localhost:8000/docs`
- **健康检查**: `http://localhost:8000/health`

## 使用场景

### 1. 与 MiniRAG 集成

可以在 MiniRAG 中使用 OpenAI 客户端连接到 vLLM 服务器：

```python
from openai import AsyncOpenAI
from minirag.llm.openai import openai_complete_if_cache

# 创建连接到 vLLM 服务器的客户端
vllm_client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# 在 MiniRAG 中使用
rag = MiniRAG(
    llm_model_func=lambda prompt, **kwargs: openai_complete_if_cache(
        model="Qwen/Qwen3-4B-Instruct-2507",
        prompt=prompt,
        base_url="http://localhost:8000/v1",
        **kwargs
    ),
    # ... 其他配置
)
```

### 2. 多客户端共享

多个客户端可以同时连接到同一个 vLLM 服务器，实现模型共享和资源复用。

### 3. 生产环境部署

- 使用 `--api-key` 保护 API 访问
- 使用反向代理（如 Nginx）进行负载均衡
- 使用 `--tensor-parallel-size` 充分利用多 GPU

## 注意事项

1. **显存要求**: Qwen3-4B-Instruct-2507 需要约 8-10GB 显存（取决于配置）
2. **端口冲突**: 确保端口 8000 未被占用，或使用 `--port` 指定其他端口
3. **GPU 选择**: 使用 `--gpu` 指定要使用的 GPU，避免与其他进程冲突
4. **模型加载**: 首次启动时模型加载可能需要几分钟时间

## 故障排除

### 问题：无法连接到服务器

- 检查服务器是否正在运行
- 检查端口是否被占用：`netstat -tuln | grep 8000`
- 检查防火墙设置

### 问题：显存不足

- 减小 `--max-model-len`
- 使用 `--dtype float16` 或 `--dtype bfloat16`
- 使用量化模型

### 问题：模型加载失败

- 检查模型路径是否正确
- 尝试添加 `--trust-remote-code` 参数
- 检查网络连接（如果从 HuggingFace 下载）

## 性能优化

1. **多 GPU 并行**: 使用 `--tensor-parallel-size` 充分利用多 GPU
2. **批处理**: vLLM 自动批处理请求，提高吞吐量
3. **PagedAttention**: vLLM 自动使用 PagedAttention 优化显存使用

