# MiniRAG 前端使用指南

## 概述

本前端是一个完整的 MiniRAG 系统可视化界面，直接对接 MiniRAG 后端的所有 API 端点，提供以下功能：

- **问答对话**：向知识图谱提问并获得 RAG 增强的回答
- **图谱可视化**：可视化知识图谱的节点和关系
- **文档管理**：上传、索引和管理文档
- **系统监控**：查看后端配置和运行状态
- **实体管理**：浏览和搜索知识图谱中的实体

---

## 快速开始

### 1. 启动 vLLM 模型服务（第一步）

MiniRAG 后端需要一个 LLM 模型服务来处理推理请求。推荐使用 vLLM 部署本地模型，它提供 OpenAI 兼容的 API 接口。

```bash
cd d:\pywork\MiniRAG1333213213

# 安装 vLLM（首次使用）
pip install vllm

# 启动 vLLM 服务器（默认使用 GPU 0，端口 8000）
python reproduce/vllm_server.py

# 指定模型和参数
python reproduce/vllm_server.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --gpu 0 \
    --port 8000 \
    --max-model-len 8192

# 多 GPU 部署
python reproduce/vllm_server.py --gpu 0,1 --tensor-parallel-size 2
```

启动成功后，vLLM 会在 `http://localhost:8000/v1` 提供 OpenAI 兼容 API。
可以通过 `http://localhost:8000/docs` 查看 API 文档。

> **验证 vLLM 是否就绪：**
> ```bash
> curl http://localhost:8000/v1/models
> ```
> 应返回可用模型列表。

### 2. 启动 MiniRAG 后端服务（第二步）

在新的终端窗口中启动 MiniRAG API 服务器，让它对接 vLLM：

```bash
cd d:\pywork\MiniRAG1333213213

# 安装依赖（首次使用）
pip install -r requirements.txt

# 推荐方式：使用 vLLM 作为 LLM 后端 + Ollama 嵌入模型
python -m minirag.api.minirag_server \
    --llm-binding openai \
    --llm-binding-host http://localhost:8000/v1 \
    --llm-binding-api-key dummy \
    --llm-model Qwen/Qwen3-4B-Instruct-2507 \
    --embedding-binding ollama \
    --embedding-model bge-m3:latest \
    --port 9721
```

**常用启动配置参考：**

```bash
# 配置 1：vLLM (LLM) + Ollama (Embedding)
# 前提：vLLM 运行在 8000 端口，Ollama 运行在 11434 端口
python -m minirag.api.minirag_server \
    --llm-binding openai \
    --llm-binding-host http://localhost:8000/v1 \
    --llm-binding-api-key dummy \
    --llm-model Qwen/Qwen3-4B-Instruct-2507 \
    --embedding-binding ollama \
    --embedding-binding-host http://localhost:11434 \
    --embedding-model bge-m3:latest \
    --port 9721

# 配置 2：vLLM 提供 LLM 和 Embedding（如果模型支持）
python -m minirag.api.minirag_server \
    --llm-binding openai \
    --llm-binding-host http://localhost:8000/v1 \
    --llm-binding-api-key dummy \
    --llm-model Qwen/Qwen3-4B-Instruct-2507 \
    --embedding-binding openai \
    --embedding-binding-host http://localhost:8000/v1 \
    --embedding-binding-api-key dummy \
    --embedding-model Qwen/Qwen3-4B-Instruct-2507 \
    --port 9721

# 配置 3：全部使用 Ollama（无需 vLLM）
python -m minirag.api.minirag_server \
    --llm-binding ollama \
    --llm-model qwen3:4b \
    --embedding-binding ollama \
    --embedding-model bge-m3:latest \
    --port 9721

# 配置 4：使用环境变量（参考 minirag/api/.env.aoi.example）
cp minirag/api/.env.aoi.example minirag/api/.env
# 编辑 .env 文件，设置以下变量：
#   LLM_BINDING=openai
#   LLM_BINDING_HOST=http://localhost:8000/v1
#   LLM_BINDING_API_KEY=dummy
#   LLM_MODEL=Qwen/Qwen3-4B-Instruct-2507
#   EMBEDDING_BINDING=ollama
#   EMBEDDING_MODEL=bge-m3:latest
python -m minirag.api.minirag_server
```

**MiniRAG 关键启动参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--llm-binding` | `ollama` | LLM 绑定类型：`ollama` / `openai` / `lollms` / `azure_openai` |
| `--llm-binding-host` | 自动 | LLM 服务地址（vLLM 填 `http://localhost:8000/v1`） |
| `--llm-binding-api-key` | 无 | LLM API Key（vLLM 填 `dummy` 即可） |
| `--llm-model` | `mistral-nemo:latest` | 模型名称（需与 vLLM 启动时的模型名一致） |
| `--embedding-binding` | `ollama` | 嵌入模型绑定类型 |
| `--embedding-model` | `bge-m3:latest` | 嵌入模型名称 |
| `--port` | `9721` | MiniRAG API 端口 |
| `--working-dir` | `./rag_storage` | RAG 索引存储目录 |
| `--input-dir` | `./inputs` | 文档输入目录（扫描目录功能使用） |
| `--max-tokens` | `32768` | 最大 token 数量 |
| `--chunk_size` | `1200` | 文档分块大小 |
| `--chunk_overlap_size` | `100` | 分块重叠大小 |
| `--key` | 无 | MiniRAG API Key（保护前端访问，可选） |

后端默认监听 `http://0.0.0.0:9721`。

> **完整启动流程总结：**
> 1. 终端 1：`python reproduce/vllm_server.py` → 等待模型加载完成（显示 API 就绪）
> 2. 终端 2：`python -m minirag.api.minirag_server --llm-binding openai --llm-binding-host http://localhost:8000/v1 ...`
> 3. 浏览器打开 `front/index.html` → 点击"测试连接"

### 2. 打开前端页面

**方式一：直接用浏览器打开**

```
直接双击打开 front/index.html
```

**方式二：使用 VS Code Live Server（推荐）**

1. 安装 VS Code 的 "Live Server" 扩展
2. 右键点击 `front/index.html` → "Open with Live Server"

**方式三：使用 Python 简易服务器**

```bash
cd d:\pywork\MiniRAG1333213213\front
python -m http.server 8080
```
然后访问 `http://localhost:8080`

### 3. 连接后端

1. 在页面左上角的 **"后端连接"** 区域，确认服务器地址为 `http://localhost:9721`
2. 如果后端配置了 API Key，在导航栏右侧的 "API Key" 输入框中填入
3. 点击 **"测试连接"** 按钮
4. 状态栏显示 "● 已连接" 即表示成功

---

## 功能说明

### 问答对话

1. 确保已连接后端且已上传文档
2. 在左侧可选择 **检索模式**：
   - `light`：路径检索，轻量高效
   - `naive`：传统 RAG 检索
   - `mini`：轻量检索模式
3. 在底部输入框输入问题，按回车或点击发送
4. 系统会调用 `POST /query` API，返回 RAG 增强的回答
5. 回答支持 Markdown 格式渲染

### 文档管理

**上传文件：**
1. 在左侧 "数据管理" 区域，点击或拖拽文件到上传区域
2. 支持格式：`.txt`, `.pdf`, `.md`, `.docx`, `.pptx`
3. 点击 "上传并索引" 按钮
4. 文件会通过 `POST /documents/upload` API 上传到后端并自动索引

**扫描目录：**
- 点击 "扫描输入目录"，后端会扫描 `--input-dir` 目录下的新文件并索引
- 调用 `POST /documents/scan` API

**清空数据：**
- 点击 "清空数据" 会调用 `DELETE /documents` 清除所有已索引文档

### 图谱可视化

1. 切换到 "图谱可视化" 标签页
2. 点击 "刷新标签" 获取所有图谱标签（调用 `GET /graph/label/list`）
3. 从下拉菜单选择一个标签
4. 系统调用 `GET /graphs?label=xxx` 获取图谱数据
5. 使用 Cytoscape.js 渲染知识图谱
6. 点击节点可查看实体信息
7. 支持拖拽、缩放和重新布局

### 系统监控

切换到 "系统监控" 标签页可查看：
- 系统健康状态
- 已索引文件数
- LLM 模型和嵌入模型配置
- 存储后端配置
- 完整配置参数列表
- 本次会话的查询日志

### 实体管理

1. 切换到 "实体管理" 标签页
2. 选择图谱标签加载实体数据
3. 在搜索框中过滤实体
4. 点击实体查看详情（描述、关联关系、原始属性）

---

## 后端 API 端点对照表

| 前端功能 | HTTP 方法 | API 端点 | 说明 |
|---------|----------|---------|------|
| 测试连接 | GET | `/health` | 返回系统状态和配置 |
| 问答查询 | POST | `/query` | 发送查询并获取 RAG 回答 |
| 上传文件 | POST | `/documents/upload` | 上传文件到输入目录并索引 |
| 扫描目录 | POST | `/documents/scan` | 扫描输入目录索引新文件 |
| 文档列表 | GET | `/health` | 从 health 中提取 indexed_files |
| 清空文档 | DELETE | `/documents` | 清除所有索引数据 |
| 图谱标签 | GET | `/graph/label/list` | 获取所有图谱标签 |
| 图谱数据 | GET | `/graphs?label=xxx` | 获取指定标签的图谱数据 |

---

## 架构说明

```
前端 (front/index.html)          后端 (minirag/api/minirag_server.py)
┌─────────────────────┐          ┌─────────────────────────────┐
│  浏览器 (纯静态 HTML) │  HTTP   │  FastAPI (默认端口 9721)     │
│                     │ ◄─────► │                             │
│  - Tailwind CSS     │  CORS   │  - CORS 已配置              │
│  - Cytoscape.js     │  JSON   │  - 支持 X-API-Key 认证      │
│  - Chart.js         │         │  - MiniRAG 核心引擎          │
│  - Marked.js        │         │  - 多种 LLM/Embedding 绑定  │
└─────────────────────┘         └─────────────────────────────┘
```

**关键设计决策：**

1. **纯静态部署**：前端是一个独立的 HTML 文件，不依赖任何构建工具，可直接用浏览器打开
2. **CORS 已内置**：后端已配置 `allow_origins=["*"]` 的 CORS 中间件，支持跨域请求
3. **API Key 可选**：如果后端未设置 API Key，前端无需配置；如果设置了，通过 `X-API-Key` 请求头传递
4. **配置持久化**：服务器地址和 API Key 保存在浏览器 localStorage 中

---

## 常见问题

### Q: 连接失败怎么办？
1. 确认后端已启动：终端中应显示 "Server is ready to accept connections!"
2. 确认端口正确（默认 9721）
3. 如果后端在远程机器上，确保防火墙允许该端口
4. 打开浏览器控制台 (F12) 查看具体错误信息

### Q: 上传文件后查询没有结果？
1. 上传文件后需要等待索引完成
2. 检查后端终端日志确认索引是否成功
3. 确认文件格式支持（.txt, .pdf, .md, .docx, .pptx）

### Q: 图谱可视化为空？
1. 必须先上传文档并成功索引
2. 索引过程中会自动构建知识图谱
3. 点击 "刷新标签" 获取可用的图谱标签
4. 如果没有标签，说明图谱尚未构建完成

### Q: 如何修改后端端口？
```bash
python -m minirag.api.minirag_server --port 8080
```
然后在前端修改服务器地址为 `http://localhost:8080`
