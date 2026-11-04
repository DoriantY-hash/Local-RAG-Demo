# 本地化中文 RAG 知识库问答系统 (Localized Chinese RAG System)

这是一个基于 **LlamaIndex** 构建的端到端检索增强生成（RAG）最小闭环演示项目。项目采用 **FastAPI + Gradio** 提供 Web 交互，所有模型（Embedding、Reranker、LLM）均在本地运行，无需依赖外部 API，确保数据隐私安全。

## 🚀 核心特性

- **全本地化部署**：使用本地 GTE 嵌入模型、本地 bge-reranker 重排模型和本地 Qwen2.5-3B 量化模型，数据不出内网。

- **两阶段检索架构**：向量相似度粗排（Top 20） + Cross-Encoder 精排（Top k），大幅提升召回准确率。

- **防幻觉与强制截断**：针对小模型容易出现的“续写复读”问题，在 Prompt 指令和流式输出代码层进行双重拦截，确保回答严格基于文档。

- **支持多种文档格式**：内置 `docx2txt` 和 `openpyxl`，支持 `.docx`、`.txt`、`.xlsx` 等格式的文档解析与切块。

- **动态配置管理**：所有模型路径、数据路径和参数均集中抽离至 `config.py`，消除硬编码，实现开箱即用。

## 🛠️ 技术栈

- **RAG 编排框架**：LlamaIndex (v0.10.67)

- **Web 框架**：FastAPI (v0.112.4) + Gradio (v4.44.1)

- **Embedding 模型**：GTE (nlp_gte_sentence-embedding_chinese-base, 768 维)

- **Reranker 模型**：bge-reranker-base

- **LLM**：Qwen2.5-3B-Instruct (GGUF, Q4_K_M 量化版)

- **LLM 推理引擎**：llama-cpp-python (v0.2.71)

## 📁 项目结构

```text
localRAG/
├── EmbeddingModel      # 存放嵌入模型的文件夹
├── File                # 文件被上传至的文件夹
├── llama.cpp           # 存放 LLM 模型的文件夹
├── RerankerModel       # 存放重排模型的文件夹
├── test_files          # 文件初始文件夹
├── VectorStore         # 存放生成的知识库的文件夹
├── config.py           # 全局配置（路径、常量）
├── main.py             # FastAPI 入口，挂载 Gradio 子应用
├── create_kb.py        # 知识库创建、增量更新、文档切块与入库
├── chat.py             # RAG 检索链路、Reranker 重排、LLM 生成与截断
├── upload_file.py      # 文件上传与类目管理
├── html_string.py      # Web 页面 HTML/CSS 模板
├── requirements.txt    # 依赖清单
└── README.md           # 项目说明
```

## 💻 环境准备

**推荐环境**：Windows，Python 3.11.9。

由于模型文件较大，项目不直接包含模型文件。请先安装模型下载工具 `modelscope`，并在项目根目录执行下载命令：

在 CMD 的项目根目录激活 Python 环境之后：

### 0. 更新 pip 至最新版本：

```bash
python -m pip install --upgrade pip
```

### 1. 安装 modelscope

```bash
pip install modelscope
```

### 1. 下载 GTE 嵌入模型

```bash
modelscope download --model iic/nlp_gte_sentence-embedding_chinese-base --local_dir ./EmbeddingModel/hub/iic/nlp_gte_sentence-embedding_chinese-base
```

### 1. 下载 bge-reranker-base 重排模型

```bash
modelscope download --model BAAI/bge-reranker-base --local_dir ./RerankerModel/bge-reranker-base
```

### 1. 下载 Qwen2.5-3B GGUF 模型（约 2GB）

```bash
modelscope download --model Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local_dir ./llama.cpp
```

## ⚙️ 安装与运行

### 1. 安装依赖

#### 1. 建议先安装 CPU 版本的 PyTorch：

```bash
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 2. Windows 用户请优先使用预编译 wheel 安装 llama-cpp：

```bash
pip install llama-cpp-python==0.2.71 --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

#### 3. 最后安装剩余依赖：

```bash
pip install -r requirements.txt
```



### 2. 启动服务

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

启动过程会加载本地 LLM，请耐心等待约 20-60 秒。

在浏览器访问 http://127.0.0.1:8000

### 3. 上传数据

点击首页的 **“上传数据”** 卡片，选择非结构化数据/结构化数据，上传文件，输入类目名称，点击“新建类目”。

### 4. 创建知识库

点击 **“创建/更新知识库”** 卡片，选择非结构化知识库/结构化知识库，输入知识库名称（如 `demo_kb`），选择文件类目，选择操作类型，点击“执行创建/增量更新”。

系统会自动将文档切块（`Chunk Size: 512`, `Overlap: 50`），生成向量并持久化到 VectorStore/ 目录。

### 5. 进行 RAG 问答

点击 **“RAG问答”** 卡片，在右侧“加载知识库”中选择刚才创建的库，在下方输入框提问，即可观察到“召回文本段”与模型的流式回答（可调整右侧参数设置）。
