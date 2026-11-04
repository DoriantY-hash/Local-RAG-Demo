import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.resolve()

# 1. 模型存放路径
MODELSCOPE_CACHE_DIR = str(BASE_DIR / "EmbeddingModel")
EMBED_MODEL_PATH = str(BASE_DIR / "EmbeddingModel" / "hub" / "iic" / "nlp_gte_sentence-embedding_chinese-base")
RERANKER_MODEL_PATH = str(BASE_DIR / "RerankerModel" / "bge-reranker-base")

# 2. LLM 模型路径（推荐把 llama.cpp 文件夹移到项目内）
LLM_MODEL_PATH = str(BASE_DIR / "llama.cpp" / "qwen2.5-3b-instruct-q4_k_m.gguf")

# 3. 数据与存储路径
DB_PATH = str(BASE_DIR / "VectorStore")
UNSTRUCTURED_FILE_PATH = str(BASE_DIR / "File" / "Unstructured")
STRUCTURED_FILE_PATH = str(BASE_DIR / "File" / "Structured")
FILE_ROOT_PATH = str(BASE_DIR / "File")  # 指向项目的 File 根目录
TMP_NAME = "tmp_abcd"