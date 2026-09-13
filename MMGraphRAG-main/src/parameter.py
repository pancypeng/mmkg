"""
多模态知识图谱构建参数配置

该模块包含多模态知识图谱构建的参数配置

目录结构：
- CACHE_PATH: LLM响应缓存、临时文件 (默认: cache/)
- WORKING_DIR: 中间处理文件（chunks、graphs、images）(默认: working/)
- OUTPUT_DIR: 最终MMKG图谱文件 (默认: output/)
"""
from sentence_transformers import SentenceTransformer

# ============ 模型配置 ============
# API Configuration
API_KEY = "Looking for the API key? So am I."
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3-max"

MM_API_KEY = "If you find it, please don’t tell me. That's safer for both of us."
MM_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MM_MODEL_NAME = "qwen-vl-max"

# embedding模型加载
EMBEDDING_MODEL_DIR = './models/all-MiniLM-L6-v2'

EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL_DIR, device="cpu")
# 其他加载方式示例：
# 1. 使用 CUDA 加速：
# EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL_DIR, device="cuda")
# 2. 直接使用模型名称（会自动下载）：
# EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

# ============ 目录配置 ============
# PDF文件路径
INPUT_PDF_PATH = r'examples/example_input/2020.acl-main.45.pdf'

# 缓存路径（LLM响应缓存、临时文件）
CACHE_PATH = 'examples/cache'

# 工作目录（中间处理文件：chunks、graphs、images）
WORKING_DIR = 'examples/example_working'

# 输出目录（最终MMKG图谱文件）
OUTPUT_DIR = 'examples/example_output'

# 知识图谱输出名称（可选，不指定时由 builder.py 自动生成带时间戳的名称）
MMKG_NAME = 'example_mmkg'

# ============ 参数配置 ============
# 实体提取与处理配置
ENTITY_EXTRACT_MAX_GLEANING = 0 # 此变量只对文本实体有效
ENTITY_SUMMARY_MAX_TOKENS = 500
SUMMARY_CONTEXT_MAX_TOKENS = 10000

# 预处理相关配置
USE_MINERU = True

# ============ RAG 检索生成配置 ============
class QueryParam:
    top_k = 5
    response_type = "Detailed System-like Response"
    # 本地上下文最大 Token 数
    local_max_token_for_local_context = 4000
    # 多模态实体数量
    number_of_mmentities = 3
    # 文本单元最大 Token 数
    local_max_token_for_text_unit = 4000

# 检索阈值
RETRIEVAL_THRESHOLD = 0.2