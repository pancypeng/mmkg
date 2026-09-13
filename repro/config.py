"""复现实验的集中配置。

原仓库把 API key / 模型 / 路径散落在 pull_llm.py 和 fusion_research.py 里，
且 embedding 模型路径硬编码到作者机器的 cuda:0。这里统一收口。
"""
import os

# 本机 SSL_CERT_FILE 指向已卸载的 miniconda，会让 huggingface/openai 的
# httpx client 在初始化时就 FileNotFoundError。指回 certifi。
_cert = os.environ.get("SSL_CERT_FILE")
if _cert and not os.path.exists(_cert):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        os.environ.pop("SSL_CERT_FILE", None)

# ---------------------------------------------------------------- 路径
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MMKG_ROOT = os.path.dirname(REPO_ROOT)

CMEL_REPO = os.path.join(MMKG_ROOT, "CMEL-dataset-main")
CMEL_RESEARCH_UPSTREAM = os.path.join(CMEL_REPO, "cmel_research")
# 上游保持原样；跑的是打过补丁的 vendor 副本（补丁见 repro/patches/）
CMEL_RESEARCH = os.path.join(REPO_ROOT, "vendor", "cmel_research")
DATASET_DIR = os.path.join(CMEL_REPO, "CMEL_dataset")

RUNS_DIR = os.path.join(REPO_ROOT, "runs")
CACHE_DIR = os.path.join(REPO_ROOT, "cache")

# 数据集里 image_path 存的是作者机器上的相对前缀，需要重写成本机真实路径
DATASET_PATH_PREFIX = "./fusion_research/fusion_dataset"

CATEGORIES = ["news", "novel", "paper"]

# 任何会改变模型输出解释或结果语义的代码改动都递增此值，并写进 run_config。
PIPELINE_SCHEMA_VERSION = 4

# ---------------------------------------------------------------- 模型
# 正式复现默认使用论文同款 Stella；也可通过环境变量切换做消融。
EMBED_MODEL_NAME = os.environ.get("CMEL_EMBED_MODEL", "/root/models/stella_en_1.5B_v5")
EMBED_DEVICE = os.environ.get("CMEL_EMBED_DEVICE", "cpu")

# 论文用 Qwen2.5-72B-Instruct（文本）+ InternVL2.5-38B-MPO（多模态）
# 通过环境变量注入，避免把 key 写进文件
API_KEY = os.environ.get("CMEL_API_KEY", "")
API_BASE = os.environ.get("CMEL_API_BASE", "")
MODEL = os.environ.get("CMEL_MODEL", "")

MM_API_KEY = os.environ.get("CMEL_MM_API_KEY", API_KEY)
MM_API_BASE = os.environ.get("CMEL_MM_API_BASE", API_BASE)
MM_MODEL = os.environ.get("CMEL_MM_MODEL", "")

# ---------------------------------------------------------------- 方法超参
EMBEDDING_THRESHOLD = float(os.environ.get("CMEL_EMBEDDING_THRESHOLD", "0.35"))

LLM_MAX_RETRIES = int(os.environ.get("CMEL_LLM_MAX_RETRIES", "4"))
LLM_TIMEOUT = float(os.environ.get("CMEL_LLM_TIMEOUT", "180"))

# 生成参数也属于实验配置，必须进入缓存键和 run_config，防止模型或推理
# 方式改变后误命中旧缓存。Qwen3.8 默认会思考；结构化抽取默认关闭思考。
LLM_TEMPERATURE = float(os.environ.get("CMEL_LLM_TEMPERATURE", "0"))
LLM_TOP_P = float(os.environ.get("CMEL_LLM_TOP_P", "1"))
LLM_TOP_K = int(os.environ.get("CMEL_LLM_TOP_K", "-1"))
LLM_PRESENCE_PENALTY = float(os.environ.get("CMEL_LLM_PRESENCE_PENALTY", "0"))
LLM_MAX_TOKENS = int(os.environ.get("CMEL_LLM_MAX_TOKENS", "2048"))
LLM_SEED = int(os.environ.get("CMEL_LLM_SEED", "0"))
LLM_ENABLE_THINKING = os.environ.get("CMEL_ENABLE_THINKING", "0").lower() in {
    "1", "true", "yes", "on"
}
LLM_REASONING_EFFORT = os.environ.get("CMEL_REASONING_EFFORT", "")
CACHE_NAMESPACE = os.environ.get("CMEL_CACHE_NAMESPACE", "cmel-v2")
TEXT_CACHE_NAMESPACE = os.environ.get("CMEL_TEXT_CACHE_NAMESPACE", CACHE_NAMESPACE)
MM_CACHE_NAMESPACE = os.environ.get("CMEL_MM_CACHE_NAMESPACE", CACHE_NAMESPACE)
# paper_v1 保留上游提示词；strict_v2 只收紧最终细粒度融合，不改变候选聚类。
ALIGNMENT_PROMPT_VERSION = os.environ.get("CMEL_ALIGNMENT_PROMPT_VERSION", "paper_v1")
DENSE_CANDIDATE_TOP_K = int(os.environ.get("CMEL_DENSE_CANDIDATE_TOP_K", "5"))


def generation_config():
    return {
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "top_k": LLM_TOP_K,
        "presence_penalty": LLM_PRESENCE_PENALTY,
        "max_tokens": LLM_MAX_TOKENS,
        "seed": LLM_SEED,
        "enable_thinking": LLM_ENABLE_THINKING,
        "reasoning_effort": LLM_REASONING_EFFORT,
    }


def api_configured():
    return bool(API_KEY and API_BASE and MODEL and MM_MODEL)
