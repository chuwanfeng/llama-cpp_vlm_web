"""
配置常量
"""
import os

# ─── 基础路径 ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
MODELS_DIR = f"D:\Scoop\LLM"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_TIMEOUT = 120  # 秒
OLLAMA_STREAM_TIMEOUT = 300  # 秒

# ─── GPU (llama-cpp-python) ──────────────────────────────────────────────────
GPU_DEFAULT_CTX = 8192
GPU_DEFAULT_LAYERS = 0  # -1 = 全部
GPU_DEFAULT_MAX_TOKENS = 4096

# ─── 推理参数默认值 ──────────────────────────────────────────────────────────
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_REPEAT_PENALTY = 1.0

# ─── Web 服务 ────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))
DEBUG = True
