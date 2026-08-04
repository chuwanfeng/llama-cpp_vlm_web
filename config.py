"""
配置常量
"""

import os

# ─── 基础路径 ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR  # 项目根目录别名，供各模块统一使用
# MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
MODELS_DIR = r"D:\Scoop\LLM"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── GPU (llama-cpp-python) ──────────────────────────────────────────────────
GPU_DEFAULT_LAYERS = 0  # -1 = 全部
GPU_DEFAULT_MAX_TOKENS = 4096

# ─── 性能优化 (llama-cpp-python >= 0.3.0) ─────────────────────────────────
GPU_N_BATCH = int(os.environ.get("GPU_N_BATCH", 512))    # 批量 prompt 处理
GPU_N_UBATCH = int(os.environ.get("GPU_N_UBATCH", 512))   # 微批量大小
GPU_FLASH_ATTN = os.environ.get("GPU_FLASH_ATTN", "1") == "1"  # Flash Attention
GPU_KV_CACHE_DTYPE = os.environ.get("GPU_KV_CACHE_DTYPE", "")  # q8_0/q4_0/f16 等
GPU_OFFLOAD_KQV = os.environ.get("GPU_OFFLOAD_KQV", "1") == "1"  # 把 KQV 也 offload 到 GPU

# 按模型家族的上下文窗口（n_ctx），全部 ≤ 原生上下文
# 全部模型原生 ≥128K，无需 RoPE 扩展
FAMILY_CONTEXT = {
    "minicpm": 65536,   # 原生 131K，64K 足够 1B 模型
    "gemma4":  65536,   # 原生 128K
    "gemma3":  65536,   # 原生 128K
    "qwen35":  65536,   # 原生 128K
    "qwen3":   65536,   # 原生 128K
    "qwen25":  65536,   # 原生 128K
    "llama4":  65536,   # 原生 128K+
    "default": 32768,   # 未知模型用保守值
}
GPU_CHAT_FORMAT = os.environ.get("GPU_CHAT_FORMAT", "")  # 空=自动从 GGUF 元数据检测


# ─── 推理参数默认值 ──────────────────────────────────────────────────────────
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 30
DEFAULT_MIN_P = 0.05
DEFAULT_TYPICAL_P = 1.0
DEFAULT_PRESENCE_PENALTY = 1.0
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_REPEAT_PENALTY = 1.0
DEFAULT_MIROSTAT_MODE = 0
DEFAULT_MIROSTAT_ETA = 0.1
DEFAULT_MIROSTAT_TAU = 5.0

# 按模型家族的推荐采样参数（作为默认值，可被前端/API 参数覆盖）
# 来源：各模型官方 generation_config.json / README
FAMILY_SAMPLING_DEFAULTS = {
    "default": {
        "temperature": 0.8, "top_p": 0.9, "top_k": 30, "min_p": 0.05,
        "typical_p": 1.0, "presence_penalty": 1.0, "frequency_penalty": 0.0,
        "repeat_penalty": 1.0, "mirostat_mode": 0, "mirostat_eta": 0.1, "mirostat_tau": 5.0,
        "max_tokens": 4096,
    },
    "minicpm": {
        "temperature": 0.7, "top_p": 0.95, "top_k": 30, "min_p": 0.0,
        "typical_p": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "repeat_penalty": 1.0, "mirostat_mode": 0, "mirostat_eta": 0.1, "mirostat_tau": 5.0,
        "max_tokens": 2048,  # 官方 256，给 1B 留冗余
    },
    "gemma4": {
        "temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0,
        "typical_p": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "repeat_penalty": 1.0, "mirostat_mode": 0, "mirostat_eta": 0.1, "mirostat_tau": 5.0,
        "max_tokens": 4096,  # Gemma4 E2B 官方 generation_config.json
    },
    "qwen35": {
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
        "typical_p": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "repeat_penalty": 1.0, "mirostat_mode": 0, "mirostat_eta": 0.1, "mirostat_tau": 5.0,
        "max_tokens": 4096,  # Qwen3.5/Qwen3.6 官方推荐
    },
}

# ─── 厂商 API ───────────────────────────────────────────────────────────────
# 各厂商 API key 通过环境变量配置（参照 api_backends.py 中的 VENDORS 定义）:
#   OPENAI_API_KEY     — OpenAI
#   DEEPSEEK_API_KEY   — DeepSeek
#   ANTHROPIC_API_KEY  — Anthropic Claude
#   GOOGLE_API_KEY     — Google Gemini
#   DASHSCOPE_API_KEY  — 通义千问
#   ZHIPUAI_API_KEY    — 智谱 AI
#   MOONSHOT_API_KEY   — Moonshot / Kimi
VENDOR_TIMEOUT = int(os.environ.get("VENDOR_TIMEOUT", 120))

# ─── Web 服务 ────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))
DEBUG = False
