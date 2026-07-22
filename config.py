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
GPU_DEFAULT_CTX = 32768
GPU_DEFAULT_LAYERS = 0  # -1 = 全部
GPU_DEFAULT_MAX_TOKENS = 4096

# ─── 性能优化 (llama-cpp-python >= 0.3.0) ─────────────────────────────────
GPU_N_BATCH = int(os.environ.get("GPU_N_BATCH", 512))    # 批量 prompt 处理
GPU_N_UBATCH = int(os.environ.get("GPU_N_UBATCH", 512))   # 微批量大小
GPU_FLASH_ATTN = os.environ.get("GPU_FLASH_ATTN", "1") == "1"  # Flash Attention
GPU_KV_CACHE_DTYPE = os.environ.get("GPU_KV_CACHE_DTYPE", "")  # q8_0/q4_0/f16 等
GPU_OFFLOAD_KQV = os.environ.get("GPU_OFFLOAD_KQV", "1") == "1"  # 把 KQV 也 offload 到 GPU
GPU_CHAT_FORMAT = os.environ.get("GPU_CHAT_FORMAT", "")  # 空=自动从 GGUF 元数据检测

# ─── 上下文扩展 (RoPE Scaling) ──────────────────────────────────────────────
# 本地模型默认 8K，可通过 RoPE 扩展到更长上下文
# 支持的 scaling 类型: "none", "linear", "yarn"
GPU_ROPE_SCALING = os.environ.get("GPU_ROPE_SCALING", "yarn")  # none|linear|yarn
GPU_ROPE_FREQ_BASE = float(os.environ.get("GPU_ROPE_FREQ_BASE", 0))  # 0=自动
GPU_ROPE_SCALE = float(os.environ.get("GPU_ROPE_SCALE", 4.0))  # 扩展倍数: 8K→32K=4.0
# 模型家族对应的推荐 RoPE freq_base
FAMILY_ROPE_BASE = {
    "gemma4": 10000.0,  # Gemma-4 默认 10K
    "gemma3": 10000.0,
    "qwen35": 1000000.0,  # Qwen3.5 默认 1M
    "qwen3": 1000000.0,  # Qwen3 默认 1M
    "qwen25": 1000000.0,  # Qwen2.5 默认 1M
    "llama4": 500000.0,  # LLaMA 4
    "minicpm": 1000000.0,  # MiniCPM R1-thinking 默认 1M
}

# ─── 推理参数默认值 ──────────────────────────────────────────────────────────
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_REPEAT_PENALTY = 1.0

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
