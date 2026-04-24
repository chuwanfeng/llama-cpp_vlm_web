"""
GPU 后端 — llama-cpp-python 原生 API
"""
import os
import gc
import json
import base64
import io
import logging
import traceback
from threading import Lock

log = logging.getLogger(__name__)

import torch
import numpy as np
from PIL import Image as PILImage

from config import (
    MODELS_DIR, GPU_DEFAULT_CTX, GPU_DEFAULT_LAYERS,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K,
    DEFAULT_REPEAT_PENALTY, GPU_DEFAULT_MAX_TOKENS,
)

# ─── 模块状态 ────────────────────────────────────────────────────────────────
LLAMA_AVAILABLE = False  # llama-cpp-python 是否可用（GPU 或 CPU）
HAVE_GPU = False          # 是否有 GPU
LLAMA_CPP = None

def detect_backend():
    """检测 llama-cpp-python 和 CUDA 可用性"""
    global LLAMA_AVAILABLE, HAVE_GPU, LLAMA_CPP
    try:
        import llama_cpp
        LLAMA_CPP = llama_cpp
        LLAMA_AVAILABLE = True
        print("[llama] llama-cpp-python 可用")
        
        # 检测 CUDA
        if torch.cuda.is_available():
            HAVE_GPU = True
            print(f"[llama] CUDA 可用: {torch.cuda.get_device_name(0)}")
        else:
            HAVE_GPU = False
            print("[llama] CUDA 不可用，将使用 CPU 模式")
        return True
    except ImportError as e:
        print(f"[llama] llama-cpp-python 未安装: {e}")
        return False
    except Exception as e:
        print(f"[llama] 检测失败: {e}")
        return False

def detect_gpu():
    """向后兼容：返回 GPU 是否可用"""
    global HAVE_GPU
    detect_backend()
    return HAVE_GPU


# ─── 模型管理 ────────────────────────────────────────────────────────────────
_model = None
_config = {}
_lock = Lock()


def _find_mmproj(model_path):
    """自动查找模型同目录下的 mmproj 文件"""
    model_dir = os.path.dirname(model_path)
    if not model_dir or not os.path.isdir(model_dir):
        return None
    for f in os.listdir(model_dir):
        low = f.lower()
        if low.startswith("mmproj") and low.endswith(".gguf"):
            return os.path.join(model_dir, f)
    return None


def _get_chat_handler(clip_model_path):
    """根据 mmproj 路径创建正确的 chat handler
    
    优先级: Qwen2.5/3 VL > LLaVA 1.6 > LLaVA 1.5 > 通用
    """
    if not clip_model_path or not os.path.exists(clip_model_path):
        return None
    try:
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        handler = Qwen25VLChatHandler(clip_model_path=clip_model_path)
        print(f"[llama] 使用 Qwen25VLChatHandler: {clip_model_path}")
        return handler
    except Exception as e:
        print(f"[llama] Qwen25VLChatHandler 加载失败: {e}")
    try:
        from llama_cpp.llama_chat_format import Llava16ChatHandler
        handler = Llava16ChatHandler(clip_model_path=clip_model_path)
        print(f"[llama] 使用 Llava16ChatHandler: {clip_model_path}")
        return handler
    except Exception as e:
        print(f"[llama] Llava16ChatHandler 加载失败: {e}")
    try:
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        handler = Llava15ChatHandler(clip_model_path=clip_model_path)
        print(f"[llama] 使用 Llava15ChatHandler: {clip_model_path}")
        return handler
    except Exception as e:
        print(f"[llama] Llava15ChatHandler 加载失败: {e}")
    print("[llama] 无可用的 chat handler")
    return None


def load_model(model_path, n_ctx=None, n_gpu_layers=None, chat_handler=None, force_cpu=False):
    """加载 GGUF 模型
    
    Args:
        model_path: 模型文件路径
        n_ctx: 上下文长度
        n_gpu_layers: GPU 层数（-1=全部, 0=仅 CPU）
        chat_handler: 多模态处理器（mmproj 文件路径，或 'auto' 自动检测）
        force_cpu: 强制使用 CPU 模式（覆盖 n_gpu_layers）
    """
    global _model, _config

    n_ctx = n_ctx or GPU_DEFAULT_CTX

    # 处理 GPU 层数
    if force_cpu:
        n_gpu_layers = 0
        print("[llama] 强制 CPU 模式")
    else:
        n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else GPU_DEFAULT_LAYERS
        if n_gpu_layers == 0:
            print("[llama] CPU 模式（n_gpu_layers=0）")
        elif n_gpu_layers > 0:
            print(f"[llama] GPU 模式: {n_gpu_layers} 层")
        else:
            print("[llama] 自动 GPU 模式（所有层）")

    with _lock:
        # 卸载旧模型
        if _model is not None:
            _model = None
            gc.collect()
            if HAVE_GPU:
                torch.cuda.empty_cache()

        # chat handler（多模态 mmproj）
        handler = None
        mmproj_path = None
        if chat_handler and chat_handler not in ("None", "", "none"):
            if chat_handler == "auto":
                mmproj_path = _find_mmproj(model_path)
                if mmproj_path:
                    print(f"[llama] 自动检测到 mmproj: {mmproj_path}")
            else:
                # 用户指定了具体路径（绝对路径或相对于 MODELS_DIR）
                if os.path.isabs(chat_handler):
                    mmproj_path = chat_handler
                else:
                    mmproj_path = os.path.join(MODELS_DIR, chat_handler)
            if mmproj_path and os.path.exists(mmproj_path):
                handler = _get_chat_handler(mmproj_path)
            elif mmproj_path:
                print(f"[llama] mmproj 文件不存在: {mmproj_path}")
        else:
            # 默认自动检测
            mmproj_path = _find_mmproj(model_path)
            if mmproj_path:
                print(f"[llama] 自动检测到 mmproj: {mmproj_path}")
                handler = _get_chat_handler(mmproj_path)

        _model = LLAMA_CPP.Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            chat_handler=handler,
        )
        _config = {
            "model": os.path.basename(model_path),
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "chat_handler": chat_handler,
            "mmproj": mmproj_path,
            "mmproj_loaded": handler is not None,
            "force_cpu": force_cpu,
        }
        print(f"[llama] 模型已加载: {_config['model']}, mmproj={'已加载' if handler else '未加载'}")


def unload_model():
    """卸载模型"""
    global _model, _config
    with _lock:
        _model = None
        _config = {}
        gc.collect()
        if HAVE_GPU:
            torch.cuda.empty_cache()
        print("[llama] 模型已卸载")


def is_loaded():
    return _model is not None


def get_config():
    return _config.copy()


def list_models():
    """列出 models 目录下的 GGUF 文件，附带 mmproj 信息"""
    models = []
    for root, dirs, files in os.walk(MODELS_DIR):
        for f in files:
            if f.lower().endswith(".gguf"):
                rel = os.path.relpath(os.path.join(root, f), MODELS_DIR)
                # 跳过 mmproj 文件本身
                if os.path.basename(rel).lower().startswith("mmproj"):
                    continue
                full_path = os.path.join(MODELS_DIR, rel)
                mmproj = _find_mmproj(full_path)
                models.append({
                    "path": rel,
                    "mmproj": os.path.relpath(mmproj, MODELS_DIR) if mmproj else None,
                    "has_vision": mmproj is not None,
                })
    return sorted(models, key=lambda m: m["path"])


# ─── 推理 ────────────────────────────────────────────────────────────────────
def _img_to_bytes(img_data):
    """base64 / bytes / PIL Image -> bytes"""
    if isinstance(img_data, bytes):
        return img_data
    if isinstance(img_data, str):
        raw = img_data
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        try:
            decoded = base64.b64decode(raw)
        except Exception as e:
            log.error("[DEBUG] base64 decode failed: %s, input length=%s", e, len(img_data))
            raise
        # 校验图片有效性
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(io.BytesIO(decoded))
            pil_img.verify()
            log.info("[DEBUG _img_to_bytes] OK: %s, %sx%s, decoded=%s bytes",
                     pil_img.format, pil_img.width, pil_img.height, len(decoded))
        except Exception as e:
            log.error("[DEBUG _img_to_bytes] PIL verify failed: %s, decoded=%s bytes", e, len(decoded))
        return decoded
    if isinstance(img_data, PILImage.Image):
        buf = io.BytesIO()
        img_data.save(buf, format="PNG")
        return buf.getvalue()
    raise ValueError(f"不支持的图片格式: {type(img_data)}")


def infer(prompt, images=None, system=None, stream=False, **params):
    """推理（支持流式输出）
    
    Args:
        prompt: 用户输入
        images: 图片列表
        system: 系统提示
        stream: 是否流式输出
        **params: 其他参数
    
    Returns:
        如果 stream=False: 返回完整字符串
        如果 stream=True: 返回生成器
    """
    if _model is None:
        raise RuntimeError("模型未加载")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    # 多模态消息
    if images:
        content = [{"type": "text", "text": prompt}]
        for img in images:
            img_bytes = _img_to_bytes(img)
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})

    # 优化参数：增加批处理、降低温度提高确定性
    gen_params = {
        "max_tokens": params.get("max_tokens", GPU_DEFAULT_MAX_TOKENS),
        "temperature": params.get("temperature", DEFAULT_TEMPERATURE),
        "top_p": params.get("top_p", DEFAULT_TOP_P),
        "top_k": params.get("top_k", DEFAULT_TOP_K),
        "repeat_penalty": params.get("repeat_penalty", DEFAULT_REPEAT_PENALTY),
        "stream": stream,
    }
    
    # CPU 模式优化：降低 top_k 和温度
    if not HAVE_GPU:
        gen_params["top_k"] = min(gen_params.get("top_k", 40), 20)
        gen_params["temperature"] = max(gen_params["temperature"], 0.5)
    
    if stream:
        def generate():
            for chunk in _model.create_chat_completion(messages=messages, **gen_params):
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
        return generate()
    else:
        response = _model.create_chat_completion(messages=messages, **gen_params)
        return response["choices"][0]["message"]["content"]


# ─── 初始化 ──────────────────────────────────────────────────────────────────
detect_backend()  # 初始化 LLAMA_AVAILABLE 和 HAVE_GPU
