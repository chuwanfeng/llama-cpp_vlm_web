"""


GPU 后端 — llama-cpp-python 原生 API


"""


import os


import gc


import json


import base64


import io


import traceback


from threading import Lock





from utils import get_logger


log = get_logger("backends.gpu")





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








def _detect_model_family(model_path):


    """根据模型文件名检测模型家族


    


    返回值用于选择正确的 chat handler / chat_format。


    """


    name = os.path.basename(model_path).lower()


    if "gemma-4" in name or "gemma4" in name:


        return "gemma4"


    if "gemma-3" in name or "gemma3" in name:


        return "gemma3"


    if "qwen3.5" in name or "qwen35" in name:


        return "qwen35"


    if "qwen3.6" in name or "qwen36" in name:


        return "qwen35"  # Qwen3.6 用 Qwen3.5 的 handler


    if "qwen3" in name:


        return "qwen3"


    if "qwen2.5" in name or "qwen25" in name:


        return "qwen25"


    if "llama-4" in name or "llama4" in name:


        return "llama4"


    if "llava-1.6" in name or "llava16" in name or "llava-v1.6" in name:


        return "llava16"


    if "llava-1.5" in name or "llava15" in name or "llava-v1.5" in name:


        return "llava15"


    if "minicpm" in name:


        # MiniCPM5 / MiniCPM-o 系列（包括 R1-thinking 变体）


        return "minicpm"


    return None








def _get_chat_handler(clip_model_path, model_path=None):


    """根据 mmproj 路径和模型名创建正确的 chat handler


    


    优先级: Gemma4 > Gemma3 > Qwen3.5 > Qwen3 VL > Qwen2.5 VL > LLaVA 1.6 > LLaVA 1.5


    


    Args:


        clip_model_path: mmproj 文件路径


        model_path: 模型文件路径（可选，用于模型家族检测）


    """


    if not clip_model_path or not os.path.exists(clip_model_path):


        return None


    


    family = _detect_model_family(model_path) if model_path else None


    


    # ── Gemma 4（原生 tool calling + 多模态） ──


    if family == "gemma4":


        try:


            from llama_cpp.llama_chat_format import Gemma4ChatHandler


            # E2B/E4B 不支持 thinking（仅 31B/26BA4B 支持）


            is_e2b = model_path and ("e2b" in os.path.basename(model_path).lower() or "e4b" in os.path.basename(model_path).lower())


            enable_thinking = not is_e2b


            handler = Gemma4ChatHandler(clip_model_path=clip_model_path, enable_thinking=enable_thinking)


            print(f"[llama] 使用 Gemma4ChatHandler: {clip_model_path} (thinking={enable_thinking})")


            return handler


        except Exception as e:


            print(f"[llama] Gemma4ChatHandler 加载失败: {e}")


    


    # ── Gemma 3 ──


    if family == "gemma3":


        try:


            from llama_cpp.llama_chat_format import Gemma3ChatHandler


            handler = Gemma3ChatHandler(clip_model_path=clip_model_path)


            print(f"[llama] 使用 Gemma3ChatHandler: {clip_model_path}")


            return handler


        except Exception as e:


            print(f"[llama] Gemma3ChatHandler 加载失败: {e}")


    


    # ── Qwen3.5 ──


    if family == "qwen35":


        try:


            from llama_cpp.llama_chat_format import Qwen35ChatHandler


            handler = Qwen35ChatHandler(clip_model_path=clip_model_path)


            print(f"[llama] 使用 Qwen35ChatHandler: {clip_model_path}")


            return handler


        except Exception as e:


            print(f"[llama] Qwen35ChatHandler 加载失败: {e}")


    


    # ── Qwen3 VL ──


    if family == "qwen3":


        try:


            from llama_cpp.llama_chat_format import Qwen3VLChatHandler


            handler = Qwen3VLChatHandler(clip_model_path=clip_model_path)


            print(f"[llama] 使用 Qwen3VLChatHandler: {clip_model_path}")


            return handler


        except Exception as e:


            print(f"[llama] Qwen3VLChatHandler 加载失败: {e}")


    


    # ── Qwen2.5 VL ──


    if family == "qwen25":


        try:


            from llama_cpp.llama_chat_format import Qwen25VLChatHandler


            handler = Qwen25VLChatHandler(clip_model_path=clip_model_path)


            print(f"[llama] 使用 Qwen25VLChatHandler: {clip_model_path}")


            return handler


        except Exception as e:


            print(f"[llama] Qwen25VLChatHandler 加载失败: {e}")


    


    # ── MiniCPM (R1-thinking 文本模型，无需 mmproj，靠 GGUF 元数据自动匹配模板) ──


    if family == "minicpm":


        print(f"[llama] MiniCPM 模型（R1-thinking），使用 GGUF 自动检测的 chat_format")


        return None  # 不需要 VLM handler，chat_format 从 GGUF 元数据自动读取


    


    # ── 通用回退（按优先级尝试已知 handler） ──


    try:


        from llama_cpp.llama_chat_format import Llava16ChatHandler


        handler = Llava16ChatHandler(clip_model_path=clip_model_path)


        print(f"[llama] 回退 LLaVA 1.6: {clip_model_path}")


        return handler


    except Exception:


        pass


    try:


        from llama_cpp.llama_chat_format import Llava15ChatHandler


        handler = Llava15ChatHandler(clip_model_path=clip_model_path)


        print(f"[llama] 回退 LLaVA 1.5: {clip_model_path}")


        return handler


    except Exception:


        pass


    


    print("[llama] 无可用的 chat handler")


    return None








def load_model(model_path, n_ctx=None, n_gpu_layers=None, chat_handler=None, force_cpu=False,


              rope_scaling=None, rope_freq_base=None, rope_scale=None):


    """加载 GGUF 模型


    


    Args:


        model_path: 模型文件路径


        n_ctx: 上下文长度（默认 8192，支持 RoPE 扩展到 32K+）


        n_gpu_layers: GPU 层数（-1=全部, 0=仅 CPU）


        chat_handler: 多模态处理器（mmproj 文件路径，或 'auto' 自动检测）


        force_cpu: 强制使用 CPU 模式（覆盖 n_gpu_layers）


        rope_scaling: RoPE 扩展类型 ("none", "linear", "yarn")


        rope_freq_base: RoPE 基础频率（0=根据模型家族自动选择）


        rope_scale: RoPE 扩展倍数（8K→16K=2.0）


    """


    global _model, _config





    n_ctx = n_ctx or GPU_DEFAULT_CTX


    


    # ── RoPE 上下文扩展 ──


    from config import GPU_ROPE_SCALING, GPU_ROPE_FREQ_BASE, GPU_ROPE_SCALE, FAMILY_ROPE_BASE


    rope_scaling = rope_scaling or GPU_ROPE_SCALING


    rope_scale = rope_scale or GPU_ROPE_SCALE


    


    # 自动检测模型家族的推荐 RoPE freq_base


    model_family = _detect_model_family(model_path)


    if rope_freq_base is None or rope_freq_base <= 0:


        rope_freq_base = GPU_ROPE_FREQ_BASE or FAMILY_ROPE_BASE.get(model_family, 0)


    


    # 上下文超过 8K 时自动启用 RoPE 扩展


    if n_ctx > 8192 and rope_scaling != "none":


        print(f"[llama] RoPE 扩展: n_ctx={n_ctx}, scaling={rope_scaling}, scale={rope_scale}, freq_base={rope_freq_base}")





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


                handler = _get_chat_handler(mmproj_path, model_path)


            elif mmproj_path:


                print(f"[llama] mmproj 文件不存在: {mmproj_path}")


        else:


            # 默认自动检测


            mmproj_path = _find_mmproj(model_path)


            if mmproj_path:


                print(f"[llama] 自动检测到 mmproj: {mmproj_path}")


                handler = _get_chat_handler(mmproj_path, model_path)





        llama_kwargs = {


            "model_path": model_path,


            "n_ctx": n_ctx,


            "n_gpu_layers": n_gpu_layers,


            "verbose": False,


        }


        


        # ── chat_handler / chat_format ──


        from config import GPU_CHAT_FORMAT


        if handler is not None:


            llama_kwargs["chat_handler"] = handler


        if GPU_CHAT_FORMAT:


            # 环境变量显式指定 chat_format（覆盖自动检测）


            llama_kwargs["chat_format"] = GPU_CHAT_FORMAT


            print(f"[llama] 使用 chat_format: {GPU_CHAT_FORMAT}")


        # 否则 llama-cpp-python 从 GGUF 元数据自动检测 chat_format


        


        # ── 性能优化 (llama-cpp-python >= 0.3.0) ──


        from config import GPU_N_BATCH, GPU_N_UBATCH, GPU_FLASH_ATTN, GPU_KV_CACHE_DTYPE, GPU_OFFLOAD_KQV


        llama_kwargs["n_batch"] = GPU_N_BATCH


        llama_kwargs["n_ubatch"] = GPU_N_UBATCH


        if HAVE_GPU and GPU_FLASH_ATTN:


            llama_kwargs["flash_attn"] = True


            print(f"[llama] Flash Attention 已启用")


        if GPU_KV_CACHE_DTYPE:


            llama_kwargs["type_k"] = GPU_KV_CACHE_DTYPE


            llama_kwargs["type_v"] = GPU_KV_CACHE_DTYPE


            print(f"[llama] KV Cache 量化: {GPU_KV_CACHE_DTYPE}")


        if HAVE_GPU and GPU_OFFLOAD_KQV:


            llama_kwargs["offload_kqv"] = True


            print(f"[llama] KQV offload 已启用")


        


        # RoPE 扩展参数


        if n_ctx > 8192 and rope_scaling != "none":


            llama_kwargs["rope_freq_base"] = rope_freq_base


            # llama-cpp-python >= 0.4: rope_scaling_type removed, freq_base handles YaRN


        _model = LLAMA_CPP.Llama(**llama_kwargs)


        _config = {


            "model": os.path.basename(model_path),


            "model_path": model_path,


            "n_ctx": n_ctx,


            "n_gpu_layers": n_gpu_layers,


            "chat_handler": chat_handler,


            "mmproj": mmproj_path,


            "mmproj_loaded": handler is not None,


            "force_cpu": force_cpu,


            "rope_scaling": rope_scaling if n_ctx > 8192 else "none",


            "rope_freq_base": rope_freq_base,


            "rope_scale": rope_scale,


            "model_family": model_family,


            "n_batch": GPU_N_BATCH,


            "flash_attn": HAVE_GPU and GPU_FLASH_ATTN,


            "kv_cache_dtype": GPU_KV_CACHE_DTYPE or "f16 (default)",


            "chat_format": GPU_CHAT_FORMAT or "auto (GGUF metadata)",


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








def infer(prompt=None, messages=None, images=None, system=None, stream=False, **params):


    """推理（支持流式输出）


    


    Args:


        prompt: 用户输入（单条消息，与 messages 二选一）


        messages: 多轮对话历史 [{role, content}, ...]，优先使用


        images: 图片列表


        system: 系统提示（会被加到 messages 最前面）


        stream: 是否流式输出


        **params: 其他参数


    


    Returns:


        如果 stream=False: 返回完整字符串


        如果 stream=True: 返回生成器


    """


    if _model is None:


        raise RuntimeError("模型未加载")





    chat_messages = []





    # 如果有 messages 数组（多轮对话），直接使用


    if messages:


        chat_messages = list(messages)  # 浅拷贝


        # system 插入到最前面（如果提供了 system 且 messages 第一条不是 system）


        if system and (not chat_messages or chat_messages[0].get("role") != "system"):


            chat_messages.insert(0, {"role": "system", "content": system})


        # 图片嵌入最后一条 user 消息（messages 模式下 images 被忽略的 bug 修复）


        if images:


            last_user = None


            for m in reversed(chat_messages):


                if m.get("role") == "user":


                    last_user = m


                    break


            if last_user:


                text_content = last_user.get("content") or ""


                multimodal_content = [{"type": "text", "text": text_content}]


                for img in images:


                    img_bytes = _img_to_bytes(img)


                    b64 = base64.b64encode(img_bytes).decode("utf-8")


                    multimodal_content.append({


                        "type": "image_url",


                        "image_url": {"url": f"data:image/png;base64,{b64}"}


                    })


                last_user["content"] = multimodal_content


    else:


        # 单条 prompt 模式（向后兼容）


        if system:


            chat_messages.append({"role": "system", "content": system})





        # 多模态消息


        if images:


            content = [{"type": "text", "text": prompt or ""}]


            for img in images:


                img_bytes = _img_to_bytes(img)


                b64 = base64.b64encode(img_bytes).decode("utf-8")


                content.append({


                    "type": "image_url",


                    "image_url": {"url": f"data:image/png;base64,{b64}"}


                })


            chat_messages.append({"role": "user", "content": content})


        else:


            chat_messages.append({"role": "user", "content": prompt or ""})





    # 优化参数：增加批处理、降低温度提高确定性


    gen_params = {


        "max_tokens": params.get("max_tokens") or GPU_DEFAULT_MAX_TOKENS,


        "temperature": params.get("temperature") or DEFAULT_TEMPERATURE,


        "top_p": params.get("top_p") or DEFAULT_TOP_P,


        "top_k": params.get("top_k") or DEFAULT_TOP_K,


        "repeat_penalty": params.get("repeat_penalty") or DEFAULT_REPEAT_PENALTY,


        "stream": stream,


    }


    


    # 提取 tools 参数（llama-cpp-python 需要显式传，否则 Qwen 模板渲染会炸）


    tools = params.get("tools", [])


    


    # 【最终防线】转换 chat_messages 中所有 tool_calls 的 arguments


    # 前端 JSON 序列化会把 dict 变回 string，Qwen 模板 |items 要求 mapping


    # 不管消息从哪来（app.py 直调 / loop.py / 前端 SSE），这里统一兜底


    for msg in chat_messages:


        if isinstance(msg, dict) and msg.get("role") == "assistant":


            for tc in msg.get("tool_calls") or []:


                func = tc.get("function", {}) if isinstance(tc, dict) else {}


                args = func.get("arguments")


                if isinstance(args, str):


                    import json


                    try:


                        func["arguments"] = json.loads(args)


                    except (json.JSONDecodeError, TypeError):


                        func["arguments"] = {}





    # ── 详细推理日志（模型/上下文/消息/参数） ──
    log.warning("[INFER] ========================================")
    log.warning("[INFER] 模型: %s  |  家族: %s  |  R1-thinking: %s  |  流式: %s",
                _config.get("model", "?"),
                _detect_model_family(_config.get("model_path", "")),
                "minicpm" in (_config.get("model_path","")+_config.get("model","")).lower() and "thinking" in (_config.get("model","")).lower(), stream)
    total_chars = sum(len(str(m.get("content",""))) for m in chat_messages)
    total_imgs = sum(1 for m in chat_messages if isinstance(m.get("content"), list))
    log.warning("[INFER] 消息: %d条, 总字符: %d, 图片: %d, 工具: %d",
                len(chat_messages), total_chars, total_imgs, len(tools) if tools else 0)
    log.warning("[INFER] 采样: temp=%.2f top_p=%.2f top_k=%d min_p=%.3f pres_pen=%.2f freq_pen=%.2f max_tok=%d",
                gen_params.get("temperature", 0.8), gen_params.get("top_p", 0.9),
                gen_params.get("top_k", 40), gen_params.get("min_p", 0.0),
                gen_params.get("present_penalty", 1.0), gen_params.get("frequency_penalty", 0.0),
                gen_params.get("max_tokens", 4096))
    for idx, m in enumerate(chat_messages):
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            text_parts = [p.get("text","") for p in content if p.get("type")=="text"]
            img_count = sum(1 for p in content if p.get("type")=="image_url")
            text = "".join(text_parts)
            log.warning("[INFER] msg[%d]%-9s 图%d 文(%d): %s", idx, role, img_count, len(text), text[:300])
        elif isinstance(content, str):
            limit = 800 if role == "system" else 300
            tag = "SYSTEM" if role == "system" else role.upper()
            preview = content[:limit] + ("..." if len(content) > limit else "")
            log.warning("[INFER] msg[%d] %-9s(%d): %s", idx, tag, len(content), preview)
        else:
            log.warning("[INFER] msg[%d] role=%-9s type=%s", idx, role, type(content).__name__)
    # tool_calls 类型检查
    for m in chat_messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            for tc_idx, tc in enumerate(m.get("tool_calls") or []):
                func = tc.get("function", {}) if isinstance(tc, dict) else {}
                args = func.get("arguments")
                log.warning("[INFER] tool_call[%d] args=%s: %s", tc_idx, type(args).__name__, repr(args)[:200])
    log.warning("[INFER] ========================================")


    # CPU 模式优化：降低 top_k 和温度


    if not HAVE_GPU:


        gen_params["top_k"] = min(gen_params.get("top_k") or 40, 20)


        gen_params["temperature"] = max(gen_params.get("temperature") or 0.7, 0.5)


    


    # 检测 R1 风格思考模型（MiniCPM5-Thinking 等）


    # 这些模型使用 <｜end▁of▁thinking｜>... 模式输出


    # llama-cpp-python 的 chat_handler 会剥离  标签，映射为 reasoning_content / content


    # 当模型 chat_format 不支持 Thinking 时，全部输出进入 reasoning_content → 需要后处理


    model_family = _detect_model_family(_config.get("model_path", ""))


    is_r1_thinking = model_family == "minicpm" and "thinking" in (_config.get("model", "").lower())





    if stream:


        def generate():


            # R1 静默缓冲：在找到  分隔点之前不输出任何内容


            # 因为一旦作为 reasoning 发出，前端会把它固定显示在思考区


            r1_buffer = ""


            r1_split_found = False  # True = 已找到  标记，后续直接作为 content 流式输出





            for chunk in _model.create_chat_completion(messages=chat_messages, tools=tools, **gen_params):


                if "choices" in chunk and len(chunk["choices"]) > 0:


                    delta = chunk["choices"][0].get("delta", {})


                    content = delta.get("content", "")


                    reasoning = delta.get("reasoning_content", "")


                    tool_calls = delta.get("tool_calls")





                    if is_r1_thinking:


                        if r1_split_found:


                            # 已在  之后，正常流式输出


                            if content:


                                yield {"content": content}


                            if reasoning:


                                yield {"reasoning_content": reasoning}


                            if tool_calls:


                                yield {"tool_calls": tool_calls}


                        elif content:


                            # 直接收到 content（模型 chat_format 正常工作了）


                            # 先把积累的 reasoning 发出，再切换到 content 模式


                            if r1_buffer:


                                yield {"reasoning_content": r1_buffer}


                                r1_buffer = ""


                            r1_split_found = True


                            yield {"content": content}


                            if reasoning:


                                yield {"reasoning_content": reasoning}


                        elif reasoning:


                            # 收到 reasoning 但没有 content → 静默缓冲


                            r1_buffer += reasoning


                            # 尝试检测  分隔点（llama-cpp-python 可能已剥离标签）


                            think_start = "<" + "think" + ">"
                            think_end = "<" + "/think" + ">"
                            sep_pos = -1


                            sep_len = 0


                            # 检测行首的结束标签


                            if think_end in r1_buffer:


                                sep_pos = r1_buffer.rfind(think_end)


                                sep_len = len(think_end)


                            if sep_pos >= 0 and sep_pos + sep_len < len(r1_buffer):


                                # 找到了标签且有后续内容


                                r1_split_found = True


                                prefix = r1_buffer[:sep_pos + sep_len]


                                suffix = r1_buffer[sep_pos + sep_len:]


                                if prefix:


                                    yield {"reasoning_content": prefix}


                                if suffix:


                                    yield {"content": suffix}


                                r1_buffer = ""


                            # 未找到分隔点 → 继续静默缓冲（不 yield ）


                        # else: 既无 reasoning 也无 content，等待下一个 chunk


                    else:


                        if reasoning:


                            yield {"reasoning_content": reasoning}


                        if content:


                            yield {"content": content}


                        if tool_calls:


                            yield {"tool_calls": tool_calls}





            # 流结束回退：如果 R1 模型缓冲了内容但没有找到分隔点，


            # 说明 llama-cpp-python 已剥离  标签且 chat_format 未正确分离


            # → 缓冲的全部内容作为 answer 输出


            if is_r1_thinking and r1_buffer and not r1_split_found:


                log.warning(


                    "[R1] 未检测到  标签，reasoning（%d字符）全量回退为 content",


                    len(r1_buffer)


                )


                yield {"content": r1_buffer}


        return generate()


    else:


        response = _model.create_chat_completion(messages=chat_messages, tools=tools, **gen_params)


        msg = response["choices"][0]["message"]


        result = {"content": msg.get("content", "")}


        reasoning = msg.get("reasoning_content", "")


        if msg.get("tool_calls"):


            result["tool_calls"] = msg["tool_calls"]





        # R1 思考模型非流式回退


        if is_r1_thinking and reasoning and not result["content"]:


            think_end_tag = "<" + "/think" + ">"
            sep_pos = -1


            sep_len = 0


            if think_end_tag in reasoning:


                sep_pos = reasoning.rfind(think_end_tag)


                sep_len = len(think_end_tag)


            if sep_pos >= 0 and sep_pos + sep_len < len(reasoning):


                result["reasoning_content"] = reasoning[:sep_pos + sep_len]


                result["content"] = reasoning[sep_pos + sep_len:].strip()


            else:


                # 没有标记 → reasoning 全量作为 content


                result["content"] = reasoning


                result["reasoning_content"] = ""


                log.warning("[R1 非流式] 未检测到  标签，reasoning 回退为 content（%d字符）",


                            len(reasoning))


        elif reasoning:


            result["reasoning_content"] = reasoning


        return result





# ─── 初始化 ──────────────────────────────────────────────────────────────────


detect_backend()  # 初始化 LLAMA_AVAILABLE 和 HAVE_GPU


