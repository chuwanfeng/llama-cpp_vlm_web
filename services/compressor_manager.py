"""
压缩管理器 — ContextCompressor 的单例包装，集成辅助模型

用法:
    from services.compressor_manager import get_compressor
    compressor = get_compressor(context_length=131072)
    compressed = compressor.compress(messages, provider="deepseek", model="deepseek-chat")
"""

from __future__ import annotations

import json
from utils import get_logger
import os
from typing import Any, Callable, Dict, List, Optional

from services.context_compressor import ContextCompressor
from services.auxiliary import make_aux_callable, is_aux_enabled

logger = get_logger("services.compressor_manager")

_compressor: Optional[ContextCompressor] = None
_compressor_context_length: int = 0


def get_compressor(
    context_length: int = 131072,
    threshold_percent: float = 0.50,
    protect_first_n: int = 3,
    protect_last_n: int = 20,
    quiet_mode: bool = False,
) -> ContextCompressor:
    """获取或创建 ContextCompressor 单例

    如果 context_length 变了会重建。
    """
    global _compressor, _compressor_context_length

    if _compressor is None or _compressor_context_length != context_length:
        _compressor = ContextCompressor(
            call_llm_fn=_make_compression_callable(),
            context_length=context_length,
            threshold_percent=threshold_percent,
            protect_first_n=protect_first_n,
            protect_last_n=protect_last_n,
            quiet_mode=quiet_mode,
        )
        _compressor_context_length = context_length
    return _compressor


def _make_compression_callable() -> Callable:
    """创建适配 ContextCompressor._call_llm 的 callable

    如果 aux_config 启用了 compression 任务 → 用辅助模型
    否则 → 返回一个在调用时动态判断的 wrapper
    """
    aux_config = _load_aux_config()

    if is_aux_enabled("compression", aux_config):
        logger.info("Compressor using auxiliary model: %s/%s",
                     aux_config.get("provider"), aux_config.get("model", "default"))
        return make_aux_callable(task="compression", aux_config=aux_config)

    # 未启用辅助模型 → 返回空 callable（ContextCompressor 会回退到 __init__ 的 call_llm_fn）
    # 但 ContextCompressor 内部 compress() 方法接受 summary_model/base_url/api_key 参数，
    # 这些会作为参数传入 _call_llm。此处返回一个占位 callable，实际由 compress() 调用者传入模型信息。
    logger.info("Auxiliary model not enabled, compressor will require explicit model info")
    return _fallback_callable


def _fallback_callable(messages: list, model: str = "", max_tokens: int = 4096, **kwargs) -> str:
    """回退 callable — 当辅助模型未启用时，compress() 调用者应提供 model/base_url/api_key"""
    # 如果调用者没提供 model，尝试用主模型
    if not model:
        raise RuntimeError(
            "Compression requires a model. Either enable auxiliary model in settings, "
            "or pass summary_model/base_url/api_key to compress()."
        )

    base_url = kwargs.get("base_url", "")
    api_key = kwargs.get("api_key", "")
    temperature = kwargs.get("temperature", 0.3)

    from backends.vendors import chat_stream
    collected = []
    for chunk in chat_stream(
        vendor_id=kwargs.get("provider", "openai"),
        model=model,
        messages=messages,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
    ):
        collected.append(chunk)
    return "".join(collected).strip()


def _load_aux_config() -> Dict[str, Any]:
    """读取 aux_config"""
    settings_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "settings.json")
    try:
        if os.path.exists(settings_file):
            with open(settings_file, "r", encoding="utf-8") as f:
                return json.load(f).get("aux_config", {})
    except Exception:
        pass
    return {}
