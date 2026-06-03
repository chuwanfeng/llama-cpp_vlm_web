"""
辅助模型路由 — 专用于侧任务（压缩/记忆/搜索）的廉价模型调用

从 hermes-agent 的 auxiliary_client.py 概念简化移植。
设计原则: 用户可指定一个便宜模型处理杂活，主模型只负责对话。

用法:
    from services.auxiliary import auxiliary_chat
    result = auxiliary_chat(
        messages=[{"role": "user", "content": "总结这段对话..."}],
        task="compression",
        aux_config=load_settings().get("aux_config", {})
    )
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from utils import get_logger, read_json
logger = get_logger("services.auxiliary")

# ─── 各厂商推荐的辅助默认模型（便宜/快速）───────────────────────
_AUX_DEFAULT_MODELS: Dict[str, str] = {
    "deepseek": "deepseek-chat",
    "zhipu": "glm-4-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
    "gemini": "gemini-2.0-flash",
    "qwen": "qwen-turbo",
    "moonshot": "moonshot-v1-8k",
    "ollama-cloud": "llama3.2:latest",
    "custom": "",
}

# ─── 默认 aux 配置 ──────────────────────────────────────────────
DEFAULT_AUX_CONFIG = {
    "enabled": False,
    "provider": "",          # 厂商 ID，空=跟随主模型
    "model": "",             # 模型名，空=用厂商默认
    "api_key": "",           # 留空=走环境变量/vendorCreds
    "base_url": "",          # 留空=用厂商默认
    "tasks": ["compression"],  # 哪些任务用辅助模型
}


def _read_aux_config(aux_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """读取辅助模型配置，合并默认值"""
    if aux_config is None:
        # 尝试从 settings.json 读取
        settings_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "settings.json")
        settings = read_json(settings_file, default={})
        if settings:
            aux_config = settings.get("aux_config", {})

    if not aux_config:
        return dict(DEFAULT_AUX_CONFIG)

    # 合并默认值
    merged = dict(DEFAULT_AUX_CONFIG)
    merged.update({k: v for k, v in aux_config.items() if k in merged})
    return merged


def is_aux_enabled(task: str, aux_config: Optional[Dict[str, Any]] = None) -> bool:
    """检查某个任务类型是否启用了辅助模型"""
    cfg = _read_aux_config(aux_config)
    if not cfg["enabled"]:
        return False
    return task in cfg.get("tasks", [])


def auxiliary_chat(
    messages: List[Dict[str, Any]],
    task: str = "general",
    aux_config: Optional[Dict[str, Any]] = None,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    base_url: str = "",
    api_key: str = "",
    model: str = "",
) -> str:
    """调用辅助模型进行非流式对话，返回完整响应文本。

    Args:
        messages: OpenAI 格式消息列表
        task: 任务类型 (compression, memory, search, vision)
        aux_config: 辅助模型配置字典，None=自动从 settings.json 读取
        max_tokens: 最大输出 token
        temperature: 温度
        base_url: 直接指定 base URL（优先级高于配置）
        api_key: 直接指定 API key（优先级高于配置）
        model: 直接指定模型（优先级高于配置）

    Returns:
        str: 模型响应文本

    Raises:
        RuntimeError: 调用失败且无降级
    """
    cfg = _read_aux_config(aux_config)

    # ── 确定 provider 和 model ──────────────────────────────
    provider = cfg.get("provider", "")
    aux_model = model or cfg.get("model", "")

    if not provider or not aux_model:
        # 未配置辅助模型 → 返回空（调用方应自行回退到主模型）
        logger.debug("Auxiliary model not configured for task '%s'", task)
        return ""

    # ── 确定 endpoint 凭据 ──────────────────────────────────
    aux_api_key = api_key or cfg.get("api_key", "")
    aux_base_url = base_url or cfg.get("base_url", "")

    # 如果没有直接提供凭据，从 vendorCreds 读取
    if not aux_api_key:
        settings_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "settings.json")
        settings = read_json(settings_file, default={})
        vendor_creds = settings.get("vendor_creds", {})

        vc = vendor_creds.get(provider, {})
        aux_api_key = aux_api_key or vc.get("api_key", "")
        aux_base_url = aux_base_url or vc.get("base_url", "")

    # ── 确定默认模型 ────────────────────────────────────────
    if not aux_model:
        aux_model = _AUX_DEFAULT_MODELS.get(provider, "")

    if not aux_model:
        logger.warning("No model specified for auxiliary provider '%s'", provider)
        return ""

    logger.info(
        "Auxiliary call [%s]: provider=%s model=%s messages=%d max_tokens=%d",
        task, provider, aux_model, len(messages), max_tokens,
    )

    # ── 执行调用 ────────────────────────────────────────────
    try:
        from backends.vendors import chat_stream

        # 收集流式输出
        collected: List[str] = []
        for chunk in chat_stream(
            vendor_id=provider,
            model=aux_model,
            messages=messages,
            api_key=aux_api_key,
            base_url=aux_base_url,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            collected.append(chunk)

        result = "".join(collected).strip()
        logger.debug("Auxiliary [%s]: got %d chars", task, len(result))
        return result

    except Exception as e:
        logger.error("Auxiliary call failed [%s]: %s", task, e)
        # 不上抛 — 调用方应有能力处理空返回
        return ""


def get_aux_config_for_frontend() -> Dict[str, Any]:
    """返回辅助模型配置给前端（屏蔽 api_key 等敏感项）"""
    cfg = _read_aux_config()
    return {
        "enabled": cfg.get("enabled", False),
        "provider": cfg.get("provider", ""),
        "model": cfg.get("model", ""),
        "tasks": cfg.get("tasks", ["compression"]),
        # 不返回 api_key / base_url — 前端从 vendorCreds 读取
    }


# ─── 便捷函数：供 ContextCompressor 等使用 ─────────────────────


def make_aux_callable(task: str = "compression", aux_config: Optional[Dict] = None):
    """创建一个适配 ContextCompressor._call_llm 签名的可调用对象。

    返回的函数签名兼容: fn(messages, model, max_tokens) -> str
    """
    def _call(messages: list, model: str = "", max_tokens: int = 4096, **kwargs) -> str:
        result = auxiliary_chat(
            messages=messages,
            task=task,
            aux_config=aux_config,
            max_tokens=max_tokens,
            model=model or "",
            temperature=kwargs.get("temperature", 0.3),
        )
        if not result:
            raise RuntimeError(f"Auxiliary model returned empty for task '{task}'")
        return result

    return _call
