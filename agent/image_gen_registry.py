"""
图片生成提供者注册表
参照 hermes-agent agent/image_gen_registry.py
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from agent.image_gen_provider import ImageGenProvider

log = logging.getLogger(__name__)

_providers: Dict[str, ImageGenProvider] = {}
_active_name: Optional[str] = None


def register_provider(provider: ImageGenProvider) -> None:
    """注册图片生成后端"""
    _providers[provider.name] = provider
    log.info("Image gen provider registered: %s", provider.name)

    # 首个注册的自动设为活跃
    global _active_name
    if _active_name is None and provider.is_available():
        _active_name = provider.name
        log.info("Auto-selected image gen provider: %s", provider.name)


def list_providers() -> List[ImageGenProvider]:
    """列出所有已注册后端"""
    return list(_providers.values())


def get_provider(name: str) -> Optional[ImageGenProvider]:
    """按名称获取"""
    return _providers.get(name)


def get_active_provider() -> Optional[ImageGenProvider]:
    """获取当前活跃的图片生成后端"""
    global _active_name
    if _active_name and _active_name in _providers:
        p = _providers[_active_name]
        if p.is_available():
            return p
        _active_name = None

    # 回退到第一个可用
    for name, p in _providers.items():
        if p.is_available():
            _active_name = name
            return p
    return None


def set_active(name: str) -> bool:
    """手动设置活跃后端"""
    if name in _providers and _providers[name].is_available():
        global _active_name
        _active_name = name
        return True
    return False


def _reset_for_tests() -> None:
    """仅测试使用"""
    _providers.clear()
    global _active_name
    _active_name = None
