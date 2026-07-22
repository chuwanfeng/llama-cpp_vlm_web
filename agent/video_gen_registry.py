"""
视频生成提供者注册表
参照 hermes-agent agent/video_gen_registry.py
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from agent.video_gen_provider import VideoGenProvider

log = logging.getLogger(__name__)

_providers: Dict[str, VideoGenProvider] = {}
_active_name: Optional[str] = None


def register_provider(provider: VideoGenProvider) -> None:
    _providers[provider.name] = provider
    log.info("Video gen provider registered: %s", provider.name)
    global _active_name
    if _active_name is None and provider.is_available():
        _active_name = provider.name


def list_providers() -> List[VideoGenProvider]:
    return list(_providers.values())


def get_provider(name: str) -> Optional[VideoGenProvider]:
    return _providers.get(name)


def get_active_provider() -> Optional[VideoGenProvider]:
    global _active_name
    if _active_name and _active_name in _providers:
        p = _providers[_active_name]
        if p.is_available():
            return p
        _active_name = None
    for name, p in _providers.items():
        if p.is_available():
            _active_name = name
            return p
    return None


def set_active(name: str) -> bool:
    if name in _providers and _providers[name].is_available():
        global _active_name
        _active_name = name
        return True
    return False


def _reset_for_tests() -> None:
    _providers.clear()
    global _active_name
    _active_name = None
