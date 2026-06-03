# -*- coding: utf-8 -*-
"""
记忆提供者插件

支持多种记忆后端：
- honcho: Honcho 用户建模
- mem0: Mem0 记忆层
- local: 本地 SQLite FTS5
"""

from .base import MemoryPlugin

__all__ = ["MemoryPlugin"]
