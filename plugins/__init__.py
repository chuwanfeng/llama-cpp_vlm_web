# -*- coding: utf-8 -*-
"""
插件系统

提供可扩展的插件架构，支持记忆提供者、工具扩展等。
"""

from .base import Plugin, PluginManager

__all__ = ["Plugin", "PluginManager"]
