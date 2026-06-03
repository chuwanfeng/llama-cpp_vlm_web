# -*- coding: utf-8 -*-
"""
插件基类和插件管理器

移植自 hermes-agent/plugins/
"""

import importlib
import inspect
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class Plugin(ABC):
    """插件基类

    所有插件必须继承此类并实现必要的方法。
    """

    # 插件元数据
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    author: str = ""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled = True

    @abstractmethod
    def initialize(self) -> bool:
        """初始化插件

        Returns:
            True 表示初始化成功
        """
        pass

    @abstractmethod
    def shutdown(self):
        """关闭插件"""
        pass

    def get_tools(self) -> List[Dict[str, Any]]:
        """返回插件提供的工具列表

        Returns:
            工具 schema 列表
        """
        return []

    def on_message(self, message: str, context: Dict[str, Any]) -> Optional[str]:
        """消息拦截钩子

        Args:
            message: 用户消息
            context: 上下文信息

        Returns:
            如果返回字符串，将替换原消息；返回 None 则保持原消息
        """
        return None

    def on_response(self, response: str, context: Dict[str, Any]) -> Optional[str]:
        """响应拦截钩子

        Args:
            response: 助手响应
            context: 上下文信息

        Returns:
            如果返回字符串，将替换原响应；返回 None 则保持原响应
        """
        return None


class PluginManager:
    """插件管理器

    负责插件的发现、加载、初始化和生命周期管理。
    """

    def __init__(self, plugin_dir: Optional[Path] = None):
        self.plugin_dir = plugin_dir or Path(__file__).parent
        self._plugins: Dict[str, Plugin] = {}
        self._hooks: Dict[str, List[Callable]] = {
            "pre_message": [],
            "post_response": [],
        }

    def discover(self) -> List[Type[Plugin]]:
        """发现可用插件

        扫描插件目录中的 Python 模块，查找 Plugin 子类。
        """
        plugins = []
        if not self.plugin_dir.exists():
            return plugins

        for item in self.plugin_dir.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                try:
                    module_name = f"plugins.{item.name}"
                    module = importlib.import_module(module_name)
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and issubclass(obj, Plugin)
                                and obj is not Plugin and not getattr(obj, "__abstractmethods__", None)):
                            plugins.append(obj)
                            logger.info("发现插件: %s (%s)", obj.name or name, module_name)
                except Exception as e:
                    logger.warning("加载插件 %s 失败: %s", item.name, e)
        return plugins

    def load(self, plugin_class: Type[Plugin], config: Optional[Dict[str, Any]] = None) -> Plugin:
        """加载并初始化插件"""
        instance = plugin_class(config)
        if instance.initialize():
            self._plugins[instance.name or plugin_class.__name__] = instance
            logger.info("插件 %s 已加载", instance.name)
            return instance
        else:
            logger.error("插件 %s 初始化失败", instance.name)
            raise RuntimeError(f"插件 {instance.name} 初始化失败")

    def unload(self, name: str):
        """卸载插件"""
        plugin = self._plugins.pop(name, None)
        if plugin:
            plugin.shutdown()
            logger.info("插件 %s 已卸载", name)

    def get(self, name: str) -> Optional[Plugin]:
        """获取插件实例"""
        return self._plugins.get(name)

    def list_plugins(self) -> List[Plugin]:
        """列出所有已加载插件"""
        return list(self._plugins.values())

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """获取所有插件提供的工具"""
        tools = []
        for plugin in self._plugins.values():
            if plugin.enabled:
                tools.extend(plugin.get_tools())
        return tools

    def run_pre_message_hooks(self, message: str, context: Dict[str, Any]) -> str:
        """运行消息前钩子"""
        for plugin in self._plugins.values():
            if plugin.enabled:
                try:
                    result = plugin.on_message(message, context)
                    if result is not None:
                        message = result
                except Exception as e:
                    logger.warning("插件 %s 的 on_message 出错: %s", plugin.name, e)
        return message

    def run_post_response_hooks(self, response: str, context: Dict[str, Any]) -> str:
        """运行响应后钩子"""
        for plugin in self._plugins.values():
            if plugin.enabled:
                try:
                    result = plugin.on_response(response, context)
                    if result is not None:
                        response = result
                except Exception as e:
                    logger.warning("插件 %s 的 on_response 出错: %s", plugin.name, e)
        return response

    def shutdown_all(self):
        """关闭所有插件"""
        for name, plugin in list(self._plugins.items()):
            try:
                plugin.shutdown()
            except Exception as e:
                logger.error("关闭插件 %s 时出错: %s", name, e)
        self._plugins.clear()
