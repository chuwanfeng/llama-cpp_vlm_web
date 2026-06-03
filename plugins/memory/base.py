# -*- coding: utf-8 -*-
"""
记忆提供者插件基类

移植自 hermes-agent/plugins/memory/
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from plugins.base import Plugin

logger = logging.getLogger(__name__)


class MemoryPlugin(Plugin):
    """记忆提供者插件基类

    提供用户记忆、会话搜索、长期记忆存储等功能。
    """

    name = "memory"
    description = "用户记忆提供者"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.user_id: Optional[str] = None

    @abstractmethod
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """存储记忆

        Args:
            content: 记忆内容
            metadata: 关联元数据

        Returns:
            True 表示存储成功
        """
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索记忆

        Args:
            query: 搜索查询
            limit: 返回结果数量

        Returns:
            记忆条目列表，每项包含 content, score, metadata
        """
        pass

    @abstractmethod
    def get_user_profile(self) -> Optional[Dict[str, Any]]:
        """获取用户画像

        Returns:
            用户画像数据
        """
        pass

    def initialize(self) -> bool:
        """初始化记忆存储"""
        return True

    def shutdown(self):
        """关闭记忆存储"""
        pass

    def on_message(self, message: str, context: Dict[str, Any]) -> Optional[str]:
        """消息拦截：可注入相关记忆"""
        # 子类可重写此方法，在消息前注入相关记忆
        return None

    def on_response(self, response: str, context: Dict[str, Any]) -> Optional[str]:
        """响应拦截：可存储对话记忆"""
        # 子类可重写此方法，存储对话到长期记忆
        return None
