"""
抽象基类 — 可插拔记忆提供者

从 hermes-agent 的 agent/memory_provider.py 完整移植，保留全部接口方法。
记忆提供者赋予 LLM 跨会话持久化记忆能力。内置提供者始终激活，
外部提供者（Honcho、Hindsight、Mem0 等）最多注册一个。

生命周期（由 MemoryManager 编排）:
  initialize()         — 连接、创建资源、预热
  is_available()       — 是否已配置、有凭据、就绪
  system_prompt_block()— 注入 system prompt 的静态文本
  prefetch(query)      — 每轮对话前的背景召回
  queue_prefetch()     — 每轮后的异步预取（供下一轮使用）
  sync_turn(user, asst)— 每轮对话后的持久化写入
  get_tool_schemas()   — 暴露给模型的工具 schema
  handle_tool_call()   — 执行工具调用
  shutdown()           — 清理退出

可选钩子:
  on_turn_start()      — 每轮触发时的运行时上下文
  on_session_end()     — 会话结束时的提取
  on_session_switch()  — 会话 ID 切换
  on_pre_compress()    — 上下文压缩前的信息提取
  on_memory_write()    — 镜像内置记忆写入
  on_delegation()      — 子代理完成观察
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryProvider(ABC):
    """抽象记忆提供者基类 — 完整保留 hermes-agent 全部接口"""

    @property
    @abstractmethod
    def name(self) -> str:
        """短标识符（如 'builtin', 'honcho', 'hindsight'）"""

    # -- 核心生命周期（必须实现）----------------------------------------------

    @abstractmethod
    def is_available(self) -> bool:
        """返回 True 表示已配置、有凭据、可用。
        在初始化阶段调用，不应发出网络请求，只检查配置和已安装的依赖。"""

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        """会话级初始化。创建资源、建立连接、启动后台线程等。

        kwargs 始终包含:
          - project_home (str): 项目目录路径
          - platform (str): 'web', 'cli', 等
        """

    def system_prompt_block(self) -> str:
        """返回注入 system prompt 的文本。静态提供者信息，与 prefetch() 分离。"""
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """当前轮次前的背景召回。返回格式化文本注入上下文，无结果返回空字符串。"""
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """排队下一轮的背景召回。默认空操作。"""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """持久化完成的轮次到后端。应非阻塞——如果后端有延迟，排队后台处理。"""

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """返回该提供者暴露的工具 schema（OpenAI function calling 格式）。
        无工具时返回空列表。"""

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """处理该提供者的工具调用。仅对 get_tool_schemas() 中的工具调用。
        必须返回 JSON 字符串。"""
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def shutdown(self) -> None:
        """清理关闭——刷新队列、关闭连接。"""

    # -- 可选钩子（覆盖以加入）------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """每轮对话开始时调用，传入用户消息。
        kwargs 可能包含: remaining_tokens, model, platform, tool_count"""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """会话结束时调用（显式退出或超时）。messages 为完整对话历史。
        不在每轮后调用——仅在真正的会话边界。"""

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        """会话 ID 切换时调用。当 reset=True 时是全新的对话，提供者应刷新缓冲区。"""

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """上下文压缩即将丢弃旧消息前调用。
        返回文本注入压缩摘要提示词，使压缩器保留提供者提取的洞察。"""
        return ""

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """子代理完成时在父代理上调用。task=委托提示词, result=子代理最终响应"""

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """内置记忆工具写入时调用。
        action: 'add', 'replace', 'remove'
        target: 'memory' 或 'user'
        content: 条目内容
        metadata: 来源元数据"""
