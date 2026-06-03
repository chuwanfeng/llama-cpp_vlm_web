"""
上下文引擎抽象基类 — 可插拔的上下文管理策略。

从 hermes-agent/agent/context_engine.py 移植，适配 llama-cpp_vlm_web 项目。

设计目标：
    - 抽象基类定义统一接口
    - 内置 ContextCompressor 作为默认实现
    - 支持第三方引擎（如 LCM）通过插件系统替换
    - 配置驱动：config.yaml 中 context.engine 选择引擎

生命周期：
    1. 引擎实例化并注册（插件 register() 或默认）
    2. on_session_start() — 对话开始时调用
    3. update_from_response() — 每次 API 响应后更新 token 使用数据
    4. should_compress() — 每轮检查后判断是否应压缩
    5. compress() — should_compress() 返回 True 时执行压缩
    6. on_session_end() — 真实会话边界时调用（CLI 退出、/reset、网关会话过期）
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ContextEngine(ABC):
    """所有上下文引擎必须实现的基类。"""

    # -- 标识 ----------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """短标识符（如 'compressor', 'lcm'）。"""

    # -- Token 状态（供 run_agent.py 读取显示/日志）--------------------
    #
    # 引擎必须维护这些字段。run_agent.py 直接读取它们。

    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0

    # -- 生命周期钩子 --------------------------------------------------

    def on_session_start(self, messages: List[Dict[str, Any]]) -> None:
        """新对话开始时调用。

        Args:
            messages: 初始消息列表（通常只有 system prompt）
        """

    def on_session_end(self) -> None:
        """真实会话边界时调用（CLI 退出、/reset、网关会话过期）。
        不是每轮调用 —— 只在会话真正结束时调用。
        """

    # -- Token 跟踪 ----------------------------------------------------

    def update_from_response(self, usage: Optional[Dict[str, int]]) -> None:
        """从 API 响应的 usage 字段更新 token 计数。

        Args:
            usage: API 响应中的 usage 字典，通常包含 prompt_tokens、
                   completion_tokens、total_tokens
        """
        if not usage:
            return
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    # -- 压缩决策与执行 ------------------------------------------------

    @abstractmethod
    def should_compress(self, messages: List[Dict[str, Any]], model: str = "") -> bool:
        """判断当前消息列表是否需要压缩。

        Args:
            messages: 当前对话消息列表
            model: 当前使用的模型名称（用于查询上下文长度限制）

        Returns:
            True 表示需要压缩，False 表示不需要
        """

    @abstractmethod
    def compress(self, messages: List[Dict[str, Any]], model: str = "") -> List[Dict[str, Any]]:
        """执行上下文压缩。

        Args:
            messages: 当前对话消息列表
            model: 当前使用的模型名称

        Returns:
            压缩后的消息列表
        """

    # -- 可选工具暴露 --------------------------------------------------

    def get_tools(self) -> List[Dict[str, Any]]:
        """返回此引擎提供的额外工具列表（如 lcm_grep）。

        默认返回空列表。子类可覆盖以暴露引擎特有的工具。

        Returns:
            工具 schema 列表
        """
        return []

    # -- 状态序列化（用于检查点/恢复）----------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """返回引擎状态的快照，用于会话持久化。

        Returns:
            可 JSON 序列化的状态字典
        """
        return {
            "name": self.name,
            "last_prompt_tokens": self.last_prompt_tokens,
            "last_completion_tokens": self.last_completion_tokens,
            "last_total_tokens": self.last_total_tokens,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
        }

    def restore(self, state: Dict[str, Any]) -> None:
        """从快照恢复引擎状态。

        Args:
            state: snapshot() 返回的状态字典
        """
        self.last_prompt_tokens = state.get("last_prompt_tokens", 0)
        self.last_completion_tokens = state.get("last_completion_tokens", 0)
        self.last_total_tokens = state.get("last_total_tokens", 0)
        self.threshold_tokens = state.get("threshold_tokens", 0)
        self.context_length = state.get("context_length", 0)


class CompressorEngine(ContextEngine):
    """
    基于 ContextCompressor 的上下文引擎实现。

    将 services/context_compressor.py 的功能包装为 ContextEngine 接口。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 引擎配置字典，可包含：
                - model: 用于摘要的模型名称
                - api_key: API 密钥
                - base_url: API 基础 URL
                - quiet_mode: 是否静默模式
                - max_output_chars: 最大输出字符数
        """
        from services.context_compressor import ContextCompressor

        self._config = config or {}
        self._compressor = ContextCompressor(
            model=self._config.get("model"),
            api_key=self._config.get("api_key"),
            base_url=self._config.get("base_url"),
            quiet_mode=self._config.get("quiet_mode", True),
            max_output_chars=self._config.get("max_output_chars"),
        )

    @property
    def name(self) -> str:
        return "compressor"

    def should_compress(self, messages: List[Dict[str, Any]], model: str = "") -> bool:
        """委托给 ContextCompressor.should_compress()。"""
        return self._compressor.should_compress(messages, model)

    def compress(self, messages: List[Dict[str, Any]], model: str = "") -> List[Dict[str, Any]]:
        """委托给 ContextCompressor.compress()。"""
        return self._compressor.compress(messages, model)

    def on_session_start(self, messages: List[Dict[str, Any]]) -> None:
        """重置压缩器状态。"""
        self._compressor.compression_count = 0
        self._compressor._ineffective_compression_count = 0

    def update_from_response(self, usage: Optional[Dict[str, int]]) -> None:
        """更新 token 计数并同步到压缩器。"""
        super().update_from_response(usage)
        if usage:
            self._compressor.total_input_tokens = usage.get("prompt_tokens", 0)
            self._compressor.total_output_tokens = usage.get("completion_tokens", 0)


# 引擎注册表
_engine_registry: Dict[str, type] = {}


def register_engine(name: str, engine_class: type) -> None:
    """注册自定义上下文引擎。

    Args:
        name: 引擎标识符
        engine_class: ContextEngine 的子类
    """
    if not issubclass(engine_class, ContextEngine):
        raise ValueError(f"Engine class must inherit from ContextEngine: {engine_class}")
    _engine_registry[name] = engine_class


def get_engine(name: str = "compressor", config: Optional[Dict[str, Any]] = None) -> ContextEngine:
    """获取指定名称的上下文引擎实例。

    Args:
        name: 引擎名称，默认 'compressor'
        config: 引擎配置

    Returns:
        ContextEngine 实例
    """
    if name == "compressor":
        return CompressorEngine(config)

    engine_class = _engine_registry.get(name)
    if engine_class:
        return engine_class(config)

    # 未知引擎回退到默认
    import logging
    logging.getLogger(__name__).warning("Unknown context engine '%s', falling back to compressor", name)
    return CompressorEngine(config)
