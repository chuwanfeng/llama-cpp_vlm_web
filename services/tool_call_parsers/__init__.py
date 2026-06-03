"""工具调用解析器注册表 — 从模型原始文本输出中提取结构化工具调用。

移植自 hermes-agent/environments/tool_call_parsers/__init__.py。
当模型（尤其是 llama.cpp 本地模型）不支持原生 function calling 时，
本模块从原始输出文本中解析 XML/自定义格式的工具调用。

用法:
    from services.tool_call_parsers import get_parser

    parser = get_parser("hermes")
    content, tool_calls = parser.parse(raw_model_output)
    # content = 剥离工具调用标记后的纯文本
    # tool_calls = ToolCall 对象列表，或 None
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)

# =============================================================================
# 类型定义（替换 openai.types 依赖，不引入 openai 包）
# =============================================================================


@dataclass
class FunctionCall:
    """工具调用中的函数信息。"""
    name: str = ""
    arguments: str = "{}"


@dataclass
class ToolCall:
    """结构化的工具调用对象，兼容 OpenAI 格式。"""
    id: str = ""
    type: str = "function"
    function: Optional[FunctionCall] = None


# 解析器返回值类型: (剥离后的文本, 工具调用列表)
ParseResult = Tuple[Optional[str], Optional[List[ToolCall]]]


# =============================================================================
# 基类
# =============================================================================


class ToolCallParser(ABC):
    """工具调用解析器基类。

    每个解析器知道如何从特定模型族的原始输出文本中提取结构化的工具调用。
    """

    @abstractmethod
    def parse(self, text: str) -> ParseResult:
        """解析模型原始输出文本，提取工具调用。

        Args:
            text: 模型输出的原始解码文本

        Returns:
            (content, tool_calls) 元组:
            - content: 剥离工具调用标记后的文本，纯工具调用时返回 None
            - tool_calls: ToolCall 对象列表，无工具调用时返回 None
        """
        raise NotImplementedError


# =============================================================================
# 注册表
# =============================================================================

# 名称 → 解析器类
PARSER_REGISTRY: Dict[str, Type[ToolCallParser]] = {}


def register_parser(name: str):
    """注册解析器类的装饰器。

    用法:
        @register_parser("hermes")
        class HermesToolCallParser(ToolCallParser):
            ...
    """
    def decorator(cls: Type[ToolCallParser]) -> Type[ToolCallParser]:
        PARSER_REGISTRY[name] = cls
        logger.debug("Registered tool call parser: %s", name)
        return cls
    return decorator


def get_parser(name: str) -> ToolCallParser:
    """根据名称获取解析器实例。

    Args:
        name: 解析器名称，如 "hermes" "llama" "deepseek_v3"

    Returns:
        实例化的解析器

    Raises:
        KeyError: 如果解析器名称未注册
    """
    if name not in PARSER_REGISTRY:
        available = sorted(PARSER_REGISTRY.keys())
        raise KeyError(
            f"Tool call parser '{name}' not found. Available parsers: {available}"
        )
    return PARSER_REGISTRY[name]()


def list_parsers() -> List[str]:
    """返回已注册解析器名称的排序列表。"""
    return sorted(PARSER_REGISTRY.keys())


# =============================================================================
# 自动注册所有解析器
# =============================================================================
# 导入每个 parser 模块会触发 @register_parser 装饰器自动注册

from services.tool_call_parsers.hermes_parser import HermesToolCallParser  # noqa: E402, F401
from services.tool_call_parsers.llama_parser import LlamaToolCallParser  # noqa: E402, F401
from services.tool_call_parsers.qwen_parser import QwenToolCallParser  # noqa: E402, F401
from services.tool_call_parsers.deepseek_v3_parser import DeepSeekV3ToolCallParser  # noqa: E402, F401
from services.tool_call_parsers.glm45_parser import Glm45ToolCallParser  # noqa: E402, F401
from services.tool_call_parsers.glm47_parser import Glm47ToolCallParser  # noqa: E402, F401
from services.tool_call_parsers.mistral_parser import MistralToolCallParser  # noqa: E402, F401
