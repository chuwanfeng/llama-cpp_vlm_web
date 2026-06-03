"""Qwen 2.5 工具调用解析器。

与 Hermes 使用相同的 <tool_call> 格式。
单独注册为 "qwen" 以便使用 --tool-parser=qwen。
"""

from services.tool_call_parsers import register_parser
from services.tool_call_parsers.hermes_parser import HermesToolCallParser


@register_parser("qwen")
class QwenToolCallParser(HermesToolCallParser):
    """Qwen 2.5 工具调用解析器。
    与 Hermes 相同的 <tool_call>{"name": ..., "arguments": ...}</tool_call> 格式。
    """

    pass  # 格式相同 -- 继承 Hermes 的全部逻辑
