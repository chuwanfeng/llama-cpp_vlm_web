"""DeepSeek V3 工具调用解析器。

格式使用特殊 unicode tokens:
    <｜tool▁call▁begin｜>
    <｜tool▁function▁callE｜>function_name
    ```json
    {"arg": "value"}
    ```
    <｜tool▁call▁end｜>

基于 VLLM 的对应实现。
"""

import re
import uuid
import logging
from typing import List, Optional

from services.tool_call_parsers import (
    FunctionCall,
    ParseResult,
    ToolCall,
    ToolCallParser,
    register_parser,
)

logger = logging.getLogger(__name__)


@register_parser("deepseek_v3")
class DeepSeekV3ToolCallParser(ToolCallParser):
    """DeepSeek V3 工具调用解析器。

    使用特殊 Unicode 标记（全角尖括号）。
    提取类型、函数名和 JSON 参数。
    确保捕获模型执行多个操作时的所有工具调用。
    """

    START_TOKEN = "<｜tool▁call▁begin｜>"

    # 正则：匹配 <｜tool▁call▁begin｜>...<｜tool▁call▁end｜> 块
    PATTERN = re.compile(
        r"<｜tool▁call▁begin｜>"
        r"(?P<type>.*?)"
        r"<｜tool▁function▁callE｜>(?P<function_name>.*?)\s*"
        r"```json\s*(?P<function_arguments>.*?)\s*```\s*"
        r"<｜tool▁call▁end｜>",
        re.DOTALL,
    )

    def parse(self, text: str) -> ParseResult:
        """解析输入文本，提取所有工具调用。"""
        if self.START_TOKEN not in text:
            return text, None

        try:
            matches = list(self.PATTERN.finditer(text))
            if not matches:
                return text, None

            tool_calls: List[ToolCall] = []

            for match in matches:
                func_name = match.group("function_name").strip()
                func_args = match.group("function_arguments").strip()

                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=FunctionCall(
                            name=func_name,
                            arguments=func_args,
                        ),
                    )
                )

            if tool_calls:
                # 内容 = 第一个工具调用块之前的文本
                content_index = text.find(self.START_TOKEN)
                content = text[:content_index].strip()
                return content if content else None, tool_calls

            return text, None

        except Exception as e:
            logger.error("Error parsing DeepSeek V3 tool calls: %s", e)
            return text, None
