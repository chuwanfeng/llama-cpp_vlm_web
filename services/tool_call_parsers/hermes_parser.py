"""Hermes 格式工具调用解析器。

格式: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
基于 VLLM 的 Hermes2ProToolParser.extract_tool_calls()
"""

import json
import re
import uuid
from typing import List

from services.tool_call_parsers import (
    FunctionCall,
    ParseResult,
    ToolCall,
    ToolCallParser,
    register_parser,
)


@register_parser("hermes")
class HermesToolCallParser(ToolCallParser):
    """Hermes 格式解析器。

    匹配 <tool_call>...</tool_call> 标签中的 JSON，
    包含 "name" 和 "arguments" 字段。
    同时处理末尾的未闭合 <tool_call>（截断的生成）。
    """

    # 匹配闭合和未闭合的 tool_call 标签
    PATTERN = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>|<tool_call>\s*(.*)", re.DOTALL
    )

    def parse(self, text: str) -> ParseResult:
        if "<tool_call>" not in text:
            return text, None

        try:
            matches = self.PATTERN.findall(text)
            if not matches:
                return text, None

            tool_calls: List[ToolCall] = []
            for match in matches:
                # match 是元组 (closed_content, unclosed_content)
                raw_json = match[0] if match[0] else match[1]
                if not raw_json.strip():
                    continue

                tc_data = json.loads(raw_json)
                if "name" not in tc_data:
                    continue

                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=FunctionCall(
                            name=tc_data["name"],
                            arguments=json.dumps(
                                tc_data.get("arguments", {}), ensure_ascii=False
                            ),
                        ),
                    )
                )

            if not tool_calls:
                return text, None

            # 内容 = 第一个 <tool_call> 之前的文本
            content = text[: text.find("<tool_call>")].strip()
            return content if content else None, tool_calls

        except Exception:
            return text, None
