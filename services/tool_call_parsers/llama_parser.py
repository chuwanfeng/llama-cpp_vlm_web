"""Llama 3.x / 4 工具调用解析器。

格式: 模型输出包含 "name" 和 "arguments"（或 "parameters"）的 JSON 对象。
可能以 <|python_tag|> 开头。支持由内容或分号分隔的多个 JSON 对象。

基于 VLLM 的 Llama3JsonToolParser.extract_tool_calls()
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


@register_parser("llama3_json")
@register_parser("llama4_json")
class LlamaToolCallParser(ToolCallParser):
    """Llama 3.x 和 4 JSON 格式工具调用解析器。

    查找包含 "name" + ("arguments" 或 "parameters") 键的 JSON 对象。
    使用 Python 的 json.JSONDecoder.raw_decode 从混合文本中鲁棒提取 JSON。
    """

    BOT_TOKEN = "<|python_tag|>"

    # 查找潜在 JSON 对象起始位置
    JSON_START = re.compile(r"\{")

    def parse(self, text: str) -> ParseResult:
        # 快速检查：需要 BOT_TOKEN 或 JSON 花括号
        if self.BOT_TOKEN not in text and "{" not in text:
            return text, None

        try:
            decoder = json.JSONDecoder()
            tool_calls: List[ToolCall] = []
            end_index = -1  # 跟踪上次解析的 JSON 结束位置

            for match in self.JSON_START.finditer(text):
                start = match.start()
                # 跳过已解析过的 JSON 对象内的花括号
                if start <= end_index:
                    continue

                try:
                    obj, json_end = decoder.raw_decode(text[start:])
                    end_index = start + json_end

                    # 必须有 "name" 和 "arguments" 或 "parameters"
                    name = obj.get("name")
                    args = obj.get("arguments", obj.get("parameters"))

                    if not name or args is None:
                        continue

                    # 规范化 arguments 为 JSON 字符串
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    elif not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)

                    tool_calls.append(
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            type="function",
                            function=FunctionCall(name=name, arguments=args),
                        )
                    )
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

            if not tool_calls:
                return text, None

            # 内容 = 第一个工具调用 JSON 之前的文本
            first_tc_start = text.find("{")
            if self.BOT_TOKEN in text:
                first_tc_start = text.find(self.BOT_TOKEN)
            content = text[:first_tc_start].strip() if first_tc_start > 0 else None

            return content, tool_calls

        except Exception:
            return text, None
