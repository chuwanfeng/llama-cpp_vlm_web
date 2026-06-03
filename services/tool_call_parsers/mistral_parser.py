"""Mistral 工具调用解析器。

支持两种格式（取决于 tokenizer 版本）：
- Pre-v11: content[TOOL_CALLS] [{"name": ..., "arguments": {...}}, ...]
- v11+:   content[TOOL_CALLS]tool_name1{"arg":"val"}[TOOL_CALLS]tool_name2{"arg":"val"}

基于 VLLM 的 MistralToolParser.extract_tool_calls()。
[TOOL_CALLS] 标记是 Mistral 模型使用的 bot_token。
"""

import json
import random
import string
import uuid
from typing import List

from services.tool_call_parsers import (
    FunctionCall,
    ParseResult,
    ToolCall,
    ToolCallParser,
    register_parser,
)


def _generate_mistral_id() -> str:
    """Mistral 工具调用 ID 为 9 位字母数字字符串。"""
    return "".join(random.choices(string.ascii_letters + string.digits, k=9))


@register_parser("mistral")
class MistralToolCallParser(ToolCallParser):
    """Mistral 格式工具调用解析器。

    自动检测格式：检查 [TOOL_CALLS] 之后的内容是否以 '[' 开头
    （Pre-v11 JSON 数组）或以工具名开头（v11+ 格式）。
    """

    # [TOOL_CALLS] 标记 —— 会根据 tokenizer 以不同形式出现
    BOT_TOKEN = "[TOOL_CALLS]"

    def parse(self, text: str) -> ParseResult:
        if self.BOT_TOKEN not in text:
            return text, None

        try:
            parts = text.split(self.BOT_TOKEN)
            content = parts[0].strip()
            raw_tool_calls = parts[1:]

            # 检测格式：第一个片段以 '[' 开头 → Pre-v11
            first_raw = raw_tool_calls[0].strip() if raw_tool_calls else ""
            is_pre_v11 = first_raw.startswith("[") or first_raw.startswith("{")

            tool_calls: List[ToolCall] = []

            if not is_pre_v11:
                # v11+ 格式: [TOOL_CALLS]tool_name{args}[TOOL_CALLS]tool_name2{args2}
                for raw in raw_tool_calls:
                    raw = raw.strip()
                    if not raw or "{" not in raw:
                        continue

                    brace_idx = raw.find("{")
                    tool_name = raw[:brace_idx].strip()
                    args_str = raw[brace_idx:]

                    # 验证并清理 JSON 参数
                    try:
                        parsed_args = json.loads(args_str)
                        args_str = json.dumps(parsed_args, ensure_ascii=False)
                    except json.JSONDecodeError:
                        pass  # 解析失败则保留原始值

                    tool_calls.append(
                        ToolCall(
                            id=_generate_mistral_id(),
                            type="function",
                            function=FunctionCall(name=tool_name, arguments=args_str),
                        )
                    )
            else:
                # Pre-v11 格式: [TOOL_CALLS] [{"name": ..., "arguments": {...}}]
                try:
                    parsed = json.loads(first_raw)
                    if isinstance(parsed, dict):
                        parsed = [parsed]

                    for tc in parsed:
                        if "name" not in tc:
                            continue
                        args = tc.get("arguments", {})
                        if isinstance(args, dict):
                            args = json.dumps(args, ensure_ascii=False)

                        tool_calls.append(
                            ToolCall(
                                id=_generate_mistral_id(),
                                type="function",
                                function=FunctionCall(
                                    name=tc["name"], arguments=args
                                ),
                            )
                        )
                except json.JSONDecodeError:
                    # 回退：使用 raw_decode 逐提取 JSON 对象
                    decoder = json.JSONDecoder()
                    idx = 0
                    while idx < len(first_raw):
                        try:
                            obj, end_idx = decoder.raw_decode(first_raw, idx)
                            if isinstance(obj, dict) and "name" in obj:
                                args = obj.get("arguments", {})
                                if isinstance(args, dict):
                                    args = json.dumps(args, ensure_ascii=False)
                                tool_calls.append(
                                    ToolCall(
                                        id=_generate_mistral_id(),
                                        type="function",
                                        function=FunctionCall(
                                            name=obj["name"], arguments=args
                                        ),
                                    )
                                )
                            idx = end_idx
                        except json.JSONDecodeError:
                            idx += 1

            if not tool_calls:
                return text, None

            return content if content else None, tool_calls

        except Exception:
            return text, None
