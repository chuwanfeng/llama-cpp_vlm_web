"""Tool call parser - extracts and formats tool calls for different backends.

For llama-cpp (no native function calling), we:
1. Inject tool definitions into the system prompt with XML format guide
2. Parse <tool_call> XML blocks from the model's text output
3. Execute the tool and inject <tool_result> blocks back

For vendors/Ollama (native function calling), we use the API's built-in
tool support - no parsing needed.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# ── XML tool call guide (injected into system prompt for llama-cpp) ─────────

TOOL_CALL_XML_GUIDE = """
## Tool Usage Rules (MANDATORY)

You MUST use the available tools whenever they are applicable. Never claim you cannot
do something that a tool can handle. If the user asks you to read a file, list files,
search the web, run a command, or write content, you MUST call the appropriate tool
immediately.

**Tool call format:**
<tool_call name="tool_name">
{{"param1": "value1", "param2": "value2"}}
</tool_call>

**Critical rules:**
- Call tools DECISIVELY and WITHOUT hesitation when they fit the task
- JSON parameters on ONE LINE, using double quotes
- NEVER invent information when a tool can provide the answer
- NEVER say "I cannot access" or "I don't have access to" if a tool exists for that
- After receiving tool results, respond to the user using that information
- Multiple tool calls are allowed in one response

{calling_tips}
"""


def build_tool_prompt(tools: List[Dict[str, Any]]) -> str:
    """Build a system prompt describing available tools for llama-cpp.

    Args:
        tools: List of tool schemas (OpenAI format).

    Returns:
        Tool prompt string to inject into system prompt.
    """
    if not tools:
        return ""

    tool_descriptions = []
    calling_tips = []

    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})

        # Build parameter descriptions
        param_lines = []
        props = params.get("properties", {})
        required = params.get("required", [])
        for pname, pinfo in props.items():
            req_mark = " (required)" if pname in required else ""
            param_lines.append(
                f"    - {pname}: {pinfo.get('description', '')}{req_mark}"
            )

        param_text = "\n".join(param_lines) if param_lines else "    (no parameters)"

        tool_descriptions.append(
            f"### {name}\n{desc}\nParameters:\n{param_text}"
        )

        # Build calling tips
        calling_tips.append(f"- `{name}`: {desc[:120]}")

    tool_text = "\n\n".join(tool_descriptions)
    tips_text = "\n".join(calling_tips)

    return TOOL_CALL_XML_GUIDE.format(calling_tips=tips_text) + "\n\n" + tool_text


# ── XML parser for extracting tool calls from model output ───────────────────

# Pattern 1: 标准格式 <tool_call name="tool_name">JSON</tool_call>
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call\s+name=[\"']([^\"']+)[\"']\s*>\s*\n?\s*(.*?)\s*\n?\s*</tool_call>",
    re.DOTALL,
)

# Pattern 2: Qwen 原生格式 <tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>
QWEN_TC_PATTERN = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
QWEN_PARAM_PATTERN = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def _parse_tool_calls_from_text(text: str):
    """从文本中提取所有 tool_calls，同时支持标准格式和 Qwen 原生格式。

    Returns:
        tuple: (clean_text, calls) — calls 是 [(name, params), ...] 列表
    """
    calls = []
    clean = text

    # 先尝试标准格式
    for match in TOOL_CALL_PATTERN.finditer(text):
        name = match.group(1)
        json_str = match.group(2).strip()
        try:
            params = json.loads(json_str)
        except json.JSONDecodeError:
            params = {"_raw": json_str}
        calls.append((name, params))

    # 再尝试 Qwen 原生格式（<tool_call><function=NAME>...</function></tool_call>）
    for match in QWEN_TC_PATTERN.finditer(text):
        fn_name = match.group(1).strip()
        fn_body = match.group(2)
        params = {}
        for pm in QWEN_PARAM_PATTERN.finditer(fn_body):
            key = pm.group(1).strip()
            value = pm.group(2).strip()
            params[key] = value
        calls.append((fn_name, params))

    # 清除所有 tool_call 块
    clean = TOOL_CALL_PATTERN.sub("", clean)
    clean = QWEN_TC_PATTERN.sub("", clean)
    clean = clean.strip()

    return (clean if clean else None, calls)


class ToolCallParser:
    """Presses tool calls from llama-cpp model output."""

    def __init__(self):
        self._buffer = ""

    def feed(self, chunk: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Feed a text chunk and return any complete tool calls found."""
        self._buffer += chunk
        _, calls = _parse_tool_calls_from_text(self._buffer)
        return calls

    def clear(self):
        """Reset the buffer."""
        self._buffer = ""

    @staticmethod
    def parse_final(text: str) -> Tuple[Optional[str], List[Tuple[str, Dict[str, Any]]]]:
        """Parse finished output: return (clean_text, tool_calls).

        Strips tool_call blocks from the text and returns:
        - clean_text: The text with tool_call blocks removed.
        - tool_calls: List of (name, params) tuples.
        """
        return _parse_tool_calls_from_text(text)


def parse_xml_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Extract tool calls from text using XML pattern matching.

    同时支持标准格式和 Qwen 原生格式。
    """
    _, calls = _parse_tool_calls_from_text(text)
    return calls