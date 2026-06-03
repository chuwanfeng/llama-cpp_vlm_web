"""Clarify 工具 — 交互式澄清问题

允许 Agent 向用户展示结构化的多选题或开放式提示。

架构设计:
- 单一 clarify(question, choices) 工具
- 返回 JSON 包含问题 + 标记，前端将其渲染为交互提示
- Web UI: 前端拦截响应，用户回答后作为后续消息追加到对话中
"""

import json
from typing import List, Optional

from tools.registry import get_registry

MAX_CHOICES = 4

# JSON Schema for clarify tool parameters (name/description handled by registry)
CLARIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "The question to present to the user.",
        },
        "choices": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": MAX_CHOICES,
            "description": (
                "Up to 4 answer choices. Omit this parameter entirely to "
                "ask an open-ended question. When provided, the UI "
                "automatically appends an 'Other (type your answer)' option."
            ),
        },
    },
    "required": ["question"],
}


def tool_error(message: str) -> str:
    """Return an error as a JSON object."""
    return json.dumps({"error": message}, ensure_ascii=False)


def clarify(question: str, choices: Optional[List[str]] = None) -> str:
    """Ask the user a question, optionally with multiple-choice options.

    Returns a JSON object with the question metadata. For web UI, the frontend
    displays this as an interactive prompt and the user's answer is appended
    to the conversation as a follow-up message.

    Args:
        question: The question text to present.
        choices:  Up to 4 predefined answer choices. When omitted the
                 question is purely open-ended.

    Returns:
        JSON string with the question and interactive metadata.
    """
    # Validate question
    if not question or not question.strip():
        return tool_error("Question text is required.")

    question = question.strip()

    # Validate and trim choices
    if choices is not None:
        if not isinstance(choices, list):
            return tool_error("choices must be a list of strings.")
        choices = [str(c).strip() for c in choices if str(c).strip()]
        if len(choices) > MAX_CHOICES:
            choices = choices[:MAX_CHOICES]
        if not choices:
            choices = None  # empty list → open-ended

    # Build the response
    result = {
        "type": "clarify_request",
        "question": question,
        "choices_offered": choices,
        "display_hint": _build_display_hint(question, choices),
    }

    return json.dumps(result, ensure_ascii=False)


def _build_display_hint(question: str, choices: Optional[List[str]]) -> str:
    """Build a human-readable display hint for the user prompt."""
    lines = [f"❓ {question}", ""]
    if choices:
        for i, c in enumerate(choices, 1):
            lines.append(f"  {i}. {c}")
        lines.append(f"  {len(choices) + 1}. Other (type your answer)")
    lines.append("")
    return "\n".join(lines)


# -- Register ---------------------------------------------------------

registry = get_registry()
registry.register(
    name="clarify",
    toolset="interaction",
    schema=CLARIFY_SCHEMA,
    handler=clarify,
)
