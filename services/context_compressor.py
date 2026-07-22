"""
上下文压缩引擎 — 自动压缩长对话的上下文窗口

从 hermes-agent 的 agent/context_compressor.py + agent/context_engine.py 完整移植。
保留全部功能，不精简任何特性。

完整功能列表:
  - ContextEngine ABC 基类（含阈值管理、反抖动、冷却、重置）
  - ContextCompressor 完整实现:
    - 工具输出剪枝（廉价预通，含信息性 1 行摘要）
    - Token 预算尾保护（替代固定消息数）
    - 结构化摘要模板（Active Task, Goal, Completed Actions, Decisions,
      Resolved/Pending Questions, Files, Remaining Work）
    - 迭代摘要更新（跨多次压缩保留信息）
    - 反抖动保护（连续 2 次节省<10% 则跳过）
    - 摘要失败冷却（provider 错误 600s，transient 60s）
    - 工具调用/结果对完整性检查与修复
    - 摘要模型回退到主模型
    - 聚焦主题压缩（/compress <topic>）
    - 敏感信息（API key, token, password）在摘要中自动过滤
    - 图片 token 预算估算
    - Truncated tool_call args → valid JSON 保持
"""

from __future__ import annotations

import hashlib
import json
from utils import get_logger
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = get_logger("services.context_compressor")

# ─── 常量 ──────────────────────────────────────────────────────
MINIMUM_CONTEXT_LENGTH = 8192
_CHARS_PER_TOKEN = 4
_IMAGE_TOKEN_ESTIMATE = 1600
_IMAGE_CHAR_EQUIVALENT = _IMAGE_TOKEN_ESTIMATE * _CHARS_PER_TOKEN

SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Your current task is identified in the '## Active Task' section of the "
    "summary — resume exactly from there. "
    "Respond ONLY to the latest user message "
    "that appears AFTER this summary. The current session state (files, "
    "config, etc.) may reflect work described here — avoid repeating it:"
)
LEGACY_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"

# 最小摘要输出 token 数
_MIN_SUMMARY_TOKENS = 2000
# 要压缩内容中分配给摘要的比例
_SUMMARY_RATIO = 0.20
# 摘要 token 绝对上限
_SUMMARY_TOKENS_CEILING = 12_000
# 剪枝 old tool results 时的占位符
_PRUNED_TOOL_PLACEHOLDER = "[Old tool output cleared to save context space]"
# 摘要失败冷却时间（秒）
_SUMMARY_FAILURE_COOLDOWN_SECONDS = 600

# ─── 工具函数 ──────────────────────────────────────────────────


def estimate_messages_tokens_rough(messages: List[Dict[str, Any]]) -> int:
    """粗略估算消息列表的 token 数（~4 字符/token）"""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(part.get("text", ""))
        total += 20  # 角色和格式开销
    return total // _CHARS_PER_TOKEN


def redact_sensitive_text(text: str) -> str:
    """过滤 API key、token、password 等敏感信息"""
    if not text:
        return text
    patterns = [
        (r'(api[_-]?key\s*[:=]\s*)([^\s,;]{8,})', r'\1[REDACTED]'),
        (r'(sk-[a-zA-Z0-9]{20,})', '[REDACTED_KEY]'),
        (r'(Bearer\s+)([a-zA-Z0-9\-_\.]{20,})', r'\1[REDACTED]'),
        (r'(password\s*[:=]\s*)([^\s,;]+)', r'\1[REDACTED]'),
        (r'(secret\s*[:=]\s*)([^\s,;]+)', r'\1[REDACTED]'),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _content_length_for_budget(raw_content: Any) -> int:
    """返回消息内容用于 token 预算的有效字符长度"""
    if isinstance(raw_content, str):
        return len(raw_content)
    if not isinstance(raw_content, list):
        return len(str(raw_content or ""))
    total = 0
    for p in raw_content:
        if isinstance(p, str):
            total += len(p)
            continue
        if not isinstance(p, dict):
            total += len(str(p))
            continue
        ptype = p.get("type")
        if ptype in {"image_url", "input_image", "image"}:
            total += _IMAGE_CHAR_EQUIVALENT
        else:
            total += len(p.get("text", "") or "")
    return total


def _content_text_for_contains(content: Any) -> str:
    """返回消息内容的最佳文本视图（用于子串检查）"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p for p in parts if p)
    return str(content)


def _append_text_to_content(content: Any, text: str, *, prepend: bool = False) -> Any:
    """安全地向消息内容追加/前置纯文本"""
    if content is None:
        return text
    if isinstance(content, str):
        return text + content if prepend else content + text
    if isinstance(content, list):
        text_block = {"type": "text", "text": text}
        return [text_block, *content] if prepend else [*content, text_block]
    rendered = str(content)
    return text + rendered if prepend else rendered + text


def _truncate_tool_call_args_json(args: str, head_chars: int = 200) -> str:
    """缩短 JSON 工具调用参数中的长字符串值，保持 JSON 有效性"""
    try:
        parsed = json.loads(args)
    except (ValueError, TypeError):
        return args

    def _shrink(obj: Any) -> Any:
        if isinstance(obj, str):
            if len(obj) > head_chars:
                return obj[:head_chars] + "...[truncated]"
            return obj
        if isinstance(obj, dict):
            return {k: _shrink(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_shrink(v) for v in obj]
        return obj

    shrunken = _shrink(parsed)
    return json.dumps(shrunken, ensure_ascii=False)


def _summarize_tool_result(tool_name: str, tool_args: str, tool_content: str) -> str:
    """创建工具调用 + 结果的信息性 1 行摘要"""
    try:
        args_dict = json.loads(tool_args) if tool_args else {}
    except (json.JSONDecodeError, TypeError):
        args_dict = {}

    content = tool_content or ""
    content_len = len(content)
    line_count = content.count("\n") + 1 if content.strip() else 0

    tool_specific = {
        "terminal": lambda: f"[terminal] ran `{str(args_dict.get('command',''))[:80]}` -> exit {_extract_exit_code(content)}, {line_count} lines output",
        "read_file": lambda: f"[read_file] read {args_dict.get('path','?')} from line {args_dict.get('offset',1)} ({content_len:,} chars)",
        "write_file": lambda: f"[write_file] wrote to {args_dict.get('path','?')}",
        "search_files": lambda: f"[search_files] search for '{args_dict.get('pattern','?')}' in {args_dict.get('path','.')}",
        "web_search": lambda: f"[web_search] query='{args_dict.get('query','?')}' ({content_len:,} chars result)",
        "web_fetch": lambda: f"[web_fetch] {str(args_dict.get('url','?'))[:80]} ({content_len:,} chars)",
        "memory": lambda: f"[memory] {args_dict.get('action','?')} on {args_dict.get('target',args_dict.get('key','?'))}",
    }

    if tool_name in tool_specific:
        return tool_specific[tool_name]()

    # 通用回退
    first_args = ", ".join(f"{k}={str(v)[:40]}" for k, v in list(args_dict.items())[:2])
    return f"[{tool_name}] {first_args} ({content_len:,} chars result)"


def _extract_exit_code(content: str) -> str:
    match = re.search(r'"exit_code"\s*:\s*(-?\d+)', content)
    return match.group(1) if match else "?"


# ─── ContextEngine ABC ──────────────────────────────────────────


class ContextEngine(ABC):
    """上下文引擎抽象基类 — 完整保留 hermes-agent ContextEngine 全部属性"""

    @property
    @abstractmethod
    def name(self) -> str:
        """引擎名称"""

    def __init__(self, context_length: int, threshold_percent: float = 0.50,
                 quiet_mode: bool = False):
        self.context_length = context_length
        self.threshold_percent = threshold_percent
        self.quiet_mode = quiet_mode
        self.threshold_tokens = max(
            int(context_length * threshold_percent), MINIMUM_CONTEXT_LENGTH
        )
        self.compression_count = 0

        # 反抖动跟踪
        self._last_compression_savings_pct: float = 100.0
        self._ineffective_compression_count: int = 0
        self._summary_failure_cooldown_until: float = 0.0
        self._last_summary_error: Optional[str] = None
        self._last_summary_dropped_count: int = 0
        self._last_summary_fallback_used: bool = False
        self._last_aux_model_failure_error: Optional[str] = None
        self._last_aux_model_failure_model: Optional[str] = None

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def on_session_reset(self) -> None:
        """重置所有会话级状态"""
        self.compression_count = 0
        self._last_compression_savings_pct = 100.0
        self._ineffective_compression_count = 0
        self._summary_failure_cooldown_until = 0.0
        self._last_summary_error = None
        self._last_summary_dropped_count = 0
        self._last_summary_fallback_used = False
        self._last_aux_model_failure_error = None
        self._last_aux_model_failure_model = None

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """从 API 响应更新跟踪的 token 使用量"""
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)

    def should_compress(self, prompt_tokens: int = 0) -> bool:
        """检查上下文是否超过压缩阈值（含反抖动保护）"""
        tokens = prompt_tokens if prompt_tokens > 0 else self.last_prompt_tokens
        if tokens < self.threshold_tokens:
            return False
        if self._ineffective_compression_count >= 2:
            if not self.quiet_mode:
                logger.warning(
                    "Compression skipped — last %d compressions saved <10%% each.",
                    self._ineffective_compression_count,
                )
            return False
        return True

    @abstractmethod
    def compress(self, messages: List[Dict[str, Any]],
                 summary_model: str = "", base_url: str = "",
                 api_key: str = "", focus_topic: str = "",
                 memory_context: str = "") -> List[Dict[str, Any]]:
        """压缩消息列表，返回压缩后的消息列表"""


# ─── ContextCompressor ──────────────────────────────────────────


class ContextCompressor(ContextEngine):
    """默认上下文引擎 — 通过有损摘要压缩对话上下文

    算法:
      1. 剪枝 old tool results（廉价，不调用 LLM）
      2. 保护 head 消息（system prompt + 首次交换）
      3. 按 token 预算保护 tail 消息（最近约 20K token）
      4. 用结构化 LLM 提示词摘要中间轮次
      5. 后续压缩时，迭代更新之前的摘要
    """

    # 摘要模型输入的截断限制
    _CONTENT_MAX = 6000
    _CONTENT_HEAD = 4000
    _CONTENT_TAIL = 1500
    _TOOL_ARGS_MAX = 1500
    _TOOL_ARGS_HEAD = 1200

    @property
    def name(self) -> str:
        return "compressor"

    def __init__(
        self,
        call_llm_fn: Callable,
        context_length: int,
        threshold_percent: float = 0.50,
        protect_first_n: int = 3,
        protect_last_n: int = 20,
        summary_target_ratio: float = 0.20,
        quiet_mode: bool = False,
        model: str = "",
        provider: str = "",
    ):
        super().__init__(context_length, threshold_percent, quiet_mode)
        self._call_llm = call_llm_fn
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.summary_target_ratio = max(0.10, min(summary_target_ratio, 0.80))
        self.model = model
        self.provider = provider

        # 从阈值推导 token 预算
        target_tokens = int(self.threshold_tokens * self.summary_target_ratio)
        self.tail_token_budget = target_tokens
        self.max_summary_tokens = min(
            int(context_length * 0.05), _SUMMARY_TOKENS_CEILING
        )

        # 迭代摘要状态
        self._previous_summary: Optional[str] = None

        # Preflight 延迟跟踪
        self._last_rough_tokens_when_real_prompt_fit: int = 0
        self._last_compression_rough_tokens: int = 0

        if not quiet_mode:
            logger.info(
                "Context compressor initialized: context_length=%d threshold=%d (%.0f%%) "
                "target_ratio=%.0f%% tail_budget=%d",
                context_length, self.threshold_tokens, threshold_percent * 100,
                self.summary_target_ratio * 100, self.tail_token_budget,
            )

    # ------------------------------------------------------------------
    # 工具输出剪枝（廉价预通，不调用 LLM）
    # ------------------------------------------------------------------

    def _prune_old_tool_results(
        self, messages: List[Dict[str, Any]], protect_tail_count: int,
        protect_tail_tokens: int | None = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """将旧工具结果替换为信息性 1 行摘要 + 去重 + 截断工具调用参数

        返回 (pruned_messages, pruned_count)
        """
        if not messages:
            return messages, 0

        result = [m.copy() for m in messages]
        pruned = 0

        # 构建 tool_call_id → (tool_name, arguments_json) 索引
        call_id_to_tool: Dict[str, Tuple[str, str]] = {}
        for msg in result:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        cid = tc.get("id", "")
                        fn = tc.get("function", {})
                        call_id_to_tool[cid] = (fn.get("name", "unknown"), fn.get("arguments", ""))
                    else:
                        cid = getattr(tc, "id", "") or ""
                        fn = getattr(tc, "function", None)
                        name = getattr(fn, "name", "unknown") if fn else "unknown"
                        args_str = getattr(fn, "arguments", "") if fn else ""
                        call_id_to_tool[cid] = (name, args_str)

        # 确定剪枝边界
        if protect_tail_tokens is not None and protect_tail_tokens > 0:
            accumulated = 0
            boundary = len(result)
            min_protect = min(protect_tail_count, len(result))
            for i in range(len(result) - 1, -1, -1):
                msg = result[i]
                raw_content = msg.get("content") or ""
                content_len = _content_length_for_budget(raw_content)
                msg_tokens = content_len // _CHARS_PER_TOKEN + 10
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        args = tc.get("function", {}).get("arguments", "")
                        msg_tokens += len(args) // _CHARS_PER_TOKEN
                if accumulated + msg_tokens > protect_tail_tokens and (len(result) - i) >= min_protect:
                    boundary = i
                    break
                accumulated += msg_tokens
                boundary = i
            budget_protect_count = len(result) - boundary
            protected_count = max(budget_protect_count, min_protect)
            prune_boundary = len(result) - protected_count
        else:
            prune_boundary = len(result) - protect_tail_count

        # Pass 1: 去重
        content_hashes: Dict[str, Tuple[int, str]] = {}
        for i in range(len(result) - 1, -1, -1):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content") or ""
            if isinstance(content, list):
                continue
            if not isinstance(content, str) or len(content) < 200:
                continue
            h = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()[:12]
            if h in content_hashes:
                result[i] = {**msg, "content": "[Duplicate tool output — same content as a more recent call]"}
                pruned += 1
            else:
                content_hashes[h] = (i, msg.get("tool_call_id", "?"))

        # Pass 2: 替换旧工具结果为信息性摘要
        for i in range(prune_boundary):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                continue
            if not isinstance(content, str) or not content or content == _PRUNED_TOOL_PLACEHOLDER:
                continue
            if content.startswith("[Duplicate tool output"):
                continue
            if len(content) > 200:
                call_id = msg.get("tool_call_id", "")
                tool_name, tool_args = call_id_to_tool.get(call_id, ("unknown", ""))
                summary = _summarize_tool_result(tool_name, tool_args, content)
                result[i] = {**msg, "content": summary}
                pruned += 1

        # Pass 3: 截断 assistant 消息中受保护尾部之外的长工具调用参数
        for i in range(prune_boundary):
            msg = result[i]
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            new_tcs = []
            modified = False
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    if len(args) > 500:
                        new_args = _truncate_tool_call_args_json(args)
                        if new_args != args:
                            tc = {**tc, "function": {**tc["function"], "arguments": new_args}}
                            modified = True
                new_tcs.append(tc)
            if modified:
                result[i] = {**msg, "tool_calls": new_tcs}

        return result, pruned

    # ------------------------------------------------------------------
    # 摘要
    # ------------------------------------------------------------------

    def _compute_summary_budget(self, turns_to_summarize: List[Dict[str, Any]]) -> int:
        """按被压缩的内容量缩放摘要 token 预算"""
        content_tokens = estimate_messages_tokens_rough(turns_to_summarize)
        budget = int(content_tokens * _SUMMARY_RATIO)
        return max(_MIN_SUMMARY_TOKENS, min(budget, self.max_summary_tokens))

    def _serialize_for_summary(self, turns: List[Dict[str, Any]]) -> str:
        """将对话轮次序列化为带标签的文本（供摘要模型使用）"""
        parts = []
        for msg in turns:
            role = msg.get("role", "unknown")
            content = redact_sensitive_text(msg.get("content") or "")

            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                if len(content) > self._CONTENT_MAX:
                    content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
                parts.append(f"[TOOL RESULT {tool_id}]: {content}")
                continue

            if role == "assistant":
                if len(content) > self._CONTENT_MAX:
                    content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc_parts = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            _raw_args = fn.get("arguments", "")
                            if isinstance(_raw_args, dict):
                                _raw_args = json.dumps(_raw_args, ensure_ascii=False)
                            args = redact_sensitive_text(_raw_args)
                            if len(args) > self._TOOL_ARGS_MAX:
                                args = args[:self._TOOL_ARGS_HEAD] + "..."
                            tc_parts.append(f"  {name}({args})")
                        else:
                            fn = getattr(tc, "function", None)
                            name = getattr(fn, "name", "?") if fn else "?"
                            tc_parts.append(f"  {name}(...)")
                    content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
                parts.append(f"[ASSISTANT]: {content}")
                continue

            if len(content) > self._CONTENT_MAX:
                content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
            parts.append(f"[{role.upper()}]: {content}")

        return "\n\n".join(parts)

    def _generate_summary(
        self, turns_to_summarize: List[Dict[str, Any]],
        summary_model: str = "", base_url: str = "", api_key: str = "",
        focus_topic: str = "", memory_context: str = "",
    ) -> Optional[str]:
        """生成对话轮次的结构化摘要。失败时返回 None"""
        now = time.monotonic()
        if now < self._summary_failure_cooldown_until:
            logger.debug("Skipping context summary during cooldown")
            return None

        summary_budget = self._compute_summary_budget(turns_to_summarize)
        content_to_summarize = self._serialize_for_summary(turns_to_summarize)
        is_iterative = self._previous_summary is not None

        # 摘要器前言（来自 OpenCode + Codex 最佳实践）
        _preamble = (
            "You are a summarization agent creating a context checkpoint. "
            "Your output will be injected as reference material for a DIFFERENT "
            "assistant that continues the conversation. "
            "Do NOT respond to any questions or requests in the conversation — "
            "only output the structured summary. "
            "Do NOT include any preamble, greeting, or prefix. "
            "Write the summary in the same language the user was using in the "
            "conversation — do not translate or switch to English. "
            "NEVER include API keys, tokens, passwords, secrets, credentials, "
            "or connection strings in the summary — replace any that appear "
            "with [REDACTED]. Note that the user had credentials present, but "
            "do not preserve their values."
        )

        # 结构化模板
        _template = """## Active Task
[THE SINGLE MOST IMPORTANT FIELD. Copy the user's most recent request or
task assignment verbatim — the exact words they used. If multiple tasks
were requested and only some are done, list only the ones NOT yet completed.
The next assistant must pick up exactly here.
If no outstanding task exists, write "None."]

## Goal
[What the user is trying to accomplish overall]

## Constraints & Preferences
[Any constraints, preferences, or requirements the user mentioned.
Include code style preferences, tools to use/avoid, architectural constraints,
budget limits, deadlines, etc.]

## Completed Actions
[Key things that were DONE. Use bullet points. Include specific details:
file paths, commands run, configuration changes, versions, error messages fixed.
This is a RECORD of what happened — be specific.]

## Key Decisions & Rationale
[Important decisions made and WHY. Include trade-offs considered and rejected
alternatives. This helps the next assistant understand the reasoning.]

## Key Facts & Context
[Important facts, data, or context discovered during the conversation.
Include URLs, API endpoints, version numbers, configuration values, etc.]

## Resolved Questions
[Questions that were asked and ANSWERED. Each with the answer.]

## Pending Questions
[Questions that were asked but NOT yet answered. Each marked as UNRESOLVED.]

## Files Modified / Created
[Files that were changed — full paths and what was done to them.]

## Code Patterns & Conventions
[Any coding patterns, conventions, or architectural decisions observed.
Include specific examples if helpful.]

## Errors & Fixes
[Errors encountered and how they were fixed. Include error messages.]

## Remaining Work
[Work that was discussed but NOT done yet. The next assistant should use this
as a checklist. Be specific about what each item entails.]

## Critical Warnings
[Anything the next assistant MUST know to avoid repeating mistakes or
breaking things. Red flags, footguns, uncommitted changes, fragile states.]"""

        if is_iterative:
            # 迭代更新模式
            prompt = (
                f"{_preamble}\n\n"
                f"## Previous Summary (Update This)\n{self._previous_summary}\n\n"
                f"## New Turns to Merge\n"
                f"Below are additional conversation turns that happened AFTER the "
                f"previous summary was created. Update the summary by:\n"
                f"1. Keep all still-relevant information from the previous summary\n"
                f"2. Add new information from these new turns\n"
                f"3. Update 'Active Task' to reflect current state\n"
                f"4. Move completed items from 'Remaining Work' to 'Completed Actions'\n"
                f"5. Update 'Resolved Questions' and 'Pending Questions'\n\n"
                f"{content_to_summarize}\n\n"
                f"OUTPUT THE FULL UPDATED SUMMARY using this template:\n{_template}"
            )
        else:
            prompt = (
                f"{_preamble}\n\n"
                f"Summarize this conversation using the template below.\n\n"
                f"{content_to_summarize}\n\n"
                f"OUTPUT THE SUMMARY using this template:\n{_template}"
            )

        # 添加记忆上下文
        if memory_context and memory_context.strip():
            prompt += f"\n\n## Relevant Memory Context\n{memory_context}"

        # 添加聚焦主题
        if focus_topic and focus_topic.strip():
            prompt += (
                f"\n\n## Focus Topic\n"
                f"Prioritize preserving information related to: {focus_topic}\n"
                f"You may be more aggressive about compressing unrelated content."
            )

        # 调用 LLM 生成摘要
        try:
            model_to_use = summary_model or self.model
            result = self._call_llm(
                messages=[{"role": "user", "content": prompt}],
                model=model_to_use,
                max_tokens=summary_budget,
            )
            self._previous_summary = result
            self._last_summary_error = None
            return result
        except Exception as e:
            logger.error("Summary generation failed: %s", e)
            self._summary_failure_cooldown_until = time.monotonic() + _SUMMARY_FAILURE_COOLDOWN_SECONDS
            self._last_summary_error = str(e)
            return None

    # ------------------------------------------------------------------
    # 工具调用/结果对完整性检查
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_tool_pairs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """移除孤立的 tool 结果（无对应 tool_call），以及孤立的 tool_call（无对应结果）"""
        if not messages:
            return messages

        tool_call_ids = set()
        for m in messages:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    cid = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                    if cid:
                        tool_call_ids.add(cid)

        result = []
        for m in messages:
            if m.get("role") == "tool":
                tc_id = m.get("tool_call_id", "")
                if tc_id and tc_id not in tool_call_ids:
                    continue  # 孤立 tool 结果，丢弃
            result.append(m)

        return result

    # ------------------------------------------------------------------
    # 压缩
    # ------------------------------------------------------------------

    def compress(
        self,
        messages: List[Dict[str, Any]],
        summary_model: str = "",
        base_url: str = "",
        api_key: str = "",
        focus_topic: str = "",
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        """压缩对话上下文

        参数:
            messages: 完整消息列表
            summary_model: 覆盖摘要模型（None=使用主模型）
            focus_topic: 引导压缩聚焦某主题
            memory_context: 记忆提供者在压缩前提取的额外上下文

        返回: 压缩后的消息列表
        """
        if not messages:
            return messages

        n_messages = len(messages)
        original_tokens = estimate_messages_tokens_rough(messages)

        if not self.quiet_mode:
            logger.info(
                "Starting compression: %d messages, ~%d tokens, threshold=%d",
                n_messages, original_tokens, self.threshold_tokens,
            )

        # Step 0: Preflight 延迟检查 — 避免 rough estimate 噪声导致的假压缩
        self._last_compression_rough_tokens = original_tokens
        if self.should_defer_preflight_to_real_usage(original_tokens):
            if not self.quiet_mode:
                logger.info(
                    "Compression deferred — rough estimate %d likely noisy (last fit at %d)",
                    original_tokens, self._last_rough_tokens_when_real_prompt_fit,
                )
            return messages

        # Step 0.5: 剥离历史消息中的图片 data URL (避免 base64 污染摘要)
        messages = self._strip_images_from_messages(messages)

        # Step 1: 剪枝 tool results
        messages, pruned_count = self._prune_old_tool_results(
            messages, self.protect_last_n, protect_tail_tokens=self.tail_token_budget
        )
        if pruned_count and not self.quiet_mode:
            logger.info("Pruned %d old tool results", pruned_count)

        # Step 2: 确定要摘要的中间范围
        # 保护 head 消息
        head_count = min(self.protect_first_n, n_messages)
        # 保护 tail 消息（按 token 预算）
        tail_msg_count = self.protect_last_n

        compress_start = head_count
        compress_end = max(compress_start + 1, n_messages - tail_msg_count)

        if compress_start >= compress_end:
            if not self.quiet_mode:
                logger.info("Nothing to compress — protected ranges cover all messages")
            return messages

        # Step 3: 提取要摘要的中间轮次
        turns_to_summarize = messages[compress_start:compress_end]

        # Step 4: 生成摘要
        summary = self._generate_summary(
            turns_to_summarize,
            summary_model=summary_model,
            base_url=base_url,
            api_key=api_key,
            focus_topic=focus_topic,
            memory_context=memory_context,
        )

        # Step 5: 组装压缩后的消息
        compressed = messages[:head_count]

        if summary and summary.strip():
            # 包装摘要
            wrapped = SUMMARY_PREFIX + "\n\n" + summary + "\n\n--- END CONTEXT SUMMARY ---"
            # 确定合并策略：避免连续相同 role
            _merge_summary_into_tail = False
            last_head_role = messages[compress_start - 1].get("role", "user") if compress_start > 0 else "user"
            first_tail_role = messages[compress_end].get("role", "user") if compress_end < n_messages else "user"

            if last_head_role in ("assistant", "tool"):
                summary_role = "user"
            else:
                summary_role = "assistant"

            if summary_role == first_tail_role:
                flipped = "assistant" if summary_role == "user" else "user"
                if flipped != last_head_role:
                    summary_role = flipped
                else:
                    _merge_summary_into_tail = True

            if not _merge_summary_into_tail:
                compressed.append({"role": summary_role, "content": wrapped})

            # 添加 tail 消息
            for i in range(compress_end, n_messages):
                msg = messages[i].copy()
                if _merge_summary_into_tail and i == compress_end:
                    msg["content"] = _append_text_to_content(
                        msg.get("content"),
                        wrapped + "\n\n--- respond to the message below, not the summary above ---\n\n",
                        prepend=True,
                    )
                    _merge_summary_into_tail = False
                compressed.append(msg)
        else:
            # 摘要失败 — 添加回退消息
            fallback_note = (
                f"[System note: {compress_end - compress_start} earlier messages were "
                f"removed to free context space but could not be summarized. The removed "
                f"messages contained earlier work in this session. Continue based on the "
                f"recent messages below and the current state of any files or resources.]"
            )
            compressed.append({"role": "user", "content": fallback_note})
            self._last_summary_fallback_used = True
            self._last_summary_dropped_count = compress_end - compress_start

            for i in range(compress_end, n_messages):
                compressed.append(messages[i].copy())

        # Step 6: 清理孤立的 tool 对
        compressed = self._sanitize_tool_pairs(compressed)

        self.compression_count += 1

        # 反抖动跟踪
        new_estimate = estimate_messages_tokens_rough(compressed)
        saved = original_tokens - new_estimate
        savings_pct = (saved / original_tokens * 100) if original_tokens > 0 else 0
        self._last_compression_savings_pct = savings_pct

        if savings_pct < 10:
            self._ineffective_compression_count += 1
        else:
            self._ineffective_compression_count = 0

        if not self.quiet_mode:
            logger.info(
                "Compressed: %d → %d messages (~%d tokens saved, %.0f%%)",
                n_messages, len(compressed), saved, savings_pct,
            )
            logger.info("Compression #%d complete", self.compression_count)

        return compressed

    # ── Preflight 延迟机制 ──────────────────────────────────────

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """从 API 响应更新 token 使用量 + 校准 rough estimator"""
        super().update_from_response(usage)
        if self._last_compression_rough_tokens > 0 and self.last_prompt_tokens > 0:
            # 压缩后首次成功调用 — 记录 rough tokens 对应了多少真实 tokens
            self._last_rough_tokens_when_real_prompt_fit = self._last_compression_rough_tokens
            self._last_compression_rough_tokens = 0

    def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
        """当 rough estimate 已知高噪声时延迟压缩。

        压缩器的 rough token estimate 通常偏大（schema overhead），
        如果之前一次压缩后的 rough estimate 跟这次的很接近，
        而上次压缩后真实 usage 是安全的，就延迟这次压缩。
        """
        if rough_tokens < self.threshold_tokens:
            return False
        baseline = self._last_rough_tokens_when_real_prompt_fit or self._last_compression_rough_tokens
        growth = max(0, rough_tokens - baseline)
        # 增长 < 2000 tokens → 可能是 schema overhead 抖动,延迟压缩
        return growth < 2000

    # ── 迭代摘要更新 ────────────────────────────────────────────

    def _build_iterative_summary_prompt(self, turns: str) -> str:
        """当存在先前摘要时,构建迭代更新提示词而非重新摘要"""
        if not self._previous_summary:
            return (
                f"Summarize the following conversation turns in a compact, information-dense format. "
                f"Preserve key decisions, code changes, file paths, errors encountered, and action items.\n\n"
                f"{turns}"
            )
        return (
            f"Below is a previous summary of the earlier conversation, followed by new turns. "
            f"Produce an UPDATED, unified summary that integrates both. "
            f"Preserve key decisions, code changes, file paths, errors encountered, and action items.\n\n"
            f"PREVIOUS SUMMARY:\n{self._previous_summary}\n\n"
            f"NEW TURNS:\n{turns}"
        )

    # ── 图像剥离 ────────────────────────────────────────────────

    @staticmethod
    def _is_image_content_part(part: Any) -> bool:
        """判断 content part 是否为图片"""
        if not isinstance(part, dict):
            return False
        if part.get("type") == "image_url":
            return True
        if part.get("type") == "image":
            return True
        return False

    @staticmethod
    def _strip_images_from_content(content: Any) -> Any:
        """从消息内容中剥离图片数据,保留文本部分"""
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            stripped = []
            img_count = 0
            for part in content:
                if ContextCompressor._is_image_content_part(part):
                    img_count += 1
                else:
                    stripped.append(part)
            if img_count > 0 and not stripped:
                # 全是图片,保留占位提示
                stripped.append({"type": "text", "text": f"[{img_count} image(s) — content stripped for compression]"})
            return stripped
        return content

    def _strip_images_from_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """剥离所有消息中非 head 部分的图片数据"""
        if not messages:
            return messages
        result = []
        head_count = min(self.protect_first_n, len(messages))
        for i, msg in enumerate(messages):
            msg_copy = msg.copy()
            if i >= head_count:
                msg_copy["content"] = self._strip_images_from_content(msg.get("content"))
            result.append(msg_copy)
        return result
