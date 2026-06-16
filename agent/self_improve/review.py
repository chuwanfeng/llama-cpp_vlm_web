"""
Background Review Fork — 移植自 hermes-agent/run_agent.py _spawn_background_review()

在每次主对话结束后，后台线程启动 review agent:
- 工具集限制为 ["memory", "skills"]
- 扫描对话历史，发现模式/偏好/错误
- 自动创建/更新 skill，标记 agent_created=True
- 自动更新记忆
- 结果推送给用户: 💾 Self-improvement review: ...

与 hermes-agent 的差异:
- 使用 AgentLoop 替代 AIAgent
- 不使用 _set_approval_callback（没有 CLI 权限系统）
- 通过 backend_type + vendor_id 指定 review agent 使用的模型
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from agent.self_improve.provenance import (
    set_current_write_origin,
    reset_current_write_origin,
    BACKGROUND_REVIEW,
)
from tools.registry import get_registry

logger = logging.getLogger(__name__)

# ── Review prompts ──────────────────────────────────────────────────────────

_MEMORY_REVIEW_PROMPT = (
    "Review the conversation above and consider saving to memory if appropriate.\n\n"
    "What to save:\n"
    "- User preferences, habits, and recurring patterns\n"
    "- Important decisions, project context, and technical details\n"
    "- Lessons learned, mistakes to avoid, and successful approaches\n"
    "- Personal information the user has shared (name, role, location, etc.)\n\n"
    "Only write to memory when you discover genuinely new or important information. "
    "If nothing stands out, say 'Nothing to save.' and stop."
)

_SKILL_REVIEW_PROMPT = (
    "Review the conversation above and update the skill library. Be "
    "selective — only create or update a skill when the conversation "
    "reveals a new pattern worth reusing, a workflow worth documenting, "
    "or a lesson worth encoding permanently.\n\n"
    "If nothing stands out, say 'Nothing to save.' and stop."
)

_COMBINED_REVIEW_PROMPT = (
    "Review the conversation above and update two things:\n\n"
    "1. MEMORY — save user preferences, project context, decisions, "
    "lessons learned, or personal info.\n\n"
    "2. SKILLS — create or update skills when the conversation reveals "
    "a new reusable pattern, workflow, methodology, or lesson.\n\n"
    "Be selective. If nothing stands out on either, say 'Nothing to save.' "
    "and stop — but don't reach for that conclusion as a default."
)


# ── Data structures ────────────────────────────────────────────────────────


@dataclass
class ReviewResult:
    """后台 review 执行结果。"""
    actions: List[str] = field(default_factory=list)
    summary: str = ""
    error: Optional[str] = None


# ── Helper: summarize review agent actions ─────────────────────────────────


def _summarize_review_actions(
    review_messages: List[Dict],
    prior_snapshot: List[Dict],
) -> List[str]:
    """从 review agent 的 session messages 中提取成功动作。

    移植自 hermes-agent run_agent.py _summarize_background_review_actions()
    
    跳过 prior_snapshot 中已存在的消息（review agent 继承了对话历史）。
    """
    existing_tool_call_ids = set()
    existing_tool_contents = set()
    for prior in prior_snapshot or []:
        if not isinstance(prior, dict) or prior.get("role") != "tool":
            continue
        tcid = prior.get("tool_call_id")
        if tcid:
            existing_tool_call_ids.add(tcid)
        else:
            content = prior.get("content")
            if isinstance(content, str):
                existing_tool_contents.add(content)

    actions: List[str] = []
    for msg in review_messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        if tcid and tcid in existing_tool_call_ids:
            continue
        if not tcid:
            content_str = msg.get("content")
            if isinstance(content_str, str) and content_str in existing_tool_contents:
                continue
        try:
            data = json.loads(msg.get("content", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict) or not data.get("success"):
            continue
        message = data.get("message", "")
        target = data.get("target", "")
        if "created" in message.lower():
            actions.append(message)
        elif "updated" in message.lower():
            actions.append(message)
        elif "added" in message.lower() or (target and "add" in message.lower()):
            label = "Memory" if target == "memory" else target
            actions.append(f"{label} updated")
        elif "Entry added" in message:
            label = "Memory" if target == "memory" else target
            actions.append(f"{label} updated")
        elif "removed" in message.lower() or "replaced" in message.lower():
            label = "Memory" if target == "memory" else target
            actions.append(f"{label} updated")
    return actions


# ── Review fork spawning ───────────────────────────────────────────────────


def spawn_background_review(
    messages_snapshot: List[Dict],
    backend_type: str = "vendor",
    vendor_id: str = "deepseek",
    model: str = None,
    review_memory: bool = False,
    review_skills: bool = False,
    on_result: Optional[Callable[[ReviewResult], None]] = None,
) -> threading.Thread:
    """启动后台 review 线程。

    移植自 hermes-agent/run_agent.py AIAgent._spawn_background_review()

    Args:
        messages_snapshot: 完整对话历史副本
        backend_type: review agent 使用的后端类型
        vendor_id: 厂商 ID
        model: 模型名
        review_memory: 是否触发 memory review
        review_skills: 是否触发 skill review
        on_result: 结果回调

    Returns:
        daemon thread (已经在运行)
    """
    # 选择 prompt
    if review_memory and review_skills:
        prompt = _COMBINED_REVIEW_PROMPT
    elif review_memory:
        prompt = _MEMORY_REVIEW_PROMPT
    else:
        prompt = _SKILL_REVIEW_PROMPT

    def _run_review():
        result = ReviewResult()
        token = set_current_write_origin(BACKGROUND_REVIEW)

        try:
            # 获取工具 schemas（只暴露 memory 和 skills）
            registry = get_registry()
            all_schemas = registry.get_schemas()
            # 过滤只保留 memory + skills 工具集
            limited_schemas = []
            valid_names = set()
            for entry in registry.list_available():
                if entry.toolset in ("memory", "skills"):
                    limited_schemas.append(entry.to_openai_schema())
                    valid_names.add(entry.name)

            if not limited_schemas:
                result.error = "no memory/skills tools available for review"
                logger.warning(result.error)
                return

            # 运行 AgentLoop 进行 review
            from agent.loop import AgentLoop

            loop = AgentLoop(
                backend_type=backend_type,
                vendor_id=vendor_id,
                model=model,
                tool_schemas=limited_schemas,
                valid_tool_names=valid_names,
                max_turns=8,
            )

            # 构建 review messages: 对话历史 + review prompt
            review_messages = list(messages_snapshot or [])
            review_messages.append({"role": "user", "content": prompt})

            async def _async_review():
                return await loop.run(review_messages)

            review_result = asyncio.run(_async_review())

            # 收集 review agent 的消息
            review_msgs = review_result.messages or []
            actions = _summarize_review_actions(review_msgs, messages_snapshot)

            result.actions = actions
            if actions:
                result.summary = " · ".join(dict.fromkeys(actions))
            else:
                result.summary = "nothing new to save"
            logger.info("Background review complete: %s", result.summary)

        except Exception as e:
            logger.warning("Background review failed: %s", e, exc_info=True)
            result.error = str(e)
            result.summary = f"review error: {e}"
        finally:
            reset_current_write_origin(token)

        if on_result:
            try:
                on_result(result)
            except Exception:
                pass

    t = threading.Thread(target=_run_review, daemon=True, name="bg-review")
    t.start()
    return t


def run_review_sync(
    messages_snapshot: List[Dict],
    backend_type: str = "vendor",
    vendor_id: str = "deepseek",
    model: str = None,
    review_memory: bool = False,
    review_skills: bool = False,
    api_key: str = "",
    base_url: str = None,
) -> ReviewResult:
    """同步运行后台 review（阻塞当前线程直到完成）。
    
    与 spawn_background_review 的区别：
    - 不创建新线程，在当前线程中同步执行
    - 返回 ReviewResult 而不是 Thread
    - 适合在端点线程中直接调用
    
    注意：调用者必须确保不在 asyncio 事件循环内调用此函数。
    （它内部会创建新的事件循环）
    """
    if review_memory and review_skills:
        prompt = _COMBINED_REVIEW_PROMPT
    elif review_memory:
        prompt = _MEMORY_REVIEW_PROMPT
    else:
        prompt = _SKILL_REVIEW_PROMPT

    result = ReviewResult()
    token = set_current_write_origin(BACKGROUND_REVIEW)

    try:
        registry = get_registry()
        limited_schemas = []
        valid_names: Set[str] = set()
        for entry in registry.list_available():
            # review 代理需要 memory + skills + file（write_file/read_file）工具
            if entry.toolset in ("memory", "skills", "file"):
                limited_schemas.append(entry.to_openai_schema())
                valid_names.add(entry.name)

        if not limited_schemas:
            result.error = "no memory/skills tools available for review"
            logger.warning(result.error)
            return result

        from agent.loop import AgentLoop

        loop = AgentLoop(
            backend_type=backend_type,
            vendor_id=vendor_id,
            model=model,
            tool_schemas=limited_schemas,
            valid_tool_names=valid_names,
            max_turns=8,
            api_key=api_key,
            base_url=base_url,
        )

        review_messages = list(messages_snapshot or [])
        review_messages.append({"role": "user", "content": prompt})

        async def _async_review():
            return await loop.run(review_messages)

        review_run_result = asyncio.run(_async_review())

        review_msgs = review_run_result.messages or []
        actions = _summarize_review_actions(review_msgs, messages_snapshot)

        result.actions = actions
        if actions:
            result.summary = " · ".join(dict.fromkeys(actions))
        else:
            result.summary = "nothing new to save"
        logger.info("Sync review complete: %s", result.summary)

    except Exception as e:
        logger.warning("Sync review failed: %s", e, exc_info=True)
        result.error = str(e)
        result.summary = f"review error: {e}"
    finally:
        reset_current_write_origin(token)

    return result