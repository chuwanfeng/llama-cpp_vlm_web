"""任务委托工具（delegate_task）。

移植自 hermes-agent/tools/delegate_tool.py。

核心功能：
  - 创建子代理（SubAgent）执行子任务
  - 支持并发多个子代理（通过 tasks 列表）
  - 自动收集结果并序列化返回
  - 迭代预算控制

使用方式（由 AgentLoop 自动调用）：
    # LLM 发出 delegate_task 工具调用时，AgentLoop 调用此函数
    result = delegate_task(
        goal="分析 performance.py 的性能瓶颈",
        context="项目是一个 Web 应用...",
        toolsets=["code_review"],
        max_iterations=20,
    )
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from tools.registry import registry
from utils import get_logger

logger = get_logger("tools.delegate_tool")


# ─── Schema ─────────────────────────────────────────────────────

DELEGATE_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {
            "type": "string",
            "description": "The goal or task for the sub-agent to accomplish. Be specific and clear about the expected deliverable.",
        },
        "context": {
            "type": "string",
            "description": "Background context the sub-agent needs to understand the task. Include file paths, relevant config, etc.",
        },
        "toolsets": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of toolset names to grant the sub-agent (e.g. ['code_review', 'web_search']). If omitted, all tools are available.",
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "context": {"type": "string"},
                    "toolsets": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "max_iterations": {"type": "integer", "default": 50},
                },
                "required": ["goal"],
            },
            "description": "Multiple tasks to run concurrently. When this is set, goal/context/toolsets on the top level are ignored. Max 5 concurrent tasks.",
        },
        "max_iterations": {
            "type": "integer",
            "description": "Maximum tool-calling iterations for the sub-agent (default: 50).",
        },
    },
    "required": ["goal"],
}

_MAX_CONCURRENT_CHILDREN = 5


# ─── 主函数 ─────────────────────────────────────────────────────


async def delegate_task(
    goal: str = "",
    context: str = "",
    toolsets: List[str] = None,
    tasks: List[Dict[str, Any]] = None,
    max_iterations: int = 50,
    acp_command: str = "",
    acp_args: Dict[str, Any] = None,
    role: str = "",
    parent_agent=None,
) -> str:
    """委托任务给子代理执行。

    这是 delegate_task 工具的主入口，由 AgentLoop 的工具调度调用。

    参数:
        goal: 任务目标（自然语言描述）
        context: 上下文信息
        toolsets: 工具集列表
        tasks: 多任务列表（并发执行）
        max_iterations: 最大迭代轮数
        acp_command: ACP 命令（hermes-agent ACP 兼容，当前未实现）
        acp_args: ACP 参数
        role: 子代理角色
        parent_agent: 父代理实例

    返回:
        JSON 字符串，包含执行结果
    """
    # 截断超过最大并发数的批量任务
    if tasks:
        tasks = tasks[:_MAX_CONCURRENT_CHILDREN]

    if tasks and len(tasks) > 0:
        return await _delegate_batch(tasks, parent_agent)
    else:
        return await _delegate_single(goal, context, toolsets, max_iterations, parent_agent)


async def _delegate_single(
    goal: str,
    context: str,
    toolsets: List[str],
    max_iterations: int,
    parent_agent,
) -> str:
    """委托单个任务（异步，直接在事件循环中 await）。"""
    from agent.sub_agent import SubAgent, sub_agent_result_to_json

    start = time.monotonic()

    try:
        sub = SubAgent(
            goal=goal,
            context=context,
            toolsets=toolsets,
            max_iterations=max_iterations,
            parent_agent=parent_agent,
        )

        result = await sub.run()

        logger.info(
            "SubAgent %s completed: success=%s turns=%d duration=%.0fms",
            sub.task_id, result.success, result.turns_used,
            (time.monotonic() - start) * 1000,
        )
        return sub_agent_result_to_json(result)

    except Exception as e:
        logger.exception("SubAgent failed: %s", e)
        return json.dumps({
            "success": False,
            "goal": goal,
            "error": f"{type(e).__name__}: {e}",
            "duration_ms": (time.monotonic() - start) * 1000,
        }, ensure_ascii=False, indent=2)


async def _delegate_batch(
    tasks: List[Dict[str, Any]],
    parent_agent,
) -> str:
    """并发委托多个任务（异步，直接在事件循环中 await）。"""
    from agent.sub_agent import SubAgentCoordinator, sub_agent_result_to_json

    start = time.monotonic()

    try:
        coordinator = SubAgentCoordinator(
            max_concurrent=_MAX_CONCURRENT_CHILDREN,
            parent_agent=parent_agent,
        )

        results = await coordinator.run_all(tasks)

        # 序列化结果
        all_results = [{
            "task_index": i,
            "goal": tasks[i].get("goal", ""),
            "success": r.success,
            "output": r.output,
            "error": r.error,
            "turns_used": r.turns_used,
            "duration_ms": r.duration_ms,
        } for i, r in enumerate(results)]

        logger.info(
            "Batch delegation: %d/%d tasks completed in %.0fms",
            sum(1 for r in results if r.success), len(tasks),
            (time.monotonic() - start) * 1000,
        )

        return json.dumps({
            "success": True,
            "batch": True,
            "total": len(tasks),
            "completed": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "results": all_results,
            "duration_ms": (time.monotonic() - start) * 1000,
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.exception("Batch delegation failed: %s", e)
        return json.dumps({
            "success": False,
            "batch": True,
            "total": len(tasks),
            "error": f"{type(e).__name__}: {e}",
            "duration_ms": (time.monotonic() - start) * 1000,
        }, ensure_ascii=False, indent=2)


# ─── 辅助函数 ──────────────────────────────────────────────────


def get_max_concurrent_children() -> int:
    """获取最大并发子代理数（内部使用）。"""
    return _MAX_CONCURRENT_CHILDREN


def register_delegate_tool(target_registry) -> None:
    """向注册表注册 delegate_task 工具。

    由 tools/__init__.py 在模块加载时调用。
    """
    target_registry.register(
        name="delegate_task",
        description="Delegate a task to a sub-agent that works independently. "
                    "The sub-agent gets its own tool set and works on the assigned goal. "
                    "Use this for parallel work or delegating well-defined subtasks. "
                    "Results are returned when the sub-agent completes.",
        schema=DELEGATE_TASK_SCHEMA,
        handler=delegate_task,
        toolset="delegation",
        emoji="🔄",
    )


# ─── 模块级注册（AST 自动发现用） ──────────────────────────────
# discover_tools() 扫描 tools/ 目录，通过 AST 检测 registry.register() 调用
# 必须是直接的 registry.register(...) 顶层调用，AST 才能检测到
# （register_delegate_tool(registry) 这种封装调用 AST 不认）
registry.register(
    name="delegate_task",
    description="Delegate a task to a sub-agent that works independently. "
                "The sub-agent gets its own tool set and works on the assigned goal. "
                "Use this for parallel work or delegating well-defined subtasks. "
                "Results are returned when the sub-agent completes.",
    schema=DELEGATE_TASK_SCHEMA,
    handler=delegate_task,
    toolset="delegation",
    emoji="🔄",
)
