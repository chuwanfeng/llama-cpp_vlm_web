"""
builtin_evolution.py — 自我进化工具
=================================

将 agent/self_improve/curator 模块的 Skill 生命周期管理暴露为 Agent 可调用工具。

工具列表:
  - skill_evolve: 触发 Curator 审查，自动合并相似技能、归档过期技能
  - curator_status: 查看 Curator 当前状态（暂停/运行记录/技能统计）

移植自 hermes-agent 的 curator + skill_evolution 机制。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)


# =============================================================================
# 工具参数 Schema（JSON Schema for parameters, NOT full OpenAI tool format）
# =============================================================================

SKILL_EVOLVE_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "force": {
            "type": "boolean",
            "description": "是否强制执行（跳过时间间隔检查）。默认 false，仅当距上次运行超过 1 小时时才执行。",
            "default": False,
        },
        "dry_run": {
            "type": "boolean",
            "description": "仅分析不执行，返回建议但不修改任何文件。",
            "default": False,
        },
    },
    "required": [],
}

CURATOR_STATUS_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
}


# =============================================================================
# 工具实现
# =============================================================================


def skill_evolve(force: bool = False, dry_run: bool = False) -> Dict[str, Any]:
    """触发技能自我进化审查。

    调用 curator 模块执行技能生命周期管理。
    如果 dry_run=True，只分析不修改。

    Args:
        force: 跳过时间间隔检查
        dry_run: 仅分析，不执行修改

    Returns:
        {"success": bool, "message": str, "report_id": str|None, "changes": list}
    """
    try:
        from agent.self_improve.curator import (
            maybe_run_curator,
            is_paused,
            _load_state,
            apply_automatic_transitions,
        )

        if is_paused():
            return {"success": False, "message": "Curator 已暂停。先调用 curator_status 查看详情。"}

        if dry_run:
            # 只执行自动转换（安全操作）+ 报告当前状态
            try:
                trans_result = apply_automatic_transitions()
            except Exception as e:
                trans_result = {"error": str(e)}

            state = _load_state()

            # 收集建议
            suggestions: List[str] = []
            if trans_result:
                suggestions.append(f"自动转换: {trans_result}")

            return {
                "success": True,
                "message": f"Dry run 完成。{len(suggestions)} 条建议。",
                "report_id": None,
                "changes": suggestions,
                "state": {
                    "last_run": state.get("last_run", "从未"),
                    "paused": is_paused(),
                    "stats": state.get("stats", {}),
                },
            }

        # 强制执行
        result = maybe_run_curator(force=force)

        return {
            "success": True,
            "message": result.get("summary", "审查完成"),
            "report_id": result.get("report_id"),
            "changes": result.get("actions", []),
            "stats": result.get("stats", {}),
        }

    except ImportError as e:
        logger.warning("skill_evolve: curator 模块不可用: %s", e)
        return {"success": False, "message": f"Curator 模块未加载: {e}"}
    except Exception as e:
        logger.error("skill_evolve 失败: %s", e, exc_info=True)
        return {"success": False, "message": str(e)}


def curator_status() -> Dict[str, Any]:
    """查看 Curator 当前状态。

    Returns:
        {"success": bool, "paused": bool, "last_run": str, "next_run": str, "stats": {}, ...}
    """
    try:
        from agent.self_improve.curator import (
            is_paused,
            _load_state,
            should_run_now,
        )

        state = _load_state()
        paused = is_paused()
        last_run = state.get("last_run", "从未运行")
        stats = state.get("stats", {})

        # 收集各状态技能数量
        try:
            from services.skill_loader import list_skills_meta
            meta = list_skills_meta()
            active_count = sum(1 for m in meta if m.get("status", "active") == "active")
            stale_count = sum(1 for m in meta if m.get("status") == "stale")
            archived_count = sum(1 for m in meta if m.get("status") == "archived")
            pinned_count = sum(1 for m in meta if m.get("status") == "pinned")
            total_count = len(meta)
        except Exception:
            active_count = stale_count = archived_count = pinned_count = total_count = -1

        ready = should_run_now() if not paused else False

        return {
            "success": True,
            "paused": paused,
            "ready_to_run": ready,
            "last_run": last_run,
            "skills": {
                "total": total_count,
                "active": active_count,
                "stale": stale_count,
                "archived": archived_count,
                "pinned": pinned_count,
            },
            "stats": stats,
        }

    except ImportError as e:
        logger.warning("curator_status: 模块不可用: %s", e)
        return {"success": False, "message": f"Curator 模块未加载: {e}"}
    except Exception as e:
        logger.error("curator_status 失败: %s", e, exc_info=True)
        return {"success": False, "message": str(e)}


# =============================================================================
# 注册
# =============================================================================

registry.register(
    name="skill_evolve",
    toolset="evolution",
    schema=SKILL_EVOLVE_PARAMS,
    handler=skill_evolve,
)

registry.register(
    name="curator_status",
    toolset="evolution",
    schema=CURATOR_STATUS_PARAMS,
    handler=curator_status,
)
