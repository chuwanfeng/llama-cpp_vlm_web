"""
技能管理工具 — 创建、更新、删除、进化技能。

从 hermes-agent/tools/skill_commands.py + agent/skill_commands.py 移植，
适配 llama-cpp_vlm_web 项目。

核心功能：
    - skill_create: 创建新技能（.skill 文件）
    - skill_update: 更新现有技能
    - skill_delete: 删除技能
    - skill_evolve: 基于对话历史自动改进技能（自我进化）

技能文件格式（.skill）：
    ---
    name: my-skill
    description: 简短描述
    priority: 10
    tools:
      - read_file
      - run_terminal
    ---
    # 技能标题
    完整指令内容...
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "skills"

# 技能名安全检查
_SKILL_INVALID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_SKILL_MULTI_HYPHEN = re.compile(r"-{2,}")


def _sanitize_skill_name(name: str) -> str:
    """将技能名清理为安全的文件名格式。

    Args:
        name: 原始技能名

    Returns:
        清理后的技能名（仅含字母、数字、连字符、下划线）
    """
    name = name.lower().strip().replace(" ", "-")
    name = _SKILL_INVALID_CHARS.sub("", name)
    name = _SKILL_MULTI_HYPHEN.sub("-", name)
    return name.strip("-")


def _is_skill_name_safe(name: str) -> bool:
    """检查技能名是否安全（只含合法字符）。"""
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", name))


def _write_skill_file(name: str, description: str, content: str, priority: int = 0,
                      tools: Optional[List[str]] = None, agent_created: bool = False) -> Path:
    """写入技能文件到磁盘。

    Args:
        name: 技能名
        description: 技能描述
        content: 技能正文内容
        priority: 优先级（越高越优先）
        tools: 技能需要的工具列表
        agent_created: 是否由 agent 自动创建

    Returns:
        写入的文件路径
    """
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    filepath = SKILLS_DIR / f"{name}.skill"

    # 构建 frontmatter
    lines = ["---"]
    lines.append(f"name: {name}")
    lines.append(f"description: {description}")
    if priority:
        lines.append(f"priority: {priority}")
    if tools:
        lines.append("tools:")
        for tool in tools:
            lines.append(f"  - {tool}")
    if agent_created:
        lines.append("agent_created: true")
    lines.append("---")
    lines.append("")
    lines.append(content.strip())

    filepath.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Skill file written: %s", filepath)
    return filepath


def skill_create(name: str, description: str, content: str, priority: int = 0,
                 tools: Optional[List[str]] = None, **kwargs) -> str:
    """
    创建新技能文件。

    参数:
        name: 技能名称（仅字母、数字、连字符、下划线）
        description: 技能描述（最多 1024 字符）
        content: 技能的完整指令内容（Markdown 格式）
        priority: 优先级，数字越大越优先（默认 0）
        tools: 此技能需要的工具列表（如 ["read_file", "run_terminal"]）

    返回:
        创建结果消息
    """
    if not name or not str(name).strip():
        return tool_error("技能名称不能为空")

    name = _sanitize_skill_name(name)
    if not _is_skill_name_safe(name):
        return tool_error(f"技能名称包含非法字符: {name}")

    # 检查是否已存在
    filepath = SKILLS_DIR / f"{name}.skill"
    if filepath.exists():
        return tool_error(f"技能 '{name}' 已存在。使用 skill_update 更新或先删除。")

    # 检查来源（自我进化标记）
    agent_created = kwargs.get("agent_created", False)
    if not agent_created:
        # 检查 provenance
        try:
            from agent.self_improve.provenance import is_agent_created
            agent_created = is_agent_created()
        except ImportError:
            pass

    try:
        _write_skill_file(
            name=name,
            description=description[:1024],
            content=content,
            priority=priority,
            tools=tools or [],
            agent_created=agent_created,
        )

        origin = " (agent-created)" if agent_created else ""
        return json.dumps({
            "status": "created",
            "name": name,
            "path": str(filepath),
            "description": description[:1024],
            "priority": priority,
            "tools": tools or [],
            "agent_created": agent_created,
            "message": f"技能 '{name}' 创建成功{origin}。使用 skill_view 查看完整内容。"
        }, ensure_ascii=False)
    except Exception as e:
        logger.exception("Failed to create skill: %s", e)
        return tool_error(f"创建技能失败: {e}")


def skill_update(name: str, description: str = None, content: str = None,
                 priority: int = None, tools: List[str] = None) -> str:
    """
    更新现有技能。

    参数:
        name: 要更新的技能名称
        description: 新描述（可选，不填则保持原值）
        content: 新内容（可选，不填则保持原值）
        priority: 新优先级（可选）
        tools: 新工具列表（可选）

    返回:
        更新结果消息
    """
    if not name or not str(name).strip():
        return tool_error("请指定要更新的技能名称")

    name = _sanitize_skill_name(name)
    filepath = SKILLS_DIR / f"{name}.skill"

    if not filepath.exists():
        return tool_error(f"技能 '{name}' 不存在。使用 skill_create 创建新技能。")

    try:
        # 读取现有内容
        from tools.builtin_skills import _load_skill
        existing = _load_skill(filepath)
        if not existing:
            return tool_error(f"无法读取技能: {name}")

        # 使用新值或保持原值
        new_description = description if description is not None else existing.get("description", "")
        new_content = content if content is not None else existing.get("content", "")
        new_priority = priority if priority is not None else existing.get("priority", 0)
        new_tools = tools if tools is not None else existing.get("tools", [])

        # 保留 agent_created 标记
        agent_created = existing.get("agent_created", False)
        raw = existing.get("raw_content", "")
        if "agent_created: true" in raw:
            agent_created = True

        _write_skill_file(
            name=name,
            description=new_description,
            content=new_content,
            priority=new_priority,
            tools=new_tools,
            agent_created=agent_created,
        )

        return json.dumps({
            "status": "updated",
            "name": name,
            "path": str(filepath),
            "message": f"技能 '{name}' 更新成功。"
        }, ensure_ascii=False)
    except Exception as e:
        logger.exception("Failed to update skill: %s", e)
        return tool_error(f"更新技能失败: {e}")


def skill_delete(name: str) -> str:
    """
    删除技能文件。

    参数:
        name: 要删除的技能名称

    返回:
        删除结果消息
    """
    if not name or not str(name).strip():
        return tool_error("请指定要删除的技能名称")

    name = _sanitize_skill_name(name)
    filepath = SKILLS_DIR / f"{name}.skill"

    if not filepath.exists():
        return tool_error(f"技能 '{name}' 不存在。")

    try:
        # 读取确认 agent_created 标记
        from tools.builtin_skills import _load_skill
        existing = _load_skill(filepath)
        agent_created = existing.get("agent_created", False)
        raw = existing.get("raw_content", "")
        if "agent_created: true" in raw:
            agent_created = True

        # 用户创建的技能需要确认
        if not agent_created:
            return tool_error(
                f"技能 '{name}' 是用户创建的技能，不能自动删除。"
                f"如需删除，请手动删除文件: {filepath}"
            )

        filepath.unlink()
        return json.dumps({
            "status": "deleted",
            "name": name,
            "message": f"技能 '{name}' 已删除。"
        }, ensure_ascii=False)
    except Exception as e:
        logger.exception("Failed to delete skill: %s", e)
        return tool_error(f"删除技能失败: {e}")


def skill_evolve(skill_name: str, observation: str, parent_agent=None) -> str:
    """
    基于观察到的模式改进现有技能（自我进化）。

    由 review agent 调用，基于对话历史中的模式自动改进技能。
    只能改进 agent_created=True 的技能。

    参数:
        skill_name: 要改进的技能名称
        observation: 观察到的模式或改进建议
        parent_agent: 父代理引用（用于获取上下文）

    返回:
        进化结果消息
    """
    if not skill_name or not str(skill_name).strip():
        return tool_error("请指定要进化的技能名称")

    name = _sanitize_skill_name(skill_name)
    filepath = SKILLS_DIR / f"{name}.skill"

    if not filepath.exists():
        return tool_error(f"技能 '{name}' 不存在。")

    try:
        from tools.builtin_skills import _load_skill
        existing = _load_skill(filepath)
        if not existing:
            return tool_error(f"无法读取技能: {name}")

        # 检查是否为 agent_created
        agent_created = existing.get("agent_created", False)
        raw = existing.get("raw_content", "")
        if "agent_created: true" in raw:
            agent_created = True

        if not agent_created:
            return tool_error(
                f"技能 '{name}' 是用户创建的，不能自动进化。"
                f"使用 skill_update 手动更新。"
            )

        # 基于观察改进内容
        current_content = existing.get("content", "")
        current_description = existing.get("description", "")

        # 简单的内容追加策略（实际可由 LLM 生成更智能的改进）
        improved_content = (
            f"{current_content}\n\n"
            f"## 自动改进（基于观察）\n\n"
            f"{observation}\n"
        )

        _write_skill_file(
            name=name,
            description=current_description,
            content=improved_content,
            priority=existing.get("priority", 0),
            tools=existing.get("tools", []),
            agent_created=True,
        )

        return json.dumps({
            "status": "evolved",
            "name": name,
            "observation": observation,
            "message": f"技能 '{name}' 已基于观察自动改进。"
        }, ensure_ascii=False)
    except Exception as e:
        logger.exception("Failed to evolve skill: %s", e)
        return tool_error(f"进化技能失败: {e}")


# =============================================================================
# 注册工具
# =============================================================================

import json

SKILL_CREATE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "技能名称（仅字母、数字、连字符、下划线）",
        },
        "description": {
            "type": "string",
            "description": "技能描述（最多 1024 字符）",
        },
        "content": {
            "type": "string",
            "description": "技能的完整指令内容（Markdown 格式）",
        },
        "priority": {
            "type": "integer",
            "description": "优先级，数字越大越优先",
            "default": 0,
        },
        "tools": {
            "type": "array",
            "items": {"type": "string"},
            "description": "此技能需要的工具列表",
        },
    },
    "required": ["name", "description", "content"],
}

SKILL_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "要更新的技能名称",
        },
        "description": {
            "type": "string",
            "description": "新描述（可选）",
        },
        "content": {
            "type": "string",
            "description": "新内容（可选）",
        },
        "priority": {
            "type": "integer",
            "description": "新优先级（可选）",
        },
        "tools": {
            "type": "array",
            "items": {"type": "string"},
            "description": "新工具列表（可选）",
        },
    },
    "required": ["name"],
}

SKILL_DELETE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "要删除的技能名称",
        },
    },
    "required": ["name"],
}

SKILL_EVOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "skill_name": {
            "type": "string",
            "description": "要改进的技能名称",
        },
        "observation": {
            "type": "string",
            "description": "观察到的模式或改进建议",
        },
    },
    "required": ["skill_name", "observation"],
}


def _skill_create_handler(args, **kw):
    return skill_create(**args)


def _skill_update_handler(args, **kw):
    return skill_update(**args)


def _skill_delete_handler(args, **kw):
    return skill_delete(**args)


def _skill_evolve_handler(args, **kw):
    # 注入 parent_agent（如果 loop.py 传递了）
    if "parent_agent" in kw:
        args["parent_agent"] = kw["parent_agent"]
    return skill_evolve(**args)


registry.register(
    name="skill_create",
    toolset="skills",
    schema=SKILL_CREATE_SCHEMA,
    handler=_skill_create_handler,
)

registry.register(
    name="skill_update",
    toolset="skills",
    schema=SKILL_UPDATE_SCHEMA,
    handler=_skill_update_handler,
)

registry.register(
    name="skill_delete",
    toolset="skills",
    schema=SKILL_DELETE_SCHEMA,
    handler=_skill_delete_handler,
)

registry.register(
    name="skill_evolve",
    toolset="skills",
    schema=SKILL_EVOLVE_SCHEMA,
    handler=_skill_evolve_handler,
)
