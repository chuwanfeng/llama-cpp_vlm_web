"""
技能管理工具 — 创建、更新、删除、进化技能。

采用 hermes-agent SKILL.md 目录格式（业界通用标准）：
    skills/
      skill-name/
        SKILL.md        # 技能内容（YAML frontmatter + Markdown）
        references/     # 可选：附加参考文件/脚本

核心功能：
    - skill_create: 创建新技能（创建目录 + SKILL.md）
    - skill_update: 更新现有技能
    - skill_delete: 删除技能目录
    - skill_improve: 基于对话历史自动改进技能（自我进化）
"""

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "skills"

# ── 目录过滤 ───────────────────────────────────────────────────────────
_EXCLUDED_SKILL_DIRS: frozenset = frozenset(
    (".git", ".github", ".hub", ".archive",
     "__pycache__", "node_modules", ".venv", "venv", "env",
     ".env", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache")
)

# ── 工具函数 ───────────────────────────────────────────────────────────
_SKILL_INVALID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_SKILL_MULTI_HYPHEN = re.compile(r"-{2,}")

# YAML 加载缓存
_yaml_load_fn = None


def _yaml_load(content: str):
    global _yaml_load_fn
    if _yaml_load_fn is None:
        import yaml
        loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader

        def _load(value: str):
            return yaml.load(value, Loader=loader)

        _yaml_load_fn = _load
    return _yaml_load_fn(content)


def _sanitize_skill_name(name: str) -> str:
    """清理技能名为安全的目录名。"""
    name = name.lower().strip().replace(" ", "-")
    name = _SKILL_INVALID_CHARS.sub("", name)
    name = _SKILL_MULTI_HYPHEN.sub("-", name)
    return name.strip("-")


def _is_skill_name_safe(name: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", name))


def _discover_skill_files() -> List[Path]:
    """递归扫描 SKILLS_DIR，找到所有 SKILL.md。"""
    if not SKILLS_DIR.exists():
        return []
    matches = []
    for root, dirs, files in os.walk(SKILLS_DIR, followlinks=True):
        dirs[:] = [d for d in dirs if d not in _EXCLUDED_SKILL_DIRS]
        if "SKILL.md" in files:
            matches.append(Path(root) / "SKILL.md")
    return sorted(matches, key=lambda p: str(p.relative_to(SKILLS_DIR)))


def _load_existing_skill(path: Path) -> Dict[str, Any]:
    """读取现有 SKILL.md 并返回解析后的数据。"""
    from tools.builtin_skills import _load_skill
    return _load_skill(path)


def _parse_frontmatter(text: str) -> Dict[str, Any]:
    """快速解析 YAML frontmatter。"""
    if not text.startswith("---"):
        return {}
    end_match = re.search(r"\n---\s*\n", text[3:])
    if not end_match:
        return {}
    yaml_content = text[3 : end_match.start() + 3]
    try:
        parsed = _yaml_load(yaml_content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def _write_skill_md(
    skill_dir: Path,
    name: str,
    description: str,
    content: str,
    priority: int = 0,
    tools: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: str = "",
    agent_created: bool = False,
    tags: Optional[List[str]] = None,
    platforms: Optional[List[str]] = None,
) -> Path:
    """写入 SKILL.md 文件到技能目录。

    Returns:
        SKILL.md 的路径。
    """
    skill_dir.mkdir(parents=True, exist_ok=True)
    filepath = skill_dir / "SKILL.md"

    # 构建 YAML frontmatter
    lines = ["---"]
    lines.append(f"name: {name}")
    lines.append(f"description: {description}")
    if version:
        lines.append(f"version: {version}")
    if author:
        lines.append(f"author: {author}")
    if priority:
        lines.append(f"priority: {priority}")
    if tools:
        lines.append("tools:")
        for tool in tools:
            lines.append(f"  - {tool}")
    if platforms:
        lines.append("platforms:")
        for p in platforms:
            lines.append(f"  - {p}")
    if tags:
        lines.append("metadata:")
        lines.append("  hermes:")
        lines.append("    tags:")
        for tag in tags:
            lines.append(f"      - {tag}")
    if agent_created:
        lines.append("agent_created: true")
    lines.append("---")
    lines.append("")
    lines.append(content.strip())

    filepath.write_text("\n".join(lines), encoding="utf-8")
    logger.info("SKILL.md 已写入: %s", filepath)
    return filepath


# ── 技能 CRUD ──────────────────────────────────────────────────────────

def skill_create(name: str, description: str, content: str, priority: int = 0,
                 tools: Optional[List[str]] = None, **kwargs) -> str:
    """
    创建新技能（hermes-agent 格式：目录 + SKILL.md）。

    参数:
        name: 技能名称（仅字母、数字、连字符、下划线）
        description: 技能描述（最多 1024 字符）
        content: 技能的完整指令内容（Markdown 格式）
        priority: 优先级，数字越大越优先（默认 0）
        tools: 此技能需要的工具列表

    返回:
        创建结果 JSON
    """
    if not name or not str(name).strip():
        return tool_error("技能名称不能为空")

    name = _sanitize_skill_name(name)
    if not _is_skill_name_safe(name):
        return tool_error(f"技能名称包含非法字符: {name}")

    # 检查是否已存在
    skill_dir = SKILLS_DIR / name
    skill_md = skill_dir / "SKILL.md"
    if skill_md.exists():
        return tool_error(f"技能 '{name}' 已存在。使用 skill_update 更新或先删除。")

    # 检查是否为 agent 创建
    agent_created = kwargs.get("agent_created", False)
    if not agent_created:
        try:
            from agent.self_improve.provenance import is_agent_created
            agent_created = is_agent_created()
        except ImportError:
            pass

    try:
        _write_skill_md(
            skill_dir=skill_dir,
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
            "path": str(skill_md),
            "description": description[:1024],
            "priority": priority,
            "tools": tools or [],
            "agent_created": agent_created,
            "message": f"技能 '{name}' 创建成功{origin}。使用 skill_view 查看完整内容。"
        }, ensure_ascii=False)
    except Exception as e:
        logger.exception("创建技能失败: %s", e)
        return tool_error(f"创建技能失败: {e}")


def skill_update(name: str, description: str = None, content: str = None,
                 priority: int = None, tools: List[str] = None) -> str:
    """
    更新现有技能。

    参数:
        name: 要更新的技能名称
        description: 新描述（可选）
        content: 新内容（可选）
        priority: 新优先级（可选）
        tools: 新工具列表（可选）

    返回:
        更新结果 JSON
    """
    if not name or not str(name).strip():
        return tool_error("请指定要更新的技能名称")

    name = _sanitize_skill_name(name)
    skill_dir = SKILLS_DIR / name
    skill_md = skill_dir / "SKILL.md"

    if not skill_md.exists():
        return tool_error(f"技能 '{name}' 不存在。使用 skill_create 创建新技能。")

    try:
        existing = _load_existing_skill(skill_md)
        if not existing:
            return tool_error(f"无法读取技能: {name}")

        new_description = description if description is not None else existing.get("description", "")
        new_content = content if content is not None else existing.get("content", "")
        new_priority = priority if priority is not None else existing.get("priority", 0)
        new_tools = tools if tools is not None else existing.get("tools", [])

        agent_created = existing.get("agent_created", False)
        raw = existing.get("raw_content", "")
        if "agent_created: true" in raw:
            agent_created = True

        _write_skill_md(
            skill_dir=skill_dir,
            name=name,
            description=new_description,
            content=new_content,
            priority=new_priority,
            tools=new_tools,
            version=existing.get("version", "1.0.0"),
            author=existing.get("author", ""),
            agent_created=agent_created,
            tags=existing.get("tags", []),
            platforms=existing.get("platforms", []),
        )

        return json.dumps({
            "status": "updated",
            "name": name,
            "path": str(skill_md),
            "message": f"技能 '{name}' 更新成功。"
        }, ensure_ascii=False)
    except Exception as e:
        logger.exception("更新技能失败: %s", e)
        return tool_error(f"更新技能失败: {e}")


def skill_delete(name: str) -> str:
    """
    删除技能目录。

    参数:
        name: 要删除的技能名称

    返回:
        删除结果 JSON
    """
    if not name or not str(name).strip():
        return tool_error("请指定要删除的技能名称")

    name = _sanitize_skill_name(name)
    skill_dir = SKILLS_DIR / name

    if not skill_dir.exists() or not (skill_dir / "SKILL.md").exists():
        return tool_error(f"技能 '{name}' 不存在。")

    try:
        existing = _load_existing_skill(skill_dir / "SKILL.md")
        agent_created = existing.get("agent_created", False)
        raw = existing.get("raw_content", "")
        if "agent_created: true" in raw:
            agent_created = True

        if not agent_created:
            return tool_error(
                f"技能 '{name}' 是用户创建的技能，不能自动删除。"
                f"如需删除，请手动删除目录: {skill_dir}"
            )

        shutil.rmtree(str(skill_dir))
        return json.dumps({
            "status": "deleted",
            "name": name,
            "message": f"技能 '{name}' 已删除。"
        }, ensure_ascii=False)
    except Exception as e:
        logger.exception("删除技能失败: %s", e)
        return tool_error(f"删除技能失败: {e}")


def skill_improve(skill_name: str, observation: str, parent_agent=None) -> str:
    """
    基于观察到的模式改进现有技能（自我进化）。

    只能改进 agent_created=True 的技能。

    参数:
        skill_name: 要改进的技能名称
        observation: 观察到的模式或改进建议
        parent_agent: 父代理引用

    返回:
        进化结果 JSON
    """
    if not skill_name or not str(skill_name).strip():
        return tool_error("请指定要进化的技能名称")

    name = _sanitize_skill_name(skill_name)
    skill_dir = SKILLS_DIR / name
    skill_md = skill_dir / "SKILL.md"

    if not skill_md.exists():
        return tool_error(f"技能 '{name}' 不存在。")

    try:
        existing = _load_existing_skill(skill_md)
        if not existing:
            return tool_error(f"无法读取技能: {name}")

        agent_created = existing.get("agent_created", False)
        raw = existing.get("raw_content", "")
        if "agent_created: true" in raw:
            agent_created = True

        if not agent_created:
            return tool_error(
                f"技能 '{name}' 是用户创建的，不能自动进化。"
                f"使用 skill_update 手动更新。"
            )

        current_content = existing.get("content", "")
        current_description = existing.get("description", "")

        improved_content = (
            f"{current_content}\n\n"
            f"## 自动改进（基于观察）\n\n"
            f"{observation}\n"
        )

        _write_skill_md(
            skill_dir=skill_dir,
            name=name,
            description=current_description,
            content=improved_content,
            priority=existing.get("priority", 0),
            tools=existing.get("tools", []),
            version=existing.get("version", "1.0.0"),
            author=existing.get("author", ""),
            agent_created=True,
            tags=existing.get("tags", []),
            platforms=existing.get("platforms", []),
        )

        return json.dumps({
            "status": "evolved",
            "name": name,
            "observation": observation,
            "message": f"技能 '{name}' 已基于观察自动改进。"
        }, ensure_ascii=False)
    except Exception as e:
        logger.exception("进化技能失败: %s", e)
        return tool_error(f"进化技能失败: {e}")


# ── Schema 定义 ────────────────────────────────────────────────────────

SKILL_CREATE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "技能名称（仅字母、数字、连字符、下划线）"},
        "description": {"type": "string", "description": "技能描述（最多 1024 字符）"},
        "content": {"type": "string", "description": "技能的完整指令内容（Markdown 格式）"},
        "priority": {"type": "integer", "description": "优先级，数字越大越优先", "default": 0},
        "tools": {"type": "array", "items": {"type": "string"}, "description": "此技能需要的工具列表"},
    },
    "required": ["name", "description", "content"],
}

SKILL_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "要更新的技能名称"},
        "description": {"type": "string", "description": "新描述（可选）"},
        "content": {"type": "string", "description": "新内容（可选）"},
        "priority": {"type": "integer", "description": "新优先级（可选）"},
        "tools": {"type": "array", "items": {"type": "string"}, "description": "新工具列表（可选）"},
    },
    "required": ["name"],
}

SKILL_DELETE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "要删除的技能名称"},
    },
    "required": ["name"],
}

SKILL_IMPROVE_SCHEMA = {
    "type": "object",
    "properties": {
        "skill_name": {"type": "string", "description": "要改进的技能名称"},
        "observation": {"type": "string", "description": "观察到的模式或改进建议"},
    },
    "required": ["skill_name", "observation"],
}


# ── Handler 包装 ───────────────────────────────────────────────────────

def _skill_create_handler(name: str, description: str, content: str,
                          priority: int = 10, tools: list = None, **kw):
    return skill_create(name, description, content, priority, tools)


def _skill_update_handler(name: str, description: str = None, content: str = None,
                          priority: int = None, tools: list = None, **kw):
    return skill_update(name, description, content, priority, tools)


def _skill_delete_handler(name: str, **kw):
    return skill_delete(name)


def _skill_improve_handler(skill_name: str, observation: str,
                           parent_agent=None, **kw):
    return skill_improve(skill_name, observation, parent_agent)


# ── 注册 ───────────────────────────────────────────────────────────────

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
    name="skill_improve",
    toolset="skills",
    schema=SKILL_IMPROVE_SCHEMA,
    handler=_skill_improve_handler,
)
