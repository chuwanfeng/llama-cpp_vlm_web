"""
技能列表/查看工具 — 移植自 hermes-agent/tools/skills_tool.py，适配本项目。

提供 agent 可调用的两个工具：
- skills_list: 列出所有可用技能（仅元数据，节省 token）
- skill_view: 查看指定技能的完整内容

技能文件格式（.skill）：
    ---
    name: my-skill
    description: 简短描述（最多 1024 字符）
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

from config import PROJECT_ROOT
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(PROJECT_ROOT) / "skills"

# Metadata 字段限制（参考 hermes-agent 的 progressive disclosure 设计）
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024

_PRE_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(text: str) -> Dict[str, Any]:
    """解析 SKILL.md / .skill 文件的 YAML frontmatter。"""
    m = _PRE_RE.match(text)
    if not m:
        return {}

    yaml_text = m.group(1)
    try:
        import yaml
        return yaml.safe_load(yaml_text) or {}
    except ImportError:
        logger.debug("PyYAML 未安装，回退到简单键值解析")
    except Exception as e:
        logger.debug("YAML 解析失败: %s", e)
        return {}

    # 无 PyYAML 时用简单正则解析常用字段
    fm: Dict[str, Any] = {}
    for line in yaml_text.split("\n"):
        line = line.strip()
        if ":" in line and not line.startswith("#"):
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key == "tools":
                continue  # 列表字段跳过简单解析
            fm[key] = val
    return fm


def _skill_files() -> List[Path]:
    """扫描 SKILLS_DIR 找所有 .skill 文件。"""
    if not SKILLS_DIR.exists():
        logger.debug("技能目录不存在: %s", SKILLS_DIR)
        return []
    return sorted(SKILLS_DIR.glob("*.skill"))


def _load_skill(path: Path) -> Dict[str, Any]:
    """加载单个技能文件，返回 {name, path, content, frontmatter}。"""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("读取技能文件失败 %s: %s", path, e)
        return {}

    fm = _parse_frontmatter(text)
    body = text

    # 去掉 frontmatter 部分得到正文
    m = _PRE_RE.match(text)
    if m:
        body = text[m.end():].strip()

    name = fm.get("name", path.stem)
    description = fm.get("description", "")

    return {
        "name": str(name)[:MAX_NAME_LENGTH],
        "description": str(description)[:MAX_DESCRIPTION_LENGTH],
        "path": str(path),
        "priority": fm.get("priority"),
        "tools": fm.get("tools", []),
        "content": body,
        "raw_content": text,
    }


def _is_skill_name_safe(name: str) -> bool:
    """路径安全检查：技能名只能包含字母、数字、下划线、连字符。"""
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", name))


def skills_list(platform: str = "") -> str:
    """
    列出所有可用技能，只返回元数据（名称 + 描述），不返回完整内容。
    遵循 progressive disclosure 原则：先轻量列表，用到时再 skill_view 加载全文。
    """
    skills = []
    for path in _skill_files():
        info = _load_skill(path)
        if not info:
            continue
        fm = _parse_frontmatter(info.get("raw_content", ""))

        # 平台过滤（如果 skill 声明了 platforms 字段）
        platforms = fm.get("platforms", [])
        if platforms:
            current_os = os.name  # 'nt' (Windows) 或 'posix' (Linux/Mac)
            platform_map = {
                "macos": "posix", "linux": "posix", "windows": "nt",
            }
            want = [platform_map.get(p, p) for p in platforms]
            if current_os not in want and "any" not in want:
                logger.debug("技能 %s 不适用当前平台 %s", fm.get("name"), current_os)
                continue

        skills.append({
            "name": info["name"],
            "description": info["description"],
            "priority": info.get("priority"),
        })

    if not skills:
        return "当前没有已安装的技能。技能文件放在 skills/ 目录下，以 .skill 为后缀。"

    # 按优先级排序
    skills.sort(key=lambda s: s.get("priority", 0) or 0, reverse=True)

    lines = [f"共 {len(skills)} 个技能："]
    for i, s in enumerate(skills):
        lines.append(f"{i + 1}. **{s['name']}**: {s['description']}")
    return "\n".join(lines)


def skill_view(name: str) -> str:
    """
    查看指定技能的完整内容。
    参数:
        name: 技能名称（不含 .skill 后缀）
    返回:
        技能的完整指令文本
    """
    if not name or not str(name).strip():
        return tool_error("请指定要查看的技能名称。")

    name = str(name).strip()

    if not _is_skill_name_safe(name):
        return tool_error(f"技能名称不安全: {name}")

    # 按名称查找
    for path in _skill_files():
        if path.stem == name:
            info = _load_skill(path)
            if not info:
                return tool_error(f"无法读取技能: {name}")

            output = f"## {info['name']}\n\n"
            if info["description"]:
                output += f"*{info['description']}*\n\n"
            output += "---\n\n"
            output += info["content"]
            return output

    return tool_error(f"未找到技能: {name}。使用 skills_list 查看可用技能列表。")


def check_skills_requirements() -> bool:
    """技能工具无外部依赖，始终可用。"""
    return True


# JSON Schema for skills tools (name/description handled by registry)

SKILLS_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "platform": {
            "type": "string",
            "description": "可选的平台过滤（macos/linux/windows），留空则不过滤",
        },
    },
    "required": [],
}

SKILL_VIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "要查看的技能名称，不包含 .skill 后缀。",
        },
    },
    "required": ["name"],
}


# =============================================================================
# 注册
# =============================================================================

registry.register(
    name="skills_list",
    toolset="skills",
    schema=SKILLS_LIST_SCHEMA,
    handler=skills_list,
    check_fn=check_skills_requirements,
)

registry.register(
    name="skill_view",
    toolset="skills",
    schema=SKILL_VIEW_SCHEMA,
    handler=skill_view,
    check_fn=check_skills_requirements,
)