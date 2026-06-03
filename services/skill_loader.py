"""技能加载器 — 读取和管理 skills/ 目录下的 .skill 文件。

核心功能:
    - 解析 YAML frontmatter（元数据：name, description, priority, tools）
    - 解析 Markdown 正文（技能指令）
    - 按优先级排序
    - 技能完整性校验

技能文件格式（.skill）：
    ---
    name: my-skill
    description: 简短描述
    priority: 10
    tools:
      - read_file
    ---
    # 技能标题
    完整指令内容...
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "skills")


class Skill:
    """A loaded skill with its metadata and content."""

    __slots__ = ("name", "description", "priority", "tools", "content", "path")

    def __init__(
        self,
        name: str,
        description: str = "",
        priority: int = 0,
        tools: Optional[List[str]] = None,
        content: str = "",
        path: str = "",
    ):
        self.name = name
        self.description = description
        self.priority = priority
        self.tools = tools or []
        self.content = content
        self.path = path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "tools": self.tools,
            "size": len(self.content),
        }

    def to_prompt_fragment(self) -> str:
        """Convert skill to a prompt fragment for injection."""
        return self.content


def _parse_skill_frontmatter(text: str) -> tuple:
    """Parse YAML frontmatter from a skill file.

    Returns:
        (metadata_dict, content_without_frontmatter)
    """
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    frontmatter = parts[1].strip()
    content = parts[2].strip()

    meta = {}
    for line in frontmatter.split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip("'\"")
            meta[key] = value

    # Parse tools list if it's a YAML list
    if "tools" in meta and isinstance(meta["tools"], str):
        meta["tools"] = [t.strip() for t in meta["tools"].split(",") if t.strip()]

    if "priority" in meta:
        try:
            meta["priority"] = int(meta["priority"])
        except (ValueError, TypeError):
            meta["priority"] = 0

    return meta, content


def load_skill(filepath: str) -> Optional[Skill]:
    """Load a single skill from a .skill file.

    Args:
        filepath: Path to the .skill file.

    Returns:
        Skill object or None if loading failed.
    """
    path = Path(filepath)
    if not path.exists() or not path.suffix == ".skill":
        return None

    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Could not read skill file %s: %s", filepath, e)
        return None

    meta, content = _parse_skill_frontmatter(raw)

    return Skill(
        name=meta.get("name", path.stem),
        description=meta.get("description", ""),
        priority=meta.get("priority", 0),
        tools=meta.get("tools", []),
        content=content,
        path=str(path),
    )


def load_all_skills(skills_dir: Optional[str] = None) -> Dict[str, Skill]:
    """Load all .skill files from the skills directory.

    Args:
        skills_dir: Path to skills directory. Defaults to project skills/.

    Returns:
        Dict of {skill_name: Skill} for all loaded skills.
    """
    if skills_dir is None:
        skills_dir = SKILLS_DIR

    if not os.path.isdir(skills_dir):
        logger.info("Skills directory not found: %s", skills_dir)
        return {}

    skills = {}
    for filename in sorted(os.listdir(skills_dir)):
        if not filename.endswith(".skill"):
            continue

        filepath = os.path.join(skills_dir, filename)
        skill = load_skill(filepath)
        if skill:
            skills[skill.name] = skill
            logger.info("Loaded skill: %s (priority=%d)", skill.name, skill.priority)

    return skills


def get_active_skills_prompt(skills: Optional[Dict[str, Skill]] = None, active_names: Optional[List[str]] = None) -> str:
    """Build a system prompt fragment from active skills.

    Args:
        skills: Dict of all skills. If None, load all.
        active_names: List of skill names to include. If None, include all.

    Returns:
        Prompt string to inject into system message.
    """
    if skills is None:
        skills = load_all_skills()

    if not skills:
        return ""

    active = list(skills.values())
    if active_names:
        active = [s for s in active if s.name in active_names]

    if not active:
        return ""

    # Sort by priority (highest first)
    active.sort(key=lambda s: s.priority, reverse=True)

    parts = ["\n## Active Skills\n"]
    for skill in active:
        parts.append(f"### Skill: {skill.name}\n{skill.to_prompt_fragment()}")

    return "\n".join(parts)