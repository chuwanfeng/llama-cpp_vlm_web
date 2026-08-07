"""技能加载器 — 读取和管理 skills/ 目录下的 SKILL.md 文件。

采用 hermes-agent 技能格式（业界通用标准）：
    skills/
      my-skill/
        SKILL.md        # 技能的完整指令（YAML frontmatter + Markdown 正文）
        references/     # 可选：技能附带的参考文件/脚本

SKILL.md 格式：
    ---
    name: my-skill
    description: 简短描述
    version: 1.0.0
    platforms: [windows, linux, macos]  # 可选，留空=全平台
    metadata:
      hermes:
        tags: [example-tag]
        related_skills: [other-skill]
    ---

    # 技能标题

    完整指令内容...
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = os.path.join(PROJECT_ROOT, "skills")

# ── 目录过滤 ───────────────────────────────────────────────────────────
# 从 hermes-agent 移植的排除目录列表
_EXCLUDED_SKILL_DIRS: frozenset = frozenset(
    (
        ".git",
        ".github",
        ".hub",
        ".archive",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".env",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    )
)

# ── YAML 加载（带缓存） ─────────────────────────────────────────────────
_yaml_load_fn = None


def _yaml_load(content: str):
    """解析 YAML，优先使用 CSafeLoader。"""
    global _yaml_load_fn
    if _yaml_load_fn is None:
        import yaml
        loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader

        def _load(value: str):
            return yaml.load(value, Loader=loader)

        _yaml_load_fn = _load
    return _yaml_load_fn(content)


# ── 平台匹配 ────────────────────────────────────────────────────────────

_PLATFORM_MAP = {
    "macos": "posix",
    "linux": "posix",
    "windows": "nt",
}


def _skill_matches_platform(frontmatter: Dict[str, Any]) -> bool:
    """检查技能是否兼容当前 OS 平台。

    如果未声明 platforms 字段则全部兼容（向后兼容默认值）。
    """
    platforms = frontmatter.get("platforms")
    if not platforms:
        return True
    if not isinstance(platforms, list):
        platforms = [platforms]
    current = os.name  # 'nt' (Windows) 或 'posix' (Linux/Mac)
    for p in platforms:
        p_lower = str(p).lower()
        mapped = _PLATFORM_MAP.get(p_lower, p_lower)
        if mapped == current or p_lower == "any":
            return True
    return False


# ── 路径过滤 ────────────────────────────────────────────────────────────

def _is_excluded_path(path: Path) -> bool:
    """检查路径的任意组件是否在排除目录列表中。"""
    parts = path.parts if hasattr(path, "parts") else Path(path).parts
    return any(part in _EXCLUDED_SKILL_DIRS for part in parts)


# ── 技能数据模型 ────────────────────────────────────────────────────────

class Skill:
    """一个已加载的技能，包含元数据和正文内容。"""

    __slots__ = (
        "name", "description", "priority", "tools",
        "content", "path", "skill_dir", "version",
        "author", "license", "platforms", "tags",
        "related_skills",
    )

    def __init__(
        self,
        name: str,
        description: str = "",
        priority: int = 0,
        tools: Optional[List[str]] = None,
        content: str = "",
        path: str = "",
        skill_dir: str = "",
        version: str = "",
        author: str = "",
        license: str = "",
        platforms: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        related_skills: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.priority = priority
        self.tools = tools or []
        self.content = content
        self.path = path
        self.skill_dir = skill_dir
        self.version = version
        self.author = author
        self.license = license
        self.platforms = platforms or []
        self.tags = tags or []
        self.related_skills = related_skills or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "tools": self.tools,
            "size": len(self.content),
            "version": self.version,
            "author": self.author,
            "platforms": self.platforms,
            "tags": self.tags,
            "related_skills": self.related_skills,
        }

    def to_prompt_fragment(self) -> str:
        """将技能转换为可注入的 prompt 片段。"""
        return self.content

    def __repr__(self) -> str:
        return f"Skill({self.name!r}, priority={self.priority})"


# ── Frontmatter 解析 ────────────────────────────────────────────────────

def _parse_skill_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """解析 SKILL.md 的 YAML frontmatter。

    使用 yaml CSafeLoader 做完整解析（支持嵌套 metadata、列表、
    多行字符串等），失败时回退到简单 key:value 解析。

    Returns:
        (frontmatter_dict, remaining_body)
    """
    frontmatter: Dict[str, Any] = {}
    body = content

    if not content.startswith("---"):
        return frontmatter, body

    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return frontmatter, body

    yaml_content = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :]

    try:
        parsed = _yaml_load(yaml_content)
        if isinstance(parsed, dict):
            frontmatter = parsed
    except Exception:
        # 回退：简单 key:value 解析
        logger.debug("YAML 解析失败，回退到简单解析")
        for line in yaml_content.strip().split("\n"):
            if ":" not in line or line.strip().startswith("#"):
                continue
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip().strip("\"'")

    return frontmatter, body


def _extract_hermes_metadata(frontmatter: Dict[str, Any]) -> Dict[str, Any]:
    """从 hermes 风格 frontmatter 中提取结构化元数据。"""
    result: Dict[str, Any] = {}
    metadata = frontmatter.get("metadata", {})
    if isinstance(metadata, dict):
        hermes = metadata.get("hermes", {})
        if isinstance(hermes, dict):
            result["tags"] = hermes.get("tags", [])
            result["related_skills"] = hermes.get("related_skills", [])
    return result


# ── 技能发现 ────────────────────────────────────────────────────────────

def _discover_skill_files(skills_dir: str = SKILLS_DIR) -> List[Path]:
    """递归扫描 skills_dir，找到所有 SKILL.md 文件。

    排除 .git、.archive、__pycache__ 等目录。
    """
    base = Path(skills_dir)
    if not base.is_dir():
        return []

    matches: List[Path] = []
    for root, dirs, files in os.walk(base, followlinks=True):
        # 排除不需要的目录
        dirs[:] = [d for d in dirs if d not in _EXCLUDED_SKILL_DIRS]
        if "SKILL.md" in files:
            matches.append(Path(root) / "SKILL.md")

    # 按相对路径排序
    return sorted(matches, key=lambda p: str(p.relative_to(base)))


# ── 技能加载 ────────────────────────────────────────────────────────────

# 全局预处理配置缓存
_preprocess_cfg: Optional[Dict[str, Any]] = None


def load_skill(filepath: str, preprocess: bool = True) -> Optional[Skill]:
    """加载单个技能文件（SKILL.md）。

    Args:
        filepath: SKILL.md 的路径。
        preprocess: 是否应用模板变量/内联 Shell 预处理 (默认 True)。

    Returns:
        Skill 对象，加载失败返回 None。
    """
    path = Path(filepath)
    if not path.exists():
        return None
    if path.name != "SKILL.md":
        return None

    if _is_excluded_path(path):
        return None

    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("无法读取技能文件 %s: %s", filepath, e)
        return None

    # ── 预处理 ──
    if preprocess:
        global _preprocess_cfg
        if _preprocess_cfg is None:
            try:
                from services.skill_preprocessing import load_skills_config
                _preprocess_cfg = load_skills_config()
            except Exception:
                _preprocess_cfg = {}
        try:
            from services.skill_preprocessing import preprocess_skill_content
            raw = preprocess_skill_content(raw, path.parent, skills_cfg=_preprocess_cfg)
        except Exception:
            logger.debug("技能预处理跳过 (预处理模块不可用)", exc_info=True)

    frontmatter, body = _parse_skill_frontmatter(raw)
    if not frontmatter:
        # 无 frontmatter — 以目录名作为技能名
        frontmatter["name"] = path.parent.name

    hermes_meta = _extract_hermes_metadata(frontmatter)

    skill_dir = str(path.parent)

    return Skill(
        name=frontmatter.get("name", path.parent.name),
        description=frontmatter.get("description", ""),
        priority=frontmatter.get("priority", 0),
        tools=frontmatter.get("tools", []),
        content=body.strip(),
        path=str(path),
        skill_dir=skill_dir,
        version=frontmatter.get("version", ""),
        author=frontmatter.get("author", ""),
        license=frontmatter.get("license", ""),
        platforms=frontmatter.get("platforms", []),
        tags=hermes_meta.get("tags", []),
        related_skills=hermes_meta.get("related_skills", []),
    )


def load_all_skills(skills_dir: Optional[str] = None, platform_filter: bool = True) -> Dict[str, Skill]:
    """加载 skills 目录下的全部技能。

    Args:
        skills_dir: 技能目录路径。默认使用项目 skills/。
        platform_filter: 是否按当前平台过滤（默认 True）。

    Returns:
        {skill_name: Skill} 字典。
    """
    if skills_dir is None:
        skills_dir = SKILLS_DIR

    if not os.path.isdir(skills_dir):
        logger.info("技能目录不存在: %s", skills_dir)
        return {}

    skills: Dict[str, Skill] = {}
    for skill_md in _discover_skill_files(skills_dir):
        skill = load_skill(str(skill_md))
        if skill is None:
            continue

        # 平台过滤
        if platform_filter:
            raw = skill_md.read_text(encoding="utf-8", errors="replace")
            fm, _ = _parse_skill_frontmatter(raw)
            if not _skill_matches_platform(fm):
                logger.debug("技能 %s 不兼容当前平台，已跳过", skill.name)
                continue

        skills[skill.name] = skill
        logger.info("已加载技能: %s (priority=%d, dir=%s)", skill.name, skill.priority, skill.skill_dir)

    return skills


def list_skills_meta(skills_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """列出所有技能的元数据（不含正文内容）— 节省 token。

    Returns:
        [{name, description, priority, tools, size, ...}, ...]
    """
    if skills_dir is None:
        skills_dir = SKILLS_DIR

    skills = load_all_skills(skills_dir)
    meta_list = [s.to_dict() for s in skills.values()]
    meta_list.sort(key=lambda m: m.get("priority", 0) or 0, reverse=True)
    return meta_list


def get_active_skills_prompt(
    skills: Optional[Dict[str, Skill]] = None,
    active_names: Optional[List[str]] = None,
    skills_dir: Optional[str] = None,
) -> str:
    """从活跃技能构建系统 prompt 片段。

    Args:
        skills: 技能字典。若为 None 则加载全部。
        active_names: 需要包含的技能名列表。若为 None 则包含全部。
        skills_dir: 技能目录（仅在 skills 为 None 时生效）。

    Returns:
        可注入到系统消息的 prompt 字符串。
    """
    if skills is None:
        skills = load_all_skills(skills_dir)

    if not skills:
        return ""

    active = list(skills.values())
    if active_names:
        active = [s for s in active if s.name in active_names]

    if not active:
        return ""

    # 按优先级排序（最高优先）
    active.sort(key=lambda s: s.priority, reverse=True)

    parts = ["\n## 活跃技能\n"]
    for skill in active:
        parts.append(f"### 技能: {skill.name}\n")
        if skill.description:
            parts.append(f"*{skill.description}*\n")
        parts.append(skill.to_prompt_fragment())

    return "\n".join(parts)





