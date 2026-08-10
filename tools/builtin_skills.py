"""
技能列表/查看工具 — 适配 hermes-agent SKILL.md 格式。

提供 agent 可调用的两个工具：
- skills_list: 列出所有可用技能（仅元数据，节省 token）
- skill_view: 查看指定技能的完整内容

技能格式（hermes-agent 标准）：
    skills/
      skill-name/
        SKILL.md        # 技能内容（YAML frontmatter + Markdown）
        references/     # 可选：附加参考文件
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from config import PROJECT_ROOT
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(PROJECT_ROOT) / "skills"

# ── 目录过滤 ───────────────────────────────────────────────────────────
_EXCLUDED_SKILL_DIRS: frozenset = frozenset(
    (".git", ".github", ".hub", ".archive",
     "__pycache__", "node_modules", ".venv", "venv", "env",
     ".env", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache")
)

# ── 元数据限制 ─────────────────────────────────────────────────────────
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024

# ── YAML 加载 ──────────────────────────────────────────────────────────
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


def _parse_frontmatter(text: str) -> Dict[str, Any]:
    """解析 SKILL.md 的 YAML frontmatter，支持嵌套 metadata。"""
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
        logger.debug("YAML 解析失败，回退到简单键值解析")

    # 回退
    fm: Dict[str, Any] = {}
    for line in yaml_content.strip().split("\n"):
        if ":" in line and not line.strip().startswith("#"):
            key, _, val = line.partition(":")
            fm[key.strip()] = val.strip().strip("\"'")
    return fm


def _discover_skill_files() -> List[Path]:
    """递归扫描 SKILLS_DIR，找到所有 SKILL.md。"""
    if not SKILLS_DIR.exists():
        logger.debug("技能目录不存在: %s", SKILLS_DIR)
        return []

    matches: List[Path] = []
    for root, dirs, files in os.walk(SKILLS_DIR, followlinks=True):
        dirs[:] = [d for d in dirs if d not in _EXCLUDED_SKILL_DIRS]
        if "SKILL.md" in files:
            matches.append(Path(root) / "SKILL.md")

    return sorted(matches, key=lambda p: str(p.relative_to(SKILLS_DIR)))


def _load_skill(path: Path) -> Dict[str, Any]:
    """加载单个 SKILL.md，返回 {name, path, content, raw_content, frontmatter}。"""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("读取技能文件失败 %s: %s", path, e)
        return {}

    fm = _parse_frontmatter(text)

    # 去掉 frontmatter
    body = text
    end_match = re.search(r"\n---\s*\n", text[3:])
    if end_match:
        body = text[end_match.end() + 3:].strip()

    name = fm.get("name", path.parent.name)

    # 提取 hermes metadata 嵌套字段
    tools = fm.get("tools", [])
    tags = []
    related = []
    meta = fm.get("metadata", {})
    if isinstance(meta, dict):
        hermes = meta.get("hermes", {})
        if isinstance(hermes, dict):
            tags = hermes.get("tags", [])
            related = hermes.get("related_skills", [])

    return {
        "name": str(name)[:MAX_NAME_LENGTH],
        "description": str(fm.get("description", ""))[:MAX_DESCRIPTION_LENGTH],
        "path": str(path),
        "skill_dir": str(path.parent),
        "priority": fm.get("priority"),
        "tools": tools if isinstance(tools, list) else [],
        "version": fm.get("version", ""),
        "author": fm.get("author", ""),
        "license": fm.get("license", ""),
        "platforms": fm.get("platforms", []),
        "tags": tags if isinstance(tags, list) else [],
        "related_skills": related if isinstance(related, list) else [],
        "content": body,
        "raw_content": text,
    }


def _is_skill_name_safe(name: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", name))


def _skill_matches_platform(fm: Dict[str, Any]) -> bool:
    platforms = fm.get("platforms")
    if not platforms:
        return True
    if not isinstance(platforms, list):
        platforms = [platforms]
    current = os.name
    plat_map = {"macos": "posix", "linux": "posix", "windows": "nt"}
    for p in platforms:
        mapped = plat_map.get(str(p).lower(), str(p).lower())
        if mapped == current or mapped == "any":
            return True
    return False


def skills_list(**kwargs) -> str:
    """
    列出所有可用技能，只返回元数据（名称 + 描述），不返回完整内容。
    遵循 progressive disclosure 原则。
    """
    skills = []
    for path in _discover_skill_files():
        info = _load_skill(path)
        if not info:
            continue
        fm = _parse_frontmatter(info.get("raw_content", ""))

        if not _skill_matches_platform(fm):
            logger.debug("技能 %s 不适用当前平台", fm.get("name"))
            continue

        skills.append({
            "name": info["name"],
            "description": info["description"],
            "priority": info.get("priority"),
            "tags": info.get("tags", []),
            "skill_dir": path.parent.name,
        })

    if not skills:
        return ("当前没有已安装的技能。\n"
                "技能以 SKILL.md 文件存放在 skills/<技能名>/ 目录下。")

    skills.sort(key=lambda s: s.get("priority", 0) or 0, reverse=True)

    # Categorize by usage scenario
    video_animation = []
    video_style = []
    prompt_tools = []
    other = []

    anim_names = {"3d-animation-short-generator", "papercraft-stop-motion-explainer",
                  "paper-collage-explainer-generator", "handdrawn-live-video-generator"}
    style_names = {"brand-promo-video-generator", "co-op-game-intro-generator",
                   "minimalist-product-ad-generator", "music-video-subtitle-generator"}
    prompt_names = {"h3-prompt-writing", "ai-video-prompting", "z_image_turbo_prompt_master"}

    for s in skills:
        n = s["name"]
        if n in anim_names:
            video_animation.append(s)
        elif n in style_names:
            video_style.append(s)
        elif n in prompt_names:
            prompt_tools.append(s)
        else:
            other.append(s)

    lines = [f"共 {len(skills)} 个技能：", ""]

    if prompt_tools:
        lines.append("## 提示词生成（先选这类！把用户需求转成规范提示词）")
        for i, s in enumerate(prompt_tools):
            lines.append(f"  • **{s['name']}** — {s['description']}")
        lines.append("")
        lines.append("**场景匹配规则：**")
        lines.append('- 自然风景/真实风格视频/慢镜头/人像 → 选 **h3-prompt-writing**（T2VA 模式）')
        lines.append('- 3D卡通动画/角色动画/叙事短片 → 选 **3d-animation-short-generator**')
        lines.append('- Z-Image 图片生成 → 选 **z_image_turbo_prompt_master**')
        lines.append('- 通用视频提示词方法论 → 选 **ai-video-prompting**')
        lines.append("")

    if video_animation:
        lines.append("## 视频风格：动画类")
        for i, s in enumerate(video_animation):
            lines.append(f"  • **{s['name']}** — {s['description']}")
        lines.append("")

    if video_style:
        lines.append("## 视频风格：营销/自媒体类")
        for i, s in enumerate(video_style):
            lines.append(f"  • **{s['name']}** — {s['description']}")
        lines.append("")

    if other:
        lines.append("## 其他")
        for i, s in enumerate(other):
            lines.append(f"  • **{s['name']}** — {s['description']}")
        lines.append("")

    lines.append("---")
    lines.append("使用方式：skills_list 查看列表 → skill_view(名称) 加载指令 → skill_reference(名称, 文件名) 读参考文件 → 生成")
    return "\n".join(lines)


def skill_view(name: str) -> str:
    """
    查看指定技能的完整内容。
    参数:
        name: 技能名称
    """
    if not name or not str(name).strip():
        return tool_error("请指定要查看的技能名称。")

    name = str(name).strip()
    if not _is_skill_name_safe(name):
        return tool_error(f"技能名称不安全: {name}")

    for path in _discover_skill_files():
        info = _load_skill(path)
        if not info:
            continue
        # Match by frontmatter name or directory name
        if info.get("name") != name and path.parent.name != name:
            continue

        output = f"## {info['name']}\n\n"
        if info["description"]:
            output += f"*{info['description']}*\n\n"
        if info.get("version"):
            output += f"版本: {info['version']} | "
        if info.get("author"):
            output += f"作者: {info['author']}\n"
        if info.get("tags"):
            output += f"标签: {', '.join(info['tags'])}\n"
        output += "\n---\n\n"
        output += info["content"]

        # List available references
        import os as _os
        refs_dir = Path(info["skill_dir"]) / "references"
        if refs_dir.exists() and refs_dir.is_dir():
            ref_files = sorted([f for f in refs_dir.iterdir() if f.is_file()])
            if ref_files:
                output += "\n\n## Available Reference Files\n\n"
                output += f"This skill has {len(ref_files)} reference file(s) in `references/`:\n\n"
                for rf in ref_files:
                    size = rf.stat().st_size
                    unit = "B"
                    if size >= 1024:
                        size //= 1024; unit = "KB"
                    output += f"- `{rf.name}` ({size}{unit})\n"
                output += "\nUse `skill_reference` tool to load a reference file on demand.\n"

        return output

    return tool_error(f"未找到技能: {name}。使用 skills_list 查看可用技能列表。")

def skill_reference(skill_name: str, ref_name: str) -> str:
    """Read a specific reference file from a skill's references/ directory.

    Use skill_view first to see available references, then call this tool
    to load the one you need.

    Args:
        skill_name: The skill name (e.g. "h3-prompt-writing")
        ref_name: The reference filename (e.g. "base-en.txt")
    """
    if not skill_name or not ref_name:
        return tool_error("Please specify both skill_name and ref_name.")

    skill_name = str(skill_name).strip()
    ref_name = str(ref_name).strip()

    if not _is_skill_name_safe(skill_name):
        return tool_error("Invalid skill name: " + skill_name)

    # Prevent path traversal
    if ".." in ref_name or "/" in ref_name or "\\" in ref_name:
        return tool_error("Invalid reference filename: " + ref_name)

    for path in _discover_skill_files():
        info = _load_skill(path)
        if not info:
            continue
        if info.get("name") != skill_name and path.parent.name != skill_name:
            continue
        ref_path = path.parent / "references" / ref_name
        if not ref_path.exists():
            available = []
            refs_dir = path.parent / "references"
            if refs_dir.exists():
                available = [f.name for f in refs_dir.iterdir() if f.is_file()]
            msg = "Reference file '" + ref_name + "' not found."
            if available:
                msg += " Available: " + ", ".join(available)
            return tool_error(msg)

        try:
            content = ref_path.read_text(encoding="utf-8")
            return "## Reference: " + ref_name + "\n\n" + content
        except Exception as e:
            return tool_error("Failed to read: " + str(e))

    return tool_error("Skill not found: " + skill_name)


def skill_delete(name: str) -> str:
    """通过技能名删除一个技能目录。（供前端 /api/skills/<id> DELETE 使用）"""
    import shutil

    if not name or not _is_skill_name_safe(str(name).strip()):
        return tool_error(f"无效的技能名称: {name}")

    name = str(name).strip()
    for path in _discover_skill_files():
        info = _load_skill(path)
        if not info:
            continue
        if info.get("name") != name and path.parent.name != name:
            continue
        try:
            shutil.rmtree(str(path.parent))
            return f"技能 '{name}' 已删除。"
        except Exception as e:
            return tool_error(f"删除失败: {e}")

    return tool_error(f"未找到技能: {name}")


def check_skills_requirements() -> bool:
    return True


# ── JSON Schema ────────────────────────────────────────────────────────

SKILLS_LIST_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
}

SKILL_VIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "要查看的技能名称。",
        },
    },
    "required": ["name"],
}


# ── 注册 ──────────────────────────────────────────────────────────────

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
SKILL_REFERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "skill_name": {"type": "string", "description": "Skill name."},
        "ref_name": {"type": "string", "description": "Reference filename to load (e.g. base-en.txt). Use skill_view first to see available files."},
    },
    "required": ["skill_name", "ref_name"],
}

registry.register(
    name="skill_reference",
    toolset="skills",
    schema=SKILL_REFERENCE_SCHEMA,
    handler=skill_reference,
    check_fn=check_skills_requirements,
)

