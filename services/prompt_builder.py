"""系统提示词构建器 - 为 AgentLoop 自动组装系统提示词。

移植自 hermes-agent/agent/prompt_builder.py，精简适配 llama-cpp_vlm_web 项目。
仅保留核心功能（工具描述、身份文件、平台提示），移除 Hermes 专属特性。

用法:
    from services.prompt_builder import build_system_prompt
    system_prompt = build_system_prompt(tool_schemas, cwd=project_root)
"""

import os
from pathlib import Path
from typing import Optional


# =============================================================================
# 配置
# =============================================================================

CONTEXT_FILE_MAX_CHARS = 20_000
CONTEXT_TRUNCATE_HEAD_RATIO = 0.7
CONTEXT_TRUNCATE_TAIL_RATIO = 0.2

_SOUL_MD_NAMES = ("SOUL.md", "soul.md")
_AGENTS_MD_NAMES = ("AGENTS.md", "agents.md")


# =============================================================================
# 工具描述生成
# =============================================================================

def build_tool_descriptions(tool_schemas: list[dict]) -> str:
    """根据 OpenAI 格式的 tool_schemas 生成工具描述文本。

    Args:
        tool_schemas: list of {"type":"function","function":{"name":...,"description":...,"parameters":...}}

    Returns:
        格式化的工具描述文本，可直接插入系统提示词
    """
    if not tool_schemas:
        return ""

    # 生成简洁的工具清单
    lines = ["## 可用工具", ""]
    for ts in tool_schemas:
        fn = ts.get("function", ts)
        name = fn.get("name", "?")
        desc = fn.get("description", "")
        # 提取参数详情
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        if props:
            param_parts = []
            for pname, pinfo in props.items():
                pdesc = pinfo.get('description', '')[:60]
                req = ' (必填)' if pname in required else ''
                param_parts.append(f"{pname}{req}: {pdesc}")
            param_str = '；'.join(param_parts)
        else:
            param_str = ""
        lines.append(f"- **{name}**: {desc}")
        if param_str:
            lines.append(f"  {param_str}")

    # 工具调用格式说明
    lines.extend([
        "",
        "## 工具调用格式",
        "",
        "需要调用工具时，输出如下 JSON:",
        "",
        "```json",
        '{"tool_call": {"name": "工具名", "arguments": {"参数": "值"}}}',
        "```",
        "",
        "每次只调用一个工具。工具执行结果会作为新的消息返回给你。",
        "如果有多个工具要调用，分多轮依次执行。",
    ])
    return "\n".join(lines)


# =============================================================================
# 上下文文件加载（SOUL.md / AGENTS.md）
# =============================================================================

def _find_git_root(start: Path) -> Optional[Path]:
    """向上查找 .git 目录，返回 git 根目录。"""
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _find_context_file(cwd: Path, names: tuple[str, ...]) -> Optional[Path]:
    """在 cwd 及其父目录中查找指定的上下文文件。

    搜索范围: cwd -> 逐级向上 -> git root -> 文件系统根
    """
    stop_at = _find_git_root(cwd)
    current = cwd.resolve()
    for directory in [current, *current.parents]:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        if stop_at and directory == stop_at:
            break
    return None


def _truncate_content(content: str, filename: str, max_chars: int = CONTEXT_FILE_MAX_CHARS) -> str:
    """内容过长时头尾截断，中间插入省略标记。"""
    if len(content) <= max_chars:
        return content
    head_chars = int(max_chars * CONTEXT_TRUNCATE_HEAD_RATIO)
    tail_chars = int(max_chars * CONTEXT_TRUNCATE_TAIL_RATIO)
    head = content[:head_chars]
    tail = content[-tail_chars:]
    marker = (
        f"\n\n[... 内容过长已截断 {filename}: "
        f"保留头部 {head_chars} + 尾部 {tail_chars} 共 {len(content)} 字符。"
        f"可使用 read_file 工具读取完整文件。]\n\n"
    )
    return head + marker + tail


def load_soul_md(cwd: Optional[str] = None) -> Optional[str]:
    """加载 SOUL.md（Agent 身份定义文件）。

    Args:
        cwd: 搜索起始目录，默认当前工作目录

    Returns:
        SOUL.md 内容字符串，未找到返回 None
    """
    start = Path(cwd) if cwd else Path.cwd()
    path = _find_context_file(start, _SOUL_MD_NAMES)
    if not path:
        return None
    try:
        content = path.read_text(encoding="utf-8")
        return _truncate_content(content, path.name)
    except Exception:
        return None


def load_agents_md(cwd: Optional[str] = None) -> Optional[str]:
    """加载 AGENTS.md（项目规则文件）。

    Args:
        cwd: 搜索起始目录，默认当前工作目录

    Returns:
        AGENTS.md 内容字符串，未找到返回 None
    """
    start = Path(cwd) if cwd else Path.cwd()
    path = _find_context_file(start, _AGENTS_MD_NAMES)
    if not path:
        return None
    try:
        content = path.read_text(encoding="utf-8")
        return _truncate_content(content, path.name)
    except Exception:
        return None


# =============================================================================
# 平台提示
# =============================================================================

def build_platform_hints() -> str:
    """返回当前运行环境的平台提示。"""
    import platform
    system = platform.system()

    hints = []
    if system == "Windows":
        hints.extend([
            "- 运行环境: Windows",
            "- 路径分隔符使用反斜杠 (\\)，但正斜杠 (/) 也通常可用",
            "- 执行命令时使用 PowerShell 语法",
            "- 文件编码注意 GBK/UTF-8 混用问题",
        ])
    elif system == "Linux":
        hints.extend([
            "- 运行环境: Linux",
            "- 使用标准 Unix 命令",
            "- 路径分隔符为正斜杠 (/)",
        ])
    elif system == "Darwin":
        hints.extend([
            "- 运行环境: macOS",
            "- 使用标准 Unix 命令",
            "- 路径分隔符为正斜杠 (/)",
        ])

    return "\n".join(hints) if hints else ""


# =============================================================================
# 完整系统提示词组装
# =============================================================================

def build_system_prompt(
    tool_schemas: Optional[list[dict]] = None,
    cwd: Optional[str] = None,
    include_soul: bool = True,
    include_agents: bool = True,
    include_platform: bool = True,
    include_tools: bool = True,
) -> str:
    """组装完整的系统提示词。

    Args:
        tool_schemas: OpenAI 格式的工具 schema 列表
        cwd: 搜索上下文文件的起始目录
        include_soul: 是否包含 SOUL.md 身份定义
        include_agents: 是否包含 AGENTS.md 规则文件
        include_platform: 是否包含平台提示
        include_tools: 是否包含工具描述

    Returns:
        组装好的系统提示词字符串
    """
    parts = []

    # ── 插槽 1: 身份定义（SOUL.md） ──
    if include_soul:
        soul = load_soul_md(cwd)
        if soul:
            parts.append(soul)

    # ── 插槽 2: 项目规则（AGENTS.md） ──
    if include_agents:
        agents = load_agents_md(cwd)
        if agents:
            parts.append(agents)

    # ── 插槽 3: 平台提示 ──
    if include_platform:
        hints = build_platform_hints()
        if hints:
            parts.append("## 运行环境\n" + hints)

    # ── 插槽 4: 工具描述 ──
    if include_tools and tool_schemas:
        tools = build_tool_descriptions(tool_schemas)
        if tools:
            parts.append(tools)

    # ── 兜底: 如果没有任何内容，给一个最小提示词 ──
    if not parts:
        parts.append("你是一个有用的 AI 助手。")

    return "\n\n".join(parts)
