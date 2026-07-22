"""
SKILL.md 预处理 — 模板变量替换和内联 Shell 执行
完整移植自 hermes-agent agent/skill_preprocessing.py

功能:
  - ${SKILL_DIR} / ${SESSION_ID} 模板变量替换
  - !`command` 内联 Shell 片段执行
  - 可配置的预处理流水线 (config 驱动)

用法:
    from services.skill_preprocessing import preprocess_skill_content
    processed = preprocess_skill_content(skill.content, skill_dir, session_id)
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 匹配 ${SKILL_DIR} / ${SESSION_ID} 模板变量
_SKILL_TEMPLATE_RE = re.compile(r"\$\{(SKILL_DIR|SESSION_ID)\}")

# 匹配内联 Shell: !`command`
_INLINE_SHELL_RE = re.compile(r"!`([^`\n]+)`")

# 内联 Shell 输出上限 (防止失控命令撑满上下文)
_INLINE_SHELL_MAX_OUTPUT = 4000


def load_skills_config() -> dict:
    """
    加载 skills 配置项 (best-effort).

    从项目 config.py 读取 skills 部分, 失败返回空字典。
    """
    try:
        # 尝试从项目配置加载
        from config import Config
        cfg = Config()
        return getattr(cfg, 'skills', {}) or {}
    except Exception:
        logger.debug("无法读取 skills 配置", exc_info=True)
    return {}


def substitute_template_vars(
    content: str,
    skill_dir: Optional[Path] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    替换 SKILL.md 中的 ${SKILL_DIR} / ${SESSION_ID} 模板变量.

    只有存在对应值时才会替换 — 无法解析的变量保持原样,
    方便作者调试。
    """
    if not content:
        return content

    skill_dir_str = str(skill_dir) if skill_dir else None

    def _replace(match: re.Match) -> str:
        token = match.group(1)
        if token == "SKILL_DIR" and skill_dir_str:
            return skill_dir_str
        if token == "SESSION_ID" and session_id:
            return str(session_id)
        return match.group(0)

    return _SKILL_TEMPLATE_RE.sub(_replace, content)


def run_inline_shell(command: str, cwd: Optional[Path] = None, timeout: int = 10) -> str:
    """
    执行单个内联 Shell 片段, 返回 stdout (裁剪后).

    失败返回 [inline-shell error: ...] 标记, 不抛出异常,
    防止单个错误的 shell 片段中断整个技能加载。
    """
    try:
        # Windows 不支持 bash, 用 cmd 或跳过
        if not _has_shell():
            return f"[inline-shell skipped: no shell available for: {command[:80]}]"

        completed = subprocess.run(
            _shell_command(command),
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout)),
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return f"[inline-shell timeout after {timeout}s: {command[:80]}]"
    except FileNotFoundError:
        return "[inline-shell error: shell not found]"
    except Exception as exc:
        return f"[inline-shell error: {exc}]"

    output = (completed.stdout or "").rstrip("\n")
    if not output and completed.stderr:
        output = completed.stderr.rstrip("\n")
    if len(output) > _INLINE_SHELL_MAX_OUTPUT:
        output = output[:_INLINE_SHELL_MAX_OUTPUT] + "...[truncated]"
    return output


def expand_inline_shell(
    content: str,
    skill_dir: Optional[Path] = None,
    timeout: int = 10,
) -> str:
    """
    替换 SKILL.md 中所有 !`command` 片段为命令输出.

    以技能目录为 CWD 执行, 这样技能附带的脚本路径可以相对引用。
    """
    if "!`" not in content:
        return content

    def _replace(match: re.Match) -> str:
        cmd = match.group(1).strip()
        if not cmd:
            return ""
        return run_inline_shell(cmd, skill_dir, timeout)

    return _INLINE_SHELL_RE.sub(_replace, content)


def preprocess_skill_content(
    content: str,
    skill_dir: Optional[Path] = None,
    session_id: Optional[str] = None,
    skills_cfg: Optional[dict] = None,
) -> str:
    """
    应用 SKILL.md 模板变量和内联 Shell 预处理.

    由配置控制 (默认只启用模板变量, 内联 Shell 需显式开启):
      skills:
        template_vars: true   (默认启用的内置预处理方法)
        inline_shell: false    (默认关闭 — 安全考量)
        inline_shell_timeout: 10
    """
    if not content:
        return content

    cfg = skills_cfg if isinstance(skills_cfg, dict) else load_skills_config()

    if cfg.get("template_vars", True):
        content = substitute_template_vars(content, skill_dir, session_id)

    if cfg.get("inline_shell", False):
        timeout = int(cfg.get("inline_shell_timeout", 10) or 10)
        content = expand_inline_shell(content, skill_dir, timeout)

    return content


# ── 平台适配 ────────────────────────────────────────────────────────────

def _has_shell() -> bool:
    """检测是否有可用的 Shell"""
    try:
        if os.name == "nt":
            return False  # Windows cmd 不支持 bash 语法, 默认跳过
        subprocess.run(
            ["bash", "--version"],
            capture_output=True, timeout=3, check=False,
            stdin=subprocess.DEVNULL,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _shell_command(command: str) -> list:
    """返回平台对应的 Shell 命令格式"""
    if os.name == "nt":
        return ["cmd", "/c", command]
    return ["bash", "-c", command]
