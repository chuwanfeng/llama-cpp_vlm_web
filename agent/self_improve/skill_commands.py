"""
技能命令处理 — 自我进化模块中的技能管理。

从 hermes-agent/agent/skill_commands.py 移植，适配 llama-cpp_vlm_web。

功能：
    - 技能命令注册（/skill-name 形式的快捷命令）
    - 技能预处理（模板变量替换、内联 shell 展开）
    - 技能配置加载

与 tools/skill_tool.py 的区别：
    - tools/skill_tool.py: Agent 可调用的工具（skill_create/update/delete/evolve）
    - agent/self_improve/skill_commands.py: 技能命令处理和预处理逻辑
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 技能命令缓存
_skill_commands: Dict[str, Dict[str, Any]] = {}
_skill_commands_platform: Optional[str] = None

# 技能名清理正则
_SKILL_INVALID_CHARS = re.compile(r"[^a-z0-9-]")
_SKILL_MULTI_HYPHEN = re.compile(r"-{2,}")


def _sanitize_skill_command(name: str) -> str:
    """清理技能命令名。"""
    name = name.lower().strip().replace(" ", "-")
    name = _SKILL_INVALID_CHARS.sub("", name)
    name = _SKILL_MULTI_HYPHEN.sub("-", name)
    return name.strip("-")


def _resolve_skill_commands_platform() -> Optional[str]:
    """解析当前平台范围，用于禁用技能过滤。"""
    return os.getenv("LLAMA_PLATFORM") or os.getenv("PLATFORM")


def load_skills_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载技能配置文件。

    Args:
        config_path: 配置文件路径，默认查找项目根目录的 skills_config.yaml

    Returns:
        技能配置字典
    """
    import yaml

    if config_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "skills_config.yaml"

    if not Path(config_path).exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.debug("Failed to load skills config: %s", e)
        return {}


def substitute_template_vars(text: str, variables: Optional[Dict[str, str]] = None) -> str:
    """替换技能文本中的模板变量。

    支持 {{variable}} 语法。

    Args:
        text: 包含模板变量的文本
        variables: 变量字典

    Returns:
        替换后的文本
    """
    if not variables:
        return text

    result = text
    for key, value in variables.items():
        result = result.replace(f"{{{{{key}}}}}", str(value))
    return result


def expand_inline_shell(text: str, cwd: Optional[str] = None) -> str:
    """展开技能文本中的内联 shell 命令。

    支持 `$(command)` 语法，执行命令并替换为输出。

    Args:
        text: 可能包含 $(command) 的文本
        cwd: 命令执行的工作目录

    Returns:
        展开后的文本
    """
    import subprocess

    pattern = re.compile(r'\$\(([^)]+)\)')

    def _replace(match):
        command = match.group(1).strip()
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip()
        except Exception as e:
            logger.debug("Inline shell expansion failed: %s", e)
            return match.group(0)  # 保持原样

    return pattern.sub(_replace, text)


def get_skill_commands() -> Dict[str, Dict[str, Any]]:
    """获取所有可用的技能命令。

    扫描 skills/ 目录，为每个 .skill 文件生成一个 /command。

    Returns:
        命令名 -> 命令信息的字典
    """
    global _skill_commands, _skill_commands_platform

    current_platform = _resolve_skill_commands_platform()

    # 如果平台变化，清除缓存
    if current_platform != _skill_commands_platform:
        _skill_commands = {}
        _skill_commands_platform = current_platform

    if _skill_commands:
        return _skill_commands

    # 扫描 skills 目录
    project_root = Path(__file__).resolve().parent.parent.parent
    skills_dir = project_root / "skills"

    if not skills_dir.exists():
        return {}

    # 加载配置（用于禁用列表）
    config = load_skills_config()
    disabled_skills = set(config.get("disabled", []))
    platform_disabled = config.get("platform_disabled", {})
    if current_platform and current_platform in platform_disabled:
        disabled_skills.update(platform_disabled[current_platform])

    for skill_file in sorted(skills_dir.glob("*.skill")):
        name = skill_file.stem
        command_name = _sanitize_skill_command(name)

        if command_name in disabled_skills:
            continue

        # 读取技能内容
        try:
            from tools.builtin_skills import _load_skill
            skill_info = _load_skill(skill_file)
            if not skill_info:
                continue

            _skill_commands[command_name] = {
                "name": command_name,
                "skill_name": name,
                "description": skill_info.get("description", ""),
                "content": skill_info.get("content", ""),
                "priority": skill_info.get("priority", 0),
                "tools": skill_info.get("tools", []),
            }
        except Exception as e:
            logger.debug("Failed to load skill command %s: %s", name, e)

    return _skill_commands


def get_skill_command(name: str) -> Optional[Dict[str, Any]]:
    """获取指定名称的技能命令。

    Args:
        name: 命令名（不含前导斜杠）

    Returns:
        命令信息字典，不存在则返回 None
    """
    commands = get_skill_commands()
    return commands.get(name)


def preprocess_skill_content(content: str, variables: Optional[Dict[str, str]] = None,
                              cwd: Optional[str] = None) -> str:
    """预处理技能内容：模板变量替换 + 内联 shell 展开。

    Args:
        content: 原始技能内容
        variables: 模板变量
        cwd: 工作目录（用于 shell 展开）

    Returns:
        处理后的内容
    """
    # 1. 模板变量替换
    content = substitute_template_vars(content, variables)

    # 2. 内联 shell 展开
    content = expand_inline_shell(content, cwd)

    return content


def build_skill_system_prompt(skill_name: str, variables: Optional[Dict[str, str]] = None) -> Optional[str]:
    """构建技能的系统提示词。

    Args:
        skill_name: 技能名称
        variables: 模板变量

    Returns:
        系统提示词字符串，技能不存在则返回 None
    """
    command = get_skill_command(skill_name)
    if not command:
        return None

    content = command.get("content", "")
    content = preprocess_skill_content(content, variables)

    return content
