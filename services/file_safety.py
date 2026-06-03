"""文件安全规则 — 保护敏感路径不被写入或读取。

移植自 hermes-agent/agent/file_safety.py，适配 llama-cpp_vlm_web 项目。

用途:
    from services.file_safety import is_write_denied
    if is_write_denied(path):
        return "Error: Access denied - protected path"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _project_root() -> Path:
    """获取项目根目录（避免循环导入）。"""
    try:
        from config import PROJECT_ROOT
        return PROJECT_ROOT
    except Exception:
        # 回退：从当前文件向上推算
        return Path(__file__).resolve().parent.parent


def build_write_denied_paths(home: str) -> set[str]:
    """返回绝对不允许写入的敏感文件路径集合。"""
    project_root = Path(_project_root())
    return {
        os.path.realpath(p)
        for p in [
            os.path.join(home, ".ssh", "authorized_keys"),
            os.path.join(home, ".ssh", "id_rsa"),
            os.path.join(home, ".ssh", "id_ed25519"),
            os.path.join(home, ".ssh", "config"),
            os.path.join(home, ".bashrc"),
            os.path.join(home, ".zshrc"),
            os.path.join(home, ".profile"),
            os.path.join(home, ".bash_profile"),
            os.path.join(home, ".zprofile"),
            os.path.join(home, ".netrc"),
            os.path.join(home, ".pgpass"),
            os.path.join(home, ".npmrc"),
            os.path.join(home, ".pypirc"),
            str(project_root / ".env"),
            str(project_root / "settings.json"),
            "/etc/sudoers",
            "/etc/passwd",
            "/etc/shadow",
        ]
    }


def build_write_denied_prefixes(home: str) -> list[str]:
    """返回不允许写入的敏感目录前缀列表。"""
    return [
        os.path.realpath(p) + os.sep
        for p in [
            os.path.join(home, ".ssh"),
            os.path.join(home, ".aws"),
            os.path.join(home, ".gnupg"),
            os.path.join(home, ".kube"),
            os.path.join(home, ".docker"),
            os.path.join(home, ".azure"),
            os.path.join(home, ".config", "gh"),
            "/etc/sudoers.d",
            "/etc/systemd",
        ]
    ]


def get_safe_write_root() -> Optional[str]:
    """返回 LLMWEB_WRITE_SAFE_ROOT 环境变量值，未设置则返回 None。"""
    root = os.getenv("LLMWEB_WRITE_SAFE_ROOT", "")
    if not root:
        return None
    try:
        return os.path.realpath(os.path.expanduser(root))
    except Exception:
        return None


def is_write_denied(path: str) -> bool:
    """检查路径是否被写入黑名单拦截。

    Args:
        path: 要检查的文件路径

    Returns:
        True 表示该路径被禁止写入
    """
    home = os.path.realpath(os.path.expanduser("~"))
    resolved = os.path.realpath(os.path.expanduser(str(path)))

    # 精确匹配
    if resolved in build_write_denied_paths(home):
        return True

    # 前缀匹配（目录级）
    for prefix in build_write_denied_prefixes(home):
        if resolved.startswith(prefix):
            return True

    # 安全写入根检查
    safe_root = get_safe_write_root()
    if safe_root and not (resolved == safe_root or resolved.startswith(safe_root + os.sep)):
        return True

    return False