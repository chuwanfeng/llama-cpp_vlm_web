"""可配置的工具输出截断限制。

移植自 hermes-agent/tools/tool_output_limits.py。

将硬编码的截断阈值集中到一个配置节（settings.json 中的
tool_output），高级用户可以不修改源码直接调整。

示例 settings.json：
    {"tool_output": {"max_bytes": 100000, "max_lines": 5000,
     "max_line_length": 2000}}

限制读取器是防御性的：任何错误（配置文件缺失、无效值类型等）
都会回退到内置默认值，不会因为配置文件错误导致工具失败。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Hardcoded defaults — these match the pre-existing values, so adding
# this module is behaviour-preserving for users who don't set
# ``tool_output`` in settings.json.
DEFAULT_MAX_BYTES = 50_000       # terminal_tool.MAX_OUTPUT_CHARS
DEFAULT_MAX_LINES = 2000         # file_operations.MAX_LINES
DEFAULT_MAX_LINE_LENGTH = 2000   # file_operations.MAX_LINE_LENGTH


def _resolve_project_root() -> Path | None:
    """Resolve the project root directory (containing settings.json).

    Tries:
        1. ``VLM_PROJECT_ROOT`` environment variable.
        2. Walk upward from this file's location (``tools/`` → parent).
        3. Current working directory.
    """
    if env_root := os.getenv("VLM_PROJECT_ROOT"):
        return Path(env_root)
    # Walk up from this file's directory
    try:
        candidate = Path(__file__).resolve().parent.parent
        if (candidate / "settings.json").exists():
            return candidate
    except Exception:
        pass
    # Fallback: CWD
    cwd = Path.cwd()
    if (cwd / "settings.json").exists():
        return cwd
    return None


def _load_tool_output_config() -> Dict[str, Any]:
    """Try to load ``tool_output`` section from settings.json.

    Returns ``{}`` on any error — this function NEVER raises.
    """
    try:
        root = _resolve_project_root()
        if root is None:
            return {}
        settings_path = root / "settings.json"
        with open(settings_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            return {}
        section = cfg.get("tool_output")
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load tool_output config: %s", exc)
        return {}


def _coerce_positive_int(value: Any, default: int) -> int:
    """Return ``value`` as a positive int, or ``default`` on any issue."""
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    if iv <= 0:
        return default
    return iv


def get_tool_output_limits() -> Dict[str, int]:
    """Return resolved tool-output limits, reading ``tool_output`` from config.

    Keys: ``max_bytes``, ``max_lines``, ``max_line_length``. Missing or
    invalid entries fall through to the ``DEFAULT_*`` constants. This
    function NEVER raises.
    """
    section = _load_tool_output_config()
    return {
        "max_bytes": _coerce_positive_int(
            section.get("max_bytes"), DEFAULT_MAX_BYTES
        ),
        "max_lines": _coerce_positive_int(
            section.get("max_lines"), DEFAULT_MAX_LINES
        ),
        "max_line_length": _coerce_positive_int(
            section.get("max_line_length"), DEFAULT_MAX_LINE_LENGTH
        ),
    }


def get_max_bytes() -> int:
    """Shortcut for terminal-tool callers that only need the byte cap."""
    return get_tool_output_limits()["max_bytes"]


def get_max_lines() -> int:
    """Shortcut for file-ops callers that only need the line cap."""
    return get_tool_output_limits()["max_lines"]


def get_max_line_length() -> int:
    """Shortcut for file-ops callers that only need the per-line cap."""
    return get_tool_output_limits()["max_line_length"]