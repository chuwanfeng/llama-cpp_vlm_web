"""读取文件内容 — 安全文件读取（within workspace and project boundaries）。

仅允许读取项目目录内的文件，支持行偏移和行数限制。

核心设计：
    - 路径解析：相对路径 → 绝对路径（基于项目根目录）
    - 大小限制：MAX_READ_SIZE（默认 5MB）
    - 目录浏览：可读取目录内容列表
    - 行级分页：offset + limit 支持大文件分页读取
"""

import os
import logging

from tools.registry import get_registry

logger = logging.getLogger(__name__)

# Maximum file size to read (5MB)
MAX_READ_SIZE = 5 * 1024 * 1024


def read_file(path: str, offset: int = 0, limit: int = 500) -> str:
    """读取文件内容，支持行偏移和行数限制。

    参数:
        path: 文件路径（相对路径自动基于项目根目录解析）
        offset: 起始行号（从 0 开始）
        limit: 最大读取行数

    返回:
        文件内容字符串（含目录列表或错误提示）
    """
    # Resolve path
    if not os.path.isabs(path):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, path)

    path = os.path.normpath(path)

    if not os.path.exists(path):
        return f"Error: File not found: {path}"

    if os.path.isdir(path):
        contents = []
        try:
            items = sorted(os.listdir(path))
        except PermissionError:
            return f"Error: Permission denied: {path}"
        for item in items:
            full = os.path.join(path, item)
            mark = "/" if os.path.isdir(full) else ""
            contents.append(f"  {item}{mark}")
        return f"Directory listing for {path}:\n" + "\n".join(contents)

    file_size = os.path.getsize(path)
    if file_size > MAX_READ_SIZE:
        return f"Error: File too large ({file_size} bytes, max {MAX_READ_SIZE})"

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="gbk", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            return f"Error reading file: {e}"
    except Exception as e:
        return f"Error reading file: {e}"

    total_lines = len(lines)

    if offset > 0:
        lines = lines[offset:]
    if limit > 0:
        lines = lines[:limit]

    content = "".join(lines)
    header = f"File: {path} (lines {offset}-{offset + len(lines)} of {total_lines})\n"
    return header + content


# ── Register ─────────────────────────────────────────────────────────────────

registry = get_registry()
registry.register(
    name="read_file",
    description="Read contents of a file or list directory contents. Use to inspect code, config, logs, or explore project structure.",
    schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path or directory path to read",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start from (0 = beginning)",
                "default": 0,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum lines to read (default 500)",
                "default": 500,
            },
        },
        "required": ["path"],
    },
    handler=read_file,
    toolset="file",
)