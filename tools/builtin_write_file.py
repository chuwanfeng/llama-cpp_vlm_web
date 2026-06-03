"""文件写入工具 — 安全文件创建（within workspace boundaries）。

支持自动创建父目录、文件大小检查、敏感路径保护。

核心设计：
    - 路径解析：相对路径 → 绝对路径（基于项目根目录）
    - 大小限制：MAX_WRITE_SIZE（默认 1MB）
    - 安全保护：调用 file_safety.is_write_denied 拦截敏感路径（token/env 等）
    - 自动创建目录：os.makedirs(parent, exist_ok=True)
"""

import os
import logging

from services.file_safety import is_write_denied

from tools.registry import get_registry

logger = logging.getLogger(__name__)

# Maximum file size to write (1MB)
MAX_WRITE_SIZE = 1 * 1024 * 1024


def write_file(path: str, content: str) -> str:
    """向文件写入内容，自动创建父目录。

    参数:
        path: 目标文件路径（相对路径自动基于项目根目录解析）
        content: 要写入的内容（UTF-8 编码）

    返回:
        确认信息（成功时含字符数和路径，失败时含错误原因）
    """
    # Resolve path
    if not os.path.isabs(path):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, path)

    path = os.path.normpath(path)

    # 文件安全检查：保护敏感路径
    if is_write_denied(path):
        return f"Error: Access denied - protected path: {path}"

    if os.path.exists(path) and os.path.isdir(path):
        return f"Error: Path is a directory: {path}"

    if len(content.encode("utf-8")) > MAX_WRITE_SIZE:
        return f"Error: Content too large ({len(content)} chars, max {MAX_WRITE_SIZE} bytes)"

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


# ── Register ─────────────────────────────────────────────────────────────────

registry = get_registry()
registry.register(
    name="write_file",
    description="Write content to a file. Creates parent directories automatically. Use to save code, notes, or generated content.",
    schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to write to (relative to project or absolute)",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["path", "content"],
    },
    handler=write_file,
    toolset="file",
)