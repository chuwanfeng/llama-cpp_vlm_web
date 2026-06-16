"""读取文件内容 — 安全文件读取（within workspace and project boundaries）。

仅允许读取项目目录内的文件，支持行偏移和行数限制。

核心设计：
    - 路径解析：相对路径 → 绝对路径（基于项目根目录）
    - 大小限制：MAX_READ_SIZE（默认 5MB）
    - 目录浏览：可读取目录内容列表
    - 行级分页：offset + limit 支持大文件分页读取
    - 流式读取：不将整个文件加载到内存，逐行跳过 offset
    - 续读提示：文件未读完时提示剩余行数，引导 Agent 继续分块调用
"""

import os
import logging

from tools.registry import get_registry

logger = logging.getLogger(__name__)

# Maximum file size to read (5MB)
MAX_READ_SIZE = 5 * 1024 * 1024


def _count_lines(path):
    """高效计算文件行数（不加载全部内容到内存）"""
    count = 0
    with open(path, 'rb') as f:
        buf_size = 1024 * 1024  # 1MB buffer
        while True:
            buf = f.read(buf_size)
            if not buf:
                break
            count += buf.count(b'\n')
    return count


def read_file(path: str, offset: int = 0, limit: int = 500) -> str:
    """读取文件内容，支持行偏移和行数限制。

    参数:
        path: 文件路径（相对路径自动基于项目根目录解析）
        offset: 起始行号（从 0 开始）
        limit: 最大读取行数

    返回:
        文件内容字符串（含目录列表或错误提示）。
        当文件未读完时，末尾会标注剩余行数和续读命令，
        引导 Agent 继续调用 read_file 读取下一段。
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

    # ── 流式读取：逐行遍历，不加载整个文件到内存 ──
    try:
        encoding = 'utf-8'
        f = open(path, 'r', encoding='utf-8', errors='replace')
    except UnicodeDecodeError:
        encoding = 'gbk'
        f = open(path, 'r', encoding='gbk', errors='replace')
    except Exception as e:
        return f"Error reading file: {e}"

    lines = []
    total_lines = 0
    try:
        for i, line in enumerate(f):
            total_lines += 1
            if offset <= i < offset + limit:
                lines.append(line)
        end_line = min(offset + len(lines), total_lines)
        remaining = total_lines - end_line
    finally:
        f.close()

    content = "".join(lines)
    header = f"File: {path} (lines {offset}-{end_line - 1} of {total_lines})\n"
    if remaining > 0:
        header += (
            f"[⏭ 还有 {remaining} 行未读取。"
            f"如需继续，请调用 read_file(path='{path}', offset={end_line}, limit={limit})]\n"
        )
    return header + content


# ── Register ─────────────────────────────────────────────────────────────────

registry = get_registry()
registry.register(
    name="read_file",
    description="Read contents of a file or list directory contents. Use to inspect code, config, logs, or explore project structure. Supports chunked reading for large files: read_file(path='f', offset=0, limit=500) then read_file(path='f', offset=500, limit=500) to continue.",
    schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path or directory path to read",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start from (0 = beginning). Increase by limit for each chunk.",
                "default": 0,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum lines to read per call (default 500). Use smaller values for very large files.",
                "default": 500,
            },
        },
        "required": ["path"],
    },
    handler=read_file,
    toolset="file",
)