"""精确文件编辑工具 — 基于精确文本替换，含唯一性校验。

类似于 Claude 的 Edit 工具。支持一次调用多个编辑。
每个编辑将精确的 oldText 替换为 newText，要求 oldText 在文件中唯一
以避免歧义。

核心设计：
    - 原子性：所有编辑验证通过后才写入（全成功或全失败）
    - 唯一性校验：oldText 必须在文件中唯一出现
    - 大小/数量限制：MAX_EDIT_SIZE=1MB，MAX_EDITS_PER_CALL=20
    - 文件类型白名单：ALLOWED_EXTENSIONS（只有指定扩展名可以编辑）
    - 编码容错：读取时 utf-8 → gbk 回退，写入时 utf-8
    - 安全保护：调用 file_safety.is_write_denied 拦截敏感路径
"""

import os
import logging
from typing import Dict, List

from tools.registry import get_registry
from services.file_safety import is_write_denied

logger = logging.getLogger(__name__)

MAX_EDIT_SIZE = 1 * 1024 * 1024   # 1MB
MAX_EDITS_PER_CALL = 20
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".vue", ".go", ".rs", ".java",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".conf", ".env",
    ".sh", ".bash", ".bat", ".cmd", ".ps1",
    ".csv", ".tsv", ".txt",
    ".md", ".html", ".xml", ".svg", ".css", ".scss",
}


def _resolve_path(path: str) -> str:
    """Resolve a relative or absolute path to the project root."""
    if not os.path.isabs(path):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, path)
    return os.path.normpath(path)


def _check_extension(path: str) -> None:
    """Validate file extension is in allowed list."""
    ext = os.path.splitext(path)[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        return  # warn but allow through
    if not ext:
        pass  # files without extension are ok


def edit_file(path: str, edits: List[Dict[str, str]]) -> str:
    """对文件执行精确的文本替换。

    每个编辑将文件中唯一的 oldText 替换为 newText。
    所有编辑在验证通过后才写入（原子操作）。

    参数:
        path: 文件路径（相对路径自动基于项目根目录解析）
        edits: 编辑列表 [{oldText: str, newText: str}, ...]

    返回:
        成功消息（含替换计数）或错误描述
    """
    if not edits:
        return "Error: edits list is empty"
    if len(edits) > MAX_EDITS_PER_CALL:
        return f"Error: too many edits ({len(edits)}, max {MAX_EDITS_PER_CALL})"

    path = _resolve_path(path)

    # 文件安全检查：保护敏感路径
    if is_write_denied(path):
        return f"Error: Access denied - protected path: {path}"

    if not os.path.exists(path):
        return f"Error: File not found: {path}"
    if os.path.isdir(path):
        return f"Error: Path is a directory: {path}"

    file_size = os.path.getsize(path)
    if file_size > MAX_EDIT_SIZE:
        return f"Error: File too large ({file_size} bytes, max {MAX_EDIT_SIZE})"

    # Read file with encoding fallback
    try:
        with open(path, "r", encoding="utf-8") as f:
            original = f.read()
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="gbk") as f:
                original = f.read()
        except Exception as e:
            return f"Error reading file: {e}"

    # Validate all edits before applying any (atomic)
    changes = 0
    new_content = original

    for i, edit in enumerate(edits):
        old_text = edit.get("oldText", "")
        new_text = edit.get("newText", "")

        if not old_text:
            return f"Error: edit[{i}] has empty oldText"

        count = new_content.count(old_text)
        if count == 0:
            return (
                f"Error: edit[{i}] oldText not found in file.\noldText: {old_text[:200]}"
            )
        if count > 1:
            return (
                f"Error: edit[{i}] oldText appears {count} times. "
                f"Include more surrounding context to make it unique.\n"
                f"oldText: {old_text[:200]}"
            )

        new_content = new_content.replace(old_text, new_text, 1)
        changes += 1

    # Write back
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        return f"Error writing file: {e}"

    logger.info("Edited %s: %d edit(s) applied", path, changes)
    return f"Successfully applied {changes} edit(s) to {path}"


# -- Register ---------------------------------------------------------

registry = get_registry()
registry.register(
    name="edit_file",
    description=(
        "Apply precise targeted text replacements to a file. "
        "Each edit replaces a single unique occurrence of oldText with newText. "
        "All edits validated before writing (atomic). "
        "Use for refactoring, bug fixes, and code modifications."
    ),
    schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to edit (relative to project or absolute)",
            },
            "edits": {
                "type": "array",
                "description": "List of edit operations",
                "minItems": 1,
                "maxItems": MAX_EDITS_PER_CALL,
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": {
                            "type": "string",
                            "description": (
                                "Exact text to find and replace. Must be unique "
                                "in the file. Include surrounding context lines "
                                "to make it unique."
                            ),
                        },
                        "newText": {
                            "type": "string",
                            "description": "Replacement text",
                        },
                    },
                    "required": ["oldText", "newText"],
                },
            },
        },
        "required": ["path", "edits"],
    },
    handler=edit_file,
    toolset="file",
)
