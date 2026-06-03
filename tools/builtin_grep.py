"""代码搜索工具 — 基于正则表达式的文件内容搜索（带上下文行）。

类似 grep/ripgrep。遍历项目目录，匹配模式，返回带有上下文行的结果。

核心设计：
    - 递归目录遍历（自动跳过 .git/__pycache__/node_modules 等）
    - 文件大小限制：MAX_FILE_SIZE=500KB
    - 结果数量限制：MAX_RESULTS=200
    - 上下文行：围绕匹配行显示前后 N 行
    - Glob 模式过滤：include 参数控制匹配的文件类型（如 *.py）
"""

import fnmatch
import logging
import os
import re
from typing import List, Optional

from tools.registry import get_registry

logger = logging.getLogger(__name__)

MAX_FILES = 500           # Max files to scan
MAX_FILE_SIZE = 500 * 1024  # 500KB per file
MAX_RESULTS = 200         # Max results to return
MAX_MATCH_LEN = 500       # Max chars per match line


def _resolve_path(path: str) -> str:
    """Resolve a relative or absolute path to the project root."""
    if not os.path.isabs(path):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, path)
    return os.path.normpath(path)


def _should_skip_dir(dirname: str) -> bool:
    """Check if a directory should be skipped during search."""
    skip_dirs = {
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        ".tox", ".eggs", "dist", "build", ".mypy_cache",
        ".pytest_cache", ".ruff_cache", ".next", ".nuxt",
        "egg-info", ".egg-info",
    }
    return dirname in skip_dirs or dirname.endswith(".egg-info")


def _collect_files(
    search_path: str, include: str, recursive: bool
) -> List[str]:
    """Collect files matching the include pattern.

    Args:
        search_path: Root directory to search.
        include: Glob pattern for file matching (e.g. "*.py").
        recursive: Whether to search subdirectories.

    Returns:
        List of absolute file paths.
    """
    files = []

    if recursive:
        for root, dirs, filenames in os.walk(search_path):
            # Skip hidden and build directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and not _should_skip_dir(d)]

            for fname in filenames:
                if fnmatch.fnmatch(fname, include):
                    files.append(os.path.join(root, fname))
                    if len(files) >= MAX_FILES:
                        return files
    else:
        try:
            for entry in os.scandir(search_path):
                if entry.is_file() and fnmatch.fnmatch(entry.name, include):
                    files.append(entry.path)
                    if len(files) >= MAX_FILES:
                        return files
        except PermissionError:
            pass

    return files


def grep_files(
    pattern: str,
    path: str = ".",
    include: str = "*.py",
    recursive: bool = True,
    case_sensitive: bool = False,
    context_lines: int = 2,
) -> str:
    """Search files for a regex pattern, returning matches with context.

    Args:
        pattern: Regex pattern to search for.
        path: Root directory to search (default: project root).
        include: File glob pattern (default: "*.py"). Use "*.js" for JS,
                 "*.md" for markdown, "*" for all files.
        recursive: Whether to search subdirectories (default: True).
        case_sensitive: Whether matching is case-sensitive (default: False).
        context_lines: Lines before/after each match to include (default: 2).

    Returns:
        Formatted search results with file paths, line numbers, and context.
    """
    search_path = _resolve_path(path)

    if not os.path.exists(search_path):
        return f"Error: Path not found: {search_path}"
    if not os.path.isdir(search_path):
        return (
            f"Error: path must be a directory for grep. "
            f"Use read_file to read a single file."
        )

    # Compile regex
    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(pattern, flags | re.MULTILINE)
    except re.error as e:
        return f"Error: invalid regex pattern '{pattern}': {e}"

    # Collect files
    files = _collect_files(search_path, include, recursive)
    if not files:
        inc_desc = f"*.{include}" if include.startswith("*") else include
        return f"No {inc_desc} files found in {search_path}"

    # Search each file
    results = []
    files_scanned = 0

    for fpath in files:
        if len(results) >= MAX_RESULTS:
            break

        try:
            fsize = os.path.getsize(fpath)
            if fsize > MAX_FILE_SIZE:
                continue
        except OSError:
            continue

        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception:
            try:
                with open(fpath, "r", encoding="gbk", errors="replace") as f:
                    lines = f.readlines()
            except Exception:
                continue

        files_scanned += 1

        for i, line in enumerate(lines):
            if len(results) >= MAX_RESULTS:
                break

            if not compiled.search(line):
                continue

            # Collect context
            ctx_start = max(0, i - context_lines)
            ctx_end = min(len(lines), i + context_lines + 1)

            result_lines = []
            for j in range(ctx_start, ctx_end):
                prefix = "> " if j == i else "  "
                line_text = lines[j].rstrip("\n\r")
                if len(line_text) > MAX_MATCH_LEN:
                    line_text = line_text[:MAX_MATCH_LEN] + "..."
                result_lines.append(f"{prefix}{j+1}: {line_text}")

            rel_path = os.path.relpath(fpath, search_path)
            results.append(f"\n--- {rel_path} ---\n" + "\n".join(result_lines))
    if not results:
        return (
            f"No matches for '{pattern}' in {files_scanned} files "
            f"(include={include})"
        )

    header = (
        f"Found {len(results)} match(es) for '{pattern}' "
        f"in {files_scanned} files (include={include}, "
        f"recursive={recursive}, case_sensitive={case_sensitive})"
    )
    if len(files) >= MAX_FILES:
        header += f" [scan limit {MAX_FILES}]"

    return header + "\n" + "\n".join(results)


# -- Register ---------------------------------------------------------

registry = get_registry()
registry.register(
    name="grep_files",
    description=(
        "Search file contents with regex. Find where functions, classes, "
        "variables, imports, or patterns are defined/used. Returns matching "
        "lines with surrounding context. Use to navigate unfamiliar codebases."
    ),
    schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "Directory to search (default: project root)",
                "default": ".",
            },
            "include": {
                "type": "string",
                "description": "File glob pattern. *.py for Python, *.js for JS, * for all (default: *.py)",
                "default": "*.py",
            },
            "recursive": {
                "type": "boolean",
                "description": "Search subdirectories recursively (default: true)",
                "default": True,
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Case-sensitive matching (default: false)",
                "default": False,
            },
            "context_lines": {
                "type": "integer",
                "description": "Lines before/after each match to show (default: 2)",
                "default": 2,
            },
        },
        "required": ["pattern"],
    },
    handler=grep_files,
    toolset="file",
)
