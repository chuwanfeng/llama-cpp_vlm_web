"""项目记忆/文档搜索工具。

允许 Agent 搜索自己的知识库（MEMORY.md、memory/*.md 文件），
获取过往决策、用户偏好、项目历史和技术积累等上下文。

核心设计：
    - 记忆文件来源：项目根目录 MEMORY.md + memory/*.md
    - 搜索方式：关键词匹配（含日期过滤）
    - 返回格式：[{path, content, date}, ...]
"""

import logging
import os
import re
from typing import List, Dict, Any

from tools.registry import get_registry

logger = logging.getLogger(__name__)

# Project root and memory paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_DIR = os.path.join(PROJECT_ROOT, "memory")


def _load_memory_files() -> List[Dict[str, Any]]:
    """Load all memory files as a list of {path, content, date}."""
    docs = []
    mem_files = []

    # MEMORY.md at project root
    root_mem = os.path.join(PROJECT_ROOT, "MEMORY.md")
    if os.path.exists(root_mem):
        mem_files.append(root_mem)

    # memory/*.md files
    if os.path.isdir(MEMORY_DIR):
        for f in sorted(os.listdir(MEMORY_DIR), reverse=True):
            if f.endswith(".md"):
                mem_files.append(os.path.join(MEMORY_DIR, f))

    for path in mem_files:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            # Extract date from filename if possible
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
            date = date_match.group(1) if date_match else ""
            docs.append({
                "path": os.path.basename(path),
                "content": content,
                "date": date,
            })
        except Exception as e:
            logger.warning("Could not read memory file %s: %s", path, e)

    return docs


def search_memory(query: str, max_chars: int = 3000) -> str:
    """Search project memory files for relevant information.

    Args:
        query: Search query (keywords or natural language).
        max_chars: Maximum characters to return from each match.

    Returns:
        Relevant memory content.
    """
    docs = _load_memory_files()

    if not docs:
        return "No memory files found in the project."

    # Simple relevance scoring: count keyword matches
    keywords = [k.lower() for k in re.findall(r"[\w\u4e00-\u9fff]+", query)]
    if not keywords:
        keywords = ["memory"]

    scored = []
    for doc in docs:
        content_lower = doc["content"].lower()
        score = sum(content_lower.count(kw) for kw in keywords)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return f"No memory entries found matching: {query}\n\nAvailable memory files: {', '.join(d['path'] for d in docs)}"

    results = []
    for score, doc in scored[:5]:
        # Truncate to max_chars
        content = doc["content"]
        if len(content) > max_chars:
            # Try to find the most relevant section
            best_pos = 0
            best_local_score = 0
            content_lower = content.lower()
            for kw in keywords:
                idx = content_lower.find(kw)
                if idx >= 0:
                    local = content_lower[max(0, idx-200):idx+200].count(kw)
                    if local > best_local_score:
                        best_local_score = local
                        best_pos = max(0, idx - max_chars // 2)

            content = "..." + content[best_pos:best_pos + max_chars] + "..."

        results.append(f"## {doc['path']} (score: {score})\n{content}\n")

    return "\n".join(results)


# ── Register ─────────────────────────────────────────────────────────────────

registry = get_registry()
registry.register(
    name="search_memory",
    description="Search project memory and documentation for past decisions, user preferences, technical learnings, and project history.",
    schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query - keywords or topic to find in memory",
            },
            "max_chars": {
                "type": "integer",
                "description": "Max characters per match (default 3000)",
                "default": 3000,
            },
        },
        "required": ["query"],
    },
    handler=search_memory,
    toolset="memory",
)