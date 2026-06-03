"""Todo 工具 — 规划 & 任务管理

提供内存中的任务列表，Agent 用来分解复杂任务、
跟踪进度、在长对话中保持专注。

设计模式:
- 写模式: 传入 'todos' 列表创建/更新任务项
- 读模式: 不传 'todos' 读取当前列表
- 合并模式: merge=true 按 id 更新已有项，追加新项
- 每次调用返回完整列表及摘要统计

移植自 hermes-agent/tools/todo_tool.py
"""

import json
import threading
from typing import Any, Dict, List, Optional

from tools.registry import get_registry

# Valid status values for todo items
VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}

# Status markers for display
STATUS_MARKERS = {
    "pending": "⬜",
    "in_progress": "🔄",
    "completed": "✅",
    "cancelled": "❌",
}


class TodoStore:
    """
    内存任务列表。线程安全单例，每个进程一个实例。

    任务项按列表顺序排列（位置即优先级）。每个任务项包含:
      - id: 唯一字符串标识符（由 Agent 分配）
      - content: 任务描述
      - status: pending | in_progress | completed | cancelled
    """

    _instance: Optional["TodoStore"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "TodoStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._items: List[Dict[str, str]] = []
        return cls._instance

    def write(
        self, todos: List[Dict[str, Any]], merge: bool = False
    ) -> List[Dict[str, str]]:
        """
        Write todos. Returns the full current list after writing.

        Args:
            todos: list of {id, content, status} dicts
            merge: if False, replace the entire list. If True, update
                   existing items by id and append new ones.
        """
        if not merge:
            # Replace mode: new list entirely
            self._items = [self._validate(t) for t in self._dedupe_by_id(todos)]
        else:
            # Merge mode: update existing items by id, append new ones
            existing = {item["id"]: item for item in self._items}
            for t in self._dedupe_by_id(todos):
                item_id = str(t.get("id", "")).strip()
                if not item_id:
                    continue  # Can't merge without an id

                if item_id in existing:
                    # Update only the fields actually provided
                    if "content" in t and t["content"]:
                        existing[item_id]["content"] = str(t["content"]).strip()
                    if "status" in t and t["status"]:
                        status = str(t["status"]).strip().lower()
                        if status in VALID_STATUSES:
                            existing[item_id]["status"] = status
                else:
                    # New item -- validate fully and append to end
                    validated = self._validate(t)
                    existing[validated["id"]] = validated
                    self._items.append(validated)
            # Rebuild _items preserving order for existing items
            seen = set()
            rebuilt = []
            for item in self._items:
                current = existing.get(item["id"], item)
                if current["id"] not in seen:
                    rebuilt.append(current)
                    seen.add(current["id"])
            self._items = rebuilt
        return self.read()

    def read(self) -> List[Dict[str, str]]:
        """Return a copy of the current list."""
        return [item.copy() for item in self._items]

    def has_items(self) -> bool:
        """Check if there are any items in the list."""
        return bool(self._items)

    def get_active_items(self) -> List[Dict[str, str]]:
        """Return only pending and in_progress items."""
        return [
            item for item in self._items
            if item["status"] in ("pending", "in_progress")
        ]

    def format_for_injection(self) -> Optional[str]:
        """
        Render active todo items for post-compression injection.

        Returns a human-readable string, or None if list is empty.
        """
        active = self.get_active_items()
        if not active:
            return None

        lines = [
            "[Your active task list was preserved across context compression]"
        ]
        for item in active:
            marker = STATUS_MARKERS.get(item["status"], "❓")
            lines.append(
                f"- {marker} {item['id']}. {item['content']} ({item['status']})"
            )
        return "\n".join(lines)

    @staticmethod
    def _validate(item: Dict[str, Any]) -> Dict[str, str]:
        """Validate and normalize a todo item."""
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            item_id = "?"

        content = str(item.get("content", "")).strip()
        if not content:
            content = "(no description)"

        status = str(item.get("status", "pending")).strip().lower()
        if status not in VALID_STATUSES:
            status = "pending"

        return {"id": item_id, "content": content, "status": status}

    @staticmethod
    def _dedupe_by_id(todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse duplicate ids, keeping the last occurrence in its position."""
        last_index: Dict[str, int] = {}
        for i, item in enumerate(todos):
            item_id = str(item.get("id", "")).strip() or "?"
            last_index[item_id] = i
        return [todos[i] for i in sorted(last_index.values())]


# Module-level store instance
_store = TodoStore()


def todo(
    todos: Optional[List[Dict[str, Any]]] = None,
    merge: Optional[bool] = None,
) -> str:
    """
    Manage task list. Writes when 'todos' is provided, reads when omitted.

    Args:
        todos: if provided, write these items. If None, read current list.
        merge: if True, update by id. If False (default), replace entire list.

    Returns:
        JSON string with the full current list and summary metadata.
    """
    if todos is not None:
        items = _store.write(todos, merge or False)
    else:
        items = _store.read()

    # Build summary counts
    pending = sum(1 for i in items if i["status"] == "pending")
    in_progress = sum(1 for i in items if i["status"] == "in_progress")
    completed = sum(1 for i in items if i["status"] == "completed")
    cancelled = sum(1 for i in items if i["status"] == "cancelled")

    return json.dumps({
        "todos": items,
        "summary": {
            "total": len(items),
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "cancelled": cancelled,
        },
    }, ensure_ascii=False)


# =============================================================================
# JSON Schema for todo tool parameters (name/description handled by registry)
TODO_SCHEMA = {
    "type": "object",
    "properties": {
        "todos": {
            "type": "array",
            "description": "Task items to write. Omit to read current list.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique item identifier"
                    },
                    "content": {
                        "type": "string",
                        "description": "Task description"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "cancelled"],
                        "description": "Current status"
                    }
                },
                "required": ["id", "content", "status"]
            }
        },
        "merge": {
            "type": "boolean",
            "description": (
                "true: update existing items by id, add new ones. "
                "false (default): replace the entire list."
            ),
            "default": False
        }
    },
    "required": []
}


# -- Register ---------------------------------------------------------

registry = get_registry()
registry.register(
    name="todo",
    toolset="planning",
    schema=TODO_SCHEMA,
    handler=todo,
)