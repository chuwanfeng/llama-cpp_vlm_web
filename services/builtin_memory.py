"""
内置记忆提供者 — 基于 SessionStore 的 SQLite 持久化记忆

从 hermes-agent 的 BuiltinMemoryProvider 完整移植，适配项目 SessionStore。
提供 key-value 记忆存储、全文搜索、自动去重和更新。
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from .memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# ─── 去重相关常量 ──────────────────────────────────────────────
_MIN_KEY_LEN = 3
_MIN_VALUE_LEN = 10
_SIMILARITY_THRESHOLD = 0.7


def _simple_similarity(a: str, b: str) -> float:
    """简单的词重叠相似度"""
    if not a or not b:
        return 0.0
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


class BuiltinMemoryProvider(MemoryProvider):
    """内置记忆提供者 — 统一管理项目的 key-value 记忆存储。

    基于 SessionStore 的 SQLite 持久化，支持全文搜索和自动去重。
    完整保留 hermes-agent BuiltinMemoryProvider 的全部功能。
    """

    @property
    def name(self) -> str:
        return "builtin"

    def __init__(self, store) -> None:
        """store: SessionStore 实例"""
        from .session_store import SessionStore as _  # noqa: F811
        self._store = store
        self._session_id: str = ""

    # -- 核心生命周期 ----------------------------------------------------------

    def is_available(self) -> bool:
        return self._store is not None

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        logger.info("BuiltinMemoryProvider initialized (session=%s)", session_id)

    def system_prompt_block(self) -> str:
        return (
            "You have access to a persistent memory system via the `memory` tool. "
            "Use `memory` to store important information, user preferences, "
            "decisions, and facts that should persist across sessions. "
            "Use `search_memory` to recall previously stored information "
            "when the user asks about something you should know. "
            "Always search memory BEFORE guessing or making up information "
            "about the user or their preferences."
        )

    # -- 记忆 CRUD -------------------------------------------------------------

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._store.get_memory(key)

    def set(self, key: str, value: str, category: str = "general",
            *, deduplicate: bool = True) -> Dict[str, Any]:
        """写入或更新记忆，支持自动去重

        返回 {'status': 'added'|'duplicate'|'replaced', ...}
        """
        key = (key or "").strip()
        value = (value or "").strip()
        if not key or not value:
            return {"status": "error", "reason": "key and value are required"}
        if len(key) > 255:
            key = key[:255]
        if len(key) < _MIN_KEY_LEN:
            return {"status": "error", "reason": f"key too short (min {_MIN_KEY_LEN})"}

        existing = self.get(key)
        if existing:
            if existing.get("value") == value:
                return {"status": "duplicate", "key": key, "reason": "identical"}
            self._store.save_memory(key, value, category)
            return {"status": "replaced", "key": key, "previous": existing.get("value")}

        # 去重
        if deduplicate:
            all_items = self._store.list_memory()
            for item in all_items:
                sim = _simple_similarity(value, item.get("value", ""))
                if sim >= _SIMILARITY_THRESHOLD:
                    self._store.save_memory(item["key"], value, category)
                    return {"status": "deduplicated", "key": item["key"],
                            "similar_to": key, "similarity": round(sim, 2)}

        self._store.save_memory(key, value, category)
        return {"status": "added", "key": key}

    def delete(self, key: str) -> bool:
        return self._store.delete_memory(key)

    def list_all(self, category: Optional[str] = None,
                 limit: int = 50) -> List[Dict[str, Any]]:
        items = self._store.list_memory(category) if category else self._store.list_memory()
        return items[:limit]

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return self.list_all(limit=limit)
        return self._store.search_memory(query, limit)

    # -- MemoryProvider 接口实现 ------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """每轮对话前搜索相关记忆并返回格式化上下文"""
        if not query or not query.strip():
            return ""
        results = self.search(query, limit=5)
        if not results:
            return ""
        lines = ["[以下是与此对话相关的已存储记忆——作为背景参考，非用户输入]"]
        for m in results:
            val = str(m.get("value", ""))[:200]
            lines.append(f"- [{m.get('category', 'general')}] {m.get('key', '')}: {val}")
        return "\n".join(lines)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """每轮后同步——自动提取显式记忆命令 [REMEMBER: key] value"""
        pattern = r'\[REMEMBER:\s*([^\]]+)\]\s*(.+?)(?=\[REMEMBER:|\Z)'
        matches = re.findall(pattern, assistant_content, re.DOTALL | re.IGNORECASE)
        for key, value in matches:
            key = key.strip()
            value = value.strip()
            if key and value:
                self.set(key, value, category="auto")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "memory",
                "description": "管理持久化记忆。使用此工具存储、更新、删除和搜索长期记忆。"
                               "支持 add/replace/remove/search 四种操作。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["add", "replace", "remove", "search"],
                                   "description": "操作类型"},
                        "key": {"type": "string", "description": "记忆键名"},
                        "value": {"type": "string", "description": "记忆值"},
                        "query": {"type": "string", "description": "搜索查询"},
                    },
                    "required": ["action"],
                },
            }
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != "memory":
            raise NotImplementedError(f"Unknown tool: {tool_name}")
        action = args.get("action", "")
        try:
            if action == "add":
                return json.dumps(self.set(args.get("key", ""), args.get("value", "")),
                                  ensure_ascii=False)
            if action == "replace":
                return json.dumps(self.set(args.get("key", ""), args.get("value", ""),
                                           deduplicate=False), ensure_ascii=False)
            if action == "remove":
                ok = self.delete(args.get("key", ""))
                return json.dumps({"status": "removed" if ok else "not_found"}, ensure_ascii=False)
            if action == "search":
                return json.dumps({"results": self.search(args.get("query", ""))}, ensure_ascii=False)
            return json.dumps({"status": "error", "reason": f"unknown action: {action}"})
        except Exception as e:
            logger.error("Memory tool error: %s", e)
            return json.dumps({"status": "error", "reason": str(e)})

    def shutdown(self) -> None:
        logger.info("BuiltinMemoryProvider shutdown")
