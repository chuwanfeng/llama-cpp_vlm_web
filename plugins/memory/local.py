# -*- coding: utf-8 -*-
"""
本地 SQLite FTS5 记忆插件

基于 SQLite FTS5 的全文搜索记忆存储。
无需外部依赖，开箱即用。
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import MemoryPlugin

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "memory.db"


class LocalMemoryPlugin(MemoryPlugin):
    """本地 SQLite FTS5 记忆插件"""

    name = "local_memory"
    description = "本地 SQLite FTS5 记忆存储"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.db_path = Path(self.config.get("db_path", str(DEFAULT_DB_PATH)))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        """初始化数据库和表"""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row

        # 创建记忆表（含 FTS5 虚拟表）
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT,
                user_id TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                content='memories',
                content_rowid='rowid'
            );
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            END;
        """)
        self._conn.commit()

    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """存储记忆"""
        try:
            mem_id = str(uuid.uuid4())[:8]
            self._conn.execute(
                "INSERT INTO memories (id, content, metadata, created_at, user_id) VALUES (?, ?, ?, ?, ?)",
                (
                    mem_id,
                    content,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                    self.user_id or "anonymous",
                ),
            )
            self._conn.commit()
            return True
        except Exception as e:
            logger.error("存储记忆失败: %s", e)
            return False

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """FTS5 全文搜索记忆"""
        try:
            cursor = self._conn.execute(
                """
                SELECT m.id, m.content, m.metadata, m.created_at,
                       rank AS score
                FROM memories_fts
                JOIN memories m ON memories_fts.rowid = m.rowid
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            )
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row["id"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"] or "{}"),
                    "created_at": row["created_at"],
                    "score": row["score"],
                })
            return results
        except Exception as e:
            logger.error("搜索记忆失败: %s", e)
            return []

    def get_user_profile(self) -> Optional[Dict[str, Any]]:
        """获取用户画像（基于记忆统计）"""
        try:
            cursor = self._conn.execute(
                "SELECT COUNT(*) as count FROM memories WHERE user_id = ?",
                (self.user_id or "anonymous",),
            )
            count = cursor.fetchone()["count"]
            return {"memory_count": count, "user_id": self.user_id or "anonymous"}
        except Exception as e:
            logger.error("获取用户画像失败: %s", e)
            return None

    def shutdown(self):
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None
