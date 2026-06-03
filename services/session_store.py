"""
会话持久化存储 — SQLite + FTS5 全文搜索

从 hermes-agent 的 hermes_state.py 提取并简化，适配 Flask Web 聊天场景。
提供会话 CRUD、消息存储、FTS5 搜索（含 trigram 中文支持）、自动 schema 迁移。
"""
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("session-store")

# ─── 中国时区 ──────────────────────────────────────────────
CST = timezone(timedelta(hours=8))

def _now_ts() -> float:
    return time.time()

def _now_iso() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")

# ─── Schema ──────────────────────────────────────────────

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    source TEXT DEFAULT 'web',
    backend TEXT,
    model TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    message_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT,
    tool_name TEXT,
    tool_calls TEXT,
    tool_call_id TEXT,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning_content TEXT,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tools (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    schema_json TEXT NOT NULL,
    handler_type TEXT DEFAULT 'python',
    enabled INTEGER DEFAULT 1,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_memory_category ON memory(category);
"""

# FTS5 + Trigram (CJK 友好)
FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '')
    );
END;
"""

FTS_TRIGRAM_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content,
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '')
    );
END;
"""


class SessionStore:
    """SQLite 会话存储，线程安全（WAL 模式）"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            # 默认在项目目录下
            db_path = os.path.join(os.path.dirname(__file__), "..", "conversations.db")
            db_path = os.path.abspath(db_path)

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=5.0,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._init_schema()

    # ─── Schema Init ─────────────────────────────────────────

    def _init_schema(self):
        cursor = self._conn.cursor()
        cursor.executescript(SCHEMA_SQL)

        # FTS5
        try:
            cursor.execute("SELECT * FROM messages_fts LIMIT 0")
        except sqlite3.OperationalError:
            cursor.executescript(FTS_SQL)

        # Trigram FTS5 (CJK)
        try:
            cursor.execute("SELECT * FROM messages_fts_trigram LIMIT 0")
        except sqlite3.OperationalError:
            cursor.executescript(FTS_TRIGRAM_SQL)

        # Schema version
        cursor.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        self._conn.commit()

    def close(self):
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # ─── Sessions ────────────────────────────────────────────

    def create_session(self, title: str = None, backend: str = None, model: str = None) -> str:
        sid = uuid.uuid4().hex[:16]
        now = _now_ts()
        with self._lock:
            self._conn.execute(
                """INSERT INTO sessions (id, title, backend, model, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (sid, title or _now_iso(), backend, model, now, now),
            )
            self._conn.commit()
        logger.info("创建会话 %s", sid)
        return sid

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
        return dict(row) if row else None

    def list_sessions(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            cursor = self._conn.execute(
                """SELECT s.*, COALESCE(
                       (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A',' '), X'0D',' '), 1, 80)
                        FROM messages m WHERE m.session_id = s.id AND m.role = 'user'
                        ORDER BY m.timestamp LIMIT 1), ''
                   ) AS preview
                   FROM sessions s
                   ORDER BY s.updated_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            )
            rows = cursor.fetchall()
        return [dict(r) for r in rows]

    def update_session(self, session_id: str, **kwargs):
        allowed = {"title", "model", "backend"}
        fields = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if not fields:
            return
        fields["updated_at"] = _now_ts()
        sets = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [session_id]
        with self._lock:
            self._conn.execute(f"UPDATE sessions SET {sets} WHERE id = ?", vals)
            self._conn.commit()

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    # ─── Messages ────────────────────────────────────────────

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str = None,
        tool_name: str = None,
        tool_calls: Any = None,
        tool_call_id: str = None,
        token_count: int = None,
        finish_reason: str = None,
        reasoning_content: str = None,
    ) -> int:
        """追加消息，返回 message id"""
        # 确保 session 存在
        session = self.get_session(session_id)
        if not session:
            self.create_session(session_id=session_id)

        tool_calls_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
        now = _now_ts()

        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO messages (session_id, role, content, tool_name, tool_calls,
                   tool_call_id, token_count, finish_reason, reasoning_content, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, role, content, tool_name, tool_calls_json,
                    tool_call_id, token_count, finish_reason, reasoning_content, now,
                ),
            )
            msg_id = cursor.lastrowid
            # 更新计数器
            self._conn.execute(
                "UPDATE sessions SET message_count = message_count + 1, updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            self._conn.commit()
        return msg_id

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp, id",
                (session_id,),
            )
            rows = cursor.fetchall()
        result = []
        for row in rows:
            msg = dict(row)
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    msg["tool_calls"] = []
            result.append(msg)
        return result

    def get_messages_as_conversation(self, session_id: str) -> List[Dict[str, Any]]:
        """返回 OpenAI 格式的消息列表"""
        msgs = self.get_messages(session_id)
        conv = []
        for m in msgs:
            msg = {"role": m["role"], "content": m.get("content") or ""}
            if m.get("tool_call_id"):
                msg["tool_call_id"] = m["tool_call_id"]
            if m.get("tool_name"):
                msg["tool_name"] = m["tool_name"]
            if m.get("tool_calls"):
                msg["tool_calls"] = m["tool_calls"]
            if m.get("finish_reason"):
                msg["finish_reason"] = m["finish_reason"]
            if m.get("reasoning_content"):
                msg["reasoning_content"] = m["reasoning_content"]
            conv.append(msg)
        return conv

    def clear_messages(self, session_id: str):
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._conn.execute(
                "UPDATE sessions SET message_count = 0, updated_at = ? WHERE id = ?",
                (_now_ts(), session_id),
            )
            self._conn.commit()

    # ─── Search ──────────────────────────────────────────────

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        for ch in text:
            cp = ord(ch)
            if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
                0x3000 <= cp <= 0x303F or 0x3040 <= cp <= 0x30FF or
                0xAC00 <= cp <= 0xD7AF):
                return True
        return False

    def search_messages(self, query: str, limit: int = 20, session_id: str = None) -> List[Dict[str, Any]]:
        """FTS5 全文搜索消息。中文查询自动走 trigram 索引。"""
        if not query or not query.strip():
            return []

        # 清洗 FTS5 查询（去掉 &"^ 等特殊字符）
        sanitized = re.sub(r'[+&{}()"^~]', ' ', query)
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        if not sanitized:
            return []

        is_cjk = self._contains_cjk(sanitized)

        with self._lock:
            if is_cjk:
                # 中文 → trigram FTS5
                tokens = sanitized.split()
                parts = []
                for tok in tokens:
                    if tok.upper() in ("AND", "OR", "NOT"):
                        parts.append(tok)
                    else:
                        parts.append(f'"{tok}"')
                fts_query = " ".join(parts) if parts else sanitized

                if session_id:
                    sql = """
                        SELECT m.id, m.session_id, m.role,
                               snippet(messages_fts_trigram, 0, '>>>', '<<<', '...', 40) AS snippet,
                               m.content, m.timestamp, m.tool_name,
                               s.title AS session_title
                        FROM messages_fts_trigram
                        JOIN messages m ON m.id = messages_fts_trigram.rowid
                        JOIN sessions s ON s.id = m.session_id
                        WHERE messages_fts_trigram MATCH ? AND m.session_id = ?
                        ORDER BY rank LIMIT ?
                    """
                    params = [fts_query, session_id, limit]
                else:
                    sql = """
                        SELECT m.id, m.session_id, m.role,
                               snippet(messages_fts_trigram, 0, '>>>', '<<<', '...', 40) AS snippet,
                               m.content, m.timestamp, m.tool_name,
                               s.title AS session_title
                        FROM messages_fts_trigram
                        JOIN messages m ON m.id = messages_fts_trigram.rowid
                        JOIN sessions s ON s.id = m.session_id
                        WHERE messages_fts_trigram MATCH ?
                        ORDER BY rank LIMIT ?
                    """
                    params = [fts_query, limit]
            else:
                if session_id:
                    sql = """
                        SELECT m.id, m.session_id, m.role,
                               snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet,
                               m.content, m.timestamp, m.tool_name,
                               s.title AS session_title
                        FROM messages_fts
                        JOIN messages m ON m.id = messages_fts.rowid
                        JOIN sessions s ON s.id = m.session_id
                        WHERE messages_fts MATCH ? AND m.session_id = ?
                        ORDER BY rank LIMIT ?
                    """
                    params = [sanitized, session_id, limit]
                else:
                    sql = """
                        SELECT m.id, m.session_id, m.role,
                               snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet,
                               m.content, m.timestamp, m.tool_name,
                               s.title AS session_title
                        FROM messages_fts
                        JOIN messages m ON m.id = messages_fts.rowid
                        JOIN sessions s ON s.id = m.session_id
                        WHERE messages_fts MATCH ?
                        ORDER BY rank LIMIT ?
                    """
                    params = [sanitized, limit]

            try:
                cursor = self._conn.execute(sql, params)
                matches = [dict(row) for row in cursor.fetchall()]
            except sqlite3.OperationalError:
                matches = []

        # 裁剪 content
        for m in matches:
            if m.get("content") and len(m["content"]) > 300:
                m["content"] = m["content"][:297] + "..."

        return matches

    # ─── Memory ──────────────────────────────────────────────

    def save_memory(self, key: str, value: str, category: str = "general"):
        now = _now_ts()
        with self._lock:
            self._conn.execute(
                """INSERT INTO memory (key, value, category, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                   category=excluded.category, updated_at=excluded.updated_at""",
                (key, value, category, now, now),
            )
            self._conn.commit()

    def get_memory(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM memory WHERE key = ?", (key,))
            row = cursor.fetchone()
        return dict(row) if row else None

    def list_memory(self, category: str = None) -> List[Dict[str, Any]]:
        with self._lock:
            if category:
                cursor = self._conn.execute(
                    "SELECT * FROM memory WHERE category = ? ORDER BY updated_at DESC", (category,)
                )
            else:
                cursor = self._conn.execute("SELECT * FROM memory ORDER BY updated_at DESC")
            rows = cursor.fetchall()
        return [dict(r) for r in rows]

    def search_memory(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """LIKE 搜索记忆"""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM memory WHERE key LIKE ? OR value LIKE ? ORDER BY updated_at DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            )
            rows = cursor.fetchall()
        return [dict(r) for r in rows]

    def delete_memory(self, key: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM memory WHERE key = ?", (key,))
            self._conn.commit()
            return cursor.rowcount > 0

    # ─── Tokens ──────────────────────────────────────────────

    def update_token_counts(self, session_id: str, input_tokens: int = 0, output_tokens: int = 0):
        with self._lock:
            self._conn.execute(
                """UPDATE sessions SET
                   input_tokens = input_tokens + ?,
                   output_tokens = output_tokens + ?,
                   updated_at = ?
                   WHERE id = ?""",
                (input_tokens, output_tokens, _now_ts(), session_id),
            )
            self._conn.commit()

    # ─── Stats ───────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_sessions = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            total_messages = self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            total_memories = self._conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
            total_input = self._conn.execute(
                "SELECT COALESCE(SUM(input_tokens), 0) FROM sessions"
            ).fetchone()[0]
            total_output = self._conn.execute(
                "SELECT COALESCE(SUM(output_tokens), 0) FROM sessions"
            ).fetchone()[0]
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_memories": total_memories,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "db_path": str(self.db_path),
            "db_size_mb": round(self.db_path.stat().st_size / 1024 / 1024, 2) if self.db_path.exists() else 0,
        }

    # ─── Wade Bridge (private context) ───────────────────────
    # 确保 wade 写入 context/memories.md 时，同步到 memory 表
    def sync_context_file(self, context_dir: str = None):
        """扫描 context/*.md 文件，同步到 memory 表（单向，文件优先）"""
        if context_dir is None:
            context_dir = os.path.join(os.path.dirname(__file__), "..", "context")
        if not os.path.isdir(context_dir):
            return 0

        count = 0
        for fname in os.listdir(context_dir):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(context_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    key = fname.replace(".md", "")
                    self.save_memory(key, content, "context-file")
                    count += 1
            except Exception as e:
                logger.warning("同步 context 文件失败 %s: %s", fname, e)
        return count


# ─── 全局单例 ──────────────────────────────────────────────

_store: Optional[SessionStore] = None
_store_lock = threading.Lock()

def get_store(db_path: str = None) -> SessionStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = SessionStore(db_path)
    return _store
