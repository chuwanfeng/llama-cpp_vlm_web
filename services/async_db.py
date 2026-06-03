"""services/async_db.py — 异步数据库层

提供 SQLite 的异步封装，支持：
- 异步 CRUD 操作
- 连接池
- 自动迁移
- 上下文管理器

使用方式：
    from services.async_db import get_db
    
    async with get_db() as db:
        result = await db.fetch_one("SELECT * FROM sessions WHERE id = ?", (sid,))
"""

import asyncio
import aiosqlite
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("async_db")


@dataclass
class DBConfig:
    """数据库配置"""
    database: str = "data/app.db"
    max_connections: int = 10
    timeout: float = 30.0


class AsyncDatabase:
    """异步数据库连接池"""

    def __init__(self, config: DBConfig = None):
        self.config = config or DBConfig()
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_connections)
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """初始化连接池"""
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            for _ in range(self.config.max_connections):
                conn = await aiosqlite.connect(
                    self.config.database,
                    timeout=self.config.timeout,
                )
                conn.row_factory = aiosqlite.Row
                await self._pool.put(conn)
            
            self._initialized = True
            logger.info("数据库连接池初始化完成: %s", self.config.database)

    async def close(self):
        """关闭所有连接"""
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()
        self._initialized = False
        logger.info("数据库连接池已关闭")

    @asynccontextmanager
    async def acquire(self):
        """获取连接上下文"""
        await self.initialize()
        conn = await self._pool.get()
        try:
            yield conn
        finally:
            await self._pool.put(conn)

    # ── 查询 API ────────────────────────────────────────────────

    async def fetch_one(self, sql: str, params: tuple = ()) -> Optional[Dict]:
        """查询单条记录"""
        async with self.acquire() as conn:
            async with conn.execute(sql, params) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def fetch_all(self, sql: str, params: tuple = ()) -> List[Dict]:
        """查询多条记录"""
        async with self.acquire() as conn:
            async with conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def execute(self, sql: str, params: tuple = ()) -> int:
        """执行 SQL，返回影响行数"""
        async with self.acquire() as conn:
            cursor = await conn.execute(sql, params)
            await conn.commit()
            return cursor.rowcount

    async def execute_many(self, sql: str, params_list: List[tuple]) -> int:
        """批量执行 SQL"""
        async with self.acquire() as conn:
            await conn.executemany(sql, params_list)
            await conn.commit()
            return len(params_list)

    async def insert(self, sql: str, params: tuple = ()) -> int:
        """插入记录，返回自增 ID"""
        async with self.acquire() as conn:
            cursor = await conn.execute(sql, params)
            await conn.commit()
            return cursor.lastrowid

    # ── 迁移 ────────────────────────────────────────────────────

    async def migrate(self):
        """执行数据库迁移"""
        migrations = [
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                messages TEXT,
                created_at REAL,
                updated_at REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                session_id TEXT,
                created_at REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """,
        ]
        
        async with self.acquire() as conn:
            for sql in migrations:
                await conn.execute(sql)
            await conn.commit()
        
        logger.info("数据库迁移完成")


# 全局实例
_db_instance: Optional[AsyncDatabase] = None


async def get_db() -> AsyncDatabase:
    """获取全局数据库实例"""
    global _db_instance
    if _db_instance is None:
        _db_instance = AsyncDatabase()
        await _db_instance.initialize()
        await _db_instance.migrate()
    return _db_instance


async def close_db():
    """关闭全局数据库"""
    global _db_instance
    if _db_instance:
        await _db_instance.close()
        _db_instance = None
