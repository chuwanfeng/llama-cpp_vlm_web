"""tests/test_async_db.py -- 异步数据库测试"""

import pytest
import pytest_asyncio
import asyncio
import os
import tempfile
from services.async_db import AsyncDatabase, DBConfig


class TestAsyncDatabase:
    @pytest_asyncio.fixture
    async def db(self):
        """创建临时数据库"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        config = DBConfig(database=db_path)
        database = AsyncDatabase(config)
        await database.initialize()
        await database.migrate()
        
        yield database
        
        await database.close()
        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_insert_and_fetch_one(self, db):
        await db.execute(
            "INSERT INTO sessions (id, title, messages) VALUES (?, ?, ?)",
            ("test-1", "Test", "[]")
        )
        
        result = await db.fetch_one(
            "SELECT * FROM sessions WHERE id = ?",
            ("test-1",)
        )
        
        assert result is not None
        assert result["id"] == "test-1"
        assert result["title"] == "Test"

    @pytest.mark.asyncio
    async def test_fetch_all(self, db):
        await db.execute(
            "INSERT INTO sessions (id, title, messages) VALUES (?, ?, ?)",
            ("test-1", "Test 1", "[]")
        )
        await db.execute(
            "INSERT INTO sessions (id, title, messages) VALUES (?, ?, ?)",
            ("test-2", "Test 2", "[]")
        )
        
        results = await db.fetch_all("SELECT * FROM sessions ORDER BY id")
        
        assert len(results) == 2
        assert results[0]["id"] == "test-1"
        assert results[1]["id"] == "test-2"

    @pytest.mark.asyncio
    async def test_execute_many(self, db):
        params = [
            ("test-1", "Test 1", "[]"),
            ("test-2", "Test 2", "[]"),
            ("test-3", "Test 3", "[]"),
        ]
        
        count = await db.execute_many(
            "INSERT INTO sessions (id, title, messages) VALUES (?, ?, ?)",
            params
        )
        
        assert count == 3
        
        results = await db.fetch_all("SELECT * FROM sessions")
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_insert_return_id(self, db):
        id1 = await db.insert(
            "INSERT INTO memories (content, session_id) VALUES (?, ?)",
            ("Memory 1", "session-1")
        )
        id2 = await db.insert(
            "INSERT INTO memories (content, session_id) VALUES (?, ?)",
            ("Memory 2", "session-1")
        )
        
        assert id1 == 1
        assert id2 == 2

    @pytest.mark.asyncio
    async def test_concurrent_access(self, db):
        """测试并发访问"""
        async def insert_one(i):
            await db.execute(
                "INSERT INTO sessions (id, title, messages) VALUES (?, ?, ?)",
                (f"test-{i}", f"Test {i}", "[]")
            )
        
        # 并发插入 10 条
        await asyncio.gather(*[insert_one(i) for i in range(10)])
        
        results = await db.fetch_all("SELECT * FROM sessions")
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, db):
        """测试事务回滚"""
        async with db.acquire() as conn:
            await conn.execute(
                "INSERT INTO sessions (id, title, messages) VALUES (?, ?, ?)",
                ("tx-test", "TX Test", "[]")
            )
            # 不提交，自动回滚
        
        result = await db.fetch_one(
            "SELECT * FROM sessions WHERE id = ?",
            ("tx-test",)
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_result(self, db):
        result = await db.fetch_one(
            "SELECT * FROM sessions WHERE id = ?",
            ("nonexistent",)
        )
        assert result is None
        
        results = await db.fetch_all(
            "SELECT * FROM sessions WHERE id = ?",
            ("nonexistent",)
        )
        assert len(results) == 0
