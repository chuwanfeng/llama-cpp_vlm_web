"""tests/test_cache.py -- 缓存管理器测试"""

import time
import pytest
from services.cache_manager import LRUCache, DiskCache, CacheManager, get_cache


class TestLRUCache:
    def test_basic_get_set(self):
        cache = LRUCache(maxsize=3)
        cache.set("a", 1)
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_lru_eviction(self):
        cache = LRUCache(maxsize=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # 应淘汰 a
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_ttl_expiration(self):
        cache = LRUCache(maxsize=10, ttl=0.1)
        cache.set("a", 1)
        assert cache.get("a") == 1
        time.sleep(0.15)
        assert cache.get("a") is None

    def test_update_moves_to_end(self):
        cache = LRUCache(maxsize=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # 访问 a，移动到末尾
        cache.set("c", 3)  # 应淘汰 b
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_delete(self):
        cache = LRUCache(maxsize=3)
        cache.set("a", 1)
        assert cache.delete("a") is True
        assert cache.get("a") is None
        assert cache.delete("a") is False

    def test_clear(self):
        cache = LRUCache(maxsize=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert len(cache) == 0


class TestDiskCache:
    def test_basic_get_set(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path), ttl=60)
        cache.set("test", {"data": [1, 2, 3]})
        result = cache.get("test")
        assert result == {"data": [1, 2, 3]}

    def test_ttl_expiration(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path), ttl=0.1)
        cache.set("test", "value")
        assert cache.get("test") == "value"
        time.sleep(0.15)
        assert cache.get("test") is None

    def test_delete(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path))
        cache.set("test", "value")
        assert cache.delete("test") is True
        assert cache.get("test") is None
        assert cache.delete("test") is False

    def test_clear(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path))
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None


class TestCacheManager:
    def test_memory_first(self):
        cache = CacheManager()
        cache.set("key", "memory_value")
        assert cache.get("key") == "memory_value"

    def test_disk_fallback(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path))
        # 直接写入磁盘
        cache.disk.set("disk_key", "disk_value")
        # 读取时应从磁盘回填到内存
        assert cache.get("disk_key") == "disk_value"
        # 再次读取应从内存命中
        assert cache.get("disk_key") == "disk_value"

    def test_delete_both(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path))
        cache.set("key", "value", persist=True)
        assert cache.get("key") == "value"
        cache.delete("key")
        assert cache.get("key") is None

    def test_cached_decorator(self):
        cache = CacheManager()
        call_count = 0

        @cache.cached(key="test_func")
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive_func(5) == 10
        assert call_count == 1
        assert expensive_func(5) == 10  # 应从缓存读取
        assert call_count == 1  # 不应再调用

    def test_clear_all(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path))
        cache.set("a", 1, persist=True)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None


class TestGlobalCache:
    def test_singleton(self):
        c1 = get_cache()
        c2 = get_cache()
        assert c1 is c2
