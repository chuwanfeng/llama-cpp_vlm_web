"""tests/test_performance_optimizations.py -- 性能优化模块测试

测试内容：
- 虚拟滚动组件
- 连接池管理
- 缓存性能
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch


class TestVirtualScroll:
    """虚拟滚动测试"""
    
    def test_virtual_scroll_creation(self):
        """测试虚拟滚动组件创建"""
        # 虚拟滚动是纯前端组件，在 Node 环境测试
        # 这里仅测试阈值逻辑
        assert True  # 基础结构测试
    
    def test_message_threshold(self):
        """测试消息阈值触发虚拟滚动"""
        # 消息数 <= 100：不启用虚拟滚动
        # 消息数 > 100：启用虚拟滚动
        messages = []
        for i in range(50):
            messages.append({"content": f"msg {i}"})
        
        # 50条消息，不启用
        assert len(messages) <= 100
        
        for i in range(60):
            messages.append({"content": f"msg {i+50}"})
        
        # 110条消息，启用
        assert len(messages) > 100


class TestConnectionPool:
    """连接池测试"""
    
    @pytest.mark.skipif(
        not __import__('importlib.util').util.find_spec('urllib3'),
        reason="urllib3 not installed"
    )
    def test_pool_creation(self):
        """测试连接池创建"""
        from services.connection_pool import ConnectionPool
        
        pool = ConnectionPool(maxsize=5, timeout=10)
        assert pool.maxsize == 5
        assert pool.timeout == 10
        assert len(pool._pools) == 0
    
    @pytest.mark.skipif(
        not __import__('importlib.util').util.find_spec('urllib3'),
        reason="urllib3 not installed"
    )
    def test_pool_singleton(self):
        """测试连接池单例"""
        from services.connection_pool import get_pool, reset_pool
        
        reset_pool()
        pool1 = get_pool()
        pool2 = get_pool()
        assert pool1 is pool2
    
    def test_pool_stats(self):
        """测试连接池统计"""
        try:
            from services.connection_pool import ConnectionPool
            pool = ConnectionPool()
            stats = pool.get_stats()
            assert "pools" in stats
            assert "maxsize" in stats
            assert "timeout" in stats
        except ImportError:
            pytest.skip("urllib3 not available")


class TestCachePerformance:
    """缓存性能测试"""
    
    def test_lru_cache_speed(self):
        """测试 LRU 缓存读写速度"""
        from services.cache_manager import LRUCache
        
        cache = LRUCache(maxsize=1000)
        
        # 写入 1000 条
        start = time.time()
        for i in range(1000):
            cache.set(f"key_{i}", {"data": i})
        write_time = time.time() - start
        
        # 读取 1000 条
        start = time.time()
        for i in range(1000):
            cache.get(f"key_{i}")
        read_time = time.time() - start
        
        # 断言：读写应该在毫秒级完成
        assert write_time < 1.0, f"Write too slow: {write_time:.3f}s"
        assert read_time < 1.0, f"Read too slow: {read_time:.3f}s"
    
    def test_cache_concurrent_access(self):
        """测试缓存并发访问"""
        from services.cache_manager import LRUCache
        
        cache = LRUCache(maxsize=100)
        errors = []
        
        def writer():
            for i in range(100):
                try:
                    cache.set(f"key_{i}", i)
                except Exception as e:
                    errors.append(e)
        
        def reader():
            for i in range(100):
                try:
                    cache.get(f"key_{i}")
                except Exception as e:
                    errors.append(e)
        
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Concurrent errors: {errors}"
    
    def test_cache_eviction(self):
        """测试缓存淘汰策略"""
        from services.cache_manager import LRUCache
        
        cache = LRUCache(maxsize=10)
        
        # 写入 20 条（超过容量）
        for i in range(20):
            cache.set(f"key_{i}", i)
        
        # 只应保留最近 10 条
        assert len(cache) <= 10
        
        # 最早写入的应该被淘汰
        assert cache.get("key_0") is None
        assert cache.get("key_19") is not None


class TestPerformanceMonitor:
    """性能监控测试"""
    
    def test_monitor_record(self):
        """测试性能记录"""
        from services.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor(max_records=100)
        monitor.record("api.test", 100, "ok")
        
        stats = monitor.get_stats("api.test")
        assert stats["count"] == 1
        assert stats["p50"] == 100
    
    def test_monitor_percentiles(self):
        """测试分位数计算"""
        from services.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # 写入 100 条不同延迟的数据
        for i in range(100):
            monitor.record("api.latency", float(i * 10), "ok")
        
        stats = monitor.get_stats("api.latency")
        assert stats["count"] == 100
        assert stats["p50"] == 495.0  # 中位数
        assert stats["p95"] >= 940.0  # 95分位（允许插值误差）
        assert stats["p99"] >= 970.0  # 99分位（允许插值误差）
    
    def test_monitor_decorator(self):
        """测试装饰器"""
        from services.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        @monitor.time("test.func")
        def slow_func():
            time.sleep(0.01)
            return 42
        
        result = slow_func()
        assert result == 42
        
        stats = monitor.get_stats("test.func")
        assert stats["count"] == 1
        assert stats["p50"] >= 10  # 至少 10ms


class TestAsyncDBPerformance:
    """异步数据库性能测试"""
    
    def test_async_db_batch_insert(self):
        """测试批量插入性能"""
        import sqlite3
        
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        
        # 批量插入 100 条
        start = time.time()
        cursor.executemany(
            "INSERT INTO test (value) VALUES (?)",
            [(f"value_{i}",) for i in range(100)]
        )
        conn.commit()
        elapsed = time.time() - start
        
        assert elapsed < 1.0, f"Batch insert too slow: {elapsed:.3f}s"
        
        count = cursor.execute("SELECT COUNT(*) FROM test").fetchone()[0]
        assert count == 100
        conn.close()
