"""tests/test_performance_monitor.py -- 性能监控器测试"""

import time
import pytest
from services.performance_monitor import PerformanceMonitor, get_monitor


class TestPerformanceMonitor:
    def test_record_and_get_stats(self):
        mon = PerformanceMonitor()
        mon.record("test.op", 10.0)
        mon.record("test.op", 20.0)
        mon.record("test.op", 30.0)

        stats = mon.get_stats("test.op")
        assert stats["count"] == 3
        assert stats["p50"] == 20.0
        assert stats["error_rate"] == 0

    def test_error_rate(self):
        mon = PerformanceMonitor()
        mon.record("test.op", 10.0, status="ok")
        mon.record("test.op", 20.0, status="error")

        stats = mon.get_stats("test.op")
        assert stats["error_rate"] == 0.5

    def test_percentile(self):
        mon = PerformanceMonitor()
        for i in range(100):
            mon.record("test.op", float(i))

        stats = mon.get_stats("test.op")
        assert stats["p50"] == 49.5
        assert stats["p95"] == 94.05
        assert stats["p99"] == 98.01

    def test_max_records(self):
        mon = PerformanceMonitor(max_records=10)
        for i in range(20):
            mon.record("test.op", float(i))

        stats = mon.get_stats("test.op")
        assert stats["count"] == 10  # 只保留最近 10 条

    def test_decorator(self):
        mon = PerformanceMonitor()

        @mon.time("decorated.fn")
        def slow_fn():
            time.sleep(0.01)
            return 42

        result = slow_fn()
        assert result == 42

        stats = mon.get_stats("decorated.fn")
        assert stats["count"] == 1
        assert stats["p50"] >= 10  # 至少 10ms

    def test_context_manager(self):
        mon = PerformanceMonitor()

        with mon.timer("context.block"):
            time.sleep(0.01)

        stats = mon.get_stats("context.block")
        assert stats["count"] == 1
        assert stats["p50"] >= 10

    def test_get_all_stats(self):
        mon = PerformanceMonitor()
        mon.record("op1", 10.0)
        mon.record("op2", 20.0)

        all_stats = mon.get_all_stats()
        assert "op1" in all_stats
        assert "op2" in all_stats

    def test_counters(self):
        mon = PerformanceMonitor()
        mon.increment("requests")
        mon.increment("requests")
        mon.increment("errors")

        counters = mon.get_counters()
        assert counters["requests"] == 2
        assert counters["errors"] == 1

    def test_empty_stats(self):
        mon = PerformanceMonitor()
        stats = mon.get_stats("nonexistent")
        assert stats["count"] == 0


class TestGlobalMonitor:
    def test_singleton(self):
        m1 = get_monitor()
        m2 = get_monitor()
        assert m1 is m2
