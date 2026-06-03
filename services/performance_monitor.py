"""services/performance_monitor.py -- 性能监控器

提供 API 端点性能指标收集和查询：
- 请求延迟统计（P50/P95/P99）
- 工具执行耗时
- SSE 流式吞吐量
- 缓存命中率

使用方式：
    from services.performance_monitor import monitor
    
    @monitor.time("api.chat")
    def agent_chat():
        ...

数据存储：内存环形缓冲区（默认保留最近 1000 条记录）
"""

import time
import threading
from collections import deque
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class MetricRecord:
    """单条性能记录"""
    timestamp: float
    name: str
    duration_ms: float
    status: str = "ok"  # ok / error / timeout
    extra: Dict = field(default_factory=dict)


class PerformanceMonitor:
    """性能监控器 — 线程安全"""

    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self._records: deque = deque(maxlen=max_records)
        self._counters: Dict[str, int] = {}
        self._lock = threading.Lock()

    # ── 记录 API ────────────────────────────────────────────────

    def record(self, name: str, duration_ms: float, status: str = "ok", **extra) -> None:
        """记录一条性能数据"""
        record = MetricRecord(
            timestamp=time.time(),
            name=name,
            duration_ms=duration_ms,
            status=status,
            extra=extra,
        )
        with self._lock:
            self._records.append(record)
            self._counters[name] = self._counters.get(name, 0) + 1

    def increment(self, name: str) -> None:
        """仅增加计数器"""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + 1

    # ── 查询 API ────────────────────────────────────────────────

    def get_stats(self, name: Optional[str] = None, last_n: int = 100) -> Dict:
        """获取性能统计

        Args:
            name: 指标名称过滤，None 表示全部
            last_n: 最近 N 条记录

        Returns:
            {
                "count": 总次数,
                "p50": 中位数延迟(ms),
                "p95": 95分位延迟(ms),
                "p99": 99分位延迟(ms),
                "error_rate": 错误率,
                "throughput": 每秒请求数,
            }
        """
        with self._lock:
            records = list(self._records)

        if name:
            records = [r for r in records if r.name == name]

        records = records[-last_n:]
        if not records:
            return {"count": 0, "p50": 0, "p95": 0, "p99": 0, "error_rate": 0, "throughput": 0}

        durations = sorted(r.duration_ms for r in records)
        count = len(durations)
        errors = sum(1 for r in records if r.status != "ok")

        # 计算分位数
        def percentile(data, p):
            k = (len(data) - 1) * p
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            if f == c:
                return data[f]
            return data[f] * (c - k) + data[c] * (k - f)

        # 时间范围
        time_span = records[-1].timestamp - records[0].timestamp if count > 1 else 1
        throughput = count / time_span if time_span > 0 else 0

        return {
            "count": count,
            "p50": percentile(durations, 0.5),
            "p95": percentile(durations, 0.95),
            "p99": percentile(durations, 0.99),
            "error_rate": errors / count,
            "throughput": round(throughput, 2),
        }

    def get_all_stats(self, last_n: int = 100) -> Dict[str, Dict]:
        """获取所有指标的统计"""
        with self._lock:
            names = set(r.name for r in self._records)
        return {name: self.get_stats(name, last_n) for name in names}

    def get_counters(self) -> Dict[str, int]:
        """获取计数器快照"""
        with self._lock:
            return dict(self._counters)

    # ── 装饰器 ──────────────────────────────────────────────────

    def time(self, name: str):
        """装饰器：自动记录函数执行时间

        用法：
            @monitor.time("api.chat")
            def agent_chat():
                ...
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    status = "ok"
                    return result
                except Exception:
                    status = "error"
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000
                    self.record(name, duration_ms, status)
            return wrapper
        return decorator

    # ── 上下文管理器 ─────────────────────────────────────────────

    def timer(self, name: str, **extra):
        """上下文管理器：记录代码块耗时

        用法：
            with monitor.timer("db.query"):
                result = db.execute(...)
        """
        return _TimerContext(self, name, **extra)


class _TimerContext:
    """计时上下文管理器"""

    def __init__(self, monitor: PerformanceMonitor, name: str, **extra):
        self.monitor = monitor
        self.name = name
        self.extra = extra
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start) * 1000
        status = "error" if exc_type else "ok"
        self.monitor.record(self.name, duration_ms, status, **self.extra)


# 全局监控实例
_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """获取全局性能监控器"""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


# 便捷别名
monitor = get_monitor()
