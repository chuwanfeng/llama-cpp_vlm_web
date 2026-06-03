"""services/rate_limiter.py -- 限流器

提供基于令牌桶算法的请求限流：
- 全局限流（所有请求）
- 按 IP 限流
- 按用户限流
- 按端点限流

使用方式：
    from services.rate_limiter import rate_limit
    
    @app.route("/api/chat")
    @rate_limit(requests_per_minute=60)
    def chat():
        ...
"""

import time
import threading
from functools import wraps
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    """限流配置"""
    requests_per_minute: int = 60
    burst_size: int = 10
    key_prefix: str = "rl"


class TokenBucket:
    """令牌桶 — 线程安全"""

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: 每秒产生令牌数
            capacity: 桶容量（突发请求数）
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> bool:
        """尝试获取令牌

        Returns:
            True: 获取成功
            False: 获取失败（限流）
        """
        now = time.time()
        with self._lock:
            # 计算新产生的令牌
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """计算需要等待的时间"""
        with self._lock:
            if self.tokens >= tokens:
                return 0.0
            needed = tokens - self.tokens
            return needed / self.rate


class RateLimiter:
    """限流器管理器"""

    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._cleanup_interval = 300  # 5 分钟清理一次
        self._last_cleanup = time.time()

    def _get_bucket(self, key: str, rate: float, capacity: int) -> TokenBucket:
        """获取或创建令牌桶"""
        with self._lock:
            # 定期清理过期桶
            now = time.time()
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup()
                self._last_cleanup = now

            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(rate, capacity)
                self._buckets[key] = bucket
            return bucket

    def _cleanup(self):
        """清理长时间未使用的桶"""
        # 简化实现：保留所有桶
        # 生产环境可添加最后访问时间，清理冷桶
        pass

    def is_allowed(self, key: str, requests_per_minute: int = 60, burst_size: int = 10) -> bool:
        """检查请求是否允许

        Args:
            key: 限流键（如 IP、用户ID）
            requests_per_minute: 每分钟请求数
            burst_size: 突发请求数

        Returns:
            True: 允许
            False: 限流
        """
        rate = requests_per_minute / 60.0
        bucket = self._get_bucket(key, rate, burst_size)
        return bucket.acquire()

    def get_wait_time(self, key: str, requests_per_minute: int = 60, burst_size: int = 10) -> float:
        """获取需要等待的时间"""
        rate = requests_per_minute / 60.0
        bucket = self._get_bucket(key, rate, burst_size)
        return bucket.get_wait_time()


# 全局限流实例
_limiter: Optional[RateLimiter] = None


def get_limiter() -> RateLimiter:
    """获取全局限流器"""
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter


# 便捷函数
def is_allowed(key: str, requests_per_minute: int = 60, burst_size: int = 10) -> bool:
    return get_limiter().is_allowed(key, requests_per_minute, burst_size)


def rate_limit(requests_per_minute: int = 60, burst_size: int = 10, key_func=None):
    """限流装饰器

    用法：
        @app.route("/api/chat")
        @rate_limit(requests_per_minute=60)
        def chat():
            ...

    Args:
        requests_per_minute: 每分钟请求数
        burst_size: 突发请求数
        key_func: 自定义限流键函数，默认使用 IP
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from flask import request

            # 获取限流键
            if key_func:
                key = key_func()
            else:
                key = request.remote_addr or "unknown"

            limiter = get_limiter()
            if not limiter.is_allowed(key, requests_per_minute, burst_size):
                wait_time = limiter.get_wait_time(key, requests_per_minute, burst_size)
                from flask import jsonify
                return jsonify({
                    "error": "Rate limit exceeded",
                    "retry_after": int(wait_time) + 1,
                }), 429

            return func(*args, **kwargs)
        return wrapper
    return decorator
