"""tests/test_rate_limiter.py -- 限流器测试"""

import time
import pytest
from services.rate_limiter import TokenBucket, RateLimiter, get_limiter


class TestTokenBucket:
    def test_initial_capacity(self):
        bucket = TokenBucket(rate=1.0, capacity=10)
        assert bucket.acquire()
        assert bucket.tokens == 9

    def test_rate_limiting(self):
        bucket = TokenBucket(rate=1.0, capacity=2)
        assert bucket.acquire()
        assert bucket.acquire()
        assert not bucket.acquire()  # 桶空

    def test_token_refill(self):
        bucket = TokenBucket(rate=10.0, capacity=1)
        assert bucket.acquire()
        assert not bucket.acquire()
        time.sleep(0.15)  # 等待 1.5 个令牌产生
        assert bucket.acquire()

    def test_wait_time(self):
        bucket = TokenBucket(rate=1.0, capacity=1)
        bucket.acquire()
        wait = bucket.get_wait_time()
        assert wait > 0.9  # 接近 1 秒

    def test_burst(self):
        bucket = TokenBucket(rate=1.0, capacity=5)
        # 连续获取 5 个
        for _ in range(5):
            assert bucket.acquire()
        # 第 6 个失败
        assert not bucket.acquire()


class TestRateLimiter:
    def test_is_allowed(self):
        limiter = RateLimiter()
        key = "test_user"
        # 前 10 个允许
        for _ in range(10):
            assert limiter.is_allowed(key, requests_per_minute=60, burst_size=10)
        # 第 11 个拒绝
        assert not limiter.is_allowed(key, requests_per_minute=60, burst_size=10)

    def test_different_keys(self):
        limiter = RateLimiter()
        # 不同 key 独立计数
        assert limiter.is_allowed("user1", burst_size=1)
        assert limiter.is_allowed("user2", burst_size=1)

    def test_wait_time(self):
        limiter = RateLimiter()
        key = "test_user"
        limiter.is_allowed(key, requests_per_minute=60, burst_size=1)
        wait = limiter.get_wait_time(key, requests_per_minute=60, burst_size=1)
        assert wait > 0.9

    def test_refill_over_time(self):
        limiter = RateLimiter()
        key = "test_user"
        # 用完配额
        for _ in range(10):
            limiter.is_allowed(key, requests_per_minute=600, burst_size=10)
        assert not limiter.is_allowed(key, requests_per_minute=600, burst_size=10)
        
        # 等待补充
        time.sleep(0.15)
        assert limiter.is_allowed(key, requests_per_minute=600, burst_size=10)


class TestGlobalLimiter:
    def test_singleton(self):
        l1 = get_limiter()
        l2 = get_limiter()
        assert l1 is l2
