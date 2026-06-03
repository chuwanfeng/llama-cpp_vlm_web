"""
重试工具 — 抖动退避（jittered backoff）实现
transplanted from hermes-agent/agent/retry_utils.py

用途：替换固定指数退避为带抖动的退避策略，防止多个会话同时
      命中同一个限速 provider 时产生雷群效应（thundering-herd）。

核心：
  - jittered_backoff(attempt) 计算退避延迟，带随机抖动
  - 使用 monotonic counter + lock 保证线程安全
  - 抖动使得不同会话的重试时间分散开来

用法:
  from services.retry_utils import jittered_backoff

  for attempt in range(1, 4):
      try:
          result = api_call()
          break
      except RateLimitError as e:
          delay = jittered_backoff(attempt, base_delay=e.retry_after or 5.0)
          time.sleep(delay)
"""

import random
import threading
import time

# ── 单调计数器 ──────────────────────────────────────────────────────────────
# 线程安全的单调递增计数器，用于生成不重复的随机种子。
# 场景：多个 gateway session 同时重试时，避免 clk/(CLOCK_MONOTONIC) 粒度
# 不够导致多个线程共享同一个随机种子。

_jitter_counter = 0
_jitter_lock = threading.Lock()


def jittered_backoff(
    attempt: int,
    *,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    jitter_ratio: float = 0.5,
) -> float:
    """计算带抖动的指数退避延迟。

    参数：
        attempt: 1-based 重试次数（第 1 次重试 = attempt=1）
        base_delay: 第一次重试的基础延迟（秒），默认 5.0
        max_delay: 最大延迟上限（秒），默认 120.0
        jitter_ratio: 抖动比例，0.5 表示抖动范围是 [0, 0.5 * delay]

    返回：
        延迟秒数: min(base * 2^(attempt-1), max_delay) + random_jitter

    抖动的作用：
        让多个并发会话的重试时间分散开来，而不是同时命中同一时间点
        重试，有效防止 thundering-herd 雷群效应。

    示例：
        attempt=1: ~5s   + random(0~2.5s)  = 5~7.5s
        attempt=2: ~10s  + random(0~5s)    = 10~15s
        attempt=3: ~20s  + random(0~10s)   = 20~30s
        attempt=4: ~40s  + random(0~20s)   = 40~60s
        attempt=5: ~80s  + random(0~40s)   = 80~120s
        attempt=6+: capping at max_delay=120s
    """
    global _jitter_counter

    # 线程安全地生成唯一计数器值
    with _jitter_lock:
        _jitter_counter += 1
        tick = _jitter_counter

    # 计算基础指数退避
    exponent = max(0, attempt - 1)
    if exponent >= 63 or base_delay <= 0:
        delay = max_delay
    else:
        delay = min(base_delay * (2 ** exponent), max_delay)

    # 用 time + counter 生成不重复的随机种子
    # 0x9E3779B9 是黄金比例的 32-bit 表示，常用于哈希混淆
    seed = (time.time_ns() ^ (tick * 0x9E3779B9)) & 0xFFFFFFFF
    rng = random.Random(seed)
    jitter = rng.uniform(0, jitter_ratio * delay)

    return delay + jitter