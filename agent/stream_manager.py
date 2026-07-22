"""
流式输出管理器 — think 标签处理 + 自适应分块
参照 hermes-agent think_scrubber.py + stream_consumer.py
"""
from __future__ import annotations

import re
import time
from typing import Optional

# Think 标签模式 — 支持主流本地模型的 thinking 输出格式
_THINK_PATTERNS = [
    # DeepSeek / Qwen:  <｜end▁of▁thinking｜>
    (re.compile(r'<\s*/\s*think\s*>', re.IGNORECASE), False, True),
    (re.compile(r'<\s*think\s*>', re.IGNORECASE), True, False),
    # 特殊结束标记
    (re.compile(r'<\s*/\s*thinking\s*>', re.IGNORECASE), False, True),
    (re.compile(r'<\s*thinking\s*>', re.IGNORECASE), True, False),
]

class StreamManager:
    """
    管理流式输出的 think 标签过滤和自适应分块

    使用方式:
        mgr = StreamManager()
        for chunk in stream:
            visible = mgr.feed(chunk)
            if visible:
                send_to_client(visible)
        final = mgr.flush()
        if final:
            send_to_client(final)
    """

    def __init__(self, chunk_threshold: int = 80, flush_interval_ms: int = 200):
        self._buffer: str = ""
        self._think_buffer: str = ""
        self._in_think: bool = False
        self._think_depth: int = 0
        self._last_flush: float = 0.0
        self._chunk_threshold: int = chunk_threshold
        self._flush_interval: float = flush_interval_ms / 1000.0
        self._total_think_tokens: int = 0
        self._total_visible_tokens: int = 0

    def reset(self) -> None:
        """重置状态 — 每次对话轮次开始时调用"""
        self._buffer = ""
        self._think_buffer = ""
        self._in_think = False
        self._think_depth = 0
        self._last_flush = 0.0
        self._total_think_tokens = 0
        self._total_visible_tokens = 0

    def feed(self, text: str) -> Optional[str]:
        """
        喂入流式文本块,返回应发送给客户端的可见文本
        Returns None 表示不需要发送
        """
        if not text:
            return None

        result_parts = []
        i = 0
        while i < len(text):
            if self._in_think:
                # 在 think 块内, 搜索  结束标签
                end_match = None
                for pattern, is_open, is_close in _THINK_PATTERNS:
                    if is_close:
                        m = pattern.search(text, i)
                        if m:
                            end_match = m
                            break

                if end_match:
                    # 记录 think 内容 (不发送)
                    self._think_buffer += text[i:end_match.start()]
                    self._total_think_tokens += len(text[i:end_match.start()].split())
                    i = end_match.end()
                    self._in_think = False
                    self._think_depth -= 1
                else:
                    # 仍在 think 块内
                    self._think_buffer += text[i:]
                    self._total_think_tokens += len(text[i:].split())
                    break
            else:
                # 不在 think 块内, 搜索 <think> 开始标签
                open_match = None
                for pattern, is_open, is_close in _THINK_PATTERNS:
                    if is_open:
                        m = pattern.search(text, i)
                        if m:
                            open_match = m
                            break

                if open_match:
                    # 开始标签前的可见文本
                    result_parts.append(text[i:open_match.start()])
                    i = open_match.end()
                    self._in_think = True
                    self._think_depth += 1
                else:
                    # 全部是可见文本
                    result_parts.append(text[i:])
                    break

        visible = "".join(result_parts)
        if not visible:
            return None

        self._buffer += visible
        self._total_visible_tokens += len(visible.split())

        # 自适应分块 — 满足阈值条件时 flush
        now = time.monotonic()
        should_flush = (
            len(self._buffer) >= self._chunk_threshold
            or (self._buffer and now - self._last_flush >= self._flush_interval)
            or "\n" in self._buffer[-20:]  # 自然段落边界
        )

        if should_flush:
            flushed = self._buffer
            self._buffer = ""
            self._last_flush = now
            return flushed

        return None

    def flush(self) -> Optional[str]:
        """清空残留缓冲区"""
        if self._buffer:
            flushed = self._buffer
            self._buffer = ""
            self._last_flush = time.monotonic()
            return flushed
        return None

    @property
    def stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_think_tokens": self._total_think_tokens,
            "total_visible_tokens": self._total_visible_tokens,
            "buffer_pending": len(self._buffer),
            "in_think": self._in_think,
        }
