"""services/cache_manager.py -- 高性能缓存管理器

为 llama-cpp_vlm_web 提供多层缓存支持：
- 内存缓存（LRU）— 热数据，O(1) 访问
- 磁盘缓存（JSON）— 持久化，跨进程共享
- TTL 过期 — 自动清理过期数据

使用场景：
- 工具 schema 缓存（避免重复 AST 解析）
- 设置缓存（避免重复文件 I/O）
- 模型列表缓存（避免重复扫描目录）
- 厂商凭据缓存（减少 settings.json 读取）

设计原则：
- 线程安全（threading.Lock）
- 内存上限保护（maxsize）
- 透明降级（缓存 miss 自动回源）
"""

import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional


class LRUCache:
    """线程安全的 LRU 内存缓存"""

    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None):
        """
        Args:
            maxsize: 最大条目数，超过时淘汰最久未使用
            ttl: 过期时间（秒），None 表示不过期
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值，过期自动清理"""
        with self._lock:
            if key not in self._cache:
                return default
            value, timestamp = self._cache[key]
            if self.ttl is not None and time.time() - timestamp > self.ttl:
                del self._cache[key]
                return default
            # 移动到末尾（最近使用）
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.time())
            # 淘汰最久未使用的
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def delete(self, key: str) -> bool:
        """删除缓存项"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()

    def keys(self) -> list:
        """返回所有有效 key"""
        with self._lock:
            now = time.time()
            valid_keys = []
            for key, (value, timestamp) in list(self._cache.items()):
                if self.ttl is not None and now - timestamp > self.ttl:
                    del self._cache[key]
                else:
                    valid_keys.append(key)
            return valid_keys

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


class DiskCache:
    """磁盘缓存（JSON 持久化）"""

    def __init__(self, cache_dir: str = ".cache", ttl: Optional[float] = None):
        self.cache_dir = cache_dir
        self.ttl = ttl
        os.makedirs(cache_dir, exist_ok=True)
        self._lock = threading.Lock()

    def _get_path(self, key: str) -> str:
        """安全的文件名转换"""
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        return os.path.join(self.cache_dir, f"{safe_key}.json")

    def get(self, key: str, default: Any = None) -> Any:
        """读取磁盘缓存"""
        path = self._get_path(key)
        with self._lock:
            if not os.path.exists(path):
                return default
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if self.ttl is not None:
                    if time.time() - data.get("_timestamp", 0) > self.ttl:
                        os.remove(path)
                        return default
                return data.get("value", default)
            except (json.JSONDecodeError, OSError):
                return default

    def set(self, key: str, value: Any) -> None:
        """写入磁盘缓存"""
        path = self._get_path(key)
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"value": value, "_timestamp": time.time()}, f)

    def delete(self, key: str) -> bool:
        """删除缓存文件"""
        path = self._get_path(key)
        with self._lock:
            if os.path.exists(path):
                os.remove(path)
                return True
            return False

    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            for fname in os.listdir(self.cache_dir):
                if fname.endswith(".json"):
                    os.remove(os.path.join(self.cache_dir, fname))


class CacheManager:
    """多级缓存管理器（内存 + 磁盘）"""

    def __init__(self, memory_maxsize: int = 256, cache_dir: str = ".cache"):
        self.memory = LRUCache(maxsize=memory_maxsize, ttl=300)  # 5分钟 TTL
        self.disk = DiskCache(cache_dir=cache_dir, ttl=3600)  # 1小时 TTL

    def get(self, key: str, default: Any = None) -> Any:
        """先读内存，miss 再读磁盘"""
        value = self.memory.get(key)
        if value is not None:
            return value
        value = self.disk.get(key)
        if value is not None:
            # 回填内存
            self.memory.set(key, value)
            return value
        return default

    def set(self, key: str, value: Any, persist: bool = False) -> None:
        """写入缓存

        Args:
            persist: 是否同时写入磁盘缓存
        """
        self.memory.set(key, value)
        if persist:
            self.disk.set(key, value)

    def delete(self, key: str) -> None:
        """同时删除内存和磁盘缓存"""
        self.memory.delete(key)
        self.disk.delete(key)

    def clear(self) -> None:
        """清空所有缓存"""
        self.memory.clear()
        self.disk.clear()

    def cached(self, key: Optional[str] = None, ttl: Optional[float] = None):
        """装饰器：缓存函数结果

        用法：
            @cache.cached(key="tool_schemas")
            def get_tool_schemas():
                return [...]
        """
        def decorator(func: Callable) -> Callable:
            cache_key = key or func.__name__

            def wrapper(*args, **kwargs):
                # 构建带参数的 key
                full_key = f"{cache_key}:{hash(str(args))}:{hash(str(kwargs))}"
                value = self.get(full_key)
                if value is not None:
                    return value
                value = func(*args, **kwargs)
                self.set(full_key, value, persist=False)
                return value

            # 附加清理方法
            wrapper.cache_clear = lambda: self.delete(cache_key)
            return wrapper
        return decorator

    # ── 工具结果缓存 (lookaside cache) ──

    # 只对幂等操作启用缓存
    _CACHEABLE_TOOLS = frozenset({
        "read_file", "list_directory", "glob", "grep",
        "web_search", "web_fetch",
    })

    @staticmethod
    def _tool_cache_key(tool_name: str, args: Dict[str, Any]) -> str:
        """基于工具名+参数哈希生成缓存 key"""
        import hashlib
        payload = json.dumps({"n": tool_name, "a": args}, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"tc:{tool_name}:{digest}"

    def get_tool_result(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """获取缓存的工具执行结果"""
        if tool_name not in self._CACHEABLE_TOOLS:
            return None
        key = self._tool_cache_key(tool_name, args)
        return self.memory.get(key)  # 内存 LRU 缓存, 5分钟 TTL

    def set_tool_result(self, tool_name: str, args: Dict[str, Any], result: str) -> None:
        """缓存工具执行结果"""
        if tool_name not in self._CACHEABLE_TOOLS:
            return
        key = self._tool_cache_key(tool_name, args)
        self.memory.set(key, result)


# 全局缓存实例（单例）
_cache_instance: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """获取全局缓存管理器"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
    return _cache_instance


def clear_all_cache() -> None:
    """清空所有缓存"""
    cache = get_cache()
    cache.clear()
