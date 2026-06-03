"""services/connection_pool.py -- HTTP 连接池管理器

为 llama-cpp_vlm_web 提供高性能 HTTP 连接复用：
- 减少 TCP 握手开销
- 支持 HTTP/1.1 Keep-Alive
- 自动连接回收和超时管理
- 线程安全

使用场景：
- 厂商 API 调用（OpenAI/DeepSeek/Anthropic 等）
- 联网搜索请求
- MCP 服务器通信
- 后端健康检查

设计原则：
- 连接复用（urllib3 PoolManager）
- 超时控制（connect/read timeout）
- 线程安全（每个线程独立连接池）
- 资源限制（最大连接数、连接过期）
"""

import threading
import time
from typing import Dict, Optional
from urllib.parse import urlparse

try:
    import urllib3
    URLLIB3_AVAILABLE = True
except ImportError:
    URLLIB3_AVAILABLE = False


class ConnectionPool:
    """HTTP 连接池管理器"""
    
    def __init__(
        self,
        maxsize: int = 10,
        timeout: float = 30.0,
        retries: int = 2,
        block: bool = False
    ):
        """
        Args:
            maxsize: 每个主机的最大连接数
            timeout: 连接超时时间（秒）
            retries: 重试次数
            block: 连接池满时是否阻塞等待
        """
        self.maxsize = maxsize
        self.timeout = timeout
        self.retries = retries
        self.block = block
        
        self._pools: Dict[str, urllib3.PoolManager] = {}
        self._lock = threading.Lock()
        self._last_used: Dict[str, float] = {}
        self._cleanup_interval = 300  # 5分钟清理一次
        self._last_cleanup = time.time()
    
    def _get_pool(self, base_url: str) -> urllib3.PoolManager:
        """获取或创建连接池"""
        # 解析主机名
        parsed = urlparse(base_url)
        host_key = f"{parsed.scheme}://{parsed.netloc}"
        
        with self._lock:
            # 定期清理过期连接池
            if time.time() - self._last_cleanup > self._cleanup_interval:
                self._cleanup_pools()
            
            if host_key not in self._pools:
                self._pools[host_key] = urllib3.PoolManager(
                    num_pools=1,
                    maxsize=self.maxsize,
                    block=self.block,
                    timeout=urllib3.Timeout(
                        connect=min(5.0, self.timeout / 3),
                        read=self.timeout
                    ),
                    retries=urllib3.Retry(
                        total=self.retries,
                        backoff_factor=0.5,
                        status_forcelist=[500, 502, 503, 504]
                    )
                )
            
            self._last_used[host_key] = time.time()
            return self._pools[host_key]
    
    def request(
        self,
        method: str,
        url: str,
        body: Optional[bytes] = None,
        headers: Optional[Dict] = None,
        **kwargs
    ):
        """发送 HTTP 请求（复用连接）"""
        if not URLLIB3_AVAILABLE:
            raise ImportError("urllib3 is required for connection pooling")
        
        pool = self._get_pool(url)
        return pool.request(method, url, body=body, headers=headers, **kwargs)
    
    def get(self, url: str, headers: Optional[Dict] = None, **kwargs):
        """GET 请求"""
        return self.request('GET', url, headers=headers, **kwargs)
    
    def post(self, url: str, body: Optional[bytes] = None, headers: Optional[Dict] = None, **kwargs):
        """POST 请求"""
        return self.request('POST', url, body=body, headers=headers, **kwargs)
    
    def _cleanup_pools(self):
        """清理长时间未使用的连接池"""
        now = time.time()
        expired = []
        
        for host_key, last_used in self._last_used.items():
            if now - last_used > 600:  # 10分钟未使用
                expired.append(host_key)
        
        for host_key in expired:
            if host_key in self._pools:
                self._pools[host_key].clear()
                del self._pools[host_key]
            del self._last_used[host_key]
        
        self._last_cleanup = now
    
    def clear(self):
        """清空所有连接池"""
        with self._lock:
            for pool in self._pools.values():
                pool.clear()
            self._pools.clear()
            self._last_used.clear()
    
    def get_stats(self) -> Dict:
        """获取连接池统计"""
        with self._lock:
            return {
                "pools": len(self._pools),
                "hosts": list(self._pools.keys()),
                "maxsize": self.maxsize,
                "timeout": self.timeout,
            }


# 全局连接池实例（单例）
_pool_instance: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    """获取全局连接池"""
    global _pool_instance
    if _pool_instance is None:
        with _pool_lock:
            if _pool_instance is None:
                _pool_instance = ConnectionPool()
    return _pool_instance


def reset_pool():
    """重置连接池"""
    global _pool_instance
    with _pool_lock:
        if _pool_instance:
            _pool_instance.clear()
        _pool_instance = ConnectionPool()


# 兼容性封装：requests 风格 API
class PooledSession:
    """兼容 requests.Session 的接口"""
    
    def __init__(self, pool: Optional[ConnectionPool] = None):
        self.pool = pool or get_pool()
    
    def request(self, method: str, url: str, **kwargs):
        """发送请求"""
        headers = kwargs.get('headers', {})
        data = kwargs.get('data')
        json_data = kwargs.get('json')
        
        if json_data and not data:
            import json
            data = json.dumps(json_data).encode('utf-8')
            headers.setdefault('Content-Type', 'application/json')
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return self.pool.request(method, url, body=data, headers=headers)
    
    def get(self, url: str, **kwargs):
        return self.request('GET', url, **kwargs)
    
    def post(self, url: str, **kwargs):
        return self.request('POST', url, **kwargs)
    
    def put(self, url: str, **kwargs):
        return self.request('PUT', url, **kwargs)
    
    def delete(self, url: str, **kwargs):
        return self.request('DELETE', url, **kwargs)
