"""
通用工具函数 — 日志、JSON、错误响应、计时等

所有模块统一从此处导入日志和通用工具，避免重复代码。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# ═══════════════════════════════════════════════════════════════════════════════
# 日志
# ═══════════════════════════════════════════════════════════════════════════════

_LOGGING_INITIALIZED = False


def setup_logging(level: str = "INFO", name: str = "llm-web") -> logging.Logger:
    """初始化根日志配置（应在 app 启动时调用一次）"""
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return logging.getLogger(name)

    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    log_fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers = [logging.StreamHandler(sys.stdout)]

    # 文件日志（供 /api/logs 读取）
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "app.log"
    fh = RotatingFileHandler(str(log_file), maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(log_fmt)
    handlers.append(fh)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
    )
    _LOGGING_INITIALIZED = True
    return logging.getLogger(name)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取日志器，默认使用项目根 logger 'llm-web'
    
    Args:
        name: 模块名，如 __name__；None 则返回根 logger
    """
    if name is None:
        return logging.getLogger("llm-web")
    # 子模块用 'llm-web.backends.gpu' 这样的层级
    if not name.startswith("llm-web"):
        name = f"llm-web.{name}" if name != "__main__" else "llm-web"
    return logging.getLogger(name)


# 预创建根 logger，方便直接 `from utils import log`
log = get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Flask 错误响应
# ═══════════════════════════════════════════════════════════════════════════════

def json_error(msg: str, code: int = 400) -> Tuple[Any, int]:
    """Flask 错误响应快捷函数
    
    用法:
        return json_error("参数缺失", 400)
    """
    from flask import jsonify
    return jsonify({"error": msg}), code


# 兼容旧名称 _err
_err = json_error


# ═══════════════════════════════════════════════════════════════════════════════
# JSON I/O
# ═══════════════════════════════════════════════════════════════════════════════

def read_json(path: Union[str, Path], default: Any = None) -> Any:
    """安全读取 JSON 文件
    
    Args:
        path: 文件路径
        default: 文件不存在或解析失败时的默认返回值
    
    Returns:
        解析后的 Python 对象，或 default
    """
    path = Path(path)
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("读取 JSON 失败 %s: %s", path, e)
        return default


def write_json(path: Union[str, Path], data: Any, indent: int = 2) -> bool:
    """安全原子写入 JSON 文件
    
    Args:
        path: 文件路径
        data: 要写入的数据
        indent: 缩进空格数
    
    Returns:
        True 成功，False 失败
    """
    import tempfile
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # 原子写入：先写临时文件，再 rename（避免 read_json 读到半截内容）
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.settings', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            os.replace(tmp, path)  # 原子替换（Windows 上原子性稍弱但远好于直接覆写）
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return True
    except (TypeError, OSError) as e:
        log.error("写入 JSON 失败 %s: %s", path, e)
        return False


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """安全 JSON 序列化，处理非序列化对象
    
    用法:
        text = safe_json_dumps({"time": datetime.now()})
    """
    def _default(o):
        if hasattr(o, "isoformat"):
            return o.isoformat()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)
    
    return json.dumps(obj, default=_default, ensure_ascii=False, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# 计时
# ═══════════════════════════════════════════════════════════════════════════════

@contextmanager
def timed(name: str = "block", logger: Optional[logging.Logger] = None):
    """计时代码块上下文管理器
    
    用法:
        with timed("API 调用"):
            response = requests.get(...)
    """
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    (logger or log).debug("%s 耗时 %.2f ms", name, elapsed)


def timed_func(logger: Optional[logging.Logger] = None):
    """函数计时装饰器
    
    用法:
        @timed_func()
        def slow_operation():
            ...
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            with timed(fn.__name__, logger):
                return fn(*args, **kwargs)
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# 其他常用
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在，不存在则创建"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def truncate(text: str, max_len: int = 200, suffix: str = "...") -> str:
    """截断文本"""
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix
