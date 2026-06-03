"""services/log_manager.py -- 日志管理器

提供生产级日志功能：
- 日志轮转（按大小/时间）
- 多级别日志
- 结构化日志（JSON）
- 日志压缩

使用方式：
    from services.log_manager import get_logger
    
    logger = get_logger("my_module")
    logger.info("User logged in", extra={"user_id": 123})
"""

import os
import json
import logging
import gzip
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class JsonFormatter(logging.Formatter):
    """JSON 格式日志"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # 添加 extra 字段
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class SizedRotatingFileHandler(logging.FileHandler):
    """按大小轮转的日志处理器"""

    def __init__(
        self,
        filename: str,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        encoding: str = "utf-8",
    ):
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        super().__init__(filename, encoding=encoding)

    def emit(self, record: logging.LogRecord):
        """写入日志，超过大小则轮转"""
        if self.should_rollover():
            self.do_rollover()
        super().emit(record)

    def should_rollover(self) -> bool:
        """检查是否需要轮转"""
        if not os.path.exists(self.baseFilename):
            return False
        return os.path.getsize(self.baseFilename) >= self.max_bytes

    def do_rollover(self):
        """执行轮转"""
        self.close()
        
        # 移动旧日志
        for i in range(self.backup_count - 1, 0, -1):
            src = f"{self.baseFilename}.{i}.gz"
            dst = f"{self.baseFilename}.{i + 1}.gz"
            if os.path.exists(src):
                shutil.move(src, dst)
        
        # 压缩当前日志
        if os.path.exists(self.baseFilename):
            with open(self.baseFilename, "rb") as f_in:
                with gzip.open(f"{self.baseFilename}.1.gz", "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(self.baseFilename)
        
        self.stream = self._open()


class TimedRotatingFileHandler(logging.FileHandler):
    """按时间轮转的日志处理器"""

    def __init__(
        self,
        filename: str,
        when: str = "midnight",  # midnight, hourly, daily
        backup_count: int = 7,
        encoding: str = "utf-8",
    ):
        self.when = when
        self.backup_count = backup_count
        self.current_date = datetime.now().date()
        super().__init__(filename, encoding=encoding)

    def emit(self, record: logging.LogRecord):
        """写入日志，日期变化则轮转"""
        now = datetime.now().date()
        if now != self.current_date:
            self.do_rollover(now)
        super().emit(record)

    def do_rollover(self, new_date: datetime.date):
        """执行轮转"""
        self.close()
        
        # 重命名旧日志
        date_str = self.current_date.strftime("%Y%m%d")
        new_name = f"{self.baseFilename}.{date_str}.gz"
        
        if os.path.exists(self.baseFilename):
            with open(self.baseFilename, "rb") as f_in:
                with gzip.open(new_name, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(self.baseFilename)
        
        self.current_date = new_date
        self.stream = self._open()


def setup_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    json_format: bool = False,
    rotate_by_size: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
):
    """配置日志系统

    Args:
        log_dir: 日志目录
        level: 日志级别
        json_format: 是否使用 JSON 格式
        rotate_by_size: 按大小轮转（False 则按时间轮转）
        max_bytes: 单个日志文件最大大小
        backup_count: 保留日志文件数
    """
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    if json_format:
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    root_logger.addHandler(console_handler)
    
    # 文件处理器
    log_file = os.path.join(log_dir, "app.log")
    if rotate_by_size:
        file_handler = SizedRotatingFileHandler(
            log_file,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )
    else:
        file_handler = TimedRotatingFileHandler(
            log_file,
            backup_count=backup_count,
        )
    
    if json_format:
        file_handler.setFormatter(JsonFormatter())
    else:
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    root_logger.addHandler(file_handler)
    
    # 错误日志单独文件
    error_file = os.path.join(log_dir, "error.log")
    error_handler = SizedRotatingFileHandler(
        error_file,
        max_bytes=max_bytes,
        backup_count=backup_count,
    )
    error_handler.setLevel(logging.ERROR)
    if json_format:
        error_handler.setFormatter(JsonFormatter())
    else:
        error_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    root_logger.addHandler(error_handler)


def get_logger(name: str) -> logging.Logger:
    """获取日志器"""
    return logging.getLogger(name)
