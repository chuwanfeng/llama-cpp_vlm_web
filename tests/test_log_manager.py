"""tests/test_log_manager.py -- 日志管理器测试"""

import os
import json
import logging
import tempfile
import gzip
from pathlib import Path
from services.log_manager import (
    JsonFormatter,
    SizedRotatingFileHandler,
    setup_logging,
    get_logger,
)


class TestJsonFormatter:
    def test_basic_format(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_extra_fields(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.extra = {"user_id": 123, "action": "login"}
        output = formatter.format(record)
        data = json.loads(output)
        assert data["user_id"] == 123
        assert data["action"] == "login"


class TestSizedRotatingFileHandler:
    def test_rotation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            handler = SizedRotatingFileHandler(
                log_file,
                max_bytes=100,
                backup_count=2,
            )
            
            # 写入超过限制的日志
            logger = logging.getLogger("test_rotate")
            logger.handlers.clear()
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            
            for i in range(20):
                logger.info("x" * 10)
            
            handler.close()
            
            # 检查是否产生轮转文件
            assert os.path.exists(log_file)
            # 至少有一个压缩备份
            gz_files = list(Path(tmpdir).glob("*.gz"))
            assert len(gz_files) > 0

    def test_no_rotation_needed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            handler = SizedRotatingFileHandler(
                log_file,
                max_bytes=10000,
                backup_count=2,
            )
            
            logger = logging.getLogger("test_no_rotate")
            logger.handlers.clear()
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            
            logger.info("Small message")
            handler.close()
            
            # 不应有压缩文件
            gz_files = list(Path(tmpdir).glob("*.gz"))
            assert len(gz_files) == 0


class TestSetupLogging:
    def test_setup(self):
        tmpdir = tempfile.mkdtemp()
        try:
            setup_logging(
                log_dir=tmpdir,
                level="DEBUG",
                json_format=True,
            )
            
            logger = get_logger("test_setup")
            logger.info("Test message")
            
            # 关闭所有处理器，释放文件句柄
            import logging
            for handler in logging.getLogger().handlers[:]:
                handler.close()
                logging.getLogger().removeHandler(handler)
            
            # 检查日志文件
            log_file = os.path.join(tmpdir, "app.log")
            assert os.path.exists(log_file)
            
            with open(log_file, "r") as f:
                line = f.readline()
                data = json.loads(line)
                assert data["message"] == "Test message"
                assert data["level"] == "INFO"
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_error_log(self):
        tmpdir = tempfile.mkdtemp()
        try:
            setup_logging(
                log_dir=tmpdir,
                level="DEBUG",
            )
            
            logger = get_logger("test_error")
            logger.error("Error message")
            
            # 关闭所有处理器
            import logging
            for handler in logging.getLogger().handlers[:]:
                handler.close()
                logging.getLogger().removeHandler(handler)
            
            # 检查错误日志文件
            error_file = os.path.join(tmpdir, "error.log")
            assert os.path.exists(error_file)
            
            with open(error_file, "r") as f:
                content = f.read()
                assert "Error message" in content
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestGetLogger:
    def test_get_logger(self):
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
