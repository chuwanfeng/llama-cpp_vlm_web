# -*- coding: utf-8 -*-
"""
审批流测试

测试危险命令检测和审批逻辑
"""

import pytest
from tools.approval import (
    check_dangerous_command,
    is_approved,
    enable_yolo_for_session,
    disable_yolo_for_session,
)


class TestDangerousCommandDetection:
    """危险命令检测测试"""

    def test_hardline_rm_rf_root(self):
        """测试 rm -rf / 被无条件阻止"""
        result = check_dangerous_command("rm -rf /", "test-session")
        assert result is not None
        # 实际返回的是 approved=False, hardline=True
        assert result.get('approved') is False
        assert result.get('hardline') is True

    def test_hardline_mkfs(self):
        """测试 mkfs 被无条件阻止"""
        result = check_dangerous_command("mkfs.ext4 /dev/sda1", "test-session")
        assert result is not None
        assert result.get('approved') is False
        assert result.get('hardline') is True

    def test_safe_echo(self):
        """测试 echo 命令被允许"""
        result = check_dangerous_command("echo hello", "test-session")
        # 安全命令返回 approved=True
        assert result is not None
        assert result.get('approved') is True

    def test_safe_ls(self):
        """测试 ls 命令被允许"""
        result = check_dangerous_command("ls -la", "test-session")
        assert result is not None
        assert result.get('approved') is True


class TestYOLOMode:
    """YOLO 模式测试 - 简化版，只测试 API 不崩溃"""

    def test_enable_disable_yolo_api(self):
        """测试 YOLO API 不崩溃"""
        session = "test-yolo-session"

        # 测试启用 YOLO 不崩溃
        enable_yolo_for_session(session)

        # 测试禁用 YOLO 不崩溃
        disable_yolo_for_session(session)

        # 测试 is_approved 不崩溃
        result = is_approved("test-pattern", session)
        assert isinstance(result, bool)
