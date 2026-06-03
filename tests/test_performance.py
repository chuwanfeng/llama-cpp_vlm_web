# -*- coding: utf-8 -*-
"""
性能测试

测试关键 API 的响应时间
"""

import time
import pytest


class TestAPIPerformance:
    """API 性能测试"""

    def test_list_skills_performance(self, client):
        """测试技能列表 API 响应时间"""
        start = time.time()
        resp = client.get('/api/skills')
        elapsed = time.time() - start

        assert resp.status_code == 200
        assert elapsed < 1.0  # 应该在 1 秒内完成

    def test_list_processes_performance(self, client):
        """测试进程列表 API 响应时间"""
        start = time.time()
        resp = client.get('/api/processes')
        elapsed = time.time() - start

        assert resp.status_code == 200
        assert elapsed < 0.5  # 应该在 0.5 秒内完成

    def test_approval_status_performance(self, client):
        """测试审批状态 API 响应时间"""
        start = time.time()
        resp = client.get('/api/approval/status')
        elapsed = time.time() - start

        assert resp.status_code == 200
        assert elapsed < 0.5
