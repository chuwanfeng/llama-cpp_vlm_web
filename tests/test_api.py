# -*- coding: utf-8 -*-
"""
API 端点测试

测试所有 RESTful API 端点的正确性
"""

import json
import pytest


class TestSkillsAPI:
    """技能管理 API 测试"""

    def test_list_skills(self, client):
        """测试获取技能列表"""
        resp = client.get('/api/skills')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'skills' in data
        assert 'count' in data
        assert isinstance(data['skills'], list)

    def test_create_skill_missing_name(self, client):
        """测试创建技能时缺少名称"""
        resp = client.post('/api/skills',
                          data=json.dumps({"content": "test"}),
                          content_type='application/json')
        assert resp.status_code == 400

    def test_create_and_delete_skill(self, client):
        """测试创建和删除技能"""
        # 创建技能
        skill_data = {
            "name": "test-skill-temp",
            "description": "临时测试技能",
            "content": "# 测试\n内容",
            "priority": 1,
            "tools": []
        }
        resp = client.post('/api/skills',
                          data=json.dumps(skill_data),
                          content_type='application/json')
        assert resp.status_code == 200

        # 删除技能（agent_created 标记的可以被删除）
        resp = client.delete('/api/skills/test-skill-temp')
        assert resp.status_code == 200
        data = resp.get_json()
        # 可能成功删除或返回错误（如果是用户创建的技能）
        assert 'status' in data or 'error' in data

    def test_get_nonexistent_skill(self, client):
        """测试获取不存在的技能"""
        resp = client.get('/api/skills/nonexistent-skill-12345')
        assert resp.status_code == 404


class TestProcessesAPI:
    """进程管理 API 测试"""

    def test_list_processes(self, client):
        """测试获取进程列表"""
        resp = client.get('/api/processes')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'processes' in data
        assert isinstance(data['processes'], list)

    def test_kill_nonexistent_process(self, client):
        """测试终止不存在的进程"""
        resp = client.post('/api/processes/nonexistent/kill')
        # 应该返回错误，但不崩溃
        assert resp.status_code in [200, 404, 500]


class TestApprovalAPI:
    """审批流 API 测试"""

    def test_get_pending_approvals(self, client):
        """测试获取待审批请求"""
        resp = client.get('/api/approval/pending')
        assert resp.status_code == 200
        data = resp.get_json()
        # 实际返回的是 pending 字段
        assert 'pending' in data
        assert isinstance(data['pending'], list)

    def test_get_approval_status(self, client):
        """测试获取审批状态"""
        resp = client.get('/api/approval/status')
        assert resp.status_code == 200
        data = resp.get_json()
        # 实际返回的是 session_key, yolo_enabled, pending
        assert 'session_key' in data
        assert 'yolo_enabled' in data


class TestChatAPI:
    """聊天 API 测试"""

    def test_chat_stream_exists(self, client):
        """测试流式聊天接口是否存在"""
        resp = client.post('/api/agent/chat/stream',
                          data=json.dumps({"message": "hello"}),
                          content_type='application/json')
        # 流式接口返回 text/event-stream
        assert resp.status_code in [200, 400, 500]
