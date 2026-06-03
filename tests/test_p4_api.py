# -*- coding: utf-8 -*-
"""
P4 API 集成测试（使用 pytest fixture 避免重复导入）
"""

import json
import pytest


class TestCronAPI:
    """测试 Cron API"""

    def test_list_jobs(self, client):
        resp = client.get("/api/cron/jobs")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "jobs" in data

    def test_create_job(self, client):
        resp = client.post(
            "/api/cron/jobs",
            json={
                "name": "API测试",
                "schedule": "0 9 * * *",
                "command": "echo morning",
                "use_agent": False,
            },
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "created"

    def test_full_lifecycle(self, client):
        # 创建
        resp = client.post(
            "/api/cron/jobs",
            json={"name": "lifecycle", "command": "echo test"},
        )
        job_id = json.loads(resp.data)["job"]["id"]

        # 获取
        resp = client.get(f"/api/cron/jobs/{job_id}")
        assert resp.status_code == 200

        # 更新
        resp = client.patch(
            f"/api/cron/jobs/{job_id}",
            json={"name": "updated"},
        )
        assert resp.status_code == 200

        # 删除
        resp = client.delete(f"/api/cron/jobs/{job_id}")
        assert resp.status_code == 200

        # 确认删除
        resp = client.get(f"/api/cron/jobs/{job_id}")
        assert resp.status_code == 404


class TestPluginAPI:
    """测试插件 API"""

    def test_list_plugins(self, client):
        resp = client.get("/api/plugins")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "plugins" in data

    def test_discover_plugins(self, client):
        resp = client.get("/api/plugins/discover")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "plugins" in data


class TestMemoryProviderAPI:
    """测试记忆提供者 API"""

    def test_memory_provider_status(self, client):
        resp = client.get("/api/memory/provider")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["provider"] == "local"

    def test_memory_search(self, client):
        resp = client.post(
            "/api/memory/provider/search",
            json={"query": "test", "limit": 3},
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "results" in data


class TestMultiAgentAPI:
    """测试多 Agent API"""

    def test_missing_task(self, client):
        resp = client.post(
            "/api/agents/team",
            json={"agents": [{"role": "test"}]},
        )
        assert resp.status_code == 400

    def test_missing_agents(self, client):
        resp = client.post(
            "/api/agents/team",
            json={"task": "test task"},
        )
        assert resp.status_code == 400
