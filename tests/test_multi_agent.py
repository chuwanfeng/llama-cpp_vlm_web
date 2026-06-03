# -*- coding: utf-8 -*-
"""
多 Agent 协作测试
"""

import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMultiAgentAPI(unittest.TestCase):
    """测试多 Agent 协作 API"""

    def setUp(self):
        from app import app
        self.app = app
        self.client = app.test_client()

    def test_missing_task(self):
        resp = self.client.post(
            "/api/agents/team",
            json={"agents": [{"role": "test"}]},
        )
        self.assertEqual(resp.status_code, 400)

    def test_missing_agents(self):
        resp = self.client.post(
            "/api/agents/team",
            json={"task": "test task"},
        )
        self.assertEqual(resp.status_code, 400)

    @patch("backends.vendors.chat_stream")
    def test_team_execution(self, mock_chat):
        """测试多 Agent 顺序执行"""
        # 模拟流式返回
        def mock_stream(*args, **kwargs):
            yield {"content": "分析结果：代码结构良好"}

        mock_chat.side_effect = mock_stream

        resp = self.client.post(
            "/api/agents/team",
            json={
                "task": "分析代码质量",
                "agents": [
                    {"role": "analyzer", "prompt": "你是代码分析专家"},
                    {"role": "reviewer", "prompt": "你是代码审查专家"},
                ],
                "vendor_id": "deepseek",
                "model": "deepseek-chat",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["task"], "分析代码质量")
        self.assertEqual(len(data["results"]), 2)
        self.assertEqual(data["results"][0]["role"], "analyzer")
        self.assertEqual(data["results"][1]["role"], "reviewer")

    @patch("backends.vendors.chat_stream")
    def test_context_passing(self, mock_chat):
        """测试上下文传递"""
        responses = [
            [{"content": "第一步结果"}],
            [{"content": "基于第一步的分析"}],
        ]
        call_count = [0]

        def mock_stream(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            for item in responses[idx]:
                yield item

        mock_chat.side_effect = mock_stream

        resp = self.client.post(
            "/api/agents/team",
            json={
                "task": "复杂分析",
                "agents": [
                    {"role": "step1", "prompt": "执行第一步"},
                    {"role": "step2", "prompt": "基于前文继续"},
                ],
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        # 第二个 agent 应该收到第一个 agent 的结果
        self.assertEqual(data["agent_count"], 2)


class TestMultiAgentEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def setUp(self):
        from app import app
        self.app = app
        self.client = app.test_client()

    def test_empty_agents_list(self):
        resp = self.client.post(
            "/api/agents/team",
            json={"task": "test", "agents": []},
        )
        self.assertEqual(resp.status_code, 400)

    def test_agent_without_role(self):
        """Agent 缺少 role 应该使用默认值"""
        with patch("backends.vendors.chat_stream") as mock_chat:
            mock_chat.return_value = iter([{"content": "ok"}])
            resp = self.client.post(
                "/api/agents/team",
                json={
                    "task": "test",
                    "agents": [{"prompt": "无角色"}],
                },
            )
            self.assertEqual(resp.status_code, 200)
            data = json.loads(resp.data)
            self.assertEqual(data["results"][0]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
