# -*- coding: utf-8 -*-
"""
AgentLoop 测试

测试 Agent 引擎核心功能
"""

import pytest
from agent.loop import AgentLoop, AgentResult


class TestAgentResult:
    """AgentResult 数据类测试"""

    def test_result_creation(self):
        """测试创建 AgentResult"""
        result = AgentResult(
            messages=[{"role": "user", "content": "test"}],
            finished_naturally=True
        )
        assert result.finished_naturally is True
        assert len(result.messages) == 1

    def test_result_defaults(self):
        """测试 AgentResult 默认值"""
        result = AgentResult(messages=[])
        assert result.turns_used == 0
        assert result.finished_naturally is False
        assert result.managed_state is None


class TestAgentLoop:
    """AgentLoop 测试"""

    def test_loop_creation(self):
        """测试创建 AgentLoop"""
        loop = AgentLoop(
            backend_type="openai",
            vendor_id="test",
            model="test-model",
            api_key="test-key"
        )
        assert loop.vendor_id == "test"
        assert loop.model == "test-model"
