# -*- coding: utf-8 -*-
"""
工具系统测试

测试工具注册、发现和执行
"""

import pytest
from tools.registry import get_registry, ToolRegistry


class TestToolRegistry:
    """工具注册中心测试"""

    def test_registry_singleton(self):
        """测试注册表是单例"""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2

    def test_get_tool_names(self):
        """测试获取工具名称列表"""
        from tools.registry import discover_tools
        discover_tools()  # 确保工具已加载
        registry = get_registry()
        names = registry.get_tool_names()
        assert isinstance(names, list)
        assert len(names) > 0
        # 核心工具应该存在
        assert 'read_file' in names
        assert 'run_terminal' in names

    def test_get_schemas(self):
        """测试获取工具 schemas"""
        from tools.registry import discover_tools
        discover_tools()  # 确保工具已加载
        registry = get_registry()
        schemas = registry.get_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) > 0
        # 每个 schema 应该有 function 字段
        for schema in schemas:
            assert 'function' in schema
            assert 'name' in schema['function']


class TestBuiltinTools:
    """内置工具测试"""

    def test_read_file_exists(self):
        """测试 read_file 工具存在"""
        from tools.builtin_read_file import read_file
        assert callable(read_file)

    def test_run_terminal_exists(self):
        """测试 run_terminal 工具存在"""
        from tools.builtin_terminal import run_terminal
        assert callable(run_terminal)

    def test_web_search_exists(self):
        """测试 web_search 工具存在"""
        from tools.builtin_web_search import web_search
        assert callable(web_search)
