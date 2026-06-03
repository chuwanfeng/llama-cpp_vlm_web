# -*- coding: utf-8 -*-
"""
插件系统测试
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plugins.base import Plugin, PluginManager
from plugins.memory.base import MemoryPlugin
from plugins.memory.local import LocalMemoryPlugin


class DummyPlugin(Plugin):
    """测试用插件"""

    name = "dummy"
    version = "0.1.0"
    description = "测试插件"

    def initialize(self) -> bool:
        return True

    def shutdown(self):
        pass

    def get_tools(self):
        return [{"type": "function", "function": {"name": "dummy_tool"}}]


class TestPlugin(unittest.TestCase):
    """测试插件基类"""

    def test_plugin_lifecycle(self):
        plugin = DummyPlugin()
        self.assertTrue(plugin.initialize())
        self.assertTrue(plugin.enabled)
        plugin.shutdown()

    def test_plugin_tools(self):
        plugin = DummyPlugin()
        tools = plugin.get_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["function"]["name"], "dummy_tool")


class TestPluginManager(unittest.TestCase):
    """测试插件管理器"""

    def setUp(self):
        self.manager = PluginManager()

    def test_load_plugin(self):
        plugin = self.manager.load(DummyPlugin)
        self.assertEqual(plugin.name, "dummy")
        self.assertIn("dummy", [p.name for p in self.manager.list_plugins()])

    def test_get_plugin(self):
        self.manager.load(DummyPlugin)
        fetched = self.manager.get("dummy")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.version, "0.1.0")

    def test_unload_plugin(self):
        self.manager.load(DummyPlugin)
        self.manager.unload("dummy")
        self.assertIsNone(self.manager.get("dummy"))

    def test_get_all_tools(self):
        self.manager.load(DummyPlugin)
        tools = self.manager.get_all_tools()
        self.assertEqual(len(tools), 1)

    def test_hooks(self):
        self.manager.load(DummyPlugin)

        # 测试消息钩子
        result = self.manager.run_pre_message_hooks("hello", {})
        self.assertEqual(result, "hello")  # 未修改

        # 测试响应钩子
        result = self.manager.run_post_response_hooks("world", {})
        self.assertEqual(result, "world")


class TestMemoryPlugin(unittest.TestCase):
    """测试记忆插件"""

    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp()
        self.plugin = LocalMemoryPlugin(config={"db_path": os.path.join(self.tmp_dir, "test.db")})

    def tearDown(self):
        self.plugin.shutdown()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_store_and_search(self):
        # 存储记忆
        ok = self.plugin.store("Python 是一种编程语言", {"topic": "programming"})
        self.assertTrue(ok)

        # 搜索记忆
        results = self.plugin.search("Python")
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("Python", results[0]["content"])

    def test_user_profile(self):
        profile = self.plugin.get_user_profile()
        self.assertIsNotNone(profile)
        self.assertEqual(profile["user_id"], "anonymous")


class TestPluginAPI(unittest.TestCase):
    """测试插件 API 端点"""

    def setUp(self):
        from app import app
        self.app = app
        self.client = app.test_client()

    def test_list_plugins(self):
        resp = self.client.get("/api/plugins")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("plugins", data)

    def test_discover_plugins(self):
        resp = self.client.get("/api/plugins/discover")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("plugins", data)


class TestMemoryProviderAPI(unittest.TestCase):
    """测试记忆提供者 API"""

    def setUp(self):
        from app import app
        self.app = app
        self.client = app.test_client()

    def test_memory_provider_status(self):
        resp = self.client.get("/api/memory/provider")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["provider"], "local")
        self.assertIn("profile", data)

    def test_memory_search(self):
        resp = self.client.post(
            "/api/memory/provider/search",
            json={"query": "test", "limit": 3},
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("results", data)


if __name__ == "__main__":
    unittest.main()
