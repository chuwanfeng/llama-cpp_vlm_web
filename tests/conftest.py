# -*- coding: utf-8 -*-
"""
Pytest 配置文件

提供测试固件（fixtures）和共享配置
"""

import pytest
import sys
import os

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture(scope="session")
def app():
    """创建 Flask 测试应用（session 级别单例）"""
    from app import app
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """创建测试客户端"""
    return app.test_client()


@pytest.fixture
def sample_skill_data():
    """示例技能数据"""
    return {
        "name": "test-skill",
        "description": "测试技能",
        "content": "# 测试技能\n\n这是一个测试技能。",
        "priority": 5,
        "tools": ["read_file", "run_terminal"]
    }
