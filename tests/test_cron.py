# -*- coding: utf-8 -*-
"""
Cron 定时任务系统测试
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cron.jobs import CronJob, JobStore
from cron.scheduler import CronScheduler


class TestCronJob(unittest.TestCase):
    """测试 CronJob 数据类"""

    def test_create_job(self):
        job = CronJob(
            name="测试任务",
            description="每小时执行一次",
            schedule="0 * * * *",
            command="echo hello",
        )
        self.assertEqual(job.name, "测试任务")
        self.assertEqual(job.schedule, "0 * * * *")
        self.assertTrue(job.enabled)
        self.assertTrue(job.use_agent)

    def test_to_dict(self):
        job = CronJob(name="test", command="echo hi")
        d = job.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["command"], "echo hi")
        self.assertIn("id", d)

    def test_from_dict(self):
        data = {
            "id": "abc123",
            "name": "test",
            "schedule": "*/5 * * * *",
            "command": "ls",
            "enabled": False,
        }
        job = CronJob.from_dict(data)
        self.assertEqual(job.id, "abc123")
        self.assertEqual(job.schedule, "*/5 * * * *")
        self.assertFalse(job.enabled)


class TestJobStore(unittest.TestCase):
    """测试 JobStore 持久化"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.store_path = Path(self.tmp_dir) / "test_jobs.json"
        self.store = JobStore(path=self.store_path)

    def tearDown(self):
        if self.store_path.exists():
            self.store_path.unlink()
        os.rmdir(self.tmp_dir)

    def test_add_and_get(self):
        job = CronJob(name="test", command="echo 1")
        added = self.store.add(job)
        self.assertEqual(added.name, "test")

        fetched = self.store.get(job.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "test")

    def test_list_all(self):
        self.store.add(CronJob(name="j1", command="echo 1"))
        self.store.add(CronJob(name="j2", command="echo 2"))
        jobs = self.store.list_all()
        self.assertEqual(len(jobs), 2)

    def test_update(self):
        job = self.store.add(CronJob(name="old", command="echo old"))
        updated = self.store.update(job.id, name="new", command="echo new")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.name, "new")

        fetched = self.store.get(job.id)
        self.assertEqual(fetched.name, "new")

    def test_delete(self):
        job = self.store.add(CronJob(name="del", command="echo del"))
        self.assertTrue(self.store.delete(job.id))
        self.assertIsNone(self.store.get(job.id))
        self.assertFalse(self.store.delete("nonexistent"))

    def test_persistence(self):
        job = self.store.add(CronJob(name="persist", command="echo persist"))
        # 创建新 store 实例，应该能加载之前的数据
        store2 = JobStore(path=self.store_path)
        fetched = store2.get(job.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "persist")


class TestCronScheduler(unittest.TestCase):
    """测试 CronScheduler"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.store_path = Path(self.tmp_dir) / "scheduler_jobs.json"
        self.scheduler = CronScheduler(job_store=JobStore(path=self.store_path))

    def tearDown(self):
        self.scheduler.stop()
        if self.store_path.exists():
            self.store_path.unlink()
        os.rmdir(self.tmp_dir)

    def test_add_job(self):
        job = CronJob(name="sched", command="echo test")
        added = self.scheduler.add_job(job)
        self.assertEqual(added.name, "sched")

    def test_list_jobs(self):
        self.scheduler.add_job(CronJob(name="j1", command="echo 1"))
        self.scheduler.add_job(CronJob(name="j2", command="echo 2"))
        jobs = self.scheduler.list_jobs()
        self.assertEqual(len(jobs), 2)

    def test_handler_registration(self):
        called = []

        def handler(job):
            called.append(job.name)

        self.scheduler.register_handler("default", handler)
        self.assertIn("default", self.scheduler._handlers)

    def test_start_stop(self):
        self.scheduler.start()
        self.assertTrue(self.scheduler._running)
        self.scheduler.stop()
        self.assertFalse(self.scheduler._running)


class TestCronAPI(unittest.TestCase):
    """测试 Cron API 端点（需要 Flask 应用上下文）"""

    def setUp(self):
        from app import app
        self.app = app
        self.client = app.test_client()

    def test_list_jobs(self):
        resp = self.client.get("/api/cron/jobs")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("jobs", data)

    def test_create_job(self):
        resp = self.client.post(
            "/api/cron/jobs",
            json={
                "name": "API测试",
                "schedule": "0 9 * * *",
                "command": "echo morning",
                "use_agent": False,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["status"], "created")
        self.assertEqual(data["job"]["name"], "API测试")

    def test_get_job(self):
        # 先创建
        resp = self.client.post(
            "/api/cron/jobs",
            json={"name": "get_test", "command": "echo get"},
        )
        job_id = json.loads(resp.data)["job"]["id"]

        # 再获取
        resp = self.client.get(f"/api/cron/jobs/{job_id}")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["job"]["name"], "get_test")

    def test_delete_job(self):
        # 先创建
        resp = self.client.post(
            "/api/cron/jobs",
            json={"name": "del_test", "command": "echo del"},
        )
        job_id = json.loads(resp.data)["job"]["id"]

        # 删除
        resp = self.client.delete(f"/api/cron/jobs/{job_id}")
        self.assertEqual(resp.status_code, 200)

        # 确认已删除
        resp = self.client.get(f"/api/cron/jobs/{job_id}")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
