# -*- coding: utf-8 -*-
"""
Cron 调度器

移植自 hermes-agent/cron/scheduler.py
基于 croniter 的定时任务调度器。
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from .jobs import CronJob, JobStore

logger = logging.getLogger(__name__)

# 全局调度器实例
_scheduler_instance: Optional["CronScheduler"] = None


def get_scheduler() -> "CronScheduler":
    """获取全局调度器实例"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = CronScheduler()
    return _scheduler_instance


class CronScheduler:
    """Cron 任务调度器

    在后台线程中运行，检查并执行到期的定时任务。
    """

    def __init__(self, job_store: Optional[JobStore] = None):
        self.store = job_store or JobStore()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._handlers: Dict[str, Callable] = {}
        self._check_interval = 30  # 每 30 秒检查一次

    def register_handler(self, name: str, handler: Callable):
        """注册任务执行处理器

        Args:
            name: 处理器名称
            handler: 回调函数，接收 CronJob 参数
        """
        self._handlers[name] = handler
        logger.info("注册任务处理器: %s", name)

    def start(self):
        """启动调度器"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="cron-scheduler")
        self._thread.start()
        logger.info("Cron 调度器已启动")

    def stop(self):
        """停止调度器"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Cron 调度器已停止")

    def _loop(self):
        """主循环"""
        while self._running:
            try:
                self._check_jobs()
            except Exception as e:
                logger.error("检查任务时出错: %s", e)
            time.sleep(self._check_interval)

    def _check_jobs(self):
        """检查并执行到期任务"""
        now = datetime.now(timezone.utc)
        for job in self.store.list_all():
            if not job.enabled:
                continue
            try:
                # 使用 croniter 检查是否到期
                from croniter import croniter
                itr = croniter(job.schedule, now)
                next_run = itr.get_next(datetime)
                # 如果下次执行时间在当前时间之前（或非常接近），则执行
                if (now - next_run).total_seconds() >= -60:  # 60 秒容差
                    self._execute_job(job)
            except Exception as e:
                logger.error("检查任务 %s 时出错: %s", job.id, e)

    def _execute_job(self, job: CronJob):
        """执行单个任务"""
        logger.info("执行任务: %s (%s)", job.name, job.id)
        self.store.record_run(job.id)

        # 调用注册的处理器
        handler = self._handlers.get("default")
        if handler:
            try:
                handler(job)
            except Exception as e:
                logger.error("任务 %s 执行失败: %s", job.id, e)
        else:
            logger.warning("没有注册默认任务处理器，任务 %s 未执行", job.id)

    def add_job(self, job: CronJob) -> CronJob:
        """添加任务"""
        return self.store.add(job)

    def get_job(self, job_id: str) -> Optional[CronJob]:
        """获取任务"""
        return self.store.get(job_id)

    def list_jobs(self) -> List[CronJob]:
        """列出所有任务"""
        return self.store.list_all()

    def delete_job(self, job_id: str) -> bool:
        """删除任务"""
        return self.store.delete(job_id)

    def update_job(self, job_id: str, **kwargs) -> Optional[CronJob]:
        """更新任务"""
        return self.store.update(job_id, **kwargs)
