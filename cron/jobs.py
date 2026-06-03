# -*- coding: utf-8 -*-
"""
Cron 任务定义和存储

移植自 hermes-agent/cron/jobs.py
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 默认存储路径
DEFAULT_STORE_PATH = Path(__file__).parent.parent / "data" / "cron_jobs.json"


@dataclass
class CronJob:
    """定时任务定义"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    # Cron 表达式 (分 时 日 月 周)
    schedule: str = "0 * * * *"  # 每小时
    # 要执行的命令或提示词
    command: str = ""
    # 是否启用
    enabled: bool = True
    # 创建时间
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # 最后执行时间
    last_run: Optional[str] = None
    # 执行次数
    run_count: int = 0
    # 是否使用 Agent 执行（True=用 AgentLoop，False=直接 shell）
    use_agent: bool = True
    # 使用的厂商和模型（use_agent=True 时）
    vendor_id: str = ""
    model: str = ""
    # 环境变量
    env: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CronJob":
        return cls(**data)


class JobStore:
    """任务持久化存储"""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or DEFAULT_STORE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, CronJob] = {}
        self._load()

    def _load(self):
        """从磁盘加载任务"""
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("jobs", []):
                job = CronJob.from_dict(item)
                self._jobs[job.id] = job
            logger.info("加载了 %d 个定时任务", len(self._jobs))
        except Exception as e:
            logger.error("加载定时任务失败: %s", e)

    def _save(self):
        """保存到磁盘"""
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(
                    {"jobs": [j.to_dict() for j in self._jobs.values()]},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            logger.error("保存定时任务失败: %s", e)

    def add(self, job: CronJob) -> CronJob:
        """添加任务"""
        self._jobs[job.id] = job
        self._save()
        return job

    def get(self, job_id: str) -> Optional[CronJob]:
        """获取任务"""
        return self._jobs.get(job_id)

    def list_all(self) -> List[CronJob]:
        """列出所有任务"""
        return list(self._jobs.values())

    def update(self, job_id: str, **kwargs) -> Optional[CronJob]:
        """更新任务"""
        job = self._jobs.get(job_id)
        if not job:
            return None
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        self._save()
        return job

    def delete(self, job_id: str) -> bool:
        """删除任务"""
        if job_id in self._jobs:
            del self._jobs[job_id]
            self._save()
            return True
        return False

    def record_run(self, job_id: str):
        """记录任务执行"""
        job = self._jobs.get(job_id)
        if job:
            job.last_run = datetime.now(timezone.utc).isoformat()
            job.run_count += 1
            self._save()
