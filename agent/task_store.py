"""任务追踪存储（TaskStore）。

移植自 hermes-agent 的 task_store.py 模式，适配 llama-cpp_vlm_web。

存储所有子代理任务的创建、执行状态和结果。
纯内存存储（生产环境可替换为 SQLite）。

功能:
  - create_task(): 创建新任务记录
  - update_status(): 更新任务状态
  - get_task(): 获取任务详情
  - list_tasks(): 列出所有/按状态过滤的任务
  - cleanup(): 清理已完成/过期的任务
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from utils import get_logger

logger = get_logger("agent.task_store")


class TaskStatus(str, Enum):
    """任务状态枚举。"""
    PENDING = "pending"       # 等待执行
    RUNNING = "running"       # 执行中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 执行失败
    CANCELLED = "cancelled"   # 已取消


@dataclass
class TaskRecord:
    """任务记录。

    属性:
        task_id: 唯一任务 ID
        goal: 任务目标
        status: 当前状态
        toolsets: 使用的工具集
        result: 执行结果（JSON 字符串）
        created_at: 创建时间（Unix timestamp）
        updated_at: 最后更新时间
        duration_ms: 执行耗时
        parent_task_id: 父任务 ID（可选，用于任务树）
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    goal: str = ""
    status: TaskStatus = TaskStatus.PENDING
    toolsets: List[str] = field(default_factory=list)
    result: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    parent_task_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转为字典（用于 JSON 序列化）。"""
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "status": self.status.value,
            "toolsets": self.toolsets,
            "result": self.result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "duration_ms": self.duration_ms,
            "parent_task_id": self.parent_task_id,
        }


class TaskStore:
    """任务追踪存储（线程安全）。"""

    def __init__(self, max_age_seconds: float = 3600.0):
        """
        参数:
            max_age_seconds: 已完成任务的保留时间（默认 1 小时）
        """
        self._tasks: Dict[str, TaskRecord] = {}
        self._lock = threading.RLock()
        self._max_age = max_age_seconds

    # ── CRUD ──────────────────────────────────────────────────────

    def create_task(
        self,
        goal: str,
        toolsets: List[str] = None,
        parent_task_id: str = None,
    ) -> TaskRecord:
        """创建新任务记录。

        参数:
            goal: 任务目标
            toolsets: 工具集列表
            parent_task_id: 父任务 ID

        返回:
            新创建的 TaskRecord
        """
        task = TaskRecord(
            goal=goal,
            toolsets=toolsets or [],
            parent_task_id=parent_task_id,
        )
        with self._lock:
            self._tasks[task.task_id] = task
        logger.debug("Task created: %s - %s", task.task_id, goal[:60])
        return task

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: str = "",
        duration_ms: float = 0.0,
    ) -> Optional[TaskRecord]:
        """更新任务状态。

        参数:
            task_id: 任务 ID
            status: 新状态
            result: 结果字符串
            duration_ms: 执行耗时

        返回:
            更新后的 TaskRecord，或 None
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = status
            task.updated_at = time.time()
            if result:
                task.result = result
            if duration_ms:
                task.duration_ms = duration_ms
            return task

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """获取任务记录。"""
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(
        self,
        status: TaskStatus = None,
        limit: int = 50,
    ) -> List[TaskRecord]:
        """列出任务。

        参数:
            status: 按状态过滤（None = 全部）
            limit: 最大返回数

        返回:
            TaskRecord 列表（按创建时间倒序）
        """
        with self._lock:
            tasks = list(self._tasks.values())
            if status:
                tasks = [t for t in tasks if t.status == status]
            # 按创建时间倒序
            tasks.sort(key=lambda t: t.created_at, reverse=True)
            return tasks[:limit]

    def get_child_tasks(self, parent_task_id: str) -> List[TaskRecord]:
        """获取指定父任务的所有子任务。"""
        with self._lock:
            return [
                t for t in self._tasks.values()
                if t.parent_task_id == parent_task_id
            ]

    def cleanup(self) -> int:
        """清理过期的已完成任务。

        返回:
            清理的任务数
        """
        now = time.time()
        removed = 0
        with self._lock:
            expired = [
                tid for tid, t in self._tasks.items()
                if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
                and (now - t.updated_at) > self._max_age
            ]
            for tid in expired:
                del self._tasks[tid]
                removed += 1
        if removed:
            logger.info("TaskStore cleanup: removed %d expired tasks", removed)
        return removed

    def cancel_all_running(self) -> int:
        """取消所有运行中的任务。

        返回:
            取消的任务数
        """
        count = 0
        with self._lock:
            for task in self._tasks.values():
                if task.status == TaskStatus.RUNNING:
                    task.status = TaskStatus.CANCELLED
                    task.updated_at = time.time()
                    count += 1
        return count

    # ── 统计 ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        """获取任务统计。"""
        with self._lock:
            counts = {}
            for task in self._tasks.values():
                key = task.status.value
                counts[key] = counts.get(key, 0) + 1
            counts["total"] = len(self._tasks)
            return counts

    def __len__(self) -> int:
        with self._lock:
            return len(self._tasks)


# ─── 全局单例 ──────────────────────────────────────────────────

_task_store: Optional[TaskStore] = None
_store_lock = threading.Lock()


def get_task_store(max_age_seconds: float = 3600.0) -> TaskStore:
    """获取全局 TaskStore 单例。

    参数:
        max_age_seconds: 已完成任务的保留时间
    """
    global _task_store
    if _task_store is None:
        with _store_lock:
            if _task_store is None:
                _task_store = TaskStore(max_age_seconds=max_age_seconds)
    return _task_store
