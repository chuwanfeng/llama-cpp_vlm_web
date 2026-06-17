"""
后台进程注册表 —— 管理通过 terminal(background=true) 启动的后台进程。

功能：
  - 输出缓冲（滚动 200KB 窗口）
  - 状态轮询和日志检索
  - 阻塞等待（支持中断）
  - 进程终止
  - 会话级跟踪

移植自 hermes-agent/tools/process_registry.py，适配 llama-cpp_vlm_web 架构。
"""

import json
import logging
import os
import platform
import shlex
import signal
import subprocess
import threading
import time
import uuid

_IS_WINDOWS = platform.system() == "Windows"

logger = logging.getLogger(__name__)

# 限制
MAX_OUTPUT_CHARS = 200_000      # 200KB 滚动输出缓冲
FINISHED_TTL_SECONDS = 1800     # 已完成进程保留 30 分钟
MAX_PROCESSES = 64              # 最大并发跟踪进程数（LRU 修剪）


def format_uptime_short(seconds: int) -> str:
    """格式化运行时间为简短可读字符串。"""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    mins, secs = divmod(s, 60)
    if mins < 60:
        return f"{mins}m {secs}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins}m"


class ProcessSession:
    """带输出缓冲的跟踪后台进程。"""

    def __init__(
        self,
        id: str,
        command: str,
        task_id: str = "",
        session_key: str = "",
        pid: int = None,
        process: subprocess.Popen = None,
        cwd: str = None,
        started_at: float = 0.0,
    ):
        self.id = id                           # 唯一会话 ID ("proc_xxxxxxxxxxxx")
        self.command = command                 # 原始命令字符串
        self.task_id = task_id                 # 任务/沙箱隔离键
        self.session_key = session_key         # 网关会话键（用于重置保护）
        self.pid = pid                         # OS 进程 ID
        self.process = process                 # Popen 句柄（仅本地）
        self.cwd = cwd                         # 工作目录
        self.started_at = started_at or time.time()
        self.exited = False                    # 进程是否已结束
        self.exit_code = None                  # 退出码（运行中为 None）
        self.output_buffer = ""                # 滚动输出（最后 MAX_OUTPUT_CHARS）
        self.max_output_chars = MAX_OUTPUT_CHARS
        self.detached = False                  # 如果从崩溃恢复则为 True（无管道）
        self._lock = threading.Lock()
        self._reader_thread = None


class ProcessRegistry:
    """
    运行中和已完成的后台进程的内存注册表。

    线程安全。访问来源：
      - 执行器线程（terminal_tool、process 工具处理程序）
      - 网关 asyncio 循环（会话重置检查）
      - 清理线程（沙箱回收协调）
    """

    _SHELL_NOISE_SUBSTRINGS = (
        "bash: cannot set terminal process group",
        "bash: no job control in this shell",
        "no job control in this shell",
        "cannot set terminal process group",
        "tcsetattr: Inappropriate ioctl for device",
    )

    def __init__(self):
        self._running: dict[str, ProcessSession] = {}
        self._finished: dict[str, ProcessSession] = {}
        self._lock = threading.Lock()

        # 完成通知队列 —— 所有后台进程事件的统一队列
        import queue as _queue_mod
        self.completion_queue: _queue_mod.Queue = _queue_mod.Queue()

        # 跟踪已通过 wait/poll/log 消耗的会话完成状态
        self._completion_consumed: set = set()

    @staticmethod
    def _clean_shell_noise(text: str) -> str:
        """从输出开头去除 shell 启动警告。"""
        lines = text.split("\n")
        while lines and any(noise in lines[0] for noise in ProcessRegistry._SHELL_NOISE_SUBSTRINGS):
            lines.pop(0)
        return "\n".join(lines)

    def _resolve_safe_cwd(self, cwd: str = None) -> str:
        """解析安全的工作目录。"""
        if cwd:
            if os.path.isdir(cwd):
                return cwd
            logger.warning("CWD 不存在: %s，回退到当前目录", cwd)
        return os.getcwd()

    def _find_shell(self) -> str:
        """查找用户的登录 shell。"""
        if _IS_WINDOWS:
            return os.environ.get("COMSPEC", "cmd.exe")
        return os.environ.get("SHELL", "/bin/bash")

    def _sanitize_subprocess_env(self, base_env: dict, extra: dict = None) -> dict:
        """构建干净的子进程环境变量。"""
        env = dict(base_env)
        # 移除可能干扰子进程的代理变量
        for key in list(env.keys()):
            if key.lower() in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
                env.pop(key, None)
        if extra:
            env.update(extra)
        return env

    # ----- Spawn -----

    def spawn(
        self,
        command: str,
        cwd: str = None,
        task_id: str = "",
        session_key: str = "",
        env_vars: dict = None,
        background: bool = True,
    ) -> ProcessSession:
        """
        在本地生成一个后台进程。

        Args:
            command: 要执行的命令
            cwd: 工作目录
            task_id: 任务/沙箱隔离键
            session_key: 网关会话键
            env_vars: 额外的环境变量
            background: 是否作为后台进程运行
        """
        session = ProcessSession(
            id=f"proc_{uuid.uuid4().hex[:12]}",
            command=command,
            task_id=task_id,
            session_key=session_key,
            cwd=self._resolve_safe_cwd(cwd or os.getcwd()),
            started_at=time.time(),
        )

        # 使用用户的登录 shell 以保持一致性
        user_shell = self._find_shell()
        # 强制无缓冲输出，以便后台执行期间可以看到进度
        bg_env = self._sanitize_subprocess_env(os.environ, env_vars)
        bg_env["PYTHONUNBUFFERED"] = "1"

        if _IS_WINDOWS:
            # Windows: 使用 cmd /c 或 PowerShell
            if user_shell.lower().endswith("powershell.exe") or user_shell.lower().endswith("pwsh.exe"):
                proc = subprocess.Popen(
                    [user_shell, "-Command", command],
                    text=True,
                    cwd=session.cwd,
                    env=bg_env,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            else:
                proc = subprocess.Popen(
                    [user_shell, "/c", command],
                    text=True,
                    cwd=session.cwd,
                    env=bg_env,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
        else:
            # Unix: 使用 shell -lic
            proc = subprocess.Popen(
                [user_shell, "-lic", f"set +m; {command}"],
                text=True,
                cwd=session.cwd,
                env=bg_env,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                preexec_fn=os.setsid,
            )

        session.process = proc
        session.pid = proc.pid

        # 启动输出读取线程
        reader = threading.Thread(
            target=self._reader_loop,
            args=(session,),
            daemon=True,
            name=f"proc-reader-{session.id}",
        )
        session._reader_thread = reader
        reader.start()

        with self._lock:
            self._prune_if_needed()
            self._running[session.id] = session

        return session

    # ----- Reader Threads -----

    def _reader_loop(self, session: ProcessSession):
        """后台线程：从本地 Popen 进程读取 stdout。"""
        first_chunk = True
        try:
            while True:
                chunk = session.process.stdout.read(4096)
                if not chunk:
                    break
                if first_chunk:
                    chunk = self._clean_shell_noise(chunk)
                    first_chunk = False
                with session._lock:
                    session.output_buffer += chunk
                    if len(session.output_buffer) > session.max_output_chars:
                        session.output_buffer = session.output_buffer[-session.max_output_chars:]
        except Exception as e:
            logger.debug("进程 stdout 读取器结束: %s", e)
        finally:
            # 始终回收子进程以防止僵尸进程
            try:
                session.process.wait(timeout=5)
            except Exception as e:
                logger.debug("进程等待超时或失败: %s", e)
            session.exited = True
            session.exit_code = session.process.returncode
            self._move_to_finished(session)

    def _move_to_finished(self, session: ProcessSession):
        """将会话从运行中移动到已完成。

        幂等：如果会话已被移动（例如 kill_process 与读取器线程竞争），
        第二次调用是无操作 —— 不会重复入队完成通知。
        """
        with self._lock:
            was_running = self._running.pop(session.id, None) is not None
            self._finished[session.id] = session

        # 仅在第一次移动时入队完成通知
        if was_running:
            self.completion_queue.put({
                "type": "completion",
                "session_id": session.id,
                "command": session.command,
                "exit_code": session.exit_code,
                "output": session.output_buffer[-2000:] if session.output_buffer else "",
            })

    # ----- Query Methods -----

    def get(self, session_id: str) -> ProcessSession | None:
        """通过 ID 获取会话（运行中或已完成）。"""
        with self._lock:
            session = self._running.get(session_id) or self._finished.get(session_id)
        return session

    def poll(self, session_id: str) -> dict:
        """检查后台进程的状态并获取新输出。"""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"未找到进程 ID {session_id}"}

        with session._lock:
            output_preview = session.output_buffer[-1000:] if session.output_buffer else ""

        result = {
            "session_id": session.id,
            "command": session.command,
            "status": "exited" if session.exited else "running",
            "pid": session.pid,
            "uptime_seconds": int(time.time() - session.started_at),
            "output_preview": output_preview,
        }
        if session.exited:
            result["exit_code"] = session.exit_code
            self._completion_consumed.add(session_id)
        return result

    def read_log(self, session_id: str, offset: int = 0, limit: int = 200) -> dict:
        """读取完整输出日志，支持按行分页。"""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"未找到进程 ID {session_id}"}

        with session._lock:
            full_output = session.output_buffer

        lines = full_output.splitlines()
        total_lines = len(lines)

        # 默认：最后 N 行
        if offset == 0 and limit > 0:
            selected = lines[-limit:]
        else:
            selected = lines[offset:offset + limit]

        result = {
            "session_id": session.id,
            "status": "exited" if session.exited else "running",
            "output": "\n".join(selected),
            "total_lines": total_lines,
            "showing": f"{len(selected)} 行",
        }
        if session.exited:
            self._completion_consumed.add(session_id)
        return result

    def wait(self, session_id: str, timeout: int = None) -> dict:
        """
        阻塞直到进程退出、超时或中断。

        Args:
            session_id: 要等待的进程
            timeout: 最大阻塞秒数。回退到 TERMINAL_TIMEOUT 配置。

        Returns:
            包含 status ("exited", "timeout", "not_found") 和输出快照的 dict
        """
        try:
            default_timeout = int(os.getenv("TERMINAL_TIMEOUT", "180"))
        except (ValueError, TypeError):
            default_timeout = 180
        max_timeout = default_timeout
        requested_timeout = timeout
        timeout_note = None

        if requested_timeout and requested_timeout > max_timeout:
            effective_timeout = max_timeout
            timeout_note = (
                f"请求的等待时间 {requested_timeout}s 被限制为 "
                f"配置的最大值 {max_timeout}s"
            )
        else:
            effective_timeout = requested_timeout or max_timeout

        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"未找到进程 ID {session_id}"}

        deadline = time.monotonic() + effective_timeout

        while time.monotonic() < deadline:
            if session.exited:
                self._completion_consumed.add(session_id)
                result = {
                    "status": "exited",
                    "exit_code": session.exit_code,
                    "output": session.output_buffer[-2000:],
                }
                if timeout_note:
                    result["timeout_note"] = timeout_note
                return result

            time.sleep(1)

        result = {
            "status": "timeout",
            "output": session.output_buffer[-1000:],
        }
        if timeout_note:
            result["timeout_note"] = timeout_note
        else:
            result["timeout_note"] = f"已等待 {effective_timeout}s，进程仍在运行"
        return result

    def kill_process(self, session_id: str) -> dict:
        """终止一个后台进程。"""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"未找到进程 ID {session_id}"}

        if session.exited:
            return {
                "status": "already_exited",
                "exit_code": session.exit_code,
            }

        # 终止进程
        try:
            if session.process:
                # 本地进程 —— 终止进程组
                try:
                    if _IS_WINDOWS:
                        session.process.terminate()
                    else:
                        os.killpg(os.getpgid(session.process.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    session.process.kill()
            else:
                return {
                    "status": "error",
                    "error": "无法终止恢复的进程",
                }
            session.exited = True
            session.exit_code = -15  # SIGTERM
            self._move_to_finished(session)
            return {"status": "killed", "session_id": session.id}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def write_stdin(self, session_id: str, data: str) -> dict:
        """向运行中的进程 stdin 发送原始数据（不附加换行符）。"""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"未找到进程 ID {session_id}"}
        if session.exited:
            return {"status": "already_exited", "error": "进程已结束"}

        if not session.process or not session.process.stdin:
            return {"status": "error", "error": "进程 stdin 不可用"}
        try:
            session.process.stdin.write(data)
            session.process.stdin.flush()
            return {"status": "ok", "bytes_written": len(data)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def submit_stdin(self, session_id: str, data: str = "") -> dict:
        """向运行中的进程 stdin 发送数据 + 换行符（相当于按 Enter）。"""
        return self.write_stdin(session_id, data + "\n")

    def close_stdin(self, session_id: str) -> dict:
        """关闭运行中进程的 stdin / 发送 EOF 而不终止进程。"""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"未找到进程 ID {session_id}"}
        if session.exited:
            return {"status": "already_exited", "error": "进程已结束"}

        if not session.process or not session.process.stdin:
            return {"status": "error", "error": "进程 stdin 不可用"}
        try:
            session.process.stdin.close()
            return {"status": "ok", "message": "stdin 已关闭"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def list_sessions(self, task_id: str = None) -> list:
        """列出所有运行中和最近完成的进程。"""
        with self._lock:
            all_sessions = list(self._running.values()) + list(self._finished.values())

        if task_id:
            all_sessions = [s for s in all_sessions if s.task_id == task_id]

        result = []
        for s in all_sessions:
            entry = {
                "session_id": s.id,
                "command": s.command[:200],
                "cwd": s.cwd,
                "pid": s.pid,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(s.started_at)),
                "uptime_seconds": int(time.time() - s.started_at),
                "status": "exited" if s.exited else "running",
                "output_preview": s.output_buffer[-200:] if s.output_buffer else "",
            }
            if s.exited:
                entry["exit_code"] = s.exit_code
            if s.detached:
                entry["detached"] = True
            result.append(entry)
        return result

    # ----- Session/Task Queries -----

    def has_active_processes(self, task_id: str) -> bool:
        """检查任务 ID 是否有活动（运行中）的进程。"""
        with self._lock:
            return any(
                s.task_id == task_id and not s.exited
                for s in self._running.values()
            )

    def has_active_for_session(self, session_key: str) -> bool:
        """检查网关会话键是否有活动的进程。"""
        with self._lock:
            return any(
                s.session_key == session_key and not s.exited
                for s in self._running.values()
            )

    def kill_all(self, task_id: str = None) -> int:
        """终止所有运行中的进程，可选按 task_id 过滤。返回终止数量。"""
        with self._lock:
            targets = [
                s for s in self._running.values()
                if (task_id is None or s.task_id == task_id) and not s.exited
            ]

        killed = 0
        for session in targets:
            result = self.kill_process(session.id)
            if result.get("status") in ("killed", "already_exited"):
                killed += 1
        return killed

    # ----- Cleanup / Pruning -----

    def _prune_if_needed(self):
        """如果超过 MAX_PROCESSES，移除最旧的已完成会话。必须持有 _lock。"""
        # 首先修剪过期的已完成会话
        now = time.time()
        expired = [
            sid for sid, s in self._finished.items()
            if (now - s.started_at) > FINISHED_TTL_SECONDS
        ]
        for sid in expired:
            del self._finished[sid]
            self._completion_consumed.discard(sid)

        # 如果仍然超过限制，移除最旧的已完成会话
        total = len(self._running) + len(self._finished)
        if total >= MAX_PROCESSES and self._finished:
            oldest_id = min(self._finished, key=lambda sid: self._finished[sid].started_at)
            del self._finished[oldest_id]
            self._completion_consumed.discard(oldest_id)

        # 清理已不在跟踪中的 _completion_consumed 条目
        tracked = self._running.keys() | self._finished.keys()
        stale = self._completion_consumed - tracked
        if stale:
            self._completion_consumed -= stale


# 模块级单例
process_registry = ProcessRegistry()


# ---------------------------------------------------------------------------
# 工具注册
# ---------------------------------------------------------------------------

def register_process_tool():
    """注册 process 工具到工具注册表。"""
    from tools.registry import registry, tool_error

    PROCESS_DESCRIPTION = (
        "管理通过 terminal(background=true) 启动的后台进程。"
        "操作: 'list' (显示所有), 'poll' (检查状态 + 新输出), "
        "'log' (完整输出带分页), 'wait' (阻塞直到完成或超时), "
        "'kill' (终止), 'write' (发送原始 stdin 数据不带换行), "
        "'submit' (发送数据 + Enter，用于回答提示), 'close' (关闭 stdin/发送 EOF)。"
    )
    PROCESS_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "poll", "log", "wait", "kill", "write", "submit", "close"],
                "description": "对后台进程执行的操作"
            },
            "session_id": {
                "type": "string",
                "description": "进程会话 ID（来自 terminal background 输出）。除 'list' 外所有操作都需要。"
            },
            "data": {
                "type": "string",
                "description": "要发送到进程 stdin 的文本（用于 'write' 和 'submit' 操作）"
            },
            "timeout": {
                "type": "integer",
                "description": "'wait' 操作的最大阻塞秒数。超时返回部分输出。",
                "minimum": 1
            },
            "offset": {
                "type": "integer",
                "description": "'log' 操作的行偏移量（默认：最后 200 行）"
            },
            "limit": {
                "type": "integer",
                "description": "'log' 操作返回的最大行数",
                "minimum": 1
            }
        },
        "required": ["action"]
    }

    def _handle_process(args, **kw):
        task_id = kw.get("task_id")
        action = args.get("action", "")
        # 强制转换为字符串 —— 某些模型将 session_id 作为整数发送
        session_id = str(args.get("session_id", "")) if args.get("session_id") is not None else ""

        if action == "list":
            return json.dumps({"processes": process_registry.list_sessions(task_id=task_id)}, ensure_ascii=False)
        elif action in ("poll", "log", "wait", "kill", "write", "submit", "close"):
            if not session_id:
                return tool_error(f"{action} 操作需要 session_id")
            if action == "poll":
                return json.dumps(process_registry.poll(session_id), ensure_ascii=False)
            elif action == "log":
                return json.dumps(process_registry.read_log(
                    session_id, offset=args.get("offset", 0), limit=args.get("limit", 200)), ensure_ascii=False)
            elif action == "wait":
                return json.dumps(process_registry.wait(session_id, timeout=args.get("timeout")), ensure_ascii=False)
            elif action == "kill":
                return json.dumps(process_registry.kill_process(session_id), ensure_ascii=False)
            elif action == "write":
                return json.dumps(process_registry.write_stdin(session_id, str(args.get("data", ""))), ensure_ascii=False)
            elif action == "submit":
                return json.dumps(process_registry.submit_stdin(session_id, str(args.get("data", ""))), ensure_ascii=False)
            elif action == "close":
                return json.dumps(process_registry.close_stdin(session_id), ensure_ascii=False)
        return tool_error(f"未知的 process 操作: {action}。可用: list, poll, log, wait, kill, write, submit, close")

    registry.register(
        name="process",
        toolset="terminal",
        schema=PROCESS_SCHEMA,
        description=PROCESS_DESCRIPTION,
        handler=_handle_process,
        emoji="⚙️",
    )
