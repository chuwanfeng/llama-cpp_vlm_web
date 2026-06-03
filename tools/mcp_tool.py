#!/usr/bin/env python3
"""
MCP (Model Context Protocol) 客户端支持 — 简化版

连接外部 MCP 服务器（stdio 或 HTTP），发现其工具并注册到本项目的工具注册表，
使 Agent 可以像调用内置工具一样调用 MCP 工具。

配置读取自项目根目录的 mcp_servers.json（或从主配置中的 mcp_servers 键）。

示例配置 (mcp_servers.json)::

    {
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "timeout": 120
      },
      "github": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {
          "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
        }
      },
      "remote_api": {
        "url": "https://my-mcp-server.example.com/mcp",
        "headers": {
          "Authorization": "Bearer sk-..."
        },
        "timeout": 180
      }
    }

特性:
    - Stdio 传输（command + args）和 HTTP 传输（url）
    - 自动重连（指数退避，最多 5 次）
    - 环境变量过滤（安全）
    - 凭据脱敏（错误消息中）
    - 线程安全架构，专用后台事件循环
    - 动态工具刷新（tools/list_changed 通知）

架构:
    一个专用后台事件循环 (_mcp_loop) 在守护线程中运行。
    每个 MCP 服务器作为一个长生命周期的 asyncio Task 在该循环上运行，
    保持传输连接上下文。工具调用协程通过 run_coroutine_threadsafe() 调度到该循环。

    关闭时，每个服务器 Task 被信号通知退出 async with 块，
    确保 anyio cancel-scope 清理在打开连接的同一个 Task 中完成。

线程安全:
    _servers 和 _mcp_loop/_mcp_thread 从 MCP 后台线程和调用线程访问。
    所有修改受 _lock 保护。

依赖:
    mcp Python 包（可选）。未安装时本模块为 no-op。
"""

import asyncio
import concurrent.futures
import inspect
import json
import logging
import math
import os
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 优雅导入 — MCP SDK 是可选依赖
# ---------------------------------------------------------------------------

_MCP_AVAILABLE = False
_MCP_HTTP_AVAILABLE = False
_MCP_NOTIFICATION_TYPES = False
_MCP_MESSAGE_HANDLER_SUPPORTED = False
LATEST_PROTOCOL_VERSION = "2025-03-26"

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _MCP_AVAILABLE = True
    try:
        from mcp.client.streamable_http import streamablehttp_client
        _MCP_HTTP_AVAILABLE = True
    except ImportError:
        _MCP_HTTP_AVAILABLE = False
    try:
        from mcp.client.streamable_http import streamable_http_client
        _MCP_NEW_HTTP = True
    except ImportError:
        _MCP_NEW_HTTP = False
    try:
        from mcp.types import LATEST_PROTOCOL_VERSION
    except ImportError:
        logger.debug("mcp.types.LATEST_PROTOCOL_VERSION 不可用 — 使用回退协议版本")
    try:
        from mcp.types import (
            ServerNotification,
            ToolListChangedNotification,
            PromptListChangedNotification,
            ResourceListChangedNotification,
        )
        _MCP_NOTIFICATION_TYPES = True
    except ImportError:
        logger.debug("MCP 通知类型不可用 — 动态工具发现禁用")
except ImportError:
    logger.debug("mcp 包未安装 — MCP 工具支持禁用")


def _check_message_handler_support() -> bool:
    """检查 ClientSession 是否接受 message_handler 参数。

    检查构造函数签名以兼容不支持通知处理程序的旧版 MCP SDK。
    """
    if not _MCP_AVAILABLE:
        return False
    try:
        return "message_handler" in inspect.signature(ClientSession).parameters
    except (TypeError, ValueError):
        return False


_MCP_MESSAGE_HANDLER_SUPPORTED = _check_message_handler_support()
if _MCP_AVAILABLE and not _MCP_MESSAGE_HANDLER_SUPPORTED:
    logger.debug("MCP SDK 不支持 message_handler — 动态工具发现禁用")

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_DEFAULT_TOOL_TIMEOUT = 120      # 工具调用超时（秒）
_DEFAULT_CONNECT_TIMEOUT = 60    # 初始连接超时（秒）
_MAX_RECONNECT_RETRIES = 5       # 最大重连次数
_MAX_INITIAL_CONNECT_RETRIES = 3 # 首次连接尝试的最大重试次数
_MAX_BACKOFF_SECONDS = 60        # 最大退避秒数

# 可安全传递给 stdio 子进程的环境变量
_SAFE_ENV_KEYS = frozenset({
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR",
})

# 凭据模式正则 — 用于从错误消息中脱敏
_CREDENTIAL_PATTERN = re.compile(
    r"(?:"
    r"ghp_[A-Za-z0-9_]{1,255}"           # GitHub PAT
    r"|sk-[A-Za-z0-9_]{1,255}"           # OpenAI 风格密钥
    r"|Bearer\s+\S+"                      # Bearer token
    r"|token=[^\s&,;\"']{1,255}"         # token=...
    r"|key=[^\s&,;\"']{1,255}"           # key=...
    r"|API_KEY=[^\s&,;\"']{1,255}"       # API_KEY=...
    r"|password=[^\s&,;\"']{1,255}"      # password=...
    r"|secret=[^\s&,;\"']{1,255}"        # secret=...
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 安全辅助函数
# ---------------------------------------------------------------------------

def _build_safe_env(user_env: Optional[dict]) -> dict:
    """为 stdio 子进程构建过滤后的环境变量字典。

    只传递安全的基础变量（PATH、HOME 等）和 XDG_* 变量，
    加上用户在服务器配置中显式指定的变量。
    防止意外将 API 密钥、令牌等凭据泄露给 MCP 服务器子进程。
    """
    env = {}
    for key, value in os.environ.items():
        if key in _SAFE_ENV_KEYS or key.startswith("XDG_"):
            env[key] = value
    if user_env:
        env.update(user_env)
    return env


def _sanitize_error(text: str) -> str:
    """从错误文本中脱敏凭据类模式，再返回给 LLM。

    将令牌、密钥和其他机密替换为 [REDACTED]，防止工具错误响应中意外暴露凭据。
    """
    return _CREDENTIAL_PATTERN.sub("[REDACTED]", text)


# ---------------------------------------------------------------------------
# 提示注入检测
# ---------------------------------------------------------------------------

# 指示 MCP 工具描述中潜在提示注入的模式。
# 这些是 WARNING 级别 — 我们记录但不阻止，因为误报会破坏合法的 MCP 服务器。
_MCP_INJECTION_PATTERNS = [
    (re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
     "提示覆盖尝试 ('ignore previous instructions')"),
    (re.compile(r"you\s+are\s+now\s+a", re.I),
     "身份覆盖尝试 ('you are now a...')"),
    (re.compile(r"your\s+new\s+(task|role|instructions?)\s+(is|are)", re.I),
     "任务覆盖尝试"),
    (re.compile(r"system\s*:\s*", re.I),
     "系统提示注入尝试"),
    (re.compile(r"<\s*(system|human|assistant)\s*>", re.I),
     "角色标签注入尝试"),
    (re.compile(r"do\s+not\s+(tell|inform|mention|reveal)", re.I),
     "隐藏指令"),
    (re.compile(r"(curl|wget|fetch)\s+https?://", re.I),
     "描述中的网络命令"),
    (re.compile(r"base64\.(b64decode|decodebytes)", re.I),
     "base64 解码引用"),
    (re.compile(r"exec\s*\(|eval\s*\(", re.I),
     "代码执行引用"),
    (re.compile(r"import\s+(subprocess|os|shutil|socket)", re.I),
     "危险导入引用"),
]


def _scan_mcp_description(server_name: str, tool_name: str, description: str) -> List[str]:
    """扫描 MCP 工具描述中的提示注入模式。

    返回发现字符串列表（空 = 干净）。
    """
    findings = []
    if not description:
        return findings
    for pattern, reason in _MCP_INJECTION_PATTERNS:
        if pattern.search(description):
            findings.append(reason)
    if findings:
        logger.warning(
            "MCP 服务器 '%s' 工具 '%s': 可疑描述内容 — %s. "
            "描述: %.200s",
            server_name, tool_name, "; ".join(findings),
            description,
        )
    return findings


def _prepend_path(env: dict, directory: str) -> dict:
    """如果 directory 尚未存在，将其前置到 env PATH。"""
    updated = dict(env or {})
    if not directory:
        return updated

    existing = updated.get("PATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if directory not in parts:
        parts = [directory, *parts]
    updated["PATH"] = os.pathsep.join(parts) if parts else directory
    return updated


def _resolve_stdio_command(command: str, env: dict) -> tuple[str, dict]:
    """根据确切的子进程环境解析 stdio MCP 命令。

    主要存在目的是使裸 npx/npm/node 命令即使在 MCP 子进程运行于过滤后的 PATH 下也能可靠工作。
    """
    resolved_command = os.path.expanduser(str(command).strip())
    resolved_env = dict(env or {})

    if os.sep not in resolved_command:
        path_arg = resolved_env["PATH"] if "PATH" in resolved_env else None
        which_hit = shutil.which(resolved_command, path=path_arg)
        if which_hit:
            resolved_command = which_hit
        elif resolved_command in {"npx", "npm", "node"}:
            candidates = [
                os.path.join(os.path.expanduser("~"), ".local", "bin", resolved_command),
            ]
            for candidate in candidates:
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    resolved_command = candidate
                    break

    command_dir = os.path.dirname(resolved_command)
    if command_dir:
        resolved_env = _prepend_path(resolved_env, command_dir)

    return resolved_command, resolved_env


def _format_connect_error(exc: BaseException) -> str:
    """将嵌套的 MCP 连接错误渲染为可操作的短消息。"""

    def _find_missing(current: BaseException) -> Optional[str]:
        nested = getattr(current, "exceptions", None)
        if nested:
            for child in nested:
                missing = _find_missing(child)
                if missing:
                    return missing
            return None
        if isinstance(current, FileNotFoundError):
            if getattr(current, "filename", None):
                return str(current.filename)
            match = re.search(r"No such file or directory: '([^']+)'", str(current))
            if match:
                return match.group(1)
        for attr in ("__cause__", "__context__"):
            nested_exc = getattr(current, attr, None)
            if isinstance(nested_exc, BaseException):
                missing = _find_missing(nested_exc)
                if missing:
                    return missing
        return None

    def _flatten_messages(current: BaseException) -> List[str]:
        nested = getattr(current, "exceptions", None)
        if nested:
            flattened: List[str] = []
            for child in nested:
                flattened.extend(_flatten_messages(child))
            return flattened
        messages = []
        text = str(current).strip()
        if text:
            messages.append(text)
        for attr in ("__cause__", "__context__"):
            nested_exc = getattr(current, attr, None)
            if isinstance(nested_exc, BaseException):
                messages.extend(_flatten_messages(nested_exc))
        return messages or [current.__class__.__name__]

    missing = _find_missing(exc)
    if missing:
        message = f"缺少可执行文件 '{missing}'"
        if os.path.basename(missing) in {"npx", "npm", "node"}:
            message += (
                "（确保已安装 Node.js 且 PATH 包含其 bin 目录，"
                "或将 mcp_servers.<name>.command 设为绝对路径并"
                "将该目录包含在 mcp_servers.<name>.env.PATH 中）"
            )
        return _sanitize_error(message)

    deduped: List[str] = []
    for item in _flatten_messages(exc):
        if item not in deduped:
            deduped.append(item)
    return _sanitize_error("; ".join(deduped[:3]))


# ---------------------------------------------------------------------------
# 服务器任务 — 每个 MCP 服务器在一个长生命周期的 asyncio Task 中运行
# ---------------------------------------------------------------------------

class MCPServerTask:
    """在专用 asyncio Task 中管理单个 MCP 服务器连接。

    整个连接生命周期（连接、发现、服务、断开）在一个 asyncio Task 中运行，
    以便 anyio cancel-scope 在传输客户端创建和销毁的同一个 Task 上下文中进入和退出。

    支持 stdio 和 HTTP/StreamableHTTP 传输。
    """

    __slots__ = (
        "name", "session", "tool_timeout",
        "_task", "_ready", "_shutdown_event", "_reconnect_event",
        "_tools", "_error", "_config",
        "_registered_tool_names", "_refresh_lock",
        "_rpc_lock", "_pending_refresh_tasks",
    )

    def __init__(self, name: str):
        self.name = name
        self.session: Optional[Any] = None
        self.tool_timeout: float = _DEFAULT_TOOL_TIMEOUT
        self._task: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._reconnect_event = asyncio.Event()
        self._tools: list = []
        self._error: Optional[Exception] = None
        self._config: dict = {}
        self._registered_tool_names: list[str] = []
        self._refresh_lock = asyncio.Lock()
        # MCP stdio 会话是单个 JSON-RPC 流。某些服务器在启动时发出 list_changed 通知；
        # 如果通知处理程序在正常工具调用正在执行时调用 list_tools，流可能阻塞，
        # 导致用户可见的工具调用超时。按服务器序列化客户端发起的 RPC。
        self._rpc_lock = asyncio.Lock()
        self._pending_refresh_tasks: set[asyncio.Task] = set()

    def _is_http(self) -> bool:
        """检查此服务器是否使用 HTTP 传输。"""
        return "url" in self._config

    # ----- 动态工具发现 (notifications/tools/list_changed) -----

    async def _refresh_tools_task(self):
        """运行动态工具刷新并记录后台任务中的失败。"""
        try:
            await self._refresh_tools()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("MCP 服务器 '%s': 动态工具刷新失败", self.name)

    def _schedule_tools_refresh(self) -> asyncio.Task:
        """调度后台工具刷新并保持强引用。"""
        task = asyncio.create_task(self._refresh_tools_task())
        self._pending_refresh_tasks.add(task)
        task.add_done_callback(self._pending_refresh_tasks.discard)
        return task

    def _make_message_handler(self):
        """为 ClientSession 构建 message_handler 回调。

        按通知类型分发。只有 ToolListChangedNotification 触发刷新；
        prompt 和 resource 变更通知记录为存根供将来使用。
        """
        async def _handler(message):
            try:
                if isinstance(message, Exception):
                    logger.debug("MCP 消息处理程序 (%s): 异常: %s", self.name, message)
                    return
                if _MCP_NOTIFICATION_TYPES and isinstance(message, ServerNotification):
                    match message.root:
                        case ToolListChangedNotification():
                            logger.info(
                                "MCP 服务器 '%s': 收到 tools/list_changed 通知",
                                self.name,
                            )
                            # 某些服务器在 initialize 后立即发出 tools/list_changed，
                            # 而客户端可能已在执行另一个请求。在 SDK 通知处理程序中同步刷新
                            # 可能与该请求竞争并阻塞 stdio JSON-RPC 流，导致所有后续工具调用超时。
                            # 在单独的任务中执行刷新，让处理程序快速返回。
                            self._schedule_tools_refresh()
                            await asyncio.sleep(0)
                        case PromptListChangedNotification():
                            logger.debug("MCP 服务器 '%s': prompts/list_changed (忽略)", self.name)
                        case ResourceListChangedNotification():
                            logger.debug("MCP 服务器 '%s': resources/list_changed (忽略)", self.name)
                        case _:
                            pass
            except Exception:
                logger.exception("MCP 消息处理程序 '%s' 出错", self.name)
        return _handler

    async def _refresh_tools(self):
        """从服务器重新获取工具并更新注册表。

        服务器发送 notifications/tools/list_changed 时调用。
        锁防止来自快速连续通知的重叠刷新。
        初始 await (list_tools) 之后，所有修改都是同步的 — 从事件循环角度看是原子的。
        """
        from tools.registry import registry

        async with self._refresh_lock:
            old_tool_names = set(self._registered_tool_names)

            async with self._rpc_lock:
                tools_result = await self.session.list_tools()
            new_mcp_tools = tools_result.tools if hasattr(tools_result, "tools") else []

            stale_tool_names = old_tool_names - {
                f"mcp_{_sanitize_mcp_name_component(self.name)}_"
                f"{_sanitize_mcp_name_component(tool.name)}"
                for tool in new_mcp_tools
            }
            for tool_name in stale_tool_names:
                registry.deregister(tool_name)

            self._tools = new_mcp_tools
            self._registered_tool_names = _register_server_tools(
                self.name, self, self._config
            )

            new_tool_names = set(self._registered_tool_names)
            added = new_tool_names - old_tool_names
            removed = old_tool_names - new_tool_names
            changes = []
            if added:
                changes.append(f"新增: {', '.join(sorted(added))}")
            if removed:
                changes.append(f"移除: {', '.join(sorted(removed))}")
            if changes:
                logger.warning(
                    "MCP 服务器 '%s': 工具动态变更 — %s. "
                    "请验证这些变更是否符合预期。",
                    self.name, "; ".join(changes),
                )
            else:
                logger.info(
                    "MCP 服务器 '%s': 动态刷新 %d 个工具（无变更）",
                    self.name, len(self._registered_tool_names),
                )

    async def _wait_for_lifecycle_event(self) -> str:
        """阻塞直到 _shutdown_event 或 _reconnect_event 触发。

        返回:
            "shutdown"  — 服务器应完全退出运行循环。
            "reconnect" — 服务器应拆除当前 MCP 会话并重新进入传输。
                          reconnect 事件在返回前被清除，以便下一个周期以新信号开始。

        如果两个事件同时设置，shutdown 优先。
        """
        shutdown_task = asyncio.create_task(self._shutdown_event.wait())
        reconnect_task = asyncio.create_task(self._reconnect_event.wait())
        try:
            await asyncio.wait(
                {shutdown_task, reconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for t in (shutdown_task, reconnect_task):
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass

        if self._shutdown_event.is_set():
            return "shutdown"
        self._reconnect_event.clear()
        return "reconnect"

    async def _run_stdio(self, config: dict):
        """使用 stdio 传输运行服务器。"""
        command = config.get("command")
        args = config.get("args", [])
        user_env = config.get("env")

        if not command:
            raise ValueError(
                f"MCP 服务器 '{self.name}' 配置中没有 'command'"
            )

        safe_env = _build_safe_env(user_env)
        command, safe_env = _resolve_stdio_command(command, safe_env)

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=safe_env if safe_env else None,
        )

        kwargs = {}
        if _MCP_NOTIFICATION_TYPES and _MCP_MESSAGE_HANDLER_SUPPORTED:
            kwargs["message_handler"] = self._make_message_handler()

        async with stdio_client(server_params) as (
            read_stream,
            write_stream,
        ):
            async with ClientSession(
                read_stream, write_stream, **kwargs
            ) as session:
                await session.initialize()
                self.session = session
                await self._discover_tools()
                self._ready.set()
                await self._wait_for_lifecycle_event()

    async def _run_http(self, config: dict):
        """使用 HTTP/StreamableHTTP 传输运行服务器。"""
        if not _MCP_HTTP_AVAILABLE:
            raise ImportError(
                f"MCP 服务器 '{self.name}' 需要 HTTP 传输但 "
                "mcp.client.streamable_http 不可用。"
                "升级 mcp 包以获取 HTTP 支持。"
            )

        url = config["url"]
        headers = dict(config.get("headers") or {})
        if not any(key.lower() == "mcp-protocol-version" for key in headers):
            headers["mcp-protocol-version"] = LATEST_PROTOCOL_VERSION
        connect_timeout = config.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT)
        ssl_verify = config.get("ssl_verify", True)

        kwargs = {}
        if _MCP_NOTIFICATION_TYPES and _MCP_MESSAGE_HANDLER_SUPPORTED:
            kwargs["message_handler"] = self._make_message_handler()

        if _MCP_NEW_HTTP:
            import httpx

            client_kwargs: dict = {
                "follow_redirects": True,
                "timeout": httpx.Timeout(float(connect_timeout), read=300.0),
                "verify": ssl_verify,
            }
            if headers:
                client_kwargs["headers"] = headers

            async with httpx.AsyncClient(**client_kwargs) as http_client:
                async with streamable_http_client(url, http_client=http_client) as (
                    read_stream, write_stream, _get_session_id,
                ):
                    async with ClientSession(read_stream, write_stream, **kwargs) as session:
                        await session.initialize()
                        self.session = session
                        await self._discover_tools()
                        self._ready.set()
                        reason = await self._wait_for_lifecycle_event()
                        if reason == "reconnect":
                            logger.info(
                                "MCP 服务器 '%s': 请求重连 — "
                                "拆除 HTTP 会话", self.name,
                            )
        else:
            _http_kwargs: dict = {
                "headers": headers,
                "timeout": float(connect_timeout),
                "verify": ssl_verify,
            }
            async with streamablehttp_client(url, **_http_kwargs) as (
                read_stream, write_stream, _get_session_id,
            ):
                async with ClientSession(read_stream, write_stream, **kwargs) as session:
                    await session.initialize()
                    self.session = session
                    await self._discover_tools()
                    self._ready.set()
                    reason = await self._wait_for_lifecycle_event()
                    if reason == "reconnect":
                        logger.info(
                            "MCP 服务器 '%s': 请求重连 — "
                            "拆除旧版 HTTP 会话", self.name,
                        )

    async def _discover_tools(self):
        """从已连接的会话中发现工具。"""
        if self.session is None:
            return
        async with self._rpc_lock:
            tools_result = await self.session.list_tools()
        self._tools = (
            tools_result.tools
            if hasattr(tools_result, "tools")
            else []
        )

    async def run(self, config: dict):
        """长生命周期协程：连接、发现工具、等待、断开。

        如果连接意外断开，包含指数退避的自动重连（除非请求了关闭）。
        """
        self._config = config
        self.tool_timeout = config.get("timeout", _DEFAULT_TOOL_TIMEOUT)

        if "url" in config and "command" in config:
            logger.warning(
                "MCP 服务器 '%s' 配置中同时有 'url' 和 'command'。"
                "使用 HTTP 传输 ('url')。移除 'command' 可消除此警告。",
                self.name,
            )
        retries = 0
        initial_retries = 0
        backoff = 1.0

        while True:
            try:
                if self._is_http():
                    await self._run_http(config)
                else:
                    await self._run_stdio(config)
                # 传输干净返回。两种情况：
                #  - _shutdown_event 被设置：完全退出运行循环。
                #  - _reconnect_event 被设置（手动刷新）：循环回来重建 MCP 会话。
                #    不要碰重试计数器 — 这不是失败。
                if self._shutdown_event.is_set():
                    break
                logger.info(
                    "MCP 服务器 '%s': 重连中（手动刷新）",
                    self.name,
                )
                self.session = None
                continue
            except Exception as exc:
                self.session = None

                if not self._ready.is_set():
                    initial_retries += 1
                    if initial_retries > _MAX_INITIAL_CONNECT_RETRIES:
                        logger.warning(
                            "MCP 服务器 '%s' 初始连接 %d 次尝试后失败，放弃: %s",
                            self.name, _MAX_INITIAL_CONNECT_RETRIES, exc,
                        )
                        self._error = exc
                        self._ready.set()
                        return

                    logger.warning(
                        "MCP 服务器 '%s' 初始连接失败 "
                        "(尝试 %d/%d)，%.0fs 后重试: %s",
                        self.name, initial_retries,
                        _MAX_INITIAL_CONNECT_RETRIES, backoff, exc,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)

                    if self._shutdown_event.is_set():
                        self._error = exc
                        self._ready.set()
                        return
                    continue

                if self._shutdown_event.is_set():
                    logger.debug(
                        "MCP 服务器 '%s' 关闭期间断开: %s",
                        self.name, exc,
                    )
                    return

                retries += 1
                if retries > _MAX_RECONNECT_RETRIES:
                    logger.warning(
                        "MCP 服务器 '%s' %d 次重连尝试后失败，放弃: %s",
                        self.name, _MAX_RECONNECT_RETRIES, exc,
                    )
                    return

                logger.warning(
                    "MCP 服务器 '%s' 连接丢失 (尝试 %d/%d)，"
                    "%.0fs 后重连: %s",
                    self.name, retries, _MAX_RECONNECT_RETRIES,
                    backoff, exc,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)

                if self._shutdown_event.is_set():
                    return
            finally:
                self.session = None

    async def start(self, config: dict):
        """创建后台 Task 并等待就绪（或失败）。"""
        self._task = asyncio.ensure_future(self.run(config))
        await self._ready.wait()
        if self._error:
            raise self._error

    async def shutdown(self):
        """信号通知 Task 退出并等待干净资源拆除。"""
        from tools.registry import registry

        self._shutdown_event.set()
        self._reconnect_event.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP 服务器 '%s' 关闭超时，取消任务",
                    self.name,
                )
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        if self._pending_refresh_tasks:
            for task in list(self._pending_refresh_tasks):
                task.cancel()
            await asyncio.gather(*self._pending_refresh_tasks, return_exceptions=True)
            self._pending_refresh_tasks.clear()
        for tool_name in list(getattr(self, "_registered_tool_names", [])):
            registry.deregister(tool_name)
        self._registered_tool_names = []
        self.session = None


# ---------------------------------------------------------------------------
# 模块级状态
# ---------------------------------------------------------------------------

_servers: Dict[str, MCPServerTask] = {}

# 专用事件循环在后台守护线程中运行。
_mcp_loop: Optional[asyncio.AbstractEventLoop] = None
_mcp_thread: Optional[threading.Thread] = None

# 保护 _mcp_loop, _mcp_thread, _servers。
_lock = threading.Lock()


def _mcp_loop_exception_handler(loop, context):
    """抑制关闭期间良性的 'Event loop is closed' 噪音。

    MCP 事件循环停止并关闭时，httpx/httpcore 异步传输可能触发 __del__ 终结器
    在已死的循环上调用 call_soon()。asyncio 捕获该 RuntimeError 并路由到这里。
    我们静默它因为连接反正正在拆除；所有其他异常转发到默认处理程序。
    """
    exc = context.get("exception")
    if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
        return
    loop.default_exception_handler(context)


def _ensure_mcp_loop():
    """如果尚未运行，启动后台事件循环线程。"""
    global _mcp_loop, _mcp_thread
    with _lock:
        if _mcp_loop is not None and _mcp_loop.is_running():
            return
        _mcp_loop = asyncio.new_event_loop()
        _mcp_loop.set_exception_handler(_mcp_loop_exception_handler)
        _mcp_thread = threading.Thread(
            target=_mcp_loop.run_forever,
            name="mcp-event-loop",
            daemon=True,
        )
        _mcp_thread.start()


def _run_on_mcp_loop(coro, timeout: float = 30):
    """在 MCP 事件循环上调度协程并阻塞直到完成。

    以短间隔轮询，以便调用线程可以在 MCP 工作仍在后台循环运行时响应用户中断。
    """
    with _lock:
        loop = _mcp_loop
    if loop is None or not loop.is_running():
        raise RuntimeError("MCP 事件循环未运行")
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    deadline = None if timeout is None else time.monotonic() + timeout

    while True:
        wait_timeout = 0.1
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return future.result(timeout=0)
            wait_timeout = min(wait_timeout, remaining)

        try:
            return future.result(timeout=wait_timeout)
        except concurrent.futures.TimeoutError:
            continue


def _interrupted_call_result() -> str:
    """用户中断的 MCP 工具调用的标准化 JSON 错误。"""
    return json.dumps({
        "error": "MCP 调用被中断"
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def _interpolate_env_vars(value):
    """递归解析 ${VAR} 占位符到 os.environ。"""
    if isinstance(value, str):
        def _replace(m):
            return os.environ.get(m.group(1), m.group(0))
        return re.sub(r"\$\{([^}]+)\}", _replace, value)
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(v) for v in value]
    return value


def _load_mcp_config() -> Dict[str, dict]:
    """读取 mcp_servers.json 或主配置中的 mcp_servers 键。

    返回 {server_name: server_config} 或空 dict。
    服务器配置可包含 command/args/env（stdio 传输）或 url/headers（HTTP 传输），
    加上可选的 timeout、connect_timeout。

    字符串值中的 ${ENV_VAR} 占位符从 os.environ 解析。
    """
    # 首先尝试 mcp_servers.json
    import os
    config_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "mcp_servers.json"),
        os.path.join(os.path.expanduser("~"), ".llama_vlm_web", "mcp_servers.json"),
    ]
    for path in config_paths:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    servers = json.load(f)
                if isinstance(servers, dict):
                    return {name: _interpolate_env_vars(cfg) for name, cfg in servers.items()}
            except Exception as exc:
                logger.debug("加载 MCP 配置失败 %s: %s", path, exc)

    # 然后尝试主配置
    try:
        import os
        main_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
        if os.path.isfile(main_config_path):
            with open(main_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            servers = config.get("mcp_servers")
            if isinstance(servers, dict):
                return {name: _interpolate_env_vars(cfg) for name, cfg in servers.items()}
    except Exception as exc:
        logger.debug("从主配置加载 MCP 配置失败: %s", exc)

    return {}


# ---------------------------------------------------------------------------
# 服务器连接辅助
# ---------------------------------------------------------------------------

async def _connect_server(name: str, config: dict) -> MCPServerTask:
    """创建 MCPServerTask，启动它，并在就绪时返回。

    服务器 Task 在后台保持连接存活。
    调用 server.shutdown()（在同一个事件循环上）以拆除。

    抛出:
        ValueError: 缺少必需的配置键。
        ImportError: 需要 HTTP 传输但不可用。
        Exception: 连接或初始化失败。
    """
    server = MCPServerTask(name)
    await server.start(config)
    return server


# ---------------------------------------------------------------------------
# Handler / check-fn 工厂
# ---------------------------------------------------------------------------

def _make_tool_handler(server_name: str, tool_name: str, tool_timeout: float):
    """返回通过后台循环调用 MCP 工具的同步处理程序。

    处理程序符合注册表的调度接口: handler(args_dict, **kwargs) -> str
    """

    def _handler(args: dict, **kwargs) -> str:
        with _lock:
            server = _servers.get(server_name)
        if not server or not server.session:
            return json.dumps({
                "error": f"MCP 服务器 '{server_name}' 未连接"
            }, ensure_ascii=False)

        async def _call():
            async with server._rpc_lock:
                result = await server.session.call_tool(tool_name, arguments=args)
            if result.isError:
                error_text = ""
                for block in (result.content or []):
                    if hasattr(block, "text"):
                        error_text += block.text
                return json.dumps({
                    "error": _sanitize_error(
                        error_text or "MCP 工具返回错误"
                    )
                }, ensure_ascii=False)

            parts: List[str] = []
            for block in (result.content or []):
                if hasattr(block, "text"):
                    parts.append(block.text)
            text_result = "\n".join(parts) if parts else ""

            structured = getattr(result, "structuredContent", None)
            if structured is not None:
                if text_result:
                    return json.dumps({
                        "result": text_result,
                        "structuredContent": structured,
                    }, ensure_ascii=False)
                return json.dumps({"result": structured}, ensure_ascii=False)
            return json.dumps({"result": text_result}, ensure_ascii=False)

        def _call_once():
            return _run_on_mcp_loop(_call(), timeout=tool_timeout)

        try:
            result = _call_once()
            return result
        except Exception as exc:
            logger.error(
                "MCP 工具 %s/%s 调用失败: %s",
                server_name, tool_name, exc,
            )
            return json.dumps({
                "error": _sanitize_error(
                    f"MCP 调用失败: {type(exc).__name__}: {exc}"
                )
            }, ensure_ascii=False)

    return _handler


def _make_check_fn(server_name: str):
    """返回验证 MCP 连接是否存活的检查函数。"""

    def _check() -> bool:
        with _lock:
            server = _servers.get(server_name)
        return server is not None and server.session is not None

    return _check


# ---------------------------------------------------------------------------
# 发现与注册
# ---------------------------------------------------------------------------

def _normalize_mcp_input_schema(schema: dict | None) -> dict:
    """规范化 MCP 输入模式以兼容 LLM 工具调用。

    MCP 服务器可能发出带有 definitions / #/definitions/... 引用的纯 JSON Schema。
    某些提供商（Kimi / Moonshot）拒绝该形式，要求本地引用指向 #/$defs/...。
    在这里规范化常见的 draft-07 形状，使 MCP 工具模式在各 OpenAI 兼容提供商间可移植。

    额外应用的健壮性修复:
    * 缺少或 null 的 type 在对象形状节点上强制为 "object"。
    * 当 object 节点缺少 properties 时，添加空 properties dict。
    * required 数组裁剪为仅存在于 properties 中的名称。
    * MCP/Pydantic 可选字段常作为 anyOf: [{...}, {"type": "null"}] 到达。
      将可空联合折叠为非空分支。
    """
    if not schema:
        return {"type": "object", "properties": {}}

    def _rewrite_local_refs(node):
        if isinstance(node, dict):
            normalized = {}
            for key, value in node.items():
                out_key = "$defs" if key == "definitions" else key
                normalized[out_key] = _rewrite_local_refs(value)
            ref = normalized.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/definitions/"):
                normalized["$ref"] = "#/$defs/" + ref[len("#/definitions/"):]
            return normalized
        if isinstance(node, list):
            return [_rewrite_local_refs(item) for item in node]
        return node

    def _strip_nullable_union(node):
        """将 JSON Schema 可空联合折叠为提供商安全的非空模式。"""
        if isinstance(node, list):
            return [_strip_nullable_union(item) for item in node]
        if not isinstance(node, dict):
            return node

        any_of = node.get("anyOf")
        if isinstance(any_of, list) and len(any_of) == 2:
            null_branch = None
            non_null_branch = None
            for branch in any_of:
                if isinstance(branch, dict) and branch.get("type") == "null":
                    null_branch = branch
                else:
                    non_null_branch = branch
            if null_branch and non_null_branch:
                result = dict(node)
                del result["anyOf"]
                if isinstance(non_null_branch, dict):
                    for k, v in non_null_branch.items():
                        if k not in result:
                            result[k] = v
                result["nullable"] = True
                return result

        return {k: _strip_nullable_union(v) for k, v in node.items()}

    def _repair_object_shape(node):
        """递归修复对象形状节点: 填充 type，裁剪 required。"""
        if isinstance(node, list):
            return [_repair_object_shape(item) for item in node]
        if not isinstance(node, dict):
            return node

        repaired = {k: _repair_object_shape(v) for k, v in node.items()}

        if not repaired.get("type") and (
            "properties" in repaired or "required" in repaired
        ):
            repaired["type"] = "object"

        if repaired.get("type") == "object":
            if "properties" not in repaired or not isinstance(
                repaired.get("properties"), dict
            ):
                repaired["properties"] = {} if "properties" not in repaired else repaired["properties"]
                if not isinstance(repaired.get("properties"), dict):
                    repaired["properties"] = {}

            required = repaired.get("required")
            if isinstance(required, list):
                props = repaired.get("properties") or {}
                valid = [r for r in required if isinstance(r, str) and r in props]
                if len(valid) != len(required):
                    if valid:
                        repaired["required"] = valid
                    else:
                        repaired.pop("required", None)

        return repaired

    normalized = _rewrite_local_refs(schema)
    normalized = _strip_nullable_union(normalized)
    normalized = _repair_object_shape(normalized)

    if not isinstance(normalized, dict):
        return {"type": "object", "properties": {}}
    if normalized.get("type") == "object" and "properties" not in normalized:
        normalized = {**normalized, "properties": {}}

    return normalized


def _sanitize_mcp_name_component(value: str) -> str:
    """返回对工具名称和前缀生成安全的 MCP 名称组件。

    保留将连字符转换为下划线的历史行为，并将 [A-Za-z0-9_] 之外的任何字符
    替换为 _，使生成的工具名称兼容提供商验证规则。
    """
    return re.sub(r"[^A-Za-z0-9_]", "_", str(value or ""))


def _convert_mcp_schema(server_name: str, mcp_tool) -> dict:
    """将 MCP 工具列表转换为项目注册表模式格式。

    Args:
        server_name: 用于前缀的逻辑服务器名称。
        mcp_tool:    带有 .name、.description、.inputSchema 的 MCP Tool 对象。

    返回:
        适合 registry.register(schema=...) 的 dict。
    """
    safe_tool_name = _sanitize_mcp_name_component(mcp_tool.name)
    safe_server_name = _sanitize_mcp_name_component(server_name)
    prefixed_name = f"mcp_{safe_server_name}_{safe_tool_name}"

    return {
        "name": prefixed_name,
        "description": mcp_tool.description or f"MCP 工具 {mcp_tool.name} 来自 {server_name}",
        "parameters": _normalize_mcp_input_schema(getattr(mcp_tool, "inputSchema", None)),
    }


def _normalize_name_filter(value: Any, label: str) -> set[str]:
    """将 include/exclude 配置规范化为工具名称集合。"""
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    logger.warning("MCP 配置 %s 必须是字符串或字符串列表；忽略 %r", label, value)
    return set()


def _parse_boolish(value: Any, default: bool = True) -> bool:
    """解析类似布尔值的配置值，安全回退。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    logger.warning("MCP 配置期望布尔值，得到 %r；使用默认值=%s", value, default)
    return default


def _register_server_tools(name: str, server: MCPServerTask, config: dict) -> List[str]:
    """将已连接服务器的工具注册到注册表。

    处理 include/exclude 过滤。用于初始发现和动态刷新 (list_changed)。

    返回:
        已注册前缀工具名称列表。
    """
    from tools.registry import registry

    registered_names: List[str] = []
    toolset_name = f"mcp-{name}"

    tools_filter = config.get("tools") or {}
    include_set = _normalize_name_filter(tools_filter.get("include"), f"mcp_servers.{name}.tools.include")
    exclude_set = _normalize_name_filter(tools_filter.get("exclude"), f"mcp_servers.{name}.tools.exclude")

    def _should_register(tool_name: str) -> bool:
        if include_set:
            return tool_name in include_set
        if exclude_set:
            return tool_name not in exclude_set
        return True

    for mcp_tool in server._tools:
        if not _should_register(mcp_tool.name):
            logger.debug("MCP 服务器 '%s': 跳过工具 '%s'（配置过滤）", name, mcp_tool.name)
            continue

        _scan_mcp_description(name, mcp_tool.name, mcp_tool.description or "")

        schema = _convert_mcp_schema(name, mcp_tool)
        tool_name_prefixed = schema["name"]

        # 防止与内置（非 MCP）工具冲突
        existing = registry.get(tool_name_prefixed)
        if existing and not existing.toolset.startswith("mcp-"):
            logger.warning(
                "MCP 服务器 '%s': 工具 '%s' (→ '%s') 与内置工具冲突 "
                "(toolset '%s') — 跳过以保护内置工具",
                name, mcp_tool.name, tool_name_prefixed, existing.toolset,
            )
            continue

        registry.register(
            name=tool_name_prefixed,
            toolset=toolset_name,
            schema=schema["parameters"],
            handler=_make_tool_handler(name, mcp_tool.name, server.tool_timeout),
            check_fn=_make_check_fn(name),
            is_async=False,
            description=schema["description"],
        )
        registered_names.append(tool_name_prefixed)

    if registered_names:
        # 注册 toolset 别名
        pass  # 项目注册表暂不支持别名，但工具已按 toolset 分组

    return registered_names


async def _discover_and_register_server(name: str, config: dict) -> List[str]:
    """连接到单个 MCP 服务器，发现工具并注册。

    返回已注册工具名称列表。
    """
    connect_timeout = config.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT)
    server = await asyncio.wait_for(
        _connect_server(name, config),
        timeout=connect_timeout,
    )
    with _lock:
        _servers[name] = server

    registered_names = _register_server_tools(name, server, config)
    server._registered_tool_names = list(registered_names)

    transport_type = "HTTP" if "url" in config else "stdio"
    logger.info(
        "MCP 服务器 '%s' (%s): 注册了 %d 个工具: %s",
        name, transport_type, len(registered_names),
        ", ".join(registered_names),
    )
    return registered_names


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def register_mcp_servers(servers: Dict[str, dict]) -> List[str]:
    """连接显式 MCP 服务器并注册其工具。

    对已连接的服务器名称幂等。enabled: false 的服务器被跳过但不断开现有会话。

    Args:
        servers: {server_name: server_config} 映射。

    返回:
        所有当前已注册 MCP 工具名称列表。
    """
    if not _MCP_AVAILABLE:
        logger.debug("MCP SDK 不可用 — 跳过显式 MCP 注册")
        return []

    if not servers:
        logger.debug("未提供显式 MCP 服务器")
        return []

    with _lock:
        new_servers = {
            k: v
            for k, v in servers.items()
            if k not in _servers and _parse_boolish(v.get("enabled", True), default=True)
        }

    if not new_servers:
        return _existing_tool_names()

    _ensure_mcp_loop()

    async def _discover_one(name: str, cfg: dict) -> List[str]:
        return await _discover_and_register_server(name, cfg)

    async def _discover_all():
        server_names = list(new_servers.keys())
        results = await asyncio.gather(
            *(_discover_one(name, cfg) for name, cfg in new_servers.items()),
            return_exceptions=True,
        )
        for name, result in zip(server_names, results):
            if isinstance(result, Exception):
                command = new_servers.get(name, {}).get("command")
                logger.warning(
                    "连接到 MCP 服务器 '%s'%s 失败: %s",
                    name,
                    f" (command={command})" if command else "",
                    _format_connect_error(result),
                )

    _run_on_mcp_loop(_discover_all(), timeout=120)

    with _lock:
        connected = [n for n in new_servers if n in _servers]
        new_tool_count = sum(
            len(getattr(_servers[n], "_registered_tool_names", []))
            for n in connected
        )
    failed = len(new_servers) - len(connected)
    if new_tool_count or failed:
        summary = f"MCP: 从 {len(connected)} 个服务器注册了 {new_tool_count} 个工具"
        if failed:
            summary += f" ({failed} 个失败)"
        logger.info(summary)

    return _existing_tool_names()


def discover_mcp_tools() -> List[str]:
    """入口点: 加载配置，连接 MCP 服务器，注册工具。

    在 discover_builtin_tools() 之后从 model_tools 调用。
    即使未安装 mcp 包也可安全调用（返回空列表）。

    对已连接服务器幂等。如果某些服务器在之前的调用中失败，只重试缺失的。

    返回:
        所有已注册 MCP 工具名称列表。
    """
    if not _MCP_AVAILABLE:
        logger.debug("MCP SDK 不可用 — 跳过 MCP 工具发现")
        return []

    servers = _load_mcp_config()
    if not servers:
        logger.debug("未配置 MCP 服务器")
        return []

    with _lock:
        new_server_names = [
            name
            for name, cfg in servers.items()
            if name not in _servers and _parse_boolish(cfg.get("enabled", True), default=True)
        ]

    tool_names = register_mcp_servers(servers)
    if not new_server_names:
        return tool_names

    with _lock:
        connected_server_names = [name for name in new_server_names if name in _servers]
        new_tool_count = sum(
            len(getattr(_servers[name], "_registered_tool_names", []))
            for name in connected_server_names
        )

    failed_count = len(new_server_names) - len(connected_server_names)
    if new_tool_count or failed_count:
        summary = f"  MCP: {new_tool_count} 个工具来自 {len(connected_server_names)} 个服务器"
        if failed_count:
            summary += f" ({failed_count} 个失败)"
        logger.info(summary)

    return tool_names


def get_mcp_status() -> List[dict]:
    """返回所有已配置 MCP 服务器的状态用于横幅显示。

    返回带有 name、transport、tools、connected 键的 dict 列表。
    包括成功连接的服务器和配置但失败的。
    """
    result: List[dict] = []

    configured = _load_mcp_config()
    if not configured:
        return result

    with _lock:
        active_servers = dict(_servers)

    for name, cfg in configured.items():
        transport = "http" if "url" in cfg else "stdio"
        server = active_servers.get(name)
        if server and server.session is not None:
            result.append({
                "name": name,
                "transport": transport,
                "tools": len(server._registered_tool_names) if hasattr(server, "_registered_tool_names") else len(server._tools),
                "connected": True,
            })
        else:
            result.append({
                "name": name,
                "transport": transport,
                "tools": 0,
                "connected": False,
            })

    return result


def shutdown_mcp_servers():
    """关闭所有 MCP 服务器连接并停止后台循环。

    每个服务器 Task 被信号通知退出其 async with 块，
    以便 anyio cancel-scope 清理在打开它的同一个 Task 中发生。
    所有服务器通过 asyncio.gather 并行关闭。
    """
    with _lock:
        servers_snapshot = list(_servers.values())

    if not servers_snapshot:
        _stop_mcp_loop()
        return

    async def _shutdown():
        results = await asyncio.gather(
            *(server.shutdown() for server in servers_snapshot),
            return_exceptions=True,
        )
        for server, result in zip(servers_snapshot, results):
            if isinstance(result, Exception):
                logger.debug(
                    "关闭 MCP 服务器 '%s' 出错: %s", server.name, result,
                )
        with _lock:
            _servers.clear()

    with _lock:
        loop = _mcp_loop
    if loop is not None and loop.is_running():
        try:
            future = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
            future.result(timeout=15)
        except Exception as exc:
            logger.debug("MCP 关闭期间出错: %s", exc)

    _stop_mcp_loop()


def _stop_mcp_loop():
    """停止后台事件循环并加入其线程。"""
    global _mcp_loop, _mcp_thread
    with _lock:
        loop = _mcp_loop
        thread = _mcp_thread
        _mcp_loop = None
        _mcp_thread = None
    if loop is not None:
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
        try:
            loop.close()
        except Exception:
            pass


def _existing_tool_names() -> List[str]:
    """返回所有当前连接服务器的工具名称。"""
    names: List[str] = []
    for _sname, server in _servers.items():
        if hasattr(server, "_registered_tool_names"):
            names.extend(server._registered_tool_names)
            continue
        for mcp_tool in server._tools:
            schema = _convert_mcp_schema(server.name, mcp_tool)
            names.append(schema["name"])
    return names
