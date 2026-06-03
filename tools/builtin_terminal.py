"""终端命令执行工具 — 集成审批流的安全 Shell 操作。

移植自 hermes-agent/tools/terminal_tool.py，集成 approval.py 审批流。

核心设计：
    - 审批流集成：危险命令通过 approval.py 进行检测和审批
    - 白名单机制：SAFE_COMMANDS 中的命令直接允许执行
    - 超时保护：COMMAND_TIMEOUT（默认 30 秒）
    - 工作目录锁定：只能在项目根目录执行

安全层级：
    1. Hardline 阻止：rm -rf /、mkfs、dd 到块设备等灾难性命令无条件阻止
    2. 危险模式检测：47 条规则检测危险命令，需用户审批
    3. 白名单检查：非危险命令检查是否在 SAFE_COMMANDS 中
"""

import logging
import os
import subprocess

from tools.registry import get_registry
from tools.approval import check_all_command_guards

logger = logging.getLogger(__name__)

# Commands that are always safe (read-only, info-gathering)
SAFE_COMMANDS = {
    "dir", "ls", "pwd", "echo", "cat", "type", "head", "tail",
    "wc", "find", "grep", "findstr", "where", "which", "python",
    "python3", "node", "npm", "pip", "git", "curl", "wget",
    "systeminfo", "ver", "date", "time", "hostname", "whoami",
    "ipconfig", "netstat", "tasklist", "tree","scoop", "ollama",
    "Get-Process", "Get-Service", "Get-ChildItem", "Get-Location",
    "Get-Content", "Select-String", "Get-Date", "Get-ComputerInfo",
}

# Timeout for command execution (seconds)
COMMAND_TIMEOUT = 30

# Working directory
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_terminal(command: str, background: bool = False, env_type: str = "local") -> str:
    """执行终端命令并返回输出。

    集成审批流：危险命令需通过 approval.py 检测和审批。
    仅允许只读、安全的命令。破坏性命令会被拦截或要求审批。

    参数:
        command: 要执行的 Shell 命令字符串
        background: 是否在后台运行（返回进程 session_id）
        env_type: 环境类型（local/container/ssh）

    返回:
        命令输出（stdout + stderr 合并）或后台进程 session_id
    """
    # Step 1: Approval flow — check dangerous commands
    approval_result = check_all_command_guards(command, env_type=env_type)
    if not approval_result.get("approved", True):
        return approval_result.get("message", "BLOCKED: Command was not approved.")

    # Step 2: Background mode — spawn via process registry
    if background:
        from tools.process_registry import process_registry
        session = process_registry.spawn(command, cwd=WORK_DIR)
        return (
            f"Background process started.\n"
            f"Session ID: {session.id}\n"
            f"PID: {session.pid}\n"
            f"Command: {session.command}\n"
            f"Use process(action='poll', session_id='{session.id}') to check status."
        )

    # Step 3: Extract the first word (command name)
    cmd_parts = command.strip().split()
    if not cmd_parts:
        return "Error: Empty command"

    cmd_name = os.path.basename(cmd_parts[0]).lower()

    # PowerShell cmdlet names
    if cmd_name.startswith("get-") or cmd_name.startswith("select-"):
        pass  # Allow Get-* and Select-* cmdlets
    elif cmd_name not in SAFE_COMMANDS:
        return (
            f"Error: Command '{cmd_name}' is not in the safe command list. "
            f"Allowed commands: {', '.join(sorted(SAFE_COMMANDS))}"
        )

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORK_DIR,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        output = result.stdout or ""
        if result.stderr:
            output += "\n[stderr]\n" + (result.stderr or "")

        if not output.strip():
            output = f"(Command completed with exit code {result.returncode}, no output)"

        return output.strip()

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {COMMAND_TIMEOUT}s"
    except Exception as e:
        return f"Error executing command: {e}"


# ── Register ─────────────────────────────────────────────────────────────────

registry = get_registry()
registry.register(
    name="run_terminal",
    description="Execute a terminal command on Windows. Use dir (not ls), type (not cat), findstr (not grep) for listing/files. Use to check system info, list files, Git clone status, or check Python/Node versions. Set background=true to run long commands in background (returns session_id for process tool).",
    schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute on Windows. Use dir, type, findstr instead of ls, cat, grep. Single command per call.",
            },
            "background": {
                "type": "boolean",
                "description": "Run in background mode. Returns a session_id that can be used with the process tool to poll/wait/kill.",
                "default": False,
            },
            "env_type": {
                "type": "string",
                "description": "Environment type: local (default), container, ssh",
                "default": "local",
            },
        },
        "required": ["command"],
    },
    handler=run_terminal,
    toolset="terminal",
)