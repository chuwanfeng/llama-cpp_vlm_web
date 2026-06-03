"""
代码执行工具 — 安全沙箱执行代码

移植自 hermes-agent/tools/code_execution_tool.py，适配 Web 场景。

核心功能：
    1. execute_python(code, timeout) — 执行 Python 代码，返回 stdout/stderr
    2. execute_javascript(code, timeout) — 执行 JavaScript 代码（Node.js）
    3. execute_shell(command, timeout) — 执行 Shell 命令（Windows: PowerShell）
    4. eval_expression(expression, language) — 安全求值简单表达式（纯 AST，无 subprocess）

安全设计（重中之重）：
    - 禁止使用危险内置函数：exec/eval/open/__import__/os.system 等
    - 禁止导入危险模块：os/sys/subprocess/builtins 等
    - 超时强制终止（subprocess timeout）
    - 输出大小限制（默认 100KB）
    - 在临时目录执行，不影响项目文件
    - 不允许文件写操作（除非显式允许）

依赖：
    - Python 标准库（无需额外安装）
    - Node.js（可选，仅 execute_javascript 需要）
    - PowerShell（Windows 默认存在）

设计参考：
    - hermes-agent 使用 Docker 隔离容器（完整沙箱）
    - 本项目使用 subprocess + 超时 + AST 检查（轻量级沙箱）
    - 生产环境建议升级到 Docker 隔离
"""

import ast
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any, Dict, Optional

from tools.registry import get_registry
# 内联截断（tool_output_limits 未导出 truncate_output）

logger = logging.getLogger(__name__)

# ── 安全常量 ─────────────────────────────────────────────────────────────

DANGEROUS_BUILTINS = {
    "exec", "eval", "open", "__import__",
    "compile", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "dir",
}

DANGEROUS_MODULES = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "builtins", "importlib", "ctypes", "multiprocessing",
    "threading", "socket", "requests", "urllib", "http",
}

MAX_OUTPUT_SIZE = 100 * 1024  # 100KB
DEFAULT_TIMEOUT = 30  # 秒
TEMP_DIR = tempfile.gettempdir()


def _truncate_output(text: str, max_bytes: int) -> str:
    """截断输出到指定字节数。"""
    if not text:
        return text
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes]
    return truncated.decode("utf-8", errors="replace") + "\n...(已截断)"

# ── 安全检查 ─────────────────────────────────────────────────────────────

def _check_code_safety(source_code: str, language: str = "python") -> Dict[str, Any]:
    """使用 AST 检查代码是否包含危险操作。

    参数：
        source_code: 源代码字符串
        language: 语言（"python" / "javascript" / "shell"）

    返回：
        {"safe": bool, "error": str} 字典
    """
    if language == "python":
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return {"safe": False, "error": f"Python 语法错误：{e}"}

        # 遍历 AST，检查危险节点
        for node in ast.walk(tree):
            # 检查危险函数调用
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in DANGEROUS_BUILTINS:
                        return {
                            "safe": False,
                            "error": f"禁止使用内置函数：{node.func.id}",
                        }
                # 检查 attribute 调用（如 os.system）
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id in DANGEROUS_MODULES:
                            return {
                                "safe": False,
                                "error": f"禁止访问危险模块：{node.func.value.id}.{node.func.attr}",
                            }

            # 检查危险导入
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    if module_name in DANGEROUS_MODULES:
                        return {
                            "safe": False,
                            "error": f"禁止导入危险模块：{module_name}",
                        }

        return {"safe": True, "error": ""}

    elif language == "javascript":
        # JavaScript 安全检查（简单字符串匹配，不够完善）
        dangerous_patterns = [
            r"require\s*\(\s*['\"]os['\"]",  # require('os')
            r"process\.exit",
            r"fs\.writeFileSync",
            r"child_process",
            r"eval\s*\(",
        ]
        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, source_code):
                return {
                    "safe": False,
                    "error": f"JavaScript 代码包含可疑模式：{pattern}",
                }
        return {"safe": True, "error": ""}

    elif language == "shell":
        # Shell 命令安全检查
        dangerous_commands = [
            "rm ", "rmdir", "del ", "format ", "mkfs",
            "dd ", "shutdown", "reboot", "kill ", "pkill",
            "> ", ">> ", "curl ", "wget ",
        ]
        for cmd in dangerous_commands:
            if cmd in source_code.lower():
                return {
                    "safe": False,
                    "error": f"Shell 命令包含危险操作：{cmd.strip()}",
                }
        return {"safe": True, "error": ""}

    else:
        return {"safe": True, "error": ""}


def _execute_via_subprocess(
    code: str,
    language: str,
    timeout: int,
    work_dir: str = TEMP_DIR,
) -> Dict[str, Any]:
    """通过子进程执行代码，带超时和输出限制。

    参数：
        code: 源代码
        language: 语言（"python" / "javascript" / "shell"）
        timeout: 超时时间（秒）
        work_dir: 工作目录

    返回：
        {
            "stdout": str,
            "stderr": str,
            "exit_code": int,
            "timeout": bool,
            "error": str,
        }
    """
    # 写入临时文件
    ext_map = {"python": ".py", "javascript": ".js", "shell": ".ps1"}
    ext = ext_map.get(language, ".txt")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=ext, dir=work_dir, delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        # 构建命令
        if language == "python":
            cmd = [sys.executable, temp_path]
        elif language == "javascript":
            cmd = ["node", temp_path]
        elif language == "shell":
            cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", temp_path]
        else:
            return {
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "timeout": False,
                "error": f"不支持的语言：{language}",
            }

        # 执行
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir,
                encoding="utf-8",
                errors="replace",
            )
            elapsed = time.time() - start_time

            return {
                "stdout": _truncate_output(result.stdout, MAX_OUTPUT_SIZE),
                "stderr": _truncate_output(result.stderr, MAX_OUTPUT_SIZE),
                "exit_code": result.returncode,
                "timeout": False,
                "error": "",
                "elapsed": round(elapsed, 3),
            }

        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"执行超时（{timeout} 秒）",
                "exit_code": -1,
                "timeout": True,
                "error": f"执行超时（{timeout} 秒）",
            }

    except Exception as e:
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "timeout": False,
            "error": f"执行失败：{type(e).__name__}: {str(e)}",
        }

    finally:
        # 清理临时文件
        try:
            os.unlink(temp_path)
        except OSError:
            pass


# ── 工具实现 ─────────────────────────────────────────────────────────────

def code_exec_python(code: str, timeout: int = DEFAULT_TIMEOUT, allow_file_write: bool = False) -> str:
    """执行 Python 代码（安全沙箱）。

    参数（JSON 字符串）：
        code: Python 代码字符串
        timeout: 超时时间（秒，默认 30）
        allow_file_write: 是否允许文件写操作（默认 false）

    返回：
        JSON 字符串，包含：
            - stdout: 标准输出
            - stderr: 标准错误
            - exit_code: 退出码
            - timeout: 是否超时
            - error: 错误信息
            - output: 合并 stdout + stderr（用于快速查看）
    """


    if not code:
        return json.dumps({"error": "code 参数必填"})

    # 安全检查
    safety = _check_code_safety(code, "python")
    if not safety["safe"]:
        return json.dumps({
            "error": f"代码安全检查失败：{safety['error']}",
            "stdout": "",
            "stderr": f"SECURITY: {safety['error']}",
            "exit_code": -1,
        }, ensure_ascii=False)

    # 如果不允许文件写，额外检查文件操作
    if not allow_file_write:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in {"open", "write", "writelines"}:
                        return json.dumps({
                            "error": "文件写操作被禁止（设置 allow_file_write=true 可允许）",
                            "stdout": "",
                            "stderr": "SECURITY: File write not allowed",
                            "exit_code": -1,
                        }, ensure_ascii=False)

    # 执行
    result = _execute_via_subprocess(code, "python", timeout)

    return json.dumps({
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "exit_code": result["exit_code"],
        "timeout": result["timeout"],
        "output": result["stdout"] + "\n" + result["stderr"] if result["stdout"] or result["stderr"] else "",
        "elapsed": result.get("elapsed", 0),
    }, ensure_ascii=False)


def code_exec_javascript(code: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """执行 JavaScript 代码（需要 Node.js）。"""


    if not code:
        return json.dumps({"error": "code 参数必填"})

    # 检查 Node.js 是否可用
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return json.dumps({
            "error": "Node.js 未安装或不在 PATH 中",
            "note": "安装 Node.js: https://nodejs.org/",
        }, ensure_ascii=False)

    # 安全检查
    safety = _check_code_safety(code, "javascript")
    if not safety["safe"]:
        return json.dumps({
            "error": f"代码安全检查失败：{safety['error']}",
        }, ensure_ascii=False)

    # 执行
    result = _execute_via_subprocess(code, "javascript", timeout)

    return json.dumps({
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "exit_code": result["exit_code"],
        "timeout": result["timeout"],
        "output": result["stdout"] + "\n" + result["stderr"] if result["stdout"] or result["stderr"] else "",
    }, ensure_ascii=False)


def code_exec_shell(command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """执行 Shell 命令（Windows: PowerShell）。"""


    if not command:
        return json.dumps({"error": "command 参数必填"})

    # 安全检查
    safety = _check_code_safety(command, "shell")
    if not safety["safe"]:
        return json.dumps({
            "error": f"命令安全检查失败：{safety['error']}",
        }, ensure_ascii=False)

    # 执行
    result = _execute_via_subprocess(command, "shell", timeout)

    return json.dumps({
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "exit_code": result["exit_code"],
        "timeout": result["timeout"],
        "output": result["stdout"] + "\n" + result["stderr"] if result["stdout"] or result["stderr"] else "",
    }, ensure_ascii=False)


def code_eval(expression: str, language: str = "python") -> str:
    """安全求值简单表达式（无 subprocess，纯 Python AST）。

    支持：
        - 数学运算：1 + 2 * 3
        - 列表/字典操作：[1, 2, 3].append(4)  ❌ 不允许
        - 但允许：sum([1, 2, 3]), len([1, 2, 3])

    不支持：
        - 函数定义
        - 循环
        - 导入
        - 赋值语句

    参数（JSON 字符串）：
        expression: 表达式字符串
        language: 语言（默认 "python"）

    返回：
        JSON 字符串，包含 result 或 error
    """

    if not expression:
        return json.dumps({"error": "expression 参数必填"})

    if language != "python":
        return json.dumps({"error": f"eval 仅支持 Python，不支持 {language}"})

    # 安全检查
    safety = _check_code_safety(expression, "python")
    if not safety["safe"]:
        return json.dumps({"error": f"表达式安全检查失败：{safety['error']}"}, ensure_ascii=False)

    # 仅允许表达式（不允许赋值、函数定义、导入）
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        return json.dumps({"error": f"表达式语法错误：{e}"}, ensure_ascii=False)

    # 执行（限制全局命名空间）
    safe_globals = {
        "__builtins__": {
            "len": len, "sum": sum, "min": min, "max": max,
            "abs": abs, "round": round, "pow": pow,
            "int": int, "float": float, "str": str, "bool": bool, "list": list, "dict": dict, "tuple": tuple,
            "range": range, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
            "sorted": sorted, "reversed": reversed,
        },
    }

    try:
        result = eval(compile(tree, "<expression>", "eval"), safe_globals, {})
        return json.dumps({"result": repr(result)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"求值失败：{type(e).__name__}: {str(e)}"}, ensure_ascii=False)


# ── OpenAI 工具 Schema ──────────────────────────────────────────────────

CODE_EXEC_PYTHON_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Python 代码字符串"},
        "timeout": {"type": "number", "description": "超时时间（秒，默认 30）"},
        "allow_file_write": {
            "type": "boolean",
            "description": "是否允许文件写操作（默认 false）",
        },
    },
    "required": ["code"],
}

CODE_EXEC_JAVASCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "JavaScript 代码字符串"},
        "timeout": {"type": "number", "description": "超时时间（秒，默认 30）"},
    },
    "required": ["code"],
}

CODE_EXEC_SHELL_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Shell 命令字符串"},
        "timeout": {"type": "number", "description": "超时时间（秒，默认 30）"},
    },
    "required": ["command"],
}

CODE_EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "expression": {"type": "string", "description": "表达式字符串（例如 '1 + 2 * 3'）"},
        "language": {"type": "string", "description": "语言（默认 'python'）"},
    },
    "required": ["expression"],
}


# ── 注册到工具系统 ──────────────────────────────────────────────────────

registry = get_registry()

registry.register(
    name="code_exec_python",
    schema=CODE_EXEC_PYTHON_SCHEMA,
    handler=code_exec_python,
)

registry.register(
    name="code_exec_javascript",
    schema=CODE_EXEC_JAVASCRIPT_SCHEMA,
    handler=code_exec_javascript,
)

registry.register(
    name="code_exec_shell",
    schema=CODE_EXEC_SHELL_SCHEMA,
    handler=code_exec_shell,
)

registry.register(
    name="code_eval",
    schema=CODE_EVAL_SCHEMA,
    handler=code_eval,
)

logger.info("代码执行工具已注册：code_exec_python, code_exec_javascript, code_exec_shell, code_eval")
