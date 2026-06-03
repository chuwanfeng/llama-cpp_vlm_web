"""危险命令审批系统 — 检测、提示和会话级状态管理。

本模块是危险命令系统的单一可信来源：
- 模式检测（DANGEROUS_PATTERNS、detect_dangerous_command）
- 会话级审批状态（线程安全，按 session_key 索引）
- 审批提示（CLI 交互式 + Gateway 异步）
- Smart Approval：通过辅助 LLM 自动审批低风险命令
- 永久白名单持久化（config.yaml）

移植自 hermes-agent/tools/approval.py，适配 llama-cpp_vlm_web 架构。
"""

import logging
import os
import re
import sys
import threading
import time
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

# =========================================================================
# 会话身份管理
# =========================================================================

# 线程级/任务级会话身份
# Gateway 在 executor 线程中并发运行 Agent 轮次，因此读取进程级环境变量
# 来识别会话是有竞争风险的。保留环境变量回退以兼容旧版单线程调用者，
# 但当上下文本地值已设置时优先使用它。
_approval_session_key: threading.local = threading.local()
_approval_session_key.value = ""


def set_current_session_key(session_key: str) -> None:
    """将当前活动的审批会话键绑定到当前上下文。"""
    _approval_session_key.value = session_key or ""


def get_current_session_key(default: str = "default") -> str:
    """返回活动会话键，优先使用上下文本地状态。

    解析顺序：
    1. approval 专用的线程本地变量（由 gateway 在 agent.run 前设置）
    2. 环境变量回退（CLI、cron、测试）
    """
    session_key = getattr(_approval_session_key, "value", "")
    if session_key:
        return session_key
    return os.getenv("LLAMA_SESSION_KEY", default)


# =========================================================================
# 敏感写入目标（触发审批）
# =========================================================================

# 即使通过 shell 展开如 $HOME 或 $LLAMA_HOME 引用，也应触发审批的敏感写入目标
_SSH_SENSITIVE_PATH = r'(?:~|\$home|\$\{home\})/\.ssh(?:/|$)'
_LLAMA_ENV_PATH = (
    r'(?:~/\.llama/|'
    r'(?:\$home|\$\{home\})/\.llama/|'
    r'(?:\$llama_home|\$\{llama_home\})/)'
    r'\.env\b'
)
_PROJECT_ENV_PATH = r'(?:(?:/|\.{1,2}/)?(?:[^\s/"\'`]+/)*\.env(?:\.[^/\s"\'`]+)*)'
_PROJECT_CONFIG_PATH = r'(?:(?:/|\.{1,2}/)?(?:[^\s/"\'`]+/)*config\.yaml)'
_SHELL_RC_FILES = (
    r'(?:~|\$home|\$\{home\})/\.'
    r'(?:bashrc|zshrc|profile|bash_profile|zprofile)\b'
)
_CREDENTIAL_FILES = (
    r'(?:~|\$home|\$\{home\})/\.'
    r'(?:netrc|pgpass|npmrc|pypirc)\b'
)
_SENSITIVE_WRITE_TARGET = (
    r'(?:/etc/|/dev/sd|'
    rf'{_SSH_SENSITIVE_PATH}|'
    rf'{_LLAMA_ENV_PATH}|'
    rf'{_SHELL_RC_FILES}|'
    rf'{_CREDENTIAL_FILES})'
)
_PROJECT_SENSITIVE_WRITE_TARGET = rf'(?:{_PROJECT_ENV_PATH}|{_PROJECT_CONFIG_PATH})'
_COMMAND_TAIL = r'(?:\s*(?:&&|\|\||;).*)?$'

# =========================================================================
# Hardline（无条件）阻止列表
# =========================================================================
#
# 这些命令极具灾难性，无论 --yolo、/yolo、approvals.mode=off 还是 cron
# 批准模式都不应通过 Agent 执行。这是 yolo 之下的底线：选择 yolo 是用户
# 信任 Agent 处理其文件和服务，而不是信任它擦除磁盘或关闭机器。
#
# Hardline 仅适用于可能实际损坏宿主机的环境（local、ssh、container-host cron）。
# 容器化后端（docker、singularity、modal、daytona）已经绕过了危险命令层，
# 因为它们无法触及宿主机，所以我们保持该行为不变。
#
# 列表刻意保持极小 —— 只有那些没有恢复路径的操作：
# 从 / 开始的文件系统销毁、原始块设备覆盖、内核关机/重启、以及使主机宕机的
# 拒绝服务命令。可恢复但代价高昂的操作（git reset --hard、rm -rf /tmp/x、
# chmod -R 777、curl|sh）保留在 DANGEROUS_PATTERNS 中，yolo 可以让它们通过
# —— 这就是 yolo 的用途。

# 匹配命令*开头*的正则片段（即 shell 开始解析新命令的位置）。
# 用于 shutdown/reboot 模式，使其不会在 "echo reboot" 或 "grep 'shutdown' log"
# 上误触发。
# 匹配：字符串开头、命令分隔符后（; && || | 换行）、子 shell 开启符后（ $(` ），
# 可选地消费前导包装命令（sudo、env VAR=VAL、exec、nohup、setsid）。
_CMDPOS = (
    r'(?:^|[;|\n`]|\$\()'         # 起始位置
    r'\s*'                          # 可选空白
    r'(?:sudo\s+(?:-[^\s]+\s+)*)?'  # 可选 sudo 及标志
    r'(?:env\s+(?:\w+=\S*\s+)*)?'   # 可选 env 及 VAR=VAL 对
    r'(?:(?:exec|nohup|setsid|time)\s+)*'  # 可选包装命令
    r'\s*'
)

HARDLINE_PATTERNS = [
    # rm 递归 targeting 根文件系统或受保护根目录
    (r'\brm\s+(-[^\s]*\s+)*(/|/\*|/ \*)(\s|$)', "recursive delete of root filesystem"),
    (r'\brm\s+(-[^\s]*\s+)*(/home|/home/\*|/root|/root/\*|/etc|/etc/\*|/usr|/usr/\*|/var|/var/\*|/bin|/bin/\*|/sbin|/sbin/\*|/boot|/boot/\*|/lib|/lib/\*)(\s|$)', "recursive delete of system directory"),
    (r'\brm\s+(-[^\s]*\s+)*(~|\$HOME)(/?|/\*)?(\s|$)', "recursive delete of home directory"),
    # 文件系统格式化
    (r'\bmkfs(\.[a-z0-9]+)?\b', "format filesystem (mkfs)"),
    # 原始块设备覆盖（dd + 重定向）
    (r'\bdd\b[^\n]*\bof=/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*', "dd to raw block device"),
    (r'>\s*/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*\b', "redirect to raw block device"),
    # Fork bomb（经典 shell 形式）
    (r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:', "fork bomb"),
    # 杀死系统上所有进程
    (r'\bkill\s+(-[^\s]+\s+)*-1\b', "kill all processes"),
    # 系统关机 / 重启 —— 锚定到命令位置（行首、命令分隔符后、或 sudo/env 包装后）
    # 避免在 "echo reboot" 或 "grep 'shutdown' logs" 上误触发。
    (_CMDPOS + r'(shutdown|reboot|halt|poweroff)\b', "system shutdown/reboot"),
    (_CMDPOS + r'init\s+[06]\b', "init 0/6 (shutdown/reboot)"),
    (_CMDPOS + r'systemctl\s+(poweroff|reboot|halt|kexec)\b', "systemctl poweroff/reboot"),
    (_CMDPOS + r'telinit\s+[06]\b', "telinit 0/6 (shutdown/reboot)"),
]

_RE_FLAGS = re.IGNORECASE | re.DOTALL
HARDLINE_PATTERNS_COMPILED = [
    (re.compile(pattern, _RE_FLAGS), description)
    for pattern, description in HARDLINE_PATTERNS
]


# =========================================================================
# 危险命令模式
# =========================================================================

DANGEROUS_PATTERNS = [
    (r'\brm\s+(-[^\s]*\s+)*/', "delete in root path"),
    (r'\brm\s+-[^\s]*r', "recursive delete"),
    (r'\brm\s+--recursive\b', "recursive delete (long flag)"),
    (r'\bchmod\s+(-[^\s]*\s+)*(777|666|o\+[rwx]*w|a\+[rwx]*w)\b', "world/other-writable permissions"),
    (r'\bchmod\s+--recursive\b.*(777|666|o\+[rwx]*w|a\+[rwx]*w)', "recursive world/other-writable (long flag)"),
    (r'\bchown\s+(-[^\s]*)?R\s+root', "recursive chown to root"),
    (r'\bchown\s+--recursive\b.*root', "recursive chown to root (long flag)"),
    (r'\bmkfs\b', "format filesystem"),
    (r'\bdd\s+.*if=', "disk copy"),
    (r'>\s*/dev/sd', "write to block device"),
    (r'\bDROP\s+(TABLE|DATABASE)\b', "SQL DROP"),
    (r'\bDELETE\s+FROM\b(?!.*\bWHERE\b)', "SQL DELETE without WHERE"),
    (r'\bTRUNCATE\s+(TABLE)?\s*\w', "SQL TRUNCATE"),
    (r'>\s*/etc/', "overwrite system config"),
    (r'\bsystemctl\s+(-[^\s]+\s+)*(stop|restart|disable|mask)\b', "stop/restart system service"),
    (r'\bkill\s+-9\s+-1\b', "kill all processes"),
    (r'\bpkill\s+-9\b', "force kill processes"),
    (r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:', "fork bomb"),
    # 通过 -c 或组合标志如 -lc、-ic 等调用 shell
    (r'\b(bash|sh|zsh|ksh)\s+-[^\s]*c(\s+|$)', "shell command via -c/-lc flag"),
    (r'\b(python[23]?|perl|ruby|node)\s+-[ec]\s+', "script execution via -e/-c flag"),
    (r'\b(curl|wget)\b.*\|\s*(ba)?sh\b', "pipe remote content to shell"),
    (r'\b(bash|sh|zsh|ksh)\s+<\s*<?\s*\(\s*(curl|wget)\b', "execute remote script via process substitution"),
    (rf'\btee\b.*["\']?{_SENSITIVE_WRITE_TARGET}', "overwrite system file via tee"),
    (rf'>>?\s*["\']?{_SENSITIVE_WRITE_TARGET}', "overwrite system file via redirection"),
    (rf'\btee\b.*["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_COMMAND_TAIL}', "overwrite project env/config via tee"),
    (rf'>>?\s*["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_COMMAND_TAIL}', "overwrite project env/config via redirection"),
    (r'\bxargs\s+.*\brm\b', "xargs with rm"),
    (r'\bfind\b.*-exec\s+(/\S*/)?rm\b', "find -exec rm"),
    (r'\bfind\b.*-delete\b', "find -delete"),
    # 网关生命周期保护：防止 Agent 杀死自己的网关进程
    (r'\bllama\s+gateway\s+(stop|restart)\b', "stop/restart llama gateway (kills running agents)"),
    (r'\bllama\s+update\b', "llama update (restarts gateway, kills running agents)"),
    # 网关保护：绝不在 systemd 管理外启动网关
    (r'gateway\s+run\b.*(&\s*$|&\s*;|\bdisown\b|\bsetsid\b)', "start gateway outside systemd (use 'systemctl --user restart llama-gateway')"),
    (r'\bnohup\b.*gateway\s+run\b', "start gateway outside systemd (use 'systemctl --user restart llama-gateway')"),
    # 自终止保护：防止 Agent 杀死自己的进程
    (r'\b(pkill|killall)\b.*\b(llama|gateway|cli\.py)\b', "kill llama/gateway process (self-termination)"),
    # 通过 kill + 命令替换（pgrep/pidof）自终止
    (r'\bkill\b.*\$\(\s*pgrep\b', "kill process via pgrep expansion (self-termination)"),
    (r'\bkill\b.*`\s*pgrep\b', "kill process via backtick pgrep expansion (self-termination)"),
    # 复制/移动/编辑到敏感系统路径
    (r'\b(cp|mv|install)\b.*\s/etc/', "copy/move file into /etc/"),
    (rf'\b(cp|mv|install)\b.*\s["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_COMMAND_TAIL}', "overwrite project env/config file"),
    (r'\bsed\s+-[^\s]*i.*\s/etc/', "in-place edit of system config"),
    (r'\bsed\s+--in-place\b.*\s/etc/', "in-place edit of system config (long flag)"),
    # 通过 heredoc 执行脚本 —— 绕过上面的 -e/-c 标志模式
    (r'\b(python[23]?|perl|ruby|node)\s+<<', "script execution via heredoc"),
    # Git 破坏性操作，可能丢失未提交的工作或重写共享历史
    (r'\bgit\s+reset\s+--hard\b', "git reset --hard (destroys uncommitted changes)"),
    (r'\bgit\s+push\b.*--force\b', "git force push (rewrites remote history)"),
    (r'\bgit\s+push\b.*-f\b', "git force push short flag (rewrites remote history)"),
    (r'\bgit\s+clean\s+-[^\s]*f', "git clean with force (deletes untracked files)"),
    (r'\bgit\s+branch\s+-D\b', "git branch force delete"),
    # chmod +x 后执行脚本 —— 捕获两步模式：脚本先被设为可执行然后立即运行
    (r'\bchmod\s+\+x\b.*[;&|]+\s*\./', "chmod +x followed by immediate execution"),
]

DANGEROUS_PATTERNS_COMPILED = [
    (re.compile(pattern, _RE_FLAGS), description)
    for pattern, description in DANGEROUS_PATTERNS
]


def _legacy_pattern_key(pattern: str) -> str:
    """为向后兼容复现旧的正则派生审批键。"""
    return pattern.split(r'\b')[1] if r'\b' in pattern else pattern[:20]


_PATTERN_KEY_ALIASES: dict[str, set[str]] = {}
for _pattern, _description in DANGEROUS_PATTERNS:
    _legacy_key = _legacy_pattern_key(_pattern)
    _canonical_key = _description
    _PATTERN_KEY_ALIASES.setdefault(_canonical_key, set()).update({_canonical_key, _legacy_key})
    _PATTERN_KEY_ALIASES.setdefault(_legacy_key, set()).update({_legacy_key, _canonical_key})


def _approval_key_aliases(pattern_key: str) -> set[str]:
    """返回应匹配此模式的所有审批键。

    新审批使用人类可读的描述字符串，但旧的 command_allowlist 条目和
    会话审批可能仍包含历史正则派生键。
    """
    return _PATTERN_KEY_ALIASES.get(pattern_key, {pattern_key})


# =========================================================================
# 检测
# =========================================================================

def _normalize_command_for_detection(command: str) -> str:
    """在危险模式匹配前规范化命令字符串。

    剥离 ANSI 转义序列、空字节，并规范化 Unicode 全角字符，
    使混淆技术无法绕过基于模式的检测。
    """
    # 剥离 ANSI 转义序列（简单实现）
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    command = ansi_escape.sub('', command)
    # 剥离空字节
    command = command.replace('\x00', '')
    # 规范化 Unicode（全角拉丁、半角片假名等）
    command = unicodedata.normalize('NFKC', command)
    return command


def detect_hardline_command(command: str) -> tuple:
    """检查命令是否匹配无条件 hardline 阻止列表。

    返回:
        (is_hardline, description) 或 (False, None)
    """
    normalized = _normalize_command_for_detection(command).lower()
    for pattern_re, description in HARDLINE_PATTERNS_COMPILED:
        if pattern_re.search(normalized):
            return (True, description)
    return (False, None)


def _hardline_block_result(description: str) -> dict:
    """为 hardline 匹配构建标准阻止结果。"""
    return {
        "approved": False,
        "hardline": True,
        "message": (
            f"BLOCKED (hardline): {description}. "
            "This command is on the unconditional blocklist and cannot "
            "be executed via the agent — not even with --yolo, /yolo, "
            "approvals.mode=off, or cron approve mode. If you genuinely "
            "need to run it, run it yourself in a terminal outside the "
            "agent."
        ),
    }


def detect_dangerous_command(command: str) -> tuple:
    """检查命令是否匹配任何危险模式。

    返回:
        (is_dangerous, pattern_key, description) 或 (False, None, None)
    """
    command_lower = _normalize_command_for_detection(command).lower()
    for pattern_re, description in DANGEROUS_PATTERNS_COMPILED:
        if pattern_re.search(command_lower):
            pattern_key = description
            return (True, pattern_key, description)
    return (False, None, None)


# =========================================================================
# 会话级审批状态（线程安全）
# =========================================================================

_lock = threading.Lock()
_pending: dict[str, dict] = {}
_session_approved: dict[str, set] = {}
_session_yolo: set[str] = set()
_permanent_approved: set = set()

# =========================================================================
# 阻塞式网关审批（镜像 CLI 的同步 input() 流程）
# =========================================================================
# 每会话的待审批队列。多个线程（并行子代理、execute_code RPC 处理器）
# 可以并发阻塞 —— 每个线程获得自己的 threading.Event。
# /approve 解决最旧的，/approve all 一次性解决会话中所有待审批。


class _ApprovalEntry:
    """网关会话中的一个待审批危险命令。"""
    __slots__ = ("event", "data", "result")

    def __init__(self, data: dict):
        self.event = threading.Event()
        self.data = data          # command, description, pattern_keys, …
        self.result: Optional[str] = None  # "once"|"session"|"always"|"deny"


_gateway_queues: dict[str, list] = {}        # session_key → [_ApprovalEntry, …]
_gateway_notify_cbs: dict[str, object] = {}  # session_key → callable(approval_data)


def register_gateway_notify(session_key: str, cb) -> None:
    """注册每会话回调，用于向用户发送审批请求。

    回调签名为 ``cb(approval_data: dict) -> None``，其中
    *approval_data* 包含 ``command``、``description`` 和
    ``pattern_keys``。回调桥接 sync→async（在 Agent 线程中运行，
    必须在事件循环上调度实际发送）。
    """
    with _lock:
        _gateway_notify_cbs[session_key] = cb


def unregister_gateway_notify(session_key: str) -> None:
    """注销网关会话的审批回调。

    向该会话的所有阻塞线程发送信号，使它们不会永远挂起
    （例如当 Agent 运行完成或被中断时）。
    """
    with _lock:
        _gateway_notify_cbs.pop(session_key, None)
        entries = _gateway_queues.pop(session_key, [])
    for entry in entries:
        entry.event.set()


def resolve_gateway_approval(session_key: str, choice: str,
                             resolve_all: bool = False) -> int:
    """由网关的 /approve 或 /deny 处理程序调用，以解除等待的 Agent 线程阻塞。

    当 *resolve_all* 为 True 时，会话中的每个待审批都会一次性解决
    （``/approve all``）。否则仅解决最旧的一个（FIFO）。

    返回已解决的审批数量（0 表示没有待审批）。
    """
    with _lock:
        queue = _gateway_queues.get(session_key)
        if not queue:
            return 0
        if resolve_all:
            targets = list(queue)
            queue.clear()
        else:
            targets = [queue.pop(0)]
        if not queue:
            _gateway_queues.pop(session_key, None)

    for entry in targets:
        entry.result = choice
        entry.event.set()
    return len(targets)


def has_blocking_approval(session_key: str) -> bool:
    """检查会话是否有一个或多个阻塞式网关审批在等待。"""
    with _lock:
        return bool(_gateway_queues.get(session_key))


def submit_pending(session_key: str, approval: dict):
    """为会话存储一个待审批请求。"""
    with _lock:
        _pending[session_key] = approval


def approve_session(session_key: str, pattern_key: str):
    """仅为此会话批准一个模式。"""
    with _lock:
        _session_approved.setdefault(session_key, set()).add(pattern_key)


def enable_session_yolo(session_key: str) -> None:
    """为单个会话键启用 YOLO 绕过。"""
    if not session_key:
        return
    with _lock:
        _session_yolo.add(session_key)


def disable_session_yolo(session_key: str) -> None:
    """为单个会话键禁用 YOLO 绕过。"""
    if not session_key:
        return
    with _lock:
        _session_yolo.discard(session_key)


def clear_session(session_key: str) -> None:
    """移除给定会话的所有审批和 yolo 状态。"""
    if not session_key:
        return
    with _lock:
        _session_approved.pop(session_key, None)
        _session_yolo.discard(session_key)
        _pending.pop(session_key, None)
        entries = _gateway_queues.pop(session_key, [])
    for entry in entries:
        # 会话边界清理应立即取消任何阻塞的审批等待，
        # 使旧运行可以 unwind 而不是 idle 到超时。
        entry.result = "deny"
        entry.event.set()


def is_session_yolo_enabled(session_key: str) -> bool:
    """当特定会话启用了 YOLO 绕过时返回 True。"""
    if not session_key:
        return False
    with _lock:
        return session_key in _session_yolo


def is_current_session_yolo_enabled() -> bool:
    """当活动审批会话启用了 YOLO 绕过时返回 True。"""
    return is_session_yolo_enabled(get_current_session_key(default=""))


def is_approved(session_key: str, pattern_key: str) -> bool:
    """检查模式是否已批准（会话级或永久）。

    同时接受当前规范键和旧版正则派生键，
    使现有 command_allowlist 条目在键迁移后继续工作。
    """
    aliases = _approval_key_aliases(pattern_key)
    with _lock:
        if any(alias in _permanent_approved for alias in aliases):
            return True
        session_approvals = _session_approved.get(session_key, set())
        return any(alias in session_approvals for alias in aliases)


def approve_permanent(pattern_key: str):
    """将一个模式添加到永久白名单。"""
    with _lock:
        _permanent_approved.add(pattern_key)


def load_permanent(patterns: set):
    """从配置批量加载永久白名单条目。"""
    with _lock:
        _permanent_approved.update(patterns)


# =========================================================================
# 永久白名单的配置持久化
# =========================================================================

def load_permanent_allowlist() -> set:
    """从配置加载永久允许的命令模式。

    同时将它们同步到审批模块，使 is_approved() 对通过
    之前会话中的 'always' 添加的模式也能工作。
    """
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        patterns = set(config.get("command_allowlist", []) or [])
        if patterns:
            load_permanent(patterns)
        return patterns
    except Exception as e:
        logger.warning("Failed to load permanent allowlist: %s", e)
        return set()


def save_permanent_allowlist(patterns: set):
    """将永久允许的命令模式保存到配置。"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        import yaml
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        config["command_allowlist"] = list(patterns)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    except Exception as e:
        logger.warning("Could not save allowlist: %s", e)


# =========================================================================
# 审批提示 + 编排
# =========================================================================

def prompt_dangerous_approval(command: str, description: str,
                              timeout_seconds: int | None = None,
                              allow_permanent: bool = True,
                              approval_callback=None) -> str:
    """提示用户批准危险命令（仅限 CLI）。

    参数:
        allow_permanent: 为 False 时隐藏 [a]lways 选项（当存在 tirith
            警告时使用，因为宽泛的永久允许列表不适用于内容级安全发现）。
        approval_callback: CLI 注册的可选回调，用于 prompt_toolkit 集成。
            签名: (command, description, *, allow_permanent=True) -> str。

    返回: 'once', 'session', 'always', 或 'deny'
    """
    if timeout_seconds is None:
        timeout_seconds = _get_approval_timeout()

    if approval_callback is not None:
        try:
            return approval_callback(command, description,
                                     allow_permanent=allow_permanent)
        except Exception as e:
            logger.error("Approval callback failed: %s", e, exc_info=True)
            return "deny"

    # 故障关闭保护：如果 prompt_toolkit 拥有终端（交互式 CLI 会话）
    # 且此线程未注册审批回调，下面的 input() 回退会生成一个守护线程，
    # 其读取永远看不到 Enter —— 用户的按键去往 prompt_toolkit 而非 input()，
    # 产生一个不可见的 60 秒死锁。
    # 改为快速拒绝并大声记录，使调用者可以向 Agent 展示真正的错误。
    try:
        from prompt_toolkit.application.current import get_app_or_none
        if get_app_or_none() is not None:
            logger.warning(
                "Dangerous-command approval requested on a thread with no "
                "approval callback while prompt_toolkit is active; denying "
                "to avoid stdin deadlock. command=%r description=%r",
                command, description,
            )
            return "deny"
    except Exception:
        pass

    os.environ["LLAMA_SPINNER_PAUSE"] = "1"
    try:
        while True:
            print()
            print(f"  ⚠️  DANGEROUS COMMAND: {description}")
            print(f"      {command}")
            print()
            if allow_permanent:
                print("      [o]nce  |  [s]ession  |  [a]lways  |  [d]eny")
            else:
                print("      [o]nce  |  [s]ession  |  [d]eny")
            print()
            sys.stdout.flush()

            result = {"choice": ""}

            def get_input():
                try:
                    prompt = "      Choice [o/s/a/D]: " if allow_permanent else "      Choice [o/s/D]: "
                    result["choice"] = input(prompt).strip().lower()
                except (EOFError, OSError):
                    result["choice"] = ""

            thread = threading.Thread(target=get_input, daemon=True)
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                print("\n      ⏱ Timeout - denying command")
                return "deny"

            choice = result["choice"]
            if choice in ('o', 'once'):
                print("      ✓ Allowed once")
                return "once"
            elif choice in ('s', 'session'):
                print("      ✓ Allowed for this session")
                return "session"
            elif choice in ('a', 'always'):
                if not allow_permanent:
                    print("      ✓ Allowed for this session")
                    return "session"
                print("      ✓ Added to permanent allowlist")
                return "always"
            else:
                print("      ✗ Denied")
                return "deny"

    except (EOFError, KeyboardInterrupt):
        print("\n      ✗ Cancelled")
        return "deny"
    finally:
        if "LLAMA_SPINNER_PAUSE" in os.environ:
            del os.environ["LLAMA_SPINNER_PAUSE"]
        print()
        sys.stdout.flush()


def _normalize_approval_mode(mode) -> str:
    """规范化从 YAML/config 加载的审批模式值。

    YAML 1.1 将裸词如 `off` 视为布尔值，所以像
    `approvals:\n  mode: off` 这样的配置条目会被解析为 False，
    除非加引号。将其视为预期的字符串模式而非回退到手动审批。
    """
    if isinstance(mode, bool):
        return "off" if mode is False else "manual"
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        return normalized or "manual"
    return "manual"


def _get_approval_config() -> dict:
    """读取审批配置块。返回包含 'mode'、'timeout' 等的字典。"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return config.get("approvals", {}) or {}
    except Exception as e:
        logger.warning("Failed to load approval config: %s", e)
        return {}


def _get_approval_mode() -> str:
    """从配置读取审批模式。返回 'manual'、'smart' 或 'off'。"""
    mode = _get_approval_config().get("mode", "manual")
    return _normalize_approval_mode(mode)


def _get_approval_timeout() -> int:
    """从配置读取审批超时。默认 60 秒。"""
    try:
        return int(_get_approval_config().get("timeout", 60))
    except (ValueError, TypeError):
        return 60


def _get_cron_approval_mode() -> str:
    """从配置读取 cron 审批模式。返回 'deny' 或 'approve'。"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        mode = str(config.get("approvals", {}).get("cron_mode", "deny")).lower().strip()
        if mode in ("approve", "off", "allow", "yes"):
            return "approve"
        return "deny"
    except Exception:
        return "deny"


def _smart_approve(command: str, description: str, call_llm_fn=None) -> str:
    """使用辅助 LLM 评估风险并决定审批。

    如果 LLM 判定命令安全则返回 'approve'，
    真正危险则返回 'deny'，不确定则返回 'escalate'。

    灵感来自 OpenAI Codex 的 Smart Approvals guardian subagent
    (openai/codex#13860)。
    """
    try:
        prompt = f"""You are a security reviewer for an AI coding agent. A terminal command was flagged by pattern matching as potentially dangerous.

Command: {command}
Flagged reason: {description}

Assess the ACTUAL risk of this command. Many flagged commands are false positives — for example, `python -c "print('hello')"` is flagged as "script execution via -c flag" but is completely harmless.

Rules:
- APPROVE if the command is clearly safe (benign script execution, safe file operations, development tools, package installs, git operations, etc.)
- DENY if the command could genuinely damage the system (recursive delete of important paths, overwriting system files, fork bombs, wiping disks, dropping databases, etc.)
- ESCALATE if you're uncertain

Respond with exactly one word: APPROVE, DENY, or ESCALATE"""

        if call_llm_fn is not None:
            response = call_llm_fn(
                task="approval",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=16,
            )
            answer = (response.choices[0].message.content or "").strip().upper()
        else:
            # 无 LLM 可用时默认 escalate
            return "escalate"

        if "APPROVE" in answer:
            return "approve"
        elif "DENY" in answer:
            return "deny"
        else:
            return "escalate"

    except Exception as e:
        logger.debug("Smart approvals: LLM call failed (%s), escalating", e)
        return "escalate"


def check_dangerous_command(command: str, env_type: str = "local",
                            approval_callback=None,
                            call_llm_fn=None) -> dict:
    """检查命令是否危险并处理审批。

    这是 terminal_tool 在执行任何命令前调用的主入口。
    编排检测、会话检查和提示。

    参数:
        command: 要检查的 shell 命令。
        env_type: 终端后端类型（'local'、'ssh'、'docker' 等）。
        approval_callback: 可选的 CLI 回调，用于交互式提示。
        call_llm_fn: 可选的 LLM 调用函数，用于 smart approval。

    返回:
        {"approved": True/False, "message": str or None, ...}
    """
    # 容器环境跳过所有审批
    if env_type in ("docker", "singularity", "modal", "daytona", "vercel_sandbox"):
        return {"approved": True, "message": None}

    # Hardline 底线：没有恢复路径的命令（rm -rf /、mkfs、dd 到原始设备、
    # shutdown/reboot、fork bomb、kill -1）被无条件阻止，在 yolo 绕过之前。
    # 选择 yolo 是信任 Agent 处理用户的文件和服务，而不是信任它擦除磁盘或关闭机器。
    is_hardline, hardline_desc = detect_hardline_command(command)
    if is_hardline:
        logger.warning("Hardline block: %s (command: %s)", hardline_desc, command[:200])
        return _hardline_block_result(hardline_desc)

    # --yolo: 绕过所有审批提示。Gateway /yolo 是会话级的；
    # CLI --yolo 通过环境变量保持进程级，供本地使用。
    if _is_truthy_value(os.getenv("LLAMA_YOLO_MODE")) or is_current_session_yolo_enabled():
        return {"approved": True, "message": None}

    is_dangerous, pattern_key, description = detect_dangerous_command(command)
    if not is_dangerous:
        return {"approved": True, "message": None}

    session_key = get_current_session_key()
    if is_approved(session_key, pattern_key):
        return {"approved": True, "message": None}

    is_cli = os.getenv("LLAMA_INTERACTIVE")
    is_gateway = os.getenv("LLAMA_GATEWAY_SESSION")
    is_ask = os.getenv("LLAMA_EXEC_ASK")

    if not is_cli and not is_gateway and not is_ask:
        # Cron 会话：尊重 cron_mode 配置
        if os.getenv("LLAMA_CRON_SESSION"):
            if _get_cron_approval_mode() == "deny":
                return {
                    "approved": False,
                    "message": (
                        f"BLOCKED: Command flagged as dangerous ({description}) "
                        "but cron jobs run without a user present to approve it. "
                        "Find an alternative approach that avoids this command. "
                        "To allow dangerous commands in cron jobs, set "
                        "approvals.cron_mode: approve in config.yaml."
                    ),
                }
        return {"approved": True, "message": None}

    if is_gateway or is_ask:
        submit_pending(session_key, {
            "command": command,
            "pattern_key": pattern_key,
            "description": description,
        })
        return {
            "approved": False,
            "pattern_key": pattern_key,
            "status": "approval_required",
            "command": command,
            "description": description,
            "message": (
                f"⚠️ This command is potentially dangerous ({description}). "
                f"Asking the user for approval.\n\n**Command:**\n```\n{command}\n```"
            ),
        }

    choice = prompt_dangerous_approval(command, description,
                                       approval_callback=approval_callback)

    if choice == "deny":
        return {
            "approved": False,
            "message": f"BLOCKED: User denied this potentially dangerous command (matched '{description}' pattern). Do NOT retry this command - the user has explicitly rejected it.",
            "pattern_key": pattern_key,
            "description": description,
        }

    if choice == "session":
        approve_session(session_key, pattern_key)
    elif choice == "always":
        approve_session(session_key, pattern_key)
        approve_permanent(pattern_key)
        save_permanent_allowlist(_permanent_approved)

    return {"approved": True, "message": None}


def check_all_command_guards(command: str, env_type: str = "local",
                             approval_callback=None,
                             call_llm_fn=None) -> dict:
    """运行所有前置执行安全检查并返回单一审批决定。

    从 tirith 和危险命令检测收集发现，然后将它们作为单一组合审批请求
    呈现。这防止了 gateway force=True 重放只绕过其中一个检查的情况。
    """
    # 对两个检查都跳过容器
    if env_type in ("docker", "singularity", "modal", "daytona", "vercel_sandbox"):
        return {"approved": True, "message": None}

    # Hardline 底线：灾难性命令无条件阻止
    is_hardline, hardline_desc = detect_hardline_command(command)
    if is_hardline:
        logger.warning("Hardline block: %s (command: %s)", hardline_desc, command[:200])
        return _hardline_block_result(hardline_desc)

    # --yolo 或 approvals.mode=off: 绕过所有审批提示
    approval_mode = _get_approval_mode()
    if _is_truthy_value(os.getenv("LLAMA_YOLO_MODE")) or is_current_session_yolo_enabled() or approval_mode == "off":
        return {"approved": True, "message": None}

    is_cli = os.getenv("LLAMA_INTERACTIVE")
    is_gateway = os.getenv("LLAMA_GATEWAY_SESSION")
    is_ask = os.getenv("LLAMA_EXEC_ASK")

    # 保留现有非交互行为：在 CLI/gateway/ask 流程之外，
    # 我们不阻塞审批，并跳过外部守卫工作。
    if not is_cli and not is_gateway and not is_ask:
        if os.getenv("LLAMA_CRON_SESSION"):
            if _get_cron_approval_mode() == "deny":
                is_dangerous, _pk, description = detect_dangerous_command(command)
                if is_dangerous:
                    return {
                        "approved": False,
                        "message": (
                            f"BLOCKED: Command flagged as dangerous ({description}) "
                            "but cron jobs run without a user present to approve it. "
                            "Find an alternative approach that avoids this command. "
                            "To allow dangerous commands in cron jobs, set "
                            "approvals.cron_mode: approve in config.yaml."
                        ),
                    }
        return {"approved": True, "message": None}

    # --- 阶段 1: 从两个检查收集发现 ---

    # 危险命令检查（仅检测，不审批）
    is_dangerous, pattern_key, description = detect_dangerous_command(command)

    # --- 阶段 2: 决定 ---

    warnings = []  # (pattern_key, description, is_tirith) 列表

    session_key = get_current_session_key()

    if is_dangerous:
        if not is_approved(session_key, pattern_key):
            warnings.append((pattern_key, description, False))

    # 没有警告
    if not warnings:
        return {"approved": True, "message": None}

    # --- 阶段 2.5: Smart approval（辅助 LLM 风险评估） ---
    if approval_mode == "smart":
        combined_desc_for_llm = "; ".join(desc for _, desc, _ in warnings)
        verdict = _smart_approve(command, combined_desc_for_llm, call_llm_fn=call_llm_fn)
        if verdict == "approve":
            for key, _, _ in warnings:
                approve_session(session_key, key)
            logger.debug("Smart approval: auto-approved '%s' (%s)",
                         command[:60], combined_desc_for_llm)
            return {"approved": True, "message": None,
                    "smart_approved": True,
                    "description": combined_desc_for_llm}
        elif verdict == "deny":
            return {
                "approved": False,
                "message": f"BLOCKED by smart approval: {combined_desc_for_llm}. "
                           "The command was assessed as genuinely dangerous. Do NOT retry.",
                "smart_denied": True,
            }
        # verdict == "escalate" → 落入手动提示

    # --- 阶段 3: 审批 ---

    combined_desc = "; ".join(desc for _, desc, _ in warnings)
    primary_key = warnings[0][0]
    all_keys = [key for key, _, _ in warnings]

    # Gateway/异步审批
    if is_gateway or is_ask:
        submit_pending(session_key, {
            "command": command,
            "pattern_key": primary_key,
            "pattern_keys": all_keys,
            "description": combined_desc,
        })
        return {
            "approved": False,
            "pattern_key": primary_key,
            "status": "approval_required",
            "command": command,
            "description": combined_desc,
            "message": (
                f"⚠️ {combined_desc}. Asking the user for approval.\n\n**Command:**\n```\n{command}\n```"
            ),
        }

    # CLI 交互式：单一组合提示
    choice = prompt_dangerous_approval(command, combined_desc,
                                       allow_permanent=True,
                                       approval_callback=approval_callback)

    if choice == "deny":
        return {
            "approved": False,
            "message": "BLOCKED: User denied. Do NOT retry.",
            "pattern_key": primary_key,
            "description": combined_desc,
        }

    # 为每个警告单独持久化审批
    for key, _, is_tirith in warnings:
        if choice == "session" or (choice == "always" and is_tirith):
            approve_session(session_key, key)
        elif choice == "always":
            approve_session(session_key, key)
            approve_permanent(key)
            save_permanent_allowlist(_permanent_approved)

    return {"approved": True, "message": None,
            "user_approved": True, "description": combined_desc}


# =========================================================================
# 辅助函数
# =========================================================================

def _is_truthy_value(value) -> bool:
    """检查值是否为真值（处理字符串 'true'、'1'、'yes' 等）。"""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on", "y")
    return bool(value)


# =========================================================================
# API 兼容函数（供 app.py 审批流 API 调用）
# =========================================================================

def get_pending_approvals(session_key: str = "") -> list:
    """获取指定会话的待审批请求列表（网关模式用）。

    返回每个待审批项的 dict 列表，包含 request_id、command、
    description 等字段。空 session_key 返回所有会话的待审批。
    """
    with _lock:
        if session_key:
            entry = _pending.get(session_key)
            if entry:
                return [{"request_id": f"{session_key}::{entry.get('pattern_key', 'unknown')}", **entry}]
            return []
        # 返回所有会话的待审批
        results = []
        for skey, entry in _pending.items():
            results.append({"request_id": f"{skey}::{entry.get('pattern_key', 'unknown')}", "session_key": skey, **entry})
        return results


def approve_request(request_id: str, session_key: str = "") -> bool:
    """批准一个等待中的请求。

    request_id 格式为 "session_key::pattern_key" 或简单 UUID。
    解析出 session_key 后调用 resolve_gateway_approval。
    """
    if "::" in request_id:
        skey, _ = request_id.split("::", 1)
    else:
        skey = session_key or get_current_session_key()
    if not skey:
        return False
    resolved = resolve_gateway_approval(skey, "once")
    # 同时清理 _pending
    with _lock:
        _pending.pop(skey, None)
    return resolved > 0


def deny_request(request_id: str, session_key: str = "") -> bool:
    """拒绝一个等待中的请求。"""
    if "::" in request_id:
        skey, _ = request_id.split("::", 1)
    else:
        skey = session_key or get_current_session_key()
    if not skey:
        return False
    resolved = resolve_gateway_approval(skey, "deny")
    with _lock:
        _pending.pop(skey, None)
    return resolved > 0


def enable_yolo_for_session(session_key: str) -> None:
    """为指定会话启用 YOLO 模式（跳过非 Hardline 审批）。"""
    enable_session_yolo(session_key)


def disable_yolo_for_session(session_key: str) -> None:
    """为指定会话禁用 YOLO 模式。"""
    disable_session_yolo(session_key)


def add_to_allowlist(command: str) -> None:
    """将命令模式添加到永久白名单。"""
    # 使用命令的规范化形式作为模式键
    normalized = _normalize_command_for_detection(command).lower()
    is_dangerous, pattern_key, description = detect_dangerous_command(command)
    if is_dangerous:
        approve_permanent(pattern_key)
    else:
        # 非危险命令也允许添加（用户自定义）
        approve_permanent(normalized[:50])
    save_permanent_allowlist(_permanent_approved)


def remove_from_allowlist(command: str) -> None:
    """从永久白名单中移除命令模式。"""
    normalized = _normalize_command_for_detection(command).lower()
    is_dangerous, pattern_key, _ = detect_dangerous_command(command)
    with _lock:
        if is_dangerous:
            for alias in _approval_key_aliases(pattern_key):
                _permanent_approved.discard(alias)
        else:
            _permanent_approved.discard(normalized[:50])
    save_permanent_allowlist(_permanent_approved)


# 模块导入时加载永久白名单
load_permanent_allowlist()
