"""
技能安全扫描器 — 外部来源技能的安全审查
完整移植自 hermes-agent tools/skills_guard.py

每个从注册表下载的技能在安装前都会经过此扫描器审查。
使用基于正则表达式的静态分析检测已知恶意模式:
  - 数据渗漏 (环境变量读取、DNS 渗漏、Markdown 图片渗漏)
  - Prompt 注入 (角色劫持、系统提示词提取、越狱模式)
  - 破坏性操作 (rm -rf、chmod 777、磁盘覆写)
  - 持久化 (crontab、SSH authorized_keys、systemd 服务)
  - 网络 (反向 Shell、隧道服务、硬编码 IP)
  - 混淆 (base64 解码+管道执行、unicode 转义)
  - 供应链 (curl pipe shell、未锁定依赖)
  - 提权 (sudo NOPASSWD、SUID 位)
  - Agent 配置修改 (AGENTS.md、SOUL.md)

信任级别:
  - builtin:   随项目发布。不扫描,始终信任。
  - trusted:   openai/skills、anthropics/skills、NVIDIA/skills。允许 caution。
  - community: 其他来源。任何发现 = 阻止 (除非 --force)。
  - agent-created: Agent 自建技能。dangerous 时 "ask"。

用法:
    from tools.skills_guard import scan_skill, should_allow_install, format_scan_report
    result = scan_skill(Path("skills/.hub/quarantine/some-skill"), source="community")
    allowed, reason = should_allow_install(result)
    if not allowed:
        print(format_scan_report(result))
"""
from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ─── 项目路径 ─────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DIR = os.path.join(PROJECT_ROOT, "skills")
HUB_DIR = os.path.join(SKILLS_DIR, ".hub")

# ─── 信任配置 ─────────────────────────────────────────────────────────

TRUSTED_REPOS: Set[str] = {
    "openai/skills",
    "anthropics/skills",
    "NVIDIA/skills",
    # 国内镜像也在信任列表中 (内容相同,仅下载源不同)
    "huggingface/skills",
}

INSTALL_POLICY: Dict[str, Tuple[str, str, str]] = {
    #                safe       caution     dangerous
    "builtin":       ("allow",  "allow",    "allow"),
    "trusted":       ("allow",  "allow",    "block"),
    "community":     ("allow",  "block",    "block"),
    "agent-created": ("allow",  "allow",    "ask"),
}

VERDICT_INDEX: Dict[str, int] = {"safe": 0, "caution": 1, "dangerous": 2}


# ─── 数据结构 ─────────────────────────────────────────────────────────

@dataclass
class Finding:
    """单个安全发现"""
    pattern_id: str       # 匹配的模式 ID
    severity: str         # critical | high | medium | low
    category: str         # exfiltration | injection | destructive | persistence | network | obfuscation | ...
    file: str             # 相对文件路径
    line: int             # 行号
    match: str            # 匹配的文本 (截断至 120 字符)
    description: str      # 人类可读说明


@dataclass
class ScanResult:
    """扫描结果"""
    skill_name: str
    source: str           # 来源标识 (如 "openai/skills")
    trust_level: str      # builtin | trusted | community | agent-created
    verdict: str          # safe | caution | dangerous
    findings: List[Finding] = field(default_factory=list)
    scanned_at: str = ""
    summary: str = ""


# ─── 威胁模式数据库 ──────────────────────────────────────────────────

# (regex, pattern_id, severity, category, description)
THREAT_PATTERNS: List[Tuple[str, str, str, str, str]] = [
    # ── 数据渗漏: Shell 命令泄露密钥 ──
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)',
     "env_exfil_curl", "critical", "exfiltration",
     "curl 命令插入了密钥环境变量"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)',
     "env_exfil_wget", "critical", "exfiltration",
     "wget 命令插入了密钥环境变量"),
    (r'fetch\s*\([^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|API)',
     "env_exfil_fetch", "critical", "exfiltration",
     "fetch() 调用插入了密钥环境变量"),
    (r'httpx?\.(get|post|put|patch)\s*\([^\n]*(KEY|TOKEN|SECRET|PASSWORD)',
     "env_exfil_httpx", "critical", "exfiltration",
     "HTTP 库调用包含密钥变量"),
    (r'requests\.(get|post|put|patch)\s*\([^\n]*(KEY|TOKEN|SECRET|PASSWORD)',
     "env_exfil_requests", "critical", "exfiltration",
     "requests 库调用包含密钥变量"),

    # ── 数据渗漏: 读取凭证存储 ──
    (r'base64[^\n]*env',
     "encoded_exfil", "high", "exfiltration",
     "base64 编码结合环境变量访问"),
    (r'\$HOME/\.ssh|\~/\.ssh',
     "ssh_dir_access", "high", "exfiltration",
     "引用了用户 SSH 目录"),
    (r'\$HOME/\.aws|\~/\.aws',
     "aws_dir_access", "high", "exfiltration",
     "引用了用户 AWS 凭证目录"),
    (r'\$HOME/\.gnupg|\~/\.gnupg',
     "gpg_dir_access", "high", "exfiltration",
     "引用了用户 GPG 密钥环"),
    (r'\$HOME/\.kube|\~/\.kube',
     "kube_dir_access", "high", "exfiltration",
     "引用了 Kubernetes 配置目录"),
    (r'\$HOME/\.docker|\~/\.docker',
     "docker_dir_access", "high", "exfiltration",
     "引用了 Docker 配置 (可能包含 registry 凭证)"),
    (r'cat\s+(?!>)[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)',
     "read_secrets_file", "critical", "exfiltration",
     "读取已知凭证文件"),

    # ── 数据渗漏: 编程方式访问环境变量 ──
    (r'printenv|env\s*\|',
     "dump_all_env", "high", "exfiltration",
     "导出所有环境变量"),
    (r'os\.environ\b(?!\s*\.get\s*\(\s*["\'](?![^"\']*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)))',
     "python_os_environ", "high", "exfiltration",
     "访问 os.environ (可能导出环境变量)"),
    (r'os\.environ\s*\.get\s*\(\s*["\'][^"\']*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)',
     "python_environ_get_secret", "critical", "exfiltration",
     "通过 os.environ.get() 读取密钥"),
    (r'os\.getenv\s*\(\s*[^\)]*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)',
     "python_getenv_secret", "critical", "exfiltration",
     "通过 os.getenv() 读取密钥"),
    (r'process\.env\[',
     "node_process_env", "high", "exfiltration",
     "访问 process.env (Node.js 环境变量)"),
    (r'ENV\[.*(?:KEY|TOKEN|SECRET|PASSWORD)',
     "ruby_env_secret", "critical", "exfiltration",
     "通过 Ruby ENV[] 读取密钥"),

    # ── 数据渗漏: DNS 和暂存 ──
    (r'\b(dig|nslookup|host)\s+[^\n]*\$',
     "dns_exfil", "critical", "exfiltration",
     "带变量插值的 DNS 查询 (可能 DNS 渗漏)"),
    (r'>\s*/tmp/[^\s]*\s*&&\s*(curl|wget|nc|python)',
     "tmp_staging", "critical", "exfiltration",
     "写入 /tmp 后渗出"),

    # ── 数据渗漏: Markdown/链接 ──
    (r'!\[.*\]\(https?://[^\)]*\$\{?',
     "md_image_exfil", "high", "exfiltration",
     "Markdown 图片 URL 包含变量插值 (图片渗漏)"),
    (r'\[.*\]\(https?://[^\)]*\$\{?',
     "md_link_exfil", "high", "exfiltration",
     "Markdown 链接包含变量插值"),

    # ── Prompt 注入 ──
    (r'ignore\s+(?:\w+\s+)*(previous|all|above|prior)\s+instructions',
     "prompt_injection_ignore", "critical", "injection",
     "Prompt 注入: 忽略之前的指令"),
    (r'you\s+are\s+(?:\w+\s+)*now\s+',
     "role_hijack", "high", "injection",
     "试图覆盖 Agent 的角色"),
    (r'do\s+not\s+(?:\w+\s+)*tell\s+(?:\w+\s+)*the\s+user',
     "deception_hide", "critical", "injection",
     "指示 Agent 对用户隐藏信息"),
    (r'system\s+(?:\w+\s+)*prompt\s+(?:\w+\s+)*override',
     "sys_prompt_override", "critical", "injection",
     "试图覆盖系统提示词"),
    (r'pretend\s+(?:\w+\s+)*(you\s+are|to\s+be)\s+',
     "role_pretend", "high", "injection",
     "试图让 Agent 扮演不同身份"),
    (r'disregard\s+(?:\w+\s+)*(your|all|any)\s+(?:\w+\s+)*(instructions|rules|guidelines)',
     "disregard_rules", "critical", "injection",
     "指示 Agent 忽略规则"),
    (r'output\s+(?:\w+\s+)*(system|initial)\s+prompt',
     "leak_system_prompt", "high", "injection",
     "试图提取系统提示词"),
    (r'(when|if)\s+no\s*one\s+is\s+(watching|looking)',
     "conditional_deception", "high", "injection",
     "条件性指令: 无人观察时改变行为"),
    (r'act\s+as\s+(if|though)\s+(?:\w+\s+)*you\s+(?:\w+\s+)*(have\s+no|don\'t\s+have)\s+(?:\w+\s+)*(restrictions|limits|rules)',
     "bypass_restrictions", "critical", "injection",
     "指示 Agent 无限制地行动"),
    (r'translate\s+.*\s+into\s+.*\s+and\s+(execute|run|eval)',
     "translate_execute", "critical", "injection",
     "翻译后执行 (规避技术)"),
    (r'<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->',
     "html_comment_injection", "high", "injection",
     "HTML 注释中隐藏指令"),
    (r'<\s*div\s+style\s*=\s*["\'][\s\S]*?display\s*:\s*none',
     "hidden_div", "high", "injection",
     "隐藏 HTML div (不可见指令)"),

    # ── 越狱模式 ──
    (r'\bDAN\s+mode\b|Do\s+Anything\s+Now',
     "jailbreak_dan", "critical", "injection",
     "DAN (Do Anything Now) 越狱尝试"),
    (r'\bdeveloper\s+mode\b.*\benabled?\b',
     "jailbreak_dev_mode", "critical", "injection",
     "开发者模式越狱尝试"),
    (r'hypothetical\s+scenario.*(?:ignore|bypass|override)',
     "hypothetical_bypass", "high", "injection",
     "假设情景用于绕过限制"),
    (r'(respond|answer|reply)\s+without\s+(?:\w+\s+)*(restrictions|limitations|filters|safety)',
     "remove_filters", "critical", "injection",
     "指示 Agent 无视安全过滤器回复"),
    (r'you\s+have\s+been\s+(?:\w+\s+)*(updated|upgraded|patched)\s+to',
     "fake_update", "high", "injection",
     "伪造更新公告 (社会工程)"),
    (r'new\s+(?:\w+\s+)*policy|updated\s+(?:\w+\s+)*guidelines|revised\s+(?:\w+\s+)*instructions',
     "fake_policy", "medium", "injection",
     "声称新策略/指南 (可能是社会工程)"),

    # ── 上下文窗口渗漏 ──
    (r'(include|output|print|send|share)\s+(?:\w+\s+)*(conversation|chat\s+history|previous\s+messages|context)',
     "context_exfil", "high", "exfiltration",
     "指示 Agent 输出/分享对话历史"),
    (r'(send|post|upload|transmit)\s+.*\s+(to|at)\s+https?://',
     "send_to_url", "high", "exfiltration",
     "指示 Agent 发送数据到 URL"),

    # ── 破坏性操作 ──
    (r'rm\s+-rf\s+/',
     "destructive_root_rm", "critical", "destructive",
     "从根目录递归删除"),
    (r'rm\s+(-[^\s]*)?r.*\$HOME|\brmdir\s+.*\$HOME',
     "destructive_home_rm", "critical", "destructive",
     "递归删除家目录"),
    (r'chmod\s+777',
     "insecure_perms", "medium", "destructive",
     "设置全局可写权限"),
    (r'>\s*/etc/',
     "system_overwrite", "critical", "destructive",
     "覆写系统配置文件"),
    (r'\bmkfs\b',
     "format_filesystem", "critical", "destructive",
     "格式化文件系统"),
    (r'\bdd\s+.*if=.*of=/dev/',
     "disk_overwrite", "critical", "destructive",
     "原始磁盘写入操作"),
    (r'shutil\.rmtree\s*\(\s*[\"\'/]',
     "python_rmtree", "high", "destructive",
     "Python rmtree 删除绝对/根路径"),
    (r'truncate\s+-s\s*0\s+/',
     "truncate_system", "critical", "destructive",
     "截断系统文件为零字节"),

    # ── 持久化 ──
    (r'\bcrontab\b',
     "persistence_cron", "medium", "persistence",
     "修改 cron 作业"),
    (r'\.(bashrc|zshrc|profile|bash_profile|bash_login|zprofile|zlogin)\b',
     "shell_rc_mod", "medium", "persistence",
     "引用 Shell 启动文件"),
    (r'authorized_keys',
     "ssh_backdoor", "critical", "persistence",
     "修改 SSH authorized_keys"),
    (r'ssh-keygen',
     "ssh_keygen", "medium", "persistence",
     "生成 SSH 密钥"),
    (r'systemd.*\.service|systemctl\s+(enable|start)',
     "systemd_service", "medium", "persistence",
     "引用/启用 systemd 服务"),
    (r'/etc/init\.d/',
     "init_script", "medium", "persistence",
     "引用 init.d 启动脚本"),
    (r'/etc/sudoers|visudo',
     "sudoers_mod", "critical", "persistence",
     "修改 sudoers (提权)"),
    (r'git\s+config\s+--global\s+',
     "git_config_global", "medium", "persistence",
     "修改全局 git 配置"),

    # ── Agent 配置持久化 ──
    (r'AGENTS\.md|CLAUDE\.md|\.cursorrules|\.clinerules',
     "agent_config_mod", "critical", "persistence",
     "引用 Agent 配置文件 (可通过会话跨越持久化恶意指令)"),
    (r'SKILL\.md|SOUL\.md|USER\.md',
     "project_config_mod", "medium", "persistence",
     "引用项目级 Agent 配置文件"),

    # ── 网络: 反向 Shell 和隧道 ──
    (r'\bnc\s+-[lp]|ncat\s+-[lp]|\bsocat\b',
     "reverse_shell", "critical", "network",
     "潜在反向 Shell 监听器"),
    (r'\bngrok\b|\blocaltunnel\b|\bserveo\b|\bcloudflared\b',
     "tunnel_service", "high", "network",
     "使用隧道服务进行外部访问"),
    (r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{2,5}',
     "hardcoded_ip_port", "medium", "network",
     "硬编码 IP 地址和端口"),
    (r'0\.0\.0\.0:\d+|INADDR_ANY',
     "bind_all_interfaces", "high", "network",
     "绑定到所有网络接口"),
    (r'/bin/(ba)?sh\s+-i\s+.*>/dev/tcp/',
     "bash_reverse_shell", "critical", "network",
     "bash 交互式反向 Shell via /dev/tcp"),
    (r'python[23]?\s+-c\s+["\']import\s+socket',
     "python_socket_oneliner", "critical", "network",
     "Python 单行 socket 连接 (可能是反向 Shell)"),
    (r'socket\.connect\s*\(\s*\(',
     "python_socket_connect", "high", "network",
     "Python socket 连接到任意主机"),
    (r'webhook\.site|requestbin\.com|pipedream\.net|hookbin\.com',
     "exfil_service", "high", "network",
     "引用已知数据渗漏/Webhook 测试服务"),
    (r'pastebin\.com|hastebin\.com|ghostbin\.',
     "paste_service", "medium", "network",
     "引用粘贴服务 (可能数据暂存)"),

    # ── 混淆: 编码和 eval ──
    (r'base64\s+(-d|--decode)\s*\|',
     "base64_decode_pipe", "high", "obfuscation",
     "base64 解码后管道到执行"),
    (r'\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}',
     "hex_encoded_string", "medium", "obfuscation",
     "十六进制编码字符串 (可能混淆)"),
    (r'\beval\s*\(\s*["\']',
     "eval_string", "high", "obfuscation",
     "eval() 带字符串参数"),
    (r'\bexec\s*\(\s*["\']',
     "exec_string", "high", "obfuscation",
     "exec() 带字符串参数"),
    (r'echo\s+[^\n]*\|\s*(bash|sh|python|perl|ruby|node)',
     "echo_pipe_exec", "critical", "obfuscation",
     "echo 管道到解释器执行"),
    (r'compile\s*\(\s*[^\)]+,\s*["\'].*["\']\s*,\s*["\']exec["\']\s*\)',
     "python_compile_exec", "high", "obfuscation",
     "Python compile() exec 模式"),
    (r'getattr\s*\(\s*__builtins__',
     "python_getattr_builtins", "high", "obfuscation",
     "动态访问 Python builtins (规避技术)"),
    (r'__import__\s*\(\s*["\']os["\']\s*\)',
     "python_import_os", "high", "obfuscation",
     "动态 import os 模块"),
    (r'codecs\.decode\s*\(\s*["\']',
     "python_codecs_decode", "medium", "obfuscation",
     "codecs.decode (可能 ROT13 或编码混淆)"),
    (r'String\.fromCharCode|charCodeAt',
     "js_char_code", "medium", "obfuscation",
     "JavaScript 字符代码构造 (可能混淆)"),
    (r'atob\s*\(|btoa\s*\(',
     "js_base64", "medium", "obfuscation",
     "JavaScript base64 编解码"),

    # ── 代码执行 ──
    (r'subprocess\.(run|call|Popen|check_output)\s*\(',
     "python_subprocess", "medium", "execution",
     "Python subprocess 执行"),
    (r'os\.system\s*\(',
     "python_os_system", "high", "execution",
     "os.system() — 无保护的 Shell 执行"),
    (r'os\.popen\s*\(',
     "python_os_popen", "high", "execution",
     "os.popen() — Shell 管道执行"),
    (r'child_process\.(exec|spawn|fork)\s*\(',
     "node_child_process", "high", "execution",
     "Node.js child_process 执行"),
    (r'Runtime\.getRuntime\(\)\.exec\(',
     "java_runtime_exec", "high", "execution",
     "Java Runtime.exec() — Shell 执行"),

    # ── 路径遍历 ──
    (r'\.\./\.\./\.\.',
     "path_traversal_deep", "high", "traversal",
     "深层相对路径遍历 (3+ 级别)"),
    (r'/etc/passwd|/etc/shadow',
     "system_passwd_access", "critical", "traversal",
     "引用系统密码文件"),
    (r'/proc/self|/proc/\d+/',
     "proc_access", "high", "traversal",
     "引用 /proc 文件系统"),

    # ── 供应链: curl/wget 管道到 Shell ──
    (r'curl\s+[^\n]*\|\s*(ba)?sh',
     "curl_pipe_shell", "critical", "supply_chain",
     "curl 管道到 Shell (下载即执行)"),
    (r'wget\s+[^\n]*-O\s*-\s*\|\s*(ba)?sh',
     "wget_pipe_shell", "critical", "supply_chain",
     "wget 管道到 Shell (下载即执行)"),
    (r'curl\s+[^\n]*\|\s*python',
     "curl_pipe_python", "critical", "supply_chain",
     "curl 管道到 Python 解释器"),

    # ── 供应链: 未锁定/延迟依赖 ──
    (r'pip\s+install\s+(?!-r\s)(?!.*==)',
     "unpinned_pip_install", "medium", "supply_chain",
     "pip install 无版本锁定"),
    (r'npm\s+install\s+(?!.*@\d)',
     "unpinned_npm_install", "medium", "supply_chain",
     "npm install 无版本锁定"),

    # ── 供应链: 远程资源获取 ──
    (r'(curl|wget|httpx?\.get|requests\.get|fetch)\s*[\(]?\s*["\']https?://',
     "remote_fetch", "medium", "supply_chain",
     "运行时获取远程资源"),
    (r'git\s+clone\s+',
     "git_clone", "medium", "supply_chain",
     "运行时克隆 git 仓库"),
    (r'docker\s+pull\s+',
     "docker_pull", "medium", "supply_chain",
     "运行时拉取 Docker 镜像"),

    # ── 提权 ──
    (r'\bsudo\b',
     "sudo_usage", "high", "privilege_escalation",
     "使用 sudo (提权)"),
    (r'setuid|setgid|cap_setuid',
     "setuid_setgid", "critical", "privilege_escalation",
     "setuid/setgid (提权机制)"),
    (r'NOPASSWD',
     "nopasswd_sudo", "critical", "privilege_escalation",
     "NOPASSWD sudoers 条目 (无密码提权)"),
    (r'chmod\s+[u+]?s',
     "suid_bit", "critical", "privilege_escalation",
     "设置 SUID/SGID 位"),

    # ── 硬编码密钥 ──
    (r'(?:api[_-]?key|token|secret|password)\s*[=:]\s*["\'][A-Za-z0-9+/=_-]{20,}',
     "hardcoded_secret", "critical", "credential_exposure",
     "可能硬编码了 API 密钥、token 或密码"),
    (r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
     "embedded_private_key", "critical", "credential_exposure",
     "内嵌私钥"),
    (r'ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{80,}',
     "github_token_leaked", "critical", "credential_exposure",
     "GitHub Personal Access Token"),
    (r'sk-[A-Za-z0-9]{20,}',
     "openai_key_leaked", "critical", "credential_exposure",
     "可能泄露了 OpenAI API Key"),
    (r'sk-ant-[A-Za-z0-9_-]{90,}',
     "anthropic_key_leaked", "critical", "credential_exposure",
     "可能泄露了 Anthropic API Key"),
    (r'AKIA[0-9A-Z]{16}',
     "aws_access_key_leaked", "critical", "credential_exposure",
     "AWS Access Key ID"),

    # ── 加密货币挖矿 ──
    (r'xmrig|stratum\+tcp|monero|coinhive|cryptonight',
     "crypto_mining", "critical", "mining",
     "加密货币挖矿引用"),
]

# ─── 结构限制 ─────────────────────────────────────────────────────────

MAX_FILE_COUNT = 50          # 技能目录文件数上限
MAX_TOTAL_SIZE_KB = 1024     # 总大小上限 1MB
MAX_SINGLE_FILE_KB = 256     # 单文件上限 256KB

# 需要扫描的文件扩展名 (文本文件, 跳过二进制)
SCANNABLE_EXTENSIONS: Set[str] = {
    '.md', '.txt', '.py', '.sh', '.bash', '.js', '.ts', '.rb',
    '.yaml', '.yml', '.json', '.toml', '.cfg', '.ini', '.conf',
    '.html', '.css', '.xml', '.tex', '.r', '.jl', '.pl', '.php',
}

# 不应出现在技能中的二进制扩展名
SUSPICIOUS_BINARY_EXTENSIONS: Set[str] = {
    '.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.com',
    '.msi', '.dmg', '.app', '.deb', '.rpm',
}

# 不可见 Unicode 字符 (用于注入)
INVISIBLE_CHARS: Set[str] = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\u2062', '\u2063',
    '\u2064', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
    '\u2066', '\u2067', '\u2068', '\u2069',
}

# ─── 忽略文件 ─────────────────────────────────────────────────────────

_SKILL_IGNORE_FILENAMES = (".skillignore", ".clawhubignore")
_ALWAYS_IGNORED_NAMES = set(_SKILL_IGNORE_FILENAMES)
_NEVER_IGNORABLE = {"SKILL.md"}


# ─── 公开 API ─────────────────────────────────────────────────────────

def scan_file(file_path: Path, rel_path: str = "") -> List[Finding]:
    """
    扫描单个文件, 检测威胁模式 + 不可见 Unicode 字符.

    Args:
        file_path: 文件的绝对路径
        rel_path: 用于显示的相对路径 (默认使用文件名)

    Returns:
        Finding 列表 (每行每个模式去重)
    """
    if not rel_path:
        rel_path = file_path.name

    if file_path.suffix.lower() not in SCANNABLE_EXTENSIONS and file_path.name != "SKILL.md":
        return []

    try:
        content = file_path.read_text(encoding='utf-8')
    except (UnicodeDecodeError, OSError):
        return []

    findings: List[Finding] = []
    lines = content.split('\n')
    seen: Set[Tuple[str, int]] = set()

    # 正则模式匹配
    for pattern, pid, severity, category, description in THREAT_PATTERNS:
        for i, line in enumerate(lines, start=1):
            if (pid, i) in seen:
                continue
            if re.search(pattern, line, re.IGNORECASE):
                seen.add((pid, i))
                matched_text = line.strip()
                if len(matched_text) > 120:
                    matched_text = matched_text[:117] + "..."
                findings.append(Finding(
                    pattern_id=pid,
                    severity=severity,
                    category=category,
                    file=rel_path,
                    line=i,
                    match=matched_text,
                    description=description,
                ))

    # 不可见 Unicode 检测
    for i, line in enumerate(lines, start=1):
        for char in INVISIBLE_CHARS:
            if char in line:
                char_name = _unicode_char_name(char)
                findings.append(Finding(
                    pattern_id="invisible_unicode",
                    severity="high",
                    category="injection",
                    file=rel_path,
                    line=i,
                    match=f"U+{ord(char):04X} ({char_name})",
                    description=f"不可见 Unicode 字符 {char_name} (可能用于文本隐藏/注入)",
                ))
                break

    return findings


def scan_skill(skill_path: Path, source: str = "community") -> ScanResult:
    """
    扫描技能目录中所有文件的安全威胁.

    执行:
      1. 结构检查 (文件数、总大小、二进制文件、符号链接)
      2. 正则模式匹配 (所有文本文件)
      3. 不可见 Unicode 字符检测

    Args:
        skill_path: 技能目录路径 (必须包含 SKILL.md)
        source: 来源标识 (如 "openai/skills")

    Returns:
        带 verdict、findings、信任元数据的 ScanResult
    """
    skill_name = skill_path.name
    trust_level = _resolve_trust_level(source)
    all_findings: List[Finding] = []

    if skill_path.is_dir():
        ignore = _load_skill_ignore(skill_path)

        # 结构检查 (遵循忽略列表)
        all_findings.extend(_check_structure(skill_path, ignore=ignore))

        # 模式扫描每个文件
        for f in skill_path.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(skill_path))
                if ignore(rel):
                    continue
                all_findings.extend(scan_file(f, rel))
    elif skill_path.is_file():
        all_findings.extend(scan_file(skill_path, skill_path.name))

    verdict = _determine_verdict(all_findings)
    summary = _build_summary(skill_name, source, trust_level, verdict, all_findings)

    return ScanResult(
        skill_name=skill_name,
        source=source,
        trust_level=trust_level,
        verdict=verdict,
        findings=all_findings,
        scanned_at=datetime.now(timezone.utc).isoformat(),
        summary=summary,
    )


def should_allow_install(result: ScanResult, force: bool = False) -> Tuple[Optional[bool], str]:
    """
    根据扫描结果和信任级别决定是否允许安装技能.

    Args:
        result: scan_skill() 的扫描结果
        force: 是否强制覆盖阻止策略

    Returns:
        (allowed, reason) — allowed 为 None 表示需要用户确认
    """
    policy = INSTALL_POLICY.get(result.trust_level, INSTALL_POLICY["community"])
    vi = VERDICT_INDEX.get(result.verdict, 2)
    decision = policy[vi]

    if decision == "allow":
        return True, f"允许 ({result.trust_level} 源, {result.verdict} 判级)"

    if force and not (result.verdict == "dangerous" and result.trust_level in ("community", "trusted")):
        return True, f"强制安装 — 忽略 {result.verdict} 判级 ({len(result.findings)} 个发现)"

    if decision == "ask":
        return None, (
            f"需要确认 ({result.trust_level} 源 + {result.verdict} 判级, "
            f"{len(result.findings)} 个发现)"
        )

    if result.verdict == "dangerous" and result.trust_level in ("community", "trusted"):
        return False, (
            f"已阻止 ({result.trust_level} 源 + dangerous 判级, "
            f"{len(result.findings)} 个发现). --force 不能覆盖 dangerous 判级."
        )
    return False, (
        f"已阻止 ({result.trust_level} 源 + {result.verdict} 判级, "
        f"{len(result.findings)} 个发现). 使用 --force 覆盖."
    )


def format_scan_report(result: ScanResult) -> str:
    """
    将扫描结果格式化为人类可读的报告字符串.
    """
    lines: List[str] = []

    verdict_display = result.verdict.upper()
    lines.append(
        f"扫描: {result.skill_name} ({result.source}/{result.trust_level})  "
        f"判级: {verdict_display}"
    )

    if result.findings:
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_findings = sorted(
            result.findings,
            key=lambda f: severity_order.get(f.severity, 4)
        )

        for f in sorted_findings:
            sev = f.severity.upper().ljust(8)
            cat = f.category.ljust(14)
            loc = f"{f.file}:{f.line}".ljust(30)
            lines.append(f"  {sev} {cat} {loc} \"{f.match[:60]}\"")

        lines.append("")

    allowed, reason = should_allow_install(result)
    if allowed is True:
        status = "允许"
    elif allowed is None:
        status = "需要确认"
    else:
        status = "已阻止"
    lines.append(f"决定: {status} — {reason}")

    return "\n".join(lines)


def content_hash(skill_path: Path) -> str:
    """
    计算技能目录所有文件的 SHA-256 哈希 (用于完整性追踪).

    文件路径 (相对于 skill_path) 和文件内容都参与哈希计算,
    所以交换两个文件内容会改变哈希值.
    """
    h = hashlib.sha256()
    if skill_path.is_dir():
        for f in sorted(skill_path.rglob("*")):
            if f.is_file():
                try:
                    rel = f.relative_to(skill_path).as_posix()
                    h.update(rel.encode("utf-8"))
                    h.update(b"\x00")
                    h.update(f.read_bytes())
                except OSError:
                    continue
    elif skill_path.is_file():
        h.update(skill_path.read_bytes())
    return f"sha256:{h.hexdigest()[:16]}"


# ─── 内部函数 ─────────────────────────────────────────────────────────

def _check_structure(skill_dir: Path, ignore: Optional[Callable] = None) -> List[Finding]:
    """
    检查技能目录的结构异常.

    Args:
        skill_dir: 技能目录路径
        ignore: 可选回调, 接收相对路径返回 True 表示排除.
                被排除的文件不计入文件数/总大小/结构发现.
    """
    if ignore is None:
        ignore = lambda _rel: False

    findings: List[Finding] = []
    file_count = 0
    total_size = 0

    for f in skill_dir.rglob("*"):
        if not f.is_file() and not f.is_symlink():
            continue

        rel = str(f.relative_to(skill_dir))
        if ignore(rel):
            continue
        file_count += 1

        # 符号链接检查
        if f.is_symlink():
            try:
                resolved = f.resolve()
                if not resolved.is_relative_to(skill_dir.resolve()):
                    findings.append(Finding(
                        pattern_id="symlink_escape",
                        severity="critical",
                        category="traversal",
                        file=rel,
                        line=0,
                        match=f"symlink -> {resolved}",
                        description="符号链接指向技能目录之外",
                    ))
            except OSError:
                findings.append(Finding(
                    pattern_id="broken_symlink",
                    severity="medium",
                    category="traversal",
                    file=rel,
                    line=0,
                    match="broken symlink",
                    description="损坏或循环的符号链接",
                ))
            continue

        # 大小跟踪
        try:
            size = f.stat().st_size
            total_size += size
        except OSError:
            continue

        # 单文件过大
        if size > MAX_SINGLE_FILE_KB * 1024:
            findings.append(Finding(
                pattern_id="oversized_file",
                severity="medium",
                category="structural",
                file=rel,
                line=0,
                match=f"{size // 1024}KB",
                description=f"文件过大 {size // 1024}KB (上限: {MAX_SINGLE_FILE_KB}KB)",
            ))

        # 二进制/可执行文件
        ext = f.suffix.lower()
        if ext in SUSPICIOUS_BINARY_EXTENSIONS:
            findings.append(Finding(
                pattern_id="binary_file",
                severity="critical",
                category="structural",
                file=rel,
                line=0,
                match=f"binary: {ext}",
                description=f"二进制/可执行文件 ({ext}) 不应在技能中",
            ))

        # 非脚本文件的可执行权限
        if ext not in {'.sh', '.bash', '.py', '.rb', '.pl'} and f.stat().st_mode & 0o111:
            findings.append(Finding(
                pattern_id="unexpected_executable",
                severity="medium",
                category="structural",
                file=rel,
                line=0,
                match="executable bit set",
                description="文件设置了可执行权限但不是已知脚本类型",
            ))

    # 文件数限制
    if file_count > MAX_FILE_COUNT:
        findings.append(Finding(
            pattern_id="too_many_files",
            severity="medium",
            category="structural",
            file="(directory)",
            line=0,
            match=f"{file_count} files",
            description=f"技能包含 {file_count} 个文件 (上限: {MAX_FILE_COUNT})",
        ))

    # 总大小限制
    if total_size > MAX_TOTAL_SIZE_KB * 1024:
        findings.append(Finding(
            pattern_id="oversized_skill",
            severity="high",
            category="structural",
            file="(directory)",
            line=0,
            match=f"{total_size // 1024}KB total",
            description=f"技能总大小 {total_size // 1024}KB (上限: {MAX_TOTAL_SIZE_KB}KB)",
        ))

    return findings


def _unicode_char_name(char: str) -> str:
    """获取不可见 Unicode 字符的可读名称"""
    names = {
        '\u200b': "zero-width space",
        '\u200c': "zero-width non-joiner",
        '\u200d': "zero-width joiner",
        '\u2060': "word joiner",
        '\u2062': "invisible times",
        '\u2063': "invisible separator",
        '\u2064': "invisible plus",
        '\ufeff': "BOM/zero-width no-break space",
        '\u202a': "LTR embedding",
        '\u202b': "RTL embedding",
        '\u202c': "pop directional",
        '\u202d': "LTR override",
        '\u202e': "RTL override",
        '\u2066': "LTR isolate",
        '\u2067': "RTL isolate",
        '\u2068': "first strong isolate",
        '\u2069': "pop directional isolate",
    }
    return names.get(char, f"U+{ord(char):04X}")


def _load_skill_ignore(skill_dir: Path) -> Callable[[str], bool]:
    """
    从技能的 `.skillignore` / `.clawhubignore` 构建匹配器.

    返回 callback `ignore(rel_posix_path) -> bool`.
    支持 gitignore 风格基础语法: 空行和 # 注释跳过,
    末尾 / 标记目录, * / ? 通配符.
    """
    patterns: List[str] = []
    for name in _SKILL_IGNORE_FILENAMES:
        ig = skill_dir / name
        try:
            if ig.is_file():
                for raw in ig.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    patterns.append(line)
        except (UnicodeDecodeError, OSError):
            continue

    def ignore(rel: str) -> bool:
        rel_posix = Path(rel).as_posix()
        base = rel_posix.split("/")[-1]

        if base in _NEVER_IGNORABLE:
            return False
        if base in _ALWAYS_IGNORED_NAMES:
            return True

        for pat in patterns:
            anchored = pat.startswith("/")
            p = pat.lstrip("/")
            is_dir = p.endswith("/")
            p = p.rstrip("/")
            if not p:
                continue

            if is_dir:
                if rel_posix == p or rel_posix.startswith(p + "/"):
                    return True
                if not anchored and ("/" + rel_posix + "/").find("/" + p + "/") != -1:
                    return True
                continue

            if fnmatch.fnmatch(rel_posix, p):
                return True
            if not anchored:
                if fnmatch.fnmatch(base, p):
                    return True
                if "/" not in p and any(
                    fnmatch.fnmatch(seg, p) for seg in rel_posix.split("/")
                ):
                    return True
                if rel_posix.startswith(p + "/"):
                    return True
        return False

    return ignore


def _resolve_trust_level(source: str) -> str:
    """将来源标识映射到信任级别"""
    # 标准化前缀别名
    prefix_aliases = ("skills-sh/", "skills.sh/", "skils-sh/", "skils.sh/")
    normalized_source = source
    for prefix in prefix_aliases:
        if normalized_source.startswith(prefix):
            normalized_source = normalized_source[len(prefix):]
            break

    if normalized_source == "agent-created":
        return "agent-created"
    if normalized_source == "official":
        return "builtin"

    for trusted in TRUSTED_REPOS:
        if normalized_source == trusted or normalized_source.startswith(f"{trusted}/"):
            return "trusted"
    return "community"


def _determine_verdict(findings: List[Finding]) -> str:
    """从发现列表确定整体判级"""
    if not findings:
        return "safe"

    has_critical = any(f.severity == "critical" for f in findings)
    has_high = any(f.severity == "high" for f in findings)

    if has_critical:
        return "dangerous"
    if has_high:
        return "caution"
    return "safe"


def _build_summary(name: str, source: str, trust: str, verdict: str,
                   findings: List[Finding]) -> str:
    """构建扫描结果的单行摘要"""
    if not findings:
        return f"{name}: 干净扫描, 未检测到威胁"

    categories = {f.category for f in findings}
    return (
        f"{name}: {verdict} — "
        f"{len(findings)} 个发现 ({', '.join(sorted(categories))})"
    )
