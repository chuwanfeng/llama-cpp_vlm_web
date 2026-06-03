"""
中央工具注册表 — hermes-agent 风格，支持 AST 自动发现。

    移植自 hermes-agent/tools/registry.py

关键特性：
    - AST 自动发现：检测包含 registry.register() 调用的模块（无需文件名模式匹配）
    - 模块级单例 registry（A 模式）+ get_registry()（B 模式）双模式兼容
    - tool_error() / tool_result() 辅助函数
    - 线程安全（RLock）
    - dispatch() 自动捕获异常，返回 JSON 错误
"""
import ast
import importlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from utils import get_logger

logger = get_logger("tools.registry")


# ── AST 自动发现 ────────────────────────────────────────────────

def _is_registry_register_call(node: ast.AST) -> bool:
    """检测 AST 节点是否为顶层 registry.register(...) 调用。"""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "register"
        and isinstance(func.value, ast.Name)
        and func.value.id == "registry"
    )


def _module_registers_tools(module_path: Path) -> bool:
    """通过 AST 检测模块是否包含顶层 registry.register(...) 调用。"""
    try:
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
    except (OSError, SyntaxError):
        return False
    return any(_is_registry_register_call(stmt) for stmt in tree.body)


# ── check_fn TTL 缓存 ────────────────────────────────────────────

_CHECK_FN_TTL_SECONDS = 30.0
_check_fn_cache: Dict[Callable, tuple] = {}
_check_fn_cache_lock = threading.Lock()


def _check_fn_cached(fn: Callable) -> bool:
    now = time.monotonic()
    with _check_fn_cache_lock:
        cached = _check_fn_cache.get(fn)
        if cached is not None:
            ts, value = cached
            if now - ts < _CHECK_FN_TTL_SECONDS:
                return value
    try:
        value = bool(fn())
    except Exception:
        value = False
    with _check_fn_cache_lock:
        _check_fn_cache[fn] = (now, value)
    return value


def invalidate_check_fn_cache() -> None:
    """清除 check_fn 缓存。配置变更后调用。"""
    with _check_fn_cache_lock:
        _check_fn_cache.clear()


# ── ToolEntry ────────────────────────────────────────────────────

class ToolEntry:
    """单个工具的元数据。"""
    __slots__ = (
        "name", "toolset", "schema", "handler", "check_fn",
        "requires_env", "is_async", "description", "emoji",
        "max_result_size_chars",
    )

    def __init__(self, name, toolset, schema, handler, check_fn,
                 requires_env, is_async, description, emoji,
                 max_result_size_chars=None):
        self.name = name
        self.toolset = toolset
        self.schema = schema
        self.handler = handler
        self.check_fn = check_fn
        self.requires_env = requires_env or []
        self.is_async = is_async
        self.description = description or schema.get("description", "")
        self.emoji = emoji
        self.max_result_size_chars = max_result_size_chars

    def is_available(self) -> bool:
        """检查工具是否可用（通过 check_fn）。"""
        if self.check_fn is None:
            return True
        return _check_fn_cached(self.check_fn)

    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI function-calling 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }


# ── ToolRegistry ─────────────────────────────────────────────────

class ToolRegistry:
    """单例注册表，收集所有工具的 schema + handler。"""

    def __init__(self):
        self._tools: Dict[str, ToolEntry] = {}
        self._toolset_checks: Dict[str, Callable] = {}
        self._lock = threading.RLock()
        self._generation: int = 0

    # ── 快照（线程安全） ──────────────────────────────────────────

    def _snapshot_entries(self) -> List[ToolEntry]:
        with self._lock:
            return list(self._tools.values())

    # ── 注册 / 注销 ─────────────────────────────────────────────

    def register(
        self,
        name: str,
        description: str = "",
        schema: Dict[str, Any] = None,
        handler: Callable = None,
        toolset: str = "general",
        check_fn: Callable = None,
        requires_env: List[str] = None,
        is_async: bool = False,
        emoji: str = "",
        max_result_size_chars: int | float | None = None,
    ):
        """注册一个工具。模块导入时由各工具文件调用。

        Args:
            name: 工具名称（唯一标识）
            description: 人类可读描述（可选，默认为 ""）
            schema: JSON Schema（参数定义）
            handler: 处理函数 callable(args, **kwargs) → str
            toolset: 工具集名称
            check_fn: 可用性检查函数
            requires_env: 所需环境变量
            is_async: handler 是否为异步函数
            emoji: 图标
            max_result_size_chars: 结果最大字符数
        """
        with self._lock:
            existing = self._tools.get(name)
            if existing and existing.toolset != toolset:
                logger.error(
                    "工具注册被拒绝: '%s' (toolset '%s') 与已有工具 (toolset '%s') 冲突",
                    name, toolset, existing.toolset,
                )
                return
            self._tools[name] = ToolEntry(
                name=name,
                toolset=toolset,
                schema=schema or {},
                handler=handler,
                check_fn=check_fn,
                requires_env=requires_env or [],
                is_async=is_async,
                description=description,
                emoji=emoji,
                max_result_size_chars=max_result_size_chars,
            )
            if check_fn and toolset not in self._toolset_checks:
                self._toolset_checks[toolset] = check_fn
            self._generation += 1
        logger.info("Registered tool: %s (toolset=%s)", name, toolset)

    def deregister(self, name: str) -> None:
        """移除工具。MCP 动态发现时使用。"""
        with self._lock:
            entry = self._tools.pop(name, None)
            if entry is None:
                return
            toolset_still_exists = any(
                e.toolset == entry.toolset for e in self._tools.values()
            )
            if not toolset_still_exists:
                self._toolset_checks.pop(entry.toolset, None)
            self._generation += 1
        logger.debug("Deregistered tool: %s", name)

    # ── 查询 ────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[ToolEntry]:
        with self._lock:
            return self._tools.get(name)

    def list_all(self) -> List[ToolEntry]:
        return self._snapshot_entries()

    def list_available(self) -> List[ToolEntry]:
        return [t for t in self._snapshot_entries() if t.is_available()]

    def get_all_tool_names(self) -> List[str]:
        return sorted(entry.name for entry in self._snapshot_entries())

    def get_tool_names(self) -> List[str]:
        """获取所有已注册工具名称（别名：兼容旧代码）。"""
        return list(self._tools.keys())

    def get_tools_by_toolset(self, toolset: str) -> List[ToolEntry]:
        return [t for t in self._snapshot_entries() if t.toolset == toolset]

    # ── Schema ──────────────────────────────────────────────────

    def get_schemas(self) -> List[Dict[str, Any]]:
        """返回所有可用工具的 OpenAI schema 列表（带缓存）。"""
        from services.cache_manager import get_cache
        cache = get_cache()
        
        # 用 generation 作为缓存版本号
        cache_key = f"tool_schemas:v{self._generation}"
        cached = cache.memory.get(cache_key)
        if cached is not None:
            return cached
        
        schemas = [t.to_openai_schema() for t in self.list_available()]
        cache.memory.set(cache_key, schemas)
        return schemas

    def get_definitions(self, tool_names: Set[str]) -> List[dict]:
        """返回指定工具的 OpenAI schema（带 check_fn 过滤）。"""
        result = []
        entries_by_name = {entry.name: entry for entry in self._snapshot_entries()}
        for name in sorted(tool_names):
            entry = entries_by_name.get(name)
            if not entry:
                continue
            if not entry.is_available():
                continue
            schema_with_name = {**entry.schema, "name": entry.name}
            result.append({"type": "function", "function": schema_with_name})
        return result

    def get_schema(self, name: str) -> Optional[dict]:
        entry = self.get(name)
        return entry.schema if entry else None

    # ── 调度 ────────────────────────────────────────────────────

    def dispatch(self, name: str, args: dict, **kwargs) -> str:
        """执行工具 handler（hermes-agent 兼容接口）。

        自动处理异步 handler、捕获异常、返回 JSON 错误。
        """
        entry = self.get(name)
        if not entry:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            result = entry.handler(args, **kwargs)
            return str(result)
        except Exception as e:
            logger.exception("Tool %s dispatch error: %s", name, e)
            return json.dumps({"error": f"Tool execution failed: {type(e).__name__}: {e}"})

    async def execute(self, name: str, params: Dict[str, Any]) -> str:
        """异步执行工具（兼容旧代码）。"""
        entry = self._tools.get(name)
        if entry is None:
            raise ValueError(f"Tool not found: {name}")
        if not entry.is_available():
            raise RuntimeError(f"Tool {name} is not available")
        try:
            result = entry.handler(**params)
            if hasattr(result, "__await__"):
                result = await result
            return str(result)
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e, exc_info=True)
            return f"Error executing {name}: {e}"

    # ── Toolset ──────────────────────────────────────────────────

    def get_registered_toolset_names(self) -> List[str]:
        return sorted({entry.toolset for entry in self._snapshot_entries()})

    def get_tool_names_for_toolset(self, toolset: str) -> List[str]:
        return sorted(
            entry.name for entry in self._snapshot_entries()
            if entry.toolset == toolset
        )

    # ── 向后兼容属性 ────────────────────────────────────────────

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation


# ════════════════════════════════════════════════════════════════
# 模块级单例
# ════════════════════════════════════════════════════════════════

_registry = ToolRegistry()

def get_registry() -> ToolRegistry:
    """获取全局注册表单例（兼容旧代码模式）。"""
    return _registry


# Module-level singleton alias — hermes-agent 兼容
# 支持两种 import 模式:
#   A 模式: from tools.registry import get_registry; registry = get_registry()
#   B 模式: from tools.registry import registry; registry.register(...)
registry = _registry


# ════════════════════════════════════════════════════════════════
# 便捷函数（兼容旧代码）
# ════════════════════════════════════════════════════════════════

def get_tool(name: str) -> Optional[ToolEntry]:
    return registry.get(name)


def get_tool_names() -> List[str]:
    return registry.get_tool_names()


def get_schemas() -> List[Dict[str, Any]]:
    return registry.get_schemas()


# ════════════════════════════════════════════════════════════════
# 工具响应序列化辅助（hermes-agent 兼容）
# ════════════════════════════════════════════════════════════════

def tool_error(message: str, **extra) -> str:
    """返回 JSON 格式的错误字符串。"""
    result = {"error": str(message)}
    if extra:
        result.update(extra)
    return json.dumps(result, ensure_ascii=False)


def tool_result(data=None, **kwargs) -> str:
    """返回 JSON 格式的结果字符串。"""
    if data is not None:
        return json.dumps(data, ensure_ascii=False)
    return json.dumps(kwargs, ensure_ascii=False)


# ════════════════════════════════════════════════════════════════
# 工具发现（AST 自动发现）
# ════════════════════════════════════════════════════════════════

def discover_tools(tools_dir: Optional[str] = None) -> List[str]:
    """AST 自动发现：扫描 tools/ 目录，导入所有包含 registry.register() 的模块。

    相比旧版 filename-based 发现（builtin_*.py），AST 检测更灵活：
    - 任何 .py 文件只要调用了 registry.register() 就会被发现
    - 排除 __init__.py、registry.py、mcp_tool.py

    Args:
        tools_dir: 工具目录路径，默认当前文件所在目录。

    Returns:
        已导入的模块名称列表。
    """
    if tools_dir is None:
        tools_dir = str(Path(__file__).resolve().parent)

    tools_path = Path(tools_dir)
    module_names = [
        f"tools.{path.stem}"
        for path in sorted(tools_path.glob("*.py"))
        if path.name not in {"__init__.py", "registry.py", "mcp_tool.py", "process_registry.py"}
        and _module_registers_tools(path)
    ]

    imported: List[str] = []
    for mod_name in module_names:
        try:
            importlib.import_module(mod_name)
            imported.append(mod_name)
        except Exception as e:
            logger.warning("Could not import tool module %s: %s", mod_name, e)

    # 发现并注册 MCP 服务器工具
    try:
        from tools.mcp_tool import discover_mcp_tools
        mcp_tools = discover_mcp_tools()
        if mcp_tools:
            logger.info("MCP 工具已注册: %s", ", ".join(mcp_tools))
            imported.append("tools.mcp_tool")
    except Exception as e:
        logger.debug("MCP 工具发现失败: %s", e)

    return imported


def discover_builtin_tools(tools_dir: Optional[str] = None) -> List[str]:
    """hermes-agent 兼容别名。"""
    return discover_tools(tools_dir)