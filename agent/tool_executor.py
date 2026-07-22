"""
并发工具执行器 - 从 hermes-agent tool_executor.py 精简移植
为本地模型场景优化：更保守的并行策略、更短的超时
"""
from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.registry import get_registry, tool_error

log = logging.getLogger(__name__)

# 本地模型场景下的保守值,避免过度并行导致超时
DEFAULT_MAX_WORKERS = 4

# 只读类工具列表 — 始终可以并行
PARALLEL_SAFE_TOOLS = frozenset({
    "read_file", "list_directory", "glob", "grep",
    "web_search", "web_fetch", "search",
})

# 冲突敏感工具 — 如果两个工具的路径参数重叠,降级为顺序执行
PATH_SENSITIVE_TOOLS = frozenset({
    "write_file", "edit_file", "delete_file", "move_file",
    "execute_command", "bash",
})


def _should_parallelize(tool_calls: List[Dict], max_workers: int = DEFAULT_MAX_WORKERS) -> bool:
    """判断工具批次是否应并发执行"""
    if len(tool_calls) < 2:
        return False

    # 检查是否有冲突敏感的路径操作
    path_targets = []
    for tc in tool_calls:
        fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
        if fn is None:
            continue
        name = fn.get("name", "") if isinstance(fn, dict) else fn.name
        if name in PATH_SENSITIVE_TOOLS:
            args = fn.get("arguments", {}) if isinstance(fn, dict) else getattr(fn, "arguments", {})
            if isinstance(args, str):
                try:
                    import json
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            path = args.get("path") or args.get("file_path") or args.get("target")
            if path:
                path_targets.append((name, str(path)))

    # 检测路径重叠
    for i, (name1, p1) in enumerate(path_targets):
        for name2, p2 in path_targets[i + 1:]:
            if p1 == p2 or p1.startswith(p2) or p2.startswith(p1):
                log.debug("Tool conflict: %s(%s) vs %s(%s) — sequential", name1, p1, name2, p2)
                return False

    return True


def _execute_one_sync(tc: Dict) -> Tuple[Dict, str]:
    """同步执行单个工具调用,用于 ThreadPoolExecutor"""
    import json as _json

    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
    if not fn and hasattr(tc, "function"):
        fn = tc.function
    if not fn and hasattr(tc, "name"):
        # 直接字段格式
        name = tc.get("name", "")
        args = tc.get("arguments", {})
        tool_id = tc.get("id", "")
    else:
        name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
        raw_args = fn.get("arguments", {}) if isinstance(fn, dict) else getattr(fn, "arguments", {})
        if isinstance(raw_args, str):
            try:
                args = _json.loads(raw_args)
            except (_json.JSONDecodeError, TypeError):
                args = {}
        else:
            args = raw_args
        tool_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")

    try:
        registry = get_registry()
        result = registry.execute_sync(name, args)
        return (tc, result)
    except Exception as exc:
        log.error("Tool '%s' failed in concurrent execution: %s", name, exc)
        return (tc, tool_error(f"[{name}] {exc}"))


def execute_tools_concurrent(
    tool_calls: List[Dict],
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout: float = 60.0,
) -> List[Tuple[Dict, str]]:
    """
    并发执行多个工具调用

    Args:
        tool_calls: 工具调用列表
        max_workers: 最大并发数
        timeout: 单个工具超时秒数

    Returns:
        [(tool_call_dict, result_string), ...]  与输入顺序一致
    """
    if len(tool_calls) <= 1 or not _should_parallelize(tool_calls):
        # 单工具或不应并行 → 顺序执行
        results = []
        for tc in tool_calls:
            r = _execute_one_sync(tc)
            results.append(r)
        return results

    n = min(len(tool_calls), max_workers)
    log.info("⚡ Concurrent: %d tool calls (%d workers)", len(tool_calls), n)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        future_to_tc = {executor.submit(_execute_one_sync, tc): tc for tc in tool_calls}

        try:
            done, not_done = concurrent.futures.wait(
                future_to_tc, timeout=timeout, return_when=concurrent.futures.ALL_COMPLETED
            )
        except KeyboardInterrupt:
            for f in future_to_tc:
                f.cancel()
            raise

        # 收集成功的结果
        for f in done:
            try:
                tc, result = f.result()
                results.append((tc, result))
            except Exception as exc:
                tc = future_to_tc[f]
                results.append((tc, tool_error(f"Concurrent execution failed: {exc}")))

        # 超时的工具
        for f in not_done:
            f.cancel()
            tc = future_to_tc[f]
            fn_name = tc.get("function", {}).get("name", "unknown") if isinstance(tc, dict) else "unknown"
            results.append((tc, tool_error(f"Tool '{fn_name}' timed out after {timeout}s")))

    # 按输入顺序排序
    tc_index = {id(tc): i for i, tc in enumerate(tool_calls)}
    results.sort(key=lambda x: tc_index.get(id(x[0]), 999))
    return results


def execute_tools_sequential(tool_calls: List[Dict]) -> List[Tuple[Dict, str]]:
    """顺序执行工具调用 (fallback)"""
    results = []
    for tc in tool_calls:
        r = _execute_one_sync(tc)
        results.append(r)
    return results
