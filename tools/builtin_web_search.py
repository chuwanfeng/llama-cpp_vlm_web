"""联网搜索工具 — 实时信息检索（ProSearch 后端）。

核心设计：
    - 使用腾讯 ProSearch 引擎（国内可达）
    - 支持 VR 卡数据：汇率、金价、天气等权威数据
    - mode=0: 自然检索 / mode=1: VR 卡 / mode=2: 混合
    - 无需 API Key，走网关鉴权
    - 时效性查询自动加 --freshness 参数

依赖：
    - Node.js（用于调用 prosearch.cjs 脚本）
    - 环境变量 AUTH_GATEWAY_PORT（默认 19000）
"""

import json
import logging
import os
import subprocess
from typing import Any

from tools.registry import get_registry, tool_result

logger = logging.getLogger(__name__)

# ProSearch 脚本路径 — 优先查找项目内置脚本，其次查找 OpenClaw skill 目录
import pathlib
_PROSEARCH_SCRIPT = str(pathlib.Path(__file__).resolve().parent.parent / "scripts" / "prosearch.cjs")
_PROSEARCH_AVAILABLE = os.path.isfile(_PROSEARCH_SCRIPT)
if not _PROSEARCH_AVAILABLE:
    logger.info("ProSearch 脚本不可用，将使用 cn.bing.com 兜底")


def _build_search_args(keyword: str, max_results: int = 5, freshness: str | None = None) -> list[str]:
    """构建 prosearch 命令行参数。"""
    args = ["node", _PROSEARCH_SCRIPT, "--keyword=" + keyword]
    
    # 数量：prosearch 支持 10/20/30/40/50
    cnt_map = {1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 20, 7: 20, 8: 20, 9: 20, 10: 20}
    cnt = cnt_map.get(max_results, 20) if max_results <= 10 else 20
    
    # 时效性查询：自动加 freshness
    if freshness:
        args.append(f"--freshness={freshness}")
    else:
        # 检测时效性关键词，自动加 freshness
        if any(kw in keyword for kw in ["今日", "今天", "现在", "当前", "实时"]):
            args.append("--freshness=24h")
        elif any(kw in keyword for kw in ["最新", "最近", "今日", "现在"]):
            args.append("--freshness=7d")
    
    # 金融类查询：使用 VR 卡模式
    finance_keywords = ["汇率", "美元", "人民币", "黄金", "白银", "油价", "股价", "汇率"]
    is_finance = any(k in keyword for k in finance_keywords)
    
    return args, is_finance


def _run_prosearch(keyword: str, max_results: int = 5) -> dict[str, Any]:
    """调用 prosearch.cjs，返回解析后的 JSON 结果。"""
    args, is_finance = _build_search_args(keyword, max_results)
    
    # 金融类查询用 mode=2（VR + 自然检索混合）
    if is_finance:
        args.append("--mode=2")
    
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            timeout=20,
            encoding="utf-8",
            errors="replace",
        )
        
        if result.returncode != 0:
            logger.warning("ProSearch failed (code %d): %s", result.returncode, result.stderr[:200])
            return {"success": False, "message": f"搜索服务异常: {result.stderr[:200]}"}
        
        output = result.stdout.strip()
        if not output:
            return {"success": False, "message": "搜索服务返回空响应"}
        
        try:
            data = json.loads(output)
            # 兜底：ProSearch 可能返回 success=false 但不带 message 字段
            if not data.get("success") and "message" not in data:
                data["message"] = f"搜索未返回结果（原始输出：{output[:300]}）"
            return data
        except json.JSONDecodeError:
            logger.warning("ProSearch returned non-JSON: %s", output[:200])
            return {"success": False, "message": f"搜索服务响应格式错误: {output[:200]}"}
            
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "搜索超时（20s），请稍后重试"}
    except FileNotFoundError:
        return {"success": False, "message": "Node.js 未安装，无法执行联网搜索"}
    except Exception as e:
        logger.warning("ProSearch exception: %s", e)
        return {"success": False, "message": f"搜索异常: {e}"}


def _duckduckgo_fallback(query: str, max_results: int = 5) -> str:
    """ProSearch 失败时的通用搜索兜底。

    使用 services.search 模块的 search() 统一入口：
    cn.bing.com 优先（国内可达），DDG 兜底。
    """
    try:
        from services.search import search as do_search
        results = do_search(query, max_results=max_results)
        if not results:
            return tool_result(f"⚠️ 搜索「{query}」未找到结果（ProSearch 不可用，通用搜索也无结果）")

        lines = [f"搜索「{query}」找到 {len(results)} 条结果（cn.bing.com 兜底）:\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            url = r.get("url", "")
            snippet = r.get("snippet", "")[:200]
            if url:
                lines.append(f"**{i}. [{title}]({url})**")
            else:
                lines.append(f"**{i}. {title}**")
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")
        return tool_result("\n".join(lines))

    except ImportError:
        logger.warning("Search fallback unavailable: services.search module not found")
        return tool_result("⚠️ 搜索服务暂时不可用（ProSearch 失败，通用搜索模块加载失败）")
    except Exception as e:
        logger.warning("Search fallback failed: %s", e)
        return tool_result(f"⚠️ 搜索失败: ProSearch 不可用，通用搜索兜底异常: {e}")


def web_search(query: str, max_results: int = 5) -> str:
    """联网搜索 — 使用腾讯 ProSearch 引擎。

    Args:
        query: 搜索关键词（简洁，2-6 词，效果最佳）
        max_results: 最大结果数（1-10，默认 5）

    Returns:
        格式化后的搜索结果字符串（含可点击链接）
    """
    max_results = min(max(1, max_results), 10)
    
    # 空关键词检查
    if not query or not query.strip():
        return tool_result("错误: 搜索关键词不能为空")
    
    query = query.strip()
    
    # ProSearch 不可用时直接走通用搜索兜底（跳过无意义的 subprocess 调用）
    if not _PROSEARCH_AVAILABLE:
        return _duckduckgo_fallback(query, max_results)
    
    result = _run_prosearch(query, max_results)
    
    if not result.get("success"):
        # ProSearch 失败时自动回退到 DuckDuckGo
        logger.info("ProSearch failed, falling back to DuckDuckGo: %s",
                     result.get("message", "")[:80])
        return _duckduckgo_fallback(query, max_results)
    
    # 返回 message 字段（原样输出，包含格式化结果）
    message = result.get("message", "")
    if message:
        return tool_result(message)
    
    # fallback: 从 data.docs 构造
    docs = result.get("data", {}).get("docs", [])
    if not docs:
        return tool_result(f"搜索「{query}」没有找到结果")
    
    lines = [f"搜索结果（{len(docs)} 条）:\n"]
    for i, doc in enumerate(docs[:max_results], 1):
        title = doc.get("title", "无标题")
        url = doc.get("url", "")
        snippet = doc.get("passage", "")[:200]
        site = doc.get("site", "")
        date = doc.get("date", "")
        
        if url:
            lines.append(f"**{i}. [{title}]({url})**")
        else:
            lines.append(f"**{i}. {title}**")
        if site:
            lines.append(f"   来源: {site}" + (f" ({date})" if date else ""))
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")
    
    return tool_result("\n".join(lines))


# ── Register ─────────────────────────────────────────────────────────────────

registry = get_registry()
registry.register(
    name="web_search",
    description="Search the web for real-time information. Use when you need current data, facts, prices, news, or anything beyond your training cutoff. Supports financial data (exchange rates, gold price) via VR cards.",
    schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string (use concise keywords, 2-6 words, not full sentences). For financial data like exchange rates, use Chinese like '1美元兑换人民币汇率'.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (1-10, default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    handler=web_search,
    toolset="web",
)