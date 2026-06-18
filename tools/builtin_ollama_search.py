"""Ollama 联网搜索/网页抓取 — 使用 Ollama Cloud API。

核心设计：
    - web_search: 调用 Ollama /api/web_search，返回 title + url + snippet
    - web_fetch: 调用 Ollama /api/web_fetch，返回标题 + 正文 + 链接列表
    - API Key 来源：环境变量 OLLAMA_API_KEY → 设置文件 vendor_creds.ollama-cloud.api_key
    - 自动回退到内置 ProSearch（web_search 失败时）
"""

import json
import logging
import os
from typing import Any, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from tools.registry import get_registry, tool_result

logger = logging.getLogger(__name__)


# ── API Key 读取 ────────────────────────────────────────────────────────────

def _get_ollama_api_key() -> str:
    """获取 Ollama API Key（环境变量优先，其次是设置文件）"""
    # 环境变量优先
    for env_key in ("OLLAMA_API_KEY", "OLLAMA_CLOUD_KEY"):
        val = os.environ.get(env_key, "")
        if val:
            return val
    # 回退到设置文件
    try:
        from utils import read_json
        settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "settings.json")
        settings = read_json(settings_path, default={})
        creds = settings.get("vendor_creds", {})
        ollama_creds = creds.get("ollama-cloud", {}) or creds.get("ollama_cloud", {})
        return ollama_creds.get("api_key", "")
    except Exception:
        return ""


OLLAMA_API_BASE = "https://ollama.com/api"


def _ollama_request(endpoint: str, body: dict, timeout: int = 15) -> Optional[dict]:
    """通用的 Ollama API JSON POST 请求"""
    api_key = _get_ollama_api_key()
    if not api_key:
        return {"success": False, "message": "Ollama API Key 未配置（请设置环境变量 OLLAMA_API_KEY 或在设置界面输入）"}

    url = f"{OLLAMA_API_BASE}/{endpoint}"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")

    req = Request(url, data=data)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")

    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body_snippet = e.read().decode("utf-8", errors="replace")[:300]
        logger.warning("Ollama %s HTTP %d: %s", endpoint, e.code, body_snippet)
        return {"success": False, "message": f"Ollama {endpoint} 请求失败 (HTTP {e.code})"}
    except URLError as e:
        logger.warning("Ollama %s URL error: %s", endpoint, e)
        return {"success": False, "message": f"Ollama {endpoint} 网络异常: {e.reason}"}
    except Exception as e:
        logger.warning("Ollama %s error: %s", endpoint, e)
        return {"success": False, "message": f"Ollama {endpoint} 异常: {e}"}


# ── 工具实现 ─────────────────────────────────────────────────────────────────

def ollama_web_search(query: str, max_results: int = 5) -> str:
    """Ollama Web Search — 使用 Ollama Cloud 的搜索 API。

    Args:
        query: 搜索关键词
        max_results: 最大结果数（1-10，默认 5）

    Returns:
        格式化搜索结果字符串
    """
    max_results = min(max(1, int(max_results)), 10)

    if not query or not query.strip():
        return tool_result("错误: 搜索关键词不能为空")

    query = query.strip()
    result = _ollama_request("web_search", {"query": query, "max_results": max_results})

    if not result or not isinstance(result, dict):
        return tool_result("搜索服务异常：未收到有效响应")

    if "success" in result and not result["success"]:
        # 回退到内置 ProSearch
        return _fallback_search(query, max_results, result.get("message", ""))

    items = result.get("results", [])
    if not items:
        return tool_result(f"搜索「{query}」没有找到结果")

    lines = [f"🔍 搜索「{query}」（来源: Ollama Cloud，共 {len(items)} 条）:\n"]
    for i, item in enumerate(items, 1):
        title = item.get("title", "无标题")
        url = item.get("url", "")
        content = item.get("content", "")[:300]
        if url:
            lines.append(f"**{i}. [{title}]({url})**")
        else:
            lines.append(f"**{i}. {title}**")
        if content:
            lines.append(f"   {content}")
        lines.append("")
    return tool_result("\n".join(lines))


def ollama_web_fetch(url: str) -> str:
    """Ollama Web Fetch — 使用 Ollama Cloud 抓取网页内容。

    Args:
        url: 要抓取的网页 URL

    Returns:
        格式化的网页内容（标题 + 正文 + 链接列表）
    """
    if not url or not url.strip():
        return tool_result("错误: URL 不能为空")

    url = url.strip()
    result = _ollama_request("web_fetch", {"url": url})

    if not result or not isinstance(result, dict):
        return tool_result("网页抓取异常：未收到有效响应")

    if "success" in result and not result["success"]:
        return tool_result(f"⚠️ 抓取失败: {result.get('message', '未知错误')}")

    lines = [f"📄 {result.get('title', url)}"]
    content = result.get("content", "")
    if content:
        lines.append("")
        lines.append(content[:3000])  # 截断到 3000 字符
    else:
        lines.append("（无正文内容）")

    links = result.get("links", [])
    if links:
        lines.append(f"\n---\n🔗 页面链接 ({len(links)} 个):")
        for link in links[:20]:
            lines.append(f"  - {link}")

    return tool_result("\n".join(lines))


def _fallback_search(query: str, max_results: int, error_msg: str = "") -> str:
    """Ollama 搜索失败时回退到 ProSearch"""
    logger.info("Ollama web_search failed, falling back to ProSearch: %s", error_msg[:80])
    try:
        from tools.builtin_web_search import web_search as _prosearch
        return _prosearch(query, max_results)
    except ImportError:
        return tool_result(f"⚠️ Ollama 搜索和 ProSearch 均不可用: {error_msg}")
    except Exception as e:
        return tool_result(f"⚠️ 搜索全通道失败: Ollama={error_msg}, ProSearch={e}")


# ── Register ─────────────────────────────────────────────────────────────────

registry = get_registry()
registry.register(
    name="ollama_web_search",
    description="Search the web using Ollama Cloud API. Returns recent web results with titles, URLs, and snippets. Use for real-time information, news, facts, and current events.",
    schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query. Use concise keywords (2-6 words) for best results.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (1-10, default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    handler=ollama_web_search,
    toolset="web",
)

registry.register(
    name="ollama_web_fetch",
    description="Fetch and extract content from a web page using Ollama Cloud API. Returns the page title, main content, and list of links. Use to read articles, documentation, or any web page content.",
    schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL of the web page to fetch (e.g., https://example.com/article).",
            },
        },
        "required": ["url"],
    },
    handler=ollama_web_fetch,
    toolset="web",
)
