"""
浏览器工具 — 网页读取与分析（Playwright 增强版）

移植自 hermes-agent/tools/browser_tool.py，适配 Web 场景。

核心功能：
    1. browser_navigate(url, use_playwright) — 访问网页，返回可读文本
    2. browser_extract_links(url) — 提取页面所有链接
    3. browser_read_page(url, selector) — 提取页面指定区域内容
    4. browser_search(query) — 调用项目已有搜索服务
    5. browser_screenshot(url, selector) — 截图（Playwright）
    6. browser_click(url, selector) — 点击元素（Playwright）
    7. browser_fill_form(url, fields) — 填写表单（Playwright）

依赖：
    - requests：HTTP 请求（已安装 ✅）
    - playwright：浏览器自动化（已安装 ✅）
    - 项目 services/search.py：联网搜索（已有 ✅）

设计说明：
    - 优先使用 Playwright（支持 JS 渲染、截图、交互）
    - Playwright 不可用时回退到 requests + 正则解析
    - 返回 Markdown 格式文本（方便 LLM 阅读）
    - 自动处理编码问题（UTF-8 / GBK / Latin-1）

安全：
    - 限制请求超时（默认 15 秒）
    - 限制响应大小（默认 5MB）
    - 禁止访问内网地址（127.0.0.1 / 192.168.* / 10.*）
    - User-Agent 伪装
"""

import json
import logging
import re
import base64
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urljoin

from tools.registry import get_registry

logger = logging.getLogger(__name__)

# ── 常量 ─────────────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT = 15  # 秒
MAX_RESPONSE_SIZE = 5 * 1024 * 1024  # 5MB
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# 禁止访问的内网地址正则
LOCAL_PATTERNS = [
    r"^https?://(127\.|0\.0\.1|localhost)(:\d+)?/",
    r"^https?://(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.)/",
]

# ── Playwright 可用性检测 ───────────────────────────────────────────────────

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright 未安装，浏览器工具将使用 requests 回退模式")

# ── 辅助函数 ─────────────────────────────────────────────────────────────────

def _is_local_url(url: str) -> bool:
    """检查 URL 是否指向内网地址。"""
    for pattern in LOCAL_PATTERNS:
        if re.match(pattern, url):
            return True
    return False

def _get_safe_headers() -> Dict[str, str]:
    """返回安全的请求头（伪装浏览器）。"""
    return {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

def _html_to_markdown(html: str, base_url: str = "") -> str:
    """将 HTML 转为 Markdown（简化版，不依赖外部库）。"""
    # 移除 script 和 style 标签
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # 移除注释
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    # 转换标题
    for i in range(6, 0, -1):
        html = re.sub(rf"<h{i}[^>]*>(.*?)</h{i}>", lambda m: "\n" + "#" * i + " " + m.group(1).strip() + "\n", html, flags=re.DOTALL | re.IGNORECASE)
    # 转换链接
    html = re.sub(r'<a[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', lambda m: f"[{m.group(2).strip()}]({urljoin(base_url, m.group(1))})", html, flags=re.DOTALL | re.IGNORECASE)
    # 转换段落
    html = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\1\n", html, flags=re.DOTALL | re.IGNORECASE)
    # 转换换行
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"</(div|li|ul|ol|blockquote)[\s>]", "\n", html, flags=re.IGNORECASE)
    # 移除所有剩余的 HTML 标签
    html = re.sub(r"<[^>]+>", "", html)
    # 解码 HTML 实体
    html = html.replace("&nbsp;", " ")
    html = html.replace("&lt;", "<")
    html = html.replace("&gt;", ">")
    html = html.replace("&amp;", "&")
    html = html.replace("&quot;", "\"")
    html = html.replace("&#39;", "'")
    # 清理多余空行
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()

def _extract_links(html: str, base_url: str) -> List[Dict[str, str]]:
    """从 HTML 中提取所有链接。"""
    links = []
    for match in re.finditer(r'<a[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.DOTALL | re.IGNORECASE):
        href = match.group(1).strip()
        text = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        full_url = urljoin(base_url, href)
        if not _is_local_url(full_url):
            links.append({"url": full_url, "text": text[:100]})
    return links[:50]

# ── requests 回退抓取 ───────────────────────────────────────────────────────

def _fetch_page_requests(url: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """使用 requests 抓取网页（Playwright 不可用时的回退）。"""
    import requests
    if _is_local_url(url):
        return {"success": False, "error": f"禁止访问内网地址: {url}"}
    try:
        resp = requests.get(url, headers=_get_safe_headers(), timeout=timeout, allow_redirects=True, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if not content_type.startswith("text/html"):
            return {"success": False, "error": f"非 HTML 内容（Content-Type: {content_type}）"}
        content = b""
        for chunk in resp.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > MAX_RESPONSE_SIZE:
                return {"success": False, "error": f"页面过大（> {MAX_RESPONSE_SIZE // 1024 // 1024}MB）"}
        encoding = resp.encoding or "utf-8"
        try:
            html = content.decode(encoding, errors="replace")
        except Exception:
            html = content.decode("utf-8", errors="replace")
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""
        text = _html_to_markdown(html, url)
        return {"success": True, "html": html[:1000], "text": text, "title": title, "status_code": resp.status_code}
    except requests.exceptions.Timeout:
        return {"success": False, "error": f"请求超时（{timeout} 秒）"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"请求失败: {type(e).__name__}: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"未知错误: {type(e).__name__}: {str(e)}"}

# ── Playwright 抓取 ─────────────────────────────────────────────────────────

def _fetch_page_playwright(url: str, timeout: int = DEFAULT_TIMEOUT, wait_for: str = "") -> Dict[str, Any]:
    """使用 Playwright 抓取网页（支持 JS 渲染）。"""
    if not PLAYWRIGHT_AVAILABLE:
        return {"success": False, "error": "Playwright 未安装", "fallback": True}
    if _is_local_url(url):
        return {"success": False, "error": f"禁止访问内网地址: {url}"}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=USER_AGENT)
            page = context.new_page()
            page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
            if wait_for:
                try:
                    page.wait_for_selector(wait_for, timeout=5000)
                except PWTimeoutError:
                    pass  # 选择器未出现，继续
            title = page.title()
            html = page.content()
            text = _html_to_markdown(html, url)
            browser.close()
            return {"success": True, "html": html[:1000], "text": text, "title": title, "status_code": 200}
    except PWTimeoutError:
        return {"success": False, "error": f"页面加载超时（{timeout} 秒）"}
    except Exception as e:
        return {"success": False, "error": f"Playwright 错误: {type(e).__name__}: {str(e)}"}

# ── 工具实现 ─────────────────────────────────────────────────────────────────

def browser_navigate(url: str, timeout: int = DEFAULT_TIMEOUT, use_playwright: bool = True) -> str:
    """访问网页并返回可读文本内容。

    参数（JSON 字符串）：
        url: 目标网页 URL
        timeout: 请求超时时间（秒，默认 15）
        use_playwright: 是否使用 Playwright（支持 JS 渲染，默认 True）

    返回：
        JSON 字符串，包含 success, text, title, url, error
    """
    if not url:
        return json.dumps({"error": "url 参数必填"})
    if use_playwright and PLAYWRIGHT_AVAILABLE:
        result = _fetch_page_playwright(url, timeout)
        if result.get("fallback"):
            result = _fetch_page_requests(url, timeout)
    else:
        result = _fetch_page_requests(url, timeout)
    if result["success"]:
        return json.dumps({
            "success": True, "url": url, "title": result.get("title", ""),
            "text": result["text"][:10000],
            "note": "内容已截断（最多 10000 字符）" if len(result["text"]) > 10000 else "",
        }, ensure_ascii=False)
    else:
        return json.dumps({"success": False, "url": url, "error": result["error"]}, ensure_ascii=False)

def browser_extract_links(url: str) -> str:
    """提取网页中的所有链接。"""
    if not url:
        return json.dumps({"error": "url 参数必填"})
    result = _fetch_page_requests(url)
    if result["success"]:
        links = _extract_links(result["html"], url)
        return json.dumps({"success": True, "url": url, "links": links, "count": len(links)}, ensure_ascii=False)
    else:
        return json.dumps({"success": False, "url": url, "error": result["error"]}, ensure_ascii=False)

def browser_read_page(url: str, selector: str = "") -> str:
    """读取网页指定区域的内容（CSS 选择器）。

    参数：
        url: 目标网页 URL
        selector: CSS 选择器（例如 "#content" ".main" "article"）

    返回：
        JSON 字符串，包含 success, text, selector, error
    """
    if not url:
        return json.dumps({"error": "url 参数必填"})
    if PLAYWRIGHT_AVAILABLE and selector:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(user_agent=USER_AGENT)
                page = context.new_page()
                page.goto(url, timeout=15000, wait_until="networkidle")
                element = page.query_selector(selector)
                if element:
                    text = element.inner_text()
                    browser.close()
                    return json.dumps({
                        "success": True, "url": url, "selector": selector,
                        "text": text[:10000],
                    }, ensure_ascii=False)
                else:
                    browser.close()
                    return json.dumps({"success": False, "url": url, "selector": selector, "error": "未找到匹配元素"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "url": url, "error": f"Playwright 错误: {type(e).__name__}: {str(e)}"}, ensure_ascii=False)
    # 回退到 requests + 正则
    result = _fetch_page_requests(url)
    if not result["success"]:
        return json.dumps({"success": False, "error": result["error"]}, ensure_ascii=False)
    html = result["html"]
    text = result["text"]
    if selector:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            selected = soup.select_one(selector)
            if selected:
                text = _html_to_markdown(str(selected), url)
        except ImportError:
            logger.warning("BeautifulSoup4 未安装，回退到全文读取")
    return json.dumps({
        "success": True, "url": url, "text": text[:10000], "selector": selector,
        "note": "BeautifulSoup4 未安装，使用了全文" if selector and "bs4" in str(locals().get("ImportError", "")) else "",
    }, ensure_ascii=False)

def browser_search(query: str, max_results: int = 5) -> str:
    """使用项目已有搜索服务进行网页搜索。"""
    if not query:
        return json.dumps({"error": "query 参数必填"})
    try:
        from services.search import search as _search
        results = _search(query, max_results=max_results)
        return json.dumps({"success": True, "query": query, "results": results, "count": len(results)}, ensure_ascii=False)
    except Exception as e:
        logger.error("搜索失败: %s", e)
        return json.dumps({"success": False, "query": query, "error": f"搜索失败: {type(e).__name__}: {str(e)}"}, ensure_ascii=False)

def browser_screenshot(url: str, selector: str = "", full_page: bool = False) -> str:
    """截图网页（Playwright）。

    参数：
        url: 目标网页 URL
        selector: CSS 选择器（可选，截取指定元素）
        full_page: 是否截取整页（默认 False，截取可视区域）

    返回：
        JSON 字符串，包含 success, image_base64, format, error
    """
    if not url:
        return json.dumps({"error": "url 参数必填"})
    if not PLAYWRIGHT_AVAILABLE:
        return json.dumps({
            "success": False,
            "error": "截图功能需要 Playwright。安装命令：pip install playwright && playwright install chromium"
        })
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=USER_AGENT, viewport={"width": 1280, "height": 720})
            page = context.new_page()
            page.goto(url, timeout=30000, wait_until="networkidle")
            if selector:
                element = page.query_selector(selector)
                if element:
                    screenshot_bytes = element.screenshot()
                else:
                    browser.close()
                    return json.dumps({"success": False, "error": f"未找到元素: {selector}"}, ensure_ascii=False)
            else:
                screenshot_bytes = page.screenshot(full_page=full_page)
            browser.close()
            image_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            return json.dumps({
                "success": True, "url": url, "format": "png",
                "image_base64": image_b64,
                "note": f"data:image/png;base64,{image_b64[:50]}..." if len(image_b64) > 50 else "",
            }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": f"截图失败: {type(e).__name__}: {str(e)}"}, ensure_ascii=False)

def browser_click(url: str, selector: str) -> str:
    """点击网页上的元素（Playwright）。

    参数：
        url: 目标网页 URL
        selector: CSS 选择器（例如 "#submit-btn" "button.login"）

    返回：
        JSON 字符串，包含 success, new_url, text, error
    """
    if not url or not selector:
        return json.dumps({"error": "url 和 selector 参数必填"})
    if not PLAYWRIGHT_AVAILABLE:
        return json.dumps({"success": False, "error": "需要 Playwright。安装命令：pip install playwright && playwright install chromium"})
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=USER_AGENT)
            page = context.new_page()
            page.goto(url, timeout=30000, wait_until="networkidle")
            element = page.query_selector(selector)
            if not element:
                browser.close()
                return json.dumps({"success": False, "error": f"未找到元素: {selector}"}, ensure_ascii=False)
            element.click()
            page.wait_for_load_state("networkidle", timeout=10000)
            new_url = page.url
            text = _html_to_markdown(page.content(), new_url)
            browser.close()
            return json.dumps({
                "success": True, "url": url, "new_url": new_url,
                "selector": selector, "text": text[:10000],
            }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": f"点击失败: {type(e).__name__}: {str(e)}"}, ensure_ascii=False)

def browser_fill_form(url: str, fields: dict) -> str:
    """填写网页表单（Playwright）。

    参数：
        url: 目标网页 URL
        fields: 字段字典，键为 CSS 选择器，值为要填写的内容
                例如 {"#username": "admin", "#password": "123456"}

    返回：
        JSON 字符串，包含 success, filled_fields, error
    """
    if not url or not fields:
        return json.dumps({"error": "url 和 fields 参数必填"})
    if not PLAYWRIGHT_AVAILABLE:
        return json.dumps({"success": False, "error": "需要 Playwright。安装命令：pip install playwright && playwright install chromium"})
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=USER_AGENT)
            page = context.new_page()
            page.goto(url, timeout=30000, wait_until="networkidle")
            filled = []
            for selector, value in fields.items():
                element = page.query_selector(selector)
                if element:
                    element.fill(str(value))
                    filled.append(selector)
                else:
                    logger.warning("表单填写: 未找到元素 %s", selector)
            browser.close()
            return json.dumps({
                "success": True, "url": url, "filled_fields": filled,
                "total": len(fields), "filled_count": len(filled),
            }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": f"表单填写失败: {type(e).__name__}: {str(e)}"}, ensure_ascii=False)

# ── OpenAI 工具 Schema ──────────────────────────────────────────────────────

BROWSER_NAVIGATE_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "目标网页 URL"},
        "timeout": {"type": "number", "description": "请求超时时间（秒，默认 15）"},
        "use_playwright": {"type": "boolean", "description": "是否使用 Playwright（支持 JS 渲染，默认 True）"},
    },
    "required": ["url"],
}

BROWSER_EXTRACT_LINKS_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "目标网页 URL"},
    },
    "required": ["url"],
}

BROWSER_READ_PAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "目标网页 URL"},
        "selector": {"type": "string", "description": "CSS 选择器（例如 '#content' '.main' 'article'）"},
    },
    "required": ["url"],
}

BROWSER_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "搜索关键词"},
        "max_results": {"type": "number", "description": "最大结果数（默认 5）"},
    },
    "required": ["query"],
}

BROWSER_SCREENSHOT_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "目标网页 URL"},
        "selector": {"type": "string", "description": "CSS 选择器（可选，截取指定元素）"},
        "full_page": {"type": "boolean", "description": "是否截取整页（默认 False）"},
    },
    "required": ["url"],
}

BROWSER_CLICK_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "目标网页 URL"},
        "selector": {"type": "string", "description": "要点击的元素的 CSS 选择器"},
    },
    "required": ["url", "selector"],
}

BROWSER_FILL_FORM_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "目标网页 URL"},
        "fields": {
            "type": "object",
            "description": "字段字典，键为 CSS 选择器，值为要填写的内容",
        },
    },
    "required": ["url", "fields"],
}

# ── 注册到工具系统 ──────────────────────────────────────────────────────────

registry = get_registry()

registry.register(name="browser_navigate", schema=BROWSER_NAVIGATE_SCHEMA, handler=browser_navigate)
registry.register(name="browser_extract_links", schema=BROWSER_EXTRACT_LINKS_SCHEMA, handler=browser_extract_links)
registry.register(name="browser_read_page", schema=BROWSER_READ_PAGE_SCHEMA, handler=browser_read_page)
registry.register(name="browser_search", schema=BROWSER_SEARCH_SCHEMA, handler=browser_search)
registry.register(name="browser_screenshot", schema=BROWSER_SCREENSHOT_SCHEMA, handler=browser_screenshot)
registry.register(name="browser_click", schema=BROWSER_CLICK_SCHEMA, handler=browser_click)
registry.register(name="browser_fill_form", schema=BROWSER_FILL_FORM_SCHEMA, handler=browser_fill_form)

logger.info("浏览器工具已注册（Playwright=%s）：browser_navigate, browser_extract_links, browser_read_page, browser_search, browser_screenshot, browser_click, browser_fill_form", PLAYWRIGHT_AVAILABLE)
