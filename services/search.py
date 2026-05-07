"""
Web 搜索 — DuckDuckGo + Bing 备选
"""
import logging
import re
import html as _html
from urllib.parse import quote

import requests

log = logging.getLogger("llm-web")


def search_ddg(query):
    """DuckDuckGo HTML 搜索"""
    url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 200:
            return []
        results = []
        pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)">([^<]+)</a>.*?<a class="result__snippet"[^>]*>([^<]+(?:<[^>]+>[^<]*</[^>]+>)*?)</a>'
        matches = re.findall(pattern, resp.text, re.DOTALL)
        for url, title, snippet in matches[:5]:
            snippet = re.sub(r"<[^>]+>", "", snippet)
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet[:200] + "..." if len(snippet) > 200 else snippet,
            })
        return results
    except Exception as e:
        log.warning("DDG 搜索失败: %s", e)
        return []


def search_bing(query):
    """Bing 搜索（cn.bing.com）"""
    log.info("Bing 搜索: %s", query)
    url = f"https://cn.bing.com/search?q={quote(query)}&setlang=zh-cn"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return []
        results = []
        blocks = re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', resp.text, re.DOTALL)
        for block in blocks:
            h2m = re.search(r'<h2[^>]*>\s*<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
            if not h2m:
                continue
            url = h2m.group(1)
            title = re.sub(r"<[^>]+>", "", h2m.group(2)).strip()
            if not url or not title:
                continue
            snippet = title
            cap = re.search(r'<div class="b_caption"[^>]*>(.*?)</div>', block, re.DOTALL)
            if cap:
                pm = re.search(r"<p[^>]*>(.*?)</p>", cap.group(1), re.DOTALL)
                if pm:
                    snippet = _html.unescape(re.sub(r"<[^>]+>", "", pm.group(1)).strip())
                    snippet = snippet[:200] + "..." if len(snippet) > 200 else snippet
            if not any(r["url"] == url for r in results):
                results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= 5:
                break
        return results
    except Exception as e:
        log.warning("Bing 搜索失败: %s", e)
        return []
