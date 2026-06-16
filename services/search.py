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


# ── 天气查询检测 ──────────────────────────────────────────────────────────────

_WEATHER_KEYWORDS = ["天气", "气温", "温度", "降雨", "降水", "风力", "湿度", "雾霾", "台风", "空气质量"]


def _is_weather_query(query: str) -> bool:
    """检测是否为天气类查询。"""
    return any(kw in query for kw in _WEATHER_KEYWORDS)


def _refine_weather_query(query: str) -> str:
    """优化天气查询词：提取地名，补全 '天气预报' 以触发 Bing 天气卡片。"""
    import re as _re
    # 台风类特殊处理：保留核心关键词
    if "台风" in query:
        cleaned = _re.sub(r"(今天|明天|后天|现在|当前|实时|今日)", "", query).strip()
        if not cleaned:
            cleaned = "台风路径"
        return cleaned
    # 先去掉末尾已有的 "天气预报"，避免后续清洗时被误拆
    query = _re.sub(r"天气预报$", "", query)
    # 去掉时间词和天气关键词，只留地名
    time_words = r"(今天|明天|后天|现在|当前|实时|今日|明日|未来"
    weather_words = "|".join(kw for kw in _WEATHER_KEYWORDS if kw != "台风")
    noise_pattern = _re.compile(time_words + "|" + weather_words + r"|的|吗|了|怎么样|如何)")
    cleaned = noise_pattern.sub("", query).strip()
    if not cleaned:
        cleaned = query  # 兜底：别洗空了
    return cleaned + "天气预报"


def search_bing(query):
    """Bing 搜索（cn.bing.com）- 天气查询自动优化。"""
    log.info("Bing 搜索: %s", query)
    actual_query = _refine_weather_query(query) if _is_weather_query(query) else query
    if actual_query != query:
        log.info("Bing 天气查询优化: %s -> %s", query, actual_query)
    url = f"https://cn.bing.com/search?q={quote(actual_query)}&setlang=zh-cn&mkt=zh-CN"
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
            # 解析标题和URL: h2内多<a>标签, 跳过r.bing.com/Javascript跟踪链接
            h2_block = re.search(r'<h2[^>]*>(.*?)</h2>', block, re.DOTALL)
            if not h2_block:
                continue
            all_as = re.findall(
                r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
                h2_block.group(1), re.DOTALL
            )
            url = ""
            title = ""
            for href, a_title in all_as:
                if 'r.bing.com' in href or 'javascript:' in href:
                    continue
                url = href
                title = re.sub(r"<[^>]+>", "", a_title).strip()
                break
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
        # 百度百科降权：如果结果多于一条且首条是百科，移到末尾
        if len(results) > 1 and "baike.baidu.com" in results[0]["url"]:
            results.append(results.pop(0))
        return results
    except Exception as e:
        log.warning("Bing 搜索失败: %s", e)
        return []


def search(query, max_results=5):
    """统一搜索入口：Bing 优先，失败降级 DDG"""
    results = search_bing(query)
    if results:
        return results[:max_results]
    log.info("Bing 无结果，尝试 DDG: %s", query)
    return search_ddg(query)[:max_results]
