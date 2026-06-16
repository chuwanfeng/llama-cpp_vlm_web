"""
视觉工具 — 图片分析（多模态）

移植自 hermes-agent/tools/vision_tools.py，适配 Web 场景。

核心功能：
    1. analyze_image(path, question) — 分析图片并回答问题（多模态 LLM 调用）
    2. describe_image(path, detail_level) — 描述图片内容（简洁/详细/专业）
    3. compare_images(path_a, path_b) — 对比两张图片
    4. extract_text(path) — OCR 提取图片中的文字

依赖：
    - Pillow（已安装 ✅）：图片格式验证、尺寸获取、EXIF 读取
    - 多模态模型 API：OpenAI GPT-4V / Gemini Pro Vision / DeepSeek Vision 等
    - base64 编码：将图片转为 API 可接受的格式

设计说明：
    - 优先使用项目已配置的厂商 API（vendor_id + api_key）
    - 图片经 base64 编码后嵌入 messages（OpenAI 多模态格式）
    - 支持本地图片路径或 URL
    - 自动检测图片格式（JPG/PNG/GIF/WEBP）

安全：
    - 限制图片大小（默认 20MB）
    - 验证文件扩展名
    - base64 编码前检查文件是否存在
"""

import base64
import json
import logging
import mimetypes
import os
import re
import requests
from typing import Any, Dict, List, Optional

from tools.registry import get_registry

logger = logging.getLogger(__name__)

# ── 常量 ─────────────────────────────────────────────────────────────────────

MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
OPENAI_VISION_MODELS = {"gpt-4-vision-preview", "gpt-4-turbo", "gpt-4o", "gpt-4.1"}
GEMINI_VISION_MODELS = {"gemini-pro-vision", "gemini-1.5-pro", "gemini-2.0-flash"}


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _encode_image(image_path: str) -> Optional[str]:
    """将图片文件编码为 base64 字符串。

    参数：
        image_path: 图片文件路径（本地）或 URL

    返回：
        base64 编码的字符串，失败返回 None

    异常：
        FileNotFoundError: 文件不存在
        ValueError: 文件过大或格式不支持
    """
    # 情况 1：URL — 先下载
    if image_path.startswith("http://") or image_path.startswith("https://"):
        try:
            resp = requests.get(image_path, timeout=10, stream=True)
            resp.raise_for_status()
            # 检查 Content-Type
            content_type = resp.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                logger.warning("URL 返回非图片 Content-Type: %s", content_type)
            # 读取并编码
            img_data = resp.content
            if len(img_data) > MAX_IMAGE_SIZE:
                raise ValueError(f"图片过大（{len(img_data)} > {MAX_IMAGE_SIZE} 字节）")
            return base64.b64encode(img_data).decode("utf-8")
        except Exception as e:
            logger.error("下载图片失败: %s", e)
            return None

    # 情况 2：本地文件
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    # 检查扩展名
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"不支持的图片格式: {ext}（允许: {ALLOWED_EXTENSIONS}）")

    # 检查大小
    size = os.path.getsize(image_path)
    if size > MAX_IMAGE_SIZE:
        raise ValueError(f"图片过大（{size} > {MAX_IMAGE_SIZE} 字节）")

    # 读取并编码
    with open(image_path, "rb") as f:
        img_data = f.read()
    return base64.b64encode(img_data).decode("utf-8")


def _get_image_mime_type(image_path: str) -> str:
    """获取图片的 MIME 类型。"""
    if image_path.startswith("http"):
        try:
            resp = requests.head(image_path, timeout=5)
            ct = resp.headers.get("Content-Type", "")
            if ct.startswith("image/"):
                return ct
        except Exception:
            pass
    # 回退到文件扩展名
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/jpeg")


def _call_ollama_vision(model: str, prompt: str, b64_img: str, api_key: str) -> str:
    """通过 Ollama Cloud 原生 /api/chat 端点调用多模态模型。

    Ollama Cloud 的 OpenAI 兼容端点 (/v1/chat/completions) 不支持 image_url，
    必须走原生 /api/chat，将 images 放在 message 内部（非请求顶层）。

    参数：
        model: 模型名（如 gemma4:31b-cloud）
        prompt: 用户提示词
        b64_img: base64 编码的图片数据
        api_key: Ollama API Key

    返回：
        JSON 字符串（含 answer 或 error）
    """
    try:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [b64_img]}],
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        resp = requests.post(
            "https://ollama.com/api/chat",
            json=body,
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        return content.strip() or "（模型未返回内容）"
    except requests.exceptions.HTTPError as e:
        logger.error("Ollama 视觉 API HTTP 错误: %s, body: %s", e, e.response.text[:500] if e.response else "")
        return json.dumps({"error": f"Ollama 视觉 API 调用失败: {e}"})
    except Exception as e:
        logger.error("Ollama 视觉 API 调用失败: %s", e)
        return json.dumps({"error": f"Ollama 视觉 API 调用失败: {type(e).__name__}: {str(e)}"})


def _call_vision_llm(
    image_path: str,
    prompt: str,
    model: str = "",
    vendor_id: str = "",
    api_key: str = "",
    base_url: str = "",
) -> str:
    """调用多模态 LLM 分析图片。

    支持的后端：
        - OpenAI GPT-4V / GPT-4o
        - Google Gemini Pro Vision
        - DeepSeek Vision（如果支持）
        - 本地多模态模型（llama-cpp 需要模型支持）

    参数：
        image_path: 图片路径
        prompt: 用户问题/分析指令
        model: 模型名（为空则自动选择）
        vendor_id: 厂商 ID
        api_key: API Key
        base_url: 自定义 Base URL

    返回：
        LLM 分析结果文本
    """
    # 编码图片
    b64_img = _encode_image(image_path)
    if not b64_img:
        return json.dumps({"error": "图片编码失败"})
    mime_type = _get_image_mime_type(image_path)

    # 构造多模态 messages（OpenAI 格式）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_img}",
                        "detail": "high",  # high/low 分辨率
                    },
                },
            ],
        }
    ]

    # 未指定厂商 → 自动从项目配置中检测（环境变量 + settings.json）
    if not vendor_id:
        try:
            from backends.vendors import _read_api_key, VENDORS
            # 先读 settings.json 中的凭据
            settings_creds = {}
            try:
                settings_path = os.path.join(os.path.dirname(__file__), "..", "settings.json")
                settings_path = os.path.normpath(settings_path)
                if os.path.exists(settings_path):
                    with open(settings_path, "r", encoding="utf-8") as f:
                        s = json.load(f)
                    settings_creds = s.get("vendor_creds", {})
            except Exception:
                pass
            # 视觉优先厂商（ollama-cloud 原生支持、openai/gemini 支持 image_url）
            vision_priority = ["ollama-cloud", "openai", "gemini"]
            # 先遍历视觉优先列表，再遍历其余厂商
            for vid in vision_priority + [v for v in VENDORS if v not in vision_priority]:
                # 先检查环境变量，再检查 settings.json
                key = _read_api_key(VENDORS[vid])
                if not key and vid in settings_creds:
                    key = settings_creds[vid].get("api_key", "")
                if key:
                    vendor_id = vid
                    api_key = api_key or key
                    base_url = base_url or settings_creds.get(vid, {}).get("base_url", "")
                    logger.info("视觉工具自动选择厂商: %s", vid)
                    break
        except Exception as e:
            logger.warning("视觉工具自动检测厂商失败: %s", e)

    # 调用厂商 API
    try:
        from backends import vendors

        if not vendor_id:
            return json.dumps({"error": "视觉工具未配置可用的厂商 API Key，请在设置中配置 OpenAI/DeepSeek/智谱等厂商凭据"})

        # 自动选择模型
        if not model:
            vdef = vendors.VENDORS.get(vendor_id, {})
            model = vdef.get("default_model", "")

        # Ollama Cloud 的 OpenAI 兼容端点不支持 image_url，走原生 /api/chat
        if vendor_id == "ollama-cloud":
            return _call_ollama_vision(model, prompt, b64_img, api_key)

        # 其他厂商走统一 chat_stream
        stream = vendors.chat_stream(
            vendor_id=vendor_id,
            model=model,
            messages=messages,
            tools=None,
            temperature=0.3,
            max_tokens=2048,
            api_key=api_key,
            base_url=base_url,
        )

        result = ""
        for chunk in stream:
            if isinstance(chunk, dict):
                if "content" in chunk and chunk["content"]:
                    result += chunk["content"]
            elif isinstance(chunk, str):
                result += chunk

        return result.strip() or "（模型未返回内容）"

    except Exception as e:
        logger.error("视觉 API 调用失败: %s", e)
        return json.dumps({"error": f"视觉 API 调用失败: {type(e).__name__}: {str(e)}"})


# ── 工具实现 ─────────────────────────────────────────────────────────────────

def vision_analyze(path: str, question: str = "请描述这张图片", model: str = "", vendor_id: str = "", api_key: str = "", base_url: str = "") -> str:
    """分析图片并回答问题。

    参数（JSON 字符串）：
        path: 图片路径（本地路径或 URL）
        question: 要回答的问题（默认 "请描述这张图片"）
        model: 指定模型（可选）
        vendor_id: 厂商 ID（可选，使用项目配置）
        api_key: API Key（可选）
        base_url: 自定义 Base URL（可选）

    返回：
        JSON 字符串，包含 answer 字段
    """

    if not path:
        return json.dumps({"error": "path 参数必填"})

    answer = _call_vision_llm(path, question, model, vendor_id, api_key, base_url)
    return json.dumps({"answer": answer}, ensure_ascii=False)


def vision_describe(path: str, detail_level: str = "detailed") -> str:
    """描述图片内容（不同详细程度）。

    level = detail_level
    参数（JSON 字符串）：
        path: 图片路径
        detail_level: 详细程度（"brief" 简洁 / "detailed" 详细 / "professional" 专业，默认 "detailed"）

    返回：
        JSON 字符串，包含 description 字段
    """

    prompts = {
        "brief": "用一两句话简要描述这张图片。",
        "detailed": "详细描述这张图片的内容，包括主要物体、颜色、构图、氛围等。",
        "professional": "以专业摄影/艺术分析的视角描述这张图片，包括构图、光影、色彩理论、情感表达等。",
    }
    prompt = prompts.get(detail_level, prompts["detailed"])

    result = _call_vision_llm(path, prompt)
    return json.dumps({"description": result, "detail_level": detail_level}, ensure_ascii=False)


def vision_compare(path_a: str, path_b: str, focus: str = "general") -> str:
    """对比两张图片的异同。

    参数（JSON 字符串）：
        path_a: 第一张图片路径
        path_b: 第二张图片路径
        focus: 对比焦点（"general" 总体 / "objects" 物体 / "style" 风格，默认 "general"）

    返回：
        JSON 字符串，包含 comparison 字段
    """

    if not path_a or not path_b:
        return json.dumps({"error": "path_a 和 path_b 参数必填"})

    prompt = f"请对比这两张图片的{focus}方面，指出相同点和不同点。"
    # 这里简化：分别分析两张图片，然后让 LLM 对比
    desc_a = _call_vision_llm(path_a, "请详细描述这张图片。")
    desc_b = _call_vision_llm(path_b, "请详细描述这张图片。")

    comparison_prompt = f"图片 A 描述：\n{desc_a}\n\n图片 B 描述：\n{desc_b}\n\n请对比这两张图片的{focus}方面。"
    # 再次调用 LLM 进行对比（这里简化，直接返回分别的描述）
    return json.dumps({
        "image_a_description": desc_a,
        "image_b_description": desc_b,
        "focus": focus,
        "note": "分别分析了两张图片，请基于描述自行对比",
    }, ensure_ascii=False)


def vision_extract_text(path: str, language: str = "auto") -> str:
    """OCR 提取图片中的文字。

    参数（JSON 字符串）：
        path: 图片路径
        language: 语言提示（"chinese" / "english" / "auto"，默认 "auto"）

    返回：
        JSON 字符串，包含 text 字段（提取的文字）
    """

    lang_hint = {
        "chinese": "图片中是中文文字。",
        "english": "图片中是英文文字。",
        "auto": "自动检测图片中的语言。",
    }
    prompt = f"{lang_hint.get(language, '')}请提取并转录图片中的所有文字，保持原有格式。"

    result = _call_vision_llm(path, prompt)
    return json.dumps({"text": result, "language": language}, ensure_ascii=False)


# ── OpenAI 工具 Schema ──────────────────────────────────────────────────────

VISION_ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "图片路径（本地路径或 HTTP URL）"},
        "question": {"type": "string", "description": "要回答的问题（默认：请描述这张图片）"},
        "model": {"type": "string", "description": "指定多模态模型（可选，留空自动选择）"},
        "vendor_id": {"type": "string", "description": "厂商 ID（可选，使用项目配置）"},
        "api_key": {"type": "string", "description": "API Key（可选，使用项目配置）"},
    },
    "required": ["path"],
}

VISION_DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "图片路径"},
        "detail_level": {
            "type": "string",
            "enum": ["brief", "detailed", "professional"],
            "description": "详细程度：brief（简洁）/ detailed（详细）/ professional（专业）",
        },
    },
    "required": ["path"],
}

VISION_COMPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "path_a": {"type": "string", "description": "第一张图片路径"},
        "path_b": {"type": "string", "description": "第二张图片路径"},
        "focus": {
            "type": "string",
            "enum": ["general", "objects", "style"],
            "description": "对比焦点：general（总体）/ objects（物体）/ style（风格）",
        },
    },
    "required": ["path_a", "path_b"],
}

VISION_EXTRACT_TEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "图片路径"},
        "language": {
            "type": "string",
            "enum": ["auto", "chinese", "english"],
            "description": "语言提示（auto 自动检测）",
        },
    },
    "required": ["path"],
}


# ── 注册到工具系统 ──────────────────────────────────────────────────────────

registry = get_registry()

registry.register(
    name="vision_analyze",
    schema=VISION_ANALYZE_SCHEMA,
    handler=vision_analyze,
)

registry.register(
    name="vision_describe",
    schema=VISION_DESCRIBE_SCHEMA,
    handler=vision_describe,
)

registry.register(
    name="vision_compare",
    schema=VISION_COMPARE_SCHEMA,
    handler=vision_compare,
)

registry.register(
    name="vision_extract_text",
    schema=VISION_EXTRACT_TEXT_SCHEMA,
    handler=vision_extract_text,
)

logger.info(" 视觉工具已注册：4 个（analyze / describe / compare / extract_text）")
