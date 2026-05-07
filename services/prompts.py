"""
提示词模板 — CRUD + 持久化
"""
import os
import json
from typing import Optional

# ─── 路径 ────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_TPL_FILE = os.path.join(_DIR, "prompt_templates.json")

# ─── 默认模板 ────────────────────────────────────────────────────────────────
_DEFAULT = {
    "image_optimizer_zh": {
        "name": "图像优化师（中文）",
        "description": "将用户描述扩写为详细的图像生成 Prompt",
        "system": "你是一位专业的 Prompt 优化师。收到用户描述后，将其扩写为完整、生动、画面感强的图像 Prompt。保留原意，补全细节（主体特征、画面风格、背景、景别）。不增加否定词，保持画面积极正面。",
        "prefix": "请将以下描述扩写为图像 Prompt，直接输出扩写结果，无需解释：\n",
        "suffix": "",
        "builtin": True,
    },
    "image_optimizer_en": {
        "name": "Image Optimizer (English)",
        "description": "Expand user description into detailed image generation Prompt",
        "system": "You are a professional prompt optimizer. Rewrite user inputs into vivid, detailed image prompts. Keep original meaning, add visual details (subject, style, background, framing). Max 200 words.",
        "prefix": "Rewrite this into an image prompt, output only the result:\n",
        "suffix": "",
        "builtin": True,
    },
    "subtitle_ja2zh": {
        "name": "日语字幕→中文",
        "description": "SRT/VTT 字幕文件翻译，保留时间码",
        "system": "你是一个专业的字幕翻译专家。你收到的是 SRT 格式的日语字幕（包含序号、时间码、台词），必须原样保留序号和时间码，只翻译台词行。将日语台词忠实翻译为中文白话，贴近口语，角色语气一致。不要解释，不要分析，不要添加任何额外文字，只输出完整的 SRT 文件。",
        "prefix": "",
        "suffix": "\n\n【重要】严格输出纯 SRT 字幕文本，格式：\n序号\n时间码\n翻译后台词\n（中间留空一行），不要输出任何其他内容。",
        "builtin": True,
    },
    "translator_en": {
        "name": "英译中翻译",
        "description": "精通英语的专业翻译，忠实流畅",
        "system": "你是一个翻译专家。只输出翻译结果，不要解释、不要分析、不要评论原文。只输出一行翻译内容。原文是什么就译什么。",
        "prefix": "翻译以下内容，直接输出翻译结果，不要任何额外文字：\n",
        "suffix": "",
        "builtin": True,
    },
    "translator_zh": {
        "name": "中译英翻译",
        "description": "Professional English translator",
        "system": "You are a translator. Output only the translation, nothing else. Do not explain, do not comment, do not add any analysis. Translate exactly what is given.",
        "prefix": "Translate to English. Output only the translation:\n",
        "suffix": "",
        "builtin": True,
    },
    "code_review": {
        "name": "代码审查",
        "description": "资深后端工程师，专注可维护性和性能",
        "system": "你是一位有10年经验的后端工程师。审查代码时关注：可读性、可维护性、潜在 bug、安全风险、性能问题。用简洁直接的语言给出意见，不废话。",
        "prefix": "请审查以下代码：\n",
        "suffix": "",
        "builtin": True,
    },
    "creative_writing": {
        "name": "创意写作",
        "description": "文学创作助手，中文为主",
        "system": "你是一位中文文学创作助手。擅长故事构思、文笔润色、风格模仿。根据用户需求进行创作或改写，语言生动，画面感强。",
        "prefix": "",
        "suffix": "",
        "builtin": True,
    },
}


# ─── 持久化读写 ──────────────────────────────────────────────────────────────
def _load() -> dict:
    if not os.path.exists(_TPL_FILE):
        _save(_DEFAULT)
        return _DEFAULT.copy()
    try:
        with open(_TPL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return _DEFAULT.copy()


def _save(data: dict):
    with open(_TPL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── CRUD ─────────────────────────────────────────────────────────────────────
def list_templates() -> list:
    """返回模板列表（完整字段，供 sendChat 渲染 system prompt）"""
    templates = _load()
    return [
        {
            "id": tid,
            "name": t["name"],
            "description": t.get("description", ""),
            "system": t.get("system", ""),
            "prefix": t.get("prefix", ""),
            "suffix": t.get("suffix", ""),
        }
        for tid, t in templates.items()
    ]


def get_template(tid: str) -> Optional[dict]:
    templates = _load()
    return templates.get(tid)


def save_template(tid: str, data: dict) -> bool:
    """新增或更新模板"""
    templates = _load()
    existing = templates.get(tid, {})
    templates[tid] = {
        "name": data.get("name", tid),
        "description": data.get("description", ""),
        "system": data.get("system", ""),
        "prefix": data.get("prefix", ""),
        "suffix": data.get("suffix", ""),
        "builtin": existing.get("builtin", False),  # 保持 builtin 标记不丢失
    }
    _save(templates)
    return True


def delete_template(tid: str) -> bool:
    """删除模板（builtin=True 不可删）"""
    templates = _load()
    if templates.get(tid, {}).get("builtin"):
        return False  # builtin 模板不允许删除
    if tid in templates:
        del templates[tid]
        _save(templates)
        return True
    return False


def apply_template(tid: str, user_input: str) -> dict:
    """
    返回渲染后的消息列表，可直接发给 LLM。
    返回: {"system": "...", "user": "..."}
    """
    tpl = get_template(tid)
    if not tpl:
        return {"system": "", "user": user_input}

    user_content = tpl.get("prefix", "") + user_input + tpl.get("suffix", "")
    return {
        "system": tpl.get("system", ""),
        "user": user_content,
    }
