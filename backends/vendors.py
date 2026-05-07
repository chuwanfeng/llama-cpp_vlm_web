"""
厂商 API 后端 — 支持 OpenAI / DeepSeek / Anthropic / Gemini / 通义千问 / 智谱 / Moonshot / 自定义
参考 hermes-agent 的 providers.py 架构，简化适配 Web Chat UI。
"""
import os
import logging

log = logging.getLogger("llm-web")

# ── 可选依赖检测 ──────────────────────────────────────────────────────────────
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass

ANTHROPIC_AVAILABLE = False
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    pass

GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    pass

# ── 厂商定义（参照 hermes-agent 的 HERMES_OVERLAYS）───────────────────────────
VENDORS = {
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "transport": "openai_chat",
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4-turbo",
            "o3-mini",
            "o4-mini",
        ],
        "default_model": "gpt-4o-mini",
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "transport": "openai_chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "base_url": "https://api.anthropic.com",
        "api_key_env": "ANTHROPIC_API_KEY",
        "transport": "anthropic_messages",
        "models": [
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
            "claude-opus-4-20250514",
        ],
        "default_model": "claude-sonnet-4-20250514",
    },
    "gemini": {
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key_env": "GOOGLE_API_KEY",
        "transport": "gemini",
        "models": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
        ],
        "default_model": "gemini-2.5-flash",
    },
    "qwen": {
        "name": "通义千问 (DashScope)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "transport": "openai_chat",
        "models": [
            "qwen-plus",
            "qwen-max",
            "qwen-turbo",
            "qwen-plus-latest",
            "qwen-max-latest",
            "qwen3-235b-a22b",
        ],
        "default_model": "qwen-plus",
    },
    "zhipu": {
        "name": "智谱 AI (GLM)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "ZHIPUAI_API_KEY",
        "transport": "openai_chat",
        "models": [
            "glm-4-plus",
            "glm-4-flash",
            "glm-4-air",
            "glm-4-airx",
            "glm-4-long",
        ],
        "default_model": "glm-4-flash",
    },
    "moonshot": {
        "name": "Moonshot (Kimi)",
        "base_url": "https://api.moonshot.cn/v1",
        "api_key_env": "MOONSHOT_API_KEY",
        "transport": "openai_chat",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        "default_model": "moonshot-v1-32k",
    },
    "ollama-cloud": {
        "name": "Ollama Cloud",
        "base_url": "https://ollama.com/v1",
        "api_key_env": "OLLAMA_API_KEY",
        "transport": "openai_chat",
        "models": [
            "glm-4.7:cloud",
            "qwen3.5:cloud",
            "deepseek-v4-pro:cloud",
            "gemma4:31b-cloud",
        ],
        "default_model": "gemma4:31b-cloud",
    },
}

def get_vendor(vendor_id: str):
    """获取厂商定义，不存在返回 None。"""
    return VENDORS.get(vendor_id)

def get_available_vendors():
    """返回依赖已安装的厂商列表（含 server 端 API key 状态）。"""
    available = []
    for vid, vdef in VENDORS.items():
        transport = vdef.get("transport", "")
        if transport == "openai_chat" and not OPENAI_AVAILABLE:
            continue
        if transport == "anthropic_messages" and not ANTHROPIC_AVAILABLE:
            continue
        if transport == "gemini" and not GEMINI_AVAILABLE:
            continue
        # custom always available if openai is
        if vid == "custom" and not OPENAI_AVAILABLE:
            continue

        api_key = _read_api_key(vdef)
        available.append({
            "id": vid,
            "name": vdef["name"],
            "transport": transport,
            "base_url": vdef["base_url"],
            "has_server_key": bool(api_key),
            "models": vdef["models"],
            "default_model": vdef["default_model"],
        })
    return available

def _read_api_key(vdef: dict) -> str:
    """读取厂商 API key（环境变量优先，其次是多个备选键）。"""
    env_keys = [vdef.get("api_key_env", "")]
    # 额外备选 env key
    extra = {
        "openai": [],
        "deepseek": [],
        "anthropic": ["ANTHROPIC_TOKEN"],
        "gemini": ["GEMINI_API_KEY"],
        "qwen": ["DASHSCOPE_API_KEY"],
        "zhipu": ["GLM_API_KEY", "ZAI_API_KEY"],
        "moonshot": ["KIMI_API_KEY"],
        "ollama-cloud": ["OLLAMA_CLOUD_KEY"],
        "custom": [],
    }
    env_keys += extra.get("", [])

    for ek in env_keys:
        if ek:
            val = os.environ.get(ek, "")
            if val:
                return val
    return ""

def chat_stream(vendor_id: str, model: str, messages: list,
                api_key: str = "", base_url: str = "", **params):
    """统一流式聊天入口，自动路由到对应传输层。

    Args:
        vendor_id: 厂商 ID
        model: 模型名
        messages: OpenAI 格式消息列表 [{"role":"user","content":"..."}]
        api_key: API key（覆盖环境变量）
        base_url: 自定义 base URL（覆盖默认值）
        **params: temperature, max_tokens, top_p 等

    Yields:
        str: 逐块文本
    """
    vdef = VENDORS.get(vendor_id)
    if not vdef:
        raise ValueError(f"未知厂商: {vendor_id}")

    transport = vdef.get("transport", "openai_chat")
    api_key = api_key or _read_api_key(vdef)
    if not api_key:
        raise ValueError(f"缺少 API Key（请设置环境变量 {vdef.get('api_key_env', '')} 或在界面中输入）")

    if not base_url:
        base_url = vdef.get("base_url", "")

    if transport == "anthropic_messages":
        yield from _anthropic_stream(model, messages, api_key, base_url, vdef, **params)
    elif transport == "gemini":
        yield from _gemini_stream(model, messages, api_key, **params)
    else:
        yield from _openai_stream(model, messages, api_key, base_url, **params)


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI 兼容传输层（OpenAI / DeepSeek / Qwen / Zhipu / Moonshot / Custom）
# ═══════════════════════════════════════════════════════════════════════════════
def _openai_stream(model: str, messages: list, api_key: str, base_url: str, **params):
    """OpenAI 兼容流式 API。"""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)

    # 对 DeepSeek 统一去掉 system role（DeepSeek 只在首个 user 消息前自动识别 system）
    if "deepseek" in (base_url or ""):
        messages = _normalize_deepseek_messages(messages)

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=params.get("max_tokens") or 4096,
        temperature=params.get("temperature", 0.7),
        top_p=params.get("top_p", 0.9),
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def _normalize_deepseek_messages(messages: list) -> list:
    """DeepSeek 不支持 system role，合并为第一个 user 消息。"""
    system_text = ""
    other = []
    for msg in messages:
        if msg.get("role") == "system":
            system_text += msg.get("content", "") + "\n"
        else:
            other.append(msg)
    if system_text and other:
        first_user = other[0]
        if first_user.get("role") == "user":
            first_user = dict(first_user)
            first_user["content"] = system_text.strip() + "\n\n" + first_user.get("content", "")
            other[0] = first_user
        else:
            other.insert(0, {"role": "user", "content": system_text.strip()})
    return other


# ═══════════════════════════════════════════════════════════════════════════════
# Anthropic 传输层
# ═══════════════════════════════════════════════════════════════════════════════
def _anthropic_stream(model: str, messages: list, api_key: str, base_url: str,
                      vdef: dict, **params):
    """Anthropic Messages API 流式传输。"""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, timeout=120)

    system_prompts = []
    chat_messages = []
    for msg in messages:
        if msg["role"] == "system":
            text = msg.get("content", "")
            if isinstance(text, str):
                system_prompts.append({"type": "text", "text": text})
            elif isinstance(text, list):
                system_prompts.extend(text)
        else:
            chat_messages.append(msg)

    kwargs = dict(
        model=model,
        max_tokens=params.get("max_tokens") or 4096,
        messages=chat_messages,
        stream=True,
    )
    if system_prompts:
        kwargs["system"] = system_prompts

    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            yield text


# ═══════════════════════════════════════════════════════════════════════════════
# Google Gemini 传输层
# ═══════════════════════════════════════════════════════════════════════════════
def _gemini_stream(model: str, messages: list, api_key: str, **params):
    """Google Gemini API 流式传输。"""
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    # Gemini generateContent 不支持多轮 history + system instruction 的简单映射，
    # 使用 chat (start_chat) 模式处理多轮对话。
    system_instruction = None
    history = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_instruction = content
        elif role == "user":
            history.append({"role": "user", "parts": [content]})
        elif role == "assistant":
            history.append({"role": "model", "parts": [content]})

    # 最后一条 user 消息作为当前轮次输入
    current_input = None
    if history and history[-1]["role"] == "user":
        current_input = history.pop()

    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_instruction,
    )

    # 把历史（不含最后一条 user）传入 start_chat
    chat = gemini_model.start_chat(history=history if len(history) > 0 else None)

    gen_config = {}
    if params.get("temperature") is not None:
        gen_config["temperature"] = params["temperature"]
    if params.get("max_tokens"):
        gen_config["max_output_tokens"] = params["max_tokens"]
    if params.get("top_p") is not None:
        gen_config["top_p"] = params["top_p"]

    if current_input:
        response = chat.send_message(
            current_input["parts"][0],
            stream=True,
            generation_config=genai.types.GenerationConfig(**gen_config) if gen_config else None,
        )
    else:
        response = chat.send_message(
            "Hello",
            stream=True,
            generation_config=genai.types.GenerationConfig(**gen_config) if gen_config else None,
        )

    for chunk in response:
        if chunk.text:
            yield chunk.text
