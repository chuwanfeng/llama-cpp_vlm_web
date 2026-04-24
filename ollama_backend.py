"""
Ollama 后端 — REST API
"""
import json
import logging
import requests
from threading import Thread

from config import (
    OLLAMA_BASE, OLLAMA_TIMEOUT, OLLAMA_STREAM_TIMEOUT,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K,
    DEFAULT_REPEAT_PENALTY,
)
from prompts import apply_template, get_template

log = logging.getLogger("llm-web")

# ─── 模块状态 ────────────────────────────────────────────────────────────────
available = False
_DEFAULT_MODEL = "qwen2.5:7b"


def check():
    """检测 Ollama 是否可用"""
    global available
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=OLLAMA_TIMEOUT)
        available = r.status_code == 200
        if available:
            log.info("Ollama 已连接")
        return available
    except Exception as e:
        available = False
        log.warning("Ollama 连接失败: %s", e)
        return False


def is_available():
    return available


def get_models():
    """获取已安装模型列表"""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=OLLAMA_TIMEOUT)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def pull_model(name):
    """后台拉取模型"""
    def _pull():
        try:
            requests.post(f"{OLLAMA_BASE}/api/pull", json={"name": name}, timeout=None)
            log.info("模型拉取完成: %s", name)
        except Exception as e:
            log.error("模型拉取失败: %s", e)
    Thread(target=_pull, daemon=True).start()
    log.info("开始拉取模型: %s", name)


def enhance_prompt(system: str, user: str, model: str = None) -> str:
    """调用 LLM 执行模板，返回 LLM 输出"""
    model = model or _DEFAULT_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    result = []
    for chunk in chat_stream(model, messages, max_tokens=512, temperature=0.3):
        if "error" in chunk:
            raise RuntimeError(chunk["error"])
        if chunk.get("message", {}).get("content"):
            result.append(chunk["message"]["content"])
    return "".join(result)


def set_default_model(name: str):
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = name
    log.info("默认模型: %s", name)


def chat_stream(model, messages, **opts):
    """流式对话"""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": opts.get("temperature", DEFAULT_TEMPERATURE),
            "num_predict": opts.get("max_tokens", 4096),
            "top_p": opts.get("top_p", DEFAULT_TOP_P),
            "top_k": opts.get("top_k", DEFAULT_TOP_K),
            "repeat_penalty": opts.get("repeat_penalty", DEFAULT_REPEAT_PENALTY),
        }
    }
    try:
        with requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            stream=True,
            timeout=OLLAMA_STREAM_TIMEOUT
        ) as r:
            for line in r.iter_lines():
                if line:
                    data = json.loads(line.decode())
                    yield data
                    if data.get("done"):
                        break
    except requests.exceptions.ConnectionError:
        yield {"error": "Ollama 连接中断，请确认服务正在运行"}
    except requests.exceptions.Timeout:
        yield {"error": "Ollama 响应超时，模型可能过载或输入过长"}
    except Exception as e:
        yield {"error": str(e)}


# ─── 初始化 ──────────────────────────────────────────────────────────────────
check()
