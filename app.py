"""
Flask 路由 — 统一后端入口（llama-cpp 优先，Ollama 备选）
"""
import os
import sys
import base64
import json
import logging
import time
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from config import MODELS_DIR, HOST, PORT, DEBUG

# ─── 日志 ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("llm-web")

# ─── 后端选择（llama-cpp 优先）────────────────────────────────────────────────
from gpu_backend import LLAMA_AVAILABLE, HAVE_GPU, is_loaded as llama_is_loaded
from gpu_backend import get_config as llama_get_config, load_model as llama_load_model
from gpu_backend import unload_model as llama_unload_model, list_models as llama_list_models
from gpu_backend import infer as llama_infer

from ollama_backend import available as ollama_available, check as ollama_check
from ollama_backend import get_models as ollama_get_models, pull_model as ollama_pull
from ollama_backend import chat_stream as ollama_chat_stream
from ollama_backend import enhance_prompt
from prompts import list_templates, get_template, save_template, delete_template, apply_template

# 确定活跃后端
# 优先级: llama-cpp (如果可用) > Ollama (如果运行中) > 无
USE_LLAMA = LLAMA_AVAILABLE
USE_OLLAMA = not USE_LLAMA and ollama_available

if USE_LLAMA:
    BACKEND = "llama-cpp"
    if HAVE_GPU:
        log.info("后端: llama-cpp (GPU 模式)")
    else:
        log.info("后端: llama-cpp (CPU 模式)")
elif USE_OLLAMA:
    BACKEND = "ollama"
    log.info("后端: Ollama")
else:
    BACKEND = "none"
    log.warning("没有可用的后端（llama-cpp 未安装，Ollama 未运行）")

# 后端切换状态
_current_backend = BACKEND
_cpu_mode = not HAVE_GPU and USE_LLAMA  # 是否处于 CPU 模式

# ─── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)


# ─── 请求日志中间件 ──────────────────────────────────────────────────────────
@app.before_request
def _log_req():
    request._start = time.time()


@app.after_request
def _log_res(resp):
    elapsed = time.time() - getattr(request, "_start", time.time())
    if request.path.startswith("/api/"):
        log.info("%s %s → %s (%.1fms)", request.method, request.path, resp.status_code, elapsed * 1000)
    return resp


# ─── 统一错误响应 ─────────────────────────────────────────────────────────────
def _err(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


# ═══════════════════════════════════════════════════════════════════════════════
# 通用路由
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def api_health():
    """健康检查 — 前端轮询用"""
    if USE_LLAMA:
        return jsonify({
            "status": "ok",
            "backend": "llama-cpp",
            "gpu_available": HAVE_GPU,
            "cpu_mode": _cpu_mode,
            "model_loaded": llama_is_loaded()
        })
    elif USE_OLLAMA:
        ollama_check()
        return jsonify({
            "status": "ok" if ollama_available else "degraded",
            "backend": "ollama",
            "running": ollama_available
        })
    else:
        return jsonify({"status": "degraded", "backend": "none"}), 503


@app.route("/api/status")
def api_status():
    if USE_LLAMA:
        return jsonify({
            "backend": "llama-cpp",
            "gpu_available": HAVE_GPU,
            "cpu_mode": _cpu_mode,
            "model_loaded": llama_is_loaded(),
            "config": llama_get_config() or None,
        })
    elif USE_OLLAMA:
        ollama_check()
        return jsonify({
            "backend": "ollama",
            "running": ollama_available
        })
    else:
        return jsonify({"backend": "none", "available": False}), 503


@app.route("/api/switch_backend", methods=["POST"])
def api_switch_backend():
    """切换后端（预留，后续实现）"""
    global _current_backend, USE_LLAMA, USE_OLLAMA, BACKEND
    data = request.json or {}
    target = data.get("backend")
    if target == "llama-cpp" and LLAMA_AVAILABLE:
        USE_LLAMA = True
        USE_OLLAMA = False
        BACKEND = "llama-cpp"
        return jsonify({"status": "switched", "backend": "llama-cpp"})
    elif target == "ollama" and ollama_available:
        USE_LLAMA = False
        USE_OLLAMA = True
        BACKEND = "ollama"
        return jsonify({"status": "switched", "backend": "ollama"})
    else:
        return _err(f"后端不可用: {target}", 400)


@app.route("/api/upload_image", methods=["POST"])
def api_upload_image():
    if "image" in request.files:
        img_bytes = request.files["image"].read()
    elif request.json and "image" in request.json:
        b64 = request.json["image"]
        img_bytes = base64.b64decode(b64.split(",", 1)[-1] if "," in b64 else b64)
    else:
        return _err("没有图片")
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    return jsonify({"base64": b64_str, "size": len(img_bytes)})


# ═══════════════════════════════════════════════════════════════════════════════
# 通用路由（模板 CRUD）— 不依赖后端类型
# ═══════════════════════════════════════════════════════════════════════════════
from prompts import list_templates, get_template, save_template, delete_template, apply_template

@app.route("/api/prompt_templates", methods=["GET"])
def api_templates_list():
    return jsonify({"templates": list_templates()})

@app.route("/api/prompt_templates/<tid>", methods=["GET"])
def api_templates_get(tid):
    tpl = get_template(tid)
    if not tpl:
        return _err("模板不存在", 404)
    return jsonify({"template": tpl, "id": tid})

@app.route("/api/prompt_templates", methods=["POST"])
def api_templates_save():
    data = request.json or {}
    tid = data.get("id", "").strip()
    if not tid:
        return _err("缺少模板 ID")
    save_template(tid, data)
    return jsonify({"status": "ok", "id": tid})

@app.route("/api/prompt_templates/<tid>", methods=["DELETE"])
def api_templates_delete(tid):
    ok = delete_template(tid)
    if not ok:
        return _err("内置模板不可删除", 403)
    return jsonify({"status": "ok", "id": tid})

@app.route("/api/enhance", methods=["POST"])
def api_enhance():
    data = request.json or {}
    user_input = data.get("prompt", "").strip()
    template_id = data.get("template", "")
    if not user_input:
        return _err("prompt 为空")
    if not template_id:
        return _err("缺少 template 参数")
    try:
        tpl = apply_template(template_id, user_input)
        # 使用当前后端进行增强
        if USE_LLAMA:
            output = llama_infer(prompt=tpl["user"], system=tpl["system"], stream=False)
        else:
            output = enhance_prompt(system=tpl["system"], user=tpl["user"])
        return jsonify({
            "original": user_input,
            "template_id": template_id,
            "output": output,
        })
    except Exception as e:
        log.error("增强失败: %s", e)
        return _err(str(e), 500)


# ═══════════════════════════════════════════════════════════════════════════════
# llama-cpp 后端路由（GPU + CPU）
# ═══════════════════════════════════════════════════════════════════════════════
if USE_LLAMA:
    @app.route("/api/llama/status")
    def api_llama_status():
        return jsonify({
            "backend": "llama-cpp",
            "gpu_available": HAVE_GPU,
            "cpu_mode": _cpu_mode,
            "model_loaded": llama_is_loaded(),
            "config": llama_get_config() or None,
            "models_dir": MODELS_DIR,
        })

    @app.route("/api/llama/models")
    def api_llama_models():
        return jsonify({"models": llama_list_models()})

    @app.route("/api/llama/load_model", methods=["POST"])
    def api_llama_load_model():
        data = request.json or {}
        model = data.get("model")
        if not model:
            return _err("缺少 model 参数")
        model_path = os.path.join(MODELS_DIR, model)
        if not os.path.exists(model_path):
            return _err(f"模型文件不存在: {model_path}")
        try:
            # chat_handler: "auto" | 具体路径 | None
            # 默认 auto — gpu_backend 会自动检测同目录下的 mmproj
            chat_handler = data.get("chat_handler", "auto")
            llama_load_model(
                model_path=model_path,
                n_ctx=data.get("n_ctx"),
                n_gpu_layers=data.get("n_gpu_layers"),
                chat_handler=chat_handler,
                force_cpu=data.get("force_cpu", False),
            )
            config = llama_get_config()
            log.info("llama-cpp 模型已加载: %s, mmproj=%s", model, config.get("mmproj_loaded"))
            return jsonify({"status": "loaded", "model": model, "config": config})
        except Exception as e:
            log.error("llama-cpp 加载失败: %s", e)
            return _err(str(e), 500)

    @app.route("/api/llama/unload", methods=["POST"])
    def api_llama_unload():
        llama_unload_model()
        return jsonify({"status": "unloaded"})

    @app.route("/api/llama/infer", methods=["POST"])
    def api_llama_infer():
        if not llama_is_loaded():
            return _err("模型未加载")
        data = request.json or {}
        images_raw = data.get("images")
        log.info("[DEBUG infer] images count=%s, types=%s, lengths=%s",
                 len(images_raw) if images_raw else 0,
                 [type(i).__name__ for i in images_raw] if images_raw else [],
                 [len(str(i)) for i in images_raw] if images_raw else [])
        if images_raw:
            for idx, img in enumerate(images_raw):
                prefix = str(img)[:80] if isinstance(img, str) else "non-string"
                log.info("[DEBUG infer] images[%s] prefix: %s", idx, prefix)
        stream = data.get("stream", False)
        try:
            if stream:
                def generate():
                    for chunk in llama_infer(
                        prompt=data.get("prompt", ""),
                        images=data.get("images"),
                        system=data.get("system_prompt"),
                        max_tokens=data.get("max_tokens"),
                        temperature=data.get("temperature"),
                        top_p=data.get("top_p"),
                        top_k=data.get("top_k"),
                        repeat_penalty=data.get("repeat_penalty"),
                        stream=True,
                    ):
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
                return Response(stream_with_context(generate()), mimetype="text/event-stream")
            else:
                result = llama_infer(
                    prompt=data.get("prompt", ""),
                    images=data.get("images"),
                    system=data.get("system_prompt"),
                    max_tokens=data.get("max_tokens"),
                    temperature=data.get("temperature"),
                    top_p=data.get("top_p"),
                    top_k=data.get("top_k"),
                    repeat_penalty=data.get("repeat_penalty"),
                    stream=False,
                )
                return jsonify({"output": result, "backend": "llama-cpp"})
        except Exception as e:
            log.error("llama-cpp 推理失败: %s", e)
            return _err(str(e), 500)


# ═══════════════════════════════════════════════════════════════════════════════
# Ollama 后端路由（当 llama-cpp 不可用时）
# ═══════════════════════════════════════════════════════════════════════════════
elif USE_OLLAMA:
    @app.route("/api/ollama_status")
    def api_ollama_status():
        ollama_check()
        return jsonify({"running": ollama_available, "models": ollama_get_models()})

    @app.route("/api/models")
    def api_models():
        ollama_check()
        if not ollama_available:
            return _err("Ollama 未运行", 503)
        return jsonify({"models": ollama_get_models()})

    @app.route("/api/pull_model", methods=["POST"])
    def api_pull_model():
        name = (request.json or {}).get("name")
        if not name:
            return _err("缺少模型名")
        ollama_pull(name)
        return jsonify({"status": "started", "model": name})

    # ─── 提示词模板 CRUD ────────────────────────────────────────────────────
    @app.route("/api/prompt_templates", methods=["GET"])
    def api_templates_list():
        return jsonify({"templates": list_templates()})

    @app.route("/api/prompt_templates/<tid>", methods=["GET"])
    def api_templates_get(tid):
        tpl = get_template(tid)
        if not tpl:
            return _err("模板不存在", 404)
        return jsonify({"template": tpl, "id": tid})

    @app.route("/api/prompt_templates", methods=["POST"])
    def api_templates_save():
        data = request.json or {}
        tid = data.get("id", "").strip()
        if not tid:
            return _err("缺少模板 ID")
        save_template(tid, data)
        return jsonify({"status": "ok", "id": tid})

    @app.route("/api/prompt_templates/<tid>", methods=["DELETE"])
    def api_templates_delete(tid):
        ok = delete_template(tid)
        if not ok:
            return _err("内置模板不可删除", 403)
        return jsonify({"status": "ok", "id": tid})

    @app.route("/api/enhance", methods=["POST"])
    def api_enhance():
        data = request.json or {}
        user_input = data.get("prompt", "").strip()
        template_id = data.get("template", "")
        if not user_input:
            return _err("prompt 为空")
        if not template_id:
            return _err("缺少 template 参数")
        try:
            tpl = apply_template(template_id, user_input)
            output = enhance_prompt(system=tpl["system"], user=tpl["user"])
            return jsonify({
                "original": user_input,
                "template_id": template_id,
                "output": output,
            })
        except Exception as e:
            log.error("增强失败: %s", e)
            return _err(str(e), 500)

    # 保留 /api/chat 端点兼容性
    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        ollama_check()
        if not ollama_available:
            return _err("Ollama 未运行", 503)

        data = request.json or {}
        model = data.get("model", "qwen2.5:7b")
        messages = data.get("messages", [])
        system = data.get("system")
        if system:
            messages = [{"role": "system", "content": system}] + messages
        opts = {
            "max_tokens": data.get("max_tokens"),
            "temperature": data.get("temperature"),
            "top_p": data.get("top_p"),
            "top_k": data.get("top_k"),
            "repeat_penalty": data.get("repeat_penalty"),
        }

        def gen():
            for chunk in ollama_chat_stream(model, messages, **opts):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        return Response(stream_with_context(gen()), mimetype="text/event-stream")

# ═══════════════════════════════════════════════════════════════════════════════
# Web 搜索功能（通用路由，放在最后确保不与其他路由冲突）
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/api/search", methods=["GET", "POST"])
def api_search():
    if request.method == "GET":
        query = request.args.get("q", "").strip()
    else:
        query = (request.json or {}).get("query", "").strip()
    if not query:
        return _err("缺少查询关键字", 400)
    try:
        import requests
        from urllib.parse import quote
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return _err("搜索失败", 500)
        import re
        results = []
        pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)">([^<]+)</a>.*?<a class="result__snippet"[^>]*>([^<]+(?:<[^>]+>[^<]*</[^>]+>)*?)</a>'
        matches = re.findall(pattern, resp.text, re.DOTALL)
        for url, title, snippet in matches[:5]:
            snippet = re.sub(r'<[^>]+>', '', snippet)
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet[:200] + "..." if len(snippet) > 200 else snippet
            })
        return jsonify({"query": query, "results": results, "count": len(results)})
    except Exception as e:
        log.error("搜索失败: %s", e)
        return _err(str(e), 500)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
