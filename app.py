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
from backends.gpu import LLAMA_AVAILABLE, HAVE_GPU, is_loaded as llama_is_loaded
from backends.gpu import get_config as llama_get_config, load_model as llama_load_model
from backends.gpu import unload_model as llama_unload_model, list_models as llama_list_models
from backends.gpu import infer as llama_infer

from backends.ollama import available as ollama_available, check as ollama_check
from backends.ollama import get_models as ollama_get_models, pull_model as ollama_pull
from backends.ollama import chat_stream as ollama_chat_stream
from backends.ollama import enhance_prompt
from services.prompts import list_templates, get_template, save_template, delete_template, apply_template

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
    ollama_check()
    available = []
    if LLAMA_AVAILABLE:
        available.append("llama-cpp")
    if ollama_available:
        available.append("ollama")
    if not available:
        return jsonify({"status": "degraded", "backend": "none"}), 503
    return jsonify({
        "status": "ok",
        "current_backend": _current_backend,
        "available_backends": available,
    })


@app.route("/api/status")
def api_status():
    ollama_check()
    available = []
    if LLAMA_AVAILABLE:
        available.append("llama-cpp")
    if ollama_available:
        available.append("ollama")
    return jsonify({
        "current_backend": _current_backend,
        "available_backends": available,
        "llama_cpp": {
            "available": LLAMA_AVAILABLE,
            "gpu_available": HAVE_GPU,
            "cpu_mode": _cpu_mode,
            "model_loaded": llama_is_loaded() if LLAMA_AVAILABLE else False,
            "config": (llama_get_config() or None) if LLAMA_AVAILABLE else None,
        },
        "ollama": {
            "available": ollama_available,
        },
    })


# ─── 设置持久化（项目目录 settings.json）──────────────────────────────────────
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    return jsonify(json.load(f))
        except Exception as e:
            log.warning("读取设置失败: %s", e)
        return jsonify({})
    else:
        try:
            data = request.json or {}
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return jsonify({"status": "saved"})
        except Exception as e:
            log.error("保存设置失败: %s", e)
            return _err(str(e), 500)


@app.route("/api/switch_backend", methods=["POST"])
def api_switch_backend():
    """切换后端"""
    global _current_backend
    data = request.json or {}
    target = data.get("backend")
    if target == "llama-cpp" and LLAMA_AVAILABLE:
        _current_backend = "llama-cpp"
        log.info("切换后端: llama-cpp")
        return jsonify({"status": "switched", "backend": "llama-cpp"})
    elif target == "ollama":
        ollama_check()  # 重新检测
        if ollama_available:
            _current_backend = "ollama"
            log.info("切换后端: Ollama")
            return jsonify({"status": "switched", "backend": "ollama"})
        return _err("Ollama 未运行", 400)
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
        if _current_backend == "llama-cpp" and LLAMA_AVAILABLE:
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
if True:  # llama-cpp routes (always registered, runtime check per-handler)
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
if True:  # Ollama routes (always registered, runtime check per-handler)
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
        from services.search import search_ddg, search_bing

        results = search_ddg(query)
        if not results:
            log.info("DDG 无结果，尝试 Bing")
            results = search_bing(query)

        return jsonify({"query": query, "results": results, "count": len(results)})
    except Exception as e:
        log.error("搜索失败: %s", e)
        return _err(str(e), 500)



# ═══════════════════════════════════════════════════════════════════════════════
# 厂商 API 路由（OpenAI / DeepSeek / Anthropic / Gemini / Qwen / Zhipu / 自定义）
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/api/vendors")
def api_vendors():
    """列出所有可用的厂商（含模型列表和 server 端 API key 状态）。"""
    try:
        from backends.vendors import get_available_vendors
        vendors = get_available_vendors()
        return jsonify({"vendors": vendors})
    except Exception as e:
        log.error("获取厂商列表失败: %s", e)
        return _err(str(e), 500)


@app.route("/api/vendors/chat", methods=["POST"])
def api_vendors_chat():
    """统一厂商聊天接口（流式）。

    请求体:
        {
            "vendor": "openai",
            "model": "gpt-4o-mini",
            "messages": [{"role":"user","content":"Hello"}],
            "api_key": "sk-...",        // 可选，覆盖环境变量
            "base_url": "https://...",   // 可选，仅 custom 厂商需要
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9
        }
    """
    data = request.json or {}
    vendor_id = data.get("vendor", "").strip()
    model = data.get("model", "").strip()
    messages = data.get("messages", [])

    if not vendor_id:
        return _err("缺少 vendor 参数")
    if not model:
        return _err("缺少 model 参数")
    if not messages:
        return _err("缺少 messages 参数")

    try:
        from backends.vendors import chat_stream

        def generate():
            try:
                for chunk in chat_stream(
                    vendor_id=vendor_id,
                    model=model,
                    messages=messages,
                    api_key=data.get("api_key", ""),
                    base_url=data.get("base_url", ""),
                    max_tokens=data.get("max_tokens"),
                    temperature=data.get("temperature"),
                    top_p=data.get("top_p"),
                ):
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                log.error("厂商流式错误: %s", e)
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    except Exception as e:
        log.error("厂商聊天失败: %s", e)
        return _err(str(e), 500)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
