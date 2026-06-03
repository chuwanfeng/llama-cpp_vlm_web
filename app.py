"""
Flask 路由 — 统一后端入口（llama-cpp + 多厂商 API）
"""
import os
import base64
import json
import logging
import time
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from config import MODELS_DIR, HOST, PORT, DEBUG
from utils import setup_logging, log, json_error, _err, read_json, write_json, timed

# ─── 日志 ─────────────────────────────────────────────────────────────────────
setup_logging()

# ─── 后端选择（llama-cpp 优先）────────────────────────────────────────────────
from backends.gpu import LLAMA_AVAILABLE, HAVE_GPU, is_loaded as llama_is_loaded
from backends.gpu import get_config as llama_get_config, load_model as llama_load_model
from backends.gpu import unload_model as llama_unload_model, list_models as llama_list_models
from backends.gpu import infer as llama_infer

from services.prompts import list_templates, get_template, save_template, delete_template, apply_template
from services.session_store import get_store
from services.agent_service import bp as agent_bp

# 确定活跃后端
# 优先级: llama-cpp (如果可用) > 厂商 API（前端手动切换）
USE_LLAMA = LLAMA_AVAILABLE

if USE_LLAMA:
    BACKEND = "llama-cpp"
    if HAVE_GPU:
        log.info("后端: llama-cpp (GPU 模式)")
    else:
        log.info("后端: llama-cpp (CPU 模式)")
else:
    BACKEND = "none"
    log.warning("llama-cpp-python 未安装，请使用厂商 API")

# 后端切换状态
_current_backend = BACKEND
_cpu_mode = not HAVE_GPU and USE_LLAMA  # 是否处于 CPU 模式

# ─── 工具系统初始化 ─────────────────────────────────────────────────────────
from tools.registry import discover_tools, get_registry
discover_tools()  # 启动时即加载所有 builtin 工具 + MCP 服务器工具

# 注册后台进程管理工具
from tools.process_registry import register_process_tool
register_process_tool()

# ─── MCP 服务器状态 ──────────────────────────────────────────────────────────
from tools.mcp_tool import get_mcp_status, shutdown_mcp_servers
import atexit
atexit.register(shutdown_mcp_servers)  # 应用退出时关闭 MCP 连接

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
# _err 已移至 utils.py


# ═══════════════════════════════════════════════════════════════════════════════
# 通用路由
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """健康检查 — Docker/K8s 用"""
    return jsonify({"status": "healthy", "timestamp": time.time()})


@app.route("/ready")
def ready():
    """就绪检查 — 服务是否可接受流量"""
    # 检查工具系统
    registry = get_registry()
    tool_count = len(registry.get_tool_names())
    
    # 检查后端
    backend_ready = LLAMA_AVAILABLE or True  # API 模式总是就绪
    
    if tool_count == 0 or not backend_ready:
        return jsonify({
            "status": "not_ready",
            "tools": tool_count,
            "backend": _current_backend,
        }), 503
    
    return jsonify({
        "status": "ready",
        "tools": tool_count,
        "backend": _current_backend,
    })


@app.route("/metrics")
def metrics():
    """性能指标 — Prometheus 格式"""
    from services.performance_monitor import get_monitor
    mon = get_monitor()
    
    counters = mon.get_counters()
    stats = mon.get_all_stats(last_n=100)
    
    lines = []
    lines.append("# HELP llm_chat_requests_total Total requests")
    lines.append("# TYPE llm_chat_requests_total counter")
    for name, count in counters.items():
        lines.append(f'llm_chat_requests_total{{name="{name}"}} {count}')
    
    lines.append("# HELP llm_chat_latency_ms Request latency")
    lines.append("# TYPE llm_chat_latency_ms summary")
    for name, stat in stats.items():
        if stat["count"] > 0:
            lines.append(f'llm_chat_latency_ms{{name="{name}",quantile="0.5"}} {stat["p50"]}')
            lines.append(f'llm_chat_latency_ms{{name="{name}",quantile="0.95"}} {stat["p95"]}')
            lines.append(f'llm_chat_latency_ms{{name="{name}",quantile="0.99"}} {stat["p99"]}')
    
    return Response("\n".join(lines), mimetype="text/plain")


@app.route("/api/health")
def api_health():
    """健康检查 — 前端轮询用"""
    available = []
    if LLAMA_AVAILABLE:
        available.append("llama-cpp")
    if not available:
        return jsonify({"status": "degraded", "backend": "none"}), 503
    return jsonify({
        "status": "ok",
        "current_backend": _current_backend,
        "available_backends": available,
    })


@app.route("/api/status")
def api_status():
    available = []
    if LLAMA_AVAILABLE:
        available.append("llama-cpp")
    # 获取 MCP 服务器状态
    mcp_servers = []
    try:
        mcp_servers = get_mcp_status()
    except Exception:
        pass

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
        "mcp_servers": mcp_servers,
    })


# ─── 设置持久化（项目目录 settings.json）──────────────────────────────────────
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        data = read_json(SETTINGS_FILE, default={})
        return jsonify(data)
    else:
        try:
            data = request.json or {}
            write_json(SETTINGS_FILE, data)
            return jsonify({"status": "saved"})
        except Exception as e:
            log.error("保存设置失败: %s", e)
            return _err(str(e), 500)


@app.route("/api/aux-config", methods=["GET", "POST"])
def api_aux_config():
    """辅助模型配置 API"""
    from services.auxiliary import DEFAULT_AUX_CONFIG, _read_aux_config

    if request.method == "GET":
        cfg = _read_aux_config()
        return jsonify({
            "enabled": cfg.get("enabled", False),
            "provider": cfg.get("provider", ""),
            "model": cfg.get("model", ""),
            "tasks": cfg.get("tasks", ["compression"]),
        })
    else:
        try:
            data = request.json or {}
            settings = read_json(SETTINGS_FILE, default={})

            aux = settings.get("aux_config", dict(DEFAULT_AUX_CONFIG))
            for key in ("enabled", "provider", "model", "tasks"):
                if key in data:
                    aux[key] = data[key]
            settings["aux_config"] = aux

            write_json(SETTINGS_FILE, settings)
            return jsonify({"status": "saved"})
        except Exception as e:
            log.error("保存辅助配置失败: %s", e)
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
    # 厂商 API 后端由前端管理，无需服务端切换
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
            # 非 llama-cpp 后端，直接返回模板结果
            output = tpl.get("user", user_input)
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
        tools = data.get("tools")

        # 如果前端传了 tools，注入工具提示词到 system message
        _tool_schemas = []  # 工具 schema 列表（传给 gpu.infer，用于模板渲染）
        if tools:
            from tools.registry import discover_tools, get_registry
            from environments.tool_parser import build_tool_prompt
            discover_tools()
            schemas = get_registry().get_schemas()
            _tool_schemas = schemas
            tool_prompt = build_tool_prompt(schemas)
            existing_system = data.get("system_prompt") or ""
            if "messages" in data and data["messages"]:
                msgs = data["messages"]
                # 在第一条 system 消息后追加工具提示
                if msgs[0].get("role") == "system":
                    msgs[0]["content"] = msgs[0]["content"] + "\n\n" + tool_prompt
                else:
                    msgs.insert(0, {"role": "system", "content": tool_prompt})
                data["messages"] = msgs
            else:
                data["system_prompt"] = existing_system + "\n\n" + tool_prompt if existing_system else tool_prompt

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
                    import re, uuid
                    from copy import deepcopy
                    from collections.abc import Generator

                    # 构建推理参数（不传 tools=，让模型通过 XML 文本调用工具）
                    infer_params = {
                        "prompt": data.get("prompt"),
                        "images": data.get("images"),
                        "system": data.get("system_prompt"),
                        "max_tokens": data.get("max_tokens"),
                        "temperature": data.get("temperature"),
                        "top_p": data.get("top_p"),
                        "top_k": data.get("top_k"),
                        "repeat_penalty": data.get("repeat_penalty"),
                    }
                    # 用 messages 模式（优先）
                    _messages = deepcopy(data.get("messages", []))

                    max_rounds = 10  # 最多执行 10 轮工具调用
                    for round_num in range(max_rounds):
                        # ── 调模型，收集完整输出 ──
                        full_text = ""
                        for chunk in llama_infer(messages=_messages, stream=True, **infer_params):
                            content = chunk if isinstance(chunk, str) else chunk.get("content", "")
                            full_text += content
                            yield f"data: {json.dumps({'type': 'content', 'content': content}, ensure_ascii=False)}\n\n"

                        # ── 如果没有启用工具，直接结束 ──
                        if not tools:
                            break

                        # ── 用 ToolCallParser 解析 XML 工具调用 ──
                        from agent.loop import ToolCallParser
                        tool_calls = ToolCallParser.parse(full_text)

                        if not tool_calls:
                            break  # 无工具调用，结束循环

                        # ── 从文本中移除 XML 块 ──
                        clean_text = re.sub(
                            r'<tool_call[^>]*>.*?</tool_call>',
                            '', full_text, flags=re.DOTALL
                        ).strip()

                        # ── 构建 OpenAI 格式的 tool_calls ──
                        tc_stubs = []
                        for tc in tool_calls:
                            tc_id = f"call_{uuid.uuid4().hex[:8]}"
                            tc_args = json.dumps(tc["arguments"], ensure_ascii=False)
                            tc_stubs.append({
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc_args,
                                }
                            })
                            yield f"data: {json.dumps({'type': 'tool_call', 'name': tc['name'], 'args': tc['arguments'], 'id': tc_id}, ensure_ascii=False)}\n\n"

                        # ── 追加助手消息（含 tool_calls）──
                        _messages.append({
                            "role": "assistant",
                            "content": clean_text or None,
                            "tool_calls": tc_stubs,
                        })

                        # ── 执行工具 ──
                        from tools.registry import get_registry as _get_reg
                        registry = _get_reg()
                        for i, tc in enumerate(tool_calls):
                            tc_id = tc_stubs[i]["id"]
                            try:
                                entry = registry._tools.get(tc["name"])
                                if entry and entry.is_available():
                                    result = entry.handler(**tc["arguments"])
                                    if hasattr(result, "__await__"):
                                        import asyncio
                                        result = asyncio.run(result.__await__())
                                    result_str = str(result)
                                else:
                                    result_str = f"未知或不可用工具: {tc['name']}"
                            except Exception as e:
                                log.error("Tool %s failed: %s", tc["name"], e)
                                result_str = f"工具执行异常: {type(e).__name__}: {e}"

                            yield f"data: {json.dumps({'type': 'tool_result', 'name': tc['name'], 'result': result_str, 'id': tc_id}, ensure_ascii=False)}\n\n"

                            _messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": result_str,
                            })

                    # ── 结束事件 ──
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"

                return Response(stream_with_context(generate()), mimetype="text/event-stream")
            else:
                result = llama_infer(
                    prompt=data.get("prompt"),
                    messages=data.get("messages"),
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
            "tools": [...],              // 可选，OpenAI 格式工具定义
            "tool_choice": "auto",       // 可选
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9
        }
    """
    data = request.json or {}
    vendor_id = data.get("vendor", "").strip()
    model = data.get("model", "").strip()
    messages = data.get("messages", [])
    tools = data.get("tools")
    tool_choice = data.get("tool_choice", "auto")

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
                    tools=tools,
                    tool_choice=tool_choice,
                ):
                    if isinstance(chunk, dict):
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    else:
                        yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                log.error("厂商流式错误: %s", e)
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    except Exception as e:
        log.error("厂商聊天失败: %s", e)
        return _err(str(e), 500)


# ═══════════════════════════════════════════════════════════════════════════════
# 会话持久化 API（从 hermes-agent 提取）
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    store = get_store()
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    sessions = store.list_sessions(limit=limit, offset=offset)
    return jsonify({"sessions": sessions})


@app.route("/api/sessions", methods=["POST"])
def create_session():
    data = request.get_json(silent=True) or {}
    store = get_store()
    sid = store.create_session(
        title=data.get("title"),
        backend=data.get("backend"),
        model=data.get("model"),
    )
    return jsonify(store.get_session(sid))


@app.route("/api/sessions/<session_id>", methods=["GET"])
def get_session(session_id):
    store = get_store()
    s = store.get_session(session_id)
    if not s:
        return _err("会话不存在", 404)
    return jsonify(s)


@app.route("/api/sessions/<session_id>", methods=["PATCH"])
def update_session(session_id):
    data = request.get_json(silent=True) or {}
    store = get_store()
    store.update_session(session_id, **data)
    s = store.get_session(session_id)
    if not s:
        return _err("会话不存在", 404)
    return jsonify(s)


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    store = get_store()
    if not store.delete_session(session_id):
        return _err("会话不存在", 404)
    return jsonify({"ok": True})


@app.route("/api/sessions/<session_id>/messages", methods=["GET"])
def get_session_messages(session_id):
    store = get_store()
    messages = store.get_messages_as_conversation(session_id)
    return jsonify({"messages": messages, "count": len(messages)})


@app.route("/api/sessions/<session_id>/messages", methods=["POST"])
def append_message(session_id):
    data = request.get_json(silent=True) or {}
    store = get_store()
    msg_id = store.append_message(
        session_id=session_id,
        role=data["role"],
        content=data.get("content"),
        tool_name=data.get("tool_name"),
        token_count=data.get("token_count"),
        finish_reason=data.get("finish_reason"),
        reasoning_content=data.get("reasoning_content"),
    )
    return jsonify({"id": msg_id, "ok": True})


@app.route("/api/sessions/<session_id>/messages", methods=["DELETE"])
def clear_session_messages(session_id):
    store = get_store()
    store.clear_messages(session_id)
    return jsonify({"ok": True})


@app.route("/api/messages/search", methods=["POST"])
def search_messages():
    data = request.get_json(silent=True) or {}
    store = get_store()
    results = store.search_messages(
        query=data.get("query", ""),
        limit=data.get("limit", 20),
        session_id=data.get("session_id"),
    )
    return jsonify({"results": results, "count": len(results)})


# ═══════════════════════════════════════════════════════════════════════════════
# 记忆 API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/memory", methods=["GET"])
def list_memory():
    store = get_store()
    items = store.list_memory(category=request.args.get("category"))
    return jsonify({"memories": items})


@app.route("/api/memory", methods=["POST"])
def save_memory():
    data = request.get_json(silent=True) or {}
    store = get_store()
    store.save_memory(
        key=data["key"],
        value=data["value"],
        category=data.get("category", "general"),
    )
    return jsonify(store.get_memory(data["key"]))


@app.route("/api/memory/<key>", methods=["GET"])
def get_memory_item(key):
    store = get_store()
    m = store.get_memory(key)
    if not m:
        return _err("未找到", 404)
    return jsonify(m)


@app.route("/api/memory/<key>", methods=["DELETE"])
def delete_memory_item(key):
    store = get_store()
    if not store.delete_memory(key):
        return _err("未找到", 404)
    return jsonify({"ok": True})


@app.route("/api/memory/search", methods=["POST", "GET"])
def search_memory_items():
    if request.method == "GET":
        query = request.args.get("q", "")
        limit = int(request.args.get("limit", 10))
    else:
        data = request.get_json(silent=True) or {}
        query = data.get("query", "")
        limit = data.get("limit", 10)
    store = get_store()
    results = store.search_memory(query=query, limit=limit)
    return jsonify({"results": results})


# ═══════════════════════════════════════════════════════════════════════════════
# 统计 API
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 上下文压缩 API（集成辅助模型）
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/compress", methods=["POST"])
def api_compress():
    """压缩对话上下文

    请求体:
        {
            "messages": [...],           // 完整消息列表
            "context_length": 131072,     // 可选，模型上下文窗口大小
            "threshold_percent": 0.50,    // 可选，触发压缩的阈值比例
            "provider": "openai",         // 可选，显式指定压缩模型厂商
            "model": "gpt-4o-mini",       // 可选，显式指定压缩模型
            "api_key": "sk-...",          // 可选，显式指定 API Key
            "base_url": "https://...",    // 可选，显式指定 Base URL
            "focus_topic": "",            // 可选，聚焦压缩主题
            "memory_context": "",         // 可选，记忆上下文注入
        }

    返回:
        {"compressed": [...], "original_count": 20, "compressed_count": 5,
         "saved_tokens": 12000, "was_compressed": true}
    """
    data = request.json or {}
    messages = data.get("messages", [])

    if not messages:
        return _err("缺少 messages 参数")

    context_length = data.get("context_length", 131072)
    threshold_percent = float(data.get("threshold_percent", 0.50))

    try:
        from services.compressor_manager import get_compressor

        compressor = get_compressor(
            context_length=context_length,
            threshold_percent=threshold_percent,
            quiet_mode=False,
        )

        original_count = len(messages)

        compressed = compressor.compress(
            messages=messages,
            summary_model=data.get("model", ""),
            base_url=data.get("base_url", ""),
            api_key=data.get("api_key", ""),
            focus_topic=data.get("focus_topic", ""),
            memory_context=data.get("memory_context", ""),
        )

        compressed_count = len(compressed)
        was_compressed = compressed_count != original_count

        # 粗略估算节省的 token
        from services.context_compressor import estimate_messages_tokens_rough
        original_tokens = estimate_messages_tokens_rough(messages)
        compressed_tokens = estimate_messages_tokens_rough(compressed)
        saved_tokens = original_tokens - compressed_tokens

        log.info("压缩完成: %d→%d 条消息, 节省 ~%d tokens",
                 original_count, compressed_count, max(0, saved_tokens))

        return jsonify({
            "compressed": compressed,
            "original_count": original_count,
            "compressed_count": compressed_count,
            "saved_tokens": max(0, saved_tokens),
            "was_compressed": was_compressed,
        })
    except Exception as e:
        log.error("压缩失败: %s", e)
        return _err(str(e), 500)


@app.route("/api/stats", methods=["GET"])
def stats():
    store = get_store()
    return jsonify(store.get_stats())


# ═══════════════════════════════════════════════════════════════════════════════
# 工具系统 API
# ═══════════════════════════════════════════════════════════════════════════════

# 懒加载：首次请求时发现并注册工具
_tools_initialized = False


def _ensure_tools():
    """确保工具已发现并注册"""
    global _tools_initialized
    if not _tools_initialized:
        from tools.registry import discover_tools, get_registry
        discover_tools()
        _tools_initialized = True


@app.route("/api/tools/list", methods=["GET"])
def api_tools_list():
    """列出所有可用工具及其 schema"""
    _ensure_tools()
    from tools.registry import get_registry

    registry = get_registry()
    tools = registry.list_available()

    return jsonify({
        "tools": [t.to_openai_schema() for t in tools],
        "count": len(tools),
    })


@app.route("/api/tools/execute", methods=["POST"])
def api_tools_execute():
    """执行一个工具调用

    请求体:
        {
            "name": "web_search",
            "params": {"query": "latest AI news", "max_results": 5}
        }

    返回:
        {"result": "...", "tool": "web_search"}
    """
    _ensure_tools()
    from tools.registry import get_registry
    import asyncio

    data = request.json or {}
    name = data.get("name", "")
    params = data.get("params", {})

    if not name:
        return _err("缺少工具名称")

    registry = get_registry()

    # Sync execution (most handlers are sync)
    result = registry.execute(name, params)
    # If async, run in event loop
    if hasattr(result, "__await__"):
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(result)
        finally:
            loop.close()

    return jsonify({
        "result": str(result),
        "tool": name,
    })


@app.route("/api/tools/prompt", methods=["GET"])
def api_tools_prompt():
    """获取 llama-cpp 工具提示词（XML 格式指南）"""
    try:
        _ensure_tools()
        from tools.registry import get_registry
        from environments.tool_parser import build_tool_prompt
        registry = get_registry()
        schemas = registry.get_schemas()
        prompt = build_tool_prompt(schemas)
        return jsonify({"prompt": prompt, "tool_count": len(schemas)})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# 技能系统 API
# ═══════════════════════════════════════════════════════════════════════════════

_skills_cache = None


def _get_skills():
    """获取所有技能（带缓存）"""
    global _skills_cache
    if _skills_cache is None:
        from services.skill_loader import load_all_skills
        _skills_cache = load_all_skills()
    return _skills_cache


@app.route("/api/skills/list", methods=["GET"])
def api_skills_list():
    """列出所有可用技能"""
    skills = _get_skills()
    return jsonify({
        "skills": [s.to_dict() for s in skills.values()],
        "count": len(skills),
    })


@app.route("/api/skills/<skill_id>", methods=["GET"])
def api_skills_get(skill_id):
    """获取技能的完整内容"""
    skills = _get_skills()
    skill = skills.get(skill_id)
    if not skill:
        return _err(f"技能不存在: {skill_id}", 404)
    return jsonify({
        "name": skill.name,
        "description": skill.description,
        "priority": skill.priority,
        "tools": skill.tools,
        "content": skill.content,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 审批流 API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/approval/status", methods=["GET"])
def api_approval_status():
    """获取当前会话的审批状态"""
    from tools.approval import (
        get_current_session_key,
        is_current_session_yolo_enabled,
        get_pending_approvals,
    )

    session_key = request.args.get("session_key") or get_current_session_key()
    return jsonify({
        "session_key": session_key,
        "yolo_enabled": is_current_session_yolo_enabled(),
        "pending": get_pending_approvals(session_key),
    })


@app.route("/api/approval/pending", methods=["GET"])
def api_approval_pending():
    """列出等待审批的请求（网关模式用）"""
    from tools.approval import get_pending_approvals

    session_key = request.args.get("session_key", "")
    pending = get_pending_approvals(session_key)
    return jsonify({
        "pending": pending,
        "count": len(pending),
    })


@app.route("/api/approval/approve", methods=["POST"])
def api_approval_approve():
    """批准一个等待中的请求"""
    from tools.approval import approve_request

    data = request.json or {}
    request_id = data.get("request_id", "")
    session_key = data.get("session_key", "")

    if not request_id:
        return _err("缺少 request_id", 400)

    success = approve_request(request_id, session_key)
    return jsonify({"status": "approved" if success else "not_found"})


@app.route("/api/approval/deny", methods=["POST"])
def api_approval_deny():
    """拒绝一个等待中的请求"""
    from tools.approval import deny_request

    data = request.json or {}
    request_id = data.get("request_id", "")
    session_key = data.get("session_key", "")

    if not request_id:
        return _err("缺少 request_id", 400)

    success = deny_request(request_id, session_key)
    return jsonify({"status": "denied" if success else "not_found"})


@app.route("/api/approval/yolo", methods=["POST"])
def api_approval_yolo():
    """为当前会话启用 YOLO 模式（跳过非 Hardline 审批）"""
    from tools.approval import enable_yolo_for_session, get_current_session_key

    data = request.json or {}
    session_key = data.get("session_key") or get_current_session_key()
    enable = data.get("enable", True)

    if enable:
        enable_yolo_for_session(session_key)
        return jsonify({"status": "yolo_enabled", "session_key": session_key})
    else:
        from tools.approval import disable_yolo_for_session
        disable_yolo_for_session(session_key)
        return jsonify({"status": "yolo_disabled", "session_key": session_key})


@app.route("/api/approval/allowlist", methods=["GET", "POST", "DELETE"])
def api_approval_allowlist():
    """管理永久白名单"""
    from tools.approval import (
        load_permanent_allowlist,
        save_permanent_allowlist,
        add_to_allowlist,
        remove_from_allowlist,
    )

    if request.method == "GET":
        return jsonify({"allowlist": list(load_permanent_allowlist())})

    elif request.method == "POST":
        data = request.json or {}
        command = data.get("command", "").strip()
        if not command:
            return _err("缺少 command", 400)
        add_to_allowlist(command)
        return jsonify({"status": "added", "command": command})

    else:  # DELETE
        data = request.json or {}
        command = data.get("command", "").strip()
        if not command:
            return _err("缺少 command", 400)
        remove_from_allowlist(command)
        return jsonify({"status": "removed", "command": command})


# ═══════════════════════════════════════════════════════════════════════════════
# 技能管理 API（RESTful）
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/skills", methods=["GET"])
def api_skills_all():
    """列出所有技能（RESTful 风格）"""
    skills = _get_skills()
    return jsonify({
        "skills": [s.to_dict() for s in skills.values()],
        "count": len(skills),
    })


@app.route("/api/skills", methods=["POST"])
def api_skills_create():
    """创建新技能"""
    data = request.json or {}
    name = data.get("name", "").strip()
    description = data.get("description", "").strip()
    content = data.get("content", "").strip()
    priority = data.get("priority", 0)
    tools = data.get("tools", [])

    if not name or not content:
        return _err("名称和内容不能为空", 400)

    try:
        from tools.skill_tool import skill_create
        result = skill_create(
            name=name,
            description=description,
            content=content,
            priority=priority,
            tools=tools,
        )
        return jsonify(json.loads(result))
    except Exception as e:
        return _err(f"创建技能失败: {e}", 500)


@app.route("/api/skills/<skill_id>", methods=["DELETE"])
def api_skills_delete(skill_id):
    """删除技能"""
    try:
        from tools.skill_tool import skill_delete
        result = skill_delete(name=skill_id)
        return jsonify(json.loads(result))
    except Exception as e:
        return _err(f"删除技能失败: {e}", 500)


# ═══════════════════════════════════════════════════════════════════════════════
# 进程管理 API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/processes", methods=["GET"])
def api_processes_list():
    """列出所有后台进程"""
    try:
        from tools.process_registry import process_registry
        sessions = process_registry.list_sessions()
        return jsonify({
            "processes": [
                {
                    "session_id": s.get("session_id", ""),
                    "pid": s.get("pid"),
                    "command": s.get("command", ""),
                    "status": s.get("status", "unknown"),
                    "exit_code": s.get("exit_code"),
                    "created_at": s.get("created_at"),
                }
                for s in sessions
            ],
            "count": len(sessions),
        })
    except Exception as e:
        return _err(f"获取进程列表失败: {e}", 500)


@app.route("/api/processes/<session_id>/kill", methods=["POST"])
def api_process_kill(session_id):
    """终止指定进程"""
    try:
        from tools.process_registry import process_registry
        process_registry.kill_process(session_id)
        return jsonify({"status": "killed", "session_id": session_id})
    except Exception as e:
        return _err(f"终止进程失败: {e}", 500)


@app.route("/api/processes/<session_id>/log", methods=["GET"])
def api_process_log(session_id):
    """获取进程日志"""
    try:
        from tools.process_registry import process_registry
        log = process_registry.read_log(session_id)
        return jsonify({"log": log, "session_id": session_id})
    except Exception as e:
        return _err(f"获取日志失败: {e}", 500)


# 注册 Agent 蓝图
app.register_blueprint(agent_bp)

# ═══════════════════════════════════════════════════════════════════════════════
# Cron 定时任务 API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/cron/jobs", methods=["GET"])
def api_cron_list():
    """列出所有定时任务"""
    try:
        from cron.scheduler import get_scheduler
        scheduler = get_scheduler()
        jobs = scheduler.list_jobs()
        return jsonify({
            "jobs": [j.to_dict() for j in jobs],
            "count": len(jobs),
        })
    except Exception as e:
        log.error("获取定时任务失败: %s", e)
        return _err(str(e), 500)


@app.route("/api/cron/jobs", methods=["POST"])
def api_cron_create():
    """创建定时任务"""
    data = request.json or {}
    try:
        from cron.jobs import CronJob
        from cron.scheduler import get_scheduler

        job = CronJob(
            name=data.get("name", ""),
            description=data.get("description", ""),
            schedule=data.get("schedule", "0 * * * *"),
            command=data.get("command", ""),
            enabled=data.get("enabled", True),
            use_agent=data.get("use_agent", True),
            vendor_id=data.get("vendor_id", ""),
            model=data.get("model", ""),
            env=data.get("env", {}),
        )
        scheduler = get_scheduler()
        scheduler.add_job(job)
        return jsonify({"status": "created", "job": job.to_dict()})
    except Exception as e:
        log.error("创建定时任务失败: %s", e)
        return _err(str(e), 500)


@app.route("/api/cron/jobs/<job_id>", methods=["GET"])
def api_cron_get(job_id):
    """获取单个任务"""
    try:
        from cron.scheduler import get_scheduler
        scheduler = get_scheduler()
        job = scheduler.get_job(job_id)
        if not job:
            return _err("任务不存在", 404)
        return jsonify({"job": job.to_dict()})
    except Exception as e:
        return _err(str(e), 500)


@app.route("/api/cron/jobs/<job_id>", methods=["PATCH"])
def api_cron_update(job_id):
    """更新定时任务"""
    data = request.json or {}
    try:
        from cron.scheduler import get_scheduler
        scheduler = get_scheduler()
        job = scheduler.update_job(job_id, **data)
        if not job:
            return _err("任务不存在", 404)
        return jsonify({"status": "updated", "job": job.to_dict()})
    except Exception as e:
        log.error("更新定时任务失败: %s", e)
        return _err(str(e), 500)


@app.route("/api/cron/jobs/<job_id>", methods=["DELETE"])
def api_cron_delete(job_id):
    """删除定时任务"""
    try:
        from cron.scheduler import get_scheduler
        scheduler = get_scheduler()
        if scheduler.delete_job(job_id):
            return jsonify({"status": "deleted"})
        return _err("任务不存在", 404)
    except Exception as e:
        return _err(str(e), 500)


# ═══════════════════════════════════════════════════════════════════════════════
# 插件系统 API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/plugins", methods=["GET"])
def api_plugins_list():
    """列出所有已加载插件"""
    try:
        from plugins.base import PluginManager
        manager = PluginManager()
        plugins = manager.list_plugins()
        return jsonify({
            "plugins": [
                {
                    "name": p.name,
                    "version": p.version,
                    "description": p.description,
                    "author": p.author,
                    "enabled": p.enabled,
                }
                for p in plugins
            ],
        })
    except Exception as e:
        log.error("获取插件列表失败: %s", e)
        return _err(str(e), 500)


@app.route("/api/plugins/discover", methods=["GET"])
def api_plugins_discover():
    """发现可用插件"""
    try:
        from plugins.base import PluginManager
        manager = PluginManager()
        classes = manager.discover()
        return jsonify({
            "plugins": [
                {
                    "name": cls.name or cls.__name__,
                    "description": cls.description,
                    "version": cls.version,
                }
                for cls in classes
            ],
        })
    except Exception as e:
        log.error("发现插件失败: %s", e)
        return _err(str(e), 500)


@app.route("/api/plugins/<name>/toggle", methods=["POST"])
def api_plugins_toggle(name):
    """启用/禁用插件"""
    try:
        from plugins.base import PluginManager
        manager = PluginManager()
        plugin = manager.get(name)
        if not plugin:
            return _err("插件不存在", 404)
        plugin.enabled = not plugin.enabled
        return jsonify({"status": "ok", "enabled": plugin.enabled})
    except Exception as e:
        return _err(str(e), 500)


# ═══════════════════════════════════════════════════════════════════════════════
# 记忆提供者 API（插件化）
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/memory/provider", methods=["GET"])
def api_memory_provider():
    """获取当前记忆提供者状态"""
    try:
        from plugins.memory.local import LocalMemoryPlugin
        plugin = LocalMemoryPlugin()
        profile = plugin.get_user_profile()
        return jsonify({
            "provider": "local",
            "profile": profile,
        })
    except Exception as e:
        log.error("获取记忆提供者失败: %s", e)
        return _err(str(e), 500)


@app.route("/api/memory/provider/search", methods=["POST"])
def api_memory_provider_search():
    """搜索长期记忆"""
    data = request.json or {}
    query = data.get("query", "").strip()
    limit = data.get("limit", 5)
    if not query:
        return _err("缺少查询参数", 400)
    try:
        from plugins.memory.local import LocalMemoryPlugin
        plugin = LocalMemoryPlugin()
        results = plugin.search(query, limit=limit)
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        log.error("搜索记忆失败: %s", e)
        return _err(str(e), 500)


# ═══════════════════════════════════════════════════════════════════════════════
# 多 Agent 协作 API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/agents/team", methods=["POST"])
def api_agent_team():
    """多 Agent 协作任务

    请求体:
        {
            "task": "分析代码并生成测试",
            "agents": [
                {"role": "analyzer", "prompt": "你是一个代码分析专家..."},
                {"role": "tester", "prompt": "你是一个测试专家..."}
            ],
            "vendor_id": "deepseek",
            "model": "deepseek-chat"
        }
    """
    data = request.json or {}
    task = data.get("task", "").strip()
    agents = data.get("agents", [])
    if not task:
        return _err("缺少 task 参数", 400)
    if not agents:
        return _err("缺少 agents 参数", 400)

    try:
        # 简单的顺序执行模式
        results = []
        context = {"task": task, "previous_results": []}

        for agent_def in agents:
            role = agent_def.get("role", "assistant")
            prompt = agent_def.get("prompt", "")

            # 构建消息
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"任务: {task}\n\n前文结果:\n" + "\n".join(context["previous_results"])},
            ]

            # 调用厂商 API
            from backends.vendors import chat_stream
            content = ""
            for chunk in chat_stream(
                vendor_id=data.get("vendor_id", "deepseek"),
                model=data.get("model", "deepseek-chat"),
                messages=messages,
                api_key=data.get("api_key", ""),
                base_url=data.get("base_url", ""),
                max_tokens=data.get("max_tokens", 4096),
                temperature=0.7,
            ):
                if isinstance(chunk, dict):
                    content += chunk.get("content", "")
                else:
                    content += str(chunk)

            results.append({"role": role, "content": content})
            context["previous_results"].append(f"[{role}]\n{content}")

        return jsonify({
            "task": task,
            "results": results,
            "agent_count": len(agents),
        })
    except Exception as e:
        log.error("多 Agent 协作失败: %s", e)
        return _err(str(e), 500)


# ═══════════════════════════════════════════════════════════════════════════════
# 监控与日志 API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """获取性能监控指标"""
    try:
        from services.performance_monitor import get_monitor
        monitor = get_monitor()
        metrics = monitor.get_metrics()

        # 补充系统信息
        metrics["system"] = {
            "backend": _current_backend,
            "uptime": time.time() - getattr(api_metrics, "_start_time", time.time()),
            "version": "2.0.0",
            "cpu_mode": _cpu_mode,
        }

        # 工具统计
        registry = get_registry()
        metrics["tools"] = {
            "registered": len(registry.tools),
            "calls_today": 0,  # TODO: 从持久化存储读取
            "success_rate": 1.0,
        }

        return jsonify({"metrics": metrics})
    except Exception as e:
        log.error("获取监控指标失败: %s", e)
        return jsonify({
            "metrics": {
                "latency": {"p50": 0, "p95": 0, "p99": 0},
                "throughput": {"rps": 0, "tps": 0, "total_requests": 0},
                "errors": {"rate": 0, "total": 0, "recent": 0},
                "cache": {"hit_rate": 0, "memory_items": 0, "disk_items": 0},
                "system": {"backend": _current_backend, "uptime": 0, "version": "2.0.0"},
                "tools": {"registered": 0, "calls_today": 0, "success_rate": 1.0},
            }
        })

# 记录启动时间
api_metrics._start_time = time.time()


@app.route("/api/logs", methods=["GET"])
def api_logs():
    """获取应用日志"""
    try:
        level = request.args.get("level", "all")
        limit = int(request.args.get("limit", 200))

        # 从日志文件读取（如果配置了文件日志）
        log_entries = []
        log_file = os.path.join(os.path.dirname(__file__), "logs", "app.log")
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            # 解析简单日志格式
            for line in lines[-limit:]:
                line = line.strip()
                if not line:
                    continue
                # 尝试解析: 2026-06-01 12:00:00,000 - logger - LEVEL - message
                import re
                match = re.match(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[\d,]*)\s+-\s+(\w+)\s+-\s+(\w+)\s+-\s+(.*)", line)
                if match:
                    timestamp_str, logger_name, log_level, message = match.groups()
                    if level != "all" and log_level.lower() != level.lower():
                        continue
                    log_entries.append({
                        "timestamp": timestamp_str,
                        "logger": logger_name,
                        "level": log_level.lower(),
                        "message": message,
                    })
                else:
                    # 无法解析的行作为原始消息
                    if level == "all":
                        log_entries.append({
                            "timestamp": "",
                            "logger": "app",
                            "level": "info",
                            "message": line,
                        })

        return jsonify({"logs": log_entries[-limit:], "count": len(log_entries)})
    except Exception as e:
        log.error("获取日志失败: %s", e)
        return jsonify({"logs": [], "count": 0})


# 注册 Agent 蓝图（幂等：避免重复注册）
if "agent" not in app.blueprints:
    app.register_blueprint(agent_bp)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
