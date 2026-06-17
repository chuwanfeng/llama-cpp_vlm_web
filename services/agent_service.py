"""services/agent_service.py -- AgentLoop Flask 路由层"""
import asyncio
import json
import logging
import os
import queue
import threading

from flask import Blueprint, request, jsonify, Response, stream_with_context
from tools.registry import get_registry
from agent.loop import AgentLoop
from services.prompt_builder import build_system_prompt
from services.context_compressor import ContextCompressor

logger = logging.getLogger(__name__)

# 使用全局变量避免重复创建 Blueprint（测试时可能多次导入）
_bp = None

def get_bp():
    global _bp
    if _bp is None:
        _bp = Blueprint("agent", __name__, url_prefix="/api/agent")
    return _bp

bp = get_bp()


def _make_call_llm_fn(vendor_id: str, api_key: str, base_url: str):
    """创建上下文压缩器使用的 LLM 调用函数（同步）
    
    用于 ContextCompressor 生成摘要时调用厂商 API。
    返回的函数签名：call_llm(**kwargs) -> str
    """
    from backends import vendors
    
    def call_llm(**kwargs):
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "")
        max_tokens = kwargs.get("max_tokens", 1024)
        
        stream = vendors.chat_stream(
            vendor_id=vendor_id,
            model=model,
            messages=messages,
            tools=None,  # 摘要不需要工具
            temperature=0.3,  # 低温度适合摘要
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )
        
        content = ""
        for chunk in stream:
            if isinstance(chunk, dict):
                if "content" in chunk:
                    content += chunk["content"]
        return content.strip()
    
    return call_llm


# ---------- helpers ----------

def _get_vendor_creds(vendor_id: str) -> dict:
    """从 settings.json 读取指定厂商的凭据（api_key, base_url）— 带缓存"""
    from services.cache_manager import get_cache
    cache = get_cache()
    cache_key = f"vendor_creds:{vendor_id}"
    
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    
    settings_path = os.path.join(os.path.dirname(__file__), "..", "settings.json")
    settings_path = os.path.normpath(settings_path)
    if not os.path.exists(settings_path):
        return {}
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        creds = s.get("vendor_creds", {}).get(vendor_id, {})
        result = {
            "api_key": creds.get("api_key", ""),
            "base_url": creds.get("base_url", ""),
        }
        cache.set(cache_key, result, persist=False)
        return result
    except Exception:
        return {}


def _build_loop(data: dict, messages: list) -> AgentLoop:
    """根据请求参数构建 AgentLoop 实例"""
    registry = get_registry()
    tool_schemas = [t.to_openai_schema() for t in registry.list_available()]
    valid_tool_names = {t.name for t in registry.list_available()}

    vendor_id = data.get("vendor_id", "")

    # Exclude external web_search if vendor has built-in search
    _SEARCH_VENDORS = {"zhipu", "moonshot"}
    if vendor_id in _SEARCH_VENDORS:
        tool_schemas = [t for t in tool_schemas if t.get("function", {}).get("name") != "web_search"]
        valid_tool_names = {n for n in valid_tool_names if n != "web_search"}

    api_key = data.get("api_key", "")
    base_url = data.get("base_url", "")
    if not api_key and vendor_id:
        creds = _get_vendor_creds(vendor_id)
        api_key = api_key or creds.get("api_key", "")
        base_url = base_url or creds.get("base_url", "")

    # 创建上下文压缩器（有 vendor_id + api_key 时可用，不限后端类型）
    compressor = None
    if vendor_id and api_key:
        try:
            call_llm_fn = _make_call_llm_fn(vendor_id, api_key, base_url)
            compressor = ContextCompressor(
                call_llm_fn=call_llm_fn,
                context_length=131072,  # 默认 128K 上下文窗口
                model=data.get("model", ""),
            )
        except Exception as e:
            logger.warning("无法创建上下文压缩器: %s", e)

    return AgentLoop(
        backend_type=data.get("backend_type", "vendor"),
        tool_schemas=tool_schemas,
        valid_tool_names=valid_tool_names,
        vendor_id=vendor_id,
        model=data.get("model", ""),
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens"),
        compressor=compressor,
        api_key=api_key,
        base_url=base_url,
        tool_choice=data.get("tool_choice", "auto"),
        plan_mode=data.get("plan_mode", False),
        web_search=data.get("web_search", False),  # 联网搜索开关（控制厂商原生搜索）
        think_output=data.get("think_output", True),
        auto_review=data.get("auto_review", False),
        ctx_ext=data.get("ctx_ext", True),
        min_prompt=data.get("min_prompt", True),
    )


def _inject_system_prompt(messages: list, tool_schemas: list[dict], cwd: str = None, min_prompt: bool = True) -> list:
    """如果 messages 中没有 system 消息，自动构建一个包含工具描述的 system prompt。

    如果已有 system 消息，则在其内容后追加工具描述。

    min_prompt=True:  仅注入工具描述（极简模式）
    min_prompt=False: 注入完整系统提示（平台信息 + 工具描述）
    """
    import copy
    msgs = copy.deepcopy(messages)

    tools_prompt = build_system_prompt(
        tool_schemas=tool_schemas,
        cwd=cwd,
        include_soul=False,   # 身份由前端控制
        include_agents=False, # 规则由前端控制
        include_platform=not min_prompt,  # min_prompt=True 时省略平台信息
        include_tools=True,              # 工具描述始终需要
    )

    if not tools_prompt:
        return msgs

    # 检查是否已有 system 消息
    for msg in msgs:
        if msg.get("role") == "system":
            msg["content"] = msg.get("content", "") + "\n\n" + tools_prompt
            return msgs

    # 没有 system 消息，插入到开头
    msgs.insert(0, {"role": "system", "content": tools_prompt})
    return msgs


# ---------- endpoints ----------

@bp.route("/chat", methods=["POST"])
def agent_chat():
    """AgentLoop 对话（非流式）"""
    data = request.json or {}
    messages = data.get("messages", [])

    if not messages:
        return jsonify({"error": "Missing messages"}), 400

    # 设置审批流会话键
    try:
        from tools.approval import set_current_session_key, get_current_session_key
        session_key = data.get("session_key", "") or get_current_session_key()
        set_current_session_key(session_key)
    except Exception:
        pass

    loop = _build_loop(data, messages)

    # 自动构建系统提示词（含工具描述 + 平台信息）
    tool_schemas = [t.to_openai_schema() for t in get_registry().list_available()]
    messages = _inject_system_prompt(messages, tool_schemas, cwd=data.get("cwd"), min_prompt=data.get("min_prompt", True))

    try:
        result = asyncio.run(loop.run(messages))
        return jsonify({
            "messages": result.messages,
            "turns_used": result.turns_used,
            "finished_naturally": result.finished_naturally,
            "reasoning_per_turn": result.reasoning_per_turn,
            "tool_errors": [
                {"turn": e.turn, "tool_name": e.tool_name, "error": e.error}
                for e in result.tool_errors
            ],
        })
    except Exception as e:
        logger.error("AgentLoop 失败: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@bp.route("/chat/stream", methods=["POST"])
def agent_chat_stream():
    """AgentLoop 对话（SSE 流式）

    数据流格式（SSE）：
      data: {"type":"token","content":"..."}
      data: {"type":"tool_call","name":"read_file","args":{...}}
      data: {"type":"tool_result","name":"read_file","content":"..."}
      data: {"type":"done","turns_used":3}
      data: [DONE]
    """
    print("=== agent_chat_stream CALLED ===", flush=True)
    logger.info("=== agent_chat_stream: vendor=%s model=%s", request.json.get("vendor_id"), request.json.get("model"))
    data = request.json or {}
    messages = data.get("messages", [])

    if not messages:
        return jsonify({"error": "Missing messages"}), 400

    # 确保工具已加载（懒加载）— 包含 MCP 服务器发现
    from tools.registry import discover_tools
    discover_tools()

    # 记录 MCP 状态
    try:
        from tools.mcp_tool import get_mcp_status
        mcp_status = get_mcp_status()
        if mcp_status:
            logger.info("MCP 状态: %s", mcp_status)
    except Exception:
        pass

    loop = _build_loop(data, messages)

    # 自动构建系统提示词（含工具描述 + 平台信息）
    tool_schemas = [t.to_openai_schema() for t in get_registry().list_available()]
    messages = _inject_system_prompt(messages, tool_schemas, cwd=data.get("cwd"), min_prompt=data.get("min_prompt", True))

    def generate():
        event_queue = queue.Queue()

        def on_token(content):
            event_queue.put({"type": "token", "content": content})

        def on_tool_call(name, args):
            # ── 审批流检查（危险命令拦截）──
            # 仅对 terminal 工具执行危险命令检测
            if name == "terminal" and isinstance(args, dict):
                command = args.get("command", "")
                if command:
                    try:
                        from tools.approval import check_dangerous_command, get_current_session_key, set_current_session_key
                        # 设置当前会话键（用于会话级审批缓存）
                        session_key = data.get("session_key", "") or get_current_session_key()
                        set_current_session_key(session_key)

                        # 检查命令是否危险
                        approval_result = check_dangerous_command(
                            command,
                            env_type=args.get("env_type", "local"),
                            call_llm_fn=_make_call_llm_fn(
                                data.get("vendor_id", ""),
                                data.get("api_key", ""),
                                data.get("base_url", ""),
                            ) if data.get("vendor_id") and data.get("api_key") else None,
                        )
                        if not approval_result.get("approved", True):
                            # 命令被阻止，发送审批事件并跳过执行
                            event_queue.put({
                                "type": "approval_required",
                                "tool": name,
                                "command": command,
                                "description": approval_result.get("description", ""),
                                "message": approval_result.get("message", ""),
                                "status": approval_result.get("status", "blocked"),
                            })
                            # 不发送 tool_call 事件，AgentLoop 会收到空结果
                            return
                    except Exception as e:
                        logger.warning("审批流检查失败: %s", e)

            event_queue.put({"type": "tool_call", "name": name, "args": args})

        def on_tool_result(name, result):
            # 确保 content 始终是有效 JSON（前端会 JSON.parse）
            if not result:
                content = json.dumps({"success": True, "result": ""})
            else:
                try:
                    json.loads(result)
                    content = result  # 已是有效 JSON
                except (json.JSONDecodeError, TypeError):
                    # 非 JSON 文本（如 read_file 输出），包装为 JSON
                    content = json.dumps({"success": True, "content": result}, ensure_ascii=False)
            event_queue.put({"type": "tool_result", "name": name, "content": content})

        def on_reasoning(content):
            event_queue.put({"type": "reasoning", "content": content})

        def run_loop():
            try:
                # Windows 上后台线程中 asyncio.run() 可能与已有事件循环冲突
                # 使用 new_event_loop() + set_event_loop() + loop.run_until_complete() 替代
                _loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_loop)
                result = _loop.run_until_complete(loop.run(
                    messages,
                    on_token=on_token,
                    on_tool_call=on_tool_call,
                    on_tool_result=on_tool_result,
                    on_reasoning=on_reasoning,
                ))
                _loop.close()
                event_queue.put({
                    "type": "done",
                    "turns_used": result.turns_used,
                    "finished_naturally": result.finished_naturally,
                    "reasoning_per_turn": result.reasoning_per_turn,
                })

                # ── Plan 模式：发送计划事件供前端渲染 ──
                if result.plan and not result.plan_approved:
                    event_queue.put({
                        "type": "plan",
                        "items": result.plan.get("items", []),
                        "assistant_message": result.plan.get("assistant_message", ""),
                        "turn": result.plan.get("turn", 1),
                    })

                # ── 后台自我进化 Review ──
                # 移植自 hermes-agent _spawn_background_review
                # 仅在对话自然完成 + auto_review 开关启用时触发
                if loop.auto_review and result.finished_naturally and len(messages) >= 2:
                    try:
                        from agent.self_improve.review import run_review_sync
                        snapshot = list(result.messages) if result.messages else list(messages)
                        # Review 使用与主对话相同的厂商和模型配置
                        _review_vendor_id = data.get("vendor_id", "")
                        _review_model = data.get("model") or loop.model
                        review_result = run_review_sync(
                            messages_snapshot=snapshot,
                            backend_type=data.get("backend_type", "vendor"),
                            vendor_id=_review_vendor_id,
                            model=_review_model,
                            review_memory=True,
                            review_skills=True,
                            api_key=data.get("api_key", "") or _get_vendor_creds(_review_vendor_id).get("api_key", ""),
                            base_url=data.get("base_url"),
                        )
                        if review_result.summary:
                            event_queue.put({
                                "type": "review",
                                "summary": review_result.summary,
                                "actions": review_result.actions,
                            })
                    except Exception as e:
                        logger.warning("Background review 失败: %s", e)

            except Exception as e:
                logger.error("AgentLoop 流式失败: %s", e, exc_info=True)
                event_queue.put({"type": "error", "content": str(e)})
            finally:
                event_queue.put(None)  # Sentinel

        thread = threading.Thread(target=run_loop, daemon=True, name="agent-loop")
        thread.start()

        while True:
            try:
                event = event_queue.get(timeout=60)  # 60秒超时
                if event is None:
                    logger.info("SSE: sentinel received, ending stream")
                    break
                event_json = json.dumps(event, ensure_ascii=False)
                logger.debug("SSE: yielding event type=%s, len=%d", event.get("type"), len(event_json))
                yield f"data: {event_json}\n\n"
            except queue.Empty:
                logger.warning("SSE: event_queue 超时 60s，结束流")
                yield f"data: {json.dumps({'type': 'error', 'content': 'Stream timeout'}, ensure_ascii=False)}\n\n"
                break

        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@bp.route("/chat/plan/execute", methods=["POST"])
def agent_plan_execute():
    """执行已审批的 Plan 计划项 — SSE 流式返回工具结果。

    请求体：
      {
        "vendor_id": "deepseek",        // 厂商 ID
        "model": "deepseek-chat",       // 模型名
        "plan_items": [{                 // 审批通过的计划项
          "tool": "web_search",
          "arguments": {"query": "..."}
        }, ...],
        "messages": [...],               // 原始对话（含 system prompt）
        "assistant_message": "我会...",   // Plan 时的助手消息文本
        "api_key": "",                   // 可选，从 settings.json 读取
      }

    SSE 流式返回:
      data: {"type":"plan_result","tool":"web_search","result":"..."}
      data: {"type":"plan_result","tool":"code_exec","result":"..."}
      data: {"type":"plan_execute_done","count":3}
      然后继续对话...
    """
    import uuid
    from tools.registry import get_registry

    data = request.json or {}
    plan_items = data.get("plan_items", [])
    messages = data.get("messages", [])
    assistant_message = data.get("assistant_message", "")

    if not plan_items:
        return jsonify({"error": "Missing plan_items"}), 400

    registry = get_registry()

    # 构建助手消息（含 tool_calls）加入对话历史
    tool_calls_stubs = []
    for i, item in enumerate(plan_items):
        tool_calls_stubs.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": item["tool"],
                "arguments": json.dumps(item["arguments"], ensure_ascii=False),
            },
        })

    def generate():
        event_queue = queue.Queue()

        def on_token(content):
            event_queue.put({"type": "token", "content": content})

        def on_tool_call(name, args):
            event_queue.put({"type": "tool_call", "name": name, "args": args})

        def on_tool_result(name, result):
            if not result:
                content = json.dumps({"success": True, "result": ""})
            else:
                try:
                    json.loads(result)
                    content = result
                except (json.JSONDecodeError, TypeError):
                    content = json.dumps({"success": True, "content": result}, ensure_ascii=False)
            event_queue.put({"type": "tool_result", "name": name, "content": content})

        def run_plan():
            try:
                # 1. 逐个执行计划项，返回结果
                executed = []
                for item in plan_items:
                    tool_name = item["tool"]
                    tool_args = item["arguments"]
                    try:
                        _loop_exec = asyncio.new_event_loop()
                        asyncio.set_event_loop(_loop_exec)
                        result = _loop_exec.run_until_complete(registry.execute(tool_name, tool_args))
                        _loop_exec.close()
                        executed.append({
                            "tool": tool_name,
                            "arguments": tool_args,
                            "result": result,
                        })
                        event_queue.put({
                            "type": "plan_result",
                            "tool": tool_name,
                            "arguments": tool_args,
                            "result": result,
                        })
                    except Exception as e:
                        logger.warning("Plan item execute 失败: %s", e)
                        executed.append({
                            "tool": tool_name,
                            "arguments": tool_args,
                            "result": json.dumps({"error": str(e)}),
                        })
                        event_queue.put({
                            "type": "plan_result",
                            "tool": tool_name,
                            "arguments": tool_args,
                            "result": json.dumps({"error": str(e)}),
                        })

                event_queue.put({
                    "type": "plan_execute_done",
                    "count": len(executed),
                })

                # 2. 构建带工具调用和结果的消息历史
                msgs = list(messages)  # copy

                # 添加助手消息（含 tool_calls）
                assistant_msg = {
                    "role": "assistant",
                    "content": assistant_message or "Executing plan...",
                }
                if tool_calls_stubs:
                    assistant_msg["tool_calls"] = tool_calls_stubs
                msgs.append(assistant_msg)

                # 添加每个工具的结果消息
                for i, ex in enumerate(executed):
                    tc_id = tool_calls_stubs[i]["id"] if i < len(tool_calls_stubs) else "call_unknown"
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": ex["result"],
                    })

                # 3. 用 AgentLoop 继续对话（plan_mode=False）
                loop = _build_loop(data, msgs)
                loop.plan_mode = False
                tool_schemas = [t.to_openai_schema() for t in registry.list_available()]
                msgs = _inject_system_prompt(msgs, tool_schemas, cwd=data.get("cwd"), min_prompt=data.get("min_prompt", True))

                _loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_loop)
                result = _loop.run_until_complete(loop.run(
                    msgs,
                    on_token=on_token,
                    on_tool_call=on_tool_call,
                    on_tool_result=on_tool_result,
                ))
                _loop.close()
                event_queue.put({
                    "type": "done",
                    "turns_used": result.turns_used,
                    "finished_naturally": result.finished_naturally,
                })
            except Exception as e:
                logger.error("Plan execute 失败: %s", e, exc_info=True)
                event_queue.put({"type": "error", "content": str(e)})
            finally:
                event_queue.put(None)

        thread = threading.Thread(target=run_plan, daemon=True, name="plan-executor")
        thread.start()

        while True:
            event = event_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
