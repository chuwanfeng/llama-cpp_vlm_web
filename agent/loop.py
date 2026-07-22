"""
Agent Loop 引擎核心

基于 hermes-agent 的 HermesAgentLoop 架构,适配 llama-cpp_vlm_web 项目。

核心功能:
- 多轮工具调用循环(Query Loop)
- 标准化 OpenAI 格式的 tool_calls 处理
- XML 工具调用回退解析(适用于不支持原生 function calling 的模型)
- 工具执行与错误处理
- 推理内容提取(reasoning_content/reasoning)

架构设计:
- AgentLoop: 主循环类,管理整个工具调用流程
- AgentResult: 返回结果数据结构
- ToolError: 工具执行错误记录
- ToolCallParser: XML 工具调用解析器(用于 llama-cpp 等不支持原生 tool calling 的后端)

参考:hermes-agent/environments/agent_loop.py
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, AsyncGenerator

# 本项目工具系统
from tools.registry import get_registry, get_tool_names
# 本项目后端
from backends import gpu, vendors
# 错误分类与重试(从 hermes-agent 移植)
from services.error_classifier import classify_api_error
from services.retry_utils import jittered_backoff

logger = logging.getLogger(__name__)


@dataclass
class ToolError:
    """
    工具执行错误记录。

    用于在 Agent Loop 中记录每次工具调用失败的信息,
    方便调试和结果分析。

    属性:
        turn: 第几轮调用(从 1 开始)
        tool_name: 被调用的工具名
        arguments: 传入的参数(截断到 200 字符)
        error: 错误信息
        tool_result: 工具返回的原始结果(截断到 500 字符)
    """
    turn: int
    tool_name: str
    arguments: str
    error: str
    tool_result: str


@dataclass
class AgentResult:
    """
    Agent Loop 执行结果。

    包含完整的对话历史、管理状态、元数据等信息。

    属性:
        messages: 完整的对话历史(OpenAI 格式)
        managed_state: 管理状态(Phase 2,当前未使用,保留接口)
        turns_used: 实际使用的轮次
        finished_naturally: 是否自然结束(vs 达到最大轮次)
        reasoning_per_turn: 每轮的推理内容列表
        tool_errors: 遇到的工具错误列表
    """
    messages: List[Dict[str, Any]]
    managed_state: Optional[Dict[str, Any]] = None
    turns_used: int = 0
    finished_naturally: bool = False
    reasoning_per_turn: List[Optional[str]] = field(default_factory=list)
    tool_errors: List[ToolError] = field(default_factory=list)
    plan: Optional[Dict[str, Any]] = None
    plan_approved: bool = False


def _extract_reasoning(message) -> Optional[str]:
    """
    从模型响应中提取推理内容(think 部分)。

    支持多种模型输出格式:
    1. message.reasoning_content 字段(大多数模型)
    2. message.reasoning 字段(部分模型)
    3. message.reasoning_details[].text(OpenRouter 风格)

    注意:<think> 块的提取不在这里处理,由响应解析器在更早阶段完成。

    参数:
        message: 助手消息对象(来自 API 响应)

    返回:
        提取的推理内容字符串,或 None(如果不存在)
    """
    # 方式 1: reasoning_content 字段(通用)
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        return message.reasoning_content

    # 方式 2: reasoning 字段
    if hasattr(message, "reasoning") and message.reasoning:
        return message.reasoning

    # 方式 3: reasoning_details(OpenRouter 风格)
    if hasattr(message, "reasoning_details") and message.reasoning_details:
        for detail in message.reasoning_details:
            if hasattr(detail, "text") and detail.text:
                return detail.text
            if isinstance(detail, dict) and detail.get("text"):
                return detail["text"]

    return None


class ToolCallParser:
    """
    XML 工具调用解析器。

    用于从模型输出的纯文本中提取工具调用。
    适用于不支持原生 function calling 的模型(如 llama-cpp)。

    支持的格式:
    - <tool_call name="xxx">\n<parameter name="key">value</parameter>\n</tool_call>
    - <invoke name="xxx">\n<parameter name="key">value</parameter>\n</invoke>
    - BEGIN_TOOL_CALL\nname: xxx\nparams: {...}\nEND_TOOL_CALL
    """

    @staticmethod
    def parse(text: str) -> List[Dict[str, Any]]:
        """
        从文本中解析工具调用。

        参数:
            text: 模型输出的完整文本

        返回:
            解析后的工具调用列表,每项包含:
            {
                "name": 工具名,
                "arguments": 参数字典
            }
        """
        import re

        calls = []
        text = text or ""

        # 方式 1: <tool_call name="xxx">...</tool_call>
        pattern1 = r'<tool_call\s+name="([^"]+)"[^>]*>(.*?)</tool_call>'
        for match in re.finditer(pattern1, text, re.DOTALL):
            name = match.group(1)
            body = match.group(2).strip()

            # 解析参数
            args = {}
            param_pattern = r'<parameter\s+name="([^"]+)"[^>]*>(.*?)</parameter>'
            for pm in re.finditer(param_pattern, body, re.DOTALL):
                key = pm.group(1)
                value = pm.group(2).strip()
                # 尝试 JSON 解析
                try:
                    args[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    args[key] = value


            # Fallback: parse body as JSON if no <parameter> tags found
            if not args and body.strip():
                try:
                    parsed = json.loads(body.strip())
                    if isinstance(parsed, dict):
                        args = parsed
                except (json.JSONDecodeError, TypeError):
                    pass
            if args:
                calls.append({"name": name, "arguments": args})

        # 方式 2: <invoke name="xxx">...</invoke>
        pattern2 = r'<invoke\s+name="([^"]+)"[^>]*>(.*?)</invoke>'
        for match in re.finditer(pattern2, text, re.DOTALL):
            name = match.group(1)
            body = match.group(2).strip()

            args = {}
            param_pattern = r'<parameter\s+name="([^"]+)"[^>]*>(.*?)</parameter>'
            for pm in re.finditer(param_pattern, body, re.DOTALL):
                key = pm.group(1)
                value = pm.group(2).strip()
                try:
                    args[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    args[key] = value


            # Fallback: parse body as JSON if no <parameter> tags found
            if not args and body.strip():
                try:
                    parsed = json.loads(body.strip())
                    if isinstance(parsed, dict):
                        args = parsed
                except (json.JSONDecodeError, TypeError):
                    pass
            if args:
                calls.append({"name": name, "arguments": args})

        # 方式 3: BEGIN_TOOL_CALL...END_TOOL_CALL
        pattern3 = r'BEGIN_TOOL_CALL\s*name:\s*(\w+)\s*params:\s*(\{.*?\})\s*END_TOOL_CALL'
        for match in re.finditer(pattern3, text, re.DOTALL):
            name = match.group(1)
            try:
                args = json.loads(match.group(2))
                calls.append({"name": name, "arguments": args})
            except json.JSONDecodeError:
                pass

        # 方式 4: Qwen 原生格式 <tool_call><function=NAME>JSON_BODY</tool_call>
        # Qwen 模型不输出 </function> 闭合标签，直接用 JSON body 作为参数
        pattern4 = r'<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</tool_call>'
        for match in re.finditer(pattern4, text, re.DOTALL):
            name = match.group(1).strip()
            body = match.group(2)
            args = {}
            # 尝试 <parameter=KEY>VALUE</parameter> 子标签格式
            param_pattern4 = r'<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>'
            for pm in re.finditer(param_pattern4, body, re.DOTALL):
                key = pm.group(1).strip()
                value = pm.group(2).strip()
                try:
                    args[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    args[key] = value
            # Fallback: 解析 body 为 JSON（Qwen 常见输出格式）
            if not args and body.strip():
                try:
                    parsed = json.loads(body.strip())
                    if isinstance(parsed, dict):
                        args = parsed
                except (json.JSONDecodeError, TypeError):
                    pass
            if args:
                calls.append({"name": name, "arguments": args})

        return calls


class AgentLoop:
    """
    多轮工具调用循环引擎。

    这是项目的核心 Agent 引擎,参考 hermes-agent 的 HermesAgentLoop 设计。
    支持标准 OpenAI tool calling 和 XML 格式回退解析。

    工作流程:
    1. 接收初始消息列表(system + user)
    2. 循环最多 max_turns 轮:
       a. 调用 LLM 获取响应
       b. 检查是否有 tool_calls
       c. 如果有工具调用:
          - 执行每个工具
          - 将工具结果追加到消息历史
       d. 如果没有工具调用,结束循环
    3. 返回完整的 AgentResult

    使用示例:
        loop = AgentLoop(
            backend_type="vendor",
            vendor_id="deepseek",
            tool_schemas=tool_schemas,
            valid_tool_names=valid_tool_names,
            max_turns=30
        )
        result = await loop.run(messages)
    """

    def __init__(
        self,
        backend_type: str,
        tool_schemas: List[Dict[str, Any]] = None,
        valid_tool_names: Set[str] = None,
        max_turns: int = 30,
        max_retries: int = 3,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        compressor = None,  # 可选的 ContextCompressor 实例(上下文压缩)
        # 后端特定参数
        vendor_id: str = None,
        model: str = None,
        tool_choice: str = "auto",
        plan_mode: bool = False,
        web_search: bool = False,   # 联网搜索开关(控制厂商原生搜索)
        think_output: bool = True,  # 是否输出思考链
        auto_review: bool = False,  # 对话完成后自动审查
        ctx_ext: bool = True,       # 上下文扩展(控制压缩阈值)
        min_prompt: bool = True,    # 最小系统提示词
        **backend_kwargs
    ):
        """
        初始化 Agent Loop。

        参数:
            backend_type: 后端类型 ("vendor", "llama-cpp")
            tool_schemas: OpenAI 格式的工具定义列表
            valid_tool_names: 允许调用的工具名集合
            max_turns: 最大轮次(默认 30)
            max_retries: API 调用最大重试次数(默认 3)
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            compressor: 可选的上下文压缩器实例
            vendor_id: 厂商 ID(backend_type="vendor" 时必填)
            model: 模型名
            tool_choice: 工具选择策略 ("auto"/"required"/"none")
            plan_mode: 开启 Plan 模式时,LLM 产生的工具调用不执行,
                      而是作为计划草稿返回(默认 False)
            **backend_kwargs: 其他后端参数
        """
        self.backend_type = backend_type
        self.tool_schemas = tool_schemas or []
        self.max_retries = max_retries
        # 如果没有提供 valid_tool_names,从 registry 获取
        self.valid_tool_names = valid_tool_names or get_tool_names()
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.plan_mode = plan_mode
        self.web_search = web_search      # 联网搜索开关
        self.think_output = think_output    # 思考链输出
        self.auto_review = auto_review      # 自动审查
        self.ctx_ext = ctx_ext              # 上下文扩展
        self.min_prompt = min_prompt        # 最小提示词

        # 后端参数
        self.vendor_id = vendor_id
        self.model = model
        self.tool_choice = tool_choice
        self.backend_kwargs = backend_kwargs

        # 工具调用解析器
        self.xml_parser = ToolCallParser()

        # 上下文压缩器(可选)
        self.compressor = compressor

        # 任务 ID(用于日志和会话隔离)
        self.task_id = str(uuid.uuid4())[:8]

    async def run(
        self,
        messages: List[Dict[str, Any]],
        on_token: callable = None,
        on_tool_call: callable = None,
        on_tool_result: callable = None,
        on_reasoning: callable = None,
    ) -> AgentResult:
        """
        执行完整的 Agent Loop。

        参数:
            messages: 初始对话消息列表，执行中会被修改
            on_token:      流式 token 回调 on_token(content_str)
            on_tool_call:  工具调用通知 on_tool_call(name, args_json)
            on_tool_result: 工具结果通知 on_tool_result(name, result_json)
            on_reasoning:   推理思考回调 on_reasoning(content_str)  # Qwen think 模式

        返回:
            AgentResult: 包含完整对话历史和元数据
        """
        reasoning_per_turn = []
        tool_errors: List[ToolError] = []
        # ── 重复调用检测 ──
        _consecutive_same_tool = 0
        _last_tool_name = None

        import time
        start_time = time.monotonic()
        logger.info(
            "[%s] >>> AgentLoop 开始 | vendor=%s model=%s backend=%s tools=%d max_turns=%d",
            self.task_id, self.vendor_id, self.model, self.backend_type,
            len(self.valid_tool_names), self.max_turns
        )

        for turn in range(self.max_turns):
            turn_start = time.monotonic()

            # === 上下文压缩检查 ===
            # 每轮开始前检查消息量是否超过阈值,超过则自动压缩
            # ctx_ext 为 False 时跳过压缩(由前端设置面板控制)
            if self.compressor and self.ctx_ext:
                from services.context_compressor import estimate_messages_tokens_rough
                estimated = estimate_messages_tokens_rough(messages)
                if self.compressor.should_compress(prompt_tokens=estimated):
                    logger.info(
                        "[%s] turn %d: 消息量超阈值,压缩上下文...",
                        self.task_id, turn + 1
                    )
                    # 收集记忆上下文(当前任务状态等)
                    memory_context = ""
                    try:
                        from tools.builtin_todo import TodoStore
                        todo_ctx = TodoStore().format_for_injection()
                        if todo_ctx:
                            memory_context = todo_ctx
                    except Exception:
                        pass

                    compress_start = time.monotonic()
                    before_count = len(messages)
                    messages = self.compressor.compress(
                        messages,
                        summary_model=self.model,
                        base_url=self.backend_kwargs.get("base_url", ""),
                        api_key=self.backend_kwargs.get("api_key", ""),
                        memory_context=memory_context,
                    )
                    compress_elapsed = time.monotonic() - compress_start
                    logger.info(
                        "[%s] 压缩完成: %d→%d 条消息, 耗时 %.2fs",
                        self.task_id, before_count, len(messages), compress_elapsed
                    )

            # 构建 API 调用参数
            chat_kwargs = {
                "messages": messages,
                "n": 1,
                "temperature": self.temperature,
            }

            # tools 始终传递(空列表也可以)，否则 Qwen 模板渲染 tool_calls 时 |items 会炸
            chat_kwargs["tools"] = self.tool_schemas if self.tool_schemas else []
            chat_kwargs["tool_choice"] = self.tool_choice

            if self.max_tokens is not None:
                chat_kwargs["max_tokens"] = self.max_tokens

            # 调用后端获取响应(含重试逻辑)
            last_error = None
            for retry_attempt in range(self.max_retries + 1):
                api_start = time.monotonic()
                try:
                    response = await self._call_backend(
                        on_token=on_token,
                        on_reasoning=on_reasoning,
                        **chat_kwargs
                    )

                    # === 更新上下文压缩器 token 用量 ===
                    if self.compressor:
                        usage = response.get("usage", {})
                        if usage:
                            self.compressor.update_from_response(usage)

                    break  # 成功,跳出重试循环
                except Exception as e:
                    api_elapsed = time.monotonic() - api_start
                    last_error = e

                    # 使用 hermes-agent 的错误分类器判断是否可重试
                    classified = classify_api_error(
                        e,
                        provider=self.vendor_id or "llama-cpp",
                        model=self.model or "",
                        approx_tokens=len(json.dumps(messages, ensure_ascii=False)) // 4,
                    )

                    # 不可重试或已达最大重试次数 -> 返回错误
                    if not classified.retryable or retry_attempt >= self.max_retries:
                        logger.error(
                            "API 调用失败 (turn %d, retry %d/%d, %.1fs, reason=%s): %s",
                            turn + 1, retry_attempt, self.max_retries,
                            api_elapsed, classified.reason.value, e
                        )
                        total_s = time.monotonic() - start_time; logger.info("[%s] END | total=%.1fs", self.task_id, total_s)
                        return AgentResult(
                            messages=messages,
                            turns_used=turn + 1,
                            finished_naturally=False,
                            reasoning_per_turn=reasoning_per_turn,
                            tool_errors=tool_errors,
                        )

                    # 计算抖动退避延迟
                    delay = jittered_backoff(retry_attempt + 1)
                    logger.warning(
                        "API 调用失败 (turn %d, retry %d/%d, %.1fs, reason=%s): "
                        "%.100s. %.0fs 后重试...",
                        turn + 1, retry_attempt + 1, self.max_retries,
                        api_elapsed, classified.reason.value,
                        classified.message, delay
                    )
                    await asyncio.sleep(delay)

            api_elapsed = time.monotonic() - api_start

            # 空响应检查
            if not response or not response.get("choices"):
                logger.warning("空响应 (turn %d)", turn + 1)
                total_s = time.monotonic() - start_time; logger.info("[%s] END | total=%.1fs", self.task_id, total_s)
                return AgentResult(
                    messages=messages,
                    turns_used=turn + 1,
                    finished_naturally=False,
                    reasoning_per_turn=reasoning_per_turn,
                    tool_errors=tool_errors,
                )

            # 提取助手消息
            assistant_msg = response["choices"][0].get("message", {})
            content = assistant_msg.get("content", "")
            tool_calls = assistant_msg.get("tool_calls", [])

            # 提取推理内容
            reasoning = assistant_msg.get("reasoning_content") or assistant_msg.get("reasoning")
            reasoning_per_turn.append(reasoning)

            # ===== 工具调用处理 =====
            if tool_calls:
                logger.info("[%s] turn %d: %d tool_calls", self.task_id, turn + 1, len(tool_calls))

                # 添加助手消息到历史(arguments 从 JSON string 转 dict,
                # Qwen 模板 |items 只接受 mapping。使用 deepcopy 避免浅拷贝
                # 导致 function 子对象共享引用的问题。
                import copy, sys
                safe_tool_calls = []
                for tc in tool_calls or []:
                    tc_copy = copy.deepcopy(tc)
                    func = tc_copy.get("function", {})
                    args = func.get("arguments")
                    if isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            func["arguments"] = {}
                    # 断言：arguments 必须是 dict，否则 Qwen 模板 |items 会炸
                    assert isinstance(func.get("arguments"), dict), \
                        f"tool_call arguments 类型错误: {type(func.get('arguments'))}, value={func.get('arguments')}"
                    safe_tool_calls.append(tc_copy)
                msg_dict = {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": safe_tool_calls,
                }
                if reasoning:
                    msg_dict["reasoning_content"] = reasoning
                messages.append(msg_dict)

                # ── Plan 模式:收集工具调用为计划草稿,不执行 ──
                if self.plan_mode:
                    plan_items = []
                    for tc in tool_calls:
                        tn = tc.get("function", {}).get("name", "")
                        ta_raw = tc.get("function", {}).get("arguments", "{}")
                        try:
                            ta = json.loads(ta_raw) if isinstance(ta_raw, str) else ta_raw
                        except json.JSONDecodeError:
                            ta = {"_raw": ta_raw}
                        plan_items.append({"tool": tn, "arguments": ta})
                    logger.info(
                        "[%s] Plan mode: %d items, awaiting approval",
                        self.task_id, len(plan_items),
                    )
                    total_s = time.monotonic() - start_time; logger.info("[%s] END | total=%.1fs", self.task_id, total_s)
                    return AgentResult(
                        messages=messages,
                        turns_used=turn + 1,
                        finished_naturally=False,
                        reasoning_per_turn=reasoning_per_turn,
                        tool_errors=tool_errors,
                        plan={
                            "items": plan_items,
                            "assistant_message": content,
                            "turn": turn + 1,
                        },
                        plan_approved=False,
                    )

                # 执行每个工具调用
                # ── 并发工具执行 (优化: 多工具并行) ──
                self._execute_tools_with_cache(
                    tool_calls, messages, tool_errors, turn,
                    on_tool_call, on_tool_result
                )

                turn_elapsed = time.monotonic() - turn_start
                logger.info(
                    "[%s] turn %d 完成: API=%.1fs, %d 工具, 总耗时=%.1fs",
                    self.task_id, turn + 1, api_elapsed, len(tool_calls), turn_elapsed
                )

                # ── 重复调用检测：同一工具连续调用 >=3 次 → 注入警告 ──
                this_turn_tools = [tc.get("function", {}).get("name", "") for tc in tool_calls]
                if len(this_turn_tools) == 1 and this_turn_tools[0] == _last_tool_name:
                    _consecutive_same_tool += 1
                else:
                    _consecutive_same_tool = 1
                _last_tool_name = this_turn_tools[0] if len(this_turn_tools) == 1 else None

                if _consecutive_same_tool >= 3:
                    logger.warning(
                        "[%s] 工具 '%s' 连续调用 %d 次，注入停止警告",
                        self.task_id, _last_tool_name, _consecutive_same_tool
                    )
                    warning_msg = {"role": "system", "content": f"⚠️ 你已经连续 {_consecutive_same_tool} 轮调用 '{_last_tool_name}' 工具。\n如果已经获取了足够的信息，请停止调用工具，直接用现有信息回答用户。\n如果没有找到相关信息，也请如实告诉用户，不要反复搜索。"}
                    messages.append(warning_msg)
                    _consecutive_same_tool = 0  # 重置，只警告一次

            # ===== 无工具调用:结束 =====
            else:
                # 将助手消息加入历史
                msg_dict = {
                    "role": "assistant",
                    "content": content or "",
                }
                if reasoning:
                    msg_dict["reasoning_content"] = reasoning
                messages.append(msg_dict)

                turn_elapsed = time.monotonic() - turn_start
                logger.info(
                    "[%s] turn %d: API=%.1fs, 无工具调用(自然结束), 总耗时=%.1fs",
                    self.task_id, turn + 1, api_elapsed, turn_elapsed
                )

                total_s = time.monotonic() - start_time; logger.info("[%s] END | total=%.1fs", self.task_id, total_s)
                return AgentResult(
                    messages=messages,
                    turns_used=turn + 1,
                    finished_naturally=True,
                    reasoning_per_turn=reasoning_per_turn,
                    tool_errors=tool_errors,
                )

        # 达到最大轮次
        logger.info("Agent 达到最大轮次 %d", self.max_turns)
        total_s = time.monotonic() - start_time; logger.info("[%s] END | total=%.1fs", self.task_id, total_s)
        return AgentResult(
            messages=messages,
            turns_used=self.max_turns,
            finished_naturally=False,
            reasoning_per_turn=reasoning_per_turn,
            tool_errors=tool_errors,
        )

    async def _call_backend(self, on_token=None, on_reasoning=None, **kwargs):
        """
        调用后端获取 LLM 响应。

        根据 backend_type 路由到对应的后端:
        - "vendor": 调用 vendors.py 的 chat_stream
        - "llama-cpp": 调用 gpu.py 的 infer

        参数:
            on_token: 可选的流式 token 回调 on_token(content_str)
            on_reasoning: 可选的思考内容回调 on_reasoning(content_str)  # Qwen think 模式

        返回:
            标准化的响应字典,包含:
            {
                "choices": [{
                    "message": {
                        "content": "...",
                        "tool_calls": [...],
                        "reasoning_content": "..."
                    }
                }]
            }
        """
        # 收集流式输出
        call_start = time.monotonic()
        chunks = []
        first_token = None  # TTFT 追踪

        if self.backend_type == "vendor":
            # 厂商 API(OpenAI 兼容格式)
            logger.info("[%s]   API 调用开始 (vendor=%s model=%s)", self.task_id, self.vendor_id, self.model)
            # 调试:打印传递的 tools 数量
            _tools = kwargs.get("tools")
            if _tools:
                _tool_names = [t.get("function", {}).get("name", "?") for t in _tools]
                logger.info("[%s]   tools 参数: %d 个 (%s)", self.task_id, len(_tools), ", ".join(sorted(_tool_names)[:10]) + ("..." if len(_tool_names) > 10 else ""))
            else:
                logger.info("[%s]   tools 参数: None/空", self.task_id)
            # ── 修复: tool_calls arguments 必须是 JSON string ──
            # OpenAI API 要求 function.arguments 为 string，但内部处理用 dict。
            # 调用 API 前将 dict 转回 string。
            import copy as _copy
            _msgs = _copy.deepcopy(kwargs.get("messages", []))
            for _m in _msgs:
                if _m.get("role") == "assistant" and _m.get("tool_calls"):
                    for _tc in _m["tool_calls"]:
                        _fn = _tc.get("function", {})
                        _arg = _fn.get("arguments")
                        if isinstance(_arg, dict):
                            _fn["arguments"] = json.dumps(_arg, ensure_ascii=False)
            # ── 调用 API ──
            stream = vendors.chat_stream(

                vendor_id=self.vendor_id,
                model=self.model,
                messages=_msgs,
                tools=kwargs.get("tools"),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens"),
                api_key=self.backend_kwargs.get("api_key"),
                base_url=self.backend_kwargs.get("base_url"),
                web_search=self.web_search,  # 联网搜索开关
            )
            for chunk in stream:
                if first_token is None:
                    first_token = time.monotonic()
                chunks.append(chunk)
                if on_token and isinstance(chunk, dict) and chunk.get("content"):
                    on_token(chunk["content"])
                if on_reasoning and isinstance(chunk, dict) and chunk.get("reasoning_content"):
                    on_reasoning(chunk["reasoning_content"])

            # TTFT 日志
            if first_token is not None:
                ttft = first_token - call_start
                logger.info("[%s]   API TTFT=%.2fs, chunks=%d", self.task_id, ttft, len(chunks))
            else:
                logger.info("[%s]   API 无内容返回, chunks=%d", self.task_id, len(chunks))

        elif self.backend_type == "llama-cpp":
            # llama-cpp (GPU 模式) - 必须传完整 messages(含 tool_calls 和 tool 结果),
            # 不能只传 system + prompt(否则工具历史全丢,第二轮推理空输出)
            messages = kwargs.get("messages", [])

            # 【关键】前端 SSE done 事件通过 JSON 序列化 messages,
            # tool_calls[].function.arguments 从 dict 变回 string。
            # 必须在传给 Qwen 模板前全量转换，否则 |items 报 TypeError
            import copy
            messages = copy.deepcopy(messages)
            for msg in messages:
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    for tc in msg.get("tool_calls") or []:
                        func = tc.get("function", {})
                        args = func.get("arguments")
                        if isinstance(args, str):
                            try:
                                func["arguments"] = json.loads(args)
                            except (json.JSONDecodeError, TypeError):
                                func["arguments"] = {}

            def sync_infer():
                return gpu.infer(
                    messages=messages,
                    tools=kwargs.get("tools"),  # 必须显式传,否则模型收不到工具定义
                    stream=True,
                    **self.backend_kwargs
                )

            loop = asyncio.get_event_loop()
            gen = await loop.run_in_executor(None, sync_infer)
            for chunk in gen:
                if first_token is None:
                    first_token = time.monotonic()
                if isinstance(chunk, dict):
                    # gpu.infer() 流式 yield dict: {"content": ..., "reasoning_content": ..., "tool_calls": ...}
                    # tool_calls 仅 Gemma4 等原生 tool calling 模型提供
                    chunks.append(chunk)
                    if on_token and chunk.get("content"):
                        on_token(chunk["content"])
                    if on_reasoning and chunk.get("reasoning_content"):
                        on_reasoning(chunk["reasoning_content"])
                else:
                    # 向后兼容:纯字符串 chunk
                    chunks.append({"content": chunk})
                    if on_token:
                        on_token(chunk)

            # TTFT 日志 (llama-cpp)
            if first_token is not None:
                ttft = first_token - call_start
                logger.info("[%s]   llama-cpp TTFT=%.2fs, chunks=%d", self.task_id, ttft, len(chunks))

        else:
            raise ValueError(f"未知后端类型: {self.backend_type}")

        # 组装响应
        return self._assemble_response(chunks)

    def _assemble_response(self, chunks: List[Dict]) -> Dict:
        """
        将流式 chunks 组装成标准化响应。

        处理三种格式:
        1. _openai_stream 顶层键:{"content": "...", "tool_calls": [...]}
        2. 旧格式嵌套:{"message": {"content": ..., "tool_calls": ...}}
        3. 纯文本格式(llama-cpp):使用 XML 解析器提取工具调用
        """
        # 合并 content
        full_content = ""
        reasoning = None  # DeepSeek/Zhipu thinking 模式的推理内容
        tool_calls = None
        usage = None  # token 用量(用于上下文压缩跟踪)

        for chunk in chunks:
            if "error" in chunk:
                raise Exception(chunk["error"])

            # _openai_stream 顶层 format(优先)
            if "content" in chunk:
                full_content += chunk["content"]
            if "reasoning_content" in chunk:
                reasoning = chunk["reasoning_content"]  # thinking 模式推理内容
            if "tool_calls" in chunk:
                tool_calls = chunk["tool_calls"]
            if "usage" in chunk:
                usage = chunk["usage"]  # 最后一个 chunk 通常携带 usage

            # 旧格式兼容(嵌套在 message 下)
            if "message" in chunk:
                msg = chunk["message"]
                if "content" in msg and msg["content"]:
                    full_content += msg["content"]
                if "tool_calls" in msg and msg["tool_calls"]:
                    tool_calls = msg["tool_calls"]

        # === 通用回退:使用 tool_call_parsers 模块解析 ===
        # 当模型不支持原生 function calling 时,从原始文本解析工具调用。
        # 支持 hermes / llama / qwen / deepseek_v3 / glm45 / glm47 / mistral 格式
        # 注意:按优先级排序,hermes/llama 最常用,glm 容易误匹配应最后尝试
        if not tool_calls:
            try:
                from services.tool_call_parsers import get_parser
                # 优先级顺序:先精确匹配,再宽松匹配
                priority_order = ["hermes", "llama3_json", "llama4_json", "qwen", "deepseek_v3", "mistral", "glm45", "glm47"]
                for parser_name in priority_order:
                    parser = get_parser(parser_name)
                    parsed_text, parsed_calls = parser.parse(full_content)
                    if parsed_calls:
                        tool_calls = []
                        for tc_data in parsed_calls:
                            tool_calls.append({
                                "id": tc_data.id,
                                "type": tc_data.type,
                                "function": {
                                    "name": tc_data.function.name,
                                    "arguments": tc_data.function.arguments,
                                }
                            })
                        full_content = parsed_text or ""
                        logger.info(
                            "tool_call_parsers: 使用 '%s' 解析器从文本提取到 %d 个工具调用",
                            parser_name, len(tool_calls)
                        )
                        break
            except ImportError:
                pass

        # llama-cpp: 旧版 XML 解析器(兼容保留)
        if self.backend_type == "llama-cpp" and not tool_calls:
            logger.debug("[_assemble] llama-cpp path, full_content len=%d, preview: %s", len(full_content), full_content[:200] if full_content else 'EMPTY')
            parsed_calls = self.xml_parser.parse(full_content)
            logger.debug("[_assemble] xml_parser.parse returned %d calls", len(parsed_calls or []))
            if parsed_calls:
                tool_calls = []
                for pc in parsed_calls:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": pc["name"],
                            "arguments": json.dumps(pc["arguments"])
                        }
                    })
                # 如果解析出工具调用,从 content 中移除 XML 部分
                if tool_calls:
                    import re
                    full_content = re.sub(
                        r'<tool_call[^>]*>.*?</tool_call>',
                        '',
                        full_content,
                        flags=re.DOTALL
                    ).strip()

        # ── 回退 J: reasoning_content 中也搜索 tool_call ──
        # DeepSeek 等模型在 thinking 模式中，tool_call 可能只出现在 reasoning_content 而非 content 中
        if not tool_calls and reasoning:
            import re
            # 复用 tool_call_parsers 对 reasoning 文本做解析
            try:
                from services.tool_call_parsers import get_parser
                priority_order = ["hermes", "llama3_json", "llama4_json", "qwen", "deepseek_v3", "mistral"]
                for parser_name in priority_order:
                    parser = get_parser(parser_name)
                    parsed_text, parsed_calls = parser.parse(reasoning)
                    if parsed_calls:
                        tool_calls = []
                        for tc_data in parsed_calls:
                            tool_calls.append({
                                "id": tc_data.id,
                                "type": tc_data.type,
                                "function": {
                                    "name": tc_data.function.name,
                                    "arguments": tc_data.function.arguments,
                                }
                            })
                        logger.info(
                            "[%s] 从 reasoning_content 用 '%s' 提取到 %d 个工具调用",
                            self.task_id, parser_name, len(tool_calls)
                        )
                        break
            except ImportError:
                pass

        # ── 回退 K: JSON {"tool_call": {...}} 格式提取（DeepSeek thinking 非标准格式）──
        # DeepSeek 在 thinking 中会输出 {"tool_call": {"name": "...", "arguments": {...}}}
        if not tool_calls:
            import re
            texts_to_check = [full_content] if full_content else []
            if reasoning:
                texts_to_check.append(reasoning)
            for check_text in texts_to_check:
                if not check_text:
                    continue
                tc_pattern = re.compile(
                    r'\{\s*"tool_call"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}\s*\}',
                    re.DOTALL
                )
                for match in tc_pattern.finditer(check_text):
                    tc_name = match.group(1)
                    tc_args_str = match.group(2)
                    try:
                        tc_args = json.loads(tc_args_str)
                    except (json.JSONDecodeError, TypeError):
                        tc_args = {}
                    tool_calls = tool_calls or []
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": tc_name,
                            "arguments": json.dumps(tc_args, ensure_ascii=False),
                        }
                    })
                    logger.info(
                        "[%s] 从文本中用 JSON 格式提取到 tool_call '%s'",
                        self.task_id, tc_name
                    )
                if tool_calls:
                    break

        resp = {
            "choices": [{
                "message": {
                    "content": full_content,
                    "tool_calls": tool_calls,
                    "reasoning_content": reasoning,  # DeepSeek/Zhipu thinking 模式
                }
            }]
        }
        if usage:
            resp["usage"] = usage
        return resp

    def _execute_tools_with_cache(
        self, tool_calls: list, messages: list, tool_errors: list,
        turn: int, on_tool_call=None, on_tool_result=None
    ):
        """
        并发执行多个工具调用,含 lookaside 缓存

        流程:
          1. 解析参数,验证工具名
          2. 对可缓存工具检查 LRU 缓存
          3. 剩余工具通过 ThreadPoolExecutor 并发执行
          4. 收集结果,追加到 messages
        """
        from agent.tool_executor import execute_tools_concurrent
        from services.cache_manager import get_cache
        import asyncio

        cache = get_cache()
        validated_calls = []  # [(tc, tool_name, args, tc_id), ...]

        for tc in tool_calls:
            tool_name = tc.get("function", {}).get("name", "")
            tool_args_raw = tc.get("function", {}).get("arguments", "{}")
            tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")

            # 验证工具名
            if tool_name not in self.valid_tool_names:
                err = json.dumps({
                    "error": f"未知工具 '{tool_name}'。可用工具: {sorted(self.valid_tool_names)}"
                })
                tool_errors.append(ToolError(
                    turn=turn + 1, tool_name=tool_name,
                    arguments=tool_args_raw[:200],
                    error=f"未知工具 '{tool_name}'",
                    tool_result=err[:500],
                ))
                messages.append({
                    "role": "tool", "tool_call_id": tc_id, "content": err
                })
                logger.warning("模型调用了未知工具 '%s'", tool_name)
                continue

            # 解析参数
            try:
                if isinstance(tool_args_raw, str):
                    args = json.loads(tool_args_raw)
                else:
                    args = tool_args_raw
            except json.JSONDecodeError as e:
                err = json.dumps({
                    "error": f"工具参数 JSON 解析失败: {e}。请使用有效 JSON。"
                })
                tool_errors.append(ToolError(
                    turn=turn + 1, tool_name=tool_name,
                    arguments=tool_args_raw[:200],
                    error=f"JSON 解析失败: {e}",
                    tool_result=err[:500],
                ))
                messages.append({
                    "role": "tool", "tool_call_id": tc_id, "content": err
                })
                logger.warning("工具 '%s' 参数解析失败: %s", tool_name, tool_args_raw[:200])
                continue

            validated_calls.append((tc, tool_name, args, tc_id))

        # 分离缓存命中和需要执行的调用
        cached_results = {}  # tc_index → result
        execute_calls = []   # tc objects to execute

        for i, (tc, tool_name, args, tc_id) in enumerate(validated_calls):
            cached = cache.get_tool_result(tool_name, args)
            if cached is not None:
                cached_results[i] = cached
                logger.debug("[%s]   🔄 缓存命中: %s", self.task_id, tool_name)
            else:
                # 为并发执行准备 args(已经是 dict)
                tc["function"]["arguments"] = args
                # 注入父代理引用
                if tool_name in ("delegate_task", "skill_evolve"):
                    tc["function"]["arguments"] = {**args, "parent_agent": self}
                execute_calls.append(tc)

        # 并发执行需要调用的工具
        execute_results = {}
        if execute_calls:
            n_tools = len(execute_calls)
            if n_tools > 1:
                logger.info("[%s] ⚡ 并发执行 %d 个工具", self.task_id, n_tools)

            # 通知前端
            if on_tool_call:
                for tc in execute_calls:
                    tn = tc.get("function", {}).get("name", "")
                    ta = tc.get("function", {}).get("arguments", {})
                    on_tool_call(tn, json.dumps(ta))

            tool_start = time.monotonic()
            results = execute_tools_concurrent(execute_calls)
            tool_elapsed = time.monotonic() - tool_start

            for tc, result in results:
                key = id(tc)
                execute_results[key] = result
                tn = tc.get("function", {}).get("name", "")
                ta = tc.get("function", {}).get("arguments", {})

                # 缓存可缓存的工具结果
                cache.set_tool_result(tn, ta, result)

                # 错误跟踪
                if '"error"' in result or '"error":' in result:
                    err_msg = result[:500]
                    tool_errors.append(ToolError(
                        turn=turn + 1, tool_name=tn,
                        arguments=json.dumps(ta)[:200],
                        error="execution error",
                        tool_result=err_msg,
                    ))

            if n_tools > 1:
                logger.info(
                    "[%s]   ✅ %d 工具执行完成 (%.2fs)",
                    self.task_id, n_tools, tool_elapsed
                )
            else:
                tn = execute_calls[0].get("function", {}).get("name", "")
                logger.info(
                    "[%s]   工具 %s 执行完成 (%.2fs)",
                    self.task_id, tn, tool_elapsed
                )

        # 按原始顺序追加结果到 messages
        for i, (tc, tool_name, args, tc_id) in enumerate(validated_calls):
            result = None
            if i in cached_results:
                result = cached_results[i]
            else:
                result = execute_results.get(id(tc), json.dumps({
                    "error": f"工具 '{tool_name}' 未执行"
                }))

            # 通知前端
            if on_tool_result:
                on_tool_result(tool_name, result)

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result or "",
            })

    async def _execute_tool(self, tool_name: str, args: Dict) -> str:
        """
        执行工具调用。

        委托给工具注册表的 execute 方法,
        Registry.execute() 内部自动处理同步/异步 handler。

        参数:
            tool_name: 工具名
            args: 参数字典

        返回:
            工具执行结果的 JSON 字符串
        """
        from tools.registry import get_registry

        try:
            # 为 delegate_task / skill_evolve 注入父代理引用
            if tool_name in ("delegate_task", "skill_evolve"):
                args = {**args, "parent_agent": self}
            result = await get_registry().execute(tool_name, args)
            return result
        except ValueError as e:
            return json.dumps({"error": str(e)})
        except RuntimeError as e:
            return json.dumps({"error": str(e)})


