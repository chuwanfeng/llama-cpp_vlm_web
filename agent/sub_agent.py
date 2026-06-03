"""异步子代理执行器（SubAgent）。

移植自 hermes-agent/run_agent.py 的 delegate_task 模式，
适配 llama-cpp_vlm_web 的 Web 异步架构。

核心设计（hermes-agent 兼容）：
  - 每个子代理独立的 IterationBudget（迭代预算）
  - 工具子集（只暴露父代理委托的工具集）
  - 异步并发执行（asyncio.create_task，替代 hermes-agent 的 ThreadPoolExecutor）
  - 结果自动收集并序列化返回给父代理

使用方式：
    # 由 tools/delegate_tool.py 的 delegate_task() 调用
    sub = SubAgent(
        goal="分析代码质量",
        context="...",
        toolsets=["code_review"],
        parent_agent=agent_loop,
    )
    result = await sub.run()

与 hermes-agent 的差异：
  - hermes-agent: ThreadPoolExecutor + run_agent.py AIAgent 递归
  - 本实现: asyncio.create_task + AgentLoop 复用
  - 去除了 CLI 依赖（display、TUI、print_fn 等）
  - 去除了 gateway/messaging 依赖
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from utils import get_logger

logger = get_logger("agent.sub_agent")


# ─── 数据结构 ──────────────────────────────────────────────────


@dataclass
class SubAgentResult:
    """子代理执行结果。

    属性:
        success: 是否成功完成
        goal: 任务目标
        summary: 执行摘要
        output: 子代理的最终输出文本
        turns_used: 使用的迭代轮次
        tool_calls_made: 发出的工具调用列表
        error: 失败时的错误信息
        duration_ms: 执行耗时（毫秒）
    """
    success: bool = False
    goal: str = ""
    summary: str = ""
    output: str = ""
    turns_used: int = 0
    tool_calls_made: List[str] = field(default_factory=list)
    error: str = ""
    duration_ms: float = 0.0


class IterationBudget:
    """线程安全的迭代预算计数器。

    移植自 hermes-agent/run_agent.py 的 IterationBudget 类。

    每个子代理有独立的预算，默认上限由 delegation.max_iterations 控制。
    父代理通过此预算限制子代理的工具调用轮数。
    """

    def __init__(self, max_iterations: int = 50):
        """
        参数:
            max_iterations: 最大迭代轮数（默认 50，hermes-agent 默认值）
        """
        self._max = max_iterations
        self._used = 0

    def consume(self) -> bool:
        """消耗一次迭代预算。返回 True 表示还有预算。"""
        if self._used >= self._max:
            return False
        self._used += 1
        return True

    @property
    def remaining(self) -> int:
        """剩余预算轮数。"""
        return max(0, self._max - self._used)

    @property
    def used(self) -> int:
        """已消耗预算轮数。"""
        return self._used

    @property
    def exhausted(self) -> bool:
        """预算是否已耗尽。"""
        return self._used >= self._max

    def reset(self) -> None:
        """重置预算计数器。"""
        self._used = 0

    def __repr__(self) -> str:
        return f"IterationBudget({self._used}/{self._max})"


# ─── SubAgent ───────────────────────────────────────────────────


class SubAgent:
    """异步子代理执行器。

    由 tools/delegate_tool.py 的 delegate_task() 创建和运行。
    每个子代理获得：
      - 独立的 IterationBudget
      - 工具子集（只包含委托的工具集）
      - 异步执行环境
      - 结果自动收集

    属性:
        goal: 任务目标（由 LLM 指定的自然语言描述）
        context: 上下文信息（父代理提供的背景）
        toolsets: 工具集名称列表（如 ["code_review", "web_search"]）
        max_iterations: 最大迭代轮数
        budget: 迭代预算计数器
        task_id: 唯一任务 ID
        result: 执行结果
    """

    def __init__(
        self,
        goal: str,
        context: str = "",
        toolsets: List[str] = None,
        max_iterations: int = 50,
        parent_agent=None,
    ):
        """
        参数:
            goal: 任务目标
            context: 上下文信息
            toolsets: 工具集名称列表
            max_iterations: 最大迭代轮数
            parent_agent: 父代理（AgentLoop 实例，用于获取后端配置等）
        """
        self.goal = goal
        self.context = context
        self.toolsets = toolsets or []
        self.max_iterations = max_iterations
        self.parent = parent_agent

        self.budget = IterationBudget(max_iterations)
        self.task_id = str(uuid.uuid4())[:12]
        self.result = SubAgentResult(goal=goal)

        # 执行的工具调用列表（用于审计/调试）
        self._tool_calls_log: List[Dict[str, Any]] = []

    async def run(self) -> SubAgentResult:
        """执行子代理任务。

        创建一个简化的 AgentLoop 实例，运行子任务，
        收集结果并返回 SubAgentResult。

        返回:
            SubAgentResult: 执行结果
        """
        start_time = time.monotonic()

        try:
            # 构建子代理的系统提示
            system_prompt = self._build_system_prompt()

            # 构建消息列表
            user_content = "## 任务目标\n" + self.goal
            if self.context:
                user_content += "\n\n## 上下文\n" + self.context
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            # 获取工具定义（仅委托的工具集）
            tool_schemas = self._get_tool_schemas()
            valid_tool_names = self._get_valid_tool_names()

            if not tool_schemas:
                self.result.success = False
                self.result.error = f"No tools available for toolsets: {self.toolsets}"
                self.result.duration_ms = (time.monotonic() - start_time) * 1000
                return self.result

            # 创建子代理的 AgentLoop
            from agent.loop import AgentLoop

            # 子代理的 max_turns 由迭代预算控制
            sub_loop = AgentLoop(
                backend_type=self._get_backend_type(),
                tool_schemas=tool_schemas,
                valid_tool_names=valid_tool_names,
                max_turns=self.max_iterations,
                max_retries=2,
                temperature=self._get_temperature(),
                vendor_id=self._get_vendor_id(),
                model=self._get_model(),
                api_key=self._get_api_key(),
                base_url=self._get_base_url(),
            )

            # 执行循环
            from agent.loop import AgentResult
            agent_result: AgentResult = await sub_loop.run(
                messages=messages,
                on_token=None,  # 子代理不需要流式回调
                on_tool_call=self._on_tool_call,
                on_tool_result=None,
            )

            # 提取最终输出
            final_output = ""
            for msg in reversed(agent_result.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    final_output = msg["content"]
                    break

            # 构建结果
            self.result.success = True
            self.result.output = final_output
            self.result.turns_used = agent_result.turns_used
            self.result.tool_calls_made = [
                tc.get("name", "unknown") for tc in self._tool_calls_log
            ]
            self.result.summary = self._generate_summary(agent_result)

        except Exception as e:
            logger.exception("SubAgent %s failed: %s", self.task_id, e)
            self.result.success = False
            self.result.error = f"{type(e).__name__}: {e}"

        finally:
            self.result.duration_ms = (time.monotonic() - start_time) * 1000

        return self.result

    def _build_system_prompt(self) -> str:
        """构建子代理的系统提示词。

        让子代理知道自己是子任务执行者，需要专注于指定的目标。
        """
        tool_list = ", ".join(self.toolsets) if self.toolsets else "通用工具"
        lines = [
            "你是一个子任务执行代理（Sub-Agent），负责完成指定的子任务。",
            "可用工具集: " + tool_list,
            "最大迭代轮数: " + str(self.max_iterations),
            "",
            "规则:",
            "1. 专注于给定的任务目标，不要偏离",
            "2. 完成后直接给出结论，不要等待进一步指令",
            "3. 如果多次尝试后无法完成，报告失败原因",
            "4. 你的输出将作为父代理的工具调用结果返回",
        ]
        return "\n".join(lines)

    def _get_tool_schemas(self) -> List[Dict[str, Any]]:
        """获取子代理可用的工具 schema。

        如果指定了 toolsets，只返回这些工具集的工具。
        如果指定的是单个工具名（非 toolset），则只返回该工具。
        否则使用父代理的全部工具。
        """
        from tools.registry import get_registry, get_tool
        registry = get_registry()

        if not self.toolsets:
            return registry.get_schemas()

        schemas = []
        for ts in self.toolsets:
            # 先尝试作为 toolset 名查找
            tools = registry.get_tools_by_toolset(ts)
            if tools:
                for entry in tools:
                    if entry.is_available():
                        schemas.append(entry.to_openai_schema())
            else:
                # 不是 toolset → 尝试作为单个工具名
                entry = get_tool(ts)
                if entry and entry.is_available():
                    schemas.append(entry.to_openai_schema())
        return schemas

    def _get_valid_tool_names(self) -> Set[str]:
        """获取有效的工具名称集合。

        支持 toolset 名和单个工具名混合。
        """
        from tools.registry import get_registry, get_tool
        registry = get_registry()

        if not self.toolsets:
            return set(registry.get_all_tool_names())

        names = set()
        for ts in self.toolsets:
            toolset_names = registry.get_tool_names_for_toolset(ts)
            if toolset_names:
                names.update(toolset_names)
            else:
                # 不是 toolset → 检查是否为有效工具名
                entry = get_tool(ts)
                if entry and entry.is_available():
                    names.add(ts)
        return names

    def _get_backend_type(self) -> str:
        """从父代理获取后端类型。"""
        if self.parent and hasattr(self.parent, "backend_type"):
            return self.parent.backend_type
        return "vendor"

    def _get_vendor_id(self) -> Optional[str]:
        """从父代理获取厂商 ID。"""
        if self.parent and hasattr(self.parent, "vendor_id"):
            return self.parent.vendor_id
        return None

    def _get_model(self) -> Optional[str]:
        """从父代理获取模型名。"""
        if self.parent and hasattr(self.parent, "model"):
            return self.parent.model
        return None

    def _get_temperature(self) -> float:
        """从父代理获取温度参数。"""
        if self.parent and hasattr(self.parent, "temperature"):
            return self.parent.temperature
        return 1.0

    def _get_api_key(self) -> str:
        """从父代理获取 API Key，fallback 到 config。"""
        if self.parent and hasattr(self.parent, "backend_kwargs"):
            key = self.parent.backend_kwargs.get("api_key", "")
            if key:
                return key
        # fallback 1: config
        try:
            from config import config
            key = config.get_backend_config().get("api_key", "")
            if key:
                return key
        except Exception:
            pass
        # fallback 2: 环境变量
        import os
        return (os.environ.get("DEEPSEEK_API_KEY", "")
                or os.environ.get("OPENAI_API_KEY", ""))

    def _get_base_url(self) -> Optional[str]:
        """从父代理获取 API Base URL。"""
        if self.parent and hasattr(self.parent, "backend_kwargs"):
            return self.parent.backend_kwargs.get("base_url")
        return None

    def _on_tool_call(self, tool_name: str, args: Dict[str, Any]):
        """工具调用回调（记录日志）。"""
        self._tool_calls_log.append({"name": tool_name, "args": args})
        logger.debug("SubAgent %s tool call: %s(%s)", self.task_id, tool_name, _truncate_args(args))

    def _generate_summary(self, agent_result) -> str:
        """生成子代理执行摘要。"""
        parts = []
        if self.result.tool_calls_made:
            parts.append(f"工具调用: {', '.join(self.result.tool_calls_made)}")
        parts.append(f"耗时 {agent_result.turns_used} 轮")
        if agent_result.finished_naturally:
            parts.append("自然完成")
        else:
            parts.append("达到最大轮数")
        return " | ".join(parts)


# ─── 并发子代理协调器 ──────────────────────────────────────────


class SubAgentCoordinator:
    """协调多个子代理的并发执行。

    使用 asyncio.gather() 或 asyncio.create_task() 实现并发。

    使用示例:
        coordinator = SubAgentCoordinator(max_concurrent=5)
        tasks = [
            {"goal": "分析 performance.py", "toolsets": ["code_review"]},
            {"goal": "分析 security.py", "toolsets": ["code_review"]},
        ]
        results = await coordinator.run_all(tasks)
    """

    def __init__(self, max_concurrent: int = 5, parent_agent=None):
        """
        参数:
            max_concurrent: 最大并发子代理数
            parent_agent: 父代理实例
        """
        self.max_concurrent = max_concurrent
        self.parent = parent_agent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def run_all(self, tasks: List[Dict[str, Any]]) -> List[SubAgentResult]:
        """并发执行多个子任务。

        参数:
            tasks: 任务列表，每项包含 goal, context, toolsets 等字段

        返回:
            按原始顺序排列的结果列表
        """

        async def _run_one(task: Dict[str, Any]) -> SubAgentResult:
            async with self._semaphore:
                sub = SubAgent(
                    goal=task.get("goal", ""),
                    context=task.get("context", ""),
                    toolsets=task.get("toolsets", []),
                    max_iterations=task.get("max_iterations", 50),
                    parent_agent=self.parent,
                )
                return await sub.run()

        return list(await asyncio.gather(*[_run_one(t) for t in tasks]))

    async def run_single(self, task: Dict[str, Any]) -> SubAgentResult:
        """执行单个子任务（带并发控制）。"""
        async with self._semaphore:
            sub = SubAgent(
                goal=task.get("goal", ""),
                context=task.get("context", ""),
                toolsets=task.get("toolsets", []),
                max_iterations=task.get("max_iterations", 50),
                parent_agent=self.parent,
            )
            return await sub.run()


# ─── 辅助函数 ──────────────────────────────────────────────────


def _truncate_args(args: Dict[str, Any], max_len: int = 100) -> str:
    """截断参数字符串用于日志。"""
    s = json.dumps(args, ensure_ascii=False)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def sub_agent_result_to_json(result: SubAgentResult) -> str:
    """将 SubAgentResult 转换为 JSON 字符串（供工具返回）。"""
    return json.dumps({
        "success": result.success,
        "goal": result.goal,
        "summary": result.summary,
        "output": result.output,
        "turns_used": result.turns_used,
        "tool_calls_made": result.tool_calls_made,
        "error": result.error,
        "duration_ms": result.duration_ms,
    }, ensure_ascii=False, indent=2)
