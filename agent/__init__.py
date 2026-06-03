"""
Agent 引擎模块

提供基于 hermes-agent 架构的多轮工具调用引擎，支持：
- 标准 OpenAI tool calling（function calling）
- XML 格式工具调用回退解析（适用于 llama-cpp 等不支持原生 tool calling 的后端）
- 多轮工具执行循环（最多 N 轮）
- 推理内容提取（支持多种模型输出格式）
- 工具执行错误恢复
- 统一后端接口（GPU/Vendors）

架构参考：hermes-agent/environments/agent_loop.py
"""

from agent.loop import AgentLoop, AgentResult, ToolError

__all__ = [
    "AgentLoop",
    "AgentResult",
    "ToolError",
]
