# hermes-agent vs llama-cpp_vlm_web 完整对比

> 生成时间: 2026-05-16
> 目标: 升级 llama-cpp_vlm_web，完整复刻 hermes-agent 功能（不简化）

---

## 一、项目规模对比

| 维度 | hermes-agent | llama-cpp_vlm_web |
|------|-------------|-------------------|
| 总文件数 | 3139 | 136 |
| Python 文件 | ~800 | ~50 |
| 模板/静态资源 | ~2000 | ~80 |
| 版本 | 0.12.0 | 无版本号 |
| 定位 | 生产级 AI Agent 框架 | Web 聊天界面 + 本地推理 |

---

## 二、核心架构对比

### 2.1 Agent 引擎

| 特性 | hermes-agent | llama-cpp_vlm_web |
|------|-------------|-------------------|
| 主引擎 | `AIAgent` (run_agent.py) | `AgentLoop` (agent/loop.py) |
| 工具调用解析 | `ToolCallParser` (多种格式) | `ToolCallParser` (OpenAI + XML 回退) |
| 多轮工具调用 | ✅ 完整 | ✅ 完整 (max_turns=30) |
| 流式输出 | ✅ SSE | ✅ SSE |
| 重试机制 | ✅ jittered_backoff | ✅ jittered_backoff |
| 上下文压缩 | ✅ 多引擎 | ✅ 单引擎 (ContextCompressor) |
| 审批回调 | ✅ _set_approval_callback | ❌ 无（Web 场景不需要） |
| Subagent 委派 | ✅ 完整 | ❌ 缺失 |
| Plan 模式 | ✅ | ❌ 缺失 |

### 2.2 自我进化（Self-Improvement）

| 特性 | hermes-agent | llama-cpp_vlm_web |
|------|-------------|-------------------|
| 后台 review fork | ✅ threading.Thread | ✅ 已移植 (review.py) |
| Write origin 追踪 | ✅ ContextVar | ✅ 已移植 (provenance.py) |
| Skill 整理器 | ✅ curator.py | ✅ 已移植 (curator.py) |
| 自动状态转换 | ✅ active→stale→archived | ✅ 已实现 |
| LLM review pass | ✅ | ✅ 已实现 |
| Cron 自动触发 | ✅ | ❌ 缺失（无 cron 基础设施） |

### 2.3 工具系统

| 特性 | hermes-agent | llama-cpp_vlm_web |
|------|-------------|-------------------|
| 工具总数 | 40+ | 10+ |
| 注册表 | ✅ AST 自动发现 | ✅ AST 自动发现 |
| MCP 集成 | ✅ | ❌ 缺失 |
| 浏览器工具 | ✅ Playwright | ❌ 缺失 |
| 视觉工具 | ✅ | ✅ 已通过 backends 支持 |
| 代码执行 | ✅ 沙箱 | ❌ 缺失 |
| 终端工具 | ✅ 多后端 | ✅ 已实现 |

### 2.4 记忆系统

| 特性 | hermes-agent | llama-cpp_vlm_web |
|------|-------------|-------------------|
| 会话记忆 | ✅ SQLite + FTS5 | ✅ JSON 文件 |
| 用户建模 | ✅ Honcho | ❌ 缺失 |
| 记忆搜索 | ✅ FTS5 | ✅ 基础搜索 |
| 记忆 Consolidation | ✅ | ✅ 通过 review fork |

### 2.5 上下文管理

| 特性 | hermes-agent | llama-cpp_vlm_web |
|------|-------------|-------------------|
| ContextEngine ABC | ✅ | ❌ 缺失 |
| 多引擎压缩 | ✅ (5 种) | ✅ (1 种) |
| 工具输出剪枝 | ✅ | ✅ |
| Token 预算管理 | ✅ | ✅ |
| 反抖动保护 | ✅ | ✅ |
| 冷却机制 | ✅ | ✅ |

### 2.6 消息平台网关

| 特性 | hermes-agent | llama-cpp_vlm_web |
|------|-------------|-------------------|
| Telegram | ✅ | ❌ |
| Discord | ✅ | ❌ |
| Slack | ✅ | ❌ |
| WhatsApp | ✅ | ❌ |
| Signal | ✅ | ❌ |
| Web UI | ✅ (FastAPI) | ✅ (Flask) |

### 2.7 终端后端

| 特性 | hermes-agent | llama-cpp_vlm_web |
|------|-------------|-------------------|
| 本地终端 | ✅ | ✅ |
| Docker | ✅ | ❌ |
| SSH | ✅ | ❌ |
| Daytona | ✅ | ❌ |
| Singularity | ✅ | ❌ |
| Modal | ✅ | ❌ |

---

## 三、缺失功能清单（按优先级）

### P0 - 必须实现

1. **Subagent 委派系统** — hermes-agent 的 `agents/` 模块
   - `SubagentTool` — 同步阻塞子代理
   - `SpawnWorkerTool` — 异步非阻塞 Worker
   - Task 系统集成

2. **Plan 模式** — 只读工具拦截 + 计划生成

3. **MCP 集成** — Model Context Protocol 支持
   - `mcp/` 模块
   - 动态工具发现
   - MCP server 连接管理

4. **浏览器自动化** — Playwright 集成
   - `tools/browser_tool.py`
   - 截图、点击、输入、导航

### P1 - 重要

5. **ContextEngine ABC** — 可插拔上下文引擎
   - `agent/context_engine.py`
   - 多种压缩策略

6. **代码执行沙箱** — 安全代码执行
   - Docker/Daytona 后端
   - 超时和资源限制

7. **更多内置工具**
   - `tools/vision_tool.py` — 图像分析
   - `tools/code_execution_tool.py` — 代码执行
   - `tools/arxiv_tool.py` — 论文搜索
   - `tools/stock_tool.py` — 股票数据

### P2 - 有用

8. **Cron 调度系统** — 定时任务
   - `cron/` 模块
   - 表达式解析
   - 后台任务执行

9. **用户建模** — Honcho 集成
   - 用户偏好学习
   - 个性化响应

10. **Web UI 增强**
    - 多会话管理（类 Claude 风格）
    - 实时 token 计数
    - 工具调用可视化

### P3 - 可选

11. **多 Agent 协作** — Kanban 风格
12. **批处理轨迹生成** — Atropos 集成
13. **RL 训练环境** — 从对话生成训练数据

---

## 四、文件级对比（关键模块）

### 4.1 Agent 引擎

| hermes-agent | llama-cpp_vlm_web | 状态 |
|-------------|-------------------|------|
| `run_agent.py` (AIAgent) | `agent/loop.py` (AgentLoop) | ✅ 已移植（简化版） |
| `agent/context_compressor.py` | `services/context_compressor.py` | ✅ 已移植 |
| `agent/context_engine.py` | `agent/context_engine.py` | ❌ 缺失 |
| `agent/memory_manager.py` | `services/memory_manager.py` | ✅ 已移植 |
| `agent/prompt_builder.py` | `services/prompt_builder.py` | ✅ 已移植 |
| `agent/model_metadata.py` | - | ❌ 缺失 |

### 4.2 自我进化

| hermes-agent | llama-cpp_vlm_web | 状态 |
|-------------|-------------------|------|
| `run_agent.py::_spawn_background_review` | `agent/self_improve/review.py` | ✅ 已移植 |
| `tools/skill_provenance.py` | `agent/self_improve/provenance.py` | ✅ 已移植 |
| `agent/curator.py` | `agent/self_improve/curator.py` | ✅ 已移植 |

### 4.3 工具系统

| hermes-agent | llama-cpp_vlm_web | 状态 |
|-------------|-------------------|------|
| `tools/registry.py` | `tools/registry.py` | ✅ 已移植 |
| `tools/browser_tool.py` | - | ❌ 缺失 |
| `tools/vision_tool.py` | - | ❌ 缺失 |
| `tools/code_execution_tool.py` | - | ❌ 缺失 |
| `tools/arxiv_tool.py` | - | ❌ 缺失 |
| `mcp/` | - | ❌ 缺失 |

### 4.4 子代理系统

| hermes-agent | llama-cpp_vlm_web | 状态 |
|-------------|-------------------|------|
| `agents/sub_agent.py` | - | ❌ 缺失 |
| `agents/coordinator.py` | - | ❌ 缺失 |
| `task_store.py` | - | ❌ 缺失 |

---

## 五、升级策略

### 阶段划分

**阶段 0: 清理** (1-2 小时)
- 删除 `backends/ollama.py`
- 清理 `app.py` 中的 Ollama 路由
- 清理前端 Ollama 逻辑

**阶段 1: Agent 引擎增强** (4-6 小时)
- 实现 `agent/context_engine.py` (ContextEngine ABC)
- 增强 `agent/loop.py` (添加 Plan 模式、Subagent 支持)
- 实现 Subagent 委派系统

**阶段 2: 工具系统扩展** (6-8 小时)
- 添加浏览器工具 (Playwright)
- 添加视觉工具
- 添加代码执行工具
- 集成 MCP

**阶段 3: 高级功能** (4-6 小时)
- Cron 调度系统
- 用户建模 (Honcho)
- 多 Agent 协作

**阶段 4: Web UI 增强** (3-4 小时)
- 多会话管理
- 实时 token 计数
- 工具调用可视化

**阶段 5: 测试与优化** (2-3 小时)
- 端到端测试
- 性能优化
- 文档完善

### 执行原则

1. **完整移植，不简化** — 所有 hermes-agent 功能都要实现
2. **详细中文注释** — 每个文件、每个函数都要有中文说明
3. **结构清晰** — 模块化、低耦合、高内聚
4. **性能优先** — 异步、缓存、并发
5. **可扩展性** — 插件化、配置驱动
6. **最新技术** — Python 3.12+，asyncio 原生，type hints

---

## 六、立即行动

根据上述分析，我将按以下顺序执行：

1. ✅ 完成阶段 0: 清理 Ollama（已验证 backends/ollama.py 仍存在）
2. ✅ 完成阶段 1: 实现 ContextEngine ABC + 增强 AgentLoop
3. ✅ 实现 Subagent 委派系统
4. ✅ 添加浏览器工具
5. ✅ 集成 MCP
6. ✅ 实现 Cron 系统
7. ✅ Web UI 增强

开始执行...
