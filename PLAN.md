# llama-cpp_vlm_web 升级落地方案

> 用户需求：前端极致 + 引擎升级（hermes-agent 能力）
> 三大原则：详细中文注释 | 结构清晰/性能/扩展/复用 | 跟进最新技术

---

## 项目定位

- **llama-cpp_vlm_web** 是 Web 界面 + 本地推理（llama-cpp-python）+ 多厂商 API
- **目标**：Web 界面极致 + Agent 引擎升级 + 自我进化能力（从 hermes-agent 移植）

### 架构原则
- **本地后端**：只保留 llama-cpp-python（GPU/CPU），去掉本地 Ollama（Ollama Cloud 已在 vendors 里走 API）
- **Agent 引擎**：服务端统一循环（替换前端 JS 工具循环）
- **自我进化**：后台 review fork，持续学习用户偏好和项目知识

---

## 阶段零：hermes-agent 核心能力解析

### 自我进化（Self-Improvement）— 最强能力 ⚡⚡⚡

hermes-agent 在每次对话轮次结束后，**自动在后台线程启动 review agent**：

```
用户提问 → 主 Agent 处理 → 对话结束
                    ↓
         [后台 review fork 启动]
                    ↓
         Review Agent（工具集限制为 memory+skills）
                    ↓
    扫描对话历史 → 发现模式/偏好/错误/教训
                    ↓
    自动创建/更新 Skill（标记 agent_created=True）
    自动更新 Memory（consolidate/prune）
                    ↓
    最终结果推送给用户：
    💾 Self-improvement review: created 'python-debug-skill' · updated 'read-memory-pattern'
```

**核心机制**（参考 `run_agent.py:3634`）：
- `threading.Thread(target=_run_review, daemon=True)` — 后台非阻塞执行
- `review_agent._memory_write_origin = "background_review"` — 标记来源
- `skill_provenance.py` — 用 ContextVar 区分「用户指令创建的 skill」vs「review agent 自动创建的 skill」
- `curator.py` — 只管理 agent-created skills，不碰用户 skill
- 工具集限制：`enabled_toolsets=["memory", "skills"]` — review agent 只能用记忆和技能工具

**关键文件**：
- `run_agent.py` — `_spawn_background_review()` + `_run_background_review()`
- `agent/memory_manager.py` — curator 逻辑
- `tools/skill_provenance.py` — write origin tracking
- `agent/skill_commands.py` — skill CRUD 在 review agent 中的行为差异

---

## 阶段一：Agent Loop 引擎（P0）

### 目标
服务端统一 Agent 引擎，替换前端 JS 工具循环，实现多轮工具调用 + 自我进化。

### 新增/改造文件

```
agent/
  __init__.py
  loop.py              # AgentLoop（已创建，~600行，中文注释）
  context_engine.py    # ContextEngine 抽象基类
  self_improve/
    __init__.py        # 自我进化模块
    review.py          # 后台 review fork（hermes-agent _spawn_background_review 移植）
    curator.py         # Skill 整理器（只管理 agent_created skills）
    provenance.py      # Write origin tracking（ContextVar 机制）
  engines/
    __init__.py
    compressor.py      # 上下文压缩引擎

tools/
  registry.py          # 已升级（添加 get_tool/get_tool_names/get_schemas）
  builtin_edit.py      # 新增：精确行级编辑
  builtin_grep.py      # 新增：代码搜索

services/
  agent_service.py     # Flask 路由 → AgentLoop 的桥梁

backends/
  gpu.py               # 保留（llama-cpp-python）
  vendors.py           # 保留（多厂商 API，含 ollama-cloud）
  ollama.py            # 【删除】本地 Ollama 不需要，Ollama Cloud 在 vendors
```

### 关键改动

#### 1. `agent/self_improve/review.py` — 自我进化核心

```python
"""
后台 Review Fork

hermes-agent 的自我进化机制：
- 主对话结束后，在后台线程启动 review agent
- review agent 工具集限制为 ["memory", "skills"]
- 扫描对话历史，发现模式/偏好/错误
- 自动创建/更新 skill，标记 agent_created=True
- 自动更新记忆（consolidate/prune）
- 结果推送给用户：💾 Self-improvement review: ...
"""

import threading
from dataclasses import dataclass

@dataclass
class ReviewResult:
    actions: List[str]   # 成功创建/更新的内容
    summary: str         # 摘要（用 · 连接）

def spawn_background_review(
    messages: List[Dict],
    memory_store,
    skill_manager,
    callback: callable = None
) -> threading.Thread:
    """启动后台 review 线程"""
    t = threading.Thread(target=_run_review, daemon=True, name="bg-review")
    t.start()
    return t
```

#### 2. `agent/self_improve/provenance.py` — 来源追踪

```python
"""
Skill Write Origin Tracking

区分两种 skill 创建来源：
- "foreground": 用户指令创建的 skill → 保护，不自动管理
- "background_review": review agent 自动创建的 skill → curator 管理

实现：ContextVar（线程安全，async-safe）
参考 hermes-agent/tools/skill_provenance.py
"""
import contextvars

_write_origin: contextvars.ContextVar[str] = contextvars.ContextVar(
    "skill_write_origin", default="foreground"
)

def set_background_review():
    """在 review fork 中调用，标记所有后续 skill 操作为 agent 创建"""
    _write_origin.set("background_review")

def is_agent_created() -> bool:
    return _write_origin.get() == "background_review"
```

#### 3. `agent/self_improve/curator.py` — Skill 整理器

```python
"""
Skill Curator

只管理 agent_created=True 的 skills：
- 合并相似技能
- 剪枝低使用率技能
- 更新已有技能（基于新学到的模式）
- 不管理用户手动创建的 skills
"""
```

#### 4. `backends/ollama.py` — 删除

本地 Ollama 不是必须的：
- llama-cpp-python 已经能跑本地模型
- Ollama Cloud 已通过 vendors.py 走厂商 API
- 减少维护负担

删除步骤：
1. 删除 `backends/ollama.py`
2. 删除 `app.py` 中所有 `/api/ollama_*` 路由
3. 删除 `config.py` 中 Ollama 配置
4. 删除 `app.js` 中所有 Ollama 相关逻辑（sendOllama、loadOllamaModels 等）
5. 更新前端 HTML（侧边栏后端选择器移除 Ollama）

### 验收标准
- [ ] AgentLoop 语法正确 ✓（上次已完成）
- [ ] `agent/self_improve/review.py` 后台 review fork 实现
- [ ] `agent/self_improve/provenance.py` write origin 追踪
- [ ] `agent/self_improve/curator.py` skill 整理器
- [ ] 删除 backends/ollama.py
- [ ] 删除 app.py 中 Ollama 路由
- [ ] 前端移除 Ollama 逻辑
- [ ] 端到端验证：对话结束 → 观察 💾 Self-improvement review 输出

---

## 阶段二：上下文管理升级（P1）

（内容同前，略）

---

## 阶段三：技能系统完善（P1）

（内容同前，略）

---

## 阶段四：前端极致打磨（P2）

（内容同前，略）

---

## 阶段五：高级功能（P3）

- Cron 定时任务系统
- 插件系统（记忆提供者：Honcho/Mem0）
- 浏览器自动化（Playwright）
- 多 Agent 协作（Kanban）

---

## 项目结构（升级后）

```
llama-cpp_vlm_web/
├── agent/
│   ├── loop.py             # AgentLoop 核心（已完成）
│   ├── context_engine.py   # ContextEngine 抽象
│   └── self_improve/       # 【新增】自我进化模块
│       ├── __init__.py
│       ├── review.py       # 后台 review fork
│       ├── provenance.py   # Write origin tracking
│       └── curator.py      # Skill 整理器
│   └── engines/
│       └── compressor.py
├── tools/
│   ├── registry.py          # 已升级
│   ├── builtin_edit.py      # 精确编辑（待实现）
│   └── builtin_grep.py      # 代码搜索（待实现）
├── skills/                  # 升级为完整框架
├── services/
│   └── agent_service.py     # Flask 路由桥梁
├── backends/
│   ├── gpu.py               # ✅ llama-cpp-python
│   ├── vendors.py           # ✅ 多厂商 API（含 ollama-cloud）
│   └── ollama.py            # ❌ 删除（本地 Ollama）
├── static/
├── templates/
├── app.py                   # 移除 Ollama 路由
├── config.py                # 移除 Ollama 配置
└── PLAN.md
```

---

## 技术选型

| 技术 | 选型 |
|------|------|
| Python 3.12 | 保持 |
| asyncio 原生 | 全部 async |
| dataclasses | AgentResult/ToolError/ReviewResult |
| threading.Thread | 后台 review（daemon=True，不阻塞主对话） |
| contextvars | Skill write origin tracking |
| Flask SSE | 流式输出 |
| ToolSet 系统 | 工具集分发 + 概率选择 |

---

## 执行顺序

1. **清理 Ollama**（阶段零）— 删除本地 Ollama 代码
2. **阶段一** — AgentLoop + 自我进化（后台 review fork）
3. **阶段二** — 上下文管理升级
4. **并行** — 阶段三（技能）+ 阶段四（前端）
5. **阶段五** — 高级功能

---

_最后更新：2026-05-11 00:00_