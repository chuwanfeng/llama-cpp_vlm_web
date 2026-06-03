# LLM Chat Web — 生产级 AI Agent 平台

基于 Flask 的多后端 AI Agent Web 平台，支持本地大模型（llama-cpp-python）和云端厂商 API（OpenAI/DeepSeek/Anthropic/智谱等），具备完整的工具调用、审批流、自我进化、多 Agent 协作、定时任务等企业级功能。

## 核心特性

### 多后端推理
- **llama-cpp-python** — GPU/CPU 直跑 GGUF，自动检测 CUDA，支持多模态图片识别
- **厂商 API** — OpenAI / DeepSeek / Anthropic / Gemini / 通义千问 / 智谱 AI / Moonshot / 自定义
- **实时切换** — 侧边栏一键切换后端，无需重启

### Agent 能力（对标 hermes-agent）
- **工具调用** — 30+ 内置工具（文件读写、终端执行、网页浏览、代码执行、联网搜索等）
- **审批流** — 双层安全检测（Hardline 无条件阻止 + Dangerous 智能风险评估），支持 YOLO 模式
- **自我进化** — 对话结束后自动 Review，生成改进建议并持久化到记忆
- **多 Agent 协作** — 顺序执行模式，上下文传递，支持自定义角色和提示词
- **MCP 集成** — 支持 stdio/HTTP 双传输，动态工具发现
- **定时任务** — Cron 表达式调度，支持 Agent 对话/HTTP 请求/Shell 命令三种处理器
- **插件系统** — 可插拔架构，支持生命周期管理和钩子系统

### 上下文管理
- **四层压缩管道** — Snip / Micro / Auto / Reactive 压缩策略
- **128K 上下文窗口** — 自动管理长对话
- **记忆系统** — SQLite FTS5 全文搜索，持久化存储

### 前端界面
- **单页应用** — 原生 HTML/CSS/JS，无需前端框架
- **流式输出** — SSE 实时推送，支持中断
- **多模态** — 图片上传识别，文件附件支持
- **管理面板** — 技能管理、审批流、进程管理、定时任务、插件管理

## 快速开始

### 环境要求
- Python 3.10+
- Windows / Linux / macOS

### 安装

```powershell
# 克隆项目
git clone <repository-url>
cd llama-cpp_vlm_web

# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
```

访问 http://localhost:5000

### GPU 机器（推荐）

```powershell
# 安装 CUDA 版 llama-cpp-python
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install llama-cpp-python --index-url https://jamepeng.github.io/whl/cu121

# 放置模型（GGUF + mmproj 同目录）
# 默认目录: D:\Scoop\LLM\
# 例: D:\Scoop\LLM\qwen3.5-4B-Q4\model.gguf
#     D:\Scoop\LLM\qwen3.5-4B-Q4\mmproj-model-f16.gguf
```

## 项目结构

```
llama-cpp_vlm_web/
├── main.py                    # 入口
├── config.py                  # 配置常量
├── app.py                     # Flask 应用 + 路由注册
├── API.md                     # API 文档
├── agent/
│   ├── loop.py               # AgentLoop 引擎（工具调用循环）
│   ├── context_engine.py     # 上下文引擎抽象基类
│   └── self_improve/         # 自我进化模块
│       ├── review.py         # 对话 Review
│       ├── provenance.py     # 来源追踪
│       └── curator.py        # 技能整理器
├── backends/                  # 推理后端
│   ├── gpu.py                # llama-cpp-python
│   └── vendors.py            # 厂商 API
├── tools/                     # 工具系统
│   ├── registry.py           # 工具注册中心（AST 自动发现）
│   ├── approval.py           # 审批流（安全检测）
│   ├── process_registry.py   # 进程管理
│   ├── mcp_tool.py           # MCP 集成
│   ├── builtin_*.py          # 内置工具（30+）
│   └── skills/               # 技能目录
├── services/                  # 服务层
│   ├── agent_service.py      # Agent 服务（SSE 流式）
│   ├── context_compressor.py # 上下文压缩
│   ├── memory_manager.py     # 记忆管理
│   └── search.py             # 联网搜索
├── cron/                      # 定时任务
│   ├── jobs.py               # 任务定义
│   └── scheduler.py          # 调度器
├── plugins/                   # 插件系统
│   ├── base.py               # 插件基类
│   └── memory/               # 记忆插件
├── tests/                     # 测试套件（67 个测试）
├── static/                    # 前端静态资源
│   ├── css/style.css
│   └── js/app.js
└── templates/
    └── index.html
```

## API 概览

### Agent 对话
```bash
POST /api/agent/chat/stream
Content-Type: application/json

{
  "message": "分析当前目录的 Python 文件",
  "vendor_id": "deepseek",
  "model": "deepseek-chat",
  "tools_enabled": true
}
```

**SSE 事件类型**: `token`, `tool_call`, `tool_result`, `reasoning`, `review`, `done`

### 工具管理
```bash
GET  /api/tools              # 列出所有工具
POST /api/tools/execute      # 执行单个工具
```

### 审批流
```bash
GET  /api/approval/status    # 审批状态
GET  /api/approval/pending   # 待审批列表
POST /api/approval/approve   # 批准请求
POST /api/approval/deny      # 拒绝请求
POST /api/approval/yolo      # 切换 YOLO 模式
```

### 定时任务
```bash
GET    /api/cron/jobs        # 列出任务
POST   /api/cron/jobs        # 创建任务
PUT    /api/cron/jobs/<id>   # 更新任务
DELETE /api/cron/jobs/<id>   # 删除任务
```

### 插件
```bash
GET  /api/plugins            # 列出插件
POST /api/plugins/discover   # 发现新插件
POST /api/plugins/<id>/toggle # 启用/禁用
```

完整 API 文档见 [API.md](API.md)

## 配置

编辑 `config.py` 或通过环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODELS_DIR` | `D:\Scoop\LLM` | llama-cpp 模型目录 |
| `PORT` | `5000` | Web 服务端口 |
| `GPU_DEFAULT_CTX` | `8192` | 默认上下文长度 |
| `WEB_SEARCH_BACKEND` | `baidu` | 搜索后端（baidu/bing/360） |

## 安全特性

- **审批流** — 47 条危险命令检测规则，LLM 智能风险评估
- **Hardline 模式** — 无条件阻止 `rm -rf /`、`mkfs` 等毁灭性命令
- **YOLO 模式** — 会话级免审批（仅用于可信环境）
- **白名单** — 持久化允许的命令模式
- **命令规范化** — ANSI 转义、Unicode 全角字符防绕过

## 测试

```powershell
# 运行全部测试
pytest tests/ -v

# 运行特定模块
pytest tests/test_api.py -v
pytest tests/test_approval.py -v
pytest tests/test_cron.py -v
```

当前测试覆盖：67/67 通过

## 技术栈

- **后端**: Python 3.12 + Flask + asyncio
- **前端**: 原生 HTML/CSS/JS，SSE 流式输出
- **模型**: GGUF (llama-cpp) / 厂商 API
- **数据库**: SQLite（记忆、会话、设置）
- **测试**: pytest

## 路线图

- [x] P0: Agent 引擎 + 审批流 + 进程管理
- [x] P1: 技能系统 + 上下文管理
- [x] P2: 前端完善
- [x] P3: 测试框架 + API 文档
- [x] P4: 自我进化 + MCP + Cron + 插件 + 多 Agent
- [ ] P5: Gateway 多平台支持（Telegram/Discord/Slack）

## License

MIT
