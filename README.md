# LLM Chat — 双后端多模态 Web 聊天

本地大模型 Web 聊天界面，支持 **llama-cpp-python**（GPU/CPU 直跑 GGUF）和 **Ollama**（本地/云端模型）双后端实时切换，支持**多模态图片识别**、流式输出、提示词模板、Web 搜索。

## 功能特性

- **双后端切换** — 侧边栏下拉框一键切换 llama-cpp / Ollama，无需重启
- **多模态图片识别** — 两个后端均支持上传图片让模型识别描述
  - llama-cpp: 自动检测同目录 mmproj 文件，动态匹配 Qwen3.5/Qwen2.5-VL/LLaVA 等 ChatHandler
  - Ollama: 图片以 `images` 字段传入，支持本地和云端视觉模型
- **流式输出** — 逐 token 实时显示，支持中断
- **提示词模板** — 内置图像优化师、翻译器等模板，可自定义 CRUD
- **Web 搜索** — DuckDuckGo 搜索结果，辅助模型回答
- **GPU/CPU 自动检测** — 启动时自动检测 CUDA，无 GPU 则走 CPU 模式或 Ollama

## 文件结构

```
llama-cpp_vlm_web/
├── main.py                # 入口，启动时打印后端状态
├── config.py              # 常量参数（端口、模型目录、推理默认值）
├── app.py                 # Flask 路由（统一入口 + 双后端路由 + 模板 CRUD + 搜索）
├── gpu_backend.py         # llama-cpp-python 后端（模型加载/推理/mmproj 自动检测）
├── ollama_backend.py      # Ollama 后端（REST API 代理/流式对话）
├── prompts.py             # 提示词模板引擎（CRUD + 持久化 JSON）
├── prompt_templates.json  # 模板存储文件
├── requirements.txt       # Python 依赖
├── start.bat              # Windows 一键启动
├── static/
│   ├── css/style.css      # 样式
│   └── js/app.js          # 前端逻辑（双后端适配/图片上传/SSE 流式）
└── templates/
    └── index.html         # 单页应用
```

## 快速开始

```powershell
# 安装 Web 依赖
pip install flask flask-cors requests pillow

# 启动
python main.py
```

访问 http://localhost:5000

启动时自动检测后端：
- llama-cpp-python 已安装 → 优先使用
- Ollama 正在运行 → 使用 Ollama
- 都没有 → 降级模式，启动后手动启动 Ollama 或安装 llama-cpp-python

## GPU 机器（llama-cpp-python）

```powershell
# 安装 CUDA 版依赖
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install llama-cpp-python --index-url https://jamepeng.github.io/whl/cu121
pip install flask flask-cors pillow numpy

# 放置模型（GGUF + mmproj 同目录自动识别）
# 默认模型目录: D:\Scoop\LLM\
# 例: D:\Scoop\LLM\qwen3.5-4B-Q4\model.gguf
#     D:\Scoop\LLM\qwen3.5-4B-Q4\mmproj-model-f16.gguf

python main.py
```

### 纯 CPU 机器编译 llama-cpp-python（0.3.36+）

> 0.3.19 版本 Python 绑定存在图片幻觉 bug（CLI 正确但 Python 输出幻觉），0.3.36 已修复。

```powershell
# 需要CMake + VS Build Tools 2022 (MSVC)
git clone https://github.com/JamePeng/llama-cpp-python.git
cd llama-cpp-python
git checkout v0.3.36
git submodule update --init  # 或手动下载 llama.cpp

# 编译（CPU 模式）
$env:CMAKE_ARGS = "-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
pip install . --no-binary :all:
```

### mmproj 自动检测

模型同目录下以 `mmproj` 开头且 `.gguf` 结尾的文件会被自动识别为视觉投影模型。加载模型时无需手动指定。

支持的 ChatHandler（按优先级尝试）：
1. `Qwen35ChatHandler` — Qwen3.5 系列
2. `Qwen3VLChatHandler` — Qwen3 VL 系列
3. `Qwen25VLChatHandler` — Qwen2.5 VL 系列
4. `Llava16ChatHandler` — LLaVA 1.6
5. `Llava15ChatHandler` — LLaVA 1.5

## CPU 机器（Ollama）

```powershell
scoop install ollama
ollama serve
ollama pull qwen2.5

pip install flask flask-cors pillow requests
python main.py
```

### Ollama 多模态图片识别

Ollama 后端支持本地模型和云端模型（如 `qwen3.5:cloud`）的图片识别。前端上传图片后自动转为纯 base64 通过 `images` 字段传给 Ollama API。

> **注意**: Ollama API 的 `images` 字段只接受纯 base64 字符串（不含 `data:image/...;base64,` 前缀），传 data URI 会报 `illegal base64 data` 错误。

## 配置

编辑 `config.py` 或通过环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODELS_DIR` | `D:\Scoop\LLM` | llama-cpp 模型目录 |
| `OLLAMA_BASE` | `http://localhost:11434` | Ollama API 地址 |
| `PORT` | `5000` | Web 服务端口 |
| `GPU_DEFAULT_CTX` | `8192` | 默认上下文长度 |
| `GPU_DEFAULT_LAYERS` | `-1` | GPU 层数（-1=全部，0=仅CPU） |

## API

### 通用

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/status` | GET | 后端状态（当前后端 + 可用后端列表） |
| `/api/switch_backend` | POST | 切换后端 `{"backend": "llama-cpp"\|"ollama"}` |
| `/api/health` | GET | 健康检查 |
| `/api/upload_image` | POST | 上传图片（返回 base64） |
| `/api/search` | GET/POST | DuckDuckGo 搜索 |

### llama-cpp 后端

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/llama/status` | GET | llama-cpp 状态（GPU/CPU/模型/配置） |
| `/api/llama/models` | GET | 可用模型列表（含 mmproj 信息） |
| `/api/llama/load_model` | POST | 加载模型 `{"model": "xxx.gguf", "chat_handler": "auto"}` |
| `/api/llama/unload` | POST | 卸载模型 |
| `/api/llama/infer` | POST | 推理（支持 `images` 和 `stream`） |

### Ollama 后端

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/ollama_status` | GET | Ollama 运行状态 + 模型列表 |
| `/api/models` | GET | 已安装模型 |
| `/api/chat` | POST | 流式对话（支持 `images` 多模态） |
| `/api/pull_model` | POST | 拉取模型 |

### 提示词模板

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/prompt_templates` | GET | 列出所有模板 |
| `/api/prompt_templates/<id>` | GET | 获取模板详情 |
| `/api/prompt_templates` | POST | 保存模板 |
| `/api/prompt_templates/<id>` | DELETE | 删除模板 |
| `/api/enhance` | POST | 执行模板增强 |

## 已知问题

- `GPU_DEFAULT_LAYERS=-1` 在纯 CPU 机器上会把所有层放 GPU（实际无 GPU），需改为 0
- Ollama 云端模型（如 `qwen3.5:cloud`）首次请求延迟较高
- llama-cpp-python 0.3.19 Python 绑定存在图片幻觉 bug，需升级到 0.3.36+

## 技术栈

- **后端**: Python 3.12 + Flask + llama-cpp-python / Ollama REST API
- **前端**: 原生 HTML/CSS/JS（无框架），SSE 流式输出
- **模型**: GGUF (llama-cpp) / Ollama 本地+云端
