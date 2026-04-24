# llama-cpp VLM Web - 更新说明

## 新增功能

### 1. ✅ llama-cpp CPU 模式支持
- 自动检测 GPU 可用性，若无 GPU 则自动降级到 CPU 模式
- 加载模型时可指定 `force_cpu=True` 强制使用 CPU
- 后端优先级：**llama-cpp (GPU/CPU) → Ollama**

### 2. ✅ 统一前端接入
- 前端自动检测后端类型（llama-cpp / Ollama）
- 统一的对话接口，无需手动切换
- 状态栏显示当前后端和模式（GPU/CPU）

### 3. ✅ Web 搜索功能
- 集成 DuckDuckGo 搜索（无需 API key）
- 点击 🔍 按钮搜索
- 搜索结果可一键插入对话

## 安装与运行

### 安装依赖
```bash
pip install -r requirements.txt
```

新增依赖：`requests`, `beautifulsoup4`（搜索功能）

### 启动服务
```bash
python app.py
```

访问 http://localhost:5000

## 使用指南

### 选择后端
- 如果安装了 `llama-cpp-python`，自动使用 llama-cpp 后端
- 否则 fallback 到 Ollama（需运行 `ollama serve`）

### 加载模型（llama-cpp）
1. 将 GGUF 模型放入 `models/` 目录
2. 在 Web 界面选择模型
3. 自动加载（首次加载较慢）

### CPU 模式说明
- 无 CUDA 环境时自动启用 CPU 模式
- 可在加载模型时设置 `force_cpu=true`
- CPU 模式速度较慢，建议使用量化模型（如 Q4_K_M）

### Web 搜索
- 点击输入框旁的 🔍 按钮
- 输入关键词
- 点击搜索结果旁的"插入链接"可将结果添加到输入框

## API 端点

### 新增端点
- `GET /api/search?q=<query>` - Web 搜索
- `POST /api/switch_backend` - 切换后端（预留）
- `POST /api/llama/load_model` - 加载 llama 模型（支持 `force_cpu`）

### 后端状态
- `/api/status` 返回 `backend` 字段：`llama-cpp` / `ollama` / `none`
- 返回 `gpu_available` 和 `cpu_mode` 信息

## 故障排除

### llama-cpp 加载失败
- 检查模型路径：`models/` 目录下是否有 `.gguf` 文件
- 检查内存：大模型需要足够 RAM（CPU 模式）或 VRAM（GPU）
- 尝试量化模型：推荐 `qwen2.5-7b-instruct-q4_k_m.gguf`

### 搜索功能失效
- 检查网络连接
- DuckDuckGo HTML API 可能被限制，可改用 Brave Search（需 API key）

## 文件修改清单

| 文件 | 修改内容 |
|------|---------|
| `gpu_backend.py` | 添加 CPU 模式支持、`detect_backend()`、`force_cpu` 参数 |
| `app.py` | 重构后端选择逻辑、添加搜索 API |
| `static/js/app.js` | 支持双后端、集成搜索功能 |
| `templates/index.html` | 添加搜索按钮 |
| `static/css/style.css` | 搜索结果显示样式 |
| `requirements.txt` | 添加 `requests`、`beautifulsoup4` |

## 注意事项

1. **CPU 模式性能**：7B 模型约需要 4-6GB RAM，生成速度约 2-5 tokens/秒
2. **多模态支持**：需要下载 mmproj 文件放入 `models/mmproj/`
3. **Ollama 兼容**：原有 Ollama 功能完全保留，作为备选后端

## 下一步优化

- [ ] 添加流式输出支持（llama-cpp）
- [ ] 搜索历史记录
- [ ] 模型下载进度显示
- [ ] 后端切换 UI 控件
