# LLM Web — 独立双后端

自动检测 GPU 切换后端。

## 文件结构

```
llama-cpp_vlm_web/
  main.py           # 入口
  config.py         # 常量参数
  gpu_backend.py    # llama-cpp-python 后端
  ollama_backend.py # Ollama 后端
  app.py            # Flask 路由
  templates/
    index.html      # 前端
  models/           # GPU 模式：放 GGUF 文件
```

## 启动

```powershell
python main.py
```

自动检测：
- 有 CUDA → GPU 模式（llama-cpp-python）
- 无 CUDA → Ollama 模式（CPU）

---

## GPU 机器

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install llama-cpp-python --index-url https://jamepeng.github.io/whl/cu121
pip install flask flask-cors pillow

python main.py
```

把 GGUF 模型放到 `models/` 目录。

---

## CPU 机器

```powershell
scoop install ollama
ollama serve
ollama pull qwen2.5

pip install flask flask-cors pillow requests
python main.py
```

---

## 配置

编辑 `config.py` 或通过环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODELS_DIR` | `./models` | GPU 模型目录 |
| `OLLAMA_BASE` | `http://localhost:11434` | Ollama API 地址 |
| `PORT` | `5000` | Web 服务端口 |

---

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/status` | GET | 后端状态 |
| `/api/gpu/models` | GET | GPU 可用模型 |
| `/api/gpu/load_model` | POST | 加载 GPU 模型 |
| `/api/gpu/infer` | POST | GPU 推理 |
| `/api/models` | GET | Ollama 已安装模型 |
| `/api/chat` | POST | Ollama 流式对话 |
| `/api/pull_model` | POST | 拉取 Ollama 模型 |