"""
入口
"""
from config import HOST, PORT, DEBUG, MODELS_DIR
from gpu_backend import HAVE_GPU, list_models, is_loaded, get_config
from ollama_backend import available as ollama_available, get_models as ollama_get_models

if __name__ == "__main__":
    from app import app

    print(f"\n{'='*50}")
    print(f"LLM Web — 双后端")
    print(f"{'='*50}")

    if HAVE_GPU:
        print(f"✅ 后端: GPU (llama-cpp-python)")
        print(f"   模型目录: {MODELS_DIR}")
        models = list_models()
        print(f"   可用模型: {len(models)} 个")
        if is_loaded():
            print(f"   当前模型: {get_config().get('model')}")
        else:
            print("   调用 GET /api/gpu/models 查看，POST /api/gpu/load_model 加载")
    else:
        print(f"🌐 后端: Ollama (CPU)")
        if ollama_available:
            print(f"   已安装模型: {ollama_get_models()}")
        else:
            print("   ⚠️  Ollama 未运行，请先启动: ollama serve")

    print(f"\n🌐 http://localhost:{PORT}")
    print(f"{'='*50}\n")

    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
