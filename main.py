"""
入口
"""
from config import HOST, PORT, DEBUG, MODELS_DIR
from backends.gpu import HAVE_GPU, list_models, is_loaded, get_config

if __name__ == "__main__":
    from app import app

    print(f"\n{'='*50}")
    print(f"LLM Web — 双后端")
    print(f"{'='*50}")

    
    print(f"[OK] Backend: llama-cpp-python")
    print(f"     Model dir: {MODELS_DIR}")
    models = list_models()
    print(f"     Available: {len(models)}")

    print(f"\n    http://localhost:{PORT}")
    print(f"{'='*50}\n")

    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
