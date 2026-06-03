"""
辅助模型端到端测试
用法: cd D:\vps\python\llama-cpp_vlm_web && python scripts/test_aux.py
前提: 1) settings.json 中 aux_config 已配置 2) 对应厂商 vendorCreds 已填
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.auxiliary import auxiliary_chat, is_aux_enabled, make_aux_callable

def load_config():
    with open("settings.json", "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    settings = load_config()
    aux = settings.get("aux_config", {})

    print("=" * 60)
    print("当前 aux_config:", json.dumps(aux, ensure_ascii=False, indent=2))
    print("=" * 60)

    # 1) 检查是否启用
    for task in ["compression", "memory", "search"]:
        enabled = is_aux_enabled(task, aux)
        print(f"  is_aux_enabled('{task}') → {enabled}")

    if not aux.get("enabled"):
        print("\n⚠️ 辅助模型未启用，跳过对话测试。请先在设置面板打开。")
        return

    # 2) 发一条简单对话
    print("\n--- 发送测试对话 ---")
    result = auxiliary_chat(
        messages=[{"role": "user", "content": "用一句话解释什么是递归"}],
        task="compression",
        aux_config=aux,
        max_tokens=256,
        temperature=0.3
    )

    if result:
        print(f"✅ 辅助模型返回: {result[:200]}...")
    else:
        print("❌ 返回为空 — 检查 provider/model/api_key 配置")

    # 3) 测试 make_aux_callable (ContextCompressor 兼容)
    print("\n--- 测试 make_aux_callable ---")
    fn = make_aux_callable(task="compression", aux_config=aux)
    result2 = fn(
        messages=[{"role": "user", "content": "用5个字总结：今天天气很好阳光明媚"}],
        model="aux-model",
        max_tokens=128
    )
    print(f"callable 返回: {result2[:200] if result2 else '(空)'}")

if __name__ == "__main__":
    main()
