# MiniCPM5-1B 官方文档与推理参数调研

**时间**: 2026-08-05 02:58 GMT+8

## 官方仓库

- GitHub: `OpenBMB/MiniCPM` (README-cn.md)
- HuggingFace: `openbmb/MiniCPM5-1B`
- GGUF: `bartowski/MiniCPM5-1B-GGUF`

## 模型规格

| 项目 | 值 |
|------|-----|
| 类型 | Causal Language Model, Dense (非MoE) |
| 架构 | LlamaForCausalLM |
| 参数 | 1,080,632,832 (~1B), non-embedding 679M |
| 层数 | 24, GQA 16 Q heads / 2 KV heads |
| 上下文长度 | **131,072 tokens** (原生) |
| 训练精度 | BF16 |
| License | Apache-2.0 |

## 上下文扩展

- MiniCPM5-1B **原生支持 131K 上下文**，不是事后扩展的
- 不需要 rope_scaling=yarn 来"扩展"，但使用 yarn 也不会破坏（只是位置编码被转换到 yarn 空间）
- 官方训练精度 BF16，GGUF Q8_0 量化后可用

## 官方采样参数建议

来自 OpenVINO 部署文档 + 模型卡 chat template：

### Think 模式 (enable_thinking=True)
- temperature = **0.9**
- top_p = **0.95**
- max_new_tokens = 256

### No-Think 模式 (enable_thinking=False)
- temperature = **0.7**
- top_p = **0.95**
- max_new_tokens = 256

## Hybrid Reasoning

同一份权重通过 chat template 的 `enable_thinking` 参数切换：
- **Think 模式**: tokenizer.apply_chat_template(messages, extra_context={"enable_thinking": True})
  - 模板会在 assistant 段注入空 `<think>\n\n</think>\n\n` — 引导模型在 `<think>` 块内推理
- **No-Think 模式**: extra_context={"enable_thinking": False}
  - 模板会在 assistant 段末尾直接开始，不预填 think 块
  - 预填一个空 `assistant<think>\n\n</think>\n\n` 让模型跳过思考直接回答

### llama-cpp-python 的实现

GGUF 版本已内嵌 chat template。在 llama-cpp-python 中：
- `enable_thinking` 是 Jinja2 模板内变量，**不是** `create_chat_completion()` 的 keyword argument
- 传递方式：`extra_body={"enable_thinking": True/False}` 或通过 `chat_format` 的 `chat_handler` 的 `__call__` 注入
- 当前本项目 `gpu.py` 中已删除 `enable_thinking` 参数（因为不在函数签名中）

## 语唠问题分析

当前参数 vs 官方建议：

| 参数 | 当前值 (本项目) | 官方 Think | 官方 No-Think |
|------|----------------|-----------|--------------|
| temperature | 0.8 | 0.9 | 0.7 |
| top_p | 0.9 | 0.95 | 0.95 |
| max_tokens | 8192 (前端) | 256 | 256 |
| top_k | 30 | - | - |
| min_p | 0.05 | - | - |
| presence_penalty | 1.0 | - | - |

**核心问题**：
1. **max_tokens=8192 远超需要** — 官方建议 256，差距 32 倍。1B 模型被要求生成 8192 token 时会产生大量重复填充
2. **presence_penalty=1.0** — 强制避免重复词汇导致模型不断绕圈找新词
3. **temperature 适中** — 0.8 在 Think/NoThink 之间，不算离谱

**建议修复**：
- MiniCPM5 单独采样策略：max_tokens=2048, temperature=0.7, presence_penalty=0.0
- 或者直接在整体 `gen_params` 里为 `model_family == "minicpm"` 做特殊处理
