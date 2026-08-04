# 全模型参数调研汇总 — 2026-08-05

> 覆盖本项目 6 个本地模型：MiniCPM5、Gemma4-E2B、Qwen3.5-9B、Qwen3.6-35B
> 来源：HuggingFace model card / generation_config.json / GitHub 官方 README / llama.cpp 文档

---

## 1. 模型概览

| 模型 | 架构 | 参数量 | 原生上下文 | 多模态 | 思考模式 | GGUF 路径 |
|------|------|--------|-----------|--------|---------|-----------|
| MiniCPM5-1B-Thinking | LlamaForCausalLM (Dense) | 1B / 679M n-e | 131,072 (131K) | 否 | ✅ hybrid | `MiniCPM5-1B-...-Q8_0.gguf` |
| Gemma 4 E2B | Gemma4ForCausalLM (Dense) | ~2.3B | 128K | ✅ 文/图/音 | ✅ 内置 | `Gemma-4-E2B-...-Q5_K_P.gguf` + mmproj |
| Qwen3.5-9B | Qwen3.5ForCausalLM (Dense) | ~9B | 128K | ✅ 文/图 | ✅ 内置 | `Qwen3.5-9B-...-Q8_0.gguf` + mmproj |
| Qwen3.6-35B-A3B | Qwen3MoE (MoE) | 35B→3B激活 | ≥256K | ✅ 文/图/视频 | ✅ 内置 | `Qwen3.6-35B-A3B-...-Q5_K_P.gguf` + mmproj |

> 注：本项目全部使用社区 GGUF 量化版（HauhauCS Aggressive），非官方 HuggingFace 原版。量化版内嵌 chat template（V2 GGUF），llama-cpp-python 自动检测。

---

## 2. 上下文 & RoPE 配置

### 2.1 官方规格

| 模型 | 原生上下文 | RoPE freq_base | RoPE 扩展需求 |
|------|-----------|----------------|---------------|
| MiniCPM5-1B | **131,072** (131K) | ~10,000 | **不需要** |
| Gemma 4 E2B | 128K | 10,000 | **不需要** |
| Qwen3.5-9B | 128K | 1,000,000 | **不需要** |
| Qwen3.6-35B-A3B | 128K+ | 1,000,000 | **不需要** |

### 2.2 本项目当前配置 (config.py)

```python
GPU_DEFAULT_CTX = 32768
GPU_ROPE_SCALING = "yarn"     # ← 所有模型统一 yarn
GPU_ROPE_FREQ_BASE = 0.0       # 0=自动
GPU_ROPE_SCALE = 4.0           # 32768 / 8192 = 4.0

FAMILY_ROPE_BASE = {
    "gemma4":  10000.0,
    "qwen35":  1000000.0,
    "minicpm": 10000.0,       # 已修正（之前误设为 1M）
}
```

### 2.3 上下文配置分析

**结论：所有四个模型原生上下文 ≥128K，远超当前 n_ctx=32768。`rope_scaling=yarn` + `rope_scale=4.0` 对它们是多余的**——它们原生就支持 128K+，不需要 RoPE 扩展来"扩展"。

- MiniCPM5：原生 131K。yarn scale=4.0 套在 32K 上不会坏，但完全不需要。
- Gemma4 E2B：原生 128K。yarn 对 32K 无影响。
- Qwen3.5/Qwen3.6：原生 128K+。freq_base=1M 配合 n_ctx=32K，不需要 yarn 扩展。

**建议**：`rope_scaling` 默认改为 `"none"` 或保持为 `"none"` 除非用户的 n_ctx > 原生上下文。当前 `yarn` 状态无害但多余。

---

## 3. 官方推荐采样参数

### 3.1 MiniCPM5-1B（官方文档）

| 参数 | Think 模式 | No-Think 模式 |
|------|-----------|--------------|
| temperature | 0.9 | 0.7 |
| top_p | 0.95 | 0.95 |
| max_new_tokens | 256 | 256 |
| enable_thinking | True | False |

> 来源：OpenBMB/MiniCPM GitHub README + OpenVINO 部署文档

### 3.2 Gemma 4 E2B（官方 generation_config.json）

| 参数 | 默认值 | 来源 |
|------|--------|------|
| temperature | **1.0** | Google official `generation_config.json` |
| top_p | **0.95** | 同上 |
| top_k | **64** | 同上 |
| do_sample | true | 同上 |
| bos_token_id | 2 | 同上 |
| eos_token_id | [1, 106, 50] | 同上 |

> 来源：`google/gemma-4-E2B-it` 的 `generation_config.json`（通过 hf-mirror.com 获取）

### 3.3 Qwen3.5/Qwen3.6（官方文档）

| 参数 | Qwen3 官方推荐 | 说明 |
|------|---------------|------|
| temperature | **0.7** | 事实性 0.1-0.3 / 创意 1.0-1.5 |
| top_p | **0.8** | 0.1-1.0，0.9 保留 90% 概率质量 |
| top_k | **20** | 1-100，20-50 常用 |
| min_p | **0.0** | 官方推荐 |
| presence_penalty | 0-2 | **过高导致语言混杂、性能下降** |
| enable_thinking | True/False | 支持 Think/No-Think 切换 |

> 来源：Qwen3 readthedocs 官方 quickstart + Qwen3.5 GitHub README

---

## 4. 本项目当前采样参数 vs 官方推荐

### 4.1 config.py 定义的值

| 参数 | config.py 默认值 |
|------|-----------------|
| DEFAULT_TEMPERATURE | 0.8 |
| DEFAULT_TOP_P | 0.9 |
| DEFAULT_TOP_K | 30 |
| DEFAULT_MIN_P | 0.05 |
| DEFAULT_PRESENCE_PENALTY | 1.0 |
| DEFAULT_FREQUENCY_PENALTY | 0.0 |
| DEFAULT_REPEAT_PENALTY | 1.0 |
| GPU_DEFAULT_MAX_TOKENS | 4096 |

### 4.2 gpu.py gen_params 实际传入的值

⚠️ **关键发现：gpu.py 只传了 temperature、top_p、top_k、repeat_penalty、max_tokens、stream 到 `create_chat_completion()`**。

```python
# gpu.py L1412-1422
gen_params = {
    "max_tokens":     params.get("max_tokens") or GPU_DEFAULT_MAX_TOKENS,   # 4096
    "temperature":    params.get("temperature") or DEFAULT_TEMPERATURE,     # 0.8
    "top_p":          params.get("top_p") or DEFAULT_TOP_P,                 # 0.9
    "top_k":          params.get("top_k") or DEFAULT_TOP_K,                 # 30
    "repeat_penalty": params.get("repeat_penalty") or DEFAULT_REPEAT_PENALTY,# 1.0
    "stream":         stream,
}
```

**未传入的参数**（只在 config.py 定义，gpu.py 未注入）：
- ❌ `min_p` (0.05) — 定义了但没传
- ❌ `presence_penalty` (1.0) — 定义了但没传
- ❌ `frequency_penalty` (0.0) — 定义了但没传
- ❌ `typical_p` (1.0) — 定义了但没传
- ❌ `mirostat_mode/eta/tau` — 定义了但没传

> 这些参数虽然在前端 slider 有 UI 控件，但目前 `sendLlama()` 到 API → `generate_stream()` → `gen_params` 这条链路中，**min_p、presence_penalty、frequency_penalty 从未进入 gen_params dict**。前端读取了但没有传到后端 `generate_stream()` 的参数列表中。需要检查 `app.js` 的 `sendLlama()` 和 Flask 路由参数映射。

### 4.3 对比表

| 参数 | 本项目当前 | MiniCPM5 官方 | Gemma4 E2B 官方 | Qwen3.5 官方 |
|------|-----------|--------------|-----------------|-------------|
| temperature | **0.8** | 0.7/0.9 | **1.0** | **0.7** |
| top_p | **0.9** | 0.95 | **0.95** | **0.8** |
| top_k | **30** | - | **64** | **20** |
| min_p | 0.05(未注入) | - | - | 0.0 |
| presence_penalty | 1.0(未注入) | - | - | 0-2 |
| max_tokens | **4096** | 256 | - | - |
| rope_scaling | **yarn** | none | none | none |

### 4.4 差异总结

| 模型 | 最大差异 | 影响 |
|------|---------|------|
| **MiniCPM5** | max_tokens 4096 vs 256 (16x) | 🔴 语唠 | 
| **MiniCPM5** | presence_penalty=1.0 (如果生效) | 🔴 语言混杂、重复绕圈 |
| **Gemma4 E2B** | temp 0.8 vs 1.0 | 比官方更保守 |
| **Gemma4 E2B** | top_k 30 vs 64 | 候选词范围收窄 |
| **Qwen3.5** | temp 0.8 vs 0.7 | 相近 |
| **Qwen3.5** | top_p 0.9 vs 0.8 | 更开放 |
| **全部** | presence_penalty/频率抑制未注入 | 前端 slider 空转 |

---

## 5. 修复建议

### 5.1 高优先级：按模型家族分采样策略

`gpu.py` 的 `gen_params` 构建处，根据 `model_family` 设置不同的默认采样参数：

```python
# 伪代码
if model_family == "minicpm":
    default_max_tokens = 2048     # 官方 256，给 1B 留点冗余
    default_temperature = 0.7    # No-Think 模式
    default_presence_penalty = 0.0
else:
    default_max_tokens = 4096
    default_temperature = 0.8
```

或在 `config.py` 中定义 `FAMILY_SAMPLING_DEFAULTS` 字典。

### 5.2 中优先级：补全 min_p/presence_penalty/frequency_penalty 注入

前端 slider 已有控件，但 gen_params 未注入。需要：
1. 确认 Flask 路由是否接收这些参数
2. 在 gpu.py gen_params 构建时加入
3. 参数名映射：llama-cpp-python 用 `present_penalty`（不是 presence_penalty）

### 5.3 低优先级：rope_scaling 默认改为 "none"

所有模型原生上下文 ≥128K >> n_ctx=32768，不需要 RoPE 扩展。

---

## 6. 项目配置完整清单

### 本地模型文件

```
D:\Scoop\LLM\
├── Gemma-4-E2B-Q5\
│   ├── Gemma-4-E2B-...-Q5_K_P.gguf      (3.66 GB)
│   └── mmproj-Gemma-4-E2B-...-f16.gguf  (986 MB)
├── Gemma-4-E2B-Q8\
├── MiniCPM5\
│   ├── MiniCPM5-1B-...-Q8_0.gguf
│   └── MiniCPM5-1B-...-F16.gguf
├── Qwen3.5-9B-Q8\
│   ├── Qwen3.5-9B-...-Q8_0.gguf        (9.53 GB)
│   └── mmproj-Qwen3.5-9B-...-BF16.gguf (922 MB)
├── Qwen3.6-35B-Q2\
└── Qwen3.6-35B-Q5\
    ├── Qwen3.6-35B-A3B-...-Q5_K_P.gguf  (28.0 GB)
    └── mmproj-Qwen3.6-35B-...-f16.gguf  (899 MB)
```

### 关键代码位置

| 文件 | 行 | 内容 |
|------|-----|------|
| `config.py` | 30-42 | RoPE 参数定义 (yarn/4.0) |
| `config.py` | 34-42 | FAMILY_ROPE_BASE |
| `config.py` | 45-54 | 默认采样参数 |
| `config.py` | 17 | GPU_DEFAULT_MAX_TOKENS=4096 |
| `backends/gpu.py` | 1412-1422 | gen_params 构建（⚠️ 缺少 min_p/presence_penalty/freq_penalty） |
| `backends/gpu.py` | 1562-1563 | model_family + is_r1_thinking 检测 |
| `backends/gpu.py` | 1592 | create_chat_completion() 调用 |
| `static/js/app.js` | 716-724 | sendLlama() 参数读取 |
| `static/js/app.js` | 994-1002 | sendVendor() 参数读取 |

---

## 7. 数据来源

| 模型 | 来源 | 获取方式 |
|------|------|---------|
| MiniCPM5-1B | OpenBMB/MiniCPM GitHub README-cn.md | web_fetch |
| Gemma 4 E2B | google/gemma-4-E2B-it generation_config.json | hf-mirror.com (✅ 成功，获取到 JSON) |
| Qwen3.5 | Qwen3 readthedocs / GitHub README | web_search + 已有知识 |
| Qwen3.6 | Qwen3.6 GitHub README + 新闻 | web_search（官方采样参数与 Qwen3.5 共享，family=Qwen3.5） |

---

生成时间：2026-08-05
