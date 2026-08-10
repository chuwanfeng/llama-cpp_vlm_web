# MiniMax H3 Skill 集成进展 — 2026-08-09

## 本阶段完成

### Skills 系统改造
1. **skills_list 分类重构** — 按「提示词生成/视频风格·动画类/视频风格·营销自媒体类/其他」分类，各类附场景匹配规则
   - 自然风景/真实风格 → 选 h3-prompt-writing
   - 3D卡通动画 → 选 3d-animation-short-generator
   - Z-Image 图片 → 选 z_image_turbo_prompt_master
   - 通用视频方法论 → 选 ai-video-prompting

2. **SDK 多轮全链路打通**（_test_multi_v2.py）:
   - skills_list() → skill_view("h3-prompt-writing") → 生成 H3 提示词 ✅
   - 场景匹配规则生效：樱花视频从选 3d-animation 改为选 h3-prompt-writing ✅

### Vendor API 工具循环修复
1. `_openai_stream` 修复 Zhipu 重复 finish_reason=tool_calls 导致工具数据丢失
   - 新增 `_tc_emitted` 标志，第二次空 finish_reason 跳过
2. 移除 Zhipu 不兼容的 `stream_options={"include_usage": True}`
3. `api_vendors_chat` 已有 MAX_TOOL_ROUNDS=10 多轮循环

### Zhipu 模型工具调用行为发现
- `glm-4-flash` 非流式 tool_calls ✅ 稳定，流式 ⚠️ 不稳定（有时不调）
- `glm-4.7-flash` 非流式 ✅，流式 ❌ — 把工具调用以 XML `<tool_call>name</tool_call>` 塞入 content
- SDK 直接调用非流式 ↔ 流式差异是模型层面的问题

## 待优化
- 流式工具调用不稳定：可对 vendor 路径降级为非流式（stream=False），保证工具调用可靠性
- `glm-4.7-flash` XML 解析支持
- 前端设置页 skill 开关需接入推理流
