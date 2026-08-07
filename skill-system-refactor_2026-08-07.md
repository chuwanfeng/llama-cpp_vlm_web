# Skill 系统重构：从模板 → 工具调用（2026-08-07 16:56）

## 问题
用户指出两个关键设计错误：
1. Skill 不应该挂在模板下拉框里 — 应该像工具一样被 LLM 自动调用
2. references 不应该全塞进 prompt — 应该按需加载

## 正确设计

```
用户: "帮我生成一段 H3 视频提示词"
  → LLM 调 skills_list → 发现 h3-prompt-writing
  → LLM 调 skill_view("h3-prompt-writing") → 拿到 SKILL.md 指令 + references 文件列表
  → SKILL.md 说 "基础模式读 base-en.txt，Ref2VA 读 ref-en.txt"
  → LLM 判断需要 base-en.txt → 调 skill_reference("h3-prompt-writing", "base-en.txt")
  → LLM 用 SKILL.md 指令 + base-en.txt 内容 → 生成最终 prompt
```

## 已完成的改动

### tools/builtin_skills.py
- `skill_view()`: 输出新增 **Available Reference Files** 列表，告知 LLM 有哪些 references 可加载
- 新增 `skill_reference()`: 按需加载指定 reference 文件内容（如 base-en.txt、ref-en.txt）
- 注册为工具（schema + registry.register）
- 修复名称匹配：同时按 YAML frontmatter name 和目录名匹配

### static/js/app.js — 回退
- 移除 `_skillList` / `_skillPrompts` / `loadSkillList()` / `getSkillPrompt()`
- `renderTemplateSelect()` 恢复原始（纯模板）
- `send()` 移除 `skill:` 前缀判断
- `loadT()` 移除 `loadSkillList()` 调用

### app.py
- `/api/enhance` 回退 `skill:` 前缀处理

## 验证通过
```
skills_list()        → 11 个技能（含 9 个 H3）
skill_view("h3-prompt-writing") → SKILL.md + Available Reference Files 列表
skill_reference("h3-prompt-writing", "base-en.txt") → 15763 chars ✓
skill_reference("h3-prompt-writing", "ref-en.txt")  → 23578 chars ✓
skill_reference("h3-prompt-writing", "bad")  → 报错 + 列出可用文件 ✓
```

## 遗留清理（待确认）
- `services/skill_loader.py` 中 `get_skill_full_prompt()` / `get_skill_select_list()` 是上一轮遗留
- `app.py` 中 `/api/skills/select` / `/api/skills/<id>/prompt` 端点也是遗留
- 可删可不删，不影响功能
