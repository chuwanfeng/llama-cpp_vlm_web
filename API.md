# llama-cpp_vlm_web API 文档

## 概述

本文档描述 llama-cpp_vlm_web 的所有 RESTful API 端点。

## 基础信息

- **Base URL**: `http://localhost:5000`
- **Content-Type**: `application/json`

---

## 技能管理 API

### 列出所有技能

```
GET /api/skills
```

**响应**:
```json
{
  "skills": [
    {
      "name": "skill-name",
      "description": "技能描述",
      "priority": 5,
      "tools": ["read_file", "run_terminal"]
    }
  ],
  "count": 4
}
```

### 创建技能

```
POST /api/skills
```

**请求体**:
```json
{
  "name": "new-skill",
  "description": "技能描述",
  "content": "# 技能内容\n\n详细说明...",
  "priority": 5,
  "tools": ["read_file"]
}
```

**响应**:
```json
{
  "status": "created",
  "name": "new-skill"
}
```

### 获取技能详情

```
GET /api/skills/<skill_id>
```

**响应**:
```json
{
  "name": "skill-name",
  "description": "描述",
  "priority": 5,
  "tools": ["read_file"],
  "content": "# 内容"
}
```

### 删除技能

```
DELETE /api/skills/<skill_id>
```

**响应**:
```json
{
  "status": "deleted",
  "name": "skill-name"
}
```

---

## 进程管理 API

### 列出所有进程

```
GET /api/processes
```

**响应**:
```json
{
  "processes": [
    {
      "session_id": "abc123",
      "pid": 12345,
      "command": "python script.py",
      "status": "running"
    }
  ],
  "count": 1
}
```

### 终止进程

```
POST /api/processes/<session_id>/kill
```

**响应**:
```json
{
  "status": "killed",
  "session_id": "abc123"
}
```

### 获取进程日志

```
GET /api/processes/<session_id>/log
```

**响应**:
```json
{
  "log": "进程输出内容...",
  "session_id": "abc123"
}
```

---

## 审批流 API

### 获取审批状态

```
GET /api/approval/status
```

**响应**:
```json
{
  "session_key": "default",
  "yolo_enabled": false,
  "pending": []
}
```

### 获取待审批请求

```
GET /api/approval/pending
```

**响应**:
```json
{
  "pending": [
    {
      "request_key": "req_123",
      "command_preview": "rm -rf /tmp",
      "session_key": "default",
      "timestamp": "2026-05-31T21:00:00"
    }
  ],
  "count": 1
}
```

### 批准请求

```
POST /api/approval/approve
```

**请求体**:
```json
{
  "request_key": "req_123"
}
```

### 拒绝请求

```
POST /api/approval/deny
```

**请求体**:
```json
{
  "request_key": "req_123"
}
```

### 切换 YOLO 模式

```
POST /api/approval/yolo
```

**请求体**:
```json
{
  "enabled": true
}
```

---

## 聊天 API

### 流式聊天

```
POST /api/agent/chat/stream
```

**请求体**:
```json
{
  "message": "你好",
  "session_id": "optional-session-id",
  "vendor_id": "deepseek",
  "model": "deepseek-chat",
  "tools_enabled": true
}
```

**响应**: SSE (Server-Sent Events)

事件类型:
- `token` - 流式 token
- `tool_call` - 工具调用
- `tool_result` - 工具结果
- `review` - 自我进化 review
- `done` - 完成

---

## 定时任务 API

### 列出所有任务

```
GET /api/cron/jobs
```

**响应**:
```json
{
  "jobs": [
    {
      "id": "job_123",
      "name": "每日报告",
      "cron_expr": "0 9 * * *",
      "handler_type": "agent_chat",
      "args": {"prompt": "生成昨日总结"},
      "enabled": true,
      "created_at": "2026-06-01T12:00:00"
    }
  ],
  "count": 1
}
```

### 创建任务

```
POST /api/cron/jobs
```

**请求体**:
```json
{
  "name": "定时任务名称",
  "cron_expr": "*/5 * * * *",
  "handler_type": "agent_chat",
  "args": {"prompt": "检查邮件"},
  "enabled": true
}
```

### 更新任务

```
PUT /api/cron/jobs/<job_id>
```

**请求体**:
```json
{
  "enabled": false
}
```

### 删除任务

```
DELETE /api/cron/jobs/<job_id>
```

---

## 插件 API

### 列出所有插件

```
GET /api/plugins
```

**响应**:
```json
{
  "plugins": [
    {
      "id": "local_memory",
      "name": "本地记忆",
      "version": "1.0.0",
      "description": "SQLite FTS5 本地记忆存储",
      "enabled": true
    }
  ],
  "count": 1
}
```

### 发现插件

```
POST /api/plugins/discover
```

**响应**:
```json
{
  "discovered": 2,
  "plugins": ["plugin_a", "plugin_b"]
}
```

### 切换插件状态

```
POST /api/plugins/<plugin_id>/toggle
```

**请求体**:
```json
{
  "enabled": true
}
```

---

## 多 Agent 协作 API

### 执行团队任务

```
POST /api/agents/team
```

**请求体**:
```json
{
  "task": "分析项目代码质量",
  "agents": [
    {"role": "reviewer", "prompt": "检查代码规范"},
    {"role": "tester", "prompt": "检查测试覆盖"}
  ],
  "context": "可选上下文"
}
```

**响应**:
```json
{
  "results": [
    {"role": "reviewer", "output": "代码规范检查结果..."},
    {"role": "tester", "output": "测试覆盖检查结果..."}
  ],
  "status": "completed"
}
```

---

## 错误响应

所有 API 在出错时返回:

```json
{
  "error": "错误描述",
  "status": "error"
}
```

HTTP 状态码:
- `400` - 请求参数错误
- `404` - 资源不存在
- `500` - 服务器内部错误
