# 通用

Gateway 默认监听 `https://localhost:8006`。以下所有端点均相对于 Gateway。

---

## 服务管理 API

以下端点用于管理系统状态、队列、配置和资源，与交互模式无关。

### 健康与状态

#### GET /health

健康检查。

**响应**：`{"status": "ok"}`

#### GET /status

服务全局状态摘要。

**响应**：
```json
{
  "total_workers": 4,
  "idle_workers": 2,
  "busy_workers": 2,
  "queue_length": 3,
  "workers": [...]
}
```

#### GET /workers

详细 Worker 列表。

**响应**：
```json
{
  "workers": [
    {
      "url": "http://localhost:22400",
      "index": 0,
      "status": "idle",
      "current_task": null,
      "current_session_id": null,
      "cached_hash": "abc123",
      "busy_since": null
    }
  ]
}
```

---

### 队列管理

Gateway 使用 FIFO 队列管理并发请求。所有交互模式共享同一个队列。

#### GET /api/queue

获取队列状态快照。

**响应**：
```json
{
  "queue_length": 3,
  "entries": [
    {
      "ticket_id": "tk_001",
      "position": 1,
      "task_type": "half_duplex",
      "eta_seconds": 15.0
    }
  ],
  "running": [
    {
      "worker_url": "http://localhost:22400",
      "task_type": "half_duplex",
      "session_id": "stream_xyz",
      "started_at": "2026-02-24T10:30:00Z",
      "elapsed_s": 5.2
    }
  ]
}
```

#### GET /api/queue/{ticket_id}

查询指定排队凭证的状态。

#### DELETE /api/queue/{ticket_id}

取消排队请求。

---

### ETA 配置

ETA（预计等待时间）为每种请求类型的基准值，运行时通过指数移动平均（EMA）动态修正。

#### GET /api/config/eta

**响应**：
```json
{
  "eta_chat_s": 15.0,
  "eta_half_duplex_s": 180.0,
  "eta_audio_duplex_s": 120.0,
  "eta_omni_duplex_s": 90.0,
  "eta_ema_alpha": 0.3,
  "eta_ema_min_samples": 3
}
```

#### PUT /api/config/eta

更新 ETA 配置。请求体使用与 GET 响应相同的字段。

---

### KV Cache

#### GET /api/cache

查询所有 Worker 的 KV Cache 状态。

---

### 配置与预设

#### GET /api/frontend_defaults

获取前端默认配置值。

**响应**：
```json
{
  "playback_delay_ms": 200,
  "chat_vocoder": "token2wav"
}
```

#### GET /api/presets

获取 System Prompt 预设列表。

**响应**：
```json
[
  {
    "id": "default_en",
    "name": "English Assistant",
    "system_prompt": "You are a helpful assistant."
  }
]
```

---

### 会话管理

会话自动录制，用于回放和调试。

#### GET /api/sessions/{session_id}

获取会话元数据。

**响应**：
```json
{
  "session_id": "omni_abc123",
  "type": "omni_duplex",
  "created_at": "2026-02-24T10:00:00Z",
  "config": {}
}
```

#### GET /api/sessions/{session_id}/recording

获取会话录制时间线数据。

#### GET /api/sessions/{session_id}/assets/{relative_path}

获取会话资源文件（音频/视频 chunk 等）。

#### GET /api/sessions/{session_id}/download

打包下载整个会话。

#### POST /api/sessions/{session_id}/upload-recording

上传前端录制的音频/视频文件。大小限制 200 MB。

---

### 应用管理

控制前端中可用的交互模式。

#### GET /api/apps

获取已启用的应用列表（前端用）。

#### GET /api/admin/apps

获取所有应用列表，包含启用状态（Admin 用）。

#### PUT /api/admin/apps

切换应用启用状态。

**请求体**：
```json
{
  "app_id": "omni",
  "enabled": false
}
```
