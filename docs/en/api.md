# Common

The Gateway listens on `https://localhost:8006` by default. All endpoints below are relative to the Gateway.

---

## Service Management APIs

These endpoints manage system status, queuing, configuration, and resources. They are independent of interaction mode.

### Health & Status

#### GET /health

Health check.

**Response**: `{"status": "ok"}`

#### GET /status

Global service status summary.

**Response**:
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

Detailed Worker list.

**Response**:
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

### Queue Management

The Gateway uses a FIFO queue to manage concurrent requests. All interaction modes share a single queue.

#### GET /api/queue

Get a snapshot of the queue status.

**Response**:
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

Query the status of a specific queue ticket.

#### DELETE /api/queue/{ticket_id}

Cancel a queued request.

---

### ETA Configuration

ETA (Estimated Time of Arrival) baselines for each request type, refined at runtime via Exponential Moving Average.

#### GET /api/config/eta

**Response**:
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

Update ETA configuration. Request body uses the same fields as the GET response.

---

### KV Cache

#### GET /api/cache

Query the KV Cache status of all Workers.

---

### Configuration & Presets

#### GET /api/frontend_defaults

Get frontend default configuration values.

**Response**:
```json
{
  "playback_delay_ms": 200,
  "chat_vocoder": "token2wav"
}
```

#### GET /api/presets

Get the list of System Prompt presets.

**Response**:
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

### Session Management

Sessions are automatically recorded for playback and debugging.

#### GET /api/sessions/{session_id}

Get session metadata.

**Response**:
```json
{
  "session_id": "omni_abc123",
  "type": "omni_duplex",
  "created_at": "2026-02-24T10:00:00Z",
  "config": {}
}
```

#### GET /api/sessions/{session_id}/recording

Get session recording timeline data.

#### GET /api/sessions/{session_id}/assets/{relative_path}

Get session asset files (audio/video chunks, etc.).

#### GET /api/sessions/{session_id}/download

Download the entire session as a package.

#### POST /api/sessions/{session_id}/upload-recording

Upload frontend-recorded audio/video files. Size limit: 200 MB.

---

### App Management

Control which interaction modes are available in the frontend.

#### GET /api/apps

Get the list of enabled apps (for frontend use).

#### GET /api/admin/apps

Get the list of all apps including enabled status (for Admin use).

#### PUT /api/admin/apps

Toggle app enabled status.

**Request Body**:
```json
{
  "app_id": "omni",
  "enabled": false
}
```
