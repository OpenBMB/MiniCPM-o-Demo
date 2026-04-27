# MiniCPM-o 音频双工 API 协议

> 本文档定义**音频双工**（Audio Full-Duplex）模式的 WebSocket 协议。
> 视频双工模式见 [video-duplex-protocol.md](video-duplex-protocol.md)。

---

## 一句话总结

> **一根 WebSocket，客户端每秒发 1 秒音频，服务端实时回音频和文字。**

---

## 模式约束

| 项目 | 值 |
|------|-----|
| 端点 | `wss://host/v1/realtime?mode=audio`（`session_id` 可选，见下） |
| 帧格式 | JSON 文本帧 |
| 上行音频 | **16 kHz**，单声道，float32 PCM，base64 编码 |
| 下行音频 | **24 kHz**，单声道，float32 PCM，base64 编码 |
| 上行视频 | 不建议发送。携带 `video_frames` 时行为未定义 |
| 会话总时长上限 | **600 秒（10 分钟）**，含所有等待和空闲时间 |
| 有效对话时间 | 待补充 |
| 上下文窗口 | **8192 tokens**，不可配置。满时服务端主动关闭会话 |

---

## 生命周期

```
┌─────────┐      ┌─────────┐      ┌─────────┐
│  连接    │ ───→ │  对话    │ ───→ │  结束    │
│  Setup   │      │  Stream  │      │  Close   │
└─────────┘      └─────────┘      └─────────┘
```

### Setup

连接 URL：`wss://host/v1/realtime?mode=audio[&session_id=xxx]`

- `session_id` **可选**。客户端可传入自定义 ID 用于日志追踪；省略时服务端自动生成（格式 `rt_{timestamp_ms}`）。
- `session.created` 返回的 `session_id` 即最终生效值（传了用客户端的，没传用服务端的）。

```
Client  ──WSS──→  Server
                  ← session.queued / queue_update / queue_done  (排队，可能跳过)
Client  → session.update      "我要中文助手，用这个音色"
Server  → session.created     "好的，准备就绪，session_id=xxx"
```

### Stream

两条独立的数据流同时工作，互不阻塞：

```
上行流（Client → Server）:            下行流（Server → Client）:
每秒发一个包，包含：                    模型随时推送：
  - 1 秒的 16kHz 音频                   - 回复的 24kHz 音频片段
  - （无视频）                           - 回复的文字
                                        - "我在听"状态信号
```

### Close

三种关闭原因：

| reason | 触发方 | 含义 |
|--------|-------|------|
| `user_stop` | 客户端 | 用户主动结束 |
| `timeout` | 服务端 | 会话总时长达到 600 秒 |
| `context_full` | 服务端 | 上下文窗口 8192 tokens 已满 |

---

## 事件总览

### 客户端 → 服务端（3 种）

| 事件 | 什么时候发 | 发什么 |
|------|-----------|--------|
| `session.update` | 开始时发一次 | 系统提示词、参考音色 |
| `input_audio_buffer.append` | 每秒发一次 | 1 秒音频（**不含视频**） |
| `session.close` | 想结束时发一次 | 关闭原因 |

### 服务端 → 客户端（3 种核心 + 辅助）

| 事件 | 什么意思 | 带什么数据 |
|------|---------|-----------|
| `session.created` | 配置完成 | session_id |
| `response.output_audio.delta` | 模型在**说话** | 音频 + 文字 + end_of_turn |
| `response.listen` | 模型在**听** | KV cache 等监控数据 |

辅助事件：`session.queued`、`session.queue_update`、`session.queue_done`、`session.go_away`、`session.closed`、`error`

---

## 消息格式

### session.update（客户端 → 服务端）

```json
{
    "type": "session.update",
    "session": {
        "instructions": "你是一个友好的中文助手",
        "ref_audio": "<base64 WAV, 16kHz>",
        "tts_ref_audio": "<base64 WAV, 16kHz>"
    }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `instructions` | string | 是 | 系统提示词 |
| `ref_audio` | string | 否 | LLM 参考音频（base64 WAV, 16kHz），用于语义风格克隆 |
| `tts_ref_audio` | string | 否 | TTS 参考音频（base64 WAV, 16kHz），用于声学特征克隆。未提供时 fallback 到 `ref_audio` |

> 音频双工模式没有 `max_slice_nums` 字段（无视频输入）。

### input_audio_buffer.append（客户端 → 服务端）

```json
{
    "type": "input_audio_buffer.append",
    "audio": "<base64, 16000 samples = 1s, float32 PCM>",
    "force_listen": false
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `audio` | string | **是** | 16 kHz 单声道 float32 PCM，1 秒 = 16000 samples = 64000 bytes，base64 编码。最小 4000 samples (250ms) |
| `force_listen` | bool | 否 | 强制模型进入 listen 状态（打断模型说话），默认 false |

> 音频双工模式下不建议携带 `video_frames` 字段，携带时行为未定义。

### session.close（客户端 → 服务端）

```json
{
    "type": "session.close",
    "reason": "user_stop"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `reason` | string | 否 | 关闭原因，建议填 `"user_stop"` |

### session.created（服务端 → 客户端）

```json
{
    "type": "session.created",
    "session_id": "rt_1714200000000",
    "prompt_length": 256
}
```

### response.output_audio.delta（服务端 → 客户端）

```json
{
    "type": "response.output_audio.delta",
    "text": "今天天气真好",
    "audio": "<base64, 24000 samples = 1s, float32 PCM>",
    "end_of_turn": false,
    "kv_cache_length": 1024
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | 是 | 本次生成的文字片段 |
| `audio` | string | 是 | 24 kHz 单声道 float32 PCM，base64 编码 |
| `end_of_turn` | bool | 是 | 本轮生成是否结束（turn EOS）。true 时表示模型说完了这句话，即将切回 listen |
| `kv_cache_length` | int | 是 | 当前 KV 缓存已使用 token 数（上限 8192） |

#### 关于文字与音频的对齐

由于模型架构特性，**文字生成领先于音频合成**。同一个 `output_audio.delta` 中的 `text` 和 `audio` 并非严格同步——文字内容通常领先音频几百毫秒。

示例（两个连续的 delta）：

```json
// delta 1
{ "text": "今天天气真好", "audio": "<音频: '看来今天天气'>" }

// delta 2
{ "text": "，比较适合散步", "audio": "<音频: '真好，比较适合'>" }
```

客户端应以**音频播放进度**为准呈现体验，文字可作为提前预览或字幕使用。

#### 关于输出音频长度

- **非首个、非末尾的 delta**：音频长度固定为 **1 秒**（24000 samples）
- **首个 delta**：音频可能短于 1 秒（模型启动延迟）
- **末尾 delta**（`end_of_turn=true`）：音频可能短于 1 秒（句尾截断）

### response.listen（服务端 → 客户端）

```json
{
    "type": "response.listen",
    "kv_cache_length": 1024
}
```

模型当前在听。客户端收到后应停止播放队列中的音频（如果有残留）。

### session.closed（服务端 → 客户端）

```json
{
    "type": "session.closed",
    "reason": "timeout"
}
```

| reason | 含义 |
|--------|------|
| `stopped` | 用户主动 `session.close` 后的确认 |
| `timeout` | 会话总时长达到上限（音频模式 600 秒） |
| `context_full` | 上下文窗口 8192 tokens 已满 |
| `server_shutdown` | 服务端正在关闭 |
| `error` | 因不可恢复错误终止 |

---

## 完整时序

```
时间 ──────────────────────────────────────────────────────────→

                           ┌─────────────────────────────────┐
                           │  Phase 1: 连接 & 排队            │
                           └─────────────────────────────────┘
Client:  WSS Connect ─────→
                           ← Server: session.queued          (你排在第 3 位，约等 45 秒)
                           ← Server: session.queue_update    (第 2 位，约 20 秒)
                           ← Server: session.queue_done      (轮到你了！)

  ⚠️ 排队阶段客户端不应发送任何消息，只被动接收排队事件。
  ⚠️ 如果 Worker 立即可用，排队阶段会被跳过。

                           ┌─────────────────────────────────┐
                           │  Phase 2: 会话初始化              │
                           └─────────────────────────────────┘
Client:  session.update ─┐  (system prompt、ref audio)
Server:  session.created ←┘  (模型就绪，返回 session_id)

                           ┌─────────────────────────────────┐
                           │  Phase 3: 全双工对话              │
                           └─────────────────────────────────┘
Client:  append(audio₁) ──→
Client:  append(audio₂) ──→
Client:  append(audio₃) ──→     ← Server: listen   (模型在听)
Client:  append(audio₄) ──→
Client:  append(audio₅) ──→     ← Server: listen   (还在听)
Client:  append(audio₆) ──→     ← Server: output_audio.delta("你好", audio, end_of_turn=false)
Client:  append(audio₇) ──→     ← Server: output_audio.delta("，", audio, end_of_turn=false)
Client:  append(audio₈) ──→     ← Server: output_audio.delta("可以帮你？", audio, end_of_turn=true)
Client:  append(audio₉) ──→     ← Server: listen   (说完了，又在听了)
...

                           ┌─────────────────────────────────┐
                           │  Phase 4: 关闭                   │
                           └─────────────────────────────────┘
                           任一条件触发：
                           - 用户发 session.close
                           - 总时长 ≥ 600s → session.closed {reason: "timeout"}
                           - KV cache ≥ 8192 → session.closed {reason: "context_full"}

Client:  session.close ──→
                           ← Server: session.closed {reason: "stopped"}
```

注意：客户端**始终**在发 append（仅音频），不管服务端是在听还是在说。这就是"全双工"。

---

## 不纳入协议的功能

| 功能 | 为什么不需要协议事件 |
|------|---------------------|
| **暂停/恢复** | 客户端停止发 `append` 即等效暂停，模型会持续 `listen` 等待 |
| **取消生成** | 全双工模式下用 `force_listen` 字段打断，不需要独立的 cancel 事件 |
| **回复结束标记** | `end_of_turn=true` 已标记，不需要额外的 `response.done` 事件 |
| **上下文窗口大小配置** | 固定 8192 tokens，不可调 |

---

## 状态机

```
          connect
             │
             ▼
    ┌─── CONNECTED ───┐
    │                  │    只允许发: session.update
    │                  │    其他一律 → error (invalid_event)
    └────────┬─────────┘
             │ 收到 session.created
             ▼
    ┌──── ACTIVE ─────┐
    │                  │    允许发: append / close
    │                  │    append 中可携带 force_listen=true
    └────────┬─────────┘
             │ close / timeout / context_full / 异常
             ▼
         CLOSED
```

---

## 错误码

### 客户端错误

| code | 含义 | WS 关闭 |
|------|------|---------|
| `not_ready` | 会话未建立就发数据 | 否 |
| `unknown_event` | 不认识的事件 type | 否 |
| `missing_field` | 必填字段缺失 | 否 |
| `invalid_payload` | 字段值非法（base64 解码失败等） | 否 |

### 服务端错误

| code | 含义 | WS 关闭 |
|------|------|---------|
| `service_unavailable` | 服务未就绪 | 是 (1013) |
| `queue_full` | 排队已满 | 是 (1013) |
| `worker_busy` | 没有空闲 Worker | 是 (1013) |
| `worker_connect_failed` | Worker 连接失败 | 是 (1013) |
| `inference_error` | 推理出错 | 否（可恢复） |

### 错误消息格式

```json
{
    "type": "error",
    "error": {
        "code": "invalid_payload",
        "message": "audio base64 decode failed",
        "type": "client_error"
    }
}
```

### 静默丢弃

客户端发送 chunk 过快时，服务端**丢弃旧 chunk**，不返回 error。保证总是处理最新的 chunk。

### 非法 JSON

WebSocket 帧无法 `JSON.parse` → 直接关闭连接，close code = **1003**。
