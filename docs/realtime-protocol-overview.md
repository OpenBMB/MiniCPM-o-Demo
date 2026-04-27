# MiniCPM-o Realtime API 协议说明

## 第一层：一句话总结

> **一根 WebSocket，客户端不停地发音频和视频，服务端不停地回音频和文字。**
> 用 JSON 文本帧通信，事件名学 OpenAI（`namespace.action` 格式）。

---

## 第二层：三阶段生命周期

整个会话只有三个阶段，按顺序发生：

```
┌─────────┐      ┌─────────┐      ┌─────────┐
│  连接    │ ───→ │  对话    │ ───→ │  结束    │
│  Setup   │      │  Stream  │      │  Close   │
└─────────┘      └─────────┘      └─────────┘
```

### 阶段 1：连接（Setup）

客户端连上 WebSocket，告诉服务端"我要用什么配置"。

```
Client  ──WSS──→  Server
Client  → session.update      "我要中文助手，用这个音色"
Server  → session.created     "好的，准备就绪"
```

完成后进入对话阶段。

### 阶段 2：对话（Stream）

**两条独立的数据流同时工作**，互不阻塞：

```
上行流（Client → Server）:           下行流（Server → Client）:
每秒发一个包，包含：                   模型随时推送：
  🎤 1秒的音频                          🔊 回复的音频片段
  📷 1帧视频截图（可选）                  💬 回复的文字
                                        👂 "我在听"状态信号
```

就像打电话——你说你的，对方说对方的，可以同时进行。

### 阶段 3：结束（Close）

任意一方都可以主动结束。

```
Client  → session.close       "我要挂了"
Server  → session.closed      "好的，再见"
```

---

## 第三层：6 种核心事件

整个协议只有 **6 种需要关心的事件**（3 种发、3 种收）。其余都是辅助性的。

### 客户端发出的（3 种）

| 事件 | 什么时候发 | 发什么 |
|------|-----------|--------|
| `session.update` | 开始时发一次 | 系统提示词、参考音色等配置 |
| `input_audio_buffer.append` | 每秒发一次 | 1秒音频 + 1帧视频（可选） |
| `session.close` | 想结束时发一次 | 无 |

### 服务端回复的（3 种）

| 事件 | 什么意思 | 带什么数据 |
|------|---------|-----------|
| `session.created` | 配置完成，可以开始了 | session_id |
| `response.output_audio.delta` | 模型在**说话** | 音频片段 + 对应文字 |
| `response.listen` | 模型在**听** | 无实质数据 |

**关键认知**：模型只有两种状态——**说**（speak）和**听**（listen）。
服务端通过 `output_audio.delta`（说）和 `listen`（听）告知客户端当前状态。

```
时间线：
  Server:  listen  listen  listen  speak  speak  speak  listen  listen ...
  含义：   在听... 在听... 在听... 在回答............. 又在听了 ...
```

---

## 第四层：一次完整对话的时序

```
时间 ──────────────────────────────────────────────────────────→

                           ┌─────────────────────────────────┐
                           │  Phase 1: 连接 & 排队            │
                           └─────────────────────────────────┘
Client:  WSS Connect ─────→
                           ← Server: session.queued          (你排在第 3 位，约等 45 秒)
                           ← Server: session.queue_update    (第 2 位，约 20 秒)
                           ← Server: session.queue_update    (第 1 位，约 5 秒)
                           ← Server: session.queue_done      (轮到你了！)

  ⚠️ 排队阶段客户端不应发送任何消息，只被动接收排队事件。
  ⚠️ 如果 Worker 立即可用，排队阶段会被跳过（不会收到 session.queued）。

                           ┌─────────────────────────────────┐
                           │  Phase 2: 会话初始化              │
                           └─────────────────────────────────┘
Client:  session.update ─┐  (发送 system prompt、ref audio 等配置)
Server:  session.created ←┘  (模型就绪，返回 session_id)

                           ┌─────────────────────────────────┐
                           │  Phase 3: 全双工对话              │
                           └─────────────────────────────────┘
Client:  append(audio₁) ──→
Client:  append(audio₂) ──→
Client:  append(audio₃) ──→     ← Server: listen   (模型在听你说话)
Client:  append(audio₄) ──→
Client:  append(audio₅) ──→     ← Server: listen   (还在听)
Client:  append(audio₆) ──→     ← Server: output_audio.delta(audio="...", text="你好")
Client:  append(audio₇) ──→     ← Server: output_audio.delta(audio="...", text="，")
Client:  append(audio₈) ──→     ← Server: output_audio.delta(audio="...", text="有什么")
Client:  append(audio₉) ──→     ← Server: output_audio.delta(audio="...", text="可以帮你？")
Client:  append(audio₁₀) ─→     ← Server: listen   (说完了，又在听了)
Client:  append(audio₁₁) ─→
...

                           ┌─────────────────────────────────┐
                           │  Phase 4: 关闭                   │
                           └─────────────────────────────────┘
Client:  session.close ──→
                           ← Server: session.closed {reason: "stopped"}
```

注意：客户端**始终**在发 append，不管服务端是在听还是在说。这就是"全双工"。

---

## 第五层：消息格式

### 5.1 session.update（客户端 → 服务端）

```json
{
    "type": "session.update",
    "session": {
        "instructions": "你是一个友好的中文助手",
        "deferred_finalize": true,
        "max_slice_nums": 1
    }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `instructions` | string | 系统提示词 |
| `deferred_finalize` | bool | 延迟 finalize（性能优化，默认 true） |
| `max_slice_nums` | int | 视频最大切片数（1=快，4=看得仔细） |
| `ref_audio` | string? | LLM 参考音频（base64 WAV），用于音色克隆 |
| `tts_ref_audio` | string? | TTS 参考音频（base64 WAV） |

### 5.2 input_audio_buffer.append（客户端 → 服务端）

```json
{
    "type": "input_audio_buffer.append",
    "seq": 42,
    "audio": "base64_pcm_float32_16khz...",
    "video_frames": ["base64_jpeg..."],
    "force_listen": false
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `audio` | string | 1秒音频，16kHz float32 PCM，base64 编码 |
| `seq` | int | 序号（从 0 开始递增，调试用） |
| `video_frames` | string[]? | JPEG 帧列表（通常 1 帧），base64 编码 |
| `force_listen` | bool? | 强制模型进入 listen 状态（打断模型说话） |
| `max_slice_nums` | int? | 覆盖本次 chunk 的视频切片数 |

### 5.3 session.created（服务端 → 客户端）

```json
{
    "type": "session.created",
    "session_id": "rt_abc123",
    "prompt_length": 256
}
```

### 5.4 response.output_audio.delta（服务端 → 客户端）

```json
{
    "type": "response.output_audio.delta",
    "text": "你好",
    "audio": "base64_pcm_float32_24khz...",
    "end_of_turn": false,
    "kv_cache_length": 1024,
    "cost_all_ms": 450
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | string | 本次生成的文字片段 |
| `audio` | string | 24kHz float32 PCM 音频，base64 |
| `end_of_turn` | bool | 本轮生成是否结束 |
| `kv_cache_length` | int | 当前 KV 缓存长度（监控用） |
| `cost_all_ms` | int | 推理耗时（毫秒） |
| `vision_slices` | int? | 处理的视觉切片数 |
| `vision_tokens` | int? | 视觉 token 数 |

### 5.5 response.listen（服务端 → 客户端）

```json
{
    "type": "response.listen",
    "kv_cache_length": 1024,
    "cost_all_ms": 120
}
```

表示模型当前在听。客户端收到后应停止播放队列中的音频（如果有）。

### 5.6 session.close / session.closed

```json
// 客户端发
{"type": "session.close", "reason": "user_stop"}

// 服务端回
{"type": "session.closed", "reason": "stopped"}
```

---

## 第六层：辅助事件（可选，不影响核心流程）

| 事件 | 方向 | 用途 |
|------|------|------|
| `session.go_away` | S→C | 服务端即将断连警告 |
| `session.queued` | S→C | 排队通知（排队期间） |
| `session.queue_update` | S→C | 排队位置更新 |
| `session.queue_done` | S→C | 排队结束，Worker 已分配 |
| `error` | S→C | 错误 |

### 不纳入协议的功能

以下功能**不属于协议层**，由实现层自行处理：

| 功能 | 为什么不需要协议事件 |
|------|---------------------|
| **暂停/恢复** | 客户端停止发 `append` 即等效暂停，模型会持续 `listen` 等待。无需 `session.pause`/`resume` 事件 |
| **取消生成** | 全双工模式下，模型同时在听和说。打断说话用 `force_listen` 字段（已在 `append` 中），不需要独立的 `response.cancel` 事件 |
| **回复结束标记** | `output_audio.delta` 中的 `end_of_turn=true` 已标记一轮回复结束，不需要额外的 `response.done` 事件 |

---

## 第七层：架构——消息在系统中如何流动

```
                        OpenAI Realtime 协议             旧协议（内部）
浏览器 ──── WSS ────→ Gateway (/v1/realtime) ──── WS ────→ Worker (/ws/duplex)
            ←                   翻译层                          ←
```

**Gateway 是一个协议翻译网关**：

| 浏览器发的（新协议） | Gateway 翻译成（旧协议） |
|---------------------|------------------------|
| `session.update` | `prepare` |
| `input_audio_buffer.append` | `audio_chunk` |
| `session.close` | `stop` |

| Worker 回的（旧协议） | Gateway 翻译成（新协议） |
|---------------------|------------------------|
| `prepared` | `session.created` |
| `result { is_listen: true }` | `response.listen` |
| `result { is_listen: false }` | `response.output_audio.delta` |
| `stopped` | `session.closed` |

这意味着：
- **Worker 不需要改动**，继续说旧协议
- **前端页面用新协议**，通过 `RealtimeSession` JS 类
- **翻译层在 Gateway 的 Python 代码里**，一个函数 ~100 行

---

## 第八层：和 OpenAI Realtime API 的关系

我们的协议是 **OpenAI Realtime API 的子集 + MiniCPM-o 的扩展**。

### 相同的部分（兼容 OpenAI）

- 事件命名风格：`namespace.action`
- 核心事件名：`session.update`、`input_audio_buffer.append`、`response.output_audio.delta`
- 连接方式：WebSocket + JSON
- 音频格式：base64 PCM in JSON

### 我们独有的扩展

| 扩展 | OpenAI 有吗 | 为什么需要 |
|------|------------|-----------|
| `video_frames` 字段 | 没有（OpenAI 只支持静态图片） | MiniCPM-o 支持连续视频流 |
| `force_listen` 字段 | 没有（OpenAI 用 VAD 打断） | MiniCPM-o 的全双工靠模型自行判断 listen/speak |
| `response.listen` 事件 | 没有 | MiniCPM-o 的 listen 是一等事件，不只是"没在说话" |
| `ref_audio` / `tts_ref_audio` | 没有 | MiniCPM-o 支持音色克隆（双参考音频） |
| `kv_cache_length` 字段 | 没有 | MiniCPM-o 有 KV cache sliding window |

### 我们没有的（OpenAI 有）

| OpenAI 特性 | 我们为什么没做 |
|-------------|--------------|
| 服务端 VAD | MiniCPM-o 的 listen/speak 由模型自主判断，不需要独立 VAD |
| `conversation.item.create` | 暂不需要手动管理对话历史 |
| `response.create` 手动触发 | 我们的模型每个 chunk 都会决策，不需要手动触发 |
| Function calling | 暂未接入 |
| WebRTC 接入 | 暂时只走 WebSocket |

---

## 快速参考卡片

```
┌──────────────────────────────────────────────────────────┐
│                  MiniCPM-o Realtime API                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  端点:  wss://host/v1/realtime?session_id=xxx            │
│  格式:  JSON 文本帧                                       │
│  音频:  上行 16kHz float32 PCM / 下行 24kHz float32 PCM   │
│  视频:  JPEG base64, ~1 FPS                              │
│                                                          │
│  核心事件:                                                │
│  ┌──────────┐                    ┌──────────────┐         │
│  │ Client   │  session.update →  │ Server       │         │
│  │          │  append →          │              │         │
│  │          │  close →           │              │         │
│  │          │                    │              │         │
│  │          │  ← session.created │              │         │
│  │          │  ← audio.delta     │              │         │
│  │          │  ← listen          │              │         │
│  │          │  ← closed          │              │         │
│  └──────────┘                    └──────────────┘         │
│                                                          │
│  模型状态:  listen ←→ speak (由模型自主切换)               │
│  打断方式:  客户端发 force_listen=true                     │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 第九层：合法调用边界与错误码

### 设计原则

1. **只约定合法的调用顺序和格式**。不在此范围内的调用，服务端直接拒绝，不做猜测和容错。
2. **错误码不是兜底**。错误码的目的是让客户端开发者在联调阶段快速定位问题，不是运行时的常规通道。
3. **算法流程不为客户端的非法调用负责**。收到非法消息，返回 error 后服务端状态不变——不 crash、不污染模型状态、不中断正在进行的推理。

### 9.1 合法的调用时序（状态机）

```
          connect
             │
             ▼
    ┌─── CONNECTED ───┐
    │                  │    只允许发: session.update
    │                  │    其他一律 → invalid_event
    └────────┬─────────┘
             │ 收到 session.created
             ▼
    ┌──── ACTIVE ─────┐
    │                  │    允许发: append / close
    │                  │    append 中可携带 force_listen=true
    └────────┬─────────┘
             │ close 或 异常
             ▼
         CLOSED
```

### 9.2 非法调用一览

| 客户端做了什么 | 为什么非法 | 错误码 |
|--------------|-----------|--------|
| 在 `session.created` 之前发 `append` | 还没配置完 | `not_ready` |
| 发了不认识的 `type` | 协议里没这个事件 | `unknown_event` |
| `append` 缺少 `audio` 字段 | 必填字段缺失 | `missing_field` |
| `audio` 不是合法的 base64 | 解码失败 | `invalid_payload` |
| `video_frames` 中的 JPEG 解码失败 | 图片损坏 | `invalid_payload` |
| 发送速度过快（队列溢出） | 超出处理能力 | **静默丢弃**（不返回 error） |
| JSON 解析失败（非法 JSON） | 二进制或损坏的帧 | 直接关闭 WebSocket (1003) |

### 9.3 服务端错误一览

| 发生了什么 | 原因 | 错误码 | 后果 |
|-----------|------|--------|------|
| Worker 模型没加载完 | 服务启动中 | `service_unavailable` | 关闭 WS (1013) |
| 排队已满 | 并发超限 | `queue_full` | 关闭 WS (1013) |
| 没有空闲 Worker | 全忙 | `worker_busy` | 关闭 WS (1013) |
| Worker 连接失败（重试耗尽） | Worker 进程崩溃 | `worker_connect_failed` | 关闭 WS (1013) |
| 推理过程出错（GPU/模型异常） | 模型内部错误 | `inference_error` | 返回 error，会话继续（模型会自动回到 listen） |
| 暂停超时 | 暂停太久未恢复 | **不是 error**，走 `session.closed` | reason=`pause_timeout`，关闭 WS |

### 9.4 错误消息格式

所有错误走统一格式：

```json
{
    "type": "error",
    "error": {
        "code": "not_ready",
        "message": "Session not prepared yet. Send session.update first.",
        "type": "client_error"
    }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | string | 机器可读的错误码（见上表） |
| `message` | string | 人可读的描述（联调用，不要依赖文案） |
| `type` | enum | `"client_error"` 或 `"server_error"` |

### 9.5 错误码速查表

| code | type | 含义 | WS 是否关闭 |
|------|------|------|-----------|
| `not_ready` | client_error | 会话未建立就发数据 | 否 |
| `unknown_event` | client_error | 不认识的事件 type | 否 |
| `missing_field` | client_error | 必填字段缺失 | 否 |
| `invalid_payload` | client_error | 字段值非法（base64/JPEG 解码失败等） | 否 |
| `service_unavailable` | server_error | 服务未就绪 | 是 (1013) |
| `queue_full` | server_error | 排队已满 | 是 (1013) |
| `worker_busy` | server_error | 没有空闲 Worker | 是 (1013) |
| `worker_connect_failed` | server_error | Worker 连接失败 | 是 (1013) |
| `inference_error` | server_error | 推理过程出错 | 否（可恢复） |

### 9.6 关于"静默丢弃"

客户端发送音频 chunk 过快时（超过模型处理速度），服务端会**直接丢弃旧的 chunk**，不返回 error。

理由：
- 这不是客户端 bug——浏览器麦克风采集速率是固定的（每秒 1 chunk）
- 只在模型推理偶尔变慢时发生
- 返回 error 会制造大量噪音，且客户端无法也不应该"降速"
- 服务端保证：**总是处理最新的 chunk**，跳过积压的旧 chunk

### 9.7 关于"非法 JSON 直接关闭"

如果客户端发来的 WebSocket 帧无法 `JSON.parse`，服务端关闭连接，WebSocket close code = **1003**（Unsupported Data）。

不返回 error JSON——因为如果对方连 JSON 都发不对，它大概率也解析不了我们返回的 error JSON。
