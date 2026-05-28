# API V2 And Backend Protocol Memo

本文是新的外部 API 与 backend protocol 对齐备忘，不是最终 JSON Schema。

目标是把当前 `/ws/chat` 和 `/ws/duplex/{session_id}` 的有效语义保留下来，同时去掉 legacy demo 命名和半 OpenAI-style 翻译层里的歧义字段。新的外部 API 保留 chat WebSocket 和 realtime WebSocket 两条主线；backend protocol 仍然可以统一成 `init/push/pull/close`。

## Scope

外部 API 分两类：

```text
WebSocket /v1/chat
WebSocket /v1/realtime
```

其中：

- `/v1/chat` 是 turn-based chat WebSocket，覆盖 streaming 和 non-streaming 输出。
- `/v1/realtime` 是 full-duplex realtime WebSocket，覆盖连续音频/视频输入和实时输出。

本文同时标注每个语义属于：

- API layer：前端、外部调用方、gateway 看到的协议。
- Backend protocol：runtime/scheduler 到 backend server 的内部协议。

外部 API 覆盖：

- session 排队与分配。
- session 初始化。
- turn-based 输入和 full-duplex 连续观察。
- 模型输出，包括 listen、文本、音频和 chat done。
- session 关闭。

backend protocol 不覆盖排队；排队只属于 API/gateway layer。

当前不定义 pause/resume 控制指令。调用方暂停输入时，只需要停止发送 input；runtime/backend 没有收到新观察时自然等待。

## API Endpoints

```text
WebSocket /v1/chat
WebSocket /v1/realtime
```

连接参数可以继续表达 mode、client identity、页面来源等路由信息，但这些不属于本文的核心消息语义。

`/ws/chat`、`/ws/duplex/{session_id}` 不再作为新的外部 API 主线，只作为 legacy compatibility 或迁移参考。

## Primitive Mapping

backend protocol 仍然可以理解成四个原语：

```text
init(params)
push(input)
pull() -> events
close()
```

API layer 到 backend protocol 的概念映射：

| API layer | Backend protocol |
|---|---|
| `session.init` | `init(params)` |
| `input.append` | `push(input)` |
| server event stream | `pull() -> events` |
| `session.close` | `close()` unary |

backend protocol 没有 `session.queued` / `session.queue_update` / `session.queue_done`。

## Client Events

### session.init

Layer: API and backend protocol.

初始化一个 session。

在 `/v1/chat` 中，它初始化 turn-based chat session。

在 `/v1/realtime` 中，它初始化 full-duplex realtime session。

它替代：

- 当前 `/ws/chat` 中隐含在首个 request 里的 chat 参数。
- 当前 `/ws/duplex` 的 `prepare`。
- 旧 `/v1/realtime` 的 `session.update` 初始化用法。

语义：

- 提供系统提示、参考音频、生成配置、视觉配置等 session 级参数。
- init 成功后 server 发送 `session.created`。
- init 完成前，client 不应发送 `input.append`。
- API/gateway 可以根据 endpoint 和 init 参数选择 worker/backend。

字段结构先保持宽松，后续 strong-types 分支再收敛。

### input.append

Layer: API and backend protocol.

提交模型输入。

在 `/v1/chat` 中：

- `input.append` 表示一次完整 turn request。
- streaming 和 non-streaming 都走同一个输入事件。
- 是否使用底层 streaming view 或 non-streaming view 是 runtime/backend adapter 的职责，不由 API runtime 聚合 stream delta 伪造。

在 `/v1/realtime` 中：

- `input.append` 表示一段新的模型观察。
- 音频输入是必选或主要输入。
- 视频/图像帧是可选附加观察。
- force listen、视觉切片等 per-input hint 可以附在 input 上。

它替代：

- 当前 `/ws/chat` 的首个完整 chat request。
- `/ws/duplex` 的 `audio_chunk`
- 旧 `/v1/realtime` 的 `input_audio_buffer.append`

语义：

- server 按收到顺序处理同一 session 的 input。

### session.close

Layer: API and backend protocol.

关闭 session。它替代 `/ws/duplex` 的 `stop`。对 chat WebSocket，它也可以用于取消或结束当前 chat session。

语义：

- close 是完成语义；server 只有在 session 已关闭、资源已释放或进入不可再用状态后，才发送关闭确认或关闭 WebSocket。
- close 后不再接受新的 `input.append`。

## Server Events

### session.queued

Layer: API only.

session 进入排队状态。

语义字段：

- 当前排队位置。
- 估计等待时间。
- 队列长度。
- ticket 标识。

### session.queue_update

Layer: API only.

排队状态更新。

语义字段同 `session.queued`，但可以只包含变化字段。

### session.queue_done

Layer: API only.

session 已经分配到 worker/runtime，可以进入 init 阶段或等待 init 被处理。

### session.created

Layer: API and backend protocol.

session 初始化完成。它替代当前 `/ws/duplex` 的 `prepared`，也替代 `/ws/chat` 的 `prefill_done` 作为 session 生命周期确认。

语义字段：

- session id。
- prompt/init 相关指标。
- recording session id，如果启用录制。

### response.output.delta

Layer: API and backend protocol.

统一的模型推进结果事件。它替代：

- `/ws/chat` 的 `chunk`
- `/ws/duplex` 的 `result`
- 旧 `/v1/realtime` 的 `response.listen`
- 旧 `/v1/realtime` 的 `response.output_audio.delta`

该事件的 `type` 保持 OpenAI-like 命名，具体输出分支由 `kind` 区分。一个 `response.output.delta` frame 只能表达一种输出：

```text
kind = listen | text | audio
```

当 `kind = listen`：

- 表示模型决定继续听。
- 不携带 text delta。
- 不携带 audio delta。
- 不应再额外发送 `response.listen`。

当 `kind = text`：

- `text` 是文本增量。
- 不携带 audio delta。
- 不表达 listen。

当 `kind = audio`：

- `audio` 是音频增量。
- 不携带 text delta。
- 不表达 listen。

`response.output.delta` 的名字保留 response 概念，但不声称兼容 OpenAI Realtime 标准事件集合。它是 MiniCPM-o realtime API 的项目事件。

### response.done

Layer: API and backend protocol for `/v1/chat`.

chat response 输出边界。它替代当前 `/ws/chat` 的 `done`。

语义：

- 仅 `/v1/chat` 必须定义该事件。
- 表示一次 turn-based response 已经完成。
- streaming chat 中，前面可以有多个 `response.output.delta`，最后发 `response.done`。
- non-streaming chat 中，可以先发完整 text/audio delta，再发 `response.done`。
- 完整文本、完整音频和 metrics 可以放在 `response.done` 或附着在最后的 delta/metrics 字段上，具体 schema 后续再定。

`/v1/realtime` 暂不强制定义 `response.done`。full-duplex 输出边界先通过 `kind=listen`、后续输入和 `session.closed` 表达，避免把 turn-based done 语义强行塞进 realtime。

### session.closed

Layer: API and backend protocol.

session 已关闭。

语义字段：

- 关闭原因。
- 可选诊断信息。
- 可选 metrics。

正常 close、超时、上下文满、连接断开后的清理，都可以归一到 `session.closed`。如果 server 发送失败，可以直接关闭 WebSocket。

## Fatal Errors

该协议不定义可恢复的模型输入错误分支，不定义 `input.rejected`。

对于 runtime/backend 来说，非法顺序、非法输入形状、backend 异常都应视为 fatal condition：

- best-effort 发送 `session.closed`。
- 关闭 WebSocket。
- 释放 session 资源。

前端可以把 WebSocket close 或 `session.closed` 当作 terminal state 处理。

## Current Mapping From Existing Duplex

Layer: migration reference only.

| Existing `/ws/duplex` | Realtime API V2 |
|---|---|
| `queued` | `session.queued` |
| `queue_update` | `session.queue_update` |
| `queue_done` | `session.queue_done` |
| `prepare` | `session.init` |
| `prepared` | `session.created` |
| `audio_chunk` | `input.append` |
| `result` with `is_listen=true` | `response.output.delta` with `kind=listen` |
| `result.text` | `response.output.delta` with `kind=text` |
| `result.audio_data` | `response.output.delta` with `kind=audio` |
| `result.end_of_turn` | `response.output.delta` with `kind=listen` after final output, if a boundary must be surfaced |
| `stop` | `session.close` |
| `stopped` / `timeout` | `session.closed` |
| `pause` / `resume` | no protocol event; caller stops/continues input |
| `interrupt` | no protocol event; use per-input force-listen hint if still needed |

## Current Mapping From Existing Chat

Layer: migration reference only.

| Existing `/ws/chat` | Chat API V2 |
|---|---|
| initial chat request | `session.init` then `input.append` |
| `prefill_done` | `session.created` or metrics attached to early events |
| `chunk.text_delta` | `response.output.delta` with `kind=text` |
| `chunk.audio_data` | `response.output.delta` with `kind=audio` |
| `done` | `response.done` |
| `error` | fatal close / `session.closed` best-effort |

## Naming Notes

- Avoid `input_audio_buffer.append`: it exposes an implementation-shaped buffer name and becomes awkward once input can include video frames and per-input hints.
- Avoid `response.listen`: it is not a standard OpenAI Realtime event and makes a model state look like a separate response type.
- Avoid `response.output_audio.delta`: current payload carries text, audio, listen state and turn boundary, so `response.output.delta` plus `kind` is more accurate.
- Do not put text and audio in the same output frame. Text/audio/listen are independent output deltas and must remain separately ordered frames.
- Keep `/v1/chat` and `/v1/realtime` separate at the external API layer. They can still share backend protocol primitives internally.
- Keep concrete payload schemas out of this memo until parameter strong typing is settled.
