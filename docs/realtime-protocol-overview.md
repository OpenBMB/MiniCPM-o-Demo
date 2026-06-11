# MiniCPM-o Realtime API 协议

本文记录当前 Docker 分支实际实现的 Realtime API。公开文档源位于
`docs-app/content/docs/zh/realtime-api/`，Docker gateway 会把该目录构建到 `/docs`。

## 连接端点

```text
wss://host/v1/realtime?mode={chat|video|audio}
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `mode` | 否 | 默认 `video`；可选 `chat`、`video`、`audio` |

`session_id` 由服务端生成并通过事件返回。客户端不在 URL 中传入，也不应假设它的格式。

## 模式

| 模式 | 上行输入 | 下行输出 | 说明 |
|------|----------|----------|------|
| `chat` | 一次 turn 的 `messages` | 文本增量、可选音频、`response.done` | 支持 streaming 和 non-streaming |
| `video` | 连续音频，可携带视频帧 | `listen`、文本增量、音频增量 | 会话总时长 300 秒 |
| `audio` | 连续音频 | `listen`、文本增量、音频增量 | 会话总时长 600 秒 |

## 事件模型

客户端事件：

- `session.init`：初始化 session，消息必须包含对象类型的 `payload` 字段。
- `input.append`：提交模型输入，消息必须包含对象类型的 `input` 字段。
- `session.close`：关闭 session。

服务端事件：

- `session.queued` / `session.queue_update` / `session.queue_done`：排队与 worker 分配状态。
- `session.created`：初始化完成。
- `response.output.delta`：统一输出事件，通过 `kind` 区分 `listen`、`text`、`audio`。
- `response.done`：仅 chat 模式使用，表示一次 turn 输出完成。
- `session.closed`：session 已关闭。
- `error`：错误事件。

## 基本时序

```text
Client connects
  <- session.queued / session.queue_update   optional
  <- session.queue_done
  -> session.init
  <- session.created
  -> input.append
  <- response.output.delta / response.done
  -> session.close
  <- session.closed
```

客户端应等到 `session.queue_done` 后再发送 `session.init`。如果没有排队，服务端也会立即发送
`session.queue_done`。

## 输出分支

一个 `response.output.delta` 只表达一种输出分支：

| `kind` | 字段 | 说明 |
|--------|------|------|
| `listen` | `metrics` | 模型继续听用户输入 |
| `text` | `text` | 文本增量 |
| `audio` | `audio` | 24 kHz 单声道 float32 PCM，base64 编码 |

文本和音频是独立 delta，不保证一一对应。客户端应按接收顺序更新字幕和播放音频。
