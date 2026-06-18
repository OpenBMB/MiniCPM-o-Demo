---
title: "Realtime API 概览"
description: "MiniCPM-o Realtime API 当前公开 WebSocket 协议"
---

MiniCPM-o Realtime API 通过一个 WebSocket 入口提供 turn-based chat、视频全双工和音频全双工三种模式。
公网客户端只连接 Gateway；Gateway 负责排队、分配 worker、会话录制与转发，Python Worker 再把 runtime 协议消息转发给实际 backend（C++ `llama-omni-server` 或 PyTorch backend）。

## API Host

```text
https://minicpmo45.modelbest.cn
```

## 连接端点

```text
wss://host/v1/realtime?mode={chat|video|audio}
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `mode` | 否 | `video` 为默认值；可选 `chat`、`video`、`audio` |

`session_id` 是服务端返回的不透明字符串，客户端不需要在 URL 中传入，也不应该假设它的格式。

## 模式

| 模式 | 端点示例 | 上行输入 | 下行输出 | 说明 |
|------|---------|---------|---------|------|
| Chat | `wss://host/v1/realtime?mode=chat` | 一次 turn 的 `messages` | 文本增量、可选音频、`response.done` | 支持 streaming 和 non-streaming |
| 视频双工 | `wss://host/v1/realtime?mode=video` | 连续音频，可携带视频帧 | `listen`、文本增量、音频增量 | 会话总时长 300 秒 |
| 音频双工 | `wss://host/v1/realtime?mode=audio` | 连续音频 | `listen`、文本增量、音频增量 | 会话总时长 600 秒 |

`mode=chat` 在 backend runtime 中映射为 `turn_based`；`mode=video` 和 `mode=audio` 都映射为 `full_duplex`，二者由 Gateway 的请求类型和会话时长限制区分。

三种模式共享同一套事件命名：

- 初始化使用 `session.init`
- 提交输入使用 `input.append`
- 输出统一使用 `response.output.delta`，并通过 `kind` 区分 `listen`、`text`、`audio`
- Chat 模式用 `response.done` 表示一次 turn 输出完成
- 关闭使用 `session.close` 和 `session.closed`

## 生命周期

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

客户端应等到 `session.queue_done` 后再发送 `session.init`。如果没有排队，服务端也会立即发送 `session.queue_done`。

## 客户端事件

### session.init

`session.init` 初始化会话。消息必须包含对象类型的 `payload` 字段。

```json
{
  "type": "session.init",
  "payload": {
    "system_prompt": "你是一个有用的助手"
  }
}
```

常用字段：

| 字段 | 模式 | 说明 |
|------|------|------|
| `system_prompt` | `video` / `audio` | 双工模式系统提示词 |
| `instructions` | `video` / `audio` | `system_prompt` 的等价字段 |
| `config` | `video` / `audio` | 双工推理配置，例如 `length_penalty` |
| `voice.ref_audio_base64` | `video` / `audio` | LLM 参考音频，base64 PCM |
| `voice.tts_ref_audio_base64` | `video` / `audio` | TTS 参考音频，base64 PCM |

Chat 模式可以发送空 payload：

```json
{ "type": "session.init", "payload": {} }
```

### input.append

`input.append` 提交一次模型输入。消息必须包含对象类型的 `input` 字段。

Chat 模式：

```json
{
  "type": "input.append",
  "input": {
    "messages": [
      { "role": "user", "content": "请只回答：测试" }
    ],
    "streaming": true,
    "generation": {
      "max_new_tokens": 64,
      "length_penalty": 1.1
    },
    "tts": {
      "enabled": false
    }
  }
}
```

视频/音频双工模式：

```json
{
  "type": "input.append",
  "input": {
    "audio": "<base64 float32 PCM, 16 kHz mono>",
    "video_frames": ["<base64 JPEG>"],
    "force_listen": false,
    "max_slice_nums": 1
  }
}
```

`audio` 是双工模式的主要输入。`video_frames` 只用于 `mode=video`，`mode=audio` 不需要携带。

### session.close

```json
{
  "type": "session.close",
  "reason": "user_stop"
}
```

发送后客户端不应再发送新的 `input.append`。

## 服务端事件

### 排队事件

```json
{
  "type": "session.queued",
  "position": 2,
  "estimated_wait_s": 20,
  "ticket_id": "ticket_xxx",
  "queue_length": 3
}
```

`session.queue_update` 使用相同字段表达排队位置更新。`session.queue_done` 表示 worker 已分配完成，可以发送 `session.init`。

### session.created

```json
{
  "type": "session.created",
  "session_id": "sess_xxx",
  "mode": "full_duplex",
  "metrics": {}
}
```

表示会话初始化完成。

`mode` 字段表示 backend runtime mode，而不是 URL 中的公开 API mode：`mode=chat` 通常返回 `turn_based`，`mode=video` / `mode=audio` 通常返回 `full_duplex`。

### response.output.delta

```json
{
  "type": "response.output.delta",
  "kind": "text",
  "session_id": "sess_xxx",
  "response_id": "resp_xxx",
  "input_id": "input_xxx",
  "text": "你好",
  "metrics": {}
}
```

| `kind` | 字段 | 说明 |
|--------|------|------|
| `listen` | `metrics` | 模型继续听用户输入 |
| `text` | `text` | 文本增量 |
| `audio` | `audio` | 24 kHz 单声道 float32 PCM，base64 编码 |

一个 `response.output.delta` 只表达一种输出分支。文本和音频可能分别到达，客户端应按接收顺序更新字幕和播放音频。

### response.done

`response.done` 只用于 `mode=chat`，表示一次 turn-based response 完成。

```json
{
  "type": "response.done",
  "session_id": "sess_xxx",
  "response_id": "resp_xxx",
  "text": "测试",
  "reason": "turn_end",
  "metrics": {}
}
```

### session.closed

```json
{
  "type": "session.closed",
  "session_id": "sess_xxx",
  "reason": "user_stop"
}
```

`reason` 可能是 `user_stop`、`client_closed`、`timeout`、`backend_error` 等。客户端收到该事件或 WebSocket 关闭后，应认为会话已经结束。

### error

```json
{
  "type": "error",
  "error": {
    "code": "queue_full",
    "message": "Queue full",
    "type": "server_error"
  }
}
```

## 协议页面

- [Chat 模式](./chat/)：turn-based 文本/可选语音输出
- [视频双工](./video/)：连续音频加视频帧输入
- [音频双工](./audio/)：连续音频输入
- [使用范例](./examples/)：命令行 probe 客户端
