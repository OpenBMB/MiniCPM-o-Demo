---
title: "音频双工"
description: "Realtime API 的音频全双工协议"
---

音频双工模式用于纯语音实时对话。客户端持续发送 16 kHz 音频；服务端以 `response.output.delta` 返回 listen、文本和音频输出。

## 连接

```text
wss://host/v1/realtime?mode=audio
```

| 项目 | 值 |
|------|-----|
| 帧格式 | JSON 文本帧 |
| 上行音频 | 16 kHz，单声道，float32 PCM，base64 编码 |
| 上行视频 | 不需要发送 |
| 下行音频 | 24 kHz，单声道，float32 PCM，base64 编码 |
| 会话总时长上限 | 600 秒 |

## 初始化

连接后等待 `session.queue_done`，然后发送 `session.init`：

```json
{
  "type": "session.init",
  "payload": {
    "system_prompt": "你是一个有用的语音助手",
    "config": {
      "length_penalty": 1.1
    },
    "voice": {
      "ref_audio_base64": "<base64 float32 PCM>",
      "tts_ref_audio_base64": "<base64 float32 PCM>"
    }
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `system_prompt` | string | 否 | 系统提示词。也可使用 `instructions` |
| `config` | object | 否 | 双工推理配置；当前常用 `length_penalty` |
| `voice.ref_audio_base64` | string | 否 | LLM 参考音频 |
| `voice.tts_ref_audio_base64` | string | 否 | TTS 参考音频；未提供时可复用 LLM 参考音频 |

初始化完成后服务端发送 `session.created`。

## 发送输入

```json
{
  "type": "input.append",
  "input": {
    "audio": "<base64 float32 PCM, 16 kHz mono>",
    "force_listen": false
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `audio` | string | 是 | 16 kHz 单声道 float32 PCM。浏览器 demo 通常每秒发送一个 chunk |
| `force_listen` | bool | 否 | 强制模型回到 listen 状态 |
| `hints.force_listen` | bool | 否 | `force_listen` 的等价位置 |

音频模式不需要携带 `video_frames`。

## 接收输出

模型继续听时：

```json
{
  "type": "response.output.delta",
  "kind": "listen",
  "session_id": "sess_xxx",
  "metrics": {
    "kv_cache_length": 1024
  }
}
```

文本增量：

```json
{
  "type": "response.output.delta",
  "kind": "text",
  "text": "你好",
  "response_id": "resp_xxx",
  "metrics": {}
}
```

音频增量：

```json
{
  "type": "response.output.delta",
  "kind": "audio",
  "audio": "<base64 float32 PCM, 24 kHz mono>",
  "response_id": "resp_xxx",
  "metrics": {}
}
```

文本和音频是独立 delta，不保证一一对应。客户端应分别更新字幕和音频播放队列；收到 `kind=listen` 可视为模型回到听用户输入的状态。

## 时序

```text
Client connects
  <- session.queued / session.queue_update   optional
  <- session.queue_done
  -> session.init { payload: {...} }
  <- session.created
  -> input.append { input: { audio } }
  -> input.append { input: { audio } }
  <- response.output.delta { kind: "listen" }
  -> input.append { input: { audio } }
  <- response.output.delta { kind: "text", text: "..." }
  <- response.output.delta { kind: "audio", audio: "..." }
  <- response.output.delta { kind: "listen" }
  -> session.close
  <- session.closed
```

音频双工不使用 `response.done` 表示每轮结束；输出边界由 `kind=listen`、后续输入和 `session.closed` 表达。

## 关闭

```json
{
  "type": "session.close",
  "reason": "user_stop"
}
```

服务端会尽力返回 `session.closed`。如果达到 600 秒上限，gateway 会发送：

```json
{
  "type": "session.closed",
  "reason": "timeout"
}
```
