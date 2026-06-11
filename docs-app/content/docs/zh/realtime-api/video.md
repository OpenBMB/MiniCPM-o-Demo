---
title: "视频双工"
description: "Realtime API 的视频全双工协议"
---

视频双工模式用于实时音视频对话。客户端持续发送 16 kHz 音频，可在每个输入包中携带 JPEG 视频帧；服务端以 `response.output.delta` 返回 listen、文本和音频输出。

## 连接

```text
wss://host/v1/realtime?mode=video
```

| 项目 | 值 |
|------|-----|
| 帧格式 | JSON 文本帧 |
| 上行音频 | 16 kHz，单声道，float32 PCM，base64 编码 |
| 上行视频 | JPEG，base64 编码，字段为 `input.video_frames` |
| 下行音频 | 24 kHz，单声道，float32 PCM，base64 编码 |
| 会话总时长上限 | 300 秒 |

## 初始化

连接后等待 `session.queue_done`，然后发送 `session.init`：

```json
{
  "type": "session.init",
  "payload": {
    "system_prompt": "你是一个有用的视频语音助手",
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
    "video_frames": ["<base64 JPEG>"],
    "force_listen": false,
    "max_slice_nums": 1
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `audio` | string | 是 | 16 kHz 单声道 float32 PCM。浏览器 demo 通常每秒发送一个 chunk |
| `video_frames` | string[] | 否 | JPEG 帧列表。视频模式建议随输入持续携带 |
| `force_listen` | bool | 否 | 强制模型回到 listen 状态 |
| `max_slice_nums` | int | 否 | 本次输入的视频切片数，默认 1 |
| `hints.force_listen` | bool | 否 | `force_listen` 的等价位置 |
| `hints.max_slice_nums` | int | 否 | `max_slice_nums` 的等价位置 |

## 接收输出

模型继续听时：

```json
{
  "type": "response.output.delta",
  "kind": "listen",
  "session_id": "sess_xxx",
  "metrics": {
    "kv_cache_length": 1024,
    "vision_slices": 1,
    "vision_tokens": 64
  }
}
```

文本增量：

```json
{
  "type": "response.output.delta",
  "kind": "text",
  "text": "我看到了",
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
  -> input.append { input: { audio, video_frames } }
  -> input.append { input: { audio, video_frames } }
  <- response.output.delta { kind: "listen" }
  -> input.append { input: { audio, video_frames } }
  <- response.output.delta { kind: "text", text: "..." }
  <- response.output.delta { kind: "audio", audio: "..." }
  <- response.output.delta { kind: "listen" }
  -> session.close
  <- session.closed
```

视频双工不使用 `response.done` 表示每轮结束；输出边界由 `kind=listen`、后续输入和 `session.closed` 表达。

## 关闭

```json
{
  "type": "session.close",
  "reason": "user_stop"
}
```

服务端会尽力返回 `session.closed`。如果达到 300 秒上限，gateway 会发送：

```json
{
  "type": "session.closed",
  "reason": "timeout"
}
```
