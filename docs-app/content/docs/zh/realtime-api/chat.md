---
title: "Chat 模式"
description: "Realtime API 的 turn-based chat 协议"
---

Chat 模式使用同一个 Realtime WebSocket 入口处理一轮或多轮 turn-based 对话。

## 连接

```text
wss://host/v1/realtime?mode=chat
```

连接后先等待 `session.queue_done`，再发送 `session.init`。

```json
{ "type": "session.init", "payload": {} }
```

服务端返回 `session.created` 后，客户端发送 `input.append`。

## 输入

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
    "image": {
      "max_slice_nums": 1
    },
    "omni_mode": false,
    "tts": {
      "enabled": false
    },
    "use_tts_template": false,
    "enable_thinking": false
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `messages` | array | 是 | 对话消息。`role` 支持 `system`、`user`、`assistant` |
| `streaming` | bool | 否 | 默认 `true`。`true` 时服务端发送文本/音频增量后再发送 `response.done` |
| `generation.max_new_tokens` | int | 否 | 生成长度上限，默认 256 |
| `generation.length_penalty` | number | 否 | 长度惩罚，默认 1.1 |
| `image.max_slice_nums` | int | 否 | 多模态图片切片数 |
| `omni_mode` | bool | 否 | 是否启用 omni chat 输入 |
| `tts.enabled` | bool | 否 | 是否生成语音输出，默认 `false` |
| `tts.ref_audio_data` | string | 否 | TTS 参考音频，base64 float32 PCM |
| `use_tts_template` | bool | 否 | 是否使用 TTS 模板 |
| `enable_thinking` | bool | 否 | 是否启用 thinking |

`messages[].content` 可以是字符串，也可以是多模态列表：

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "描述这张图" },
    { "type": "image", "data": "<base64 image>" }
  ]
}
```

## 输出

Streaming chat 会先收到若干 `response.output.delta`：

```json
{
  "type": "response.output.delta",
  "kind": "text",
  "text": "测试",
  "response_id": "resp_xxx",
  "session_id": "sess_xxx"
}
```

如果启用 TTS，音频作为独立 delta 下发：

```json
{
  "type": "response.output.delta",
  "kind": "audio",
  "audio": "<base64 float32 PCM, 24 kHz mono>",
  "response_id": "resp_xxx",
  "session_id": "sess_xxx"
}
```

一次 turn 结束时服务端发送 `response.done`：

```json
{
  "type": "response.done",
  "text": "测试",
  "reason": "turn_end",
  "metrics": {}
}
```

Non-streaming chat 可以只返回 `response.done`，完整文本在 `text` 字段中；如果生成音频，完整音频在 `audio` 字段中。

## 关闭

```json
{ "type": "session.close", "reason": "turn_done" }
```

服务端关闭后返回 `session.closed` 或直接关闭 WebSocket。
