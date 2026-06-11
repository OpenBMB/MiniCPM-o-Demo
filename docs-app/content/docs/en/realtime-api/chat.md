---
title: "Chat Mode"
description: "Turn-based chat protocol for the Realtime API"
---

Chat mode uses the same Realtime WebSocket endpoint for one or more turn-based chat requests.

## Connection

```text
wss://host/v1/realtime?mode=chat
```

After connecting, wait for `session.queue_done`, then send `session.init`.

```json
{ "type": "session.init", "payload": {} }
```

After the server returns `session.created`, send `input.append`.

## Input

```json
{
  "type": "input.append",
  "input": {
    "messages": [
      { "role": "user", "content": "Reply with exactly: test" }
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

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `messages` | array | Yes | Conversation messages. `role` supports `system`, `user`, and `assistant` |
| `streaming` | bool | No | Defaults to `true`. When true, the server streams text/audio deltas before `response.done` |
| `generation.max_new_tokens` | int | No | Generation length limit, default 256 |
| `generation.length_penalty` | number | No | Length penalty, default 1.1 |
| `image.max_slice_nums` | int | No | Multimodal image slicing count |
| `omni_mode` | bool | No | Enables omni chat input |
| `tts.enabled` | bool | No | Enables speech output, default `false` |
| `tts.ref_audio_data` | string | No | TTS reference audio, base64 float32 PCM |
| `use_tts_template` | bool | No | Whether to use the TTS template |
| `enable_thinking` | bool | No | Whether to enable thinking |

`messages[].content` can be a string or a multimodal content list:

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "Describe this image" },
    { "type": "image", "data": "<base64 image>" }
  ]
}
```

## Output

Streaming chat first returns zero or more `response.output.delta` events:

```json
{
  "type": "response.output.delta",
  "kind": "text",
  "text": "test",
  "response_id": "resp_xxx",
  "session_id": "sess_xxx"
}
```

If TTS is enabled, audio is sent as a separate delta:

```json
{
  "type": "response.output.delta",
  "kind": "audio",
  "audio": "<base64 float32 PCM, 24 kHz mono>",
  "response_id": "resp_xxx",
  "session_id": "sess_xxx"
}
```

At the end of one turn, the server sends `response.done`:

```json
{
  "type": "response.done",
  "text": "test",
  "reason": "turn_end",
  "metrics": {}
}
```

Non-streaming chat can return only `response.done`, with the full text in `text`; if audio is generated, the full audio is in `audio`.

## Close

```json
{ "type": "session.close", "reason": "turn_done" }
```

After closing, the server returns `session.closed` or closes the WebSocket directly.
