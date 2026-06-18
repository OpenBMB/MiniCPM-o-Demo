---
title: "Realtime API Overview"
description: "Current public WebSocket protocol for the MiniCPM-o Realtime API"
---

The MiniCPM-o Realtime API exposes turn-based chat, video full-duplex, and audio full-duplex through one WebSocket endpoint.
Public clients connect only to the Gateway. The Gateway handles queueing, worker assignment, session recording, and forwarding; the Python Worker then forwards runtime protocol messages to the actual backend, either C++ `llama-omni-server` or the PyTorch backend.

## API Host

```text
https://minicpmo45.modelbest.cn
```

## Endpoint

```text
wss://host/v1/realtime?mode={chat|video|audio}
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `mode` | No | Defaults to `video`; allowed values are `chat`, `video`, and `audio` |

`session_id` is an opaque server value. Clients do not pass it in the URL and should not rely on a specific format.

## Modes

| Mode | Endpoint example | Input | Output | Notes |
|------|------------------|-------|--------|-------|
| Chat | `wss://host/v1/realtime?mode=chat` | One turn of `messages` | Text deltas, optional audio, `response.done` | Supports streaming and non-streaming |
| Video full-duplex | `wss://host/v1/realtime?mode=video` | Continuous audio, optionally with video frames | `listen`, text deltas, audio deltas | 300 second session limit |
| Audio full-duplex | `wss://host/v1/realtime?mode=audio` | Continuous audio | `listen`, text deltas, audio deltas | 600 second session limit |

`mode=chat` maps to backend runtime mode `turn_based`; both `mode=video` and `mode=audio` map to `full_duplex`, with the Gateway request type and session duration limit distinguishing the two public modes.

All modes use the same event names:

- Initialize with `session.init`
- Submit input with `input.append`
- Receive model output as `response.output.delta`, with `kind` set to `listen`, `text`, or `audio`
- Chat mode ends each turn with `response.done`
- Close with `session.close` and `session.closed`

## Lifecycle

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

Clients should wait for `session.queue_done` before sending `session.init`. If no queueing is needed, the server sends `session.queue_done` immediately.

## Client Events

### session.init

`session.init` initializes the session. The message must contain an object-valued `payload` field.

```json
{
  "type": "session.init",
  "payload": {
    "system_prompt": "You are a helpful assistant."
  }
}
```

Common fields:

| Field | Modes | Description |
|-------|-------|-------------|
| `system_prompt` | `video` / `audio` | System prompt for full-duplex modes |
| `instructions` | `video` / `audio` | Alias for `system_prompt` |
| `config` | `video` / `audio` | Duplex inference settings, for example `length_penalty` |
| `voice.ref_audio_base64` | `video` / `audio` | LLM reference audio, base64 PCM |
| `voice.tts_ref_audio_base64` | `video` / `audio` | TTS reference audio, base64 PCM |

Chat mode can initialize with an empty payload:

```json
{ "type": "session.init", "payload": {} }
```

### input.append

`input.append` submits model input. The message must contain an object-valued `input` field.

Chat mode:

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
    "tts": {
      "enabled": false
    }
  }
}
```

Video/audio full-duplex modes:

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

`audio` is the primary input for full-duplex modes. `video_frames` is only used with `mode=video`; `mode=audio` does not need it.

### session.close

```json
{
  "type": "session.close",
  "reason": "user_stop"
}
```

After this event, the client should not send more `input.append` events.

## Server Events

### Queue Events

```json
{
  "type": "session.queued",
  "position": 2,
  "estimated_wait_s": 20,
  "ticket_id": "ticket_xxx",
  "queue_length": 3
}
```

`session.queue_update` uses the same fields for queue position changes. `session.queue_done` means a worker has been assigned and the client can send `session.init`.

### session.created

```json
{
  "type": "session.created",
  "session_id": "sess_xxx",
  "mode": "full_duplex",
  "metrics": {}
}
```

The session has been initialized.

The `mode` field is the backend runtime mode, not the public API mode from the URL: `mode=chat` usually returns `turn_based`, while `mode=video` / `mode=audio` usually return `full_duplex`.

### response.output.delta

```json
{
  "type": "response.output.delta",
  "kind": "text",
  "session_id": "sess_xxx",
  "response_id": "resp_xxx",
  "input_id": "input_xxx",
  "text": "Hello",
  "metrics": {}
}
```

| `kind` | Fields | Description |
|--------|--------|-------------|
| `listen` | `metrics` | The model is listening for more user input |
| `text` | `text` | Text delta |
| `audio` | `audio` | 24 kHz mono float32 PCM, base64 encoded |

One `response.output.delta` frame describes one output branch. Text and audio can arrive separately; clients should update captions and audio playback in receive order.

### response.done

`response.done` is used only with `mode=chat` and marks the end of one turn-based response.

```json
{
  "type": "response.done",
  "session_id": "sess_xxx",
  "response_id": "resp_xxx",
  "text": "test",
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

`reason` may be `user_stop`, `client_closed`, `timeout`, `backend_error`, or another server diagnostic. After receiving this event or a WebSocket close, the client should treat the session as ended.

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

## Protocol Pages

- [Chat mode](./chat/): turn-based text and optional speech output
- [Video full-duplex](./video/): continuous audio plus video frame input
- [Audio full-duplex](./audio/): continuous audio input
- [Examples](./examples/): command-line probe clients
