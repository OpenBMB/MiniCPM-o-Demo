---
title: "Video Full-Duplex"
description: "Video full-duplex protocol for the Realtime API"
---

Video full-duplex mode is for realtime audio/video conversation. The client continuously sends 16 kHz audio and can include JPEG video frames in each input packet; the server returns listen, text, and audio output as `response.output.delta`.

## Connection

```text
wss://host/v1/realtime?mode=video
```

| Item | Value |
|------|-------|
| Frame format | JSON text frames |
| Input audio | 16 kHz mono float32 PCM, base64 encoded |
| Input video | JPEG, base64 encoded, under `input.video_frames` |
| Output audio | 24 kHz mono float32 PCM, base64 encoded |
| Total session duration limit | 300 seconds |

## Initialize

After connecting, wait for `session.queue_done`, then send `session.init`:

```json
{
  "type": "session.init",
  "payload": {
    "system_prompt": "You are a helpful video voice assistant.",
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

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `system_prompt` | string | No | System prompt. `instructions` is also accepted |
| `config` | object | No | Duplex inference settings; `length_penalty` is commonly used |
| `voice.ref_audio_base64` | string | No | LLM reference audio |
| `voice.tts_ref_audio_base64` | string | No | TTS reference audio; can fall back to the LLM reference audio |

The server sends `session.created` after initialization completes.

## Send Input

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

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | string | Yes | 16 kHz mono float32 PCM. The browser demo usually sends one chunk per second |
| `video_frames` | string[] | No | JPEG frame list. In video mode, send frames continuously when available |
| `force_listen` | bool | No | Force the model back to listen state |
| `max_slice_nums` | int | No | Video slice count for this input, default 1 |
| `hints.force_listen` | bool | No | Equivalent position for `force_listen` |
| `hints.max_slice_nums` | int | No | Equivalent position for `max_slice_nums` |

## Receive Output

Model keeps listening:

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

Text delta:

```json
{
  "type": "response.output.delta",
  "kind": "text",
  "text": "I can see",
  "response_id": "resp_xxx",
  "metrics": {}
}
```

Audio delta:

```json
{
  "type": "response.output.delta",
  "kind": "audio",
  "audio": "<base64 float32 PCM, 24 kHz mono>",
  "response_id": "resp_xxx",
  "metrics": {}
}
```

Text and audio are separate deltas and are not guaranteed to map one-to-one. Clients should update captions and the audio playback queue independently; `kind=listen` means the model has returned to listening for user input.

## Timeline

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

Video full-duplex mode does not use `response.done` for every output turn. The boundary is represented by `kind=listen`, later input, and `session.closed`.

## Close

```json
{
  "type": "session.close",
  "reason": "user_stop"
}
```

The server makes a best effort to return `session.closed`. If the 300 second limit is reached, the gateway sends:

```json
{
  "type": "session.closed",
  "reason": "timeout"
}
```
