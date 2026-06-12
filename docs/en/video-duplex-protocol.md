# MiniCPM-o Video Full-Duplex Protocol

Video full-duplex uses the current Realtime API:

```text
wss://host/v1/realtime?mode=video
```

The client continuously sends 16 kHz mono float32 PCM audio, base64-encoded, and can include JPEG
video frames with each input. The server returns text and 24 kHz mono float32 PCM audio. The session
limit is 300 seconds.

## Initialization

After connecting, wait for `session.queue_done`, then send:

```json
{
  "type": "session.init",
  "payload": {
    "system_prompt": "You are a helpful audio-video assistant.",
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

The server returns `session.created` when initialization completes.

## Sending Input

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

`video_frames` may be empty or omitted; audio-only input still runs in a video full-duplex session.

## Receiving Output

The model continues listening:

```json
{
  "type": "response.output.delta",
  "kind": "listen",
  "metrics": {}
}
```

Text delta:

```json
{
  "type": "response.output.delta",
  "kind": "text",
  "text": "Hello",
  "response_id": "resp_xxx",
  "session_id": "sess_xxx"
}
```

Audio delta:

```json
{
  "type": "response.output.delta",
  "kind": "audio",
  "audio": "<base64 float32 PCM, 24 kHz mono>",
  "response_id": "resp_xxx",
  "session_id": "sess_xxx"
}
```

Text and audio are independent deltas and are not guaranteed to align one-to-one. Video full-duplex
does not use `response.done` for turn boundaries; boundaries are expressed by `kind=listen`,
subsequent input, and `session.closed`.

## Closing

```json
{
  "type": "session.close",
  "reason": "user_stop"
}
```

The server tries to return `session.closed`; when the duration limit is reached, it returns
`reason=timeout`.
