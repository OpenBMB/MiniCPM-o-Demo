---
title: "Realtime API Examples"
description: "Minimal Python clients for testing MiniCPM-o Realtime audio and video APIs."
---

This page shows minimal Python clients for calling the MiniCPM-o Realtime API
without a browser. The examples live in the repository under
[`examples/realtime/`](https://github.com/OpenBMB/MiniCPM-o-Demo/tree/main/examples/realtime).

## Install

```bash
cd examples/realtime
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

For the video example, make sure `ffmpeg` is available:

```bash
ffmpeg -version
```

## Audio probe

The audio probe opens `wss://host/v1/realtime?mode=audio`, waits for
`session.queue_done`, sends `session.init`, and then sends audio chunks as
`input.append` events. Input audio is 16 kHz mono float32 PCM.

```bash
python audio_probe.py \
  --url https://minicpmo45.modelbest.cn \
  --input-wav assets/test.wav \
  --region local-audio \
  --pretty-json
```

The bundled `assets/test.wav` is intentionally small and suitable for a quick
smoke test.

## Video probe

The video probe extracts 16 kHz mono audio and JPEG frames from an MP4 file with
`ffmpeg`, then sends `input.append` events with `input.video_frames`.

```bash
python video_probe.py \
  --url https://minicpmo45.modelbest.cn \
  --video /path/to/video.mp4 \
  --region local-video \
  --pretty-json
```

You can also download the hosted sample video first:

```bash
curl -L -o sample.mp4 \
  https://pub-4c730b83fc564d6a85ec9be6da99f10c.r2.dev/minicpmo/realtime-api/examples/VID_20260511_174245.mp4

python video_probe.py \
  --url https://minicpmo45.modelbest.cn \
  --video sample.mp4 \
  --region local-video \
  --pretty-json
```

## What to look at

- `ws_ready_ms`: WebSocket connection time.
- `first_text_ms`: time from connection start to the first text delta.
- `first_audio_ms`: time from connection start to the first output audio chunk.
- `output_audio_chunks`: number of downstream audio chunks.
- `chunk_interarrival_*_ms`: output audio chunk arrival interval distribution.
- `underrun_count`: simulated playback buffer underruns in the audio probe.
