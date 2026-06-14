# Realtime API examples

Minimal command-line examples for the MiniCPM-o Realtime API.

Repository path:

```text
https://github.com/OpenBMB/MiniCPM-o-Demo/tree/main/examples/realtime
```

## Install

```bash
cd examples/realtime
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

`video_probe.py` also requires `ffmpeg`:

```bash
ffmpeg -version
```

## Audio example

```bash
python audio_probe.py \
  --url https://minicpmo45.modelbest.cn \
  --input-wav assets/test.wav \
  --region local-audio \
  --pretty-json
```

The script opens `wss://host/v1/realtime?mode=audio`, sends 16 kHz
mono float32 PCM audio chunks, and reports client-observed latency and
streaming smoothness metrics.

## Video example

Use your own MP4 file:

```bash
python video_probe.py \
  --url https://minicpmo45.modelbest.cn \
  --video /path/to/video.mp4 \
  --region local-video \
  --pretty-json
```

Or download the hosted sample video first:

```bash
curl -L -o sample.mp4 \
  https://pub-4c730b83fc564d6a85ec9be6da99f10c.r2.dev/minicpmo/realtime-api/examples/VID_20260511_174245.mp4

python video_probe.py \
  --url https://minicpmo45.modelbest.cn \
  --video sample.mp4 \
  --region local-video \
  --pretty-json
```

The script extracts 16 kHz mono audio and JPEG frames with `ffmpeg`,
then sends `input.append` events with `input.video_frames`.
