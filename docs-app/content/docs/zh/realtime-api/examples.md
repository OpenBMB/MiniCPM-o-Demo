---
title: "Realtime API 使用范例"
description: "用于测试 MiniCPM-o Realtime 音频和视频 API 的最小 Python 客户端。"
---

本页提供不依赖浏览器的最小 Python 客户端，用于调用 MiniCPM-o Realtime
API。示例代码位于项目仓库
[`examples/realtime/`](https://github.com/OpenBMB/MiniCPM-o-Demo/tree/main/examples/realtime)
目录。

## 安装依赖

```bash
cd examples/realtime
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

视频示例还需要系统中安装 `ffmpeg`：

```bash
ffmpeg -version
```

## 音频探测

音频探测脚本会连接 `wss://host/v1/realtime?mode=audio`，等待
`session.queue_done` 后发送 `session.init`，再按 chunk 发送 `input.append`
事件。音频为 16 kHz 单声道 float32 PCM。

```bash
python audio_probe.py \
  --url https://minicpmo45.modelbest.cn \
  --input-wav assets/test.wav \
  --region local-audio \
  --pretty-json
```

仓库内置的 `assets/test.wav` 很小，适合快速 smoke test。

## 视频探测

视频探测脚本会用 `ffmpeg` 从 MP4 中提取 16 kHz 单声道音频和 JPEG 帧，然后发送携带
`input.video_frames` 的 `input.append` 事件。

```bash
python video_probe.py \
  --url https://minicpmo45.modelbest.cn \
  --video /path/to/video.mp4 \
  --region local-video \
  --pretty-json
```

也可以先下载托管的示例视频：

```bash
curl -L -o sample.mp4 \
  https://pub-4c730b83fc564d6a85ec9be6da99f10c.r2.dev/minicpmo/realtime-api/examples/VID_20260511_174245.mp4

python video_probe.py \
  --url https://minicpmo45.modelbest.cn \
  --video sample.mp4 \
  --region local-video \
  --pretty-json
```

## 关键指标

- `ws_ready_ms`：WebSocket 建连耗时。
- `first_text_ms`：从连接开始到收到第一个文本增量的耗时。
- `first_audio_ms`：从连接开始到收到第一个输出音频 chunk 的耗时。
- `output_audio_chunks`：下行音频 chunk 数量。
- `chunk_interarrival_*_ms`：下行音频 chunk 到达间隔分布。
- `underrun_count`：音频探测脚本模拟播放缓冲时的 underrun 次数。
