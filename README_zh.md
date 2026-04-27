# MiniCPM-o 4.5 PyTorch 演示 — Realtime API 协议

> **本仓库是 [OpenBMB/MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo) 的 fork**（`realtime-protocol` 分支）。
> 在原有演示系统基础上新增了 **OpenAI Realtime 风格的 WebSocket API**，使第三方客户端可以通过标准化协议与 MiniCPM-o 4.5 进行全双工对话。

[English Documentation](README.md) | [详细文档](https://openbmb.github.io/MiniCPM-o-Demo/site/zh/index.html) | [**Realtime API 协议文档**](docs/realtime-protocol-overview.md)

## 本 Fork 新增内容

本 fork 引入了 **MiniCPM-o Realtime API** — 基于 WebSocket 的全双工音视频对话协议。API 遵循 OpenAI Realtime 命名约定（`namespace.action` 事件格式），在 Gateway 层以**纯翻译层**实现，未修改任何模型或 Worker 代码。

### Realtime API 概览

| | |
|---|---|
| **端点** | `wss://host/v1/realtime?mode={video\|audio}` |
| **协议** | JSON over WebSocket，事件驱动 |
| **模式** | 视频双工（音频 + 视频，5 分钟）/ 音频双工（纯音频，10 分钟）|
| **音频格式** | 上行 16 kHz float32 PCM，下行 24 kHz float32 PCM |
| **文档** | [协议总览](docs/realtime-protocol-overview.md) · [视频双工](docs/video-duplex-protocol.md) · [音频双工](docs/audio-duplex-protocol.md) · [JSON Schema](docs/realtime-protocol-schema.json) |

### 示例代码

本仓库的全双工 demo 页面直接使用 Realtime API，可作为生产级的客户端实现参考：

| 页面 | 说明 |
|------|------|
| [`static/omni/`](static/omni/) | 视频双工 — 实时音视频对话，含摄像头画面 |
| [`static/audio-duplex/`](static/audio-duplex/) | 音频双工 — 实时纯音频对话 |

两个页面均包含 **Protocol Data Flow** 面板，可实时查看所有 WebSocket 事件流。

### 架构

```
┌─────────────┐     OpenAI Realtime      ┌──────────────┐    旧协议         ┌────────────┐
│   客户端     │ ←──── WebSocket ────→    │   Gateway    │ ←── WebSocket ──→ │   Worker   │
│ (浏览器 /   │   session.update         │  /v1/realtime│   prepare         │  (PyTorch) │
│  Python)    │   append / listen / ...  │   翻译层      │   audio_chunk     │   未修改    │
└─────────────┘                          └──────────────┘   result / ...    └────────────┘
```

Gateway（`gateway.py`）在 Realtime API 事件和原有 Worker 协议之间做双向翻译。**模型代码和 Worker 代码均未修改。**

## 快速开始

部署方式与 [main 分支](https://github.com/OpenBMB/MiniCPM-o-Demo) 完全一致。服务运行后可访问以下页面：

| 页面 | URL | 说明 |
|------|-----|------|
| 视频双工 | `https://host:port/omni` | 实时音视频对话（使用 Realtime API） |
| 音频双工 | `https://host:port/audio_duplex` | 实时纯音频对话（使用 Realtime API） |
| Realtime API 端点 | `wss://host:port/v1/realtime?mode={video\|audio}` | 供第三方客户端接入的 WebSocket API |
| 协议文档 | `https://host:port/docs/overview` | 可浏览的协议文档 |
| 轮次对话 | `https://host:port/` | 原有轮次对话（未改动） |
| 半双工语音 | `https://host:port/half_duplex` | 原有半双工语音通话（未改动） |
| 仪表盘 | `https://host:port/admin` | 管理面板 |
