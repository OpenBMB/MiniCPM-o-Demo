# MiniCPM-o 4.5 PyTorch Demo — Realtime API Protocol

> **This is a fork of [OpenBMB/MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo)** (`realtime-protocol` branch).
> It adds an **OpenAI Realtime-style WebSocket API** on top of the original demo system, enabling third-party clients to interact with MiniCPM-o 4.5 via a standardized protocol.

[中文简介](README_zh.md) | [**Realtime API Docs**](docs/realtime-protocol-overview.md)

## What's New in This Fork

This fork introduces the **MiniCPM-o Realtime API** — a WebSocket-based protocol for full-duplex audio/video conversations with MiniCPM-o 4.5. The API follows OpenAI Realtime conventions (`namespace.action` event naming) and is implemented as a **pure translation layer** in the gateway, with zero changes to the model or worker.

### Realtime API at a Glance

| | |
|---|---|
| **Endpoint** | `wss://host/v1/realtime?mode={video\|audio}` |
| **Protocol** | JSON over WebSocket, event-driven |
| **Modes** | Video duplex (audio + video, 5 min) / Audio duplex (audio only, 10 min) |
| **Audio format** | 16 kHz float32 PCM input, 24 kHz float32 PCM output |
| **Documentation** | [Protocol Overview](docs/realtime-protocol-overview.md) · [Video Duplex](docs/video-duplex-protocol.md) · [Audio Duplex](docs/audio-duplex-protocol.md) · [JSON Schema](docs/realtime-protocol-schema.json) |

### Example Code

The full-duplex demo pages in this repository serve as production-quality examples of Realtime API usage:

| Page | Description |
|------|-------------|
| [`static/omni/`](static/omni/) | Video duplex — real-time audio + video conversation with camera |
| [`static/audio-duplex/`](static/audio-duplex/) | Audio duplex — real-time audio-only conversation |

Both pages include a **Protocol Data Flow** panel that visualizes all WebSocket events in real time, making them useful for understanding the protocol in action.

### Architecture

```
┌─────────────┐     OpenAI Realtime      ┌──────────────┐    old protocol    ┌────────────┐
│   Client    │ ←──── WebSocket ────→    │   Gateway    │ ←── WebSocket ──→  │   Worker   │
│ (browser /  │   session.update         │  /v1/realtime│   prepare          │  (PyTorch) │
│  Python)    │   append / listen / ...  │  translation │   audio_chunk      │  unchanged │
└─────────────┘                          └──────────────┘   result / ...     └────────────┘
```

The gateway (`gateway.py`) translates between Realtime API events and the existing worker protocol. **No model code or worker code was modified.**

## Quick Start

Deployment is identical to the [main branch](https://github.com/OpenBMB/MiniCPM-o-Demo). After the service is running, the following pages are available:

| Page | URL | Description |
|------|-----|-------------|
| Video Duplex | `https://host:port/omni` | Real-time audio + video conversation (uses Realtime API) |
| Audio Duplex | `https://host:port/audio_duplex` | Real-time audio-only conversation (uses Realtime API) |
| Realtime API Endpoint | `wss://host:port/v1/realtime?mode={video\|audio}` | WebSocket API for third-party clients |
| Protocol Docs | `https://host:port/docs/overview` | Browsable protocol documentation |
| Turn-based Chat | `https://host:port/` | Original turn-based chat (unchanged) |
| Half-Duplex Audio | `https://host:port/half_duplex` | Original half-duplex voice call (unchanged) |
| Dashboard | `https://host:port/admin` | Admin dashboard |
