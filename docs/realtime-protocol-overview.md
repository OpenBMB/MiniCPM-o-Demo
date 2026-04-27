# MiniCPM-o Realtime API 协议

MiniCPM-o 提供两种全双工实时对话模式，通过 WebSocket 协议通信。

## 两种模式

| 模式 | 端点 | 上行数据 | 会话时长 | 有效对话 |
|------|------|---------|---------|---------|
| **视频双工** | `wss://host/v1/realtime?mode=video` | 音频 + 视频帧（必须） | 5 分钟 | ~90 秒 |
| **音频双工** | `wss://host/v1/realtime?mode=audio` | 仅音频（不允许视频） | 10 分钟 | 待补充 |

两种模式共享相同的事件命名和消息结构，区别在于：
- **视频双工**：`input_audio_buffer.append` **必须**携带 `video_frames`
- **音频双工**：`input_audio_buffer.append` **不允许**携带 `video_frames`

选择一种模式后，整个会话期间不能切换。

## 协议文档

- [视频双工协议](video-duplex-protocol.md) — 含视频帧的全双工对话
- [音频双工协议](audio-duplex-protocol.md) — 纯音频的全双工对话
- [JSON Schema](realtime-protocol-schema.json) — 机器可读的消息格式定义

## 共同特性

- 事件命名兼容 OpenAI Realtime API（`namespace.action` 格式）
- 上行音频：16 kHz float32 PCM，1 秒/chunk
- 下行音频：24 kHz float32 PCM，中间 chunk 固定 1 秒
- 上下文窗口：8192 tokens，固定不可调
- 文字领先音频：由于模型架构，`text` 内容领先 `audio` 数百毫秒
- 排队机制：FIFO 队列，含位置和 ETA 估算
- 打断机制：通过 `force_listen=true` 字段
