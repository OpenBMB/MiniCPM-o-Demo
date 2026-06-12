# MiniCPM-o 视频双工协议

视频双工使用当前 Realtime API：

```text
wss://host/v1/realtime?mode=video
```

客户端持续发送 16 kHz 单声道 float32 PCM 音频，base64 编码，并可随输入携带 JPEG 视频帧；
服务端返回文本和 24 kHz 单声道 float32 PCM 音频。会话总时长上限为 300 秒。

## 初始化

连接后等待 `session.queue_done`，然后发送：

```json
{
  "type": "session.init",
  "payload": {
    "system_prompt": "你是一个有用的音视频助手",
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

初始化完成后，服务端返回 `session.created`。

## 发送输入

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

`video_frames` 可以为空或省略；只发送音频时，该模式仍按视频双工 session 运行。

## 接收输出

模型继续听：

```json
{
  "type": "response.output.delta",
  "kind": "listen",
  "metrics": {}
}
```

文本增量：

```json
{
  "type": "response.output.delta",
  "kind": "text",
  "text": "你好",
  "response_id": "resp_xxx",
  "session_id": "sess_xxx"
}
```

音频增量：

```json
{
  "type": "response.output.delta",
  "kind": "audio",
  "audio": "<base64 float32 PCM, 24 kHz mono>",
  "response_id": "resp_xxx",
  "session_id": "sess_xxx"
}
```

文本和音频是独立 delta，不保证一一对应。视频双工不使用 `response.done` 表示每轮结束；
输出边界由 `kind=listen`、后续输入和 `session.closed` 表达。

## 关闭

```json
{
  "type": "session.close",
  "reason": "user_stop"
}
```

服务端会尽力返回 `session.closed`；达到时长上限时返回 `reason=timeout`。
