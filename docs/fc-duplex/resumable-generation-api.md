# FC Duplex Resumable Generation API

## 1. 目的与范围

本文定义 FC Duplex 公共 WebSocket API 的可恢复生成日志。协议允许客户端在不接收
token ID、且服务端不保存 Session/KV 状态的前提下，保存完整双向事件历史，并在满足
明确条件的 Unit 边界重新构造模型状态。

本协议只支持：

- 相同模型、相同 tokenizer target 与相同协议版本；
- 从 `response.unit.committed` 声明为 `available` 的 Unit 边界恢复；
- 客户端提交从 Session 开始到目标 Unit 的完整双向历史；
- 通过 public text delta 重新 encode ordinary token，通过 protocol semantic key
  重建协议结构 token。

本协议不承诺所有 Unit 边界都可恢复。无法无损恢复时必须明确失败，不回滚、不猜测、
不依赖服务端残留状态。

## 2. 术语

### 2.1 Generation step

模型生成一个 token 对应一个 generation step。即使底层 primitive 一次返回多个 token，
View 也必须按 token 顺序展开为多个 step。

### 2.2 Text stream

一条连续 ordinary token 序列。每个 think span、tool-call span 和 spoken turn 分别拥有
独立 `stream_id`。

### 2.3 Safe text delta

SDK ordinary text decode stream 输出的完整 Unicode 文本。每个 delta 的边界必须保留，
不能先与相邻 delta 拼接再重新 encode。

### 2.4 Protocol structural token

表达 Unit、lane、span、turn 和终止原因的协议 token。公共 API 只暴露稳定 semantic key，
不暴露 token ID 或 tokenizer 原始 token 字符串。

### 2.5 Resume checkpoint

Unit 完成时产生的恢复资格声明。Checkpoint 只说明客户端保存的历史是否足以重建当前
Unit 后的模型状态，不引用服务端保存的 opaque state。

## 3. 传输 Envelope

沿用 `/v1/realtime` 的扁平 WebSocket JSON envelope：

```json
{
  "type": "response.generation.step_batch",
  "...": "event fields"
}
```

不引入 `version/event_id/payload` 外层包装。所有新增下行事件都必须携带：

- `session_id`
- `event_index`：Session 内严格单调递增
- `server_send_ts`

## 4. Canonical generation step batch

`response.generation.step_batch` 是 semantic resume 的 canonical 下行日志。现有
`response.think.*`、`response.tool_call.*`、`response.output.delta` 和
`response.output.sp_tokens` 可以继续作为业务便利事件，但恢复不得依赖它们推测遗漏的
generation step。

```json
{
  "type": "response.generation.step_batch",
  "session_id": "sess_xxx",
  "event_index": 1042,
  "batch_index": 7,
  "stream_id": "think_1",
  "track": "non_spoken",
  "steps": [
    {
      "step_index": 120,
      "unit_index": 7,
      "output": {
        "kind": "text_pending"
      }
    },
    {
      "step_index": 121,
      "unit_index": 8,
      "output": {
        "kind": "text_delta",
        "delta_index": 31,
        "text": "龘",
        "source_step_indices": [120, 121]
      }
    }
  ]
}
```

### 4.1 Step output variants

`output` 是以 `kind` 为 discriminator 的 union。

#### `text_pending`

```json
{"kind": "text_pending"}
```

表示该 ordinary generation step 没有产生新的安全 Unicode 文本。它不承诺具体原因，
也不等价于“半个字符”；它只保留 step 和 Unit provenance。

#### `text_delta`

```json
{
  "kind": "text_delta",
  "delta_index": 31,
  "text": "龘",
  "source_step_indices": [120, 121]
}
```

约束：

- `text` 必须是非空、完整 Unicode。
- `delta_index` 在同一 `stream_id` 内从 0 开始严格递增。
- `source_step_indices` 必须按顺序列出自上一次非空 delta 后、属于同一 stream 的全部
  ordinary steps。不同 track 的 step 可能插在这些全局 step index 之间，因此不能用
  单一闭区间替代该列表。
- 使用相同 tokenizer target 时必须满足：

```python
tokenizer.encode_ordinary(text) == source_token_ids
```

- API transport 可以 batch，但不能合并、拆分或改写单个 safe text delta。

#### `protocol`

```json
{
  "kind": "protocol",
  "semantic_key": "think_end"
}
```

`semantic_key` 使用 SDK 协议 registry 的稳定 key，例如：

```text
listen
speak
spoken_slot_eos
spoken_turn_eos
tts_pad
think_start
think_end
tool_call_start
tool_call_end
no_action
non_spoken_eos
non_spoken_budget_reached
non_spoken_hold
non_spoken_abort
```

纯模板骨架（`<unit>`、`</unit>`、slot start/end、audio placeholder）默认不逐条下发，
由 replay canonicalizer 按 Unit 模板重建。

### 4.2 Batch flush

满足任一条件必须 flush：

- 累积 50ms；
- 累积 16 个 steps；
- 遇到 protocol structural token；
- Unit 即将 committed；
- stream/span/turn 即将闭合；
- Session 即将关闭。

一个 batch 不得跨 `stream_id` 或 Unit commit，steps 顺序不得改变。

## 5. 业务便利事件

现有事件继续承担 UI 与工具执行语义：

```text
response.think.begin/delta/done
response.tool_call.args.begin/delta/end/raw
response.output.delta kind=text/audio/listen
response.output.sp_tokens
```

它们必须可由 canonical generation steps 与完整解析结果确定性投影。恢复的 ordinary
token 事实来源是 generation step batch 中逐项保留的 safe text delta，不是把便利事件的
文本任意拼接后重新 encode。

`session.created` 为可恢复客户端提供当前服务的 identity：

```json
{
  "type": "session.created",
  "session_id": "sess_xxx",
  "resume": {
    "protocol_version": "fc-duplex-resume-v1",
    "model": "/path/or/model-id",
    "tokenizer_target": "o45_fc",
    "tokenizer_fingerprint": {
      "vocab_hash": "...",
      "merges_hash": "..."
    },
    "ref_audio_sha256": "...",
    "prompt_wav_sha256": "..."
  }
}
```

客户端构造 `session.resume` 时必须原样带回该 identity。

Tool-call 的 canonical text delta 保存模型实际生成的 ordinary wire；业务执行只使用
`response.tool_call.args.raw` 的结构化结果。

## 6. Unit checkpoint

每个 Unit 完成后发送：

```json
{
  "type": "response.unit.committed",
  "session_id": "sess_xxx",
  "event_index": 1043,
  "unit_index": 8,
  "input_id": "input_000008",
  "last_step_index": 121,
  "resume": {
    "status": "available"
  }
}
```

不可恢复边界：

```json
{
  "type": "response.unit.committed",
  "session_id": "sess_xxx",
  "event_index": 1043,
  "unit_index": 7,
  "input_id": "input_000007",
  "last_step_index": 120,
  "resume": {
    "status": "unavailable",
    "reason": "pending_text_delta",
    "stream_id": "think_1",
    "pending_from_step": 120
  }
}
```

### 6.1 MVP 可恢复条件

Unit checkpoint 只有同时满足以下条件才可标为 `available`：

1. 所有 ordinary steps 已被某个 safe text delta 覆盖，并通过 re-encode 不变量。
2. 当前没有未闭合的 think/tool-call span。
3. 当前没有跨 Unit 延续的 spoken turn/TTS 状态。
4. 当前没有 deferred non-spoken close 等待下一次 prefill 才进入 KV。
5. 本 Unit 所需上行输入已完整记录。
6. 当前 Session 尚未产生需要恢复 tool-call ID 映射的工具调用或 tool result。
7. 模型、tokenizer fingerprint、LLM reference audio 与 TTS prompt wav 的内容 hash
   与原 Session 完全一致。

每个 `input.append.input_id` 必须在 Session 内唯一；`response.unit.committed.input_id`
必须引用实际被 Worker 处理的输入。Latency 调度中被丢弃且没有 checkpoint 的 input
不参与 replay。

条件不满足时标记 `unavailable`。后续 Unit 重新满足条件后可以再次标记为
`available`。例外：当前 MVP 一旦出现 deferred close 或 tool-call/tool-result 状态，
后续 checkpoint 都保持 `unavailable`，因为 public history 尚不足以重建这些状态。

## 7. Resume 请求

Resume 使用新 WebSocket 连接，第一帧发送 `session.resume`：

```json
{
  "type": "session.resume",
  "payload": {
    "protocol_version": "fc-duplex-resume-v1",
    "model": "minicpm-o-4.5",
    "tokenizer_target": "o45_fc",
    "tokenizer_fingerprint": {
      "vocab_hash": "...",
      "merges_hash": "..."
    },
    "ref_audio_sha256": "...",
    "prompt_wav_sha256": "...",
    "through_unit_index": 8,
    "history": [
      {"type": "session.init", "...": "..."},
      {"type": "input.append", "...": "..."},
      {"type": "response.generation.step_batch", "...": "..."},
      {"type": "response.unit.committed", "...": "..."}
    ]
  }
}
```

`history` 必须包含从 Session 开始到目标 checkpoint 的完整双向语义历史，包括：

- `session.init`
- 全部 `input.append` 原始输入（包含音频/图像或可读取引用）
- 全部 `input.tool_result*`
- 全部 canonical generation step batches
- 全部 Unit checkpoints

成功：

```json
{
  "type": "session.resumed",
  "through_unit_index": 8,
  "next_unit_index": 9
}
```

## 8. Stateless replay

服务端不得依赖旧 Session/KV、内存缓存或 opaque cursor。恢复过程：

1. 校验协议版本、模型、tokenizer target 与 fingerprint。
2. 校验 `event_index`、`step_index`、`batch_index`、`delta_index` 与 Unit 顺序连续。
3. 重放 `session.init` 与全部上行输入。
4. protocol structural semantic key 映射到当前 target 的固定 token ID。
5. 对每个 safe text delta 独立调用 `encode_ordinary(text)`。
6. 校验 re-encode token 数等于其 source step 数，并按 step 顺序恢复 ordinary token ID。
7. 按 Unit 模板重建未显式发送的结构骨架。
8. 以 deterministic feed（不重新采样）恢复历史输出 token 到模型/KV。
9. 从 `next_unit_index` 继续实时推理。

## 9. Resume 失败

当前只定义以下失败码：

```text
non_resumable_text_boundary
incomplete_event_history
model_or_tokenizer_mismatch
text_delta_roundtrip_mismatch
unsupported_open_span
unsupported_spoken_turn_state
unsupported_deferred_close
unsupported_tool_state
```

```json
{
  "type": "session.resume.failed",
  "code": "non_resumable_text_boundary",
  "unit_index": 7,
  "stream_id": "think_1",
  "pending_from_step": 120
}
```

失败时不回滚到更早 checkpoint、不猜测 token、不重新采样历史输出。

## 10. 分层职责

### SDK

- target-aware ordinary text decode stream；
- ordinary/control 边界校验；
- safe delta re-encode；
- target fingerprint。

### Capability

- 模型采样与原始 token IDs；
- Unit/KV/slot 状态；
- deterministic replay feed；
- 完整 span 批量 decode 校验；
- 音频生成。

### View

- 把 primitive 返回展开为逐 token steps；
- 持有各 text stream 的 DecodeStream；
- 产生 `text_pending` / `text_delta` / `protocol`；
- 维护 stream/step/delta/Unit 序号；
- 判断 checkpoint 是否可恢复；
- 校验 safe delta re-encode 不变量。

### API Runtime

- 聚合并发送 generation step batch；
- 发送 Unit checkpoint；
- 生成业务便利事件；
- 接收 history 并驱动 stateless replay；
- 不直接访问 tokenizer。

### Client

- 普通客户端只消费业务便利事件；
- 可恢复客户端保存完整双向历史；
- 只在 `resume.status=available` 的 Unit 发起 resume；
- 不接收或构造 token ID。

## 11. 非目标

本版本不支持：

- 任意 step 位置恢复；
- `resume.status=unavailable` 的 Unit 边界恢复；
- 跨模型、跨 tokenizer target 或 fingerprint 恢复；
- 缺失输入媒体或历史事件时恢复；
- text delta roundtrip 不一致时 fallback；
- 服务端 Session/KV 持久化；
- spoken turn/TTS 中途恢复；
- open think/tool-call span 中途恢复；
- deferred close 边界恢复。
- 包含 tool-call/tool-result 状态的 Session 恢复。
