# FC Duplex Semantic Realtime API v2

## 1. 目标

本协议以单 WebSocket 有序状态机为基础，只公开业务和 stateless resume 真正需要的信息。
协议不暴露 token ID，不重复传输同一语义，不为当前不存在的并发能力预埋 ID。

## 2. 不变量

- 一个 WebSocket 只承载一个 Session。
- 同时最多一个 active spoken turn。
- 同时最多一个 active think。
- Tool-call 可以异步等待结果，因此只有 tool-call 需要 `tool_call_id`。
- Unit 是模型调度边界；think、tool-call 和 spoken turn 可以跨 Unit。
- WebSocket 接收顺序就是事件顺序，不额外定义 event/step/batch/delta 序号。

### 2.1 Checkpoint Profile 参数

通用 Runtime 不维护 checkpoint 专属训练事实。调用方必须在 `session.init` 中显式携带由
外部 Checkpoint Profile 展开的身份和两类 budget：

```json
{
  "checkpoint_profile_id": "profile_id",
  "config": {
    "non_spoken_scheduling": "quality",
    "non_spoken_budget_while_listening": 30,
    "non_spoken_budget_while_speaking": 15
  }
}
```

示例值只表示调用方展开后的结果，不是 API 默认值。调度模式同样属于 Profile：当前
checkpoint 若只验证了 quality，就不能由页面切到 latency。Runtime 根据当前 Unit 的
spoken 决策选择 listening 或 speaking budget。若进程环境已经绑定 Profile，
`session.init` 给出的值必须完全一致，否则 fail-fast；两侧都没有提供时也必须失败。

## 3. 必要关联字段

协议只保留：

```text
unit_index     输出所属的模型调度 Unit
input_id       实际被该 Unit 消费的输入
tool_call_id   异步 tool result 关联
```

删除以下公开字段：

```text
block_id
span_id
response_id
每个事件重复的 session_id
event_index
step_index
batch_index
delta_index
source_step_indices
source_steps
```

## 4. Text step

每个 step 是 discriminated union：

```json
{"kind": "pending"}
```

```json
{
  "kind": "text",
  "text": "动物"
}
```

`pending` 表示一个 ordinary token 尚未产生安全 Unicode。每个 `text` step 自动覆盖同一
有序 semantic stream 中从上次 `text` 后累计的全部 `pending` 加当前 step；客户端可直接
从 steps 顺序推导，不重复发送计数。Transport 可以把多个 step 放进同一事件，但不能合并
或拆分单个 safe text。

## 5. Unit 生命周期

Backend 在实际完成 prefill 后发送：

```json
{
  "type": "response.unit.started",
  "unit_index": 12,
  "input_id": "input_000015",
  "tool_events": [
    {
      "type": "tool_result",
      "tool_call_id": "tc_000002"
    }
  ]
}
```

每个实际处理 Unit 都必须发送，包括空 events：

```json
{
  "type": "response.unit.started",
  "unit_index": 13,
  "input_id": "input_000016",
  "tool_events": []
}
```

`tool_events` 只表达实际 Unit 归属，不重复 tool result 内容；内容从此前
`input.tool_result` 读取。

Unit 完成：

```json
{
  "type": "response.unit.committed",
  "unit_index": 12,
  "non_spoken_end": "budget_reached",
  "resume": {
    "status": "unavailable",
    "reason": "deferred_close"
  }
}
```

`non_spoken_end`：

```text
no_action
eos
budget_reached
hold
abort
```

`budget_reached` 按协议固定表示 close token、slot end 和 Unit end 在下一 Unit prefill 前
进入 KV，因此不再额外发送 `deferred_model_feed`。

## 6. Think

同一时刻最多一个 active think，因此不需要 ID。

```json
{
  "type": "response.think.begin",
  "unit_index": 5
}
```

```json
{
  "type": "response.think.delta",
  "unit_index": 5,
  "steps": [
    {"kind": "text", "text": "用户给我"},
    {"kind": "pending"}
  ]
}
```

跨 Unit 继续时仍属于同一个 think：

```json
{
  "type": "response.think.delta",
  "unit_index": 6,
  "steps": [
    {"kind": "text", "text": "设了个规则"}
  ]
}
```

只有模型产生 `</think>` 时发送：

```json
{
  "type": "response.think.end",
  "unit_index": 8,
  "full_text": "用户给我设了个规则……"
}
```

Budget 不发送 `think.end`，只重置内部 BPE decoder segment。Capability 的完整 think
aggregate 保留到 matching end。

## 7. Tool-call

Tool-call 需要 ID，因为外部结果异步返回。

```json
{
  "type": "response.tool_call.begin",
  "tool_call_id": "tc_000002",
  "unit_index": 16
}
```

```json
{
  "type": "response.tool_call.delta",
  "tool_call_id": "tc_000002",
  "unit_index": 16,
  "steps": [
    {
      "kind": "text",
      "text": "<function name=\"display_object_on_board\">"
    }
  ]
}
```

Budget 不结束 semantic tool-call，也不更换 `tool_call_id`。只有模型产生
`</tool_call>` 时发送一个最终事件：

```json
{
  "type": "response.tool_call.done",
  "tool_call_id": "tc_000002",
  "unit_index": 17,
  "full_text": "<function ...>...</function>",
  "call": {
    "name": "display_object_on_board",
    "arguments": {
      "name": "老鼠"
    }
  }
}
```

解析失败：

```json
{
  "type": "response.tool_call.done",
  "tool_call_id": "tc_000002",
  "unit_index": 17,
  "full_text": "...",
  "error": "invalid tool-call XML"
}
```

不再拆成 `args.end` 和 `args.raw` 两条事件。

## 8. Tool result

```json
{
  "type": "input.tool_result",
  "tool_call_id": "tc_000002",
  "content": {
    "status": "displayed",
    "name": "老鼠"
  }
}
```

等待结果或等待下一 Unit 注入时 checkpoint 暂时不可恢复：

```json
{
  "status": "unavailable",
  "reason": "pending_tool_result"
}
```

`response.unit.started.tool_events` 确认结果被某 Unit 消费后，可以重新 available。

## 9. Spoken

同时最多一个 active spoken turn，因此不需要 turn ID。

```json
{
  "type": "response.spoken.delta",
  "unit_index": 5,
  "steps": [
    {"kind": "text", "text": "好的"}
  ],
  "audio": "...base64...",
  "sample_rate": 24000
}
```

每个 Unit 的 spoken slot 结束：

```json
{
  "type": "response.spoken.end",
  "unit_index": 5,
  "reason": "slot_eos"
}
```

如果模型没有生成 `spoken_slot_eos`、仅由模板关闭 slot，则使用
`reason: "slot_end"`；该 reason 不对应额外模型 token。

模型产生 `SPEAK` 但没有 text/audio 时仍必须发送
`response.spoken.delta {steps: []}`，用于无歧义表达 SPEAK；纯模板空 slot 则只发送
`reason: "slot_end"`。

整个 turn 结束：

```json
{
  "type": "response.spoken.end",
  "unit_index": 8,
  "reason": "turn_eos",
  "full_text": "好的，你继续说。"
}
```

Listen：

```json
{
  "type": "response.spoken.end",
  "unit_index": 9,
  "reason": "listen"
}
```

Active turn 未 `turn_eos` 就出现 listen，Backend 必须 RuntimeError 并关闭 Session。

## 10. Warning

只在无法无损表达时发送：

```json
{
  "type": "response.warning",
  "unit_index": 8,
  "code": "incomplete_bpe_at_stream_end",
  "message": "文本边界包含未完成 BPE，Resume 不保证可复现"
}
```

## 11. Resume

客户端保存从 `session.init` 到 available `response.unit.committed` 的完整有序历史。
服务端：

1. 校验 Unit started/committed 连续。
2. 使用 semantic begin/end/done 恢复 protocol token。
3. 按有序 steps 统计每个 text 前累计的 pending ordinary token，并用 text re-encode
   结果逐项恢复。
4. 按 Unit started 显式归属恢复 tool events。
5. 按 `non_spoken_end` 恢复 lane terminator 与 deferred feed。
6. 恢复到 checkpoint 后继续下一 Unit。

协议不依赖服务端旧 Session/KV，也不暴露 token ID。

## 12. 前端显示

- 一次 think begin→end 对应一张卡片。
- 一次 tool-call begin→done 对应一张卡片。
- Streaming 只显示 text steps。
- 同一卡片内部按 `unit_index` 显示 segment 样式。
- Full 只显示 end/done 的 `full_text`。
- 不自行添加 `<think>` / `<tool_call>`。
- 必须校验：

```text
join(all text steps) == full_text
```

Budget 只增加 Unit segment boundary，不关闭卡片。
