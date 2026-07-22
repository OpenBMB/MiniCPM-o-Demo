# Duplex Tool Calling Protocol Draft

本文描述引入 duplex tool-call 功能之后，对 backend 与 runtime 既有协议的补充说明。
它不替代现有 session 生命周期、WebSocket 有序流、close 完成语义。
本补充继续适用于原 full-duplex 输入里的两种模态：`audio` 与 `video`；tool-call
能力是在同一条双工音视频流上新增的事件能力。

FC Duplex 可恢复 generation step batch、Unit checkpoint 与 stateless
`session.resume` 的权威规范见
[`../fc-duplex/resumable-generation-api.md`](../fc-duplex/resumable-generation-api.md)。

本文只定义最小必要协议面：

## 宏观上的注意事项
- 1.客户端不需要对 unit 的组织负责。 比如，客户端可以在任意时候发送 tool-result ，不需要把它跟某个视频帧绑定到一个 unit。  服务端足够智能地处理这些 unit 组织关系。
- 2.为了支持 resume，服务端会推送 canonical generation step batch 与 Unit checkpoint；`response.output.sp_tokens` 保留为兼容性/便利观测。普通客户端不根据这些日志事件执行业务控制，可恢复客户端必须按顺序持久化。
- 3.`temperature` / `top_p` / `top_k` 这类采样参数优先在 `session.init.payload.config.generation` 中声明为 session 级默认值；普通音视频 chunk 不携带一份完整 generation config。
- 4.tool definitions 在 init 时传入，格式跟 openai 对齐


## 1. 传输与排序

本扩展沿用现有 WebSocket 数据通道。backend 下行事件与 runtime 上行输入都在同一个
session 内按接收顺序生效。

普通业务便利事件不要求 chunk-level `seq`。可恢复事件
`response.generation.step_batch` / `response.unit.committed` 必须携带 Session 内严格单调
递增的 `event_index`；batch 内 step 还必须携带 `step_index` 与 `unit_index`。这些序号属于
resume 协议，不改变普通业务事件按 WebSocket 顺序生效的语义。

## 2. Init: tool definitions and sampling config

duplex tool-call 的工具定义在 `session.init.payload.tools` 中传入。格式采用 OpenAI-compatible
`tools` 数组；backend 负责校验、保存，并把它按当前模型模板注入模型上下文。

采样参数在 `session.init.payload.config.generation` 中传入，作为 session 级默认值。

```json
{
  "type": "session.init",
  "payload": {
    "mode": "full_duplex",
    "config": {
      "generation": {
        "decode_mode": "sampling",
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20
      }
    },
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "read_file",
          "description": "Read a UTF-8 text file from the workspace.",
          "parameters": {
            "type": "object",
            "properties": {
              "path": {
                "type": "string",
                "description": "Workspace-relative path."
              }
            },
            "required": ["path"],
            "additionalProperties": false
          }
        }
      }
    ]
  }
}
```

约束：

- `payload.tools` MAY 缺省；缺省表示本 session 不启用 tool calling。
- `payload.tools` MUST 在 `session.init` 时一次性给出。当前草案不定义 session 中途增删工具。
- backend MUST 按 tool name 建立 definition 索引，用于后续
  `response.tool_call.args.raw` 的收束和 schema-guided 参数解析。
- backend 下发 `response.tool_call.args.raw` 时，MUST 使用与模型生成时同一份 tool definition。
- runtime 执行工具时不需要也不应该重新调用 O5 SDK serializer；它消费 backend 下发的
  `response.tool_call.args.raw`。
- 如果 `tools` 非法、重名、或 backend 不支持其中某个 schema，backend 应按既有 fail-fast
  规则终止 session，而不是在运行中下发可恢复错误事件。

`payload.tools` 是协议字段；工具具体实现和权限仍属于 runtime/tool 执行层，不属于 backend
推理协议。

### 2.1 sampling config

`session.init.payload.config.generation` 用于声明采样参数的 session 级默认值。普通
`input.audio` / `input.video` / `input.standalone` chunk 不携带完整 generation config。

字段语义：

- `decode_mode`：`greedy` 或 `sampling`。当为 `greedy` 时，backend MAY 忽略
  `temperature` / `top_p` / `top_k`。
- `temperature`：sampling temperature。backend MAY 根据模型能力限制范围。
- `top_p`：nucleus sampling 阈值。省略表示 backend 默认。
- `top_k`：top-k sampling 阈值。省略表示 backend 默认。

约束：

- 这些字段是 session 级默认值。
- backend MAY 支持 session update 类事件修改这些默认值；修改只影响 update 生效后的后续
  decode，不回溯已进入模型上下文的 unit。
- `temperature` / `top_p` / `top_k` 是模型相关采样 hint。backend 不支持某个字段时，应忽略
  或 fail-fast；不能悄悄改变 tool-call 事件语义。

### 2.2 session.resume

`session.resume` 是新 WebSocket 连接的第一帧，用客户端保存的完整双向历史在相同
模型/tokenizer 下重建到一个可恢复 Unit checkpoint。服务端不依赖旧 Session/KV 或
opaque cursor。

请求、成功响应、失败码和可恢复边界见
[`FC Duplex Resumable Generation API`](../fc-duplex/resumable-generation-api.md)。
它与 legacy duplex 的 `pause` / `resume` 不同：legacy 操作复用仍驻留的服务端状态，
`session.resume` 则从客户端历史执行 stateless replay。

## 3. 上行输入

### 3.1 input.standalone

`input.standalone` 表示双工模式下用户或上游系统给模型看的独立文本输入。

```json
{
  "type": "input.standalone",
  "contents": [
    {
      "kind": "text",
      "text": "https://arxiv.org/abs/..."
    }
  ]
}
```

`contents` 预留多模态结构，但当前只实现单个 `text`。

### 3.2 input.tool_result

`input.tool_result` 表示外部工具执行完成后的结果。backend 将其作为 tool response 注入
模型上下文，并用 `tool_call_id` 与之前的模型 tool call 配对。它是非流式完整结果。

```json
{
  "type": "input.tool_result",
  "tool_call_id": "tc_xxx",
  "contents": [
    {
      "kind": "text",
      "text": "工具返回内容"
    }
  ]
}
```

约束：

- `tool_call_id` MUST 等于 backend 已经下发过的某个 `response.tool_call.args.begin`
  中的 id。
- `contents` 是完整工具结果内容列表。
- 成功与失败不在协议字段中区分；错误也作为 `contents` 返回。

### 3.3 input.tool_result.delta / done

如果工具本身支持流式输出，runtime MAY 用 `input.tool_result.delta` 把工具结果分块回填给
backend，并用 `input.tool_result.done` 收束。backend 将这些 delta 按接收顺序作为同一个
tool response 注入模型上下文。

```json
{
  "type": "input.tool_result.delta",
  "tool_call_id": "tc_xxx",
  "delta": {
    "kind": "text",
    "text": "第一段结果..."
  }
}
```

```json
{
  "type": "input.tool_result.done",
  "tool_call_id": "tc_xxx"
}
```

约束：

- `input.tool_result` 和 `input.tool_result.delta` / `done` 是互斥的两种回填方式；同一个
  `tool_call_id` MUST 只选择其中一种。
- 每个 `input.tool_result.delta` MUST 属于一个已开始但尚未 done 的 streaming tool result。
- `delta.kind` 当前只定义 `text`；后续可扩展其他内容类型。
- runtime MUST 在最后一个 delta 后发送 `input.tool_result.done`。
- backend MAY 在模型内部用 `<|tool_response_streaming|>` 等 input-event special token 标记
  流式工具结果，但这些 token 不作为 `response.output.sp_tokens` 下发。

## 4. 下行输出

### 4.1 response.output.sp_tokens

`response.output.sp_tokens` 表示模型输出侧 special-token 的只读语义观测。它主要服务于
semantic resume / 事件日志，避免把 `budget_reached`、`tts_pad`、slot/turn eos
这类信息在拼接文本时丢掉。

```json
{
  "type": "response.output.sp_tokens",
  "token": "spoken_slot_eos"
}
```

约束：

- `token` 是协议枚举，不是 tokenizer 的 raw token 文本，也不是 token id。
- 一个 special token MUST 独占一条 `response.output.sp_tokens` 事件。连续 special token
  MUST 按模型输出顺序拆成多条事件下发。
- 事件顺序就是模型输出顺序；runtime SHOULD 按接收顺序把它与 text/audio/think/tool_call
  事件一起写入 semantic resume 日志。
- `response.output.sp_tokens` 是只读信息。runtime MUST NOT 根据它执行控制行为，例如启动工具、
  取消工具、打断模型、关闭 session、强制 listen。
- backend SHOULD NOT 下发纯结构骨架 token，例如 `<unit>`、`</unit>`、slot start/end、
  image/audio placeholder。这些由 backend 在 canonicalize/replay 时按模板重建。
- input-event 侧 special token，例如 `<|tool_started|>`、`<|event_budget_reached|>`、
  `<|tool_response_streaming|>`，不属于 `response.output.sp_tokens` 的默认输出范围。
- 如果同一个模型推进步骤同时产生文本和 sp token，backend SHOULD 按模型可见顺序拆成多条事件。

当前输出侧 sp-token 枚举：

| `token` | 来源语义 | 说明 |
|------------|----------|------|
| `listen` | `<|listen|>` | 当前 unit 模型决定继续听，不产生 spoken text。可作为旧的 `response.output.delta kind=listen` 的只读日志补充。 |
| `tts_pad` | `<|tts_pad|>` | spoken lane 仍在时间结构中，但当前 unit 没有新增 spoken text。 |
| `speak` | `<|speak|>` | 当前 unit 开始/包含 spoken text；如果 text/audio delta 已明确表达发声，backend MAY 省略。 |
| `spoken_slot_eos` | `<|spoken_slot_eos|>` / 兼容目标中的 `<|chunk_eos|>` | 当前 spoken slot 结束，但 spoken turn 未结束。 |
| `spoken_turn_eos` | `<|spoken_turn_eos|>` / 兼容目标中的 `<|turn_eos|>` | 当前 spoken turn 结束。 |
| `no_action` | `<|no_action|>` | 当前 non-spoken lane 无动作。 |
| `non_spoken_eos` | `<|non_spoken_eos|>` | 当前 non-spoken decode 正常结束。 |
| `non_spoken_budget_reached` | `<|non_spoken_budget_reached|>` | 当前 non-spoken decode 因预算用尽中断，并终止当前 think/tool-call text stream；后续 Unit 如有 opener，会创建新 stream。 |
| `non_spoken_hold` | `<|non_spoken_hold|>` | 模型要求 non-spoken lane 暂停/保持。 |
| `non_spoken_abort` | `<|non_spoken_abort|>` | 模型要求中止当前 non-spoken 动作。 |

#### 4.1.1 为什么这些 token 需要保留

`response.output.sp_tokens` 的目的不是让普通应用层消费模型内部 token，而是在
semantic resume / replay 时保留单靠 text/audio/tool-call 事件无法稳定推出的状态机信息。

O5 duplex 输出不是一条单独的 assistant text stream，而是按 unit 交错展开 spoken lane 和
non-spoken lane：

```text
unit_k:
  ai_spoken_slot
  ai_non_spoken_slot
```

因此，lane 内的决策 token 和 terminator token 会影响后续模型可见序列与状态机，不能简单
依赖“有没有文本”来推理。

必要性分层如下：

- `listen`：如果同一协议流已经下发 `response.output.delta kind=listen`，则该 token 语义可由
  `kind=listen` 推出，属于冗余但有用的只读 trace。保留它可以让 resume 日志更接近模型输出序列。
- `speak`：如果同一 unit 已经有 spoken text/audio delta，通常可以推出 `<|speak|>`；但在
  边界异常、空 spoken slice、或仅做 exact trace 时，显式记录仍有价值。backend MAY 在 text/audio
  delta 已经明确表达发声时省略。
- `tts_pad`：不能由“本 unit 没有 spoken text”稳定推出。它表示当前 unit 仍在同一个 spoken turn
  timeline 内，但没有新增 spoken text token；这与 `listen`、异常无输出、或被省略的空 slice 不同。
- `spoken_slot_eos`：表示当前 unit 内 spoken slot 结束。若没有显式 per-unit spoken slot boundary，
  resume 时很难无歧义重建该 terminator 的插入位置。
- `spoken_turn_eos`：表示整个 spoken turn 结束，而不只是当前 slot 结束。它决定后续 unit 是继续
  同一个 spoken turn、进入 `tts_pad`、还是切回 listen，因此建议保留。
- `no_action`：表示模型明确决定当前 non-spoken lane 无动作。它不同于 backend 没有运行
  non-spoken decode、事件被省略、或异常无输出。
- `non_spoken_eos`：表示 non-spoken lane 正常结束。它不是某个 `think.end` 或
  `tool_call.args.end` 的简单同义词；span 可以闭合，但 lane 还需要自己的结束状态。
- `non_spoken_budget_reached`：表示当前 Unit 因预算耗尽而中断，并终止当前
  think/tool-call text stream。模型 token、slot end 与 Unit end 可以延迟到下一 Unit
  prefill 前进入 KV；canonical history 必须记录该 deferred feed 时序。
- `non_spoken_hold`：表示模型要求 non-spoken lane 暂停/保持。它是显式状态，不等价于没有新 delta。
- `non_spoken_abort`：表示模型中止当前 non-spoken 动作。它和 parser error、runtime cancel、
  budget stop 都不同；tool-call 场景下还需要与 `response.tool_call.abort` 的语义对齐。

如果未来协议新增更高层的语义事件，例如：

```text
response.spoken.done reason=slot_eos|turn_eos
response.non_spoken.done reason=no_action|eos|budget_reached|hold|abort
```

则这些事件可以承载同样的 resume 信息，`response.output.sp_tokens` 可以进一步降级为可选
debug / exact-trace 字段。但无论用哪种表示，上述 lane 边界和终止原因本身不能丢失。

纯结构骨架 token（`<unit>`、slot start/end、image/audio placeholder 等）不在此列：它们可由
backend 根据模板、unit 输入和模态信息确定性重建，默认不下发。

示例：

```json
{ "type": "response.output.sp_tokens", "token": "spoken_slot_eos" }
```

```json
{ "type": "response.output.sp_tokens", "token": "non_spoken_budget_reached" }
```

```json
{ "type": "response.output.sp_tokens", "token": "spoken_turn_eos" }
```

#### 4.1.2 Canonical resumable generation

`response.output.sp_tokens` 是历史兼容和业务便利事件。精确 resume 使用
`response.generation.step_batch` 保存逐 token generation step，并用
`response.unit.committed` 声明 Unit checkpoint 是否可恢复。

Batch 内通过 `text_pending` / `text_delta` / `protocol` 三种 discriminated variant
同时保存：

- 没有安全 Unicode 输出的 step；
- 可逐项 re-encode 的 safe text delta 及其有序 `source_step_indices`；
- protocol structural semantic key。

完整字段和 batching 约束见
[`FC Duplex Resumable Generation API`](../fc-duplex/resumable-generation-api.md)。

#### 4.1.3 Stream boundary warning

Think/tool-call 在 matching end 或 `non_spoken_budget_reached` 终止时，如果增量 decoder
仍有 incomplete BPE，backend 下发：

```json
{
  "type": "response.warning",
  "code": "incomplete_bpe_at_stream_end",
  "stream_id": "think_2",
  "reason": "budget_reached",
  "message": "文本边界包含未完成 BPE，公共 API 历史无法保证精确复现"
}
```

Warning 不终止 live Session，但后续 resume 不保证可用。Backend 不得用替换字符补齐。

### 4.2 response.think

`think` 是模型生成的非口语思考文本流。它需要 begin/end 边界，但不需要 id 或 seq。

```json
{ "type": "response.think.begin" }
```

```json
{ "type": "response.think.delta", "delta": "我需要先判断..." }
```

```json
{ "type": "response.think.end" }
```

约束：

- 同一 session 内同一时刻最多一个 active think。
- `response.think.delta` 的文本按 WebSocket 接收顺序拼接。
- `response.think.end` 表示当前 think span 正常闭合。

### 4.3 response.tool_call.args

`response.tool_call.args.*` 表示模型正在生成一个工具调用参数流。`tool_call_id` 由 backend
分配并贴到所有相关事件上。这里保留 chunk，同时在结束后给出 backend 已解析好的
raw tool call。

```json
{
  "type": "response.tool_call.args.begin",
  "tool_call_id": "tc_xxx"
}
```

```json
{
  "type": "response.tool_call.args.delta",
  "tool_call_id": "tc_xxx",
  "delta": "<name>write_file</name>"
}
```

```json
{
  "type": "response.tool_call.args.end",
  "tool_call_id": "tc_xxx"
}
```

```json
{
  "type": "response.tool_call.args.raw",
  "tool_call_id": "tc_xxx",
  "raw": {
    "type": "function_call",
    "name": "write_file",
    "arguments": "{\"path\":\"a.txt\",\"content\":\"hello\"}"
  }
}
```

如果模型采样没有产生合法的 tool-call 序列，backend 仍然下发同一个
`response.tool_call.args.raw` 事件，但 `raw` 只包含 `error` 字段，值为解析失败的 reason
字符串：

```json
{
  "type": "response.tool_call.args.raw",
  "tool_call_id": "tc_xxx",
  "raw": {
    "error": "failed to parse tool call: missing required argument `path`"
  }
}
```

约束：

- `tool_call_id` MUST 由 backend 分配。
- runtime MUST 按接收顺序拼接同一个 `tool_call_id` 的 `delta`。
- `response.tool_call.args.end` 表示参数流闭合。
- `response.tool_call.args.raw` 表示 backend 已完成收束和解析后的 tool call 结果，runtime
  MUST 以它作为执行工具的依据。
- 当 `raw.error` 存在时，该 tool call 解析失败，runtime MUST NOT 执行该工具，也不需要回填
  `input.tool_result`。
- runtime MAY 拼接 `delta` 用于展示、日志或诊断，但执行工具时不需要、也不应该再调用 SDK
  serializer 解析参数流。
- `raw` 内 MUST NOT 重复携带 `id`、`call_id` 或 `tool_call_id`；事件外层的
  `tool_call_id` 是 runtime 回填结果时使用的唯一关联 id。
- 当前草案不定义 chunk-level `seq`。

`tool_call_id` 是 backend-worker wire 层对象 id。模型 token 流内部是否显式包含 id，
不由本文规定。

### 4.4 response.tool_call.abort

模型可能在参数流完成前放弃一个工具调用。backend 用 `response.tool_call.abort` 通知
runtime。

```json
{
  "type": "response.tool_call.abort",
  "tool_call_id": "tc_xxx"
}
```

约束：

- 如果 abort 发生在 `args.end` 之前，runtime MUST NOT 执行该 tool call。
- 如果 runtime 已经在 `args.end` 后开始执行，abort 表示取消请求；具体工具能否取消由
  runtime/tool 实现决定。
- 被 abort 的 tool call 不要求 runtime 回传 `input.tool_result`。

## 5. 最小生命周期

一次普通工具调用的下行与上行顺序如下：

```text
backend -> runtime: response.think.begin
backend -> runtime: response.think.delta*
backend -> runtime: response.think.end

backend -> runtime: response.tool_call.args.begin { tool_call_id }
backend -> runtime: response.tool_call.args.delta*
backend -> runtime: response.tool_call.args.end { tool_call_id }
backend -> runtime: response.tool_call.args.raw { tool_call_id, raw }

runtime -> backend: input.tool_result { tool_call_id, contents }
```

一次流式工具结果回填：

```text
backend -> runtime: response.tool_call.args.begin { tool_call_id }
backend -> runtime: response.tool_call.args.delta*
backend -> runtime: response.tool_call.args.end { tool_call_id }
backend -> runtime: response.tool_call.args.raw { tool_call_id, raw }

runtime -> backend: input.tool_result.delta { tool_call_id, delta }
runtime -> backend: input.tool_result.delta { tool_call_id, delta }
runtime -> backend: input.tool_result.done { tool_call_id }
```

一次被放弃的工具调用：

```text
backend -> runtime: response.tool_call.args.begin { tool_call_id }
backend -> runtime: response.tool_call.args.delta*
backend -> runtime: response.tool_call.abort { tool_call_id }
```

独立文本输入与工具结果可以独立进入同一 session：

```text
runtime -> backend: input.standalone
runtime -> backend: input.tool_result
runtime -> backend: input.tool_result.delta / input.tool_result.done
```

backend 负责把这些上行事件按模型内部 unit/text input slot 策略注入上下文。本文不规定
具体 unit 分配算法。

## 7. 可选 debug 事件

backend MAY 下发 `response.debug` 事件，用于调试、观测、性能估算、运行时状态展示或实验性
trace。协议只定义事件 envelope，不定义 `debug` 内部结构。

```json
{
  "type": "response.debug",
  "session_id": "sess_xxx",
  "response_id": "resp_xxx",
  "input_id": "input_xxx",
  "debug": {
    "estimated_max_budget_1s": 37,
    "used": 13
  }
}
```

约束：

- `type` MUST 为 `response.debug`。
- `debug` MUST 是一个 JSON object。
- 协议不规定 `debug` 内部字段；字段语义完全由 backend/runtime 当前实现自定义。
- runtime MAY 忽略任意 `response.debug` 事件。
- runtime MUST NOT 依赖 `response.debug` 恢复会话语义、执行工具、控制模型或重建模型状态。
- backend MAY 省略、延迟、聚合、采样或裁剪 `response.debug` 事件。

## 8. 内部 token 观测

公共 FC Duplex WebSocket API 不返回 token ID、vocab piece 或原始 bytes。
`token_observations` 不属于公共 wire schema。`session.created.resume` 只返回
tokenizer target 与稳定 fingerprint，用于拒绝跨 tokenizer resume；它不包含任何生成
token 内容。

模型实现可以在服务端内部 trace 中记录 token ID、bytes、logprob 或 top-logprobs，用于
调试、审计和测试；这些内部字段：

- 不发送给客户端；
- 不参与业务控制；
- 不作为 semantic resume 输入；
- 不承诺跨模型/tokenizer 兼容。

公共 resume 只依赖
[`response.generation.step_batch`](../fc-duplex/resumable-generation-api.md) 中的
step/Unit provenance、safe text delta 与 protocol semantic key。工具执行仍只使用
`response.tool_call.args.raw` 的结构化结果。
