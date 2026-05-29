# Backend Server Network Protocol

本文描述 scheduler 与远端 inference backend server 之间的下层网络协议。它不是公网 `/v1/realtime` 协议，也不描述 UI 事件；它描述的是调度层如何初始化一个有状态 backend session、持续提交模型输入，并从 backend 读取模型输出、状态和指标。

本文当前只定义过程、原语和完成语义，不定义具体 JSON Schema。具体字段名、参数结构、音频/视频编码、metrics 字段和诊断字段见 [Message Schema](./schema.md)；交互时序与示例见 [Sequences](./sequences.md)。

## 1. Terminology

本协议有两个端：**backend** 一端，**scheduler/runtime** 一端。二者通过本协议
（WebSocket 数据通道 + unary 控制通道）通信。

| 术语 | 含义 |
|------|------|
| **backend** | 协议的服务端，持有模型、执行推理。它接收 `init`/`push`，产出 `pull` 事件流，响应 `close`。**这是本协议要新实现的一侧**（如 C++/llama-server backend）。backend 不感知公网 API、UI、调度或队列。 |
| **scheduler / runtime** | 协议的客户端，是 backend 的**上游**，负责建立 session、按节奏 `push` 输入、消费 `pull` 事件、发起 `close`。它把上层（公网 API、UI 事件、队列调度）翻译成本协议原语。**backend 不是它的一部分，也无需了解其内部结构。** 本文中 `scheduler` 与 `runtime` 指同一侧的职责，不区分。 |
| **session** | 一次有状态的推理会话，由 `init` 建立、`close` 结束，生命周期见 §4。 |
| **inference / 模型** | backend 内部实际执行生成的部分。本协议不规定其实现。 |

## 2. Scope

该协议覆盖两类有状态推理会话：

| Mode | Meaning |
|------|---------|
| `turn_based` | 一次输入产生一次 response；backend 可以按请求选择 stream 或 non-stream 推理路径，但输出都通过 pull 事件表达。 |
| `full_duplex` | 持续接收音频/视频输入，模型可以在 listen/speak 状态间切换。 |

当前版本只定义 `turn_based` 与 `full_duplex`，**不包括半双工（half-duplex）**。半双工
能力将来若实现，会在本协议的 `push` / `pull` 原语之上新增一种交互类型。

该协议把 backend 类原语映射到网络：

```text
init(params)
push(input)
pull() -> events
unary(request) -> result
```

其中：

- `init` 创建一个有状态 backend session。
- `push` 只承载模型输入。turn-based 的请求、full-duplex 的连续观察都通过 `push(input)` 进入 backend。
- `pull` 承载该 session 的单一有序事件流，包括初始化确认、模型输出、response 完成和关闭。metrics 不是独立事件，而是附着在这些下行事件上的字段（见 §6.7）。
- `unary` 承载不进入模型上下文的一次性控制请求。当前版本只定义 `close`。
- session identity 是网络层概念，由 backend 在 init 成功后分配，不接受客户端指定。

当前版本不定义 `pause` / `resume`。如果 runtime 或 scheduler 想暂停输入，只需停止调用 `push`；backend 没有新的模型观察时自然处于等待输入状态。暂停是 backend 上游的行为：可能发生在客户端采集层，或 runtime/scheduler 层（如暂停队列调度、停止向 backend 发送 input），都不是 backend 的控制命令。

## 3. Transport Layout

### 3.1 WebSocket Data Channel

WebSocket 是主数据通道。连接建立后，scheduler 必须先发起 init。backend 完成初始化后，在同一条下行事件流中确认 session 进入 active 状态。

scheduler 只有在 init 成功后，才能继续 `push(input)`。init 成功前到达的 input 是协议错误，应触发 fatal termination。

active 后，WebSocket 同时承载：

- scheduler 到 backend 的 `push(input)`。
- backend 到 scheduler 的 `pull()` 事件流。

下行事件是单连接内的有序流。scheduler 必须按接收顺序消费事件。协议不要求 backend 在下行事件中发送 `idx`、`event_seq` 或类似序号；WebSocket/TCP 已经提供单连接内可靠、有序传输。

如果后续 schema 需要上行顺序标识或业务关联标识，它们只用于上行诊断、metrics 关联或 response 关联，不改变下行事件按接收顺序消费的语义。

### 3.2 Unary Control Channel

当前 unary control 只定义 `close`，使用独立请求通道，并定位到一个已初始化 session。

`unary` 是完成语义。unary 返回成功时，backend 必须已经完成对应操作。对于 `close`，成功返回表示：

- session 已经关闭。
- 推理资源已经释放或进入不可再用状态。
- 该 session 不再可接收新的 `push` 或 `unary` 请求。

backend 可以在 WebSocket 下行事件流中额外发送 `session.closed`，用于统一日志和状态同步；但 unary 返回值本身就是完成确认，不能只表示“命令已接收”。

`close` 不走 WebSocket `push` 通道的原因是：关闭请求应避免被输入数据流、输出写队列或长时间 decode 阻塞。实现可以取消正在进行的 decode，也可以在有界时间内完成清理；如果 close 过程中发生任何异常，backend 必须把 session 视为终止并关闭 WebSocket。unary 不能返回成功，除非关闭和资源释放已经完成。

## 4. Session Lifecycle

### 4.1 Init

init 阶段完成以下事情：

1. scheduler 建立数据连接。
2. scheduler 发起 init。
3. backend 初始化有状态 session。
4. backend 确认 session 进入 active。

init 成功前：

- backend 不接受 input。
- unary control 不能生效，因为目标 session 还未建立。

init 成功后：

- WebSocket 进入 active 状态。
- scheduler 可以按 session 支持的模式 `push(input)`。
- scheduler 可以通过 unary close 关闭该 session。

### 4.2 Active

active 状态下，scheduler 持续 push 输入，backend 持续在 pull 事件流中返回模型输出和状态。

turn-based 的过程是：

1. scheduler push 一次 turn request。
2. backend 处理该 request。
3. backend 在 pull 事件流中产生 response 生命周期事件和内容 delta。
4. backend 发送 response 完成边界。

turn-based streaming / non-streaming 不是两套 backend 网络原语。二者都通过 `push(input)` 进入 backend，并通过同一条 `pull()` 事件流返回。区别在 backend 内部执行路径：

- streaming 请求可以调用底层 stream view，边生成边输出 delta。
- non-streaming 请求可以调用底层 non-stream view，在生成完成后一次性或少量输出 delta，再输出 response 完成边界。

> **设计说明（暂定，待确认）：** 当前设计倾向于把"用哪种 view"作为 backend adapter 的职责，runtime 只负责转发、记录、协议翻译和 emit，不为了得到 non-stream 语义而在 runtime 侧聚合 stream delta。此约定尚未最终确认，可能调整；见 §11 Open Issues。

full-duplex 的过程是：

1. scheduler 持续 push 音频/视频观察。
2. backend 对每个或若干观察推进模型状态。
3. 当模型决定继续听，backend 在 pull 事件流中报告 listen 状态。
4. 当模型决定说话或产生输出，backend 在 pull 事件流中产生 response 生命周期事件和内容 delta。
5. 当一次语义 response 结束，backend 发送 response 完成边界。

backend 输出是该 session 的单一有序事件流。scheduler 不应假设某个输出事件一定能和某个输入包一一对应。尤其是 C++ backend 中，文字和音频不是一一对应关系，因此协议把文字和音频作为独立的内容 delta 语义处理。

### 4.3 Close

close 使用 unary control，且必须满足完成语义：

1. scheduler 发起 close。
2. backend 终止正在进行的推理或等待有界清理点。
3. backend 关闭 session 并释放资源。
4. backend 可以 best-effort 发送 `session.closed`。
5. backend 关闭 WebSocket。
6. unary 返回成功。

close 后：

- 该 session 不再可用。
- 后续 WebSocket input 无效，因为连接已经关闭。
- backend 在 close 后遗忘该 session：对同一 session 再次 close 返回 session 不存在
  （HTTP 404）。协议层不保证 close 幂等，重复 close 的安全性由上游自己保证（见 schema §7.3）。
- 如果 WebSocket 在 close unary 返回前已经断开，unary 成功返回仍然表示 session 已关闭。

## 5. Input Semantics

### 5.1 Common Input Semantics

`push(input)` 表示把新的模型输入交给 backend。当前文档不规定 input envelope 的具体字段。

scheduler 必须保证：

- init 成功后才 push。
- 同一 session 内 input 顺序合法。
- input 形状与 session mode 匹配。
- input 内容符合 backend 能力。
- push 速率符合 backend/runtime 调度策略。

backend 必须保证：

- 按接收顺序处理同一 session 的 input。
- 不把非法输入当作可恢复业务分支。
- 一旦发现协议错误或无法继续处理，立即终止 session 并断开连接。

`push` 没有 reject 语义。对 scheduler 来说，push 成功写入连接只表示输入进入该 session 的有序输入流；backend 后续要么继续通过 pull 产出事件，要么在 fatal condition 下终止 session。

### 5.2 Turn-Based Input

turn-based input 表示一次新的对话请求。它通过 `push(input)` 进入 backend，输出通过 `pull()` 的 response 事件返回。

turn-based 可以是文本、多模态输入、带生成选项的输入等；具体 payload schema 当前不在本文定义。

### 5.3 Full-Duplex Input

full-duplex input 表示一次新的模型观察。它通常来自连续音频流，也可以带随时间采样的视频/图像观察。

full-duplex input 的具体编码、分块大小、是否允许 image-only、视频帧数量和 hint 结构，当前不在本文定义。本文只要求这些输入走同一个 `push(input)` 通道，并按 session 内顺序进入 backend。

## 6. Pull Event Semantics

backend 输出分为三类：

- session 生命周期事件。
- response 生命周期事件。
- 模型内容事件。

核心事件语义包括：

```text
session.created
response.output.delta kind=listen
response.output.delta kind=text
response.output.delta kind=audio
response.done
session.closed
```

这些名称描述语义，不代表最终 payload schema 已经稳定。

### 6.1 Session Created

`session.created` 表示 init 成功，WebSocket 进入 active 状态。该事件之后，scheduler 才能开始 push input。

### 6.2 Response Output Delta

`response.output.delta` 表示一次原子下行输出。用 `kind` 区分输出分支：

```text
kind = listen | text | audio
```

同一个 output frame 只能表达一种输出，不把文字和音频放在同一个 frame 中。

### 6.3 Listen

`response.output.delta` with `kind=listen` 表示 full-duplex 模型决定继续听用户输入，本次模型推进没有内容输出。内部 `<|listen|>` token 不应直接暴露到协议层；协议层只暴露 listen 语义。

### 6.4 Text Delta

`response.output.delta` with `kind=text` 表示一段模型文本输出。文字输出是增量语义，需要由上层按 pull 顺序拼接或展示。

### 6.5 Audio Delta

`response.output.delta` with `kind=audio` 表示一段模型音频输出。音频输出是增量语义，需要由上层按 pull 顺序播放或拼接。

音频格式、采样率、编码和分片边界由后续 schema 定义。

### 6.6 Response Done

`response.done` 表示 turn-based chat 当前 response 的全部可输出内容已经结束。

response 完成原因可以表达正常 turn 结束、达到生成上限、或 close 导致 response 被终止；具体枚举由后续 schema 定义。

错误不通过 `response.done` 作为正常 response 收尾。任何错误都是 terminal condition，见 §6.8。

full-duplex 当前不要求 `response.done`。如果需要显式表达模型切回听的边界，使用 `response.output.delta` with `kind=listen`。

### 6.7 Metrics

metrics 不通过独立请求获取，而是附着在下行事件上。实现可以在每个事件上附带 metrics，也可以只在有新观测值的事件上附带。

本文不定义 metrics 字段集合。后续 schema 可以根据 Python backend、C++ backend 和 runtime 需要统一指标名称。

### 6.8 Lifecycle And Fatal Termination

除模型内容事件外，backend 可以输出 `session.closed`。

`session.closed` 表示 session 已经结束或即将结束。它可以携带关闭原因和诊断信息，但具体字段由后续 schema 定义。

backend 协议不定义可恢复的业务错误，不定义 `error` 事件，也不定义 `input.rejected`。scheduler 与 backend 之间是受信任的内部协议：scheduler 必须保证输入顺序、输入形状、mode、容量和生命周期状态都合法。backend 收到非法输入，或 backend/engine 发生异常，都表示 scheduler 或 backend 实现存在 bug，应当 fail fast。

fatal condition 的处理规则：

- backend 必须立即终止 session。
- backend 必须关闭 WebSocket。
- backend 可以在关闭前 best-effort 发送一个 `session.closed` 事件作为诊断。
- `session.closed` 发送失败时，直接关闭连接即可。
- fatal condition 后不能继续发送模型内容事件，也不能继续接收新的 input。

## 7. Backpressure And Capacity

backend 可以限制同一 session 内同时处理的 input 数量，但不应该把容量控制暴露成 `input.rejected` 或 `busy` 业务事件。

容量与背压语义：

- scheduler 应按 backend 能力控制 push 速率。
- backend 可以内部排队，也可以让 WebSocket receive / processing loop 自然施加背压。
- 如果 scheduler 违反速率或顺序约束，这是协议 bug，不是可恢复的 reject 分支。
- backend 必须按 fatal protocol violation 终止 session 并关闭 WebSocket。

无论策略如何，backend 必须保持同一 session 内 input 的顺序语义。

session 的 mode 在 init 时确定，且在 session 生命周期内固定。turn-based input 与 full-duplex input MUST NOT 在同一 session 中混合：scheduler 只能发送与该 session mode 匹配的 input。backend 收到与 mode 不匹配的 input 时，视为 fatal protocol violation，立即断开（见 §6.8）。

## 8. Pause Semantics

当前版本不定义 backend-level `pause` / `resume`。

理由：

- 对 backend 来说，没有新的 `push(input)` 就没有新的模型观察，session 会自然停在等待输入状态。
- 运行时的暂停发生在 backend 上游：可能在客户端采集层，也可能在 runtime/scheduler 层（暂停队列调度、停止向 backend 发送 input）。
- backend 若提供 `pause/resume`，容易和输入队列、正在进行的 decode/finalize、TTS 队列产生不清晰的完成语义。

因此：

- scheduler/runtime 暂停时应停止 push。
- scheduler/runtime 恢复时继续 push。
- backend 协议只保留 terminal control：`close`。

## 9. Disconnect Semantics

WebSocket 断开后，不支持恢复。

原因：

- WebSocket 底层已有 ping/pong。
- TCP 本身提供可靠、有序传输。
- 短暂网络抖动通常不会立即导致 WebSocket 断开。
- 如果 WebSocket 已断开，继续恢复同一个 inference session 的收益较低，且需要额外的 event journal、message replay 和 exactly-once 语义。

因此：

- backend 应将该 session 标记为 failed 或 closed，并释放推理资源。
- scheduler 应丢弃该 session，并在排查原因后重新建立新 session。
- 协议不要求支持 message replay、Last-Event-ID、event journal、断线续传或 exactly-once 语义。

## 10. Non-goals

本文不定义：

- 公网 `/v1/realtime` 协议。
- UI / frontend 事件命名。
- 具体 JSON Schema。
- 具体字段名、参数结构或示例 payload。
- 音频编码、视频帧编码和压缩格式的完整枚举。
- metrics 字段集合。
- 独立 `get_metrics` 请求。
- 独立 pause/resume 控制。
- public completion / chat-completion API 形态。
- 多 session 并发调度策略。
- 断线恢复。
- 事件补发。
- 输入重放。
- exactly-once / at-least-once 语义。
- 可恢复 backend error / reject 分支。
- tool-calling。

## 11. Open Issues

以下为尚未最终确认的设计点，可能在后续版本调整。实现可参考当前倾向，但不应将其视为
稳定契约：

- **stream/non-stream 的 view 选择归属（见 §4.2 设计说明）。** 当前倾向：由 backend
  adapter 决定用 stream view 还是 non-stream view，runtime 不在自己一侧聚合 stream
  delta 来模拟 non-stream。此职责划分尚未定稿。
