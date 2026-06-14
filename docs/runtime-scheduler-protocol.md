# MiniCPM-o Inference Backend Protocol

本文档分为两层：

1. `InferenceBackend Interface Contract`：单个有状态 backend 实例的类协议。
2. `Runtime / Transport Binding`：runtime 如何调用 backend，并把类协议映射到网络、IPC 或跨进程调用。

当前 draft 只定义第一层。第一层假设调用方已经拿到一个可靠的 backend 对象；连接管理、重连、heartbeat、session id 映射、消息重放、数据落盘和对外协议转换属于第二层。

## 1. InferenceBackend Interface Contract

### 1.1 Scope

一个 `InferenceBackend` 实例表示一个有状态推理会话。它持有 prompt、配置、KV/cache、流式状态、duplex 状态和 metrics。调用方通过 `push(message)` 提交模型输入和运行控制，并通过 `pull()` 读取文本、音频、状态、指标和错误事件。

这一层不要求 `session_id`。在 backend 类协议中，对象身份就是会话身份；如果需要跨连接恢复或远程路由，`session_id` 由第二层 runtime / transport binding 定义。

上层 runtime 是 backend 的调用者。它可以负责转发输入、连接外部 transport、记录输入输出、落盘媒体文件、聚合 metrics 和做协议转换，但不应该重新定义 backend 内部的 listen/speak、pause/resume、prefill/decode pipeline 等执行语义。

同一套接口覆盖两类 mode：

| Mode | 语义 |
|---------|------|
| `turn_based` | 一次或多次输入后生成一个完整 response，可流式输出文本/音频 |
| `full_duplex` | 持续接收音频/视频输入，模型可在 listen/speak 状态间切换 |

### 1.2 Terminology

| Term | Meaning |
|------|---------|
| `backend` | 驱动单个有状态推理会话的对象，可由 PyTorch、C++ 或其他引擎实现 |
| `runtime` | backend 的上层调用者，负责转发、记录、transport 映射和运行期编排 |
| `input bundle` | 一次提交给 backend 的用户侧输入，可包含文本、音频、图像、视频帧和输入级配置 |
| `control command` | 不作为模型观察内容的运行命令，例如暂停、恢复、关闭、查询指标 |
| `event` | backend 对外发出的可观察事实，例如文本增量、音频增量、metrics、done、error |
| `response` | backend 针对输入产生的一段模型输出，可由多个 delta event 组成 |
| `turn` | 模型输出的一段语义轮次；在 full-duplex 中，turn 可能被 listen/speak 状态切分 |

### 1.3 Class Surface

Backend 暴露四个核心原语。`push` / `pull` 用于流式或持续交互；`unary` 用于一次性应答。

```text
init(params) -> void
push(message) -> void
pull() -> Iterator[BackendEvent]
unary(request) -> UnaryResult
```

参考接口：

```python
from collections.abc import Iterator
from typing import Any, Literal, NotRequired, TypedDict

class InferenceBackend:
    def init(self, params: BackendInitParams) -> None: ...
    def push(self, message: BackendMessage) -> None: ...
    def pull(self) -> Iterator[BackendEvent]: ...
    def unary(self, request: UnaryRequest) -> UnaryResult: ...

BackendMode = Literal["turn_based", "full_duplex"]
BackendMessageType = Literal["input", "control", "close"]
BackendEventType = Literal[
    "backend.initialized",
    "backend.state",
    "backend.closed",
    "input.accepted",
    "input.rejected",
    "input.committed",
    "response.started",
    "response.text.delta",
    "response.audio.delta",
    "response.output.delta",
    "response.speak",
    "response.done",
    "metrics.snapshot",
    "metrics.frame",
    "metrics.response",
    "error",
]

class SamplingOptions(TypedDict, total=False):
    temperature: float
    top_p: float
    top_k: int

class TurnBasedGenerationOptions(SamplingOptions, total=False):
    max_new_tokens: int
    min_new_tokens: int
    do_sample: bool
    max_inp_length: int
    length_penalty: float

class FullDuplexGenerationOptions(SamplingOptions, total=False):
    max_new_speak_tokens_per_chunk: int
    decode_mode: str
    length_penalty: float
    text_repetition_penalty: float
    text_repetition_window_size: int
    listen_prob_scale: float
    listen_top_k: int
    force_listen_count: int

class VisionOptions(TypedDict, total=False):
    max_slice_nums: int
    use_image_id: bool

class VoiceRef(TypedDict, total=False):
    ref_audio: str | bytes
    ref_audio_path: str
    ref_audio_data: str
    ref_audio_max_ms: int

class TtsSamplingOptions(TypedDict, total=False):
    top_p: float
    min_p: float
    top_k: int
    repetition_penalty: float
    temperature: float
    win_size: int
    tau_r: float

class TtsRequestOptions(TypedDict, total=False):
    enabled: bool
    mode: str
    voice: VoiceRef
    output_path: str
    language: str
    sampling: TtsSamplingOptions

class FullDuplexTtsOptions(TypedDict, total=False):
    temperature: float

class StreamTimingOptions(TypedDict, total=False):
    chunk_ms: int
    sample_rate: int

class DuplexOptions(TypedDict, total=False):
    generate_audio: bool
    ls_mode: str
    generation: FullDuplexGenerationOptions
    tts: FullDuplexTtsOptions
    timing: StreamTimingOptions

class BackendResources(TypedDict, total=False):
    model_path: str
    pt_path: str
    device: str
    gpu_id: int
    attn_implementation: Literal["auto", "flash_attention_2", "sdpa", "eager"]
    compile: bool
    preload_both_tts: bool
    default_voice: VoiceRef

class BackendInitCommon(TypedDict, total=False):
    resources: BackendResources
    metadata: dict[str, Any]

class SessionDefaultsCommon(TypedDict, total=False):
    system_prompt: str
    voice: VoiceRef

class TurnBasedDefaults(SessionDefaultsCommon, total=False):
    pass

class FullDuplexDefaults(SessionDefaultsCommon, total=False):
    tts_voice: VoiceRef
    vision: VisionOptions
    deferred_finalize: bool

class TurnBasedInitParams(BackendInitCommon, total=False):
    mode: Literal["turn_based"]
    defaults: TurnBasedDefaults
    chat_vocoder: Literal["token2wav", "cosyvoice2"]

class FullDuplexInitParams(BackendInitCommon, total=False):
    mode: Literal["full_duplex"]
    defaults: FullDuplexDefaults
    duplex: DuplexOptions
    pause_timeout_s: float

BackendInitParams = TurnBasedInitParams | FullDuplexInitParams

class TextContent(TypedDict):
    type: Literal["text"]
    text: str

class AudioContent(TypedDict):
    type: Literal["audio"]
    format: str
    sample_rate: int
    data: bytes | str

class ImageContent(TypedDict):
    type: Literal["image"]
    format: str
    data: bytes | str

class VideoFramesContent(TypedDict):
    type: Literal["video_frames"]
    format: str
    frames: list[bytes | str]

class MessagesContent(TypedDict):
    type: Literal["messages"]
    messages: list[dict[str, Any]]

InputContent = TextContent | AudioContent | ImageContent | VideoFramesContent | MessagesContent

class ConversationMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str | list[InputContent]

class InputBundle(TypedDict, total=False):
    input_id: str
    timestamp_ms: int
    role: Literal["user", "system"]
    content: list[InputContent]
    hints: dict[str, Any]

class BackendControl(TypedDict, total=False):
    command_id: str
    type: Literal["backend.pause", "backend.resume", "metrics.get", "backend.close"]
    payload: dict[str, Any]

class TurnBasedFlags(TypedDict, total=False):
    use_tts_template: bool
    omni_mode: bool
    enable_thinking: bool
    return_prompt: bool

class TurnBasedUnaryRequest(TypedDict, total=False):
    type: Literal["turn_based"]
    request_id: str
    timestamp_ms: int
    messages: list[ConversationMessage]
    generation: TurnBasedGenerationOptions
    vision: VisionOptions
    tts: TtsRequestOptions
    flags: TurnBasedFlags
    metadata: dict[str, Any]

class TurnBasedUnaryResult(TypedDict, total=False):
    type: Literal["turn_based"]
    request_id: str
    text: str
    audio: bytes | str
    usage: dict[str, Any]
    metrics: dict[str, Any]

UnaryRequest = TurnBasedUnaryRequest
UnaryResult = TurnBasedUnaryResult

class BackendMessage(TypedDict):
    message_id: str
    type: BackendMessageType
    timestamp_ms: int
    payload: InputBundle | BackendControl | dict[str, Any]

class BackendEvent(TypedDict, total=False):
    version: Literal["backend.class.v1"]
    event_id: str
    type: BackendEventType
    message_id: NotRequired[str]
    response_id: NotRequired[str]
    input_id: NotRequired[str]
    timestamp_ms: int
    payload: dict[str, Any]
```

### 1.4 `init(params)`

创建或重置 backend 内部推理状态。

`params` 使用 mode-specific union。类型上可以理解为：

```text
BackendInitParams =
  BackendInitCommon
  & (TurnBasedInitParams | FullDuplexInitParams)
```

其中 `resources` 和 `metadata` 是 backend init 的交集部分；`defaults` 也有 `SessionDefaultsCommon` 交集，但每个 mode 可以扩展自己的 defaults。`chat_vocoder`、`duplex`、`pause_timeout_s` 是不同 mode 的差异部分。

| Field | Meaning |
|-------|---------|
| `mode` | 选择 backend 交互语义：`turn_based` 或 `full_duplex` |
| `resources` | backend 初始化时绑定的资源，例如模型路径、设备、attention 实现、默认参考音频 |
| `defaults` | backend 初始化后的默认上下文；包含 common 的 `system_prompt` / `voice`，也可以有 mode-specific 字段 |
| `chat_vocoder` | `turn_based` 专属；Chat 非流式 vocoder 选择 |
| `duplex` | `full_duplex` 专属；listen/speak、采样、force-listen、chunk 参数 |
| `pause_timeout_s` | `full_duplex` 专属；暂停超时 |

`resources` 通常不能随请求改变。Turn-based 的 generation、vision、TTS 等选项属于当次 `unary` 请求，不属于 backend init defaults。Backend 初始化后不提供中途更新配置的 control；如需更换 `resources` 或 mode-specific 初始化字段，应创建或重置 backend。

部署层参数不属于 `BackendInitParams`，例如 gateway port、worker port、队列大小、HTTP timeout、录制开关和数据保留策略。这些由 runtime / transport 层持有。

初始化完成后，backend 通过 `pull()` 输出 `backend.initialized` 事件。事件中的 `BackendInfo` 至少包含：

```json
{
  "mode": "full_duplex",
  "resources": {
    "model_path": "/models/MiniCPM-o",
    "device": "cuda:0",
    "attn_implementation": "flash_attention_2",
    "default_voice": {
      "ref_audio_path": "assets/ref_audio/ref_minicpm_signature.wav"
    }
  },
  "defaults": {
    "system_prompt": "You are a helpful assistant."
  },
  "state": "initialized",
  "capabilities": {
    "text_output": true,
    "audio_output": true,
    "vision_input": true,
    "metrics": true
  }
}
```

### 1.5 `push(message)`

向 backend 提交输入消息。`push` 不同步返回模型结果，也不通过返回值报告接收或拒绝；相关结果通过后续 `pull()` 读取。

`message` 使用统一 envelope：

```json
{
  "message_id": "msg_001",
  "type": "input",
  "timestamp_ms": 12345,
  "payload": {}
}
```

消息类型：

| Type | Payload | Meaning |
|------|---------|---------|
| `input` | `InputBundle` | 新的模型观察内容，输出通过普通 event stream 表达 |
| `control` | `BackendControl` | 不进入模型上下文的运行控制 |
| `close` | `reason` | 关闭 backend |

#### Input Message

向 backend 提交用户侧输入。该调用表达“有新的模型观察内容进入 backend”，不表达“立刻 decode”或“立刻产生 response”。

`InputBundle` 是可变的 multimodal bundle：

```json
{
  "input_id": "in_001",
  "timestamp_ms": 12345,
  "role": "user",
  "content": [
    {
      "type": "audio",
      "format": "pcm_f32",
      "sample_rate": 16000,
      "data": "base64..."
    },
    {
      "type": "image",
      "format": "jpeg",
      "data": "base64..."
    }
  ],
  "hints": {
    "force_listen": false,
    "max_slice_nums": 2
  }
}
```

一个 `input bundle` 可以包含：

| Content Type | Meaning |
|--------------|---------|
| `text` | 用户文本 |
| `audio` | 一段音频输入 |
| `image` | 单张图像 |
| `video_frames` | 一组视频帧 |
| `messages` | turn-based chat 消息列表 |

`content` 的粒度由调用方和 backend mode 决定。例如 full-duplex 可以每秒 push 一组 `{audio + frames}`，也可以把音频和图像拆成多个 bundle。接口只要求同一个 backend 实例内 `input_id` 可追踪、输入顺序可观察。

#### Control Message

向 backend 提交运行控制命令。Control 不进入模型上下文。

```json
{
  "command_id": "cmd_001",
  "type": "metrics.get",
  "payload": {
    "scope": "backend"
  }
}
```

核心 control 类型：

| Type | Meaning |
|------|---------|
| `backend.pause` | 暂停处理新的模型输入 |
| `backend.resume` | 恢复处理输入 |
| `metrics.get` | 请求一次 metrics snapshot |
| `backend.close` | 关闭 backend |

Backend 不通过返回值报告 control 结果。对于会改变 backend 状态的控制命令，backend 应产出对应 state event；对于 `metrics.get`，backend 应通过 `pull()` 产出 `metrics.snapshot`。

### 1.6 `pull()`

`pull` 承载 backend 的全部输出。各类 chunk 的参数签名定义在 `pull()` 返回的 event 类型中，而不是定义在 `push` 的返回值中。

`pull()` 表示同一个 backend 实例的单一有序输出流。Backend 可以在 `push()` 返回前就产生 event，因此调用方不能依赖“先 `push`，再开始 `pull`”这种顺序来避免丢事件。Runtime 若需要提供一问一答 API，必须先建立持续消费 `pull()` 的 pump，并在 `push(input)` 之前注册用于聚合该 `input_id` / `response_id` 的 collector。

常用输出签名：

```text
pull() -> backend.initialized(info: BackendInfo)
pull() -> backend.state(state: BackendState, reason?: string)
pull() -> backend.closed(reason?: string)

pull() -> input.accepted(input_id: string)
pull() -> input.rejected(input_id: string, error: BackendError)
pull() -> input.committed(input_id: string, metrics?: BackendMetrics)

pull() -> response.started(response_id: string, input_id?: string)
pull() -> response.text.delta(response_id: string, text: string)
pull() -> response.audio.delta(response_id: string, audio: bytes | string, format: string, sample_rate: int)
pull() -> response.output.delta(kind=listen, reason?: string)
pull() -> response.speak(reason?: string)
pull() -> response.done(response_id: string, reason: ResponseDoneReason, usage?: Usage)

pull() -> metrics.snapshot(metrics: BackendMetrics)
pull() -> metrics.frame(input_id: string, metrics: BackendMetrics)
pull() -> metrics.response(response_id: string, metrics: BackendMetrics)

pull() -> error(error: BackendError)
```

### 1.7 Unary Exchange Semantics

一次性应答使用独立的 `unary(request)` 原语表达。它类似 backend 内部的 unary RPC：一次调用进入 backend，一次返回完整结果。Turn-based chat 是当前最明确的使用场景，但 `unary` 本身不绑定 turn-based；后续可以承载其他“一次请求、一次结果”的 backend 能力。

```text
init(params)
push(message)
pull() -> Iterator[BackendEvent]
unary(request) -> UnaryResult
```

`unary` 与 `push` / `pull` 的区别：

| Primitive | Semantics |
|-----------|-----------|
| `push(input)` | 提交流式或持续交互输入，输出通过 `pull()` 的 `response.*` event 表达 |
| `unary(request)` | 提交一次性请求，直接返回聚合后的 `UnaryResult` |

`UnaryRequest` 是 sum type，而不是带 `payload` 的通用 envelope。每一种 unary 语义都应该定义自己的 request/result 结构：

```python
UnaryRequest = TurnBasedUnaryRequest  # | OtherUnaryRequest | ...
UnaryResult = TurnBasedUnaryResult    # | OtherUnaryResult | ...
```

当前已定义的 variant 是 `turn_based`：

```json
{
  "request_id": "unary_001",
  "type": "turn_based",
  "timestamp_ms": 12345,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "介绍一下这辆车"
        }
      ]
    }
  ],
  "generation": {
    "max_new_tokens": 256,
    "length_penalty": 1.1
  },
  "vision": {
    "max_slice_nums": 1
  },
  "tts": {
    "enabled": false,
    "voice": {
      "ref_audio_data": "base64..."
    }
  },
  "flags": {
    "use_tts_template": false,
    "omni_mode": false,
    "enable_thinking": false,
    "return_prompt": false
  }
}
```

对应的 `UnaryResult`：

```json
{
  "request_id": "unary_001",
  "type": "turn_based",
  "text": "这是一辆...",
  "audio": null,
  "usage": {
    "output_tokens": 32
  },
  "metrics": {
    "prefill_ms": 18.2,
    "generate_ms": 63.5
  }
}
```

Backend 可以在 `unary` 内部复用同一套 prefill/decode/TTS pipeline，也可以内部流式生成；但这些中间过程不暴露为该调用的协议结果。`unary` 的外部语义是一次调用对应一次完整结果。后续如果增加其它一次性能力，应新增 union variant，而不是往通用 `payload` 里塞字段。

Runtime 若要对外提供 `ask()` 这类 API，可以直接调用 backend 的 `unary`：

```python
class Runtime:
    async def ask(self, request: UnaryRequest) -> UnaryResult:
        result = await to_thread(self.backend.unary, request)
        await self.record_unary(request, result)
        return result
```

这样做的含义是：流式交互使用 `push` + `pull`；一次性应答使用 `unary`。二者共享同一个 backend 对象和初始化状态，但不是同一个输出形态。

### 1.8 Event Model

Backend event 使用统一 envelope：

```json
{
  "version": "backend.class.v1",
  "event_id": "evt_0001",
  "type": "response.text.delta",
  "response_id": "resp_001",
  "input_id": "in_001",
  "timestamp_ms": 12356,
  "payload": {
    "text": "这是一辆..."
  }
}
```

字段语义：

| Field | Meaning |
|-------|---------|
| `version` | backend 类协议版本 |
| `event_id` | event 标识；同一 backend 实例内建议单调递增 |
| `type` | event 类型 |
| `message_id` | event 关联的 pushed message，可选 |
| `response_id` | event 所属 response，可选 |
| `input_id` | event 关联的 input bundle，可选 |
| `timestamp_ms` | backend 产生 event 的时间戳 |
| `payload` | event 载荷 |

#### Backend Events

| Type | Payload | Meaning |
|------|---------|---------|
| `backend.initialized` | `BackendInfo` | backend 初始化完成 |
| `backend.state` | `state`, `reason` | backend 状态变化 |
| `backend.closed` | `reason` | backend 已关闭 |

#### Input Events

| Type | Payload | Meaning |
|------|---------|---------|
| `input.accepted` | `input_id` | backend 已接收输入 |
| `input.rejected` | `input_id`, `error` | backend 拒绝输入 |
| `input.committed` | `input_id`, `metrics` | 输入已进入模型上下文或 backend 状态 |

`input.committed` 表示输入已经被 backend 接纳到推理状态中。对某些实现，它可能对应 prefill 完成；对另一些实现，它可能只是 pipeline 中的可观察提交点。

#### Response Events

| Type | Payload | Meaning |
|------|---------|---------|
| `response.started` | `response_id`, `input_id` | 开始产生 response |
| `response.text.delta` | `text` | 文本增量 |
| `response.audio.delta` | `audio`, `format`, `sample_rate` | 音频增量 |
| `response.output.delta` | `kind=listen`, `reason` | 模型选择继续听 |
| `response.speak` | `reason` | 模型进入说话状态 |
| `response.done` | `reason`, `usage` | response 结束 |

`response.done.reason` 建议使用稳定枚举：

| Reason | Meaning |
|--------|---------|
| `turn_end` | 当前语义轮次结束 |
| `listen` | 模型切换到听状态 |
| `max_tokens` | 达到生成上限 |
| `error` | 由于错误结束 |

#### Metrics Events

| Type | Payload | Meaning |
|------|---------|---------|
| `metrics.snapshot` | `BackendMetrics` | 当前 metrics 快照 |
| `metrics.frame` | `BackendMetrics`, `input_id` | 某个 input/frame 的处理指标 |
| `metrics.response` | `BackendMetrics`, `response_id` | 某个 response 的生成指标 |

常用 metrics 字段：

```json
{
  "backend": "cpp",
  "kv_cache_length": 1024,
  "prefill_ms": 18.2,
  "generate_ms": 63.5,
  "n_tokens": 12,
  "n_tts_tokens": 50,
  "vision_slices": 2
}
```

#### Error Events

错误通过 `error` event 表达：

```json
{
  "type": "error",
  "payload": {
    "code": "invalid_state",
    "message": "backend is paused",
    "scope": "backend",
    "terminal": false
  }
}
```

字段：

| Field | Meaning |
|-------|---------|
| `code` | 稳定错误码 |
| `message` | 人类可读说明 |
| `scope` | `input` / `response` / `backend` |
| `terminal` | 是否终止对应 scope |

建议错误码：

| Code | Meaning |
|------|---------|
| `invalid_input` | 输入内容或格式不合法 |
| `invalid_state` | 当前状态不接受该操作 |
| `unsupported` | backend 不支持该能力或字段 |
| `context_full` | 上下文容量不足 |
| `backend_busy` | backend 当前无法接收更多工作 |
| `backend_error` | backend 内部错误 |
| `engine_error` | 推理引擎错误 |

### 1.9 Backend State Machine

```text
NEW
  |
  | init
  v
ACTIVE
  |  ^
  |  | backend.resume
  | backend.pause
  v
PAUSED
  |
  | close / terminal error
  v
CLOSED
```

状态语义：

| State | Meaning |
|-------|---------|
| `NEW` | backend 对象已创建，尚未初始化 |
| `ACTIVE` | backend 正常接收 input/control 并产生 event |
| `PAUSED` | backend 暂停处理新的模型输入 |
| `CLOSED` | backend 已结束 |

Backend 在 `ACTIVE` 内部可以继续细分执行阶段。推荐表达为带参数的状态，而不是把组合压平成多个枚举值：

```text
ActivePhase =
  | Listening
  | Ingesting(InputModalities)
  | Responding(OutputModalities)

OutputModalities = {
  text: boolean,
  audio: boolean
}
```

例如同时生成文本和音频时是 `Responding({text: true, audio: true})`，而不是 `RespondingTextAudio`。

### 1.10 Ordering Semantics

Backend 应保证同一实例内以下顺序可观察：

- `backend.initialized` 早于模型输出事件。
- 对同一 response，`response.*.delta` 早于 `response.done`。
- `backend.closed` 之后不再产生新的文本、音频或 response event。
- 如果 backend 发出 `input.committed`，它应早于由该 input 触发的 `response.done`。

文本和音频不要求严格同步。Full-duplex 中，文本通常早于音频，音频也可能由独立 TTS pipeline 延迟产生。

### 1.11 Backend Modes

#### Turn-Based Mode

Turn-based mode 适用于一次请求产生一次 response 的场景。

典型流程：

```text
init(mode=turn_based)
push(input: messages/text/audio/image)
  -> input.accepted
  -> input.committed
  -> response.started
  -> response.text.delta*
  -> response.audio.delta*
  -> response.done
push(close)
```

一次性应答流程：

```text
init(mode=turn_based)
unary(TurnBasedUnaryRequest{messages, generation?, vision?, tts?, flags?})
  -> TurnBasedUnaryResult{text, audio?, usage?, metrics?}
push(close)
```

Turn-based backend 可以在 `response.done` 后继续接收下一次 `push(input)`，也可以选择单请求完成后关闭。具体生命周期由调用方和 backend 配置决定。

#### Full-Duplex Mode

Full-duplex mode 适用于持续输入和模型 listen/speak 切换。它复用 `SamplingOptions`、`VisionOptions` 和 `VoiceRef` 这些 leaf type，但组合方式不同：generation / listen-speak 策略在 `duplex` 中，session 默认视觉和参考音频在 `defaults` 中，逐 chunk 输入仍可带 `max_slice_nums` 与 `force_listen`。

典型流程：

```text
init(mode=full_duplex, defaults={system_prompt, voice, tts_voice, vision}, duplex={generation, tts, timing})
push(input: audio + frames)
push(input: audio + frames)
  -> response.output.delta(kind=listen)
push(input: audio + frames)
  -> response.started
  -> response.text.delta*
  -> response.audio.delta*
  -> response.done(reason=listen | turn_end)
push(close)
```

Full-duplex backend 可以内部实现 prefill/decode pipeline、listen/speak 状态机、TTS 队列、KV 滑窗和音频输出缓冲。对外语义由 input、control 和 event 表达。

### 1.12 Conformance Checks

一个 backend 若声称支持 `backend.class.v1`，至少应满足：

- `init` 成功后能通过 `pull()` 产出 `backend.initialized`。
- `push(input)` 不把 control 字段当作模型输入。
- `unary(request)` 返回一次完整 `UnaryResult`，或以错误终止该调用。
- `push(close)` 最终产生 `backend.closed`。
- 同一 response 的 delta 事件早于 `response.done`。
- terminal error 必须通过 `error.terminal=true` 或 `backend.closed` 表达。
- full-duplex mode 支持 `response.output.delta(kind=listen)` 或等价 listen 状态事件。
- turn-based mode 支持 `response.done` 作为一次 response 的完成事件。

### 1.13 Design Rule

> Backend 接收模型输入和运行控制，发出结构化事件；具体实现可以使用 prefill/decode/finalize、pipeline、队列、异步 TTS 或其他机制，但这些机制通过统一 backend event 呈现给调用方。Runtime 位于 backend 之上，负责转发、记录、落盘和对外协议适配。

## 2. Runtime / Transport Binding

TBD.
