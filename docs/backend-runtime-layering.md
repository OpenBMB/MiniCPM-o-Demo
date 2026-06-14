# Backend / Runtime Layering Design

本文档描述 MiniCPM-o 推理链路的代码分层。协议文档定义 backend contract；本文档说明这些 contract 在代码里如何落位。

## 1. Layer Overview

推荐分为四层：

| Layer | Owns | Does Not Own |
|-------|------|--------------|
| `Transport` | WebSocket、HTTP、gRPC、IPC、连接生命周期、重连、heartbeat、session id 映射 | 模型状态、推理 pipeline |
| `Runtime` | 转发、数据记录、媒体落盘、metrics 聚合、协议转换、调用 backend | backend 内部 listen/speak、prefill/decode pipeline |
| `InferenceBackend` | 有状态推理会话、KV/cache、模型输入队列、输出事件队列、pause/resume/close、metrics | WebSocket、UI 协议、session 录制格式 |
| `Engine / Compute` | PyTorch / llama.cpp 具体算子、prefill、decode、TTS、KV 操作 | 上层事件协议 |

核心原则：

```text
Transport owns connection.
Runtime owns orchestration and recording.
Backend owns inference state and backend-level control.
Engine owns compute.
```

## 2. Backend Contract

Backend 面向 runtime 暴露 pull-based 接口。它不接收 `emit` callback，也不知道上层事件要发往哪里。

```python
class InferenceBackend:
    def init(self, params: BackendInitParams) -> None: ...
    def push(self, message: BackendMessage) -> None: ...
    def pull(self) -> Iterator[BackendEvent]: ...
    def unary(self, request: UnaryRequest) -> UnaryResult: ...
```

Backend 内部可以是同步队列、异步队列、线程池、C++ worker thread、SSE stream 或 gRPC stream；对 runtime 暴露的语义是：

```text
push(message) 进入 backend
backend 内部推进推理状态
pull 产出 BackendEvent

unary(request) 进入 backend
backend 根据 request.type 完成一次性推理或查询
unary 返回 UnaryResult
```

`BackendMessage` 至少包含三类：

```text
input:   新的模型观察内容
control: pause/resume/metrics.get 等运行控制
close:   关闭 backend
```

Backend event 包括：

```text
backend.initialized
backend.state
backend.closed
input.accepted
input.rejected
input.committed
response.started
response.text.delta
response.audio.delta
response.output.delta(kind=listen)
response.speak
response.done
metrics.snapshot
metrics.frame
metrics.response
error
```

## 3. Runtime Contract

Runtime 面向 transport 暴露 emit-based 接口。`emit` 来自上层 transport handler，由 runtime 调用。

```python
class Runtime:
    def __init__(self, backend: InferenceBackend, recorder: Recorder | None = None):
        self.backend = backend
        self.recorder = recorder
        self.emit = None

    async def start(self, emit: EventSink) -> None:
        self.emit = emit
        create_task(self._pump_backend_events())

    async def push_data(self, data: InputBundle) -> None:
        self.backend.push({"type": "input", "payload": data})

    async def push_control(self, command: BackendControl) -> None:
        self.backend.push({"type": "control", "payload": command})

    async def ask(self, request: UnaryRequest) -> UnaryResult:
        result = await to_thread(self.backend.unary, request)
        await self._record_unary(request, result)
        return result

    async def close(self, reason: str | None = None) -> None:
        self.backend.push({"type": "close", "payload": {"reason": reason}})

    async def _pump_backend_events(self) -> None:
        for event in self.backend.pull():
            await self._record(event)
            await self.emit(self._translate(event))
```

Runtime 可以做：

- 把 transport message 转成 backend `BackendMessage`。
- 从 `backend.pull()` 读取事件。
- 记录输入、输出、音频、图片、metrics。
- 将 backend event 转成 worker protocol、OpenAI Realtime 风格事件或前端旧协议。
- 处理 transport 断开后的 cleanup。

Runtime 不应该做：

- 手动拆分 backend 内部的 prefill/decode/finalize 生命周期。
- 重新实现 backend 的 listen/speak 状态机。
- 在 backend 外部维护 KV/cache 语义。
- 把 WebSocket 连接状态当作 backend 推理状态。

## 4. Pull And Emit

`pull` 和 `emit` 分别属于不同边界：

```text
Backend -> Runtime: pull() -> BackendEvent
Runtime -> Transport: emit(event) -> void
```

伪代码：

```python
async def handle_worker_ws(ws):
    backend = CppBackend()
    runtime = Runtime(backend, recorder=SessionRecorder())

    async def emit(event):
        await ws.send_json(translate_to_wire_event(event))

    await runtime.start(emit)

    async for raw in ws.iter_text():
        msg = parse_wire_message(raw)
        if msg.kind == "input":
            await runtime.push_data(to_backend_input(msg))
        elif msg.kind == "control":
            await runtime.push_control(to_backend_control(msg))
```

事件方向：

```text
client
  -> transport
  -> runtime.push_data(...)
  -> backend.push(input)

backend.pull()
  -> runtime record / persist / translate
  -> emit(event)
  -> transport send
  -> client
```

一句话：

> `pull` 是 backend 给 runtime 的输出接口；`emit` 是 runtime 给 transport 的输出接口。

## 5. Current Code Mapping

当前代码已经有一部分接近这个分层，但命名和边界还没有完全对齐。

| Target Concept | Current Code |
|----------------|--------------|
| Runtime receives `emit` | `core/runtime/duplex.py::DuplexSessionRuntime.start(emit)` |
| Runtime calls `emit(event)` | `DuplexSessionRuntime.process_frame(..., emit=emit)` |
| Transport provides `emit` | `core/runtime/worker_handlers.py::_emit_event` closure |
| Runtime event shape | `core/runtime/events.py::RuntimeEvent` |
| Runtime-to-worker translation | `core/runtime/worker_protocol.py::runtime_event_to_worker_messages` |
| Gateway public translation | `gateway_modules/runtime_protocol.py` |
| Backend implementation | `core/processors/pytorch_backend.py`, `core/processors/cpp_backend.py` |

Current duplex path:

```text
worker websocket
  -> handle_worker_duplex_runtime_ws
  -> DuplexSessionRuntime.push_frame
  -> backend.prefill / backend.generate / backend.finalize
  -> RuntimeEvent
  -> _emit_event
  -> runtime_event_to_worker_messages
  -> ws.send_json
```

Target duplex path:

```text
worker websocket
  -> Runtime.push_data
  -> backend.push(input)
  -> backend internal pipeline
  -> backend.pull
  -> Runtime record / translate
  -> emit
  -> ws.send_json
```

The main difference is that current runtime still directly drives `prefill/generate/finalize`; target backend should be allowed to hide those steps behind `push` / `pull` for streaming interactions and `unary` for one-shot interactions.

## 6. Migration Direction

Suggested migration order:

1. Keep current `DuplexSessionRuntime.start(emit)` shape. It already matches the runtime-to-transport side.
2. Introduce a backend-facing event type, e.g. `BackendEvent`, separate from current transport-oriented `RuntimeEvent`.
3. Add an experimental backend wrapper that exposes `push`, `pull`, and `unary` while internally calling existing `prefill/generate/finalize`.
4. Move pause/resume/close handling into backend-facing control methods.
5. Update runtime to pump `backend.pull()` and use the existing `_emit_event` as the transport emit.
6. Apply `unary` to non-streaming turn-based chat, and keep streaming turn-based chat on `push` / `pull`.

This lets PyTorch and C++ backends share the same stateful backend contract without forcing either side to expose low-level compute steps as public protocol.
