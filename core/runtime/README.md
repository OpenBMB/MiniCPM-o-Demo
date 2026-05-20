# Runtime Layer

This package is the worker-local boundary between transport handlers and model
backends.

## Current Shape

The public and legacy worker protocols are unchanged.  Pages and direct API
clients can still use:

- `/ws/chat`
- `/ws/half_duplex`
- `/ws/duplex`

Inside the worker, duplex sessions now flow through:

```text
worker.py WebSocket handler
  -> RuntimeManager
  -> DuplexSessionRuntime
  -> DuplexBackendAdapter
  -> PyTorch model methods (or future C++/SGLang/vLLM adapters)
```

Gateway worker scheduling now also has a small capability contract.  Worker
health responses report capabilities such as `chat`, `streaming`,
`audio_duplex`, and `omni_duplex`; `WorkerPool` keeps these on
`WorkerConnection` and only assigns a request to an idle worker that advertises
the required capability.  Existing workers default to all capabilities, so this
is a compatibility-preserving boundary for future specialized runtimes.

## Responsibilities

### worker.py

- Parse legacy WebSocket messages.
- Decode or persist transport-facing artifacts when still required.
- Send legacy WebSocket responses.
- Keep existing `/ws/duplex` behavior stable.

### RuntimeManager

- Own worker-local runtime instances by session id.
- Close runtimes on session end or worker shutdown.

### DuplexSessionRuntime

- Own per-session inference lifecycle.
- Convert one input frame into backend prefill/generate work.
- Manage deferred finalize internally.
- Drain finalize before stop/cleanup.

### BackendAdapter

- Hide backend-specific execution mechanics.
- PyTorch may need explicit finalize.
- C++ may treat finalize as a no-op and produce audio asynchronously.
- Future backends should satisfy the same adapter contract.

## Design Rule

`finalize`, KV-cache edits, prompt-cache artifacts, and backend-specific stream
mechanics should not leak into gateway or scheduler code.  They belong inside
runtime/backend boundaries.

