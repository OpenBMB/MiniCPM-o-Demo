# FC Duplex API 化与后续 Refactor 迁移计划

本文记录当前 FC duplex MVP 如何迁入正式 `/v1/realtime` API 体系，以及后续如何再迁到 `o45-refactor` 分支。目标不是重写推理代码，而是在保持 FC tool-call 行为正确的前提下，把 standalone demo 外周逐步替换为正式 API 外周。

## 涉及项目与职责

### `MiniCPM-o-Demo-wt-duplex-tool-api-2026-07-02`

当前工作基线，也可称为 `tool-api`。它来自 `o45-fc-try`，已经包含 FC 推理代码和 standalone `audio_duplex_board` MVP。

职责：

- 作为第一阶段 API 化改造的主工作区。
- 保留已验证的 FC 推理代码，先不大幅重构 `FcDuplexCapability` / `FcDuplexView`。
- 把 `audio_duplex_board` 中已跑通的会话调度与 tool-call 行为迁入正式 API 体系。
- 产出一个可以通过 `/v1/realtime` 体验 FC think/tool-call/tool-result 的版本。

### `MiniCPM-o-Demo-o45-fc-try`

FC MVP 源分支 worktree。它包含 sunweiyue 的 `audio_duplex_board` MVP 以及后续合入的本地调整。

职责：

- 作为 FC 行为的黄金参考。
- 参考其中的 `audio_duplex_board/session.py`、tool service、前端表现和实际运行行为。
- 不作为最终 API 主入口。

### `MiniCPM-o-Demo-wt-inference-refactor-2026-06-30`

此前的 o45 inference refactor worktree。它的目标是让 `modeling_minicpmo_unified.py` 尽量复用 vendored/base modeling，让 unified 只承担 demo/runtime/capability 外壳。

职责：

- 第二阶段迁移目标。
- 等 `tool-api` 先跑通正式 API FC 行为后，再把 FC 推理核心和 API runtime 迁入这里。
- 迁移时以 `tool-api` 的行为作为对照，避免同时引入推理迁移错误和 API 外周错误。

### `MiniCPM-o-Demo-wt-agent-harness-2026-06-09`

backend protocol / agent harness 参考 worktree。当前 FC tool-call API 补充协议不在
`tool-api` worktree 内，而是在这里：

```text
/user/weihongliang/MiniCPM-o-Demo-wt-agent-harness-2026-06-09/docs/backend-protocol/duplex-tool-calling.md
```

下文提到的 `response.think.*`、`response.tool_call.args.*`、`input.tool_result` 和
`response.output.sp_tokens` 语义，以这个文件为准。不要把它误写成当前 worktree 下的
`docs/backend-protocol/duplex-tool-calling.md`。

### `main`

当前 upstream/main 对应的普通 demo 主线。历史上 FC feature 曾进入过主线，随后被 revert。当前 `o45-fc-try` 不是从当前 `main` 直接分出，而是沿着被 revert 前的 FC feature 继续开发。

职责：

- 作为普通 API / gateway / worker / py_backend 体系的参考。
- 不作为直接 cherry-pick FC 的目标，因为 FC 分支和当前 main 的历史关系较复杂。

## 当前代码现状

## 运行环境与权重路径

### 本机环境

本机没有 GPU，主要用于代码编辑、轻量静态检查和无 GPU fake/runtime 单测。依赖安装不能全局安装，必须使用项目路径下的 venv。

当前 `tool-api` worktree：

```text
/user/weihongliang/MiniCPM-o-Demo-wt-duplex-tool-api-2026-07-02
```

当前 `o45-refactor` worktree：

```text
/user/weihongliang/MiniCPM-o-Demo-wt-inference-refactor-2026-06-30
```

当前 `o45-fc-try` 参考 worktree：

```text
/user/weihongliang/MiniCPM-o-Demo-o45-fc-try
```

### 本机 FC 默认权重

`tool-api` / `o45-fc-try` 默认使用：

```text
base model: /user/weihongliang/autoshow_omni/models/MiniCPM-o-4_5
overlay pt: /user/weihongliang/o45_fc_assets/checkpoints/minicpm-v_100.pt
```

对应配置位置：

```text
audio_duplex_board/config.py
scripts/run_o45_fc_model_server_cctl.sh
```

### sunweiyue MVP 默认权重

sunweiyue 的 MVP 启动脚本默认使用同一个 base model，但 `.pt` 是训练目录下的原始产物：

```text
base model: /user/weihongliang/autoshow_omni/models/MiniCPM-o-4_5
overlay pt: /user/sunweiyue/lib/swy-dev/omni_agent_research/minicpm_o5_training/experiments/overfit_midtrain_v2_100/runs/job_137673/checkpoints/minicpm-v/swy_o5_midtrain_v2_overfit/o5_midtrain_v2_overfit100_sdk005a0_v1/job_137673_ckpt_100/minicpm-v_100.pt
```

对应脚本：

```text
/user/sunweiyue/lib/swy-dev/omni_agent_research/minicpm_o5_demo/scripts/run_audio_duplex_board_server_cybertron.sh
```

本地 `/user/weihongliang/o45_fc_assets/checkpoints/minicpm-v_100.pt` 应视为该训练产物的本地副本或等价 overlay。

### 34 机器

34 机器指：

```text
34.66.244.63
ssh user: weihongliang
```

如果本机访问外网不通，可以借用：

```bash
ssh weihongliang@34.66.244.63
```

34 上的 FC 权重路径和本机不同，位于 `/home/weihongliang`：

```text
base model: /home/weihongliang/models/MiniCPM-o-4_5
overlay pt: /home/weihongliang/o45_fc_assets/checkpoints/minicpm-v_100.pt
```

34 上还存在若干 base model 候选副本：

```text
/home/weihongliang/MiniCPM-o-Demo-o45-fc-try/models/MiniCPM-o-4_5
/home/weihongliang/MiniCPM-o-Demo/models/MiniCPM-o-4_5
/home/weihongliang/MiniCPM-o-Demo-vendored-duplex-o45-probe/models/MiniCPM-o-4_5
```

34 上也有 GGUF 权重候选，但 FC/PyTorch demo 默认应使用 HF-style base model + `.pt` overlay：

```text
/home/weihongliang/models/MiniCPM-o-4_5-gguf
```

34 上可复用 venv：

```text
o45/base venv: /home/weihongliang/MiniCPM-o-Demo/.venv/base/bin/python
o5 venv:       /home/weihongliang/MiniCPM-o-Demo-o5-inference-refactor/.venv/o5/bin/python
```

当前 FC / o45 smoke 优先使用 o45/base venv：

```text
/home/weihongliang/MiniCPM-o-Demo/.venv/base/bin/python
```

已确认的关键包版本：

```text
python       3.12.3
torch        2.8.0+cu128
transformers 4.51.0
edge_tts     7.2.8
librosa      0.9.0
soundfile    0.12.1
fastapi      0.136.1
uvicorn      0.46.0
numpy        2.4.4
```

o5 venv 包含 transformers 5.5.4，更适合 o5 trusted code，不优先用于当前 o45 FC smoke。

34 使用原则：

- 优先只读检查资源，再启动最小 smoke。
- 不污染系统目录，不写他人目录。
- 不杀别人的 GPU 任务。
- 需要产物时放在 `/home/weihongliang` 或项目自己的输出目录。

### cctl

本机无 GPU 时可以使用 `cctl` 启动集群任务。优先使用 `agent-dev` 资源池。

使用原则：

- 先查询资源和任务状态。
- GPU 资源不足时不杀别人的任务。
- 只启动最小必要 smoke，不长时间占用资源。
- 如果 smoke 无法触发 tool-call 或模型行为不稳定，不反复调参硬跑。

### edge-tts 测试音频

`o45-refactor` worktree 里曾用 `edge_tts` 合成用户输入音频，脚本包括：

```text
/user/weihongliang/MiniCPM-o-Demo-wt-inference-refactor-2026-06-30/scripts/minimal_o45_unified_model_duplex.py
/user/weihongliang/MiniCPM-o-Demo-wt-inference-refactor-2026-06-30/scripts/minimal_o45_vendored_duplex.py
```

用法：

```python
import edge_tts
await edge_tts.Communicate(USER_TEXT, "zh-CN-XiaoxiaoNeural").save(str(path))
```

生成后用 `librosa.load(path, sr=16000, mono=True)` 转为 16k float waveform，可作为 FC/duplex 用户音频输入。

当前可用于 tool-call smoke 的故事文本：

```text
接下来我给小朋友会讲一个故事，故事中出现的动物你给我放到画板上
清晨的森林刚刚醒来，一只小松鼠从老橡树上探出头，蹦蹦跳跳地跑下来，草地上一只梅花鹿正在吃草，不远处的小河边一只灰兔子蹲在河边喝水
```

该文本预期可能触发 `display_object_on_board`，目标对象包括：小松鼠、梅花鹿、灰兔子；老橡树是否作为展示对象取决于模型/tool 规则。

### FC 推理代码已经在 `tool-api`

核心文件：

- `MiniCPMO45/modeling_minicpmo_unified.py`
  - 包含 `FcDuplexCapability`。
  - `MiniCPMO` 暴露 `fc_duplex_*` 转发方法。
- `core/processors/unified.py`
  - 包含 `FcDuplexView`。
  - 管理 `ToolCallStateManager`、tool-call id、tool started / response 注入。
- `core/schemas/fc_duplex.py`
  - FC duplex 的 Pydantic schema。

因此第一阶段不应把主要精力放在重新搬推理代码上，而应放在把 standalone MVP 的会话调度和 tool-call 事件迁入正式 API。

### Standalone MVP 外周不应整体迁入

旧 `audio_duplex_board` 是独立 demo 应用，不是正式 backend/gateway API 体系；当前分支已移除该 standalone 目录，历史参考保留在 `/user/weihongliang/MiniCPM-o-Demo-o45-fc-try`。

不应作为最终主入口迁入：

- `audio_duplex_board/run_server.py`
- `audio_duplex_board/model_server.py`
- `audio_duplex_board/web/*`
- `audio_duplex_board/remote_view.py`
- `audio_duplex_board/ws_remote_view.py`

这些文件只作为历史参考，不在当前正式分支保留。正式路径应落在 `py_backend/server.py`、`worker.py`、`gateway.py` 和 runtime/API 事件体系中。

## 应吸收的逻辑

### 来自 `audio_duplex_board/session.py`

应吸收的核心逻辑：

- per-unit 调度：`prefill -> spoken_generate -> non_spoken decode loop -> finalize`。
- non-spoken polling / stop / budget 逻辑。
- closed span 转 higher-level event 的思路。
- `_pending_tool_responses` 在下一次 prefill 注入模型的机制。

不建议直接 import `AudioDuplexBoardSession`，因为它绑定了 board 私有事件、standalone server、`BoardEvent`、页面状态和工具展示逻辑。更合适的方式是复制核心文本到新的 API runtime 类，并只改输入/输出边界。

### 来自 `audio_duplex_board/tools/display_object_on_board/service.py`

已迁入当前分支的 demo 层：`demos/fc_board/tools/display_object_on_board/service.py`。

价值：

- 已经实现 `display_object_on_board` 工具执行。
- 已经生成训练对齐的 `tool_response_content` JSON 字符串。
- 能快速保持 MVP 行为，减少变量。

注意：

- 下载目录已改到 `demos/fc_board/tools/display_object_on_board/live_image_downloads`，不再落在旧 standalone 目录。
- 后续如果正式 API 由 client/runtime 执行工具，则该 service 可以退化为本地 demo tool executor，而不是 backend 必需组件。

### 来自 `audio_duplex_board/schemas.py` 和 `events.py`

只作为参考，不直接迁。

原因：

- 它们定义的是 board 页面事件，例如 `non_spoken_delta`、`tool_call_final`、`board_card_created`。
- 正式 API 应该发 agent-harness worktree 中
  `/user/weihongliang/MiniCPM-o-Demo-wt-agent-harness-2026-06-09/docs/backend-protocol/duplex-tool-calling.md`
  定义的事件，例如 `response.think.*`、`response.tool_call.args.*`、`input.tool_result`。

### `response.output.sp_tokens` 约束

`duplex-tool-calling.md` 明确定义了 `response.output.sp_tokens`，但它不是任意调试事件通道。

关键约束：

- `token` 是协议枚举，不是 tokenizer raw token 文本，也不是 token id。
- 每个 special token 独占一条 `response.output.sp_tokens` 事件。
- `response.output.sp_tokens` 是只读信息，runtime/client 不能根据它执行工具、取消工具、关闭 session 或强制 listen。
- input-event 侧 special token，例如 `<|tool_started|>`、`<|event_budget_reached|>`、`<|tool_response_streaming|>`，不属于默认输出范围。

当前输出侧允许的 `token` 枚举是：

```text
listen
tts_pad
speak
spoken_slot_eos
spoken_turn_eos
no_action
non_spoken_eos
non_spoken_budget_reached
non_spoken_hold
non_spoken_abort
```

因此 `fc_prefill`、`fc_finalize` 这类 adapter 生命周期标记不是协议 token，不应通过
`response.output.sp_tokens` 下发。

## Tool-call id 设计

需要区分 API id 和模型内部 id。

### 两类 id

- `api_tool_call_id`
  - 对外暴露给 runtime/client。
  - 用于 `response.tool_call.args.begin/delta/raw/abort` 和 `input.tool_result`。
- `internal_tool_call_id`
  - `FcDuplexView/ToolCallStateManager` 在 tool-call 闭合后生成。
  - 当前形态类似 `fc_call_000001`。
  - 模型侧 `tool_started` / `tool_response` 注入需要使用这个 id。

### 映射

API runtime 维护：

```python
api_to_internal: dict[str, str]
internal_to_api: dict[str, str]
```

闭合流程：

1. API runtime 看到 tool-call 开始，生成 `api_tool_call_id`。
2. 对外发 `response.tool_call.args.begin`。
3. 参数 token 持续生成时，对外发 `response.tool_call.args.delta`。
4. tool-call 闭合后，`FcDuplexView` 返回 `internal_tool_call_id`。
5. runtime 建立 `api_tool_call_id <-> internal_tool_call_id` 映射。
6. 对外发 `response.tool_call.args.raw`，其中可以附带 `internal_tool_call_id` 供诊断，但 runtime/client 的主 id 仍是 API id。

工具结果回灌流程：

1. runtime/client 发 `input.tool_result`，携带 `api_tool_call_id`。
2. backend runtime 查映射得到 `internal_tool_call_id`。
3. backend runtime 构造 `FcToolResponse(call_id=internal_tool_call_id, content=...)`。
4. 下一次 `fc.streaming_prefill(tool_responses=...)` 注入模型。

abort 流程：

- 如果 tool-call 未闭合就 abort，则只有 `api_tool_call_id`。
- 不生成 `internal_tool_call_id`。
- 不允许后续 `input.tool_result`。
- 对外发 `response.tool_call.abort`。

长期更干净的方向是让 `FcDuplexView` 接收外部指定 id，使 API id 和 internal id 合一。但第一阶段为减少推理代码改动，先维护双 id 映射。

## 第一阶段：在 `tool-api` 上 API 化 FC

### 自动执行边界

后续如果在用户不实时盯着的情况下继续执行，应遵守以下边界：

- 只做 `tool-api` 上的最小 API 化接线，不启动大规模 refactor。
- 优先复制并改造 `audio_duplex_board/session.py` 的核心 runtime 逻辑，尽量少发明新结构。
- tool-call response 由 client/runtime 上层执行并通过 `input.tool_result` 回填；backend runtime 不自动执行业务工具。
- 可以新增一个纯 CLI smoke：连接 API、发送音频、监听 tool-call、回 `input.tool_result`。
- 如果 CLI smoke 没触发 tool-call，不反复调参刷到触发；记录现象后停下。
- 不为听感、模型选择行为或偶发输出做硬编码修补。
- 遇到复杂问题、模型行为不稳定、架构需要取舍时停下等用户指令。
- 不污染其他目录，不写 `/user/sunweiyue`，不全局安装依赖。

### 1. 新增 API runtime 类

建议新增：

```text
py_backend/fc_duplex_runtime.py
```

其中放 `FcDuplexSessionRuntime`。

职责：

- 接收 API `input.append`。
- 维护输入节奏和 backpressure。
- 调用 FC backend primitive：prepare、prefill、spoken generate、non-spoken generate、finalize。
- 控制 non-spoken decode loop 和 budget reached。
- 维护 pending tool responses。
- 维护 API tool id 与 internal tool id 映射。
- 输出正式 API event。

### 2. 复制而不是抽象核心 loop

第一版建议从 `audio_duplex_board/session.py` 文本复制核心函数，保持变量名和结构尽量一致。

优先复制并改造：

- `_run_non_spoken_loop`
- `_run_non_spoken_loop_polling`
- `_emit_step_events`
- `_emit_close_for_span`
- `_dispatch_tool_call` 的语义事件部分
- pending tool response 相关状态

可以先不迁：

- `_maybe_dump_speak_wav`
- board UI 事件细节
- standalone WS / remote view 逻辑
- replay/offline 逻辑

预估规模：

- 新增 `FcDuplexSessionRuntime`：约 350-500 行。
- API event adapter / tool id mapping：约 100-200 行。
- 总体第一版新增/改写约 450-700 行，不应复制整个 `audio_duplex_board` 的 1500+ 行外周。

### 3. 接入 `py_backend/server.py`

当前 full duplex 逻辑在 `BackendProtocolSession._push_full_duplex()` 内直接执行：

```text
input.append
  -> wait previous finalize
  -> backend.duplex_prefill
  -> backend.duplex_generate
  -> send output
  -> schedule finalize
```

FC API 化后，应改为：

```text
input.append
  -> FcDuplexSessionRuntime.push_input(...)
  -> runtime 内部完成 per-unit 调度和事件发送
```

短期可以让 `BackendProtocolSession` 只负责协议收发和调用 runtime。中长期应避免把 polling、budget、tool-call mapping 继续堆进 `BackendProtocolSession`。

### 4. 扩展 `PyTorchBackend`

需要在 `core/processors/pytorch_backend.py` 暴露 FC primitive。

建议方法：

- `fc_duplex_prepare(...)`
- `fc_duplex_prefill(...)`
- `fc_duplex_spoken_generate(...)`
- `fc_duplex_non_spoken_generate(...)`
- `fc_duplex_finalize(...)`
- `fc_duplex_cleanup(...)`

这些方法只做适配，不放调度逻辑。

### 5. 输出正式 API 事件

从 board event 改成正式 API event。

映射方向：

- spoken output -> `response.output.delta(kind="audio" / kind="text")`
- think begin/delta/end -> `response.think.begin/delta/end`
- tool-call begin/delta/raw -> `response.tool_call.args.begin/delta/raw`
- tool-call abort -> `response.tool_call.abort`
- tool result input -> `input.tool_result`

第一版可以保留额外 debug metadata，但不要让前端/runtime 依赖模型 XML。

### 6. 工具执行策略

正式路径不允许 backend runtime 自动执行业务工具。工具执行属于 client / 上层 runtime：

- backend 只发 `response.tool_call.args.begin/delta/end/raw`。
- client 解析 `response.tool_call.args.raw`，执行对应工具。
- client 通过 `input.tool_result` 回填结果。
- backend runtime 维护 api id -> internal id 映射，再转成 `FcToolResponse`。
- 下一次 `streaming_prefill(tool_responses=...)` 注入模型。

`display_object_on_board` 的模型可见 tool response content 需要保持训练对齐格式：JSON **字符串**，不是 dict。例如：

```json
{"status":"displayed","name":"小松鼠","reason":"已在画板显示该对象。"}
```

因此正式 API 回填建议使用：

```json
{
  "type": "input.tool_result",
  "tool_call_id": "tc_000001",
  "contents": [
    {
      "kind": "text",
      "text": "{\"status\":\"displayed\",\"name\":\"小松鼠\",\"reason\":\"已在画板显示该对象。\"}"
    }
  ]
}
```

backend runtime 只把 `contents[*].text` 拼成模型输入文本，不理解/执行具体业务工具。

### 6.1 non-spoken 调度模式

`session.init` 需要支持两种 non-spoken 调度模式，用于权衡实时性和完整 tool-call 质量：

```json
{
  "fc_duplex": {
    "non_spoken_scheduling": "latency"
  }
}
```

可选值：

- `latency` / 性能模式：下一 audio chunk 到来时，当前 unit 应尽快以 `budget_reached` 收束，优先处理新输入。这个模式要求 API 层正确维护跨 unit open span：`budget_reached` 不代表 `<think>` / `<tool_call>` 闭合，不能发 tool-call done。
- `quality` / 质量模式：audio chunk 进入队列，当前 unit 跑完固定 `non_spoken_budget_per_unit` 或自然 terminated 后再处理下一个 chunk。这样更容易让 think/tool-call 在当前 unit 内自然闭合，但会积压延迟。

这属于 view 之上的 session/runtime 节奏控制；token 合法性、special token 插入和 `budget_reached` 如何进入模型上下文仍属于 `FcDuplexCapability` / modeling 层。

当前 `tool-api` 实现状态：

- `FcDuplexSessionRuntime` 按 MVP session 的主循环执行 `prefill -> spoken -> non-spoken loop -> finalize`。
- `latency` 模式下 audio 输入进入单元素待处理队列；新 chunk 到来会触发当前 non-spoken loop 以 `budget_reached` 收束，并保留最新待处理 chunk。
- `quality` 模式下 audio 输入按队列顺序处理，不丢弃待处理 chunk。
- `response.output.delta kind=non_spoken` 已从 backend 正式输出中移除；未能实时归类到 think/tool_call 的中间 token 只通过 `debug.fc_non_spoken.delta` 观察。
- `input.tool_result` 与 `input.tool_result.delta/done` 都已接入；流式 tool result 在 backend 聚合到 `done` 后作为完整 `FcToolResponse` 注入下一次 prefill。
- `MiniCPMO45/modeling_minicpmo_unified.py`、`core/processors/unified.py`、`core/schemas/fc_duplex.py` 与 clean MVP worktree 保持字面一致，view 以下没有为 API 外周额外 fork。

## 第二阶段：验证 `tool-api` 行为

验证重点：

- FC 推理能通过 `/v1/realtime` 跑起来。
- think 能流式输出并正确闭合。
- tool-call 参数能流式输出并正确闭合。
- tool-call raw 能被 runtime/client 消费，不要求 client 解析 XML。
- tool result 能通过 pending response 在下一次 prefill 注入模型。
- 注入 tool result 后模型能继续产生正确 spoken / non-spoken 输出。
- budget reached 由 runtime 层决定，special token 由 capability/view 内部正确补齐。

对齐方式：

- 使用 `/user/weihongliang/MiniCPM-o-Demo-o45-fc-try/audio_duplex_board/session.py` 作为行为参考；当前分支不再保留 standalone 目录。
- 对复制出的 runtime 核心函数做文本 diff。
- 如果行为异常，优先对比 per-unit 调度、stop/budget、pending tool response 注入时机。

## 第三阶段：迁入 `o45-refactor`

前提：

- `tool-api` 已经跑通正式 API FC 行为。
- tool-call 行为已经有可复现 demo 或最小 client。

迁移内容：

- `FcDuplexCapability`。
- `FcDuplexView` 和 `ToolCallStateManager`。
- `core/schemas/fc_duplex.py`。
- `PyTorchBackend` 的 FC primitive。
- `py_backend/fc_duplex_runtime.py`。
- `py_backend/server.py` 的 FC API 接入。

迁移原则：

- 以 `tool-api` 的可跑行为为金标准。
- 每次只迁一小块，迁完立即对照 API 行为。
- 避免同时改推理核心和 API 外周。
- 不把 `audio_duplex_board` standalone 外周迁成正式主入口。

## 分层原则

最终希望形成如下分层：

```text
gateway.py
  - 排队、worker 分配、API WS 透传

worker.py / runtime/backend_client.py
  - worker 内部 runtime 协议转发

py_backend/server.py / FcDuplexSessionRuntime
  - API session 协议
  - input queue / polling / budget / tool-call id mapping
  - 正式 API event 输出

PyTorchBackend
  - 推理后端适配器
  - 暴露 fc_duplex_* primitive

UnifiedProcessor / FcDuplexView
  - 模式 view
  - schema 包装
  - tool started / tool response 状态管理

MiniCPMO / FcDuplexCapability
  - 真正模型状态机
  - KV cache / special tokens / streaming_prefill / streaming_non_spoken_generate
```

核心原则：

- polling、时钟、budget 决策放在 runtime/session 层。
- special token 的具体补齐放在 view/capability 层。
- gateway/worker 不放业务 tool-call 状态机。
- 旧 `audio_duplex_board` 不作为正式 API 架构的一部分，也不在当前分支保留；需要对照时使用 `o45-fc-try` worktree。

## 2026-07-02 本轮最小实现状态

已在 `tool-api` worktree 做了第一版最小 API 接线，目标是能通过 backend protocol 跑 FC duplex/tool-call smoke，而不是迁移 standalone board UI。

新增/修改文件：

```text
py_backend/fc_duplex_runtime.py
py_backend/server.py
worker.py
core/processors/pytorch_backend.py
scripts/fc_duplex_api_smoke.py
docs/fc-duplex/api-migration-plan.md
```

当前接线方式：

- `session.init.payload.fc_duplex=true` 或 `runtime/config.runtime=fc_duplex` 时，`py_backend/server.py` 创建 `FcDuplexSessionRuntime`。
- 普通 full-duplex 不带 `fc_duplex` 时仍走旧 `duplex_prefill -> duplex_generate` 路径。
- FC runtime 每个音频 chunk 执行：

```text
fc_duplex_prefill -> fc_duplex_spoken_generate -> fc_duplex_non_spoken_generate loop -> fc_duplex_finalize
```

- `core/processors/pytorch_backend.py` 新增 `fc_duplex_*` primitive wrapper，并通过 `processor.set_fc_duplex_mode()` 获取 `FcDuplexView`。
- backend websocket 已能接受 `input.tool_result`，并转为内部 payload：`{"type":"tool_result", ...}`。
- worker websocket 也允许 `input.tool_result` 透传给 backend runtime。
- 如果 init 没传 tools，FC runtime 默认注入 `display_object_on_board` tool definition，和 standalone MVP fallback 保持一致。

当前事件策略：

- spoken 文本/音频继续用现有 `response.output.delta kind=text/audio/listen`。
- non-spoken token 先作为 `response.output.delta kind=non_spoken` 输出，便于 smoke 观察。
- think closed span 输出为：

```text
response.think.begin
response.think.delta
response.think.end
```

- tool-call closed span 输出为：

```text
response.tool_call.args.begin
response.tool_call.args.delta
response.tool_call.args.end
response.tool_call.args.raw
```

- 外部 API id 使用 `tc_000001` 递增；内部 FC id 例如 `fc_call_000001` 只保存在 runtime 映射里。
- 自动执行 `display_object_on_board` 时，会发送临时事件 `response.tool_result`，并把模型需要的 `FcToolResponse(call_id=<internal_id>, content=<训练格式 JSON 字符串>)` 排入下一次 prefill。

当前限制：

- 没有实现 `input.tool_result.delta/done`；MVP 暂不支持 tool response streaming。
- 没有迁移 standalone board 的页面事件、远程 view、run_server/model_server/web。
- 没有为 `input.standalone` 单独接模型注入路径。
- 如果 smoke 没触发 tool-call，不应反复改 prompt 或做复杂 workaround；记录结果后等下一步指令。

本地静态检查：

```bash
python -m py_compile \
  core/processors/pytorch_backend.py \
  py_backend/server.py \
  py_backend/fc_duplex_runtime.py \
  worker.py \
  scripts/fc_duplex_api_smoke.py
```

已通过。

## 最小 smoke 脚本

路径：

```text
/user/weihongliang/MiniCPM-o-Demo-wt-duplex-tool-api-2026-07-02/scripts/fc_duplex_api_smoke.py
```

用途：

- 用 `edge-tts` 合成上面的故事音频。
- 用 `librosa.load(..., sr=16000, mono=True)` 读取为 16k float32。
- 按 1 秒 chunk 发送 `input.append`。
- 打印 `response.output.delta`、`response.think.*`、`response.tool_call.args.raw`、`response.tool_result`、`response.output.sp_tokens`。
- 保存模型返回的音频 delta 到 output dir，方便人工试听。

示例：

```bash
python scripts/fc_duplex_api_smoke.py --backend-url http://127.0.0.1:22500
```

或远端：

```bash
/home/weihongliang/MiniCPM-o-Demo/.venv/base/bin/python \
  scripts/fc_duplex_api_smoke.py \
  --backend-url http://127.0.0.1:22500 \
  --output-dir /home/weihongliang/fc_duplex_api_smoke
```

## 自动运行边界

用户明确要求自动阶段不要做太多 workaround：

- 可以做最小接线、静态检查、一次 smoke。
- 遇到复杂模型行为、工具不触发、音频异常、依赖/资源问题，不要长时间调参或大改。
- 不写 `/user/sunweiyue`。
- 不全局安装依赖。
- 不杀其他人的 GPU 进程。
- cctl 不可用时可用 34，但同样不能抢/杀他人进程。

## 当前资源检查结果

本轮尝试检查 cctl 与 34：

- `cctl pool list -o json` 请求 `http://cybertron.thunlp.org/api/resource_pool/` 超时，因此没有继续在 cctl 上启动任务。
- 34 机器可 ssh，但两张 RTX PRO 6000 当前显存均被占用较多：
  - GPU0 约 72.9G used / 24.3G free。
  - GPU1 约 75.8G used / 21.4G free。
- 进程包括一个 weihongliang 自己的 o5 backend，以及 root/yanyunhe 的长期服务。按约束没有 kill 任何进程。

因此本轮只完成本地静态检查，没有启动模型 smoke。

## 34 o45 base 权重更正

2026-07-02 后续检查发现，34 上原先记录的：

```text
/home/weihongliang/models/MiniCPM-o-4_5
```

不是完整 HF 权重目录，启动 `py_backend/server.py` 时 transformers 报：

```text
OSError: Error no file named pytorch_model.bin, model.safetensors, tf_model.h5, model.ckpt.index or flax_model.msgpack found
```

只读搜索 `/home`、`/data`、`/mnt` 后，只找到 o46 safetensors：

```text
/home/weihongliang/models/minicpmv46-final/checkpoints/MiniCPM-V-4.6-Thinking/model.safetensors
/home/weihongliang/models/minicpmv46-final/checkpoints/MiniCPM-V-4.6/model.safetensors
```

因此当前 34 上还不能直接运行 o45 FC smoke，除非补齐 o45 base 权重，或明确改用其他兼容 base。

## 34 smoke 最新结果更正

此前关于 34 缺 o45 base 权重的判断不完整：完整 o45 base 实际通过 symlink 存在于：

```text
/home/weihongliang/MiniCPM-o-Demo/models/MiniCPM-o-4_5 -> /root/xubokai/MiniCPM-o-4_5
```

该路径可看到完整 HF shard：

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
```

34 上 FC SDK 源码路径：

```text
/home/weihongliang/o45_fc_assets/sdk/src
```

## SDK 与当前推理代码的保障边界

根据 `minicpm_o5_sdk` guidebook 的推理章节，FC/O5 duplex 推理最硬的不变量是：训练时构造出的 token id 流必须和推理时喂给模型的 token id 流按位一致。它特别强调不要把 HF `AutoTokenizer.from_pretrained(...)` 当成 duplex 协议 tokenizer 直接使用，因为：

- HF added token longest-match 可能把用户普通文本里的 `<image>` / `<tool_call>` / `<think>` 等字面误编码成协议 control token id。
- BPE 是相邻 token greedy merge，`encode(chunk1) + encode(chunk2)` 不保证等于 `encode(chunk1 + chunk2)`；unit/budget 切分如果由 runtime 自己按字符串切，会造成训练和推理 token 边界漂移。
- tool definitions、tool call、tool response 等模型可见字符串必须复用 SDK 的 stringify / normalize / serializer 入口，不能各层自己拼一个“看起来等价”的格式。

当前 `tool-api` 代码已经接入了 SDK 的一部分硬事实，但不是全链路由 SDK 协议状态机兜底：

| guidebook 事项 | 当前代码是否由 SDK 保证 | 当前落点 |
|---|---:|---|
| special token id 不手猜 | 基本是 | `MiniCPMO45/modeling_minicpmo_unified.py` 的 `FcDuplexCapability._ensure_protocol()` 使用 `load_builtin_o45_fc_tokenizer()` 和 `O5SpecialTokenRegistry.from_tokenizer(...)`。 |
| ordinary content 不撞 control id | 基本是 | `FcDuplexCapability.encode_text()` 调 SDK `encode_ordinary_with_offsets()`；`decode_text()` 调 SDK `decode_ordinary()`。 |
| tool definition / tool call serializer | 部分是 | `get_o5_tool_serializer()` 用于 tool system block 渲染和 tool call deserialize。 |
| tool response content normalize | 是 | `_input_event_slot_ids()` 里调用 `normalize_tool_response_content(raw)`。 |
| target 显式选择 | 当前写死 O45_FC | FC MVP 当前使用 `load_builtin_o45_fc_tokenizer()` / `O5TokenizerID.O45_FC`；未来切 O5 需要显式替换，不应靠路径猜。 |
| 实时推理完整走 SDK `tokenize_training_data(...)` | 否 | 实时 prefill 仍由 `FcDuplexCapability` 手写 `_system_prefill_parts()`、`_unit_input_ids()`、`_input_event_slot_ids()` 等 skeleton。 |
| unit-aware budget / BPE 切分完全由 SDK pipeline 保证 | 不完全 | 单段 ordinary encode 用 SDK；但 unit/budget 调度和 slot 组织由推理代码手写。只要不自行把长 content 字符串切碎再分别 encode，风险较低，但不是 SDK 全链路兜底。 |
| decode 输出用 SDK streaming parser | 否 | `FcDuplexCapability` 自己维护 `_non_spoken_mode`、`_think_buf`、`_tool_call_buf` 并产生 `closed_spans`。 |
| API 层跨 unit open span 状态 | 否 | `py_backend/fc_duplex_runtime.py` 没有 import SDK，也没有完整复用 standalone MVP `session.py` 的 block 状态机；它只根据 `closed_spans` 发 API event。 |

因此，当前结构应理解为：

```text
py_backend/fc_duplex_runtime.py
  手写 session 调度 / budget / API event
      ↓
core/processors/unified.py FcDuplexView
  thin wrapper
      ↓
MiniCPMO45/modeling_minicpmo_unified.py FcDuplexCapability
  使用 SDK tokenizer / special token registry / tool serializer
  但 prefill skeleton、unit slot、non-spoken 状态机仍主要手写
```

这意味着：SDK 已经保护了 token id、ordinary encode、special token、tool serializer 这些底层协议事实；但 budget 跨 unit 的 open span、tool-call streaming event、API id 映射和前端可见事件仍由 `py_backend/fc_duplex_runtime.py` / MVP session 逻辑负责。后续如果出现 tool-call 片段泄露到 speech、budget 后 span 类型丢失、或 API 事件不符合正式协议，不能简单归因于“SDK 已经保证”，需要检查 runtime/session 状态机。

启动 backend 时需要把它加入 `PYTHONPATH`，否则 `session.init` 会报：

```text
No module named 'minicpm_o5_sdk'
```

当前可用 backend 启动命令：

```bash
cd /home/weihongliang/MiniCPM-o-Demo-wt-duplex-tool-api-2026-07-02
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/weihongliang/o45_fc_assets/sdk/src:. \
  /home/weihongliang/MiniCPM-o-Demo/.venv/base/bin/python py_backend/server.py \
  --host 127.0.0.1 \
  --port 22500 \
  --model-path /home/weihongliang/MiniCPM-o-Demo/models/MiniCPM-o-4_5 \
  --pt-path /home/weihongliang/o45_fc_assets/checkpoints/minicpm-v_100.pt \
  --gpu-id 0
```

本轮已在 34 GPU1 上启动成功，health 为 ready：

```text
http://127.0.0.1:22500/health -> {"status":"ready","backend":"pytorch","worker_status":"ready","active_session_id":null}
```

最小 API smoke 命令：

```bash
cd /home/weihongliang/MiniCPM-o-Demo-wt-duplex-tool-api-2026-07-02
PYTHONPATH=/home/weihongliang/o45_fc_assets/sdk/src:. \
  /home/weihongliang/MiniCPM-o-Demo/.venv/base/bin/python scripts/fc_duplex_api_smoke.py \
  --backend-url http://127.0.0.1:22500 \
  --output-dir /home/weihongliang/fc_duplex_api_smoke
```

结果：

- `session.init` 成功，`fc_prepare` 成功。
- 每个 1s audio chunk 都完成了 FC runtime 的 prefill/listen/non-spoken/finalize 调用链。
- 当时实现曾通过 `response.output.sp_tokens` 暴露 `fc_prefill` / `fc_finalize` 这类 adapter 生命周期标记；按上文协议约束，这些不是合法输出侧 sp-token，后续应删除或改为内部日志。
- 没有 backend crash。
- 没有触发 `response.think.*` 或 `response.tool_call.args.*`。
- 没有生成模型音频 delta。

这说明第一版 API 接线和 primitive 调用可以跑完整 session，但当前最小 smoke 没有复现 standalone MVP 的 tool-call 行为。按用户约束，暂不通过调 prompt/调度做复杂 workaround。
