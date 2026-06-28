# FC Slot 双工推理集成开发计划

## 目标

将 `project/MiniCPM-o-4_5-fc_duplex_infer` 中的新 FC slot 双工推理协议，以新增能力的方式集成到当前 `MiniCPM-o-Demo-FC` 项目中。

核心原则：

- 保留现有旧双工协议链路，不改动 `DuplexView` / `DuplexCapability` / `duplex_*` 的既有行为。
- 新协议使用独立命名空间：`fc_duplex` / `FcDuplexCapability` / `FcDuplexView`。
- 与老 `DuplexCapability` 相似的能力尽量使用相同方法名，便于维护和对照。
- `core/schemas/fc_duplex.py` 只定义 API 数据结构，不承载协议状态机行为。
- budget 判断由调用方负责；模型侧只按调用方指令继续生成或闭合 slot 并插入对应 special token。

## 总体架构

```text
调用方 / 服务层
  - 管理 unit 调度
  - 管理 non-spoken budget
  - 决定何时继续生成、何时 budget_reached 闭合
        |
        v
core/processors/unified.py
  FcDuplexView
  - API 参数整理
  - schema <-> 模型调用转换
  - offline_inference 离线便捷封装
        |
        v
MiniCPMO45/modeling_minicpmo_unified.py
  MiniCPMO.fc_duplex: FcDuplexCapability
  - FC slot 协议状态机
  - KV cache / StreamDecoder
  - spoken / non-spoken 生成
  - special token 插入与 slot 闭合
        |
        v
LLM / audio encoder / processor
```

## 新增模型层能力

### 1. `FcDuplexCapability`

放在 `MiniCPMO45/modeling_minicpmo_unified.py` 中。

职责：

- 管理 FC slot 协议 token 编码/解码。
- 管理独立 `StreamDecoder` 和 KV cache。
- 维护当前 unit 的状态：是否已经 prepare、是否处于 unit 中、当前 slot、累计 `output_ids`。
- 负责插入结构 token：`<unit>`、slot start/end、`</unit>`。
- 负责 spoken / non-spoken 两段生成。
- 根据调用方传入的 `close_reason` 插入终止 special token。

命名原则：

- 与老 `DuplexCapability` 相似的生命周期能力使用相同方法名。
- 新协议真正不同的地方，是把老协议的 `streaming_generate()` 拆成 `streaming_spoken_generate()` 和 `streaming_non_spoken_generate()`。
- 不使用 `start_unit()` 作为公开方法名；对应职责放入 `streaming_prefill()`。

方法映射：

```text
旧 DuplexCapability              新 FcDuplexCapability
prepare()                    ->  prepare()
streaming_prefill()          ->  streaming_prefill()
streaming_generate()         ->  streaming_spoken_generate()
                              +  streaming_non_spoken_generate()
finalize_unit()              ->  finalize_unit()
_reset_streaming_state()     ->  _reset_streaming_state()
set_break_event()            ->  set_break_event()            可选
clear_break_event()          ->  clear_break_event()          可选
set_session_stop()           ->  set_session_stop()           可选
clear_session_stop()         ->  clear_session_stop()         可选
is_break_set()               ->  is_break_set()               可选
is_session_stop_set()        ->  is_session_stop_set()        可选
```

建议方法：

```python
class FcDuplexCapability:
    def __init__(
        self,
        model,
        device="cuda",
        temperature=0.7,
        tool_format="minicpm4_xml",
        **kwargs,
    ): ...

    @property
    def protocol(self): ...

    def prepare(self, system_prompt: str, tools=None) -> dict: ...

    def streaming_prefill(
        self,
        audio_waveform=None,
        frame_list=None,
        tool_responses=None,
        sample_rate: int = 16000,
    ) -> dict: ...

    def streaming_spoken_generate(
        self,
        max_tokens: int = 24,
        decode_mode: str = "greedy",
    ) -> dict: ...

    def streaming_non_spoken_generate(
        self,
        decode_mode: str = "greedy",
        max_tokens: int = 1,
        close_reason: str | None = None,
    ) -> dict: ...

    def finalize_unit(self) -> dict: ...

    def decode_output_ids(self, output_ids=None, tools=None) -> dict: ...

    def _ensure_previous_unit_closed(self) -> None: ...

    def _reset_streaming_state(self) -> None: ...

    def cleanup(self) -> None: ...
```

`streaming_prefill()` 的职责：

- 在开始新 unit 前调用 `_ensure_previous_unit_closed()`，兜底闭合上一个未完成 unit。
- 写入 `<unit>`。
- 写入 `user_video_slot`。
- 写入 `user_audio_slot`，并把真实音频 embedding 喂入 decoder。
- 写入 `input_event_slot`，支持接收工具调用结果 `tool_responses`。
- 写入 `<ai_spoken_slot>`，为 spoken 生成准备 logits。

`streaming_spoken_generate()` 的职责：

- 在 `ai_spoken_slot` 内采样。
- 生成 `<|listen|>`，或生成 `<|speak|>` + spoken text + spoken terminator。
- 闭合 `</ai_spoken_slot>`。
- 返回本 unit 是否 speaking、spoken token、spoken 文本、是否 spoken turn eos。

`streaming_non_spoken_generate()` 的职责：

- 首次调用时写入 `<ai_non_spoken_slot>`。
- 当 `close_reason is None` 时，生成最多 `max_tokens` 个 non-spoken token，并返回是否自然终止。
- 当 `close_reason` 不为空时，不再继续采样，直接插入对应 terminator 并闭合 `</ai_non_spoken_slot>`。
- budget 是否耗尽由调用方判断；模型侧只负责执行 `close_reason="budget_reached"` 对应的 token 插入和 slot 闭合。
- 在线维护 `<think>...</think>` 和 `<tool_call>...</tool_call>` 的 span 状态。
- 当生成到 `</think>` 或 `</tool_call>` 时，在返回值中额外附带该完整 span 的解析结果，方便调用方在线处理，而不必等整轮 `decode_output_ids()`。

支持的 `close_reason`：

- `"eos"` -> `<|non_spoken_eos|>`
- `"no_action"` -> `<|no_action|>`
- `"budget_reached"` -> `<|non_spoken_budget_reached|>`
- `"hold"` -> `<|non_spoken_hold|>`
- `"abort"` -> `<|non_spoken_abort|>`

`streaming_non_spoken_generate()` 返回结构建议：

```python
{
    "token_ids": [...],
    "terminated": False,
    "close_reason": None,
    "closed_spans": [
        {
            "type": "think",
            "text": "...",
        },
        {
            "type": "tool_call",
            "wire": "...",
            "tool_call": {"name": "...", "arguments": "...", "error": None},
        },
    ],
}
```

为支持在线 span 解析，`FcDuplexCapability` 需要维护状态：

```python
self._non_spoken_slot_open = False
self._non_spoken_mode = None  # None / "think" / "tool_call"
self._think_buf = []
self._tool_call_buf = []
```

状态处理规则：

- 生成 `<think>`：进入 `"think"` 模式，清空 `think_buf`。
- 生成普通文本 token 且处于 `"think"`：追加到 `think_buf`。
- 生成 `</think>`：闭合 think span，解码 `think_buf`，返回 `closed_spans=[{"type": "think", "text": ...}]`。
- 生成 `<tool_call>`：进入 `"tool_call"` 模式，清空 `tool_call_buf`。
- 生成普通文本 token 且处于 `"tool_call"`：追加到 `tool_call_buf`。
- 生成 `</tool_call>`：闭合 tool span，反序列化 tool call，返回 `closed_spans=[{"type": "tool_call", ...}]`。
- 生成 non-spoken terminator：设置 `terminated=True`，闭合 `</ai_non_spoken_slot>`。

## 协议 Helper 设计

原新项目中的 `FcDuplexProtocol` 可以合并进 `FcDuplexCapability`，作为内部 helper 方法存在，而不是独立对外类。

需要保留的协议能力：

- special token id 查询：`sid(key)`。
- 文本编码/解码：`encode_text()` / `decode_text()` / `safe_decode_text()`。
- system prefill：`system_prefill_ids(system_prompt, tools)`。
- 输入 slot 编码：
  - `user_video_slot_ids(...)`
  - `user_audio_slot_ids(...)`
  - `input_event_slot_ids(tool_responses)`
  - `unit_input_ids(...)`
- 输出解析：
  - `decode_output_ids(ids, tool_definitions=None)`
  - tool call 反序列化。
- 在线 span 解析：
  - `_decode_closed_think_span(token_ids)`
  - `_decode_closed_tool_call_span(token_ids, tool_definitions=None)`

这些 helper 可以作为私有方法，例如 `_system_prefill_ids()`、`_input_event_slot_ids()`，避免暴露过多模型层 API。

## Budget 管理策略

不迁移 `RuntimeNonSpokenBudget` 作为模型侧必需组件。

调用方负责：

- 当前 unit 的 non-spoken budget 上限。
- 当前是否允许继续生成 non-spoken token。
- 当前是否预算耗尽。
- 预算耗尽时调用模型侧闭合接口。

模型侧负责：

- 继续生成 non-spoken token。
- 收到调用方指令后插入 `<|non_spoken_budget_reached|>`。
- 闭合 `</ai_non_spoken_slot>`。
- 保持 KV cache 与 `output_ids` 一致。
- 如果调用方没有显式发送 budget reached，而是直接开始下一个 `streaming_prefill()`，框架需要在新 unit 开始前自行补齐未闭合结构。

推荐实时调用方式：

```python
model.fc_duplex.streaming_prefill(audio_waveform=audio_chunk, tool_responses=events)
spoken = model.fc_duplex.streaming_spoken_generate(...)

while caller_budget_remains():
    step = model.fc_duplex.streaming_non_spoken_generate(max_tokens=1)
    if step["terminated"]:
        break

if caller_budget_reached():
    model.fc_duplex.streaming_non_spoken_generate(close_reason="budget_reached")
else:
    model.fc_duplex.streaming_non_spoken_generate(close_reason="eos")

model.fc_duplex.finalize_unit()
```

### 未显式 budget_reached 的兜底闭合

调用方在线调度时，可能发现 non-spoken budget 已到，但直接进入下一次 `streaming_prefill()`，没有显式调用：

```python
streaming_non_spoken_generate(close_reason="budget_reached")
```

为保证 token 流合法，`streaming_prefill()` 开头必须调用内部兜底：

```python
def _ensure_previous_unit_closed(self):
    if self._non_spoken_slot_open:
        self.streaming_non_spoken_generate(close_reason="budget_reached")

    if self._current_unit_open:
        self.finalize_unit()
```

效果：

- 如果上一个 unit 的 `ai_non_spoken_slot` 还没闭合，自动插入 `<|non_spoken_budget_reached|>` 和 `</ai_non_spoken_slot>`。
- 如果上一个 `<unit>` 还没闭合，自动插入 `</unit>`。
- 然后再开始新的 `streaming_prefill()`。

这只是结构兜底，不代表模型侧接管 budget；budget 判断仍归调用方。

## 主模型 `MiniCPMO` 需要新增的设定

### 1. 初始化字段

在 `MiniCPMO.__init__()` 中新增：

```python
self.fc_duplex: Optional["FcDuplexCapability"] = None
self._fc_duplex_config = {
    "temperature": 0.7,
    "tool_format": "minicpm4_xml",
    "default_unit_sec": 1.0,
    "max_spoken_tokens_per_unit": 24,
    "extra_response_units": 4,
}
```

### 2. `init_unified()` 中创建新能力

在不影响旧 `self.duplex = DuplexCapability(...)` 的前提下，新增：

```python
self.fc_duplex = FcDuplexCapability(
    model=self,
    device=device,
    **self._fc_duplex_config,
)
```

### 3. 可选透传方法

为了上层调用统一，可以新增主模型透传方法：

- `fc_duplex_prepare(...)`
- `fc_duplex_streaming_prefill(...)`
- `fc_duplex_streaming_spoken_generate(...)`
- `fc_duplex_streaming_non_spoken_generate(...)`
- `fc_duplex_finalize_unit()`
- `fc_duplex_decode_output_ids(...)`
- `fc_duplex_cleanup()`

这些方法只做空值检查和转发，核心逻辑仍在 `FcDuplexCapability`。

## 新增 Schema

文件：`core/schemas/fc_duplex.py`

只定义 Pydantic 数据结构，不实现协议状态机。

建议类：

- `FcDuplexConfig`
- `FcToolResponse`
- `FcDuplexPrepareRequest`
- `FcDuplexPrefillRequest`
- `FcSpokenGenerateRequest`
- `FcNonSpokenGenerateRequest`
- `FcFinalizeUnitRequest`
- `FcDuplexOfflineInput`
- `FcDuplexOfflineOutput`
- `FcDuplexUnitInfo`
- `FcDuplexStepResult`
- `FcDuplexOutput`

重点字段：

- `system_prompt`
- `tools`
- `audio_data` / `audio_path`
- `frame_list` / `image_paths`
- `tool_responses`
- `tool_call_id`
- `tool_started`
- `tool_call_id_generator`
- `decode_mode`
- `max_spoken_tokens`
- `close_reason`
- `closed_spans`
- `non_spoken_budget_per_unit`
- `non_spoken_budget_listening`
- `non_spoken_budget_speaking`
- `output_ids`
- `output_render`
- `spoken_text`
- `think_text`
- `tool_calls`
- `units_info`

## 新增 Processor View

文件：`core/processors/unified.py`

新增：

```python
class FcDuplexView:
    def __init__(self, model, config=None): ...

    def prepare(self, request: FcDuplexPrepareRequest): ...
    def streaming_prefill(self, request: FcDuplexPrefillRequest): ...
    def streaming_spoken_generate(self, request: FcSpokenGenerateRequest): ...
    def streaming_non_spoken_generate(self, request: FcNonSpokenGenerateRequest): ...
    def finalize_unit(self, request: FcFinalizeUnitRequest): ...
    def decode_output(self, output_ids=None, tools=None): ...
    def offline_inference(
        self,
        task_input: FcDuplexOfflineInput,
        non_spoken_budget_per_unit: int = 12,
    ) -> FcDuplexOfflineOutput: ...
    def cleanup(self): ...
```

`UnifiedProcessor` 新增：

- `_fc_duplex_view`
- `set_fc_duplex_mode()`
- `fc_duplex` property

是否新增 `ProcessorMode.FC_DUPLEX` 可按服务层需求决定：

- 如果只是本地脚本调用，可不新增 mode，直接 `processor.fc_duplex`。
- 如果要接入 worker/gateway 调度，建议新增 `FC_DUPLEX`，避免和旧 `DUPLEX` 混淆。

## Tool Call ID 与 `tool_started` 事件管理

该能力放在 `FcDuplexView` 层实现，不下沉到 `FcDuplexCapability`。底层模型只负责生成和解析 `<tool_call>...</tool_call>`；View 层负责 ID 分配、状态维护、input event 注入和调用方 response 校验。

当 `streaming_non_spoken_generate()` 闭合一个 tool call span 后：

1. View 层调用 `tool_call_id_generator` 生成唯一 `tool_call_id`。
2. 把该 ID 附加到本次 step 返回的 tool call 结果中。
3. 记录该 ID 的状态为待发送 started 事件。
4. 在下一次 `streaming_prefill()` 的 `input_event_slot` 中自动注入标准 created/started 事件：

```text
<input_event_slot>
  <tool_response_event>
    <tool_call_id>call_0</tool_call_id>
    <|tool_started|>
  </tool_response_event>
</input_event_slot>
```

这是 SDK / 协议标准结构，不用自然语言描述“创建成功”。对应 semantic token 是 `TOOL_STARTED`。

### ID 生成器

View 层新增 ID generator 概念：

```python
class ToolCallIdGenerator:
    def next_id(self) -> str: ...
```

默认实现可以先用简单递增：

```text
fc_call_000001
fc_call_000002
...
```

后续 SDK 如果提供默认 generator，可替换为 SDK 实现。

离线 overfit / 训推一致测试时，`prepare()` 或 `offline_inference()` 支持传入自定义 generator，按训练数据中的 ID 顺序消费：

```text
dob_dev_plan_0001_call_01
dob_dev_plan_0001_call_02
dob_dev_plan_0001_call_03
dob_dev_plan_0001_call_04
```

这样能让推理时生成的 tool call 与训练数据里的 tool response ID 对齐。

### Tool Call 状态管理器

View 层维护 `ToolCallStateManager`：

```python
class ToolCallState:
    id: str
    tool_call: dict | None
    parse_error: str | None
    started_sent: bool
    response_received: bool

class ToolCallStateManager:
    def register_tool_call(tool_call: dict) -> str: ...
    def register_parse_error(error: str, wire: str) -> str: ...
    def consume_pending_started_events() -> list[dict]: ...
    def validate_tool_response(tool_call_id: str) -> None: ...
    def mark_response_received(tool_call_id: str) -> None: ...
```

规则：

- 同一个上下文内不能生成重复 ID。
- 调用方不能返回未知 ID 的 tool response。
- 调用方不能对同一个 ID 返回多次 response。
- 违反规则时直接 `raise ValueError`，错误信息要说明原因，例如：
  - `unknown tool_call_id: ...`
  - `duplicate tool_response for tool_call_id: ...`
  - `duplicate generated tool_call_id: ...`

外部调用者可以 `try/except` 捕获这些错误，然后进入 `finalize` 流程终止 session。

### `tool_started` 自动注入

`FcDuplexView.streaming_prefill()` 在调用底层 `FcDuplexCapability.streaming_prefill()` 前，需要合并两类 input event：

1. 调用方传入的 `tool_responses`。
2. View 层状态管理器中待发送的 `tool_started` 事件。

伪代码：

```python
def streaming_prefill(request):
    pending_started = self.tool_call_manager.consume_pending_started_events()
    checked_responses = self.tool_call_manager.validate_and_mark_responses(request.tool_responses)
    input_events = pending_started + checked_responses
    return self._model.fc_duplex_streaming_prefill(
        audio_waveform=...,
        tool_responses=input_events,
    )
```

底层协议 helper 需要能编码两种事件：

```text
started event:
  <tool_response_event>
    <tool_call_id>id</tool_call_id>
    <|tool_started|>
  </tool_response_event>

response event:
  <tool_response_event>
    <tool_call_id>id</tool_call_id>
    <tool_response>content</tool_response>
  </tool_response_event>
```

### 解析失败处理

如果 SDK 解析 `<tool_call>...</tool_call>` XML 失败：

- 当前 step 不把它作为有效 tool call 返回给调用方执行。
- View 层仍然生成一个 `tool_call_id` 并登记该失败调用。
- 下一次 `streaming_prefill()` 仍然可以发送 `<|tool_started|>`，表示框架已接收这个调用生命周期。
- 同时构造一个失败 `tool_response`，使用同一个 ID，把错误信息用自然语言描述放在 `<tool_response>` content 中，例如：

```text
工具调用解析失败，无法执行该工具调用。错误信息：...
```

当前没有专用的 `tool_create_failed` token，失败也统一走 `tool_response`：

```text
<tool_response>{"ok": false, "error": "SDK deserialize failed: ..."}</tool_response>
```

这个 content 格式先使用临时结构，后续由 SDK 补标准错误 schema 后再替换。编码 content 时仍应走 SDK 的 `normalize_tool_response_content(...)`。

### Step 返回结构扩展

`FcDuplexView.streaming_non_spoken_generate()` 返回给调用方的 `closed_spans` 中，tool call span 需要附带框架分配的 ID：

```python
{
    "type": "tool_call",
    "tool_call_id": "fc_call_000001",
    "wire": "<function ...></function>",
    "tool_call": {"name": "...", "arguments": {...}, "error": None},
}
```

如果解析失败：

```python
{
    "type": "tool_call",
    "tool_call_id": "fc_call_000002",
    "wire": "<function malformed ...>",
    "tool_call": None,
    "error": "SDK deserialize failed: ...",
}
```

调用方不应该执行解析失败的 tool call；它只需要知道该 step 有一个解析失败事件。框架会在后续 input event 中写回错误 `tool_response`。

## `offline_inference()` 设计

`offline_inference()` 放在 `FcDuplexView`，作为测试、演示、离线批处理入口。底层 `FcDuplexCapability` 只保留实时原语。

建议签名：

```python
def offline_inference(
    self,
    task_input: FcDuplexOfflineInput,
    non_spoken_budget_per_unit: int = 12,
) -> FcDuplexOfflineOutput:
    ...
```

简化 budget 策略：

- 每个 unit 最多调用 `k = non_spoken_budget_per_unit` 次 `streaming_non_spoken_generate(max_tokens=1)`。
- 如果 k 次内自然终止，则直接进入 `finalize_unit()`。
- 如果 k 次后仍未自然终止，则调用 `streaming_non_spoken_generate(close_reason="budget_reached")`，强制插入 `<|non_spoken_budget_reached|>` 并闭合 non-spoken slot。

离线流程：

```text
prepare()
for each audio unit:
    streaming_prefill(audio_chunk, tool_responses_for_this_unit)
    spoken = streaming_spoken_generate()

    for _ in range(non_spoken_budget_per_unit):
        step = streaming_non_spoken_generate(max_tokens=1)
        if step.terminated:
            break

    if not step.terminated:
        streaming_non_spoken_generate(close_reason="budget_reached")

    finalize_unit()

decode_output()
return FcDuplexOfflineOutput
```

后续可扩展为 listen/speak 双预算：

```python
def offline_inference(
    self,
    task_input: FcDuplexOfflineInput,
    non_spoken_budget_listening: int = 12,
    non_spoken_budget_speaking: int = 6,
) -> FcDuplexOfflineOutput:
    ...
```

选择预算：

```python
k = non_spoken_budget_speaking if spoken["is_speaking"] else non_spoken_budget_listening
```

初版建议先实现 `non_spoken_budget_per_unit`，后续再扩展 listen/speak 双预算。

## 新增脚本

### 1. `demo_fc_duplex.py`

放在项目根目录。

用途：

- `mock`：验证 FC 协议编码/解码与训练数据 token 流一致。
- `real`：加载模型后跑真实前向 smoke test。

可以从 `project/MiniCPM-o-4_5-fc_duplex_infer/demo_fc_duplex.py` 迁移，但需要改 import 到当前项目的新类。

### 2. `tests/test_fc_duplex_protocol.py`

测试内容：

- system prefill token 是否可构造。
- unit input slot 是否可构造。
- tool response 是否可编码。
- output ids 是否可 decode 成 spoken / think / tool_calls。

### 3. `tests/test_fc_duplex_capability.py`

测试内容：

- `prepare()` 后 decoder 有 cache。
- `streaming_prefill()` 能喂音频占位。
- `streaming_spoken_generate()` 返回 listen/speak 状态。
- `streaming_non_spoken_generate(max_tokens=1)` 能返回 token。
- `streaming_non_spoken_generate()` 在闭合 `</think>` 时返回完整 think 解析结果。
- `streaming_non_spoken_generate()` 在闭合 `</tool_call>` 时返回完整 tool call 解析结果。
- `streaming_non_spoken_generate(close_reason="budget_reached")` 会插入 budget reached token 并闭合 non-spoken slot。
- 未显式发送 budget reached 时，下一次 `streaming_prefill()` 会自动补 `<|non_spoken_budget_reached|>`、闭合 non-spoken slot，并 finalize 上一个 unit。
- `finalize_unit()` 会插入 `</unit>`。

### 4. `tests/test_fc_duplex_offline.py`

测试内容：

- `offline_inference(non_spoken_budget_per_unit=k)` 能完整跑通。
- 每个 unit 最多生成 k 次 non-spoken step。
- 超过 k 后会插入 budget reached。
- 输出可通过 `decode_output()` 解析。

### 5. `tests/test_fc_duplex_tool_call_manager.py`

测试内容：

- tool call 闭合后自动分配唯一 `tool_call_id`。
- 下一次 `streaming_prefill()` 自动注入 `<|tool_started|>` 事件。
- 调用方返回未知 ID 的 tool response 会 raise。
- 调用方重复返回同一个 ID 的 tool response 会 raise。
- generator 生成重复 ID 会 raise。
- SDK 解析 tool call XML 失败时，框架生成同 ID 的错误 `tool_response` content。
- offline 模式可以使用训练数据中的 tool call ID generator，按顺序消费固定 ID。

## 服务层后续接入

如果要接入现有后端 runtime，需要新增独立 endpoint / message type，不复用旧 `/duplex`：

- worker backend 新增 `fc_duplex_*` 方法。
- runtime protocol 新增 request type：`fc_duplex` 或 `omni_fc_duplex`。
- gateway 能力声明新增 `fc_duplex`。
- 前端按新协议驱动 unit、budget、tool response 注入。

## 开发顺序

1. 在 `core/schemas/fc_duplex.py` 定义 FC 双工请求/响应 schema。
2. 在模型层新增 `FcDuplexCapability`，先迁移协议 helper 和 `prepare()`。
3. 加 `streaming_prefill()`，打通 system prefill + user audio slot + input event slot。
4. 加 `streaming_spoken_generate()`。
5. 加 `streaming_non_spoken_generate(max_tokens=1 / close_reason=...)`。
6. 加 non-spoken 在线 span 状态机：闭合 `</think>` / `</tool_call>` 时返回解析结果。
7. 加 `_ensure_previous_unit_closed()`，让下一次 `streaming_prefill()` 能兜底补 budget reached 和 finalize。
8. 加 `finalize_unit()`、`decode_output_ids()`、`cleanup()`。
9. 在 `MiniCPMO.__init__()` / `init_unified()` 中挂载 `self.fc_duplex`。
10. 在 `core/processors/unified.py` 新增 `FcDuplexView`。
11. 在 `FcDuplexView` 新增 `offline_inference(non_spoken_budget_per_unit=k)`。
12. 在 `FcDuplexView` 新增 tool call ID manager：
    - tool call 闭合后分配唯一 ID。
    - 下个 unit 自动注入 `<|tool_started|>`。
    - 校验外部 tool response ID 合法性。
    - 解析失败时写回错误 `tool_response`。
13. 迁移 `demo_fc_duplex.py`，先跑 mock，再跑 real smoke test。
14. 补协议测试、capability 状态测试、span 闭合测试、兜底闭合测试、tool call manager 测试和 offline 测试。
15. 如需线上服务，最后接 worker/gateway/frontend。

## 风险与注意事项

- 新协议 special token 可能超出当前 embedding size，需要在真实前向前 resize。
- 新增 embedding 行如果未训练，真实生成只适合 smoke test，不代表最终效果。
- `schemas` 不应依赖 SDK tokenizer，否则 API 层会变重。
- `FcDuplexCapability` 应使用独立 decoder，不复用旧 `DuplexCapability.decoder`。
- 不要让旧 `set_duplex_mode()` 切到新协议，避免旧前端和旧测试行为变化。
- budget 由调用方管理时，需要明确调用顺序，否则容易出现 slot 未闭合或 unit 未 finalize 的状态错误。
- `offline_inference()` 内部的 `non_spoken_budget_per_unit=k` 只是离线便捷策略，不代表实时调用方必须使用同样策略。
- 即使调用方不显式发送 `budget_reached`，框架也必须保证下一次 `streaming_prefill()` 前自动补齐结构 token，避免破坏训练/推理协议格式。
- 在线返回 tool call 解析结果时要保留原始 `wire` 文本和解析错误信息，避免反序列化失败导致工具调用丢失。
- `<|tool_started|>` 是标准 input event，不要用自然语言描述工具创建成功。
- 工具创建/解析失败当前没有专用 token，失败信息先作为同 ID 的 `<tool_response>` content 写回。
- Tool call ID 属于 View 层运行时状态，不应由模型生成，也不应由调用方自行决定。
