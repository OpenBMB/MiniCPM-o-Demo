# FC Slot 双工推理说明与使用文档

本文档说明 `MiniCPM-o-Demo-FC` 中新增的 FC slot 双工推理链路。该链路与原有 `DuplexView` / `DuplexCapability` 并行存在，不改变旧双工推理接口。

## 入口与文件

主要入口：

- `core.processors.UnifiedProcessor`
- `processor.fc_duplex`
- `core.schemas.fc_duplex`

主要实现文件：

- `core/schemas/fc_duplex.py`：FC 双工请求、配置、返回值 schema。
- `core/processors/unified.py`：`FcDuplexView`，负责上层 API 封装、音频读取、tool call id 管理、离线推理。
- `MiniCPMO45/modeling_minicpmo_unified.py`：`FcDuplexCapability`，负责 FC slot 协议 token、KV cache、spoken/non-spoken 生成。
- `test_fc_duplex_overfit.py`：基于 overfit checkpoint 和训练样本的 smoke test。

## 整体调用模型

FC 双工协议把一次实时交互拆成连续的 `unit`。每个 unit 内大致包含：

```text
<unit>
  <user_video_slot>...</user_video_slot>      # 可选
  <user_audio_slot>...</user_audio_slot>      # 可选
  <input_event_slot>...</input_event_slot>    # 可选，主要放工具事件/工具结果
  <ai_spoken_slot>...</ai_spoken_slot>
  <ai_non_spoken_slot>...</ai_non_spoken_slot>
</unit>
```

上层调用方负责 unit 调度和 non-spoken budget 管理；模型层只负责根据调用方指令继续生成或闭合 slot。

## 初始化

```python
from core.processors import UnifiedProcessor

processor = UnifiedProcessor(
    model_path="/path/to/base/model",
    pt_path="/path/to/checkpoint.pt",
    device="cuda",
    compile=False,
    attn_implementation="sdpa",
)

fc = processor.fc_duplex
```

`UnifiedProcessor` 初始化后会在主模型上挂载 `MiniCPMO.fc_duplex`，上层通过 `processor.fc_duplex` 获取 `FcDuplexView`。

## 在线推理流程

### 1. Prepare

```python
from core.schemas.fc_duplex import FcDuplexPrepareRequest

fc.prepare(
    FcDuplexPrepareRequest(
        system_prompt=system_prompt,
        tools=tools,
    )
)
```

`prepare()` 会：

- 重置 FC 双工状态。
- 写入 system prompt。
- 写入工具定义。
- 初始化 tool call id 状态管理器。

### 2. 每个 unit 输入 prefill

```python
from core.schemas.fc_duplex import FcDuplexPrefillRequest

fc.streaming_prefill(
    FcDuplexPrefillRequest(
        audio_path="/path/to/audio.wav",
        tool_responses=None,
        sample_rate=16000,
    )
)
```

`streaming_prefill()` 支持：

- `audio_path`：音频文件路径。
- `audio_data`：base64 编码的 float32 PCM。
- `frame_list`：预留视频/图像输入。
- `tool_responses`：调用方返回的工具执行结果。

如果上一 unit 里模型刚生成了 tool call，`FcDuplexView` 会在本次 prefill 自动注入对应的 `tool_started` 事件：

```text
<input_event_slot>
  <tool_response_event>
    <tool_call_id>...</tool_call_id>
    <|tool_started|>
  </tool_response_event>
</input_event_slot>
```

### 3. 生成 spoken slot

```python
from core.schemas.fc_duplex import FcSpokenGenerateRequest

spoken_result = fc.streaming_spoken_generate(
    FcSpokenGenerateRequest(
        max_tokens=24,
        decode_mode="greedy",
    )
)
```

该方法生成当前 unit 的 `ai_spoken_slot`。模型会选择 `<|listen|>` 或 `<|speak|>...`。

### 4. 生成 non-spoken slot

```python
from core.schemas.fc_duplex import FcNonSpokenGenerateRequest

budget = 12
terminated = False
for _ in range(budget):
    step = fc.streaming_non_spoken_generate(
        FcNonSpokenGenerateRequest(
            max_tokens=1,
            decode_mode="greedy",
        )
    )
    if step.closed_spans:
        for span in step.closed_spans:
            print(span.type, span.tool_call_id, span.tool_call)
    if step.terminated:
        terminated = True
        break

if not terminated:
    fc.streaming_non_spoken_generate(
        FcNonSpokenGenerateRequest(
            max_tokens=0,
            close_reason="budget_reached",
        )
    )
```

`streaming_non_spoken_generate()` 会在 `<think>` 或 `<tool_call>` 闭合时，把闭合 span 放到 `step.closed_spans` 里返回。

对 tool call span，View 层会自动：

- 生成唯一 `tool_call_id`。
- 把 `tool_call_id` 写入 `FcClosedSpan.tool_call_id`。
- 把 `tool_call_id` 写入解析后的 `tool_call` dict。
- 排队在下一次 `streaming_prefill()` 中注入 `<|tool_started|>`。

如果调用方达到 non-spoken budget，可以用 `close_reason="budget_reached"` 强制闭合当前 non-spoken slot。

### 5. Finalize unit

```python
unit_info = fc.finalize_unit()
```

`finalize_unit()` 返回当前 unit 的摘要信息，包括：

- `unit`
- `n_audio`
- `has_event`
- `is_listen`
- `is_speaking`
- `spoken_ids`
- `non_spoken_ids`
- `non_spoken_terminator`
- `closed_spans`

### 6. Decode output

```python
decoded = fc.decode_output(tools=tools)
print(decoded["output_render"])
print(decoded["spoken_text"])
print(decoded["think_text"])
print(decoded["tool_calls"])
```

`output_render` 是完整 token 流的可读渲染，适合调试协议结构。

## Tool Call 生命周期

当前实现中，tool call 生命周期由 `FcDuplexView` 管理。

### 模型生成 tool call

模型在 `ai_non_spoken_slot` 中生成：

```text
<tool_call>
  <function name="display_object_on_board">
    <param name="name">咖啡杯</param>
  </function>
</tool_call>
```

当 `</tool_call>` 闭合时，`streaming_non_spoken_generate()` 会返回类似：

```python
FcClosedSpan(
    type="tool_call",
    tool_call_id="fc_call_000001",
    wire='<function name="display_object_on_board"><param name="name">咖啡杯</param></function>',
    tool_call={
        "tool_call_id": "fc_call_000001",
        "wire": "...",
        "name": "display_object_on_board",
        "arguments": {"name": "咖啡杯"},
        "error": None,
    },
)
```

### 下一 unit 注入 tool_started

下一个 `streaming_prefill()` 会自动注入：

```text
<input_event_slot>
  <tool_response_event>
    <tool_call_id>fc_call_000001</tool_call_id>
    <|tool_started|>
  </tool_response_event>
</input_event_slot>
```

这表示框架已经接收/创建了该工具调用。工具后续执行成功或失败，都不影响这个 started 事件的存在。

### 调用方返回工具执行结果

调用方执行工具后，可以在后续 unit 传入：

```python
from core.schemas.fc_duplex import FcToolResponse, FcDuplexPrefillRequest

fc.streaming_prefill(
    FcDuplexPrefillRequest(
        audio_data=audio_data,
        tool_responses=[
            FcToolResponse(
                call_id="fc_call_000001",
                content={"ok": True, "result": "..."},
            )
        ],
    )
)
```

View 层会校验：

- `call_id` 必须是框架已生成过的 ID。
- 同一个 `call_id` 不能重复提交 tool response。

如果传入未知 ID 或重复响应，会抛出 `ValueError`。

### tool call 解析失败

如果模型生成的 `<tool_call>` XML 无法被 SDK 解析，框架仍会：

- 分配一个 `tool_call_id`。
- 在下一 unit 注入 `<|tool_started|>`。
- 排队注入一个错误 `tool_response`，content 使用自然语言描述，形如：

```text
工具调用解析失败，无法执行该工具调用。错误信息：...
```

## 离线推理

`FcDuplexView.offline_inference()` 是便捷封装，适合调试、overfit 测试和协议验证。

```python
from core.schemas.fc_duplex import FcDuplexConfig, FcDuplexOfflineInput

result = fc.offline_inference(
    FcDuplexOfflineInput(
        system_prompt=system_prompt,
        tools=tools,
        user_audio_path="/path/to/user_audio.opus",
        tool_call_ids=[
            "sample_call_01",
            "sample_call_02",
        ],
        config=FcDuplexConfig(
            decode_mode="greedy",
            non_spoken_budget_per_unit=10000,
            extra_response_units=4,
        ),
    ),
    non_spoken_budget_per_unit=10000,
)
```

离线推理会：

1. 调用 `prepare()`。
2. 按 `unit_sec` 切分音频。
3. 每个 unit 依次执行 `streaming_prefill()`、`streaming_spoken_generate()`、`streaming_non_spoken_generate()`。
4. 达到 budget 时用 `budget_reached` 强制闭合 non-spoken slot。
5. 每个 unit 调用 `finalize_unit()`。
6. 最后调用 `decode_output()` 并返回 `FcDuplexOfflineOutput`。

`tool_call_ids` 是可选字段。传入后会使用固定 ID generator，适合让 overfit 推理结果和训练数据中的 tool_call_id 对齐。

## Overfit Smoke Test

项目根目录提供了 `test_fc_duplex_overfit.py`。

默认路径：

- base model：`/user/heweiquan/project/MiniCPM-o-4_5`
- checkpoint：`/user/heweiquan/models/minicpm-o45-fc-overfit/minicpm-v_100.pt`
- data：`/user/heweiquan/project/MiniCPM-o-4_5-fc_duplex_infer/data/training_data/dob_dev_plan_0001.json`

运行：

```bash
/user/heweiquan/envs/miniconda3/envs/minicpm-o4_5/bin/python \
  /user/heweiquan/project/MiniCPM-o-Demo-FC/test_fc_duplex_overfit.py \
  --budget 10000
```

常用参数：

- `--model-path`：base model 路径。
- `--pt-path`：额外 checkpoint 路径。
- `--data`：训练样本 JSON。
- `--device`：默认 `cuda`。
- `--attn-implementation`：默认 `sdpa`。
- `--budget`：每个 unit 的 non-spoken 生成预算。
- `--extra-response-units`：音频结束后额外补的静音响应 unit 数。
- `--decode-mode`：`greedy` 或 sampling 模式。

输出中重点看：

- `[result].success`
- `spoken_text`
- `think_text`
- `tool_calls`
- `[render head]` 中的 token 流

如果功能正常，tool call 后的下一 unit 会出现：

```text
<input_event_slot><tool_response_event><tool_call_id>...</tool_call_id><|tool_started|></tool_response_event></input_event_slot>
```

## 结果字段说明

`FcDuplexStepResult`：

- `token_ids`：本步生成或插入的 token id。
- `terminated`：当前 slot 是否已经结束。
- `close_reason`：结束原因，如 `eos`、`no_action`、`budget_reached`。
- `closed_spans`：本步闭合的 `<think>` 或 `<tool_call>`。
- `text`：本步可解码文本。
- `metadata`：底层返回的其他调试字段。

`FcClosedSpan`：

- `type`：`think` 或 `tool_call`。
- `tool_call_id`：框架分配的 tool call ID，仅 tool call span 使用。
- `text`：think 文本。
- `wire`：tool call 原始 XML 文本。
- `tool_call`：SDK 解析后的结构化 tool call。
- `error`：解析或状态管理错误。

`FcDuplexUnitInfo`：

- `unit`：unit 下标。
- `n_audio`：本 unit 音频 placeholder 数。
- `has_event`：是否包含 input event。
- `is_listen`：spoken slot 是否选择 listen。
- `is_speaking`：spoken slot 是否处于 speak。
- `spoken_ids`：spoken slot token。
- `non_spoken_ids`：non-spoken slot token。
- `non_spoken_terminator`：non-spoken 结束 token 对应原因。
- `closed_spans`：该 unit 内闭合的 span。

## 注意事项

- `non_spoken_budget_per_unit` 由调用方管理。在线服务中建议每个 unit 自己循环调用 `streaming_non_spoken_generate(max_tokens=1)`，达到预算后显式传 `close_reason="budget_reached"`。
- 如果调用方忘记闭合上一 unit，底层 `FcDuplexCapability` 会在下一次 prefill 前尝试自动闭合，保证 token 流结构合法。
- 当前 `tool_started` 是框架自动注入的创建/启动事件；真正工具执行完成后的结果仍需要调用方通过 `tool_responses` 传入。
- 当前成功启动事件默认只注入 `<|tool_started|>`，没有额外构造成功 content。若协议后续要求“成功/失败都必须有 `<tool_response>` content”，需要在 `ToolCallStateManager.consume_pending_started_events()` 附近扩展成功 content。
- IDE 可能提示无法解析 `minicpm_o5_sdk`，但只要使用 `/user/heweiquan/envs/miniconda3/envs/minicpm-o4_5` 环境运行即可。
