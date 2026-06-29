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
- `evaluate_fc_duplex_batch.py`：批量训练数据验证 runner，内部调用 `FcDuplexView.offline_inference_from_train_data()`。

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

## 主要接口输入输出

正规化后，`FcDuplexView` 对调用方公开的方法都使用 `core.schemas.fc_duplex` 中的 Pydantic 模型表达输入和输出。底层 `FcDuplexCapability` 内部仍可能使用临时 dict，但不会直接暴露给上层调用方。

### `prepare(request) -> FcDuplexPrepareResult`

输入类型：`FcDuplexPrepareRequest`

- `system_prompt: str`：系统提示词。
- `tools: list[dict] | None`：OpenAI tool definition 形式的工具定义。
- `ref_audio_path: str | None`：可选参考音频，会以 16kHz mono 加载并作为 reference audio embedding 喂给 LLM。
- `prompt_wav_path: str | None`：可选 Token2Wav prompt 音频路径，用于初始化 TTS vocoder streaming cache。
- `generate_audio: bool`：是否启用 spoken TTS waveform 生成。

输出类型：`FcDuplexPrepareResult`

- `prefill_ids: list[int]`：system/tool prefill token ids。
- `output_render: str`：prefill token stream 的可读渲染。
- `resized: bool`：是否因为 FC special token 扩展了 embedding 表。
- `old_vocab_size/new_vocab_size/required_vocab_size`：embedding resize 相关信息。
- `generate_audio: bool`：本 session 是否开启 TTS。
- `has_ref_audio: bool`：是否实际加载并喂入了 reference audio。
- `prompt_wav_path: str | None`：实际使用的 Token2Wav prompt path。

### `streaming_prefill(request) -> FcDuplexPrefillResult`

输入类型：`FcDuplexPrefillRequest`

- `audio_path: str | None`：当前 unit 的用户音频文件路径。
- `audio_data: str | None`：base64 编码的 float32 PCM 音频。
- `frame_list: list[Any] | None`：预留图像/视频帧输入。
- `tool_responses: list[FcToolResponse] | None`：调用方传回的工具执行结果。
- `sample_rate: int`：输入音频采样率，默认 16000。

输出类型：`FcDuplexPrefillResult`

- `unit_index: int`：当前 unit 下标。
- `n_audio_placeholders: int`：当前 unit 写入的用户音频 placeholder embedding 数。
- `has_input_event: bool`：是否写入 input event slot。
- `is_listen: bool | None`：当前 unit 的 listen 状态，prefill 阶段通常还未知。
- `is_speaking: bool`：当前 unit 是否已标记为 speaking。
- `inserted_token_ids: list[int]`：预留字段，表示 prefill 显式插入的 token ids。

### `streaming_spoken_generate(request) -> FcSpokenGenerateResult`

输入类型：`FcSpokenGenerateRequest`

- `max_tokens: int`：当前 unit 的 spoken slot 最大生成 token 数。
- `decode_mode: str`：解码模式，常用 `greedy`。

输出类型：`FcSpokenGenerateResult`

- `is_listen: bool`：模型是否选择 `<|listen|>`。
- `is_speaking: bool`：模型是否选择 `<|speak|>`。
- `spoken_token_ids: list[int]`：当前 spoken slot 生成的 token ids。
- `spoken_text: str`：当前 unit 生成的 spoken 文本。
- `spoken_turn_eos: bool`：是否生成 spoken turn 结束。
- `audio_waveform: Any | None`：如果 `generate_audio=True`，这里可能返回 24kHz float32 waveform。
- `audio_sample_rate: int | None`：`audio_waveform` 的采样率，当前为 24000。
- `n_audio_samples: int`：生成 waveform 的采样点数。
- `n_tts_tokens: int`：TTS audio token 数。
- `cost_llm/cost_tts_prep/cost_tts/cost_token2wav: float`：分阶段耗时。

### `streaming_non_spoken_generate(request) -> FcNonSpokenGenerateResult`

输入类型：`FcNonSpokenGenerateRequest`

- `max_tokens: int`：本次最多生成多少个 non-spoken token。在线建议每次传 `1`。
- `decode_mode: str`：解码模式，常用 `greedy`。
- `close_reason: FcNonSpokenCloseReason | None`：强制闭合原因，可选 `eos`、`no_action`、`budget_reached`、`hold`、`abort`。

输出类型：`FcNonSpokenGenerateResult`

- `token_ids: list[int]`：本次生成或插入的 token ids。
- `terminated: bool`：当前 non-spoken slot 是否已经结束。
- `close_reason: str | None`：结束原因。
- `generation_flag: NonSpokenStepGenerationFlag`：明确的调用方循环控制信号。
- `closed_spans: list[FcClosedSpan]`：本次闭合的 `<think>` 或 `<tool_call>` span。
- `text: str`：本次普通文本 token 的解码文本。
- `audio_waveform/audio_sample_rate/n_tts_tokens`：从通用 step result 继承，目前 non-spoken 阶段通常不用。
- `metadata: dict`：底层调试信息。

`NonSpokenStepGenerationFlag`：

- `continue_non_spoken_generation`：当前 non-spoken slot 尚未结束，调用方可继续 step generate。
- `no_action`：模型生成了 no-action 语义，当前 non-spoken slot 可结束。
- `non_spoken_slot_eos`：当前 non-spoken slot 已结束。它是 API 控制状态，不是 SDK special token 名；可能对应 `<|non_spoken_eos|>`、`</ai_non_spoken_slot>` 或调用方强制闭合。

`FcClosedSpan` 字段含义：

- `type: "think" | "tool_call"`：闭合 span 类型。
- `tool_call_id: str | None`：框架分配的 tool call id。
- `text: str | None`：think span 的完整文本。
- `wire: str | None`：tool call 原始 XML/wire 文本。
- `tool_call: dict | None`：SDK 解析出的工具调用结构。
- `error: str | None`：解析失败或状态错误。

### `finalize_unit(request=None) -> FcDuplexUnitInfo`

输入类型：`FcFinalizeUnitRequest | None`

输出类型：`FcDuplexUnitInfo`

- `unit: int`：unit 下标。
- `n_audio: int`：用户音频 placeholder 数。
- `has_event: bool`：是否包含 input event。
- `is_listen: bool | None`：spoken slot 是否选择 listen。
- `is_speaking: bool`：spoken slot 是否选择 speak。
- `spoken_ids: list[int]`：spoken slot token ids。
- `non_spoken_ids: list[int]`：non-spoken slot token ids。
- `non_spoken_terminator: str | None`：non-spoken 结束原因。
- `closed_spans: list[FcClosedSpan]`：当前 unit 中闭合的 think/tool_call span。
- `audio_sample_rate/n_audio_samples`：如果生成了 TTS waveform，会记录音频信息。

### `decode_output(request) -> FcDecodeOutputResult`

输入类型：`FcDecodeOutputRequest`

- `output_ids: list[int] | None`：要解码的 token ids；不传则解码当前 session 累计输出。
- `tools: list[dict] | None`：用于反序列化 tool call 的工具定义。

输出类型：`FcDecodeOutputResult`

- `units: list[FcDecodedUnit]`：按 unit 解码后的结构。
- `spoken_text: str`：所有 spoken slot 文本拼接。
- `think_text: str`：所有 think span 文本拼接。
- `tool_calls: list[FcDecodedToolCall]`：解码后的 tool call。
- `output_ids: list[int]`：完整 token ids。
- `output_render: str`：完整 token stream 可读渲染。

## 在线推理流程

### 1. Prepare

```python
from core.schemas.fc_duplex import FcDuplexPrepareRequest

prepare_result = fc.prepare(
    FcDuplexPrepareRequest(
        system_prompt=system_prompt,
        tools=tools,
        ref_audio_path="/path/to/ref.wav",
        prompt_wav_path="/path/to/short_prompt.wav",
        generate_audio=True,
    )
)
```

`prepare()` 会：

- 重置 FC 双工状态。
- 写入 system prompt。
- 写入工具定义。
- 加载并喂入 reference audio。
- 初始化 Token2Wav prompt cache，用于后续 spoken TTS waveform 生成。
- 初始化 tool call id 状态管理器。

### 2. 每个 unit 输入 prefill

```python
from core.schemas.fc_duplex import FcDuplexPrefillRequest

prefill_result = fc.streaming_prefill(
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
from core.schemas.fc_duplex import FcDecodeOutputRequest

decoded = fc.decode_output(FcDecodeOutputRequest(tools=tools))
print(decoded.output_render)
print(decoded.spoken_text)
print(decoded.think_text)
print(decoded.tool_calls)
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

## 离线推理接口

### `offline_inference(task_input) -> FcDuplexOfflineOutput`

`FcDuplexView.offline_inference()` 是直接面向推理参数的离线封装，适合调试、overfit 测试和协议验证。

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

输入类型：`FcDuplexOfflineInput`

- `system_prompt: str`：system prompt 文本。
- `tools: list[dict] | None`：工具定义。
- `user_audio_path: str | None`：用户输入音频路径。
- `audio_data: str | None`：base64 float32 PCM 音频。
- `ref_audio_path: str | None`：TTS/reference audio 路径。
- `prompt_wav_path: str | None`：Token2Wav prompt 路径。
- `generate_audio: bool`：是否生成 TTS waveform。
- `tool_responses_by_unit: dict[int, list[FcToolResponse]]`：按 unit 固定注入的 tool response。
- `tool_responses_by_call_id: dict[str, Any]`：按 tool call id 动态注入的 GT tool response。
- `tool_call_ids: list[str] | None`：固定 tool call id 列表，用于训练数据一致性验证。
- `config: FcDuplexConfig`：离线推理配置。

输出类型：`FcDuplexOfflineOutput`

- `success: bool`：推理是否成功。
- `error: str | None`：失败信息。
- `output_ids/output_render`：预测 token ids 和可读 token stream。
- `spoken_text/think_text`：解码后的 spoken/think 文本。
- `tool_calls: list[FcDecodedToolCall]`：预测 tool calls。
- `units_info: list[FcDuplexUnitInfo]`：每个 unit 的结构化摘要。
- `audio_waveforms: list[Any]`：生成的 24kHz waveform 列表，仅 TTS 开启时可能有值。
- `total_units/n_audio_units/total_duration_ms`：推理统计信息。

### `offline_inference_from_train_data(request) -> FcDuplexTrainDataResult`

`offline_inference_from_train_data()` 是正规化后的训练数据验证入口。调用方不需要自己从训练 JSON 里拆 system prompt、tools、user audio、tool call ids、tool responses，也不需要自己做 GT tokenization 和对比。

```python
from core.schemas.fc_duplex import FcDuplexConfig, FcDuplexTrainDataRequest

result = fc.offline_inference_from_train_data(
    FcDuplexTrainDataRequest(
        train_data_path="/path/to/dob_dev_plan_0001.json",
        config=FcDuplexConfig(
            decode_mode="greedy",
            extra_response_units=0,
        ),
        generate_audio=True,
        ref_audio_path="/path/to/ref.wav",
        prompt_wav_path="/path/to/short_prompt.wav",
        output_artifact_dir="/path/to/output/dob_dev_plan_0001",
    )
)
```

输入类型：`FcDuplexTrainDataRequest`

- `train_data_path: str | None`：训练数据 JSON 路径。
- `train_data: Any | None`：已经加载好的训练数据结构或 SDK 对象。
- `data_root: str | None`：媒体目录。传 `train_data_path` 时默认使用训练 JSON 所在目录。
- `config: FcDuplexConfig`：离线推理配置。
- `non_spoken_budget_per_unit: int | None`：debug override。默认不传，使用 SDK train data 的 per-unit listening/speaking budget。
- `generate_audio: bool`：是否生成 TTS 音频。
- `ref_audio_path: str | None`：TTS/reference audio。
- `prompt_wav_path: str | None`：Token2Wav prompt。建议使用短参考音频。
- `output_artifact_dir: str | None`：输出目录；传入后会写入 `source.json`、GT/pred token stream、`units_info.json`、`comparison.json` 和可选音频文件。
- `use_train_tool_call_ids: bool`：是否使用训练数据里的 tool call id 作为固定 id generator。
- `inject_train_tool_responses: bool`：是否从训练数据里提取 tool response，并在对应 tool call 闭合后的下一 unit 动态注入。

输出类型：`FcDuplexTrainDataResult`

- `sample_id`：样本 id。
- `success/error`：验证是否成功及错误信息。
- `source_path/user_audio_path`：源训练 JSON 和用户音频路径。
- `gt_output_ids/pred_output_ids`：GT 与预测 token ids。
- `gt_output_render/pred_output_render`：GT 与预测 token stream 渲染。
- `gt_spoken_text/pred_spoken_text`：GT 与预测 spoken 文本。
- `gt_think_text/pred_think_text`：GT 与预测 think 文本。
- `gt_tool_calls/pred_tool_calls`：GT 与预测 tool calls。
- `tool_call_ids/tool_responses_by_call_id`：从训练数据提取的 tool call id 和 tool response。
- `units_info`：预测 unit 摘要。
- `comparison: FcDuplexComparisonResult | None`：GT/pred 对比结果。
- `audio_artifact: FcDuplexAudioArtifact | None`：TTS 音频落盘结果。
- `total_duration_ms`：总耗时。

`FcDuplexComparisonResult`：

- `token_ids_exact`：GT/pred token ids 是否完全一致。
- `rendered_token_stream_exact`：GT/pred rendered token stream 是否完全一致。
- `spoken_text_exact`：spoken text 是否一致。
- `think_text_exact`：think text 是否一致。
- `tool_calls_semantic_exact`：tool call 的 name/arguments/error 是否一致，不比较 id。
- `tool_call_ids_exact`：tool call id 是否与训练数据一致。
- `first_rendered_token_stream_diff`：首个 rendered token stream 差异位置和上下文。

`FcDuplexAudioArtifact`：

- `sample_rate`：音频文件采样率，当前为 24000。
- `unit_audio_paths`：每个 spoken unit 生成的 wav 文件。
- `full_audio_path`：拼接后的完整 wav 文件。
- `n_audio_units`：成功写出的音频 unit 数。

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

## 批量训练数据验证

项目根目录提供 `evaluate_fc_duplex_batch.py`。该脚本现在只是 runner，核心训练数据解析、GT tokenization、tool response 注入、GT/pred 对比和可选 TTS artifact 保存都在 `FcDuplexView.offline_inference_from_train_data()` 内完成。

先跑一条样本，启用 TTS：

```bash
cd /user/heweiquan/project/MiniCPM-o-Demo-FC

/user/heweiquan/envs/miniconda3/envs/minicpm-o4_5/bin/python evaluate_fc_duplex_batch.py \
  --data-dir /user/heweiquan/dataset/DuplexFcTest/delivery_train_data \
  --output-dir /user/heweiquan/project/MiniCPM-o-Demo-FC/fc_duplex_test_results_tts \
  --limit 1 \
  --skip-mutated \
  --extra-response-units 0 \
  --ref-audio-path /user/heweiquan/dataset/DuplexFcTest/delivery_train_data/media/system_reference/HTRef06.wav \
  --tts-prompt-path /user/heweiquan/dataset/DuplexFcTest/delivery_train_data/media/system_reference/HTRef06.wav \
  --attn-implementation sdpa
```

默认路径：

- base model：`/user/heweiquan/project/MiniCPM-o-4_5`
- checkpoint：`/user/heweiquan/models/minicpm-o45-fc-overfit/20260629/minicpm-v_50.pt`
- data dir：`/user/heweiquan/dataset/DuplexFcTest/delivery_train_data`
- output dir：`/user/heweiquan/project/MiniCPM-o-Demo-FC/fc_duplex_test_results`

常用参数：

- `--limit N`：只跑前 N 条样本。
- `--skip-mutated`：跳过轻微改写样本。
- `--budget N`：debug override。默认不传，使用 SDK train data 的 per-unit budget。
- `--extra-response-units N`：用户音频结束后额外补的静音响应 unit 数。
- `--decode-mode greedy`：解码模式。
- `--ref-audio-path PATH`：显式 TTS reference audio。
- `--tts-prompt-path PATH`：显式 Token2Wav prompt audio。

本文档示例以启用 TTS 为主：需要同时传入 `--ref-audio-path` 和 `--tts-prompt-path`。只传其中一个脚本会直接报参数错误。这样可以避免把很长的 `user_audio_0.opus` 默认当作 Token2Wav prompt，导致 streaming cache 超长。

如需跑全部 delivery 样本并生成 TTS：

```bash
/user/heweiquan/envs/miniconda3/envs/minicpm-o4_5/bin/python evaluate_fc_duplex_batch.py \
  --data-dir /user/heweiquan/dataset/DuplexFcTest/delivery_train_data \
  --output-dir /user/heweiquan/project/MiniCPM-o-Demo-FC/fc_duplex_test_results_tts \
  --skip-mutated \
  --extra-response-units 0 \
  --ref-audio-path /user/heweiquan/dataset/DuplexFcTest/delivery_train_data/media/system_reference/HTRef06.wav \
  --tts-prompt-path /user/heweiquan/dataset/DuplexFcTest/delivery_train_data/media/system_reference/HTRef06.wav \
  --attn-implementation sdpa
```

如需关闭 TTS，省略 `--ref-audio-path` 和 `--tts-prompt-path` 即可。

输出目录示例：

```text
fc_duplex_test_results/
  original/
    dob_dev_plan_0001/
      source.json
      gt_token_stream.txt
      pred_token_stream.txt
      units_info.json
      comparison.json
      pred_audio/                 # 仅启用 TTS 且成功生成时存在
        pred_audio_unit_000.wav
        pred_audio_full.wav
  original_summary.json
  summary.json
```

`comparison.json` 是 `FcDuplexTrainDataResult.model_dump()` 的结果，字段含义与上文 `FcDuplexTrainDataResult` 一致。

## 结果字段说明

`FcDuplexStepResult` / `FcNonSpokenGenerateResult`：

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
- `audio_sample_rate`：如果本 unit 生成了 TTS 音频，记录采样率。
- `n_audio_samples`：如果本 unit 生成了 TTS 音频，记录采样点数。

## 注意事项

- `non_spoken_budget_per_unit` 由调用方管理。在线服务中建议每个 unit 自己循环调用 `streaming_non_spoken_generate(max_tokens=1)`，达到预算后显式传 `close_reason="budget_reached"`。
- 如果调用方忘记闭合上一 unit，底层 `FcDuplexCapability` 会在下一次 prefill 前尝试自动闭合，保证 token 流结构合法。
- 当前 `tool_started` 是框架自动注入的创建/启动事件；真正工具执行完成后的结果仍需要调用方通过 `tool_responses` 传入。
- 当前成功启动事件默认只注入 `<|tool_started|>`，没有额外构造成功 content。若协议后续要求“成功/失败都必须有 `<tool_response>` content”，需要在 `ToolCallStateManager.consume_pending_started_events()` 附近扩展成功 content。
- IDE 可能提示无法解析 `minicpm_o5_sdk`，但只要使用 `/user/heweiquan/envs/miniconda3/envs/minicpm-o4_5` 环境运行即可。
