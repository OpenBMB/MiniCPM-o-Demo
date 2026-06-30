# FC Slot 双工在线接口接入说明

本文档面向要把新 FC slot 双工协议接入在线语音双工对话的调用方。目标是说明调用方应该如何使用 `FcDuplexView` 的公开接口，如何按 unit 驱动模型，如何处理 spoken / non-spoken / tool call / TTS 音频输出，以及哪些状态需要调用方自己维护。

本文只描述新协议接口。旧的 `DuplexView` / `DuplexCapability` 不在本文范围内。

## 1. 核心概念

### 1.1 一个会话由多个 unit 组成

FC 双工把实时语音对话拆成连续的 `unit`。每个 unit 可以理解为一小段时间窗口，典型是 1 秒。

每个 unit 的协议结构大致是：

```text
<unit>
  <user_video_slot>...</user_video_slot>      # 当前实现预留，通常不用
  <user_audio_slot>...</user_audio_slot>      # 用户音频输入
  <input_event_slot>...</input_event_slot>    # 工具 started / tool response
  <ai_spoken_slot>...</ai_spoken_slot>        # 模型是否说话，以及说什么
  <ai_non_spoken_slot>...</ai_non_spoken_slot># 模型后台思考 / tool call / no_action
</unit>
```

在线调用方每收到一个音频窗口，就推进一个 unit。每个 unit 的推荐调用顺序固定为：

```text
streaming_prefill()
streaming_spoken_generate()
streaming_non_spoken_generate()  # 循环若干步，直到终止或 budget 到达
finalize_unit()
```

### 1.2 调用方负责调度，模型负责生成

调用方负责：

- 切分用户音频为 unit。
- 决定每个 unit 的 non-spoken budget。
- 决定是否继续调用 `streaming_non_spoken_generate()`。
- 执行 tool call。
- 把 tool response 在后续 unit 注入。
- 播放 TTS waveform。
- 管理前端/服务端会话状态。

`FcDuplexView` 负责：

- 把 Pydantic request 转成底层模型调用。
- 加载 reference audio。
- 自动维护 tool call id。
- 自动注入 `<|tool_started|>`。
- 校验 tool response id。
- 把底层 dict 转成 Pydantic result。

模型层负责：

- 写入 FC special tokens。
- 维护 KV cache。
- 生成 spoken / non-spoken token。
- 生成 TTS waveform。
- 确保 unit / slot 结构可闭合。

## 2. 入口对象

在线调用方一般只需要使用：

```python
from core.processors import UnifiedProcessor

processor = UnifiedProcessor(
    model_path="/user/heweiquan/project/MiniCPM-o-4_5",
    pt_path="/path/to/fc/checkpoint.pt",
    device="cuda",
    compile=False,
    attn_implementation="sdpa",
)

fc = processor.fc_duplex
```

`fc` 就是 `FcDuplexView` 实例。

`attn_implementation="sdpa"` 表示使用 PyTorch SDPA attention，避免部分视觉模块不支持 flash attention 的兼容性问题。在线服务如果已确认 flash attention 可用，可以自行调整。

## 3. 依赖的 schema

所有公开接口的输入输出都定义在：

```python
from core.schemas.fc_duplex import (
    FcDuplexPrepareRequest,
    FcDuplexPrepareResult,
    FcDuplexPrefillRequest,
    FcDuplexPrefillResult,
    FcSpokenGenerateRequest,
    FcSpokenGenerateResult,
    FcNonSpokenGenerateRequest,
    FcNonSpokenGenerateResult,
    FcFinalizeUnitRequest,
    FcDuplexUnitInfo,
    FcDecodeOutputRequest,
    FcDecodeOutputResult,
    FcToolResponse,
    NonSpokenStepGenerationFlag,
)
```

调用方应以这些 Pydantic 模型作为接口契约，不要直接依赖底层 `MiniCPMO.fc_duplex_*` 的 dict 返回值。

## 4. 会话初始化：prepare

### 4.1 接口

```python
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

### 4.2 入参说明

`system_prompt: str`

系统提示词。通常包含业务规则、工具调用策略、展示策略等。

`tools: list[dict] | None`

OpenAI tool definition 形式的工具定义。例如：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "display_object_on_board",
            "description": "把指定对象显示到用户正在看的 visual board 上。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "要展示的对象名称"}
                },
                "required": ["name"],
            },
        },
    }
]
```

`ref_audio_path: str | None`

参考音频路径。`FcDuplexView` 会用 16kHz mono 加载，并作为 reference audio embedding 喂给 LLM。它影响模型的 reference audio 上下文，也用于让系统 prompt 中出现 `<|audio_start|><|audio_end|>` 对应的音频 embedding。

在线语音对话建议传入。delivery train data 中常见的 system reference audio 可以作为这个参数。

`prompt_wav_path: str | None`

Token2Wav 的 prompt 音频路径，用于初始化 vocoder streaming cache 和音色条件。建议使用较短、质量稳定的 prompt wav。不要默认使用很长的用户音频作为 prompt，否则可能导致 Token2Wav cache 过长。

`generate_audio: bool`

是否启用 spoken TTS waveform 生成。在线语音双工对话通常应设为 `True`。

如果 `generate_audio=True`，必须能提供 `prompt_wav_path`。当前实现中 `prompt_wav_path` 为空时会 fallback 到 `ref_audio_path`，但线上建议显式传两个参数，避免歧义。

### 4.3 返回值

`FcDuplexPrepareResult` 重要字段：

- `prefill_ids`: system/tool prefill token ids。
- `output_render`: system/tool prefill 的可读 token stream。
- `resized`: 是否扩展了 embedding 表。
- `generate_audio`: 当前 session 是否开启 TTS。
- `has_ref_audio`: 是否实际加载并喂入 reference audio。
- `prompt_wav_path`: 实际使用的 Token2Wav prompt path。

### 4.4 prepare 的语义

`prepare()` 会重置当前 FC duplex 状态。因此一个新的在线会话必须先调用一次 `prepare()`。

不要在同一个在线会话中间重复调用 `prepare()`。如果需要开启新对话，应先结束旧 session，再新建或重置 session。

## 5. 每个 unit 的输入：streaming_prefill

### 5.1 接口

```python
prefill_result = fc.streaming_prefill(
    FcDuplexPrefillRequest(
        audio_data=audio_data_base64,
        tool_responses=pending_tool_responses or None,
        sample_rate=16000,
    )
)
```

也可以直接传音频文件路径：

```python
prefill_result = fc.streaming_prefill(
    FcDuplexPrefillRequest(
        audio_path="/path/to/unit_audio.wav",
        tool_responses=None,
        sample_rate=16000,
    )
)
```

### 5.2 音频输入格式

推荐在线服务传 `audio_data`：

- 内容是 `float32` PCM。
- 单声道。
- 采样率通过 `sample_rate` 指定，推荐 16000。
- 再用 base64 编码。

示例：

```python
import base64
import numpy as np

def encode_float32_pcm(audio: np.ndarray) -> str:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return base64.b64encode(audio.tobytes()).decode("utf-8")
```

如果服务端已经有文件，也可以传 `audio_path`。`FcDuplexView` 会用 `librosa.load(audio_path, sr=sample_rate, mono=True)` 加载。

### 5.3 每个 unit 多长

通常按 1 秒一个 unit。也就是 16kHz 下每个 unit 约 16000 个 sample。

如果实时音频 chunk 不是刚好 1 秒，调用方需要统一自己的调度策略：

- 可以按固定 1 秒聚合后再调用。
- 或者按业务定义传入短 chunk，但要清楚这会影响模型看到的音频节奏。

训练数据一致性场景中 unit 由 SDK arrangement 管理；在线场景由调用方管理。

### 5.4 tool response 注入

如果之前某个 unit 中模型生成了 tool call，调用方执行工具后，需要在后续某个 unit 的 `streaming_prefill()` 中传入：

```python
from core.schemas.fc_duplex import FcToolResponse

tool_responses = [
    FcToolResponse(
        call_id="fc_call_000001",
        content={"status": "displayed", "name": "水杯"},
    )
]
```

`FcDuplexView` 会把它编码成：

```text
<input_event_slot>
  <tool_response_event>
    <tool_call_id>fc_call_000001</tool_call_id>
    <tool_response>...</tool_response>
  </tool_response_event>
</input_event_slot>
```

### 5.5 自动 tool_started

调用方不需要自己构造 `<|tool_started|>`。

当 `streaming_non_spoken_generate()` 发现一个完整 tool call span 后，`FcDuplexView` 会自动分配 `tool_call_id`，并在下一次 `streaming_prefill()` 时自动注入：

```text
<input_event_slot>
  <tool_response_event>
    <tool_call_id>fc_call_000001</tool_call_id>
    <|tool_started|>
  </tool_response_event>
</input_event_slot>
```

也就是说，调用方只需要执行工具并传回真实 `tool_response`。

### 5.6 返回值

`FcDuplexPrefillResult` 重要字段：

- `unit_index`: 当前 unit 下标。
- `n_audio_placeholders`: 当前 user audio slot 写入的音频 placeholder 数。
- `has_input_event`: 当前 unit 是否包含 input event。
- `inserted_token_ids`: 预留调试字段。

## 6. 生成 spoken slot：streaming_spoken_generate

### 6.1 接口

```python
spoken_result = fc.streaming_spoken_generate(
    FcSpokenGenerateRequest(
        max_tokens=24,
        decode_mode="greedy",
    )
)
```

### 6.2 模型可能输出什么

当前 unit 的 `ai_spoken_slot` 可能是：

```text
<ai_spoken_slot><|listen|></ai_spoken_slot>
```

表示模型不说话，只听。

也可能是：

```text
<ai_spoken_slot><|speak|>好的，我明白了<|spoken_turn_eos|><|spoken_slot_eos|></ai_spoken_slot>
```

表示模型说话。

如果当前 unit 只是同一轮语音的中间片段，也可能是：

```text
<ai_spoken_slot><|speak|>我先继续听你说<|spoken_slot_eos|></ai_spoken_slot>
```

SDK 004 后，spoken 结束 token 的语义需要区分 turn 结束和 slot 结束：

- `<|spoken_slot_eos|>`：当前 unit 的 spoken 片段结束，但同一轮语音可能下一 unit 继续。
- `<|spoken_turn_eos|>`：这一轮 spoken turn 完整结束，TTS / Token2Wav 可以 flush；它本身不闭合当前 spoken slot。
- `<|tts_pad|>`：当前 unit 没有新的 spoken 文本，但还属于 alignment / TTS padding 结构。

因此 `<|spoken_turn_eos|>` 和 `<|spoken_slot_eos|>` 可能连续出现。调用方不要把 `spoken_turn_eos=True` 理解成“当前 spoken slot 已由模型正常结束”；它只表示这一轮语音 turn 已结束。当前 spoken slot 是否由模型预测的终止 token 正常结束，应看返回值中的 `metadata["spoken_slot_terminated"]` 和 `metadata["spoken_termination_reason"]`。

### 6.3 返回值

`FcSpokenGenerateResult` 重要字段：

- `is_listen`: 是否输出 `<|listen|>`。
- `is_speaking`: 是否输出 `<|speak|>`。
- `spoken_token_ids`: spoken slot 生成 token ids。
- `spoken_text`: 当前 unit 生成的 spoken 文本。
- `spoken_turn_eos`: 是否结束本轮 spoken turn。
- `audio_waveform`: 如果 `generate_audio=True`，可能返回 24kHz float32 waveform。
- `audio_sample_rate`: waveform 采样率，当前为 24000。
- `n_audio_samples`: waveform sample 数。
- `n_tts_tokens`: TTS audio token 数。
- `cost_llm`, `cost_tts_prep`, `cost_tts`, `cost_token2wav`: 分阶段耗时。
- `metadata`: spoken slot 终止相关调试信息。

`metadata` 中当前包含：

- `spoken_slot_terminated: bool`：模型是否生成了 spoken slot 终止 token。
- `spoken_slot_unterminated: bool`：模型未生成 spoken slot 终止 token，框架只闭合了结构边界 `</ai_spoken_slot>`。
- `spoken_generation_reached_max_tokens: bool`：是否因为达到 `max_tokens` 才停止 spoken 生成。
- `spoken_termination_reason: str | None`：`spoken_slot_eos`、`listen`、`tts_pad` 或 `ai_spoken_slot_end`。
- `spoken_termination_token_id: int | None`：终止 token id。

正常 speak 分支中，SDK 004 期望由模型预测 `<|spoken_slot_eos|>` 结束当前 spoken slot。如果模型只预测了 `<|spoken_turn_eos|>`，但没有继续预测 `<|spoken_slot_eos|>`，`spoken_turn_eos=True` 仍会返回，但 `metadata["spoken_slot_terminated"]` 会是 `False`。

### 6.4 如何播放 TTS

如果 `spoken_result.audio_waveform is not None`，调用方可以直接播放或发送给前端。

示例：

```python
if spoken_result.audio_waveform is not None:
    send_audio_to_client(
        waveform=spoken_result.audio_waveform,
        sample_rate=spoken_result.audio_sample_rate or 24000,
    )
```

注意：

- TTS waveform 不是每个 speaking unit 都一定有。
- Token2Wav 是 streaming cache 逻辑，可能在某些 unit 返回一小段音频，在 `spoken_turn_eos=True` 时 flush 更多音频。
- `spoken_turn_eos=True` 只表示 TTS turn 需要 flush，不表示 spoken slot 已经由模型预测 `<|spoken_slot_eos|>` 正常结束。
- 前端播放层建议按返回顺序排队播放。

## 7. 生成 non-spoken slot：streaming_non_spoken_generate

### 7.1 调用方为什么要循环

`ai_non_spoken_slot` 可能包含：

- `<|no_action|>`：本 unit 没有后台动作。
- `<think>...</think>`：模型后台思考。
- `<tool_call>...</tool_call>`：模型发起工具调用。
- `<|non_spoken_eos|>`：自然结束。
- `<|non_spoken_budget_reached|>`：调用方 budget 到了，强制切到下个 unit。

在线场景中，调用方通常每次只让模型生成 1 个 non-spoken token：

```python
step = fc.streaming_non_spoken_generate(
    FcNonSpokenGenerateRequest(
        max_tokens=1,
        decode_mode="greedy",
    )
)
```

然后根据 `generation_flag` 决定下一步。

### 7.2 推荐循环

```python
from core.schemas.fc_duplex import (
    FcNonSpokenGenerateRequest,
    NonSpokenStepGenerationFlag,
)

budget = 30 if not spoken_result.is_speaking else 15

for _ in range(budget):
    step = fc.streaming_non_spoken_generate(
        FcNonSpokenGenerateRequest(
            max_tokens=1,
            decode_mode="greedy",
        )
    )

    handle_closed_spans(step.closed_spans)

    if step.generation_flag != NonSpokenStepGenerationFlag.continue_non_spoken_generation:
        break
else:
    # budget 用完，强制闭合当前 non-spoken slot
    step = fc.streaming_non_spoken_generate(
        FcNonSpokenGenerateRequest(
            max_tokens=0,
            decode_mode="greedy",
            close_reason="budget_reached",
        )
    )
```

### 7.3 generation_flag 的含义

`FcNonSpokenGenerateResult.generation_flag` 是调用方最应该看的一级字段。

可取值：

`continue_non_spoken_generation`

当前 non-spoken slot 还没结束。如果 budget 允许，继续调用 `streaming_non_spoken_generate(max_tokens=1)`。

`no_action`

模型输出 `<|no_action|>`，当前 non-spoken slot 已结束。本 unit 可以 finalize。

`non_spoken_slot_eos`

当前 non-spoken slot 已结束。包括自然 eos、budget reached、hold、abort 等结束情况。本 unit 可以 finalize。

### 7.4 terminated / close_reason

`terminated` 和 `close_reason` 仍然保留，适合日志和调试。

常见值：

- `close_reason=None`：还没结束。
- `close_reason="no_action"`：输出 `<|no_action|>`。
- `close_reason="eos"`：输出 `<|non_spoken_eos|>`。
- `close_reason="budget_reached"`：调用方强制预算结束。
- `close_reason="hold"`：保留。
- `close_reason="abort"`：中止。

### 7.5 close_reason 如何使用

调用方可以主动闭合 non-spoken slot：

```python
fc.streaming_non_spoken_generate(
    FcNonSpokenGenerateRequest(
        max_tokens=0,
        close_reason="budget_reached",
    )
)
```

常用场景：

- budget 用完：`budget_reached`
- 服务端要中止本 unit：`abort`
- 服务端决定本 unit 无动作：`no_action`

线上最常见的是 `budget_reached`。

## 8. 处理 think / tool_call closed spans

### 8.1 closed_spans 是什么

`streaming_non_spoken_generate()` 每一步都可能返回 `closed_spans`。

当模型刚好生成到 `</think>`，会返回一个完整 think span。

当模型刚好生成到 `</tool_call>`，会返回一个完整 tool call span。

调用方不需要自己拼 token 或等待整轮 decode。

### 8.2 think span

结构：

```python
FcClosedSpan(
    type="think",
    text="用户要求我展示具体可见对象...",
)
```

调用方通常只做日志记录，不展示给用户。

### 8.3 tool call span

结构：

```python
FcClosedSpan(
    type="tool_call",
    tool_call_id="fc_call_000001",
    wire="<function name=\"display_object_on_board\">...</function>",
    tool_call={
        "name": "display_object_on_board",
        "arguments": {"name": "水杯"},
        "error": None,
        "tool_call_id": "fc_call_000001",
    },
    error=None,
)
```

调用方处理方式：

```python
def handle_closed_spans(spans):
    for span in spans:
        if span.type == "think":
            log_debug("think", span.text)
            continue

        if span.type == "tool_call":
            if span.error or not span.tool_call:
                # 不执行解析失败的工具调用。
                log_error("tool_call_parse_failed", span.error, span.wire)
                continue

            enqueue_tool_call(
                call_id=span.tool_call_id,
                name=span.tool_call["name"],
                arguments=span.tool_call.get("arguments"),
            )
```

### 8.4 工具执行结果怎么传回

工具执行完成后，在后续某个 unit 的 `streaming_prefill()` 里传：

```python
pending_tool_responses.append(
    FcToolResponse(
        call_id=tool_call_id,
        content={
            "status": "displayed",
            "name": "水杯",
        },
    )
)
```

然后：

```python
fc.streaming_prefill(
    FcDuplexPrefillRequest(
        audio_data=audio_data_base64,
        tool_responses=pending_tool_responses,
        sample_rate=16000,
    )
)
```

### 8.5 工具错误怎么传

工具执行失败也传 `FcToolResponse`，content 可以是自然语言或结构化对象。

```python
FcToolResponse(
    call_id=tool_call_id,
    content="工具执行失败：board 当前不可用。",
)
```

当前协议要求成功失败都通过 tool response 写回。

### 8.6 tool response 的校验

`FcDuplexView` 会校验：

- 不能返回未知 `tool_call_id`。
- 不能对同一个 `tool_call_id` 重复返回 response。
- tool call id 由框架生成，不由调用方自己生成。

如果违反，会抛 `ValueError`。线上服务应捕获异常，记录 session 错误，并按业务策略中止或重建会话。

## 9. finalize_unit

每个 unit 的 spoken / non-spoken 都处理完后，必须调用：

```python
unit_info = fc.finalize_unit()
```

返回 `FcDuplexUnitInfo`。

重要字段：

- `unit`: unit 下标。
- `n_audio`: 当前 unit 音频 placeholder 数。
- `has_event`: 是否包含 input event。
- `is_listen`: 当前 unit 是否 listen。
- `is_speaking`: 当前 unit 是否 speaking。
- `spoken_ids`: 当前 unit spoken token ids。
- `spoken_slot_terminated`: 当前 spoken slot 是否由模型预测终止 token 正常结束。
- `spoken_slot_unterminated`: 是否由框架只闭合结构边界、但模型未预测 spoken slot 终止 token。
- `spoken_generation_reached_max_tokens`: spoken 生成是否达到 `max_tokens`。
- `spoken_termination_reason`: spoken slot 终止原因，例如 `spoken_slot_eos`、`listen`、`tts_pad`。
- `spoken_termination_token_id`: spoken slot 终止 token id。
- `non_spoken_ids`: 当前 unit non-spoken token ids。
- `non_spoken_terminator`: non-spoken 结束原因。
- `closed_spans`: 当前 unit 闭合的 think / tool_call span。
- `audio_sample_rate`: 如果本 unit 生成 TTS 音频，记录采样率。
- `n_audio_samples`: 如果本 unit 生成 TTS 音频，记录 sample 数。

如果调用方忘记 finalize，下一个 `streaming_prefill()` 会尽力兜底闭合上一个未完成 unit。但在线服务不应该依赖兜底逻辑，推荐每个 unit 显式 finalize。

## 10. decode_output

在线会话结束后，可以调用：

```python
decoded = fc.decode_output(FcDecodeOutputRequest(tools=tools))
```

返回 `FcDecodeOutputResult`：

- `units`: 解码后的 per-unit 摘要。
- `spoken_text`: 所有 spoken 文本拼接。
- `think_text`: 所有 think 文本拼接。
- `tool_calls`: 解码出的 tool calls。
- `output_ids`: 完整 token ids。
- `output_render`: 完整 token stream 的可读渲染。

`decode_output()` 主要用于：

- debug。
- 落日志。
- 离线回放。
- 问题定位。

在线实时逻辑不应该依赖最终 `decode_output()` 才执行工具，因为 tool call 已经能通过 `closed_spans` 在线返回。

## 11. cleanup

会话结束或服务释放资源时调用：

```python
fc.cleanup()
```

它会清理底层 FC duplex 状态。长连接服务中，如果每个用户 session 独占一个 processor/view，session 结束时应调用。

如果多个用户共享同一个模型实例，需要额外设计 session 隔离。当前 `FcDuplexView` / `FcDuplexCapability` 是有状态对象，不适合多个在线 session 并发复用同一个 view。

## 12. 推荐在线主循环

下面是一个简化的完整伪代码。

```python
import base64
import numpy as np

from core.processors import UnifiedProcessor
from core.schemas.fc_duplex import (
    FcDuplexPrepareRequest,
    FcDuplexPrefillRequest,
    FcSpokenGenerateRequest,
    FcNonSpokenGenerateRequest,
    FcToolResponse,
    NonSpokenStepGenerationFlag,
)


def encode_audio(audio: np.ndarray) -> str:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return base64.b64encode(audio.tobytes()).decode("utf-8")


processor = UnifiedProcessor(
    model_path="/user/heweiquan/project/MiniCPM-o-4_5",
    pt_path="/path/to/fc/checkpoint.pt",
    device="cuda",
    compile=False,
    attn_implementation="sdpa",
)
fc = processor.fc_duplex

tools = [...]
pending_tool_responses: list[FcToolResponse] = []

fc.prepare(
    FcDuplexPrepareRequest(
        system_prompt="你是一个实时 visual board 助手...",
        tools=tools,
        ref_audio_path="/path/to/ref.wav",
        prompt_wav_path="/path/to/short_prompt.wav",
        generate_audio=True,
    )
)

for unit_index, audio_chunk in enumerate(realtime_audio_units()):
    # 1. 输入当前 unit 用户音频 + 上一轮工具结果
    fc.streaming_prefill(
        FcDuplexPrefillRequest(
            audio_data=encode_audio(audio_chunk),
            tool_responses=pending_tool_responses or None,
            sample_rate=16000,
        )
    )
    pending_tool_responses = []

    # 2. 生成 spoken slot
    spoken = fc.streaming_spoken_generate(
        FcSpokenGenerateRequest(
            max_tokens=24,
            decode_mode="greedy",
        )
    )

    if spoken.audio_waveform is not None:
        send_audio_to_client(spoken.audio_waveform, spoken.audio_sample_rate or 24000)

    # 3. 根据 speaking/listening 决定 non-spoken budget
    budget = 15 if spoken.is_speaking else 30

    # 4. 生成 non-spoken slot
    for _ in range(budget):
        step = fc.streaming_non_spoken_generate(
            FcNonSpokenGenerateRequest(
                max_tokens=1,
                decode_mode="greedy",
            )
        )

        for span in step.closed_spans:
            if span.type == "think":
                log_debug("think", span.text)
            elif span.type == "tool_call":
                if span.error or not span.tool_call:
                    log_error("tool_call_parse_failed", span.error)
                    continue
                result = execute_tool(
                    name=span.tool_call["name"],
                    arguments=span.tool_call.get("arguments"),
                )
                pending_tool_responses.append(
                    FcToolResponse(
                        call_id=span.tool_call_id,
                        content=result,
                    )
                )

        if step.generation_flag != NonSpokenStepGenerationFlag.continue_non_spoken_generation:
            break
    else:
        # budget 到达，强制闭合 non-spoken slot
        fc.streaming_non_spoken_generate(
            FcNonSpokenGenerateRequest(
                max_tokens=0,
                close_reason="budget_reached",
            )
        )

    # 5. 结束当前 unit
    unit_info = fc.finalize_unit()
    log_unit(unit_info)

decoded = fc.decode_output()
fc.cleanup()
```

## 13. 在线 budget 建议

调用方可以按业务决定预算。当前训练数据中常见配置是：

- listening unit：30
- speaking unit：15

含义：

- 模型不说话时，可以给更多 non-spoken token 做后台思考和 tool call。
- 模型正在说话时，non-spoken lane 预算更小，避免后台生成过多影响实时性。

推荐初始配置：

```python
LISTENING_NON_SPOKEN_BUDGET = 30
SPEAKING_NON_SPOKEN_BUDGET = 15
```

如果线上延迟压力较大，可以降低。

如果模型经常来不及闭合 `<think>` 或 `<tool_call>`，可以适当提高。

## 14. TTS 接入说明

### 14.1 ref_audio_path 和 prompt_wav_path 的区别

`ref_audio_path`

用于把参考音频作为 LLM 上下文喂入，让模型知道 reference audio。它走的是音频 encoder / LLM embedding 路径。

`prompt_wav_path`

用于 Token2Wav 的 vocoder streaming cache 和音色 prompt。它走的是 TTS waveform 生成路径。

两者可以指向同一个短 wav，也可以不同。

线上推荐：

- `ref_audio_path` 使用系统或角色参考音频。
- `prompt_wav_path` 使用短、干净、稳定的音色 prompt。

### 14.2 TTS 输出的时机

`streaming_spoken_generate()` 返回的 `audio_waveform` 可能是：

- `None`：本 unit 没有可播放音频。
- 一段 24kHz float32 waveform：可以立即送前端播放。

多 unit 连续 speaking 时，Token2Wav 会维护 streaming cache。`spoken_turn_eos=True` 时会 flush 并重置当前 turn 的 TTS 状态。

注意 SDK 004 下，turn-final spoken slot 的模型输出通常是：

```text
<|spoken_turn_eos|><|spoken_slot_eos|>
```

其中 `<|spoken_turn_eos|>` 驱动 TTS turn flush，`<|spoken_slot_eos|>` 才表示当前 spoken slot 结束。在线调用方通常不需要手动处理这两个 token，但排查日志时应区分它们的语义。

### 14.3 前端播放建议

前端或网关建议按 unit 顺序排队播放：

```text
unit 11 audio -> unit 12 audio -> unit 13 audio -> ...
```

不要按网络返回时间乱序播放。

如果要做 barge-in 或打断，需要服务层额外设计 stop / clear audio queue 逻辑。

## 15. 工具调用时序

一个典型 tool call 生命周期：

```text
unit N:
  模型在 non-spoken slot 生成 <tool_call>...</tool_call>
  streaming_non_spoken_generate() 返回 closed_span，带 tool_call_id

unit N+1:
  下一次 streaming_prefill() 自动注入 <|tool_started|>
  调用方可以同时或稍后传入真实 tool_response

unit N+2 或更后:
  如果工具结果稍后才返回，调用方在任意后续 streaming_prefill() 传 FcToolResponse
```

如果工具执行很快，也可以在下一次 prefill 同时传入 response。`FcDuplexView` 会把自动 pending started event 和调用方传入 response 合并进 input event slot。

但如果希望完全贴合某些训练数据的时序，可以让 started 和 response 分别落在相邻两个 unit。

## 16. 异常处理

### 16.1 tool response id 错误

如果调用方传了未知 id：

```text
unknown tool_call_id: ...
```

如果重复传同一个 id：

```text
duplicate tool_response for tool_call_id: ...
```

这类异常说明调用方 tool 状态管理有问题，建议中止当前 session 或重建 FC duplex session。

### 16.2 tool call XML 解析失败

如果模型生成的 tool call XML 无法被 SDK 解析：

- `closed_span.error` 会有错误信息。
- `closed_span.tool_call` 为空或不可执行。
- 调用方不应该执行该工具。
- `FcDuplexView` 会为该失败调用生成 tool_call_id，并在后续 input event 中写回自然语言错误 `tool_response`。

### 16.3 TTS 参数错误

如果 `generate_audio=True` 但没有有效 prompt wav，可能抛出：

```text
prompt_wav_path is required when generate_audio=True
```

线上建议在调用 `prepare()` 前校验 `ref_audio_path` 和 `prompt_wav_path` 是否存在。

## 17. 并发与会话隔离

当前 `FcDuplexView` 是有状态对象，内部包含：

- 当前 KV cache。
- 当前 unit index。
- 当前 tool call id manager。
- 当前 TTS / Token2Wav streaming cache。

因此：

- 一个 `FcDuplexView` 同一时间只服务一个在线 session。
- 多用户并发时，不要共享同一个 `fc` 对象交叉调用。
- 可以为每个 session 创建独立 processor/view，或者在服务层实现 session 队列和互斥。

如果要复用同一模型权重支持多 session，需要进一步做 session state 隔离；当前接口按单 session stateful 方式设计。

## 18. 最小接入检查清单

上线前建议确认：

- 已调用 `prepare()`，且 `has_ref_audio=True`。
- `generate_audio=True` 时，`prompt_wav_path` 有效。
- 每个 unit 都按顺序调用 `prefill -> spoken -> non-spoken -> finalize`。
- non-spoken 循环使用 `generation_flag` 控制。
- budget 到达时调用 `close_reason="budget_reached"`。
- tool call 只在 `closed_spans` 返回完整 span 后执行。
- tool response 使用框架返回的 `tool_call_id`。
- 每个 tool_call_id 只返回一次 response。
- TTS waveform 按 unit 顺序播放。
- session 结束后调用 `decode_output()` 做日志，再调用 `cleanup()`。

## 19. 推荐日志字段

线上排查问题时，建议每个 unit 打印或记录：

- `session_id`
- `unit_index`
- `n_audio_placeholders`
- `spoken.is_listen`
- `spoken.is_speaking`
- `spoken.spoken_text`
- `spoken.spoken_turn_eos`
- `spoken.metadata.spoken_slot_terminated`
- `spoken.metadata.spoken_slot_unterminated`
- `spoken.metadata.spoken_generation_reached_max_tokens`
- `spoken.metadata.spoken_termination_reason`
- `spoken.n_audio_samples`
- non-spoken `generation_flag`
- non-spoken `close_reason`
- closed think span 长度
- closed tool call id / name / arguments
- tool response ids
- `unit_info.non_spoken_terminator`

会话结束时记录：

- `decoded.output_render`
- `decoded.spoken_text`
- `decoded.think_text`
- `decoded.tool_calls`

这些字段足够定位绝大多数协议时序、budget、tool call 和 TTS 问题。

