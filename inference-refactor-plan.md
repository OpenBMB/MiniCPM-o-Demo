# 推理代码重构计划

## 目标

把模型 trusted code 和 omni-dev 的 demo runtime 分开。

重构后的目标形态：

- `MiniCPMO45/modeling_minicpmo.py` 承担真正的模型代码，尽量贴近权重目录里的 trusted code。
- `MiniCPMO45/modeling_minicpmo_unified.py` 保留现有对外接口，但只做模型之外的调度和适配。
- 后续接入 o5 时，原则上替换 `modeling_minicpmo.py` 及配套 trusted code 文件即可，unified runtime 尽量不动。

## 当前问题

现在的 `modeling_minicpmo_unified.py` 同时承担了两类职责：

- 模型基础能力：LLM、APM、VPM、TTS、embedding、forward、generate 等。
- demo runtime：mode 切换、duplex 状态机、单工 streaming/non-streaming API 包装、backend 需要的 prepare/generate/finalize 接口。

这导致 duplex 逻辑和模型实现混在一起。替换 o5 时，很难判断哪些 diff 是模型必须变化，哪些只是 omni-dev demo runtime 的历史改动。

另外，当前 vendored 的 `MiniCPMO45/modeling_minicpmo.py` 在 unified 主线里基本不承担推理职责，主要是占位和少量旧 helper 引用。它更适合改造成真正的 trusted code 承载文件。

## 目标分层

模型层：

- `modeling_minicpmo.py`
- `configuration_minicpmo.py`
- `processing_minicpmo.py`
- `utils.py`

这一层尽量贴近 o45/o5 weight trusted code，暴露基础能力：

- `MiniCPMO`
- `MiniCPMODuplex`，如果 trusted code 原生提供
- `llm / apm / vpm / tts`
- `get_audio_embedding()`
- `get_audio_embedding_streaming()`
- `get_vision_embedding()`
- `get_omni_embedding()`
- `chat()`
- `streaming_prefill()`
- `streaming_generate()`
- `as_duplex()`

runtime 层：

- `modeling_minicpmo_unified.py`

这一层保留 omni-dev 现有对外接口：

- `MiniCPMO.init_unified()`
- `MiniCPMO.set_mode()`
- `MiniCPMO.duplex_prepare()`
- `MiniCPMO.duplex_prefill()`
- `MiniCPMO.duplex_generate()`
- `MiniCPMO.duplex_finalize()`
- `non_streaming_prefill/generate`
- `streaming_prefill/generate` 的 demo API 适配

但这些接口内部应该尽量调用 `modeling_minicpmo.py` 的基础模型，而不是重新复制一整份模型实现。

duplex runtime：

- `DuplexCapability` 仍可先放在 `modeling_minicpmo_unified.py`。
- 它持有 `model: MiniCPMO`，调用模型基础能力。
- 它负责 full-duplex 的 prepare/prefill/generate/finalize、turn 状态、TTS 调度、token2wav flush、listen/speak 策略。
- 它不应该包含模型结构定义。

## 当前共识

`MiniCPMO45/modeling_minicpmo.py` 不应该继续只是一个基本没被主线使用的占位文件。它更适合承担真正的 trusted/weight model code。

`MiniCPMO45/modeling_minicpmo_unified.py` 仍然保留，并且保持当前对外接口不变。但它的职责应该收敛为 runtime/facade：

- 对外维持 omni-dev backend 现在调用的 API。
- 对内调度 `modeling_minicpmo.py` 暴露的模型能力。
- 保留 `DuplexCapability` 等 demo runtime 逻辑。
- 不再复制或改写一整份模型结构。

这样后续切 o5 时，预期主要替换 `modeling_minicpmo.py` 及配套 trusted code 文件，而不是重新 fork 一份 o5 unified modeling。

## Model 和 Runtime 的边界

模型层可以暴露内部模块和基础方法。runtime 不一定把 model 当黑盒，允许白盒式访问这些稳定内部能力：

- `model.llm`
- `model.tts`
- `model.processor`
- `model.config`
- `model.device`
- `model.llm_past_key_values`
- `model.audio_past_key_values`
- `model.get_audio_embedding()`
- `model.get_audio_embedding_streaming()`
- `model.get_vision_embedding()`
- `model.get_omni_embedding()`
- `model.reset_session()`
- `model.init_streaming_processor()`

但这些访问应该集中在 unified/runtime 层，不应该反过来让 trusted model code 持有 demo runtime。

## Unified Runtime 依赖 Base Model 的方式

多数 unified 独有方法不要求改写 base model，只需要调用 base model 暴露的属性和方法。

优先实现方式是让 `modeling_minicpmo_unified.MiniCPMO` 继承
`modeling_minicpmo.MiniCPMO`：

```python
from .modeling_minicpmo import MiniCPMO as BaseMiniCPMO

class MiniCPMO(BaseMiniCPMO):
    ...
```

这样 `modeling_minicpmo.py` 负责承载 weight/trusted model 的基础能力，
`modeling_minicpmo_unified.py` 只在子类里补 omni-dev 需要的 runtime/facade
接口。默认不重复实现父类已经提供的 model 能力；只有证明某个差异是主线
跑通必需时，才在 unified 子类里做最小 override 或 shim。

纯 runtime/facade 方法：

- `init_unified()`
- `set_mode()`
- `current_mode()`
- `duplex_prepare()`
- `duplex_prefill()`
- `duplex_generate()`
- `duplex_finalize()`
- `duplex_stop()`
- `duplex_chat()`
- `apply_torch_compile()`
- `set_compile_enabled()`
- `warmup_compile()`
- `benchmark()`

这些方法主要使用 `llm/tts/processor/reset_session/duplex` 等能力，不需要把模型结构重新 fork 一遍。

单工 runtime 方法：

- `non_streaming_prefill()`
- `non_streaming_generate()`
- `streaming_prefill()`
- `streaming_generate()`
- `_save_speculative_snapshot()`
- `restore_speculative_snapshot()`

这些方法会读写 model 的内部会话状态，例如：

- `llm_past_key_values`
- `audio_past_key_values`
- `tts_last_turn_tokens`
- `new_user_msg`
- `audio_chunk_idx`
- `_pending_round_id`
- `_omni_chunk_history`

这说明它们属于 runtime，但不是完全黑盒 runtime。它们可以继续放在 unified 层，通过继承或包装访问 base model 状态。

## 需要 Shim 或 Override 的位置

有些 unified 期望的接口和 weight trusted code 的命名不同，不应该因此改写 base model，可以在 unified 层补 shim：

- unified 的 `init_speech_decoder()` 对应 weight 的 `init_tts_module()`。
- unified 的 `init_token2wav()` 对应 weight 的 `init_tts()`。

还有一些方法包含 demo 行为差异，短期可以在 unified 层 override，后续再讨论是否拆成更独立的 runtime/helper：

- `chat()`
- `get_sys_prompt()`
- `_generate_speech_non_streaming()`
- `streaming_prefill()`
- `streaming_generate()`

这些差异不意味着 base model 必须被改写成 unified model。

## Vendored Modeling 的当前使用

当前 `MiniCPMO45/modeling_minicpmo.py` 在 unified 主线中基本不承担推理职责。

已确认的代码依赖主要是：

- `core/processors/base.py` 里为了调用 `MiniCPMO.get_sys_prompt()` import 了 `MiniCPMO45.modeling_minicpmo.MiniCPMO`。

这只是一个 prompt helper 依赖，不是推理依赖。后续可以改为：

- 从 unified 当前 model 获取 `get_sys_prompt()`。
- 或抽独立 prompt helper。

不需要为了这处引用保留整份旧 vendored modeling。

## 判断标准

重构完成后，应满足：

- `modeling_minicpmo.py` 看起来像 weight trusted code，而不是 demo runtime。
- `modeling_minicpmo_unified.py` 仍然提供当前 omni-dev backend 需要的接口。
- duplex 状态机在 unified/runtime 层，不混进模型结构定义。
- 单工和双工主线仍可通过 docs-app/API 示例跑通。
- o5 接入时，主要变更集中在 trusted code 文件替换，而不是重写 unified runtime。

## 分步重构路径

每一步都应该保持 demo 可运行。除非该步明确只改文档或纯死代码，否则完成后都启动
docs-app/API 主线，让人工验证至少覆盖：

- 文本输出正常。
- 单工语音输出正常。
- duplex/full-duplex 语音输出正常。

### Step 0：建立当前基线

不改推理代码，只确认当前分支、当前服务启动方式、docs-app/API 调用方式和人工验证样例。
后续每一步都和这个基线比对。

### Step 1：解除主线对旧 vendored modeling 的误用

当前 `core/processors/base.py` 只为了 `MiniCPMO.get_sys_prompt()` 引用
`MiniCPMO45/modeling_minicpmo.py`。先把这个依赖迁到稳定位置：

- 或从 unified facade 取 prompt helper。
- 或抽一个独立 prompt helper。

目标是让 `modeling_minicpmo.py` 后续可以替换成 weight trusted code，而不因为
prompt helper 改动影响 demo 主线。

验收：demo 行为应该完全不变。

### Step 2：把 `modeling_minicpmo.py` 替换为 o45 weight trusted code

在 unified 仍然自包含的前提下，先让 vendored modeling 变成真正的 weight model
承载文件。

这一阶段不让主线推理立刻依赖它，只确保：

- 文件内容贴近 o45 weight trusted code。
- package import 不报错。
- prompt/helper 依赖已经不再绑死旧 vendored modeling。

验收：demo 行为应该完全不变。这个 step 的意义是把“可信 base”先放到正确位置。

### Step 3：建立继承关系，但暂不迁移行为

让 `modeling_minicpmo_unified.MiniCPMO` 继承
`modeling_minicpmo.MiniCPMO`，但先保留 unified 现有实现。

这一阶段可以用过渡写法保持行为不变：`__init__` 仍按 unified 当前逻辑初始化，
runtime 方法也都还在 unified 子类里。此时继承关系只是为后续逐步删除重复代码
建立 MRO。

验收：demo 行为应该完全不变。

### Step 4：删除最安全的重复 model 方法

先迁移不会改变 module global 绑定的简单方法，让它们自然落到父类实现：

- input/output embedding getter/setter
- `get_decoder()/set_decoder()`
- `_decode()`
- `_decode_stream()`
- `_decode_text()`
- `forward()`
- `generate()`
- `subsequent_chunk_mask()`
- `_get_feat_extract_output_lengths()`
- 明确只访问 `self` 状态且父类代码字面一致的 cache helper

注意：即便方法字面一样，如果方法体会引用模块级类名或函数名，也不能在这一批
无脑删除，因为落到父类后 global namespace 会变。

验收：单工、duplex 都应该正常。

### Step 5：处理组件类和 module global 绑定

再处理底部组件类：

- 字面一致的类可以从 vendored modeling import 或直接停止在 unified 维护。
- 不字面一致的类暂时不要动，例如 `MiniCPMWhisperEncoder`、`MiniCPMTTS`。

这个 step 的重点不是减少行数，而是确认哪些 model primitive 真正可以完全来自
vendored model，哪些仍然是 demo 主线需要的 patch。

验收：重点听音频，尤其 duplex 音频。

### Step 6：切换 `__init__` 到父类初始化

等前面边界清楚后，再让 unified 子类调用 `BaseMiniCPMO.__init__()`，然后只追加
unified runtime 状态：

- `_current_mode`
- `_unified_initialized`
- `duplex`
- `_duplex_config`
- compile/benchmark/runtime 需要的缓存字段

如果父类命名和 unified 旧命名不同，只在 unified 子类里补最小 shim，例如：

- `init_speech_decoder()` 到 `init_tts_module()` 的兼容。
- `init_token2wav()` 到 `init_tts()` 的兼容。

验收：这是风险较高的一步，需要完整跑单工和 duplex。

### Step 7：逐个收敛近似但不完全相同的 model 方法

对这些方法逐个判断，不能整批搬：

- `get_audio_embedding()`
- `get_omni_embedding()`
- `_generate_mel_spec()`
- `init_token2wav_cache()`
- `get_audio_embedding_streaming()`

默认策略是先尝试使用父类；如果 demo 主线坏了，再把差异压缩成最小 override，
而不是整段复制旧 unified 方法。

验收：每删或改一组都启动 demo 让人工验证。

### Step 8：保留 runtime/facade，停止收缩

最后 unified 里应该主要剩：

- `init_unified()`
- `set_mode()`
- `current_mode`
- `non_streaming_prefill/generate`
- `streaming_prefill/generate` 的 demo API 适配
- `duplex_prepare/prefill/generate/finalize`
- `DuplexCapability`
- 少量已证明必须保留的 override/shim

到这个状态就先停止，不追求把所有代码都删干净。目标是主线稳定、边界清楚、
后续替换 o5 trusted code 容易。
