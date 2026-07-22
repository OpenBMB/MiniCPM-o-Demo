# o45-fc → add-API 合并审计（2026-07-10）

> 背景：`o45-fc`（`OpenBMB/minicpm-o-4_5-pytorch-simple-demo` 仓库分支，HEAD `ea26b2d`）和本分支 `wt/fc-api-on-refactor-2026-07-05`（`chmod777john` fork，HEAD `4beb31b`）共享 merge-base `0cec9dd`（2026-06-20），之后各自独立演化 38 个 / 24 个 commit。两边没有走 git merge/rebase 关联，本文档记录把 `o45-fc` 独有的修复手工核对/移植到本分支的过程和结论。

## A 类：`core/processors/unified.py` + `core/schemas/fc_duplex.py`（已合并，见 commit `9d8f094`）

两个文件在两个分支上几乎逐字相同（同样的 `FcDuplexView`/`ToolCallStateManager` 类结构和方法列表），说明是同一份代码快照分叉而非独立实现。核对出的三处真实差异：

- **`token_strs`/`spoken_token_strs`/`inserted_token_strs` 字段**：本分支原本保留（`_token_strs` 用 `fc_duplex.id2name` 改进过特殊 token 识别，比 o45-fc 早期版本更准），但 o45-fc 实测证明（`tmp/token_strs_vs_text_test.py`）普通内容 token 反查出来仍是 byte-level BPE 字节代理编码乱码，`id2name` 的改进只解决协议控制 token 误识别问题，不解决内容 token 乱码问题，所以字段本身依然不适合作为 streaming UI 的通用数据源。已按 o45-fc 的结论删除，`_step_result`/`_prefill_result`/`_spoken_result` 改回 `@staticmethod`。
- **`trace_snapshot`/`dump_trace`**：本分支独有，o45-fc 没有，已保留（纯增量，无冲突）。
- **`MiniCPMO45/utils.py`**：o45-fc 的 `get_seq_length()` 兼容性修复（`StreamDecoder.get_cache_length`）和本分支新增的 `normalize_content_item`/`_is_url`/`_download_url_to_tempfile` 辅助函数在文件里完全不相交，两边都保留。

## B 类：`MiniCPMO45/modeling_minicpmo_unified.py`（审计完成，无需移植）

o45-fc 在旧的单体 `class MiniCPMO(MiniCPMOPreTrainedModel)` 里直接加了约 1100 行 `FcDuplexCapability` 相关代码；本分支把同名的 `FcDuplexCapability` 抽到了新文件 `MiniCPMO45/fc_duplex_capability.py`（`inference-refactor-plan.md` 说明了动机：把贴近权重 trusted code 的模型层和 demo runtime 层分开，方便以后接入新模型版本）。

逐方法对比（`FcDuplexCapability` 类体 + `MiniCPMO.fc_duplex_*` 委托方法）结论：**o45-fc 加的所有方法名和方法体，在本分支里逐字存在（同名同参数同实现），o45-fc 没有任何独有逻辑缺失**。本分支额外多出的部分都是纯增量，不涉及协议语义变化：

- Trace 诊断：`_token_pieces`、`_trace_span`、`_record_trace`、`trace_snapshot`、`dump_trace`，以及散布在 `prepare`/`streaming_prefill`/`streaming_spoken_generate`/`streaming_non_spoken_generate`/`finalize_unit` 里的 `self._record_trace(...)` 调用点。
- 性能优化 1（合批 feed）：`_feed_mixed_embeddings`/`_audio_embeds` 把系统 token + 音频 embedding + 收尾 token 合并成一次 `decoder.feed()` 调用，取代 o45-fc 里分三次 `_feed_ids`/`_feed_audio` 单独喂的写法。
- 性能优化 2（延迟收尾反馈，commit `2ca8d08` "perf(fc-duplex): defer budget close feed to next prefill"）：`_close_non_spoken_slot(reason, *, defer_feed=False)` 新增 `defer_feed` 参数，`budget_reached` 路径下把收尾 token 的 decoder feed 推迟到下一次 `streaming_prefill` 时跟其它 token 一起合批喂，减少一次独立前向传播。

**已知的 `spoken_turn_eos` 语义 bug 两边完全一样**：`streaming_spoken_generate` 里 `elif nid == self.sid(K.SPOKEN_TURN_EOS): turn_eos = True` 后直接 `self._close_spoken_slot(append_spoken_slot_eos=turn_eos)`，两个分支逐字相同——本分支没有吸收 `origin/feature-fc`（何维佺分支，跟本次讨论的两个分支都不是同一条线）修的那个 spoken slot 语义拆分。这是 `feature_fc_alignment_checklist.md` Part 2 里记录的独立问题，跟这次 add-API 合并无关，需要的话仍需单独处理。

**结论：B 类不需要移植任何代码**，直接采用本分支现状即可。

## C 类：board demo（`audio_duplex_board` vs `demos/fc_board` + `static/fc-board`）

### 工具目录（`demos/fc_board/tools/display_object_on_board`）

跟 `audio_duplex_board/tools/display_object_on_board` 逐行对比：核心逻辑完全一致，只有 docstring 文案（"audio_duplex_board 原型" vs "FC board demo"）和 `BoardImageResult` 所在模块路径（我们放在顶层 `audio_duplex_board/schemas.py`，这边放在 `demos/fc_board/tools/display_object_on_board/schemas.py`）不同。**无需移植。**

### 业务编排层（`audio_duplex_board/session.py` vs `py_backend/fc_duplex_runtime.py`）

`py_backend/fc_duplex_runtime.py` 的模块 docstring 和多处方法 docstring 明确写着"mirrors audio_duplex_board.session"——同样是同一份代码快照分叉演化。已核对并移植的差异：

- **`_guess_block_kind_from_tokens(token_strs)` → `_guess_block_kind_from_token_ids(token_ids, backend)`**：跟 A 类同样的问题（字符串子串匹配不可靠），同样的修复方向。但这里用了比 o45-fc 更好的实现：不需要单独 import `minicpm_o5_sdk` 加载 tokenizer，直接复用模型自己已经初始化好的 `backend.processor.model.fc_duplex.ids["think_start"/"tool_call_start"]`（`FcDuplexCapability._ensure_protocol()` 早已经用 `O5SpecialTokenRegistry` 建好的同一份映射）。**注意 `backend` 不是模型本身**：`FcDuplexSessionRuntime.backend` 是 `core/processors/pytorch_backend.py::PyTorchBackend`，要经过 `backend.processor.model.fc_duplex` 三层才能拿到真正的 `FcDuplexCapability` 实例（抽成了 `_resolve_fc_duplex_capability(backend)` 复用）；`getattr` 链在任一环节拿不到都会优雅返回 `None`（`unknown`），不会抛异常。
- **`_token_observations` 第一版删错了，已修正**：最初判断"前端 `fc-board-app.js` 从未消费这个字段 + 数据源 `token_strs` 没了只会一直吐空字符串"，就把它和它在 `response.think.delta`/`response.tool_call.args.delta` 上的调用点整个删掉了。**这个判断是错的**——`docs/backend-protocol/duplex-tool-calling.md` §8「可选 token 观测」把 `token_observations` 定义为正式的（哪怕可选）协议能力，类比 OpenAI 的 `logprobs`，专门给调试/审计/exact trace 用，协议原文明确写了"不作为 semantic resume 的主接口"——前端不消费恰恰是设计允许的，不代表这是死代码。经用户提醒纠正为：**保留 `token_observations`，但内容必须用 SDK 提供的方法解码，不能用 `tokenizer.convert_ids_to_tokens`**。
  - 新实现：普通 token 用 `fc_duplex.decode_text([tid])`（即 `FcDuplexCapability.decode_text` → `self._sdk_tokenizer.decode_ordinary([tid])`，跟产出可靠的 `text`/`step_text` 字段用的是同一个 SDK 方法，只是传入单 token 列表）；特殊 token（`fc_duplex.is_special(tid)`）走 `fc_duplex.id2name` 给出协议展示名，不走 decode。
  - **实测验证两种方法对"跨字节边界拆分的 token"的行为本质不同**：`convert_ids_to_tokens` 暴露的是 byte-level BPE 内部字节→Unicode 代理字符的映射表，跟真实文本毫无关系，是彻头彻尾的乱码，还可能被误认成正常文本；`decode_ordinary` 是标准 UTF-8 decode-with-replacement 语义，单独解码"半个多字节字符"的 token 时干净地吐出一个标准 `U+FFFD`（`�`），是业界通用的"这里编码不完整"标记，不会被误读成别的内容。用已知的跨边界拆分对（79621/243，组合起来才是"展"）实测：`decode_ordinary([79621])` → `' \ufffd'`，`decode_ordinary([243])` → `'\ufffd'`，符合预期。
  - `_short_list`（唯一调用点是旧版 `_token_observations` 之外那两行日志格式化，已经在其它编辑里删了）保持删除，跟 `token_observations` 无关。
  - 同步给 `debug.fc_non_spoken.delta` 这个非正式 debug 事件也加上了 `token_observations`（同一个函数），跟正式协议事件保持一致，而不是只删 `token_strs` 不补。
- **`static/fc-board/fc-board-app.js` 的两处 `event.token_strs` fallback** 同步删除，跟 A 类前端修复一致，只用 `event.text`（后端 `step_text`）。前端没有消费 `token_observations`（协议允许），未改动。

**未动的 `token_strs`**：`MiniCPMO45/fc_duplex_capability.py` 里 `_record_trace(..., token_strs=self._token_pieces(...))` 那几处，是 `trace_snapshot`/`dump_trace`（他们自己的整段 trace 落盘诊断工具，见 B 类审计）内部用的字段，走的是 `_token_pieces`（已经用 `id2name` 改进过特殊 token 识别的版本），不是 o45-fc 删除的那个 pydantic schema 字段，也不是我们移植范围内的东西——保留不动。

### 前端 UI（`audio_duplex_board/web/*` vs `static/fc-board/*`）

逐文件核对结论：

- `audio-player.js`/`board-state.js`/`duplex-utils.js`/`file-audio-provider.js`/`live-mic-provider.js`：**逐字节相同**，无需移植。
- 打字机流式呈现（`enqueueSpeech`/`pumpSpeechQueue`/`appendSpeechChar`）、`scrollNsToBottom`（滚动到 `.nsStream.parentElement` 而不是自身）、turn-based 音频播放（`handleSpokenOutput` 依据 `isListen`/`isSpeaking` 调 `beginTurn`/`endTurn`，debug 面板的 `appendAiAudio` 只在抽屉打开时才建 `<audio>` 且没有 autoplay，不会重叠播放）：**已经存在，逐段核对没有回归，无需移植。**
- **LUFS mic 表**：确认缺失——他们的 `updateMicLevel` 还是原始 RMS（`level.toFixed(2)`，无 dB 换算、无目标响度区间），已移植 o45-fc 的 `rmsToDb`/`MIC_METER_MIN/MAX_DB`/`MIC_TARGET_LO/HI_DB`/`resetMicPeak` 全套实现到 `fc-board-app.js`，CSS 目标区间遮罩（`.mic-meter .meter-track::before`）加进 `fc-board.css`。
- **训练分布对齐的空状态引导文案**：他们原来的 empty-state 是泛化文案（"说出具体物体，例如「你看这只猫」「桌上有个苹果」"），已替换成 o45-fc 的训练句式引导（先说指令→等 AI 应声→再自由描述），文案根据这个分支的通用 system prompt 措辞做了适配（不再写"农场"这类 o45-fc 特定训练子集的场景词）。同步改了 HTML 里的初始文案和 `renderBoard()` 里重新渲染时的 innerHTML。

### 真实模型端到端验证（cctl job `142631`）

`cctl job copy`/`job create` 起了一个 `py_backend.server` + `worker.py` bundle（不开 `--compile`，config.example.json 默认 `compile: false`，跳过 ~10 分钟 warmup，纯正确性验证不测延迟），devbox 本地起 `gateway.py --http` 指向它，走正式 `/v1/realtime` 协议（`session.init` + `input.append`，跟 `fc-realtime-client.js`/`fc-board-app.js` 的真实用法一致）灌同一条训练音频（`dob_midtrain_v1_20260628_animal_seed_ct01_and_04842`）。

第一次尝试（比 o45-fc 板子当时验证幸运，那边试了 3 次才中）就拿到了决定性结果：

```
[EVT] response.think.begin @ 22803607.472
[EVT] response.think.delta (FIRST) @ 22803607.505 gap_since_begin_ms=33.7
```

`response.think.begin` 在第一个内容 token 那一步就发出，跟第一条 `response.think.delta` 只差 34ms（同一个生成 step），证明 `_guess_block_kind_from_token_ids` 在真实模型输出上确实是"span 一开就精确命中"，不是等 span 闭合才知道。后续 think 内容正常流式吐出可读中文（"用户希望我在...接下来的描述中，识别出兼具生活与农务属性的物品...我先爽快地答应他，然后保持倾听，不打断他的叙述。过程中只要听到符合这类特征的工具，就调用 display_object_on_board 展示，对于纯粹的生活用品则直接跳过。"），中间出现的一处 " �"+"�" 是已知的多字节 UTF-8 跨 token 边界的增量解码伪影（跟昨天 o45-fc 板子验证时看到的一样，最终整体解码不受影响，不是本次改动引入的问题）。这次模型选择只推理规则、不触发 tool_call（这条训练 case 的 `user_instruction_0.wav` 本身只是"立规则"阶段，没有提到具体物体），跟 tool_call 检测逻辑是否正确无关——`_guess_block_kind_from_token_ids` 对 `think_start`/`tool_call_start` 走的是同一段代码，think 侧已经拿到真实证据。

服务端日志唯一的 ERROR 跟这次改动无关：`_dump_model_trace`（他们自己原有的、写死路径 `/user/weihongliang/fc_trace_logs` 的诊断功能）在我们的运行环境下路径只读，被自身的 `try/except` 优雅捕获、不影响主流程，只是这个日志行本身值得他们后续把路径改成可配置。

验证完成后已清理资源：devbox 上的 `gateway.py` 进程已停，cctl job `142631` 已 stop。

### 音频落盘诊断（新增，他们完全没有）

`py_backend/fc_duplex_runtime.py` 新增 `_maybe_dump_user_wav`/`_maybe_dump_speak_wav`/`_append_audio_dump_manifest`，逻辑从 `audio_duplex_board/session.py` 移植，语义不变（用户音频 rms<0.001 跳过、AI 音频仅 `is_speaking=True` 且有 waveform 才落盘，同一个 `manifest.jsonl` 用 `role` 字段区分）。**开关方式不同**：o45-fc 走显式 config 字段 `audio_dump_dir`（CLI 参数一路传进 session）；这个分支为了不改动 `config.py`/`gateway.py`/`worker.py` 的参数传递链，改用环境变量 `FC_DUPLEX_AUDIO_DUMP_DIR`，跟这个分支已有的同类诊断开关（`FC_DUPLEX_TRACE_DIR`，见 `_dump_model_trace`）保持同一种约定，默认不设置 = 关闭。已用 mock backend 跑过端到端单测（写 wav + manifest + 静音/非说话跳过 + 默认关闭四种场景）。

## `token_observations` / 流式 decode 深挖（2026-07-11，commit `0b605e1` 之后的后续修正）

commit `0b605e1` 把 `_token_observations` 从"直接删掉"改成了"用 `fc_duplex.decode_text([tid])`（即 SDK 的 `decode_ordinary`）逐 token decode"。**这个实现本身还不是最终版本，下面记录为什么，以及正确的做法应该是什么。**

### 核心原则：不允许 fallback，每个函数只做自己能保证做对的事

`decode_ordinary`（本质是 HF `tokenizers` 库 Rust 层的 `String::from_utf8_lossy`，已用 subagent 精确定位到 `tokenizers/src/pre_tokenizers/byte_level.rs` 的 `ByteLevel::decode_chain`，本地不可配置、无法关闭）对"字节不完整"这件事的处理方式是**用 `U+FFFD` 顶上**——这是一个丢信息的 fallback：拿到 `U+FFFD` 之后无法反推原始字节是什么，也无法区分"这是哪个字被切断的"（实测过："展"和"放"被切断后单独 decode 出来的码点完全一样，`['0x20', '0xfffd']`，证明 `U+FFFD` 不携带任何原字符信息）。

对"逐 token 观测"这个场景（`token_observations` 的契约就是"一个 token 一条记录"），`decode_ordinary` 天然不适用——它的契约假设"输入的字节能形成合法字符"，逐 token 场景经常不满足这个假设。**正确工具是 `tokenizer.convert_ids_to_tokens`**：它只做字节级的可逆映射（GPT2 风格的 byte↔unicode 表，`transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode`），从不尝试判断"这些字节是不是一个合法字符"，所以对任何 token（完整字符还是半个字符）都精确、可逆，没有 fallback。

### 已落地的修正（`_token_observations`，`py_backend/fc_duplex_runtime.py`）

```python
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

_BYTE_DECODER = {v: k for k, v in bytes_to_unicode().items()}


def _token_raw_bytes(piece: str) -> bytes:
    """把 convert_ids_to_tokens 反查出的 vocab piece 还原成原始字节。

    这一步只是字节级映射表反查，从不做 UTF-8 判断，对任何 piece 都精确、
    可逆，没有 fallback。
    """
    return bytes(_BYTE_DECODER[ch] for ch in piece)


def _token_observations(token_ids: List[int], backend: Any) -> Optional[List[Dict[str, Any]]]:
    if not token_ids:
        return None
    fc_duplex = _resolve_fc_duplex_capability(backend)
    if fc_duplex is None:
        return None
    tokenizer = getattr(fc_duplex, "tokenizer", None)  # FcDuplexCapability.tokenizer 是公开属性（self.processor.tokenizer），已经在用，不需要新加
    if tokenizer is None:
        return None
    observations = []
    for tid in token_ids:
        tid_int = int(tid)
        if fc_duplex.is_special(tid_int):
            observations.append({"id": tid_int, "text": fc_duplex.id2name.get(tid_int), "bytes_hex": None})
            continue
        try:
            piece = tokenizer.convert_ids_to_tokens([tid_int])[0]
            raw = _token_raw_bytes(piece)
        except Exception:  # noqa: BLE001 - vocab 反查在极端情况下也不能挂主流程
            observations.append({"id": tid_int, "text": None, "bytes_hex": None})
            continue
        try:
            text = raw.decode("utf-8")  # 默认 strict，不用 errors='replace' 这种 fallback
        except UnicodeDecodeError:
            text = None  # 诚实信号"这个 token 单独不构成合法字符"，不是伪造的占位符
        observations.append({"id": tid_int, "text": text, "bytes_hex": raw.hex()})
    return observations
```

**不需要改模型层**：`FcDuplexCapability.tokenizer`（`self.processor.tokenizer`，`__init__` 第 64 行）已经是公开属性，直接暴露原始 HF tokenizer，`convert_ids_to_tokens` 是标准方法，不用新加任何东西到 `MiniCPMO45/fc_duplex_capability.py`。

**实测验证过的关键事实**（用 `/tmp` 下的脚本 + subagent，都可以重新跑）：
- `decode_ordinary([243])` → `'\ufffd'`（Python `str`，恰好一个字符，不抛异常，不会有变长/失控的输出）；跟 `raw.decode('utf-8')`（strict）比较：默认 strict 会抛 `UnicodeDecodeError`，只有显式传 `errors='replace'` 才会变成 `U+FFFD`——这行为在 SDK/transformers/tokenizers 整条链路里都找不到显式的 `errors=` 参数，因为真正起作用的是 Rust `String::from_utf8_lossy`（编译进 `.abi3.so`，本地不可见源码，已用跟本机版本一致的 GitHub tag 源码核对）。
- `convert_ids_to_tokens` 反查出的 vocab piece，**任何 token（完整字符还是半个字符）都一视同仁**，只做字节级映射，从不做 UTF-8 判断——`decode_ordinary([20002])`（"用户"，完整字符）反查出来同样是 `'çĶ¨æĪ·'` 这种"看起来像乱码"的东西，跟半个字符的 token 没有本质区别，区别只在于**后续要不要再做一次 UTF-8 decode**。
- 用 `bytes_to_unicode()` 反查表可以把 vocab piece 精确还原成原始字节（`token_raw_bytes(79621) == b' \xe5\xb1'`，`token_raw_bytes(243) == b'\x95'`），拼起来 `decode('utf-8')` 能正确还原"展"——证明 `convert_ids_to_tokens` 路径无损，只是把"什么时候做 UTF-8 decode"这个决定权留给了调用者。

### 顺带发现的一个更大的问题：流式增量 `step_text` 本身也会撞到同一个坑，不只是 `token_observations`

`MiniCPMO45/fc_duplex_capability.py` 的 `streaming_non_spoken_generate`/`streaming_spoken_generate` 里，`text_ids = []` 在**每次调用**开头重新初始化（`streaming_non_spoken_generate` 第 1008 行，`streaming_spoken_generate` 第 846 行），`text = self._flush(text_ids)` 只 decode**这一次调用新采样的 token**。业务层调用约定是 `max_tokens=1`（`py_backend/fc_duplex_runtime.py::_run_non_spoken_loop`），也就是说**大多数情况下每个 step 只 decode 1 个新 token**——这跟 `token_observations` 的逐 token 场景是**同一个问题**，只是发生在"当前吐给前端看的正文"（`response.think.delta`/`response.tool_call.args.delta` 的 `delta` 字段）上，不是可选的调试字段。

**已经在真实生成里实测确认过这个问题会真实发生**（2026-07-08 晚验证 token-id block-kind 检测那次的原始记录）：

```
[EVT] response.think.delta delta=' �'
[EVT] response.think.delta delta='�'
[EVT] response.think.delta delta='示'
```

"展"被拆到两个 step，前端会先看到两个 `�` 短暂闪一下，再看到"示"（"展"永久性地在流式展示里变成了两个问号，不会被后续修正——只有整段 span 闭合时的 `full_text`/`closed_span.text` 是整体 decode，最终是对的，但**流式过程中用户会看到错的**）。之前的分析笔记把这个记成"已知的、无害的增量解码伪影"，现在看来**不是无害，是一个真实的、有确定修复方案的 bug**。

### 根本修法（还没有落地实现，留给下一轮）：`tokenizers.decoders.DecodeStream`

已经系统性调研过业界标准做法（vLLM/SGLang/TGI/HF `TextStreamer`），结论都是同一个套路——**缓冲，等字节/字符攒够了再吐**，区别只在缓冲粒度。本机环境（`tokenizers==0.22.2`）已经验证 HuggingFace `tokenizers` 库自带现成的、按 token id 粒度工作的增量 decode 类，vLLM 自己也是直接调它当 fast path，不用我们发明状态机：

```python
from tokenizers.decoders import DecodeStream

stream = DecodeStream(skip_special_tokens=False)
piece = stream.step(backend_tokenizer, token_id)   # backend_tokenizer = fc_duplex.tokenizer.backend_tokenizer（Rust Tokenizer 对象）
# piece is None   -> 这个 token 单独字节不够，先不吐，状态留在 stream 内部
# piece is str    -> 攒够了，这次吐出的完整文本（可能包含好几个 step 之前攒的内容）
```

**当时样本上验证过、但后来确认不能当作全局不变量的性质**（`/tmp/demo_decode_stream_vs_naive.py`、`/tmp/verify_stream_chunk_reencode.py`）：
- 早期样本中，`DecodeStream` chunk 重新 encode 恰好等于原 token ids；2026-07-22
  在真实 O45_FC token 序列上发现同长度但不同 ID 的反例。因此实现必须逐 delta
  显式验证 exact token round-trip，不成立时只允许 live 展示，checkpoint 必须
  unavailable。
- 内部状态**不会无限增长**：一旦确认吐出一个 chunk，源码（`tokenizers/src/tokenizer/mod.rs::step_decode_stream`，`*ids = ids.drain(*prefix_index..).collect()` 这一行）会把已经消费掉的 token id 从缓冲区里丢弃，只留下"还在等下一个 token 才能凑齐"的那一小段。
- 算法本身是"整体重新 decode + 跟上次比长度 + 看结尾是不是 `U+FFFD`"（vLLM/SGLang 抄的也是同一套，只是有的用字节级缓冲、有的用这种字符串级 diff），不是按字节精确切分——比如"空格+半个展"这个 token，`DecodeStream` 会把空格也留在缓冲区里一起等，不会提前把空格吐出来；这个粒度上的"不够精确"对我们的场景没有实际影响（最多晚一步吐出一个空格，不会看到 `U+FFFD`，不会丢字）。
- 早期 corpus 上的 round-trip（9894/9894）与分段拼接（290360/290360）只说明
  该 corpus 没触发 tokenizer 非规范编码反例，不能证明任意生成 token 序列都可
  re-encode。最终实现以运行时 exact ID 比较作为安全条件。

**修复方案设计（分层，还没写代码）**：

- **落地位置：业务层 `py_backend/fc_duplex_runtime.py`，不是模型层 `MiniCPMO45/fc_duplex_capability.py`。** 原因：（1）模型层的 `text_ids`/`_flush` 逐 call 重置这件事本身不用改，模型层继续按自己的节奏吐 `token_ids` 就行；（2）业务层已经拿到了可靠的 `step.token_ids`（原始采样 id，从来没有过失真问题），有它就足够重建正确文本，不需要依赖模型层的 `step.text` 字段；（3）改业务层不需要重新加载 9B 模型、不需要 GPU，能在本机快速验证，改模型层则需要重新起 cctl job 才能验证，成本高很多。
- **状态放在哪**：`FcDuplexSessionRuntime`（每个 session 一个实例）需要新增至少 2 个 `DecodeStream` 实例——非语音（think/tool_call 混合的 non_spoken lane）和语音（spoken lane）各一个，因为这是两条独立的 token 序列，不能共用一个 `DecodeStream` 的内部状态。
- **需要处理的细节（还没完全想清楚，下一轮要专门确认）**：
  1. `DecodeStream` 不认识"special/control token"这个概念，会把喂给它的任何 token id 都当普通字节去尝试 decode——如果非语音 lane 里混进了 `<think>`/`<tool_call>`/`<|non_spoken_eos|>` 这类协议 token（`is_special(tid)==True`），不能直接喂给 `DecodeStream`，需要在喂之前先用 `fc_duplex.is_special(tid)` 过滤掉，只把"ordinary content token"喂进去（跟 `decode_ordinary` 自己会对 control token 抛 `ValueError` 是同一个约束）。
  2. 一个 non_spoken block（比如一个 `<think>...</think>` span）闭合之后，下一个 span（可能是 `<tool_call>`）开始时，`DecodeStream` 的内部 `prefix`/`ids` 状态要不要重置？直觉上应该重置（两个 span 的内容语义上不连续，没必要让上一个 span 未消费完的缓冲字节"泄漏"到下一个 span），但需要确认："span 闭合"这个时间点，`DecodeStream` 内部缓冲区是不是保证已经空了（如果 span 内最后一个字符恰好也被跨 token 切断，缓冲区可能非空）——这种边界情况下要不要在 span 强制闭合时把 `DecodeStream` 剩余缓冲的内容强制吐出来（哪怕它当时还没等到"确认"），需要设计一个"flush"语义。
  3. spoken lane 同理：每个 unit 的 `streaming_spoken_generate` 调用之间，`DecodeStream` 状态要不要跨 unit 保留？如果一个字被切在两个 unit 之间（理论上可能，因为 spoken 也是按 unit 调度的），跨 unit 保留状态才能修复；但这会让 spoken lane 的 `DecodeStream` 生命周期跟 `FcDuplexSessionRuntime` 整个 session 一样长，需要确认这样设计没有其它副作用（比如 session 内 spoken lane 有没有"turn 切换"之类需要重置的语义边界，跟 non_spoken 的 span 边界是不是类似的问题）。
  4. 验证方式：这个改动改变了对外协议 `response.think.delta`/`response.tool_call.args.delta`/`response.output.delta(kind=text)` 的吐出时机（可能某个 delta 会因为在等字节而"迟一步"才出现，具体内容不受影响），需要一次真实 GPU 端到端验证（复用之前验证 block-kind 检测用的同一套 cctl job + WS 脚本流程），确认没有引入新的乱序/丢字问题，再合入。

**执行顺序**：`_token_observations` 的低风险修正已落地并用真实 tokenizer + mock backend 验证；`DecodeStream` 流式修法涉及的细节多、需要真实 GPU 验证，留到下一轮专门做，不在同一个 commit 里混着改。

## 2026-07-22：安全增量文本、canonical step batch 与 stateless resume 落地

本轮先新增规范
[`resumable-generation-api.md`](./resumable-generation-api.md)，再按规范实现：

- SDK tokenizer bundle 的 `create_ordinary_text_decode_stream()` 由 View 持有；
  non-spoken 每个 think/tool-call span 独立，spoken 每个 turn 独立并跨 Unit。
- View 把 capability token IDs 展开为逐 token `text_pending` / `text_delta` /
  `protocol` step；safe delta 在内部验证
  `encode_ordinary(text_delta) == original_pending_token_ids`，失败则后续 checkpoint
  永久标为 `text_delta_roundtrip_mismatch`。
- API runtime 以 50ms / 16 steps / protocol / Unit 边界为 flush 条件发送
  `response.generation.step_batch`，公共 step 不包含 token ID。
- 每个实际处理 Unit 发送 `response.unit.committed`，checkpoint 用唯一
  `input_id` 绑定实际处理输入；latency queue 丢弃但未处理的 input 不会错配到 Unit。
- 新 `session.resume` 使用客户端提交的完整双向历史；canonicalizer 校验
  event/batch/step/checkpoint 连续性、Unit 内 spoken→non_spoken 顺序、媒体
  base64/float32 形状、模型/tokenizer/reference-audio identity 与 safe delta
  round-trip，再构造 Unit replay plan。
- Capability 新增 deterministic `replay_completed_unit()`，按 live skeleton
  `prefill → spoken IDs → spoken slot end → non-spoken IDs → slot/unit end`
  feed 历史输出，不重新采样、不重新生成 TTS。
- 当前 resume MVP 对 pending text、open span、跨 Unit spoken turn、deferred close、
  tool-call/tool-result 状态明确返回 unavailable/failure，不做 fallback。
- FC Board client 自动保存双向 history、Unit checkpoint 和 resume identity，并可
  构造 `session.resume` payload。

验证：

- 新增 4 个测试文件，覆盖 capability replay skeleton、跨 Unit/跨 track safe delta
  恢复、媒体与 input 绑定、View decoder 生命周期、round-trip 不变量、runtime
  batching/checkpoint/resume。
- 定向 + schema 回归：61 passed。
- 核心 resume 模块 mypy 通过；Python 编译、ES module 语法、Bash 语法和
  `git diff --check` 通过。
- 仓库全量测试在当前 worktree 无 `.venv/base` 且历史 case 使用
  `/path/to/MiniCPM-o-4_5` 的环境下无法作为有效全绿 gate；本轮新增测试全部通过。
- 旧任务 `144802` 因错误使用历史 `thunlp/agent-dev` 入口且无容量已取消。
- `modelbest/langfang_train/deploy` 任务 `606905` 完成真实
  live→Unit 0 available checkpoint→disconnect→`session.resumed` E2E。
- 首轮 E2E 发现 trace 默认写入只读的 `/user/weihongliang/fc_trace_logs`；已将 runtime
  默认改为 `/tmp/minicpmo45_fc_trace_logs`，启动脚本显式使用
  `${LOG_DIR}/fc_traces`。
- 修复后任务 `606931` 通过 trace 等价验证：
  `output_ids` 完全一致（252 IDs）、`kv_cache_length=322`、
  `current_unit_idx=1`，证明 public-history replay 在该真实 Unit 上恢复了同一 LLM
  token/KV 边界。
