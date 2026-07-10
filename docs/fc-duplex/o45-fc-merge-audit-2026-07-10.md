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
