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

见后续提交，进行中。
