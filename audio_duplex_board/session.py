"""Async board prototype session wrapper around `FcDuplexView`.

Responsibilities:

- Drive one ws session through the FC duplex 1-second-per-unit loop.
- Emit page-facing `BoardEvent` instances over an async sink:
    session_started → unit_started
                    → spoken_final (if speaking, with TTS waveform)
                    → non_spoken_block_started
                    → non_spoken_delta (per-step, with token_id + token_str)
                    → non_spoken_block_closed (with parsed full_text)
                    → think_final / tool_call_final  (derived, for log readers)
                    → board_card_created (status=searching) once tool_call closes
                    → board_card_updated (status=ready/error) once async search returns
                    → unit_finished → ... → session_finished
- Dispatch each tool's image search as `asyncio.create_task` so that:
    * the next prefill is not blocked on the previous tool's network round-trip
    * multiple in-flight tool calls run concurrently
    * `_pending_tool_responses` is only appended when a search finishes; that
      list is drained into the next `streaming_prefill(tool_responses=...)`

Threading model:

- This module runs entirely inside an asyncio event loop (the FastAPI ws
  handler). The model's `streaming_*` calls are CPU/GPU bound and blocking,
  so they go through `asyncio.to_thread` to avoid stalling the event loop.
- Tool image search (`tool_service.search`) is also blocking I/O, so it goes
  through `asyncio.to_thread` from a background task spawned per tool call.
- A single `asyncio.Lock` guards `_pending_tool_responses` because both the
  unit loop and the tool tasks touch it.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import numpy as np
import soundfile as sf

from core.processors import UnifiedProcessor
from core.schemas.fc_duplex import (
    FcDuplexPrepareRequest,
    FcDuplexPrefillRequest,
    FcNonSpokenGenerateRequest,
    FcSpokenGenerateRequest,
    FcToolResponse,
)

# When the session is wired to a WS-based remote view, fc methods are async
# and stream_non_spoken_decode returns an async iterator. We duck-type on the
# `stream_non_spoken_decode` attribute rather than isinstance.

from audio_duplex_board.config import AudioDuplexBoardConfig
from audio_duplex_board.schemas import (
    BoardCard,
    BoardEvent,
    StreamAudioChunkRequest,
    StreamPrepareRequest,
    ToolCallView,
)
from audio_duplex_board.tools.display_object_on_board.service import (
    DisplayObjectOnBoardService,
    board_image_result_from_tool_result,
)


EventSink = Callable[[BoardEvent], Awaitable[None]]


class AudioDuplexBoardSession:
    """Async session for one board prototype ws connection.

    Args:
        config: Board prototype runtime config.
        processor: Shared `UnifiedProcessor` (or `MockUnifiedProcessor`) instance.
        send_event: Async callback invoked for every page-facing event. Sending
            order must be preserved by the caller; this session calls it
            sequentially from one event-loop coroutine for unit events, and
            concurrently from tool tasks for `board_card_updated`. The caller
            is responsible for serializing writes onto the underlying ws.
        tool_service: Display object board tool service.
    """

    # 每个 unit 内 non-spoken slot 最多 step 数。这是死循环兜底；real-time
    # 节奏下应该靠 next-prefill-arrived 信号提前跳出。
    _MAX_NON_SPOKEN_STEPS_PER_UNIT = 200

    def __init__(
        self,
        *,
        config: AudioDuplexBoardConfig,
        processor: UnifiedProcessor,
        send_event: EventSink,
        tool_service: DisplayObjectOnBoardService | None = None,
    ) -> None:
        self.config = config
        self.processor = processor
        self.fc = processor.fc_duplex
        self.tool_service = tool_service or DisplayObjectOnBoardService()
        self._send_event = send_event
        self.session_id = f"board-{uuid.uuid4().hex[:10]}"
        self._prepared = False
        # 可选：把每个 speak unit 的 TTS waveform 落盘，供事后回听诊断
        # 重叠/截断/TTS 质量。
        self._audio_dump_session_dir: Path | None = None
        self._audio_dump_manifest_path: Path | None = None
        if config.audio_dump_dir:
            self._audio_dump_session_dir = Path(config.audio_dump_dir) / self.session_id
            self._audio_dump_session_dir.mkdir(parents=True, exist_ok=True)
            self._audio_dump_manifest_path = self._audio_dump_session_dir / "manifest.jsonl"
            print(
                f"[session {self.session_id}] audio dump: {self._audio_dump_session_dir}",
                flush=True,
            )
        self._unit_index = 0
        self._max_spoken_tokens = 24
        self._decode_mode = "greedy"
        # tool responses produced by background search tasks, drained at next prefill
        self._pending_tool_responses: list[FcToolResponse] = []
        self._pending_lock = asyncio.Lock()
        # background tool tasks, kept so we can await them at session finish
        self._tool_tasks: set[asyncio.Task[None]] = set()
        # per-unit non-spoken block sequence, used to build stable block_id
        # Monotonic block sequence counter. A "block" corresponds to one
        # <think>...</think> or one <tool_call>...</tool_call> segment. It
        # MUST NOT reset per unit — the same think/tool_call span can span
        # multiple 1s units (SDK 0.0.5a0 cross-unit non-spoken continuation:
        # slot closes with <|non_spoken_budget_reached|> but the SPAN keeps
        # going in the next unit's non-spoken slot). We only reset
        # `_current_block_*` when a closed_span arrives.
        # 原始需求（original_requirement_20260630.md）明确：进入 <think> 开
        # 一个 block，遇到 </think> 才闭合；工具调用同理。之前按 unit 拆导致
        # 一个 think 被切成 N 个 block，UI 全是空 UNKNOWN 卡片。
        self._block_seq_global = 0
        self._current_block_id: str | None = None
        self._current_block_kind: str | None = None  # "think" | "tool_call" | None=unknown
        # Real-time signaling: the ws layer sets this asyncio.Event whenever
        # a NEW audio_chunk arrives but the previous unit's non-spoken decode
        # loop is still running. The decode loop polls the event between
        # steps and exits cleanly when set, so the next prefill can proceed
        # without waiting for the model to terminate the non-spoken slot.
        #
        # When the loop exits this way, control returns to process_audio_chunk
        # which calls view.finalize_unit. The view's finalize_unit auto-emits
        # `<|non_spoken_budget_reached|>` + `<|ai_non_spoken_slot_end|>` if
        # the non-spoken slot is still open (unified.py finalize_unit). Per
        # SDK protocol (`ai_non_spoken.py` slot rules + `parser.py`
        # `active_non_spoken_wrapper` continuation), this token does NOT
        # close the active think/tool_call wrapper — it marks "this unit's
        # budget is up; the same span continues in the next unit". The
        # view's `_non_spoken_mode` / `_think_buf` / `_tool_call_buf` state
        # is NOT reset by `_close_non_spoken_slot`, so when the next unit's
        # slot opens, sampled tokens continue accumulating into the same
        # span buffer until the model emits `</think>` / `</tool_call>`.
        #
        # Whether the model actually emits the cross-unit continuation tokens
        # is a model-behavior property of the ckpt and is independent of this
        # protocol-level cooperative stop signal.
        self._next_prefill_event = asyncio.Event()
        # Per-unit timing for real-time health checks. Keyed by unit_index.
        # Each entry: dict with prefill_arrival_ms, prefill_done_ms,
        # spoken_done_ms, non_spoken_done_ms, finalize_done_ms.
        # We log a one-line summary at unit_finished.
        self._last_prefill_arrival_ts: float | None = None
        # Rolling window of unit total durations (ms), used to log p50/p95
        # every 10 units.
        self._unit_duration_ms_window: list[float] = []
        # Count of non-spoken steps that were short-circuited by next_prefill.
        self._budget_short_circuit_count = 0

    # ----------------------------------------------------------- public API

    async def prepare_stream(self, request: StreamPrepareRequest) -> None:
        """Prepare the session and emit `session_started`."""

        if request.session_id:
            self.session_id = request.session_id
        self._max_spoken_tokens = request.max_spoken_tokens
        self._decode_mode = request.decode_mode
        async with self._pending_lock:
            self._pending_tool_responses = []
        self._unit_index = 0

        prepare_request = FcDuplexPrepareRequest(
            system_prompt=request.system_prompt,
            tools=request.tools,
            ref_audio_path=request.ref_audio_path,
            prompt_wav_path=request.prompt_wav_path,
            generate_audio=request.generate_audio,
        )
        print(
            f"[session {self.session_id}] prepare: "
            f"system_prompt_len={len(request.system_prompt or '')} "
            f"tools={len(request.tools or [])} "
            f"ref_audio_path={request.ref_audio_path!r} "
            f"generate_audio={request.generate_audio}",
            flush=True,
        )
        await self._call_fc(self.fc.prepare, prepare_request)
        self._prepared = True
        await self._send_event(
            BoardEvent(
                type="session_started",
                session_id=self.session_id,
                payload={
                    "mode": "stream",
                    "generate_audio": request.generate_audio,
                },
            )
        )

    def signal_next_prefill_arrived(self) -> None:
        """Called from the ws layer the moment a NEW audio_chunk arrives, even
        if the previous unit's decode loop is still running. Lets the in-flight
        non-spoken decode bail out before its hard step cap.
        """
        self._next_prefill_event.set()

    async def process_audio_chunk(self, request: StreamAudioChunkRequest) -> None:
        """Process one browser-sent audio unit end-to-end (async)."""

        if not self._prepared:
            raise RuntimeError("stream session is not prepared")

        # This call IS the next prefill — clear the cross-unit signal so the
        # FOLLOWING unit's decode loop starts fresh waiting for the unit AFTER
        # this one.
        self._next_prefill_event.clear()

        prefill_arrival_ts = time.perf_counter()
        # Inter-prefill spacing: how long between this prefill's arrival at the
        # server and the previous one's. Frontend pushes at ~1s cadence so a
        # healthy value is ~1000ms. < 1000ms means the frontend is bunching
        # chunks (often because the previous unit took > 1s and the browser
        # finally caught up), which is the symptom of being non-real-time.
        inter_prefill_ms: float | None = None
        if self._last_prefill_arrival_ts is not None:
            inter_prefill_ms = (
                prefill_arrival_ts - self._last_prefill_arrival_ts
            ) * 1000.0
        self._last_prefill_arrival_ts = prefill_arrival_ts

        # 1) drain tool responses produced by background search since last prefill
        async with self._pending_lock:
            tool_responses = list(self._pending_tool_responses)
            self._pending_tool_responses = []

        # 用户反馈过"模型一直不响应"，很多时候是麦克风采到了静音（浏览器权限/
        # noiseSuppression 太狠 / 用户没真说话）。在这里解一次 base64 算 RMS +
        # duration，写到日志，出问题时能立刻区分是"音频没进来"还是"进来了但
        # 模型不动"。开销 O(1s of audio) 无所谓。
        audio_rms: float | None = None
        audio_peak: float | None = None
        audio_ms: float | None = None
        _samples: np.ndarray | None = None
        try:
            _raw = base64.b64decode(request.audio_base64)
            _samples = np.frombuffer(_raw, dtype=np.float32)
            if _samples.size > 0:
                audio_rms = float(np.sqrt(np.mean(_samples * _samples)))
                audio_peak = float(np.max(np.abs(_samples)))
                audio_ms = 1000.0 * _samples.size / max(1, request.sample_rate)
        except Exception:  # noqa: BLE001 - RMS 是诊断日志，出错不能影响主流程
            pass

        prefill = await self._call_fc(
            self.fc.streaming_prefill,
            FcDuplexPrefillRequest(
                audio_data=request.audio_base64,
                sample_rate=request.sample_rate,
                tool_responses=tool_responses or None,
            ),
        )
        prefill_done_ts = time.perf_counter()
        unit_index = prefill.unit_index

        # 用户输入音频也落一份 wav（unit_NNNN_user.wav），跟 AI TTS 输出的
        # unit_NNNN_speak.wav 放同一个 session 目录、同一个 manifest。用户反
        # 馈"一半 case 模型不回答"时，只有听到当时用户到底说了什么，才能
        # 判定是模型 OOD 还是 pipeline 有问题。
        self._maybe_dump_user_wav(
            unit_index=unit_index,
            samples=_samples,
            sample_rate=request.sample_rate,
        )
        # 不重置 `_current_block_id/_current_block_kind`：一个 <think> 或
        # <tool_call> segment 可能跨多个 unit，只有 span 闭合才 reset（见
        # `_emit_close_for_span`）。这里如果 reset 会让下一个 unit 的非空
        # non_spoken 步骤误开新 block。

        await self._send_event(
            BoardEvent(
                type="unit_started",
                session_id=self.session_id,
                unit_index=unit_index,
                payload={
                    "n_audio": prefill.n_audio_placeholders,
                    "tool_response_count": len(tool_responses),
                    "is_speaking": prefill.is_speaking,
                    "is_listen": prefill.is_listen,
                },
            )
        )

        # 2) spoken slot
        spoken = await self._call_fc(
            self.fc.streaming_spoken_generate,
            FcSpokenGenerateRequest(
                max_tokens=self._max_spoken_tokens,
                decode_mode=self._decode_mode,
            ),
        )
        spoken_done_ts = time.perf_counter()
        # 每个 unit 都要发 spoken_final，不只是 speak unit。
        # 原因：前端 AudioPlayer 是 turn-based（demo static/duplex/lib/
        # realtime-session.js 的模式）——speak unit 上如果 turn 未开就
        # beginTurn 然后 playChunk；**listen unit 上必须 endTurn**，否则
        # 下一次 speak turn 打开时 AudioPlayer.beginTurn() 会
        # _stopAllSources() 把上一 turn 尾巴截断（这就是用户看到的"结尾
        # 有截断"）。同理不区分 listen/speak 的话又会不停 begin/end 造成
        # 重叠。所以前端必须知道 is_listen / is_speaking。
        audio_float32_base64 = _audio_waveform_to_float32_base64(
            spoken.audio_waveform
        )
        audio_wav_base64 = _audio_waveform_to_wav_base64(
            spoken.audio_waveform,
            spoken.audio_sample_rate or 24000,
        )
        # 事后回听诊断用：is_speaking + 有 waveform 的 unit 落盘一份 wav +
        # 一行 manifest。同步 IO，10 KB/unit 量级，直接写不占大头。
        self._maybe_dump_speak_wav(
            unit_index=unit_index,
            spoken=spoken,
        )
        await self._send_event(
            BoardEvent(
                type="spoken_final",
                session_id=self.session_id,
                unit_index=unit_index,
                text=spoken.spoken_text,
                token_ids=list(spoken.spoken_token_ids),
                token_strs=list(spoken.spoken_token_strs),
                payload={
                    "is_speaking": bool(spoken.is_speaking),
                    "is_listen": bool(spoken.is_listen),
                    "spoken_turn_eos": spoken.spoken_turn_eos,
                    "audio_float32_base64": audio_float32_base64,
                    "audio_wav_base64": audio_wav_base64,
                    "audio_sample_rate": spoken.audio_sample_rate or 24000,
                },
            )
        )

        # 3) non-spoken slot: step loop, emit deltas, react to closed spans.
        #    Bails out early when next prefill arrives — but ONLY for speak
        #    units (real-time is critical when user is listening to AI). For
        #    listen units (user is talking, AI works in the background) we
        #    let the model run to view-side termination or the hard cap, so
        #    long <think>...</think> + <tool_call>...</tool_call> sequences
        #    have enough budget to actually emit a closing tag and produce a
        #    closed_span the business can dispatch. Mid-stream budget cuts
        #    leave the model stuck in think and never reach tool_call.
        is_speaking = bool(spoken.is_speaking)
        non_spoken_steps, short_circuited = await self._run_non_spoken_loop(
            unit_index, is_speaking=is_speaking
        )
        non_spoken_done_ts = time.perf_counter()

        # 4) finalize unit
        unit = await self._call_fc(self.fc.finalize_unit)
        finalize_done_ts = time.perf_counter()

        # Timing breakdown
        prefill_ms = (prefill_done_ts - prefill_arrival_ts) * 1000.0
        spoken_ms = (spoken_done_ts - prefill_done_ts) * 1000.0
        non_spoken_ms = (non_spoken_done_ts - spoken_done_ts) * 1000.0
        finalize_ms = (finalize_done_ts - non_spoken_done_ts) * 1000.0
        total_ms = (finalize_done_ts - prefill_arrival_ts) * 1000.0

        self._unit_duration_ms_window.append(total_ms)
        if len(self._unit_duration_ms_window) > 50:
            self._unit_duration_ms_window.pop(0)
        if short_circuited:
            self._budget_short_circuit_count += 1

        n_spoken = len(spoken.spoken_token_ids or [])
        n_non_spoken = len(unit.closed_spans)
        speak_text = (spoken.spoken_text or "")[:40].replace("\n", "\\n")
        inter_str = f"{inter_prefill_ms:.0f}" if inter_prefill_ms is not None else "—"
        rt_marker = "RT" if total_ms < 1000.0 else "SLOW"
        # RMS < 0.005 基本就是静音；正常说话 RMS 一般在 0.02 ~ 0.2 之间。
        if audio_rms is None:
            audio_tag = "audio=?"
        else:
            silent_tag = "SILENT" if audio_rms < 0.005 else "OK"
            audio_tag = (
                f"audio=[rms={audio_rms:.4f} peak={audio_peak:.3f} "
                f"ms={audio_ms:.0f} sr={request.sample_rate} {silent_tag}]"
            )
        print(
            f"[session {self.session_id} unit={unit.unit}] "
            f"listen={unit.is_listen} speak={unit.is_speaking} "
            f"spoken_tok={n_spoken} text={speak_text!r} "
            f"non_spoken_term={unit.non_spoken_terminator} "
            f"closed_spans={n_non_spoken} "
            f"ns_steps={non_spoken_steps} short_circuit={short_circuited} "
            f"{audio_tag} "
            f"timing_ms=[total={total_ms:.0f} prefill={prefill_ms:.0f} "
            f"spoken={spoken_ms:.0f} non_spoken={non_spoken_ms:.0f} "
            f"finalize={finalize_ms:.0f}] inter_prefill_ms={inter_str} [{rt_marker}]",
            flush=True,
        )

        # Periodic p50/p95 summary so the operator can read real-time health at a glance.
        if len(self._unit_duration_ms_window) >= 10 and unit.unit % 10 == 9:
            sorted_w = sorted(self._unit_duration_ms_window)
            p50 = statistics.median(sorted_w)
            p95 = sorted_w[int(len(sorted_w) * 0.95)]
            print(
                f"[session {self.session_id} rolling_stats over_last={len(sorted_w)}] "
                f"unit_total_ms p50={p50:.0f} p95={p95:.0f} "
                f"budget_short_circuits={self._budget_short_circuit_count}",
                flush=True,
            )
        await self._send_event(
            BoardEvent(
                type="unit_finished",
                session_id=self.session_id,
                unit_index=unit.unit,
                payload={
                    "is_listen": unit.is_listen,
                    "is_speaking": unit.is_speaking,
                    "non_spoken_terminator": unit.non_spoken_terminator,
                    "closed_span_count": len(unit.closed_spans),
                },
            )
        )
        self._unit_index = unit_index + 1

    async def finish_stream(self, reason: str = "client_finished") -> None:
        """Finish the session, await any pending tool tasks, emit session_finished."""

        self._prepared = False
        if self._tool_tasks:
            # Wait briefly so still-running searches can emit board_card_updated.
            # We do not cancel them — completing search and pushing tool response
            # is fast (sub-second typically); a hung backend will be capped by
            # urllib timeouts inside live_image_search_backend.
            await asyncio.gather(*self._tool_tasks, return_exceptions=True)
        await self._send_event(
            BoardEvent(
                type="session_finished",
                session_id=self.session_id,
                payload={"reason": reason, "unit_count": self._unit_index},
            )
        )

    # --------------------------------------------------- non-spoken loop

    async def _call_fc(self, method, *args, **kwargs):
        """Call an fc method that may be sync (RemoteFcDuplexView / UnifiedProcessor
        local) or async (WsRemoteFcDuplexView). Duck-typed via inspect.

        Sync methods run in a thread to keep the event loop responsive.
        """

        import inspect

        result = method(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        # sync call returned a real value; offload via to_thread to keep loop responsive
        # (we can't to_thread retroactively, so this path is only safe when the call
        # is already fast; for the heavy paths we rely on async fc).
        return result

    def _maybe_dump_speak_wav(self, *, unit_index: int, spoken: Any) -> None:
        """将本 unit 的 TTS waveform 落成一个 WAV + 一行 manifest。

        只对 `is_speaking=True 且 audio_waveform 非空` 的 unit 生效。目录：
            <audio_dump_dir>/<session_id>/unit_<idx>_speak.wav
        manifest.jsonl 每行：
            {"unit": N, "text": "...", "sample_rate": 24000,
             "n_samples": M, "spoken_turn_eos": bool, "ts": <epoch>}

        用于事后回听诊断"重叠/截断/TTS 走音"这些主观问题——只有事后能拿到
        每个 unit 的原始波形，才能判断是 TTS 本身的问题还是前端播放路径的
        问题。为了不阻塞事件循环，写盘走 asyncio.to_thread 也不至于（10KB
        /unit，同步 write 也就微秒级），这里直接同步写。
        """

        if self._audio_dump_session_dir is None:
            return
        if not getattr(spoken, "is_speaking", False):
            return
        waveform = getattr(spoken, "audio_waveform", None)
        if waveform is None:
            return
        array = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if array.size == 0:
            return
        sr = int(getattr(spoken, "audio_sample_rate", None) or 24000)
        wav_path = self._audio_dump_session_dir / f"unit_{unit_index:04d}_speak.wav"
        try:
            sf.write(str(wav_path), array, sr, format="WAV")
        except Exception as exc:  # noqa: BLE001 - dump 是诊断能力，出错不能挂主流程
            print(
                f"[session {self.session_id} audio_dump] wav write failed unit={unit_index}: {exc}",
                flush=True,
            )
            return
        manifest_line = {
            "unit": unit_index,
            "role": "speak",
            "text": getattr(spoken, "spoken_text", "") or "",
            "sample_rate": sr,
            "n_samples": int(array.size),
            "duration_ms": round(1000.0 * array.size / sr, 1),
            "spoken_turn_eos": bool(getattr(spoken, "spoken_turn_eos", False)),
            "wav": wav_path.name,
            "ts": time.time(),
        }
        self._append_manifest(manifest_line)

    def _maybe_dump_user_wav(
        self, *, unit_index: int, samples: np.ndarray | None, sample_rate: int
    ) -> None:
        """把这一 unit 的用户输入音频落成 unit_NNNN_user.wav。

        同一个 manifest.jsonl 用 `role` 字段区分 user vs speak。RMS 也一起
        记录，事后 `jq 'select(.role=="user") | .rms > 0.03'` 就能过滤出用户
        真说话的 unit，跟 speak 事件对齐看模型响应时机。

        用户不说话（rms<0.001 / n_samples=0）就跳过，避免落一堆空静音 wav。
        """

        if self._audio_dump_session_dir is None:
            return
        if samples is None or samples.size == 0:
            return
        rms = float(np.sqrt(np.mean(samples * samples)))
        peak = float(np.max(np.abs(samples)))
        # 只落有内容的 unit（rms>0.001 排除完全静音，允许收进背景噪声）；
        # 完全静音的 unit 太多会污染目录，靠 manifest 里的 unit 序号就能
        # 推断哪些是没落的静默段。
        if rms < 0.001:
            return
        wav_path = self._audio_dump_session_dir / f"unit_{unit_index:04d}_user.wav"
        try:
            sf.write(str(wav_path), samples, int(sample_rate), format="WAV")
        except Exception as exc:  # noqa: BLE001 - 诊断路径，出错不能挂主流程
            print(
                f"[session {self.session_id} audio_dump] user wav write failed unit={unit_index}: {exc}",
                flush=True,
            )
            return
        self._append_manifest({
            "unit": unit_index,
            "role": "user",
            "sample_rate": int(sample_rate),
            "n_samples": int(samples.size),
            "duration_ms": round(1000.0 * samples.size / sample_rate, 1),
            "rms": round(rms, 5),
            "peak": round(peak, 4),
            "wav": wav_path.name,
            "ts": time.time(),
        })

    def _append_manifest(self, line: dict[str, Any]) -> None:
        """向 session dump manifest 追加一行 JSON。写盘失败仅打印不抛。"""

        if self._audio_dump_manifest_path is None:
            return
        try:
            with self._audio_dump_manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            print(
                f"[session {self.session_id} audio_dump] manifest append failed: {exc}",
                flush=True,
            )

    async def _run_non_spoken_loop(
        self, unit_index: int, *, is_speaking: bool
    ) -> tuple[int, bool]:
        """Step through non-spoken tokens, emitting delta events and reacting to closed spans.

        Stop semantics:

        - **Always protocol-correct**: stopping the decode loop on the
          next-prefill signal is *not* a model-state corruption. Per SDK
          (`ai_non_spoken.py` slot rules + `parser.py` cross-unit wrapper
          handling), when the loop exits early, `process_audio_chunk` calls
          `view.finalize_unit`, which auto-emits
          `<|non_spoken_budget_reached|>` + `<|ai_non_spoken_slot_end|>` if
          the slot is still open. That terminator marks "this unit's budget
          is up; the same think/tool_call span continues in the next unit",
          and the view preserves `_non_spoken_mode` / `_think_buf` /
          `_tool_call_buf` across `finalize_unit`. So whether or not we
          short-circuit, the on-wire token sequence is protocol-valid.

        - **is_speaking=True**: cooperative stop on the next-prefill signal.
          Real-time matters because the user is hearing AI speech now and
          will hear the next 1s soon; we cannot let the unit blow past 1s
          wall-clock.

        - **is_speaking=False (listen)**: pass a permanently-clear stop
          event, so the server's decode loop runs to view-side terminate
          (`no_action` / `non_spoken_eos` / `non_spoken_budget_reached`) or
          the hard cap. The user is talking, AI is silent in this unit, so
          a few hundred extra ms of background decode are imperceptible. We
          do this NOT to "fix" any protocol issue but as a pragmatic
          workaround for current ckpts whose cross-unit continuation
          quality is inconsistent: if the model can emit
          `</think>...<tool_call>...</tool_call>` entirely within one
          long-enough decode window, we never have to rely on its
          cross-unit continuation ability at all.

        When the fc client supports `stream_non_spoken_decode`, the decode
        loop runs server-side and we just consume `decode_step` events; the
        stop signal flows over ws as `decode_stop`. The server checks the
        stop flag at the TOP of every iteration, so any in-flight step
        completes and arrives as a regular `decode_step` before the loop
        exits — average extra latency on stop ≈ half a token's decode time.

        Otherwise we fall back to the legacy per-step RPC pattern.

        Args:
            unit_index: current FC duplex unit index, used for event tagging.
            is_speaking: whether this unit's spoken slot produced output.

        Returns:
            (n_steps, short_circuited): short_circuited is True iff the
            server (or this loop, in the fallback path) exited via the stop
            signal rather than via view-side terminate.
        """

        # Listen units: never short-circuit. See docstring for rationale.
        stop_event = (
            self._next_prefill_event if is_speaking else asyncio.Event()
        )

        if hasattr(self.fc, "stream_non_spoken_decode"):
            return await self._run_non_spoken_loop_streaming(
                unit_index, stop_event=stop_event
            )
        return await self._run_non_spoken_loop_polling(
            unit_index, stop_event=stop_event
        )

    async def _run_non_spoken_loop_streaming(
        self, unit_index: int, *, stop_event: asyncio.Event
    ) -> tuple[int, bool]:
        """Streaming variant: one ws `decode_start` → many `decode_step` → `decode_end`.

        Server-side loop reads `decode_stop` (sent when stop_event fires)
        from a separate ws frame and checks at the top of every iteration.
        """

        stream = self.fc.stream_non_spoken_decode(
            decode_mode=self._decode_mode,
            stop_event=stop_event,
        )
        n_steps = 0
        async for step in stream:
            n_steps += 1
            await self._emit_step_events(step, unit_index)
            if step.terminated:
                # view-side terminated (no_action / slot_eos / non_spoken_budget_reached etc.)
                # short_circuited = False because the model itself decided.
                return n_steps, False
        # iteration ended via decode_end. last_reason tells us why.
        short_circuited = stream.last_reason == "stopped_by_client"
        return stream.last_n_steps or n_steps, short_circuited

    async def _run_non_spoken_loop_polling(
        self, unit_index: int, *, stop_event: asyncio.Event
    ) -> tuple[int, bool]:
        """Legacy variant: per-step RPC poll loop with stop check at top."""

        n_steps = 0
        for _ in range(self._MAX_NON_SPOKEN_STEPS_PER_UNIT):
            # Check stop flag at the TOP of every iteration. Any in-flight
            # step is already in `step` from the previous iteration and was
            # emitted, so we can exit cleanly here.
            if stop_event.is_set():
                return n_steps, True
            step = await self._call_fc(
                self.fc.streaming_non_spoken_generate,
                FcNonSpokenGenerateRequest(
                    max_tokens=1, decode_mode=self._decode_mode
                ),
            )
            n_steps += 1
            await self._emit_step_events(step, unit_index)
            if step.terminated:
                return n_steps, False
        # Hard cap fallback (death-loop protection).
        close = await self._call_fc(
            self.fc.streaming_non_spoken_generate,
            FcNonSpokenGenerateRequest(
                max_tokens=0,
                decode_mode=self._decode_mode,
                close_reason="budget_reached",
            ),
        )
        await self._emit_step_events(close, unit_index)
        return n_steps, False

    async def _emit_step_events(self, step: Any, unit_index: int) -> None:
        """Convert one FcNonSpokenGenerateResult into board events."""

        token_ids = list(step.token_ids or [])
        token_strs = list(step.token_strs or [])
        step_text = step.text or ""

        # 判断这一步是否是"有内容"的一步：只要 step 产出的 BPE decode 文本
        # 非空、或者带任何 token id 就算。之前只在 token_strs 里含
        # <think>/<tool_call> 才开 block —— 但实际 ckpt 的 tokenizer 把
        # opener 编成 special id，convert_ids_to_tokens 回来的 str 可能是
        # `<|think|>` 甚至空串，永远匹配不上，结果 units 9-11 的所有 delta
        # 都被静默丢掉，只在 unit 12 的 closed_span 里才被合成一个空 block
        # 立即 close —— 就是用户截图 STREAMING 层完全空的原因。
        # 现在改成：只要有 token 就开 block，kind 先记 unknown，等 closed_span
        # 或后续步骤能识别时再 upgrade。
        # 需要过滤掉的例外：`<|non_spoken_no_action|>` 单 token + 立刻
        # terminated —— 这不是内容，不该开 block。
        is_no_action_marker = (
            bool(step.terminated) and getattr(step, "close_reason", None) == "no_action"
        )
        should_open = (
            self._current_block_id is None
            and (token_ids or step_text)
            and not is_no_action_marker
        )
        if should_open:
            # 先给一个 tentative kind：能从 token_strs 一眼看出就直接标注，
            # 看不出就 unknown（等 closed_span 或后续 step 揭晓）
            tentative_kind = _guess_block_kind_from_tokens(token_strs) if token_strs else None
            await self._begin_block(tentative_kind, unit_index)

        # 每一步都发 delta —— 前端字符级 streaming 就靠这个。
        # 原始需求："每当从后端收到这个的，就要把这个字给它显示，及时地
        #   显示上去" + "streaming 里边就去呈现它的原始的那个内容就行了，
        #   你不需要做 XML 解析。包括 <tool_call> 和 </tool_call> 你也是
        #   都是带上的。"
        # 所以 step_text（BPE 逐 step 增量，包含所有 XML 标签）是唯一可靠
        # 的 streaming 来源；token_strs 带上做 fallback / 展示用。
        if (token_ids or step_text) and self._current_block_id is not None:
            await self._send_event(
                BoardEvent(
                    type="non_spoken_delta",
                    session_id=self.session_id,
                    unit_index=unit_index,
                    block_id=self._current_block_id,
                    block_kind=_kind_for_event(self._current_block_kind),
                    token_ids=token_ids,
                    token_strs=token_strs,
                    step_text=step_text,
                )
            )

        # closed_spans 可能在同一 step 里出现（如 single-token EOS 同步带闭合，
        # 或 mock view 一步直接给出 closed span 而无 token）
        for span in (step.closed_spans or []):
            await self._emit_close_for_span(span, unit_index)

    async def _begin_block(self, kind: str | None, unit_index: int) -> None:
        """Create a new non-spoken block and emit `non_spoken_block_started` (awaited)."""

        self._block_seq_global += 1
        # block_id 使用全局递增序号；unit_index 只做展示用途，不用来标识块。
        block_id = f"nsb-{self._block_seq_global}"
        self._current_block_id = block_id
        self._current_block_kind = kind
        await self._send_event(
            BoardEvent(
                type="non_spoken_block_started",
                session_id=self.session_id,
                unit_index=unit_index,
                block_id=block_id,
                block_kind=_kind_for_event(kind),
            )
        )

    async def _emit_close_for_span(self, span: Any, unit_index: int) -> None:
        """Emit non_spoken_block_closed + the derived think_final / tool_call_final event.

        如果当前没有 active block（例如 mock view 一步直接给出 closed span 且无 token），
        先合成一个 block_started 事件，让前端有容器可关闭。
        """

        span_type = getattr(span, "type", None)
        # Business log: every closed span deserves a line so we can tell whether
        # the model is emitting `think` blocks (no board impact) vs `tool_call`
        # blocks (potential board card), and what the raw wire / parse result is.
        wire_preview = (getattr(span, "wire", None) or "")[:120].replace("\n", "\\n")
        tool_call = getattr(span, "tool_call", None)
        span_error = getattr(span, "error", None)
        text_preview = (getattr(span, "text", None) or "")[:80].replace("\n", "\\n")
        print(
            f"[session {self.session_id} unit={unit_index} closed_span] "
            f"type={span_type} "
            f"tool_call_id={getattr(span, 'tool_call_id', None)!r} "
            f"name={(tool_call or {}).get('name') if isinstance(tool_call, dict) else None!r} "
            f"args={(tool_call or {}).get('arguments') if isinstance(tool_call, dict) else None!r} "
            f"error={span_error!r} "
            f"text_preview={text_preview!r} "
            f"wire_preview={wire_preview!r}",
            flush=True,
        )

        if self._current_block_id is None:
            await self._begin_block(span_type, unit_index)

        block_id = self._current_block_id
        block_kind = span_type or self._current_block_kind
        full_text = _full_text_from_span(span)

        await self._send_event(
            BoardEvent(
                type="non_spoken_block_closed",
                session_id=self.session_id,
                unit_index=unit_index,
                block_id=block_id,
                block_kind=_kind_for_event(block_kind),
                full_text=full_text,
            )
        )

        if span_type == "think":
            await self._send_event(
                BoardEvent(
                    type="think_final",
                    session_id=self.session_id,
                    unit_index=unit_index,
                    block_id=block_id,
                    think_text=full_text or "",
                )
            )
        elif span_type == "tool_call":
            tool_call = _tool_call_view(span)
            await self._send_event(
                BoardEvent(
                    type="tool_call_final",
                    session_id=self.session_id,
                    unit_index=unit_index,
                    block_id=block_id,
                    tool_call=tool_call,
                )
            )
            if tool_call.error is None and tool_call.name:
                await self._dispatch_tool_call(tool_call, unit_index)

        # close current block
        self._current_block_id = None
        self._current_block_kind = None

    # --------------------------------------------------- tool dispatch

    async def _dispatch_tool_call(
        self,
        tool_call: ToolCallView,
        unit_index: int,
    ) -> None:
        """Emit `board_card_created` immediately, then spawn async image search."""

        query = _query_from_tool_call(tool_call)
        card_id = f"card:{tool_call.tool_call_id or query}"
        created_card = BoardCard(
            card_id=card_id,
            tool_call_id=tool_call.tool_call_id,
            query=query,
            status="searching",
        )
        await self._send_event(
            BoardEvent(
                type="board_card_created",
                session_id=self.session_id,
                unit_index=unit_index,
                card=created_card,
            )
        )
        task = asyncio.create_task(
            self._run_tool_search(
                tool_call=tool_call,
                card_id=card_id,
                query=query,
                unit_index=unit_index,
            )
        )
        self._tool_tasks.add(task)
        task.add_done_callback(self._tool_tasks.discard)

    async def _run_tool_search(
        self,
        *,
        tool_call: ToolCallView,
        card_id: str,
        query: str,
        unit_index: int,
    ) -> None:
        """Background task: search image, emit board_card_updated, push tool response."""

        started_at = time.perf_counter()
        try:
            result = await asyncio.to_thread(self.tool_service.search, query)
            image_result = board_image_result_from_tool_result(
                result, tool_call_id=tool_call.tool_call_id
            )
            updated_card = BoardCard(
                card_id=card_id,
                tool_call_id=tool_call.tool_call_id,
                query=query,
                status="error" if result.error else "ready",
                image=image_result,
                error=result.error,
            )
            await self._send_event(
                BoardEvent(
                    type="board_card_updated",
                    session_id=self.session_id,
                    unit_index=unit_index,
                    card=updated_card,
                )
            )
            if tool_call.tool_call_id:
                async with self._pending_lock:
                    self._pending_tool_responses.append(
                        FcToolResponse(
                            call_id=tool_call.tool_call_id,
                            content=result.tool_response_content,
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            updated_card = BoardCard(
                card_id=card_id,
                tool_call_id=tool_call.tool_call_id,
                query=query,
                status="error",
                error=error,
            )
            await self._send_event(
                BoardEvent(
                    type="board_card_updated",
                    session_id=self.session_id,
                    unit_index=unit_index,
                    card=updated_card,
                )
            )
            if tool_call.tool_call_id:
                async with self._pending_lock:
                    # 即便 search 挂了，也要给模型一个 tool_response，避免悬空 tool_call_id
                    self._pending_tool_responses.append(
                        FcToolResponse(
                            call_id=tool_call.tool_call_id,
                            content=json.dumps(
                                {
                                    "status": "error",
                                    "name": query,
                                    "reason": f"运行时搜图失败：{error}",
                                },
                                ensure_ascii=False,
                            ),
                        )
                    )
        finally:
            del started_at  # placeholder for future timing log


# ============================================================ helpers


def _tool_call_view(span: Any) -> ToolCallView:
    raw = getattr(span, "tool_call", None) or {}
    name = raw.get("name") if isinstance(raw, dict) else None
    arguments = raw.get("arguments") if isinstance(raw, dict) else None
    span_error = getattr(span, "error", None)
    raw_error = raw.get("error") if isinstance(raw, dict) else None
    return ToolCallView(
        tool_call_id=getattr(span, "tool_call_id", None),
        name=name,
        arguments=arguments,
        error=span_error or raw_error,
        wire=getattr(span, "wire", None),
    )


def _query_from_tool_call(tool_call: ToolCallView) -> str:
    args = tool_call.arguments
    if isinstance(args, dict):
        value = args.get("name") or args.get("query")
        if value is not None:
            return str(value)
    return tool_call.name or "unknown object"


def _full_text_from_span(span: Any) -> str | None:
    span_type = getattr(span, "type", None)
    if span_type == "think":
        return getattr(span, "text", None) or ""
    if span_type == "tool_call":
        tool_call = getattr(span, "tool_call", None)
        if isinstance(tool_call, dict) and tool_call.get("name"):
            # Reconstruct a readable JSON-ish full_text from the parsed tool call;
            # if parser failed, we leave full_text empty so the frontend's full
            # layer stays empty and only the streaming layer is visible (as the
            # original requirement specified).
            try:
                return json.dumps(
                    {
                        "name": tool_call.get("name"),
                        "arguments": tool_call.get("arguments"),
                    },
                    ensure_ascii=False,
                )
            except Exception:
                return None
        return None
    return None


def _guess_block_kind_from_tokens(token_strs: list[str]) -> str | None:
    """Best-effort kind detection from the first token piece of a block.

    Returns "think" or "tool_call" when an opener token appears, otherwise None
    (caller renders as `unknown` until kind becomes clear or closed span tells us).
    """

    for piece in token_strs:
        if not isinstance(piece, str):
            continue
        if "think" in piece and "<" in piece:
            return "think"
        if "tool_call" in piece and "<" in piece:
            return "tool_call"
    return None


def _kind_for_event(kind: str | None) -> str:
    if kind in ("think", "tool_call"):
        return kind
    return "unknown"


def _audio_waveform_to_wav_base64(audio_waveform: Any, sample_rate: int) -> str | None:
    """Encode a generated waveform as base64 wav for browser playback."""

    if audio_waveform is None:
        return None
    array = np.asarray(audio_waveform, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return None
    buffer = io.BytesIO()
    sf.write(buffer, array, sample_rate, format="WAV")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _audio_waveform_to_float32_base64(audio_waveform: Any) -> str | None:
    """Encode a generated waveform as base64-encoded little-endian Float32 PCM.

    This is the format the demo's `static/duplex/lib/audio-player.js`
    consumes for pre-scheduled gapless playback via `AudioBufferSourceNode`.
    """

    if audio_waveform is None:
        return None
    array = np.asarray(audio_waveform, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return None
    # numpy default is platform-native; on x86_64 that's little-endian which
    # is what AudioBufferSourceNode + Float32Array view expects.
    return base64.b64encode(array.tobytes()).decode("ascii")
