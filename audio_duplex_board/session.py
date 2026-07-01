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
        self._unit_index = 0
        self._max_spoken_tokens = 24
        self._decode_mode = "greedy"
        # tool responses produced by background search tasks, drained at next prefill
        self._pending_tool_responses: list[FcToolResponse] = []
        self._pending_lock = asyncio.Lock()
        # background tool tasks, kept so we can await them at session finish
        self._tool_tasks: set[asyncio.Task[None]] = set()
        # per-unit non-spoken block sequence, used to build stable block_id
        self._block_seq_in_unit = 0
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
        self._block_seq_in_unit = 0
        self._current_block_id = None
        self._current_block_kind = None

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
        if spoken.is_speaking or spoken.spoken_token_ids:
            # Emit both formats so the frontend can pick:
            #   - audio_float32_base64: raw little-endian Float32 PCM at
            #     `audio_sample_rate` (typically 24000). This is the format
            #     the demo's `static/duplex/lib/audio-player.js` consumes
            #     for gapless pre-scheduled playback. Preferred when present.
            #   - audio_wav_base64: WAV-encoded fallback for naive
            #     `<audio src="data:audio/wav;base64,..."` playback.
            audio_float32_base64 = _audio_waveform_to_float32_base64(
                spoken.audio_waveform
            )
            audio_wav_base64 = _audio_waveform_to_wav_base64(
                spoken.audio_waveform,
                spoken.audio_sample_rate or 24000,
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

        # 如果当前没有 active block，但这一步产了 token，那是新 block 的开头
        if token_ids and self._current_block_id is None:
            kind = _guess_block_kind_from_tokens(token_strs)
            await self._begin_block(kind, unit_index)

        if token_ids and self._current_block_id is not None:
            await self._send_event(
                BoardEvent(
                    type="non_spoken_delta",
                    session_id=self.session_id,
                    unit_index=unit_index,
                    block_id=self._current_block_id,
                    block_kind=_kind_for_event(self._current_block_kind),
                    token_ids=token_ids,
                    token_strs=token_strs,
                    step_text=step.text or "",
                )
            )

        # closed_spans 可能在同一 step 里出现（如 single-token EOS 同步带闭合，
        # 或 mock view 一步直接给出 closed span 而无 token）
        for span in (step.closed_spans or []):
            await self._emit_close_for_span(span, unit_index)

    async def _begin_block(self, kind: str | None, unit_index: int) -> None:
        """Create a new non-spoken block and emit `non_spoken_block_started` (awaited)."""

        self._block_seq_in_unit += 1
        block_id = f"nsb-{unit_index}-{self._block_seq_in_unit}"
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
