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

    # 每个 unit 内 non-spoken slot 最多 step 数。real view 内部会用 generation_flag
    # / close_reason 主动停下；这里设硬上限只是兜底，避免极端坏例死循环。
    _MAX_NON_SPOKEN_STEPS_PER_UNIT = 40

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
            generate_audio=request.generate_audio,
        )
        await asyncio.to_thread(self.fc.prepare, prepare_request)
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

    async def process_audio_chunk(self, request: StreamAudioChunkRequest) -> None:
        """Process one browser-sent audio unit end-to-end (async)."""

        if not self._prepared:
            raise RuntimeError("stream session is not prepared")

        # 1) drain tool responses produced by background search since last prefill
        async with self._pending_lock:
            tool_responses = list(self._pending_tool_responses)
            self._pending_tool_responses = []

        prefill = await asyncio.to_thread(
            self.fc.streaming_prefill,
            FcDuplexPrefillRequest(
                audio_data=request.audio_base64,
                sample_rate=request.sample_rate,
                tool_responses=tool_responses or None,
            ),
        )
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
        spoken = await asyncio.to_thread(
            self.fc.streaming_spoken_generate,
            FcSpokenGenerateRequest(
                max_tokens=self._max_spoken_tokens,
                decode_mode=self._decode_mode,
            ),
        )
        if spoken.is_speaking or spoken.spoken_token_ids:
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
                        "audio_wav_base64": audio_wav_base64,
                        "audio_sample_rate": spoken.audio_sample_rate,
                    },
                )
            )

        # 3) non-spoken slot: step loop, emit deltas, react to closed spans
        await self._run_non_spoken_loop(unit_index)

        # 4) finalize unit
        unit = await asyncio.to_thread(self.fc.finalize_unit)
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

    async def _run_non_spoken_loop(self, unit_index: int) -> None:
        """Step through non-spoken tokens, emitting delta events and reacting to closed spans."""

        for _ in range(self._MAX_NON_SPOKEN_STEPS_PER_UNIT):
            step = await asyncio.to_thread(
                self.fc.streaming_non_spoken_generate,
                FcNonSpokenGenerateRequest(
                    max_tokens=1, decode_mode=self._decode_mode
                ),
            )
            await self._emit_step_events(step, unit_index)
            if step.terminated:
                return
        # 兜底：超 step 上限，主动闭合一次
        close = await asyncio.to_thread(
            self.fc.streaming_non_spoken_generate,
            FcNonSpokenGenerateRequest(
                max_tokens=0,
                decode_mode=self._decode_mode,
                close_reason="budget_reached",
            ),
        )
        await self._emit_step_events(close, unit_index)

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
