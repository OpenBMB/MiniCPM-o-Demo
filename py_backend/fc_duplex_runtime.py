"""Minimal FC duplex runtime for the backend protocol server.

This module adapts the standalone FC board MVP scheduling pattern to the
backend protocol without importing the old standalone server/web stack. It keeps the
model-facing FC primitive calls intact and only owns API event shaping plus the
external-tool-id to internal-tool-id mapping.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Dict, List, Optional, Protocol, cast

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from core.fc_duplex_resume import (
    FcDuplexResumeError,
    FcResumeFailureCode,
    build_fc_duplex_resume_plan,
)
from core.schemas.fc_duplex import FcToolResponse, NonSpokenStepGenerationFlag
from py_backend.media import decode_audio_base64, decode_frame_base64_list

logger = logging.getLogger(__name__)


def _deferred_budget_reached_step(
    *,
    generation_steps: Optional[List[Any]] = None,
    warnings: Optional[List[Any]] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        token_ids=[],
        terminated=True,
        close_reason="budget_reached",
        generation_flag="continue_non_spoken_generation",
        closed_spans=[],
        text="",
        text_delta="",
        generation_steps=list(generation_steps or []),
        warnings=list(warnings or []),
        metadata={"deferred_model_feed": True},
    )


DEFAULT_DISPLAY_OBJECT_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "display_object_on_board",
        "description": (
            "Display a named concrete object on the visual board so the user can "
            "see it. Use only for concrete, visualizable objects mentioned in "
            "user speech."
        ),
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
}


class _SendCallable(Protocol):
    """Async backend event sender accepting arbitrary public event fields."""

    def __call__(self, event_type: str, **fields: Any) -> Awaitable[None]:
        """Send one public backend event."""


class _FatalCallable(Protocol):
    """Async callback used when detached FC processing cannot continue."""

    def __call__(self, error: Exception) -> Awaitable[None]:
        """Close the owning protocol Session with a public fatal event."""


class FcDuplexSessionRuntime:
    """Small per-session scheduler for FC duplex protocol input/output."""

    def __init__(
        self,
        *,
        session_id: str,
        backend: Any,
        send: _SendCallable,
        on_fatal: Optional[_FatalCallable] = None,
    ) -> None:
        self.session_id = session_id
        self.backend = backend
        self._send = send
        self._on_fatal = on_fatal
        self._response_id: Optional[str] = None
        self._tools: List[Dict[str, Any]] = []
        self._pending_tool_responses: List[FcToolResponse] = []
        self._streaming_tool_results: Dict[str, List[Any]] = {}
        self._api_to_internal: Dict[str, str] = {}
        self._internal_to_api: Dict[str, str] = {}
        self._tool_seq = 0
        self._max_spoken_tokens = 24
        self._checkpoint_profile_id: Optional[str] = None
        self._non_spoken_budget_while_listening: Optional[int] = None
        self._non_spoken_budget_while_speaking: Optional[int] = None
        self._non_spoken_scheduling: Optional[str] = None
        self._decode_mode = "greedy"
        self._sample_rate = 16000
        self._input_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._seen_input_ids: set[str] = set()
        self._queue_worker: Optional[asyncio.Task[None]] = None
        self._next_input_event = asyncio.Event()
        self._closed = False
        self._current_block_id: Optional[str] = None
        self._current_block_kind: Optional[str] = None
        self._current_tool_call_id: Optional[str] = None
        self._current_block_streamed = False
        self._block_seq = 0
        self._block_started_sent = False
        self._audio_dump_unit_seq = 0
        self._audio_dump_session_dir: Optional[Path] = None
        self._audio_dump_manifest_path: Optional[Path] = None
        self._generation_step_index = 0
        self._generation_batch_index = 0
        self._generation_event_index = 0
        self._delta_index_by_stream: Dict[str, int] = {}
        self._pending_text_steps_by_stream: Dict[str, List[int]] = {}
        self._generation_batch_stream_id: Optional[str] = None
        self._generation_batch_track: Optional[str] = None
        self._generation_batch_input_id: Optional[str] = None
        self._generation_batch_unit_index: Optional[int] = None
        self._generation_batch_started_at: Optional[float] = None
        self._generation_batch_steps: List[Dict[str, Any]] = []
        self._unit_has_deferred_close = False
        self._unit_non_spoken_end: Optional[str] = None
        self._resume_identity: Dict[str, Any] = {}
        self._emit_debug_events = (
            os.environ.get("FC_DUPLEX_DEBUG_EVENTS", "0") == "1"
        )
        audio_dump_root = os.environ.get("FC_DUPLEX_AUDIO_DUMP_DIR")
        if audio_dump_root:
            self._audio_dump_session_dir = Path(audio_dump_root) / self.session_id
            self._audio_dump_session_dir.mkdir(parents=True, exist_ok=True)
            self._audio_dump_manifest_path = self._audio_dump_session_dir / "manifest.jsonl"
            logger.info(
                "fc_duplex audio dump enabled: session=%s dir=%s",
                self.session_id,
                self._audio_dump_session_dir,
            )

    async def prepare(self, params: Dict[str, Any]) -> None:
        config = _first_dict(params.get("config"), params.get("duplex"), params.get("fc_duplex"))
        self._max_spoken_tokens = int(config.get("max_spoken_tokens", params.get("max_spoken_tokens", 24)) or 24)
        self._checkpoint_profile_id = str(
            _coalesce(
                config.get("checkpoint_profile_id"),
                params.get("checkpoint_profile_id"),
                os.environ.get("CHECKPOINT_PROFILE_ID"),
                default="",
            )
        ).strip() or None
        profile_listening_budget = _optional_positive_int(
            os.environ.get("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING"),
            field_name="FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING",
        )
        profile_speaking_budget = _optional_positive_int(
            os.environ.get("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING"),
            field_name="FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING",
        )
        requested_listening_budget = _optional_positive_int(
            _coalesce(
                config.get("non_spoken_budget_while_listening"),
                params.get("non_spoken_budget_while_listening"),
            ),
            field_name="non_spoken_budget_while_listening",
        )
        requested_speaking_budget = _optional_positive_int(
            _coalesce(
                config.get("non_spoken_budget_while_speaking"),
                params.get("non_spoken_budget_while_speaking"),
            ),
            field_name="non_spoken_budget_while_speaking",
        )
        legacy_budget = _optional_positive_int(
            _coalesce(
                config.get("non_spoken_budget_per_unit"),
                params.get("non_spoken_budget_per_unit"),
            ),
            field_name="non_spoken_budget_per_unit",
        )
        if requested_listening_budget is None and legacy_budget is not None:
            requested_listening_budget = legacy_budget
        if requested_speaking_budget is None and legacy_budget is not None:
            requested_speaking_budget = legacy_budget
        self._non_spoken_budget_while_listening = _resolve_profile_bound_budget(
            requested=requested_listening_budget,
            registered=profile_listening_budget,
            field_name="non_spoken_budget_while_listening",
        )
        self._non_spoken_budget_while_speaking = _resolve_profile_bound_budget(
            requested=requested_speaking_budget,
            registered=profile_speaking_budget,
            field_name="non_spoken_budget_while_speaking",
        )
        if (
            self._non_spoken_budget_while_listening is None
            or self._non_spoken_budget_while_speaking is None
        ):
            raise RuntimeError(
                "fc_duplex non-spoken budgets must be provided explicitly by "
                "Checkpoint Profile or session.init"
            )
        registered_scheduling = _optional_scheduling(
            os.environ.get("FC_DUPLEX_NON_SPOKEN_SCHEDULING"),
            field_name="FC_DUPLEX_NON_SPOKEN_SCHEDULING",
        )
        requested_scheduling = _optional_scheduling(
            _coalesce(
                config.get("non_spoken_scheduling"),
                params.get("non_spoken_scheduling"),
            ),
            field_name="non_spoken_scheduling",
        )
        self._non_spoken_scheduling = _resolve_profile_bound_choice(
            requested=requested_scheduling,
            registered=registered_scheduling,
            field_name="non_spoken_scheduling",
        )
        if self._non_spoken_scheduling is None:
            raise RuntimeError(
                "fc_duplex non_spoken_scheduling must be provided explicitly by "
                "Checkpoint Profile or session.init"
            )
        if self._non_spoken_scheduling not in {"latency", "quality"}:
            raise RuntimeError("fc_duplex non_spoken_scheduling must be 'latency' or 'quality'")
        self._decode_mode = str(config.get("decode_mode", params.get("decode_mode", "greedy")) or "greedy")
        self._sample_rate = int(config.get("sample_rate", params.get("sample_rate", 16000)) or 16000)
        self._tools = list(params.get("tools") or [DEFAULT_DISPLAY_OBJECT_TOOL])

        voice = _first_dict(params.get("voice"), params.get("defaults"))
        ref_audio_path = _coalesce(params.get("ref_audio_path"), voice.get("ref_audio_path"))
        prompt_wav_path = _coalesce(params.get("prompt_wav_path"), params.get("tts_ref_audio_path"), voice.get("tts_ref_audio_path"), ref_audio_path)
        await asyncio.to_thread(
            self.backend.fc_duplex_prepare,
            system_prompt=str(_coalesce(params.get("system_prompt"), params.get("instructions"), default="")),
            tools=self._tools,
            ref_audio_path=ref_audio_path,
            prompt_wav_path=prompt_wav_path,
            generate_audio=bool(params.get("generate_audio", True)),
        )
        resume_identity_fn = getattr(self.backend, "fc_duplex_resume_identity", None)
        self._resume_identity = (
            await asyncio.to_thread(resume_identity_fn)
            if resume_identity_fn is not None
            else {}
        )
        self._resume_identity.update(
            {
                "checkpoint_profile_id": self._checkpoint_profile_id,
                "non_spoken_budget_while_listening": (
                    self._non_spoken_budget_while_listening
                ),
                "non_spoken_budget_while_speaking": (
                    self._non_spoken_budget_while_speaking
                ),
                "non_spoken_scheduling": self._non_spoken_scheduling,
            }
        )

    @property
    def resume_identity(self) -> Dict[str, Any]:
        """Return a copy of the current public resume identity metadata."""

        return dict(self._resume_identity)

    async def resume(self, params: Dict[str, Any]) -> None:
        """Rebuild a safe Unit checkpoint exclusively from client-provided history."""

        try:
            plan = build_fc_duplex_resume_plan(
                protocol_version=str(params.get("protocol_version") or ""),
                model=str(params.get("model") or ""),
                tokenizer_target=str(params.get("tokenizer_target") or ""),  # type: ignore[arg-type]
                tokenizer_fingerprint=dict(
                    params.get("tokenizer_fingerprint") or {}
                ),
                through_unit_index=int(params.get("through_unit_index", -1)),
                history=list(params.get("history") or []),
            )
        except FcDuplexResumeError:
            raise
        except (TypeError, ValueError) as exc:
            raise FcDuplexResumeError(
                "incomplete_event_history",
                f"invalid session.resume payload: {exc}",
            ) from exc
        await self.prepare(plan.session_init_payload)
        has_model_identity = any(
            key in self._resume_identity
            for key in (
                "tokenizer_target",
                "tokenizer_fingerprint",
                "model",
                "ref_audio_sha256",
                "prompt_wav_sha256",
            )
        )
        if has_model_identity:
            requested_fingerprint = dict(params.get("tokenizer_fingerprint") or {})
            current_fingerprint = dict(
                self._resume_identity.get("tokenizer_fingerprint") or {}
            )
            current_model = str(self._resume_identity.get("model") or "")
            requested_model = str(params.get("model") or "")
            identity_mismatch = (
                str(params.get("tokenizer_target") or "")
                != str(self._resume_identity.get("tokenizer_target") or "")
                or (
                    current_model not in {"", "unknown"}
                    and requested_model != current_model
                )
                or (
                    requested_fingerprint
                    and requested_fingerprint != current_fingerprint
                )
                or params.get("ref_audio_sha256")
                != self._resume_identity.get("ref_audio_sha256")
                or params.get("prompt_wav_sha256")
                != self._resume_identity.get("prompt_wav_sha256")
            )
            if identity_mismatch:
                raise FcDuplexResumeError(
                    "model_or_tokenizer_mismatch",
                    "resume request does not match current model/tokenizer identity",
                    unit_index=plan.through_unit_index,
                )
        for unit in plan.units:
            payload = unit.input_payload
            audio_base64 = _extract_audio_base64(payload)
            frame_list = decode_frame_base64_list(
                _extract_frame_base64_list(payload)
            ).frame_list
            sample_rate = int(payload.get("sample_rate") or self._sample_rate)
            await asyncio.to_thread(
                self.backend.fc_duplex_replay_completed_unit,
                audio_data=audio_base64,
                frame_list=frame_list,
                tool_responses=list(unit.tool_events) or None,
                sample_rate=sample_rate,
                spoken_token_ids=unit.spoken_token_ids,
                non_spoken_token_ids=unit.non_spoken_token_ids,
                deferred_non_spoken_close=unit.deferred_non_spoken_close,
            )
        restore_stream_sequence = getattr(
            self.backend,
            "fc_duplex_restore_generation_stream_sequence",
            None,
        )
        if restore_stream_sequence is not None:
            await asyncio.to_thread(
                restore_stream_sequence,
                next_stream_sequence=plan.next_stream_sequence,
            )
        restore_tool_call_sequence = getattr(
            self.backend,
            "fc_duplex_restore_tool_call_sequence",
            None,
        )
        if restore_tool_call_sequence is not None:
            await asyncio.to_thread(
                restore_tool_call_sequence,
                tool_call_count=plan.tool_call_count,
            )
        self._tool_seq = plan.api_tool_call_sequence

        boundary_status_fn = getattr(
            self.backend,
            "fc_duplex_resume_boundary_status",
            None,
        )
        boundary_status = (
            await asyncio.to_thread(boundary_status_fn)
            if boundary_status_fn is not None
            else {"status": "unavailable", "reason": "unsupported_open_span"}
        )
        if boundary_status.get("status") != "available":
            raise FcDuplexResumeError(
                cast(
                    FcResumeFailureCode,
                    str(
                        boundary_status.get(
                            "reason",
                            "unsupported_open_span",
                        )
                    ),
                ),
                "replayed View state is not resumable",
                unit_index=plan.through_unit_index,
                stream_id=boundary_status.get("stream_id"),
            )

        self._audio_dump_unit_seq = plan.through_unit_index + 1
        self._generation_event_index = plan.next_event_index
        self._generation_step_index = plan.next_step_index
        self._generation_batch_index = plan.next_batch_index
        self._delta_index_by_stream = dict(plan.next_delta_index_by_stream)
        self._pending_text_steps_by_stream = {}
        self._seen_input_ids = set(plan.seen_input_ids)
        await self._send(
            "session.resumed",
            session_id=self.session_id,
            through_unit_index=plan.through_unit_index,
            next_unit_index=plan.through_unit_index + 1,
        )

    async def push(self, payload: Dict[str, Any]) -> None:
        payload_type = str(payload.get("type") or payload.get("event_type") or "")
        if payload_type == "tool_result":
            await self.queue_tool_result(payload)
            return
        if payload_type == "tool_result.delta":
            await self.queue_tool_result_delta(payload)
            return
        if payload_type == "tool_result.done":
            await self.finish_tool_result_stream(payload)
            return
        await self.enqueue_audio_input(payload)

    async def queue_tool_result(self, payload: Dict[str, Any]) -> None:
        api_id = str(payload.get("tool_call_id") or "")
        if not api_id:
            raise RuntimeError("input.tool_result requires tool_call_id")
        internal_id = self._api_to_internal.get(api_id)
        if not internal_id:
            raise RuntimeError(f"unknown tool_call_id: {api_id}")
        self._pending_tool_responses.append(
            FcToolResponse(
                call_id=internal_id,
                content=_contents_to_text(
                    payload.get("content", payload.get("contents"))
                ),
            )
        )

    async def queue_tool_result_delta(self, payload: Dict[str, Any]) -> None:
        api_id = str(payload.get("tool_call_id") or "")
        if not api_id:
            raise RuntimeError("input.tool_result.delta requires tool_call_id")
        if api_id not in self._api_to_internal:
            raise RuntimeError(f"unknown tool_call_id: {api_id}")
        self._streaming_tool_results.setdefault(api_id, []).append(payload.get("delta"))

    async def finish_tool_result_stream(self, payload: Dict[str, Any]) -> None:
        api_id = str(payload.get("tool_call_id") or "")
        if not api_id:
            raise RuntimeError("input.tool_result.done requires tool_call_id")
        internal_id = self._api_to_internal.get(api_id)
        if not internal_id:
            raise RuntimeError(f"unknown tool_call_id: {api_id}")
        deltas = self._streaming_tool_results.pop(api_id, None)
        if deltas is None:
            raise RuntimeError(f"tool_result.done without prior delta for tool_call_id: {api_id}")
        self._pending_tool_responses.append(
            FcToolResponse(call_id=internal_id, content=_contents_to_text(deltas))
        )

    async def enqueue_audio_input(self, payload: Dict[str, Any]) -> None:
        audio_base64 = _extract_audio_base64(payload)
        if not audio_base64:
            raise RuntimeError("fc_duplex input requires audio")
        input_id = str(payload.get("input_id") or "")
        if not input_id:
            raise RuntimeError("fc_duplex input requires input_id")
        if input_id in self._seen_input_ids:
            raise RuntimeError(f"duplicate fc_duplex input_id: {input_id}")
        self._seen_input_ids.add(input_id)
        worker_running = self._queue_worker is not None and not self._queue_worker.done()
        if self._non_spoken_scheduling == "latency":
            while not self._input_queue.empty():
                with suppress(asyncio.QueueEmpty):
                    self._input_queue.get_nowait()
                    self._input_queue.task_done()
        await self._input_queue.put(payload)
        if self._non_spoken_scheduling == "latency" and worker_running:
            self._next_input_event.set()
        if self._queue_worker is None or self._queue_worker.done():
            self._queue_worker = asyncio.create_task(self._consume_audio_queue())

    async def _consume_audio_queue(self) -> None:
        while not self._closed:
            payload = await self._input_queue.get()
            try:
                await self._process_audio_payload(payload)
            except Exception as exc:
                logger.exception("FC runtime failed to process audio payload")
                if self._on_fatal is None:
                    raise
                self._closed = True
                await self._on_fatal(exc)
                await self._dump_model_trace(reason="runtime_error")
                await asyncio.to_thread(self.backend.fc_duplex_cleanup)
                return
            finally:
                self._input_queue.task_done()
            if self._input_queue.empty():
                return

    async def _process_audio_payload(self, payload: Dict[str, Any]) -> None:
        unit_t0 = time.perf_counter()
        input_id = payload.get("input_id")
        self._response_id = self._response_id or str(payload.get("response_id") or f"resp_{uuid.uuid4().hex[:12]}")
        audio_base64 = _extract_audio_base64(payload)
        self._next_input_event.clear()
        if self._non_spoken_scheduling == "latency" and not self._input_queue.empty():
            self._next_input_event.set()
        frame_list = decode_frame_base64_list(_extract_frame_base64_list(payload)).frame_list
        tool_responses = list(self._pending_tool_responses)
        self._pending_tool_responses.clear()
        sample_rate = int(payload.get("sample_rate") or self._sample_rate)
        unit_index = self._audio_dump_unit_seq
        self._audio_dump_unit_seq += 1
        self._unit_has_deferred_close = False
        self._unit_non_spoken_end = None

        if self._audio_dump_session_dir is not None:
            self._maybe_dump_user_wav(
                unit_index=unit_index,
                samples=_safe_decode_audio_base64(audio_base64),
                sample_rate=sample_rate,
            )

        prefill = await asyncio.to_thread(
            self.backend.fc_duplex_prefill,
            audio_data=audio_base64,
            frame_list=frame_list,
            tool_responses=tool_responses or None,
            sample_rate=sample_rate,
        )
        await self._emit_unit_input_events(
            prefill,
            unit_index=unit_index,
            input_id=input_id,
        )

        spoken = await asyncio.to_thread(
            self.backend.fc_duplex_spoken_generate,
            max_tokens=self._max_spoken_tokens,
            decode_mode=self._decode_mode,
        )
        await self._emit_spoken(
            spoken,
            input_id=input_id,
            unit_index=unit_index,
        )
        if self._audio_dump_session_dir is not None:
            self._maybe_dump_speak_wav(unit_index=unit_index, spoken=spoken)
        spoken_done_elapsed_ms = (time.perf_counter() - unit_t0) * 1000

        await self._run_non_spoken_loop(
            input_id=input_id,
            unit_index=unit_index,
            pre_non_spoken_elapsed_ms=spoken_done_elapsed_ms,
            unit_budget=self._select_non_spoken_budget(spoken),
        )

        await asyncio.to_thread(self.backend.fc_duplex_finalize)
        await self._flush_generation_batch()
        resume_status_fn = getattr(
            self.backend, "fc_duplex_resume_boundary_status", None
        )
        resume_status = (
            await asyncio.to_thread(resume_status_fn)
            if resume_status_fn is not None
            else {
                "status": "unavailable",
                "reason": "unsupported_open_span",
            }
        )
        if self._unit_has_deferred_close:
            resume_status = {
                "status": "unavailable",
                "reason": "deferred_close",
            }
        pending_candidates = [
            (stream_id, step_indexes[0])
            for stream_id, step_indexes in self._pending_text_steps_by_stream.items()
            if step_indexes
        ]
        if pending_candidates:
            stream_id, pending_from_step = min(
                pending_candidates,
                key=lambda item: item[1],
            )
            resume_status = {
                "status": "unavailable",
                "reason": "pending_text_delta",
                "stream_id": stream_id,
                "pending_from_step": pending_from_step,
            }
        if self._unit_non_spoken_end is None:
            raise RuntimeError(
                f"Unit {unit_index} finalized without response.non_spoken.end"
            )
        await self._send(
            "response.unit.committed",
            unit_index=unit_index,
            resume=resume_status,
        )

    async def close(self) -> None:
        self._closed = True
        if self._queue_worker is not None:
            self._queue_worker.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._queue_worker
        await self._dump_model_trace(reason="session_close")
        await asyncio.to_thread(self.backend.fc_duplex_cleanup)

    def _maybe_dump_user_wav(self, *, unit_index: int, samples: Optional[np.ndarray], sample_rate: int) -> None:
        """把这一 unit 的用户输入音频落成 unit_NNNN_user.wav。

        Ported from audio_duplex_board.session._maybe_dump_user_wav (see
        docs/fc-duplex/o45-fc-merge-audit-2026-07-10.md). Only active when
        FC_DUPLEX_AUDIO_DUMP_DIR is set; opt-in, off by default. Diagnostic
        purpose: users reported ~50% non-response rates that turned out to be
        partly a loudness/phrasing mismatch with training data — only by
        listening back to what was actually captured per unit can you tell
        "mic captured silence" apart from "model chose not to respond".
        Same manifest.jsonl as the spoken-audio dump, disambiguated by
        `role`. Skips near-silent units (rms<0.001) to avoid flooding the
        dump dir with empty wavs.
        """

        if self._audio_dump_session_dir is None or samples is None or samples.size == 0:
            return
        rms = float(np.sqrt(np.mean(samples * samples)))
        peak = float(np.max(np.abs(samples)))
        if rms < 0.001:
            return
        wav_path = self._audio_dump_session_dir / f"unit_{unit_index:04d}_user.wav"
        try:
            sf.write(str(wav_path), samples, int(sample_rate), format="WAV")
        except Exception as exc:  # noqa: BLE001 - dump 是诊断能力，出错不能挂主流程
            logger.warning("audio dump: user wav write failed unit=%s: %s", unit_index, exc)
            return
        self._append_audio_dump_manifest({
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

    def _maybe_dump_speak_wav(self, *, unit_index: int, spoken: Any) -> None:
        """将本 unit 的 TTS waveform 落成一个 WAV + 一行 manifest。

        Ported from audio_duplex_board.session._maybe_dump_speak_wav (see
        docs/fc-duplex/o45-fc-merge-audit-2026-07-10.md). Only active when
        FC_DUPLEX_AUDIO_DUMP_DIR is set. Diagnostic purpose: lets you listen
        back to exactly what the model actually synthesized per unit, to
        separate "TTS itself glitched" from "frontend playback path glitched"
        when a user reports garbled/overlapping/cut-off AI speech.
        """

        if self._audio_dump_session_dir is None or not getattr(spoken, "is_speaking", False):
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
            logger.warning("audio dump: speak wav write failed unit=%s: %s", unit_index, exc)
            return
        self._append_audio_dump_manifest({
            "unit": unit_index,
            "role": "speak",
            "text": getattr(spoken, "spoken_text", "") or "",
            "sample_rate": sr,
            "n_samples": int(array.size),
            "duration_ms": round(1000.0 * array.size / sr, 1),
            "spoken_turn_eos": bool(getattr(spoken, "spoken_turn_eos", False)),
            "wav": wav_path.name,
            "ts": time.time(),
        })

    def _append_audio_dump_manifest(self, line: Dict[str, Any]) -> None:
        """向 session dump manifest 追加一行 JSON。写盘失败仅打印不抛。"""

        if self._audio_dump_manifest_path is None:
            return
        try:
            with self._audio_dump_manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning("audio dump: manifest append failed: %s", exc)

    async def _dump_model_trace(self, *, reason: str) -> None:
        dump = getattr(self.backend, "fc_duplex_dump_trace", None)
        if dump is None:
            return
        trace_dir = os.environ.get(
            "FC_DUPLEX_TRACE_DIR",
            "/tmp/minicpmo45_fc_trace_logs",
        )
        session = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.session_id or "session")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(trace_dir, f"fc_trace_{session}_{stamp}.json")
        try:
            info = await asyncio.to_thread(dump, path=path, session_id=self.session_id, reason=reason)
            logger.info("fc_model_trace_dumped session=%s path=%s info=%s", self.session_id, path, info)
        except Exception:
            logger.exception("failed to dump fc model trace: session=%s path=%s", self.session_id, path)

    async def _build_deferred_budget_reached_step(self) -> SimpleNamespace:
        """Close the View text stream while deferring the model close to finalize."""

        terminate = getattr(
            self.backend,
            "fc_duplex_terminate_non_spoken_text_stream",
            None,
        )
        if terminate is None:
            raise RuntimeError(
                "backend lacks fc_duplex_terminate_non_spoken_text_stream; "
                "cannot produce resumable deferred-close history"
            )
        result = await asyncio.to_thread(terminate, reason="budget_reached")
        return _deferred_budget_reached_step(
            generation_steps=list(
                getattr(result, "generation_steps", None) or []
            ),
            warnings=list(getattr(result, "warnings", None) or []),
        )

    async def _run_non_spoken_loop(
        self,
        *,
        input_id: Optional[str],
        unit_index: int,
        pre_non_spoken_elapsed_ms: float,
        unit_budget: int,
    ) -> None:
        used = 0
        step_durations_ms: List[float] = []
        for _ in range(unit_budget):
            if self._non_spoken_scheduling == "latency" and self._next_input_event.is_set():
                step = await self._build_deferred_budget_reached_step()
                self._unit_has_deferred_close = True
                await self._emit_step_events(
                    step,
                    input_id=input_id,
                    unit_index=unit_index,
                )
                await self._emit_budget_debug(
                    input_id=input_id,
                    used=used,
                    step_durations_ms=step_durations_ms,
                    pre_non_spoken_elapsed_ms=pre_non_spoken_elapsed_ms,
                )
                return
            step_t0 = time.perf_counter()
            step = await asyncio.to_thread(
                self.backend.fc_duplex_non_spoken_generate,
                max_tokens=1,
                decode_mode=self._decode_mode,
            )
            step_durations_ms.append((time.perf_counter() - step_t0) * 1000)
            used += 1
            await self._emit_step_events(
                step,
                input_id=input_id,
                unit_index=unit_index,
            )
            raw_flag = getattr(step, "generation_flag", "") or ""
            flag = str(getattr(raw_flag, "value", raw_flag))
            terminated = bool(getattr(step, "terminated", False))
            if terminated or flag in {
                NonSpokenStepGenerationFlag.no_action.value,
                NonSpokenStepGenerationFlag.non_spoken_slot_eos.value,
            }:
                await self._emit_budget_debug(
                    input_id=input_id,
                    used=used,
                    step_durations_ms=step_durations_ms,
                    pre_non_spoken_elapsed_ms=pre_non_spoken_elapsed_ms,
                )
                return
        step = await self._build_deferred_budget_reached_step()
        self._unit_has_deferred_close = True
        await self._emit_step_events(
            step,
            input_id=input_id,
            unit_index=unit_index,
        )
        await self._emit_budget_debug(
            input_id=input_id,
            used=used,
            step_durations_ms=step_durations_ms,
            pre_non_spoken_elapsed_ms=pre_non_spoken_elapsed_ms,
        )

    def _select_non_spoken_budget(self, spoken: Any) -> int:
        """根据当前 Unit 的 spoken 决策选择 Checkpoint Profile budget。

        参数:
            spoken: 当前 Unit 的 spoken generation 结果，至少包含 ``is_speaking``。

        返回:
            当前 Unit 可执行的 non-spoken 模型解码次数。

        异常:
            RuntimeError: Session 尚未通过 Profile 或显式参数配置两类 budget。
        """

        budget = (
            self._non_spoken_budget_while_speaking
            if bool(getattr(spoken, "is_speaking", False))
            else self._non_spoken_budget_while_listening
        )
        if budget is None:
            raise RuntimeError("fc_duplex non-spoken budgets are not configured")
        return budget

    async def _emit_budget_debug(
        self,
        *,
        input_id: Optional[str],
        used: int,
        step_durations_ms: List[float],
        pre_non_spoken_elapsed_ms: float,
    ) -> None:
        if not self._emit_debug_events:
            return
        await self._send(
            "response.debug",
            session_id=self.session_id,
            response_id=self._response_id,
            input_id=input_id,
            debug={
                "estimated_max_budget_1s": _estimate_remaining_budget_1s(
                    step_durations_ms,
                    pre_non_spoken_elapsed_ms=pre_non_spoken_elapsed_ms,
                ),
                "used": int(used),
            },
        )

    async def _emit_unit_input_events(
        self,
        prefill: Any,
        *,
        unit_index: int,
        input_id: Optional[str],
    ) -> None:
        """Expose the exact processed-Unit attribution of internal tool events."""

        public_events: List[Dict[str, Any]] = []
        for raw_event in list(getattr(prefill, "tool_events", None) or []):
            event = dict(raw_event)
            internal_call_id = str(event.get("call_id") or "")
            api_call_id = self._internal_to_api.get(internal_call_id)
            if not api_call_id:
                raise RuntimeError(
                    "processed tool event has no API id mapping: "
                    f"{internal_call_id!r}"
                )
            public_event = {
                "type": (
                    "tool_result"
                    if str(event.get("type") or "") == "tool_response"
                    else str(event.get("type") or "tool_result")
                ),
                "tool_call_id": api_call_id,
            }
            public_events.append(public_event)
        await self._send(
            "response.unit.started",
            input_id=input_id,
            unit_index=unit_index,
            tool_events=public_events,
        )

    async def _record_generation_steps(
        self,
        generation_steps: List[Any],
        *,
        unit_index: int,
        input_id: Optional[str],
    ) -> None:
        """Project View steps directly into minimal semantic API delta events."""

        for view_step in generation_steps:
            stream_id = str(getattr(view_step, "stream_id", "") or "")
            track = str(getattr(view_step, "track", "") or "")
            if not stream_id or track not in {"spoken", "non_spoken"}:
                raise RuntimeError(
                    f"invalid FC View generation step stream/track: {stream_id}/{track}"
                )
            output_model = getattr(view_step, "output", None)
            if output_model is None:
                output: Dict[str, Any] = {}
            elif hasattr(output_model, "model_dump"):
                output = dict(output_model.model_dump())
            else:
                output = dict(output_model)
            kind = str(output.get("kind") or "")
            if kind == "protocol":
                await self._flush_generation_batch()
                continue
            if kind not in {"text_pending", "text_delta"}:
                raise RuntimeError(
                    f"unsupported FC View generation step kind: {kind}"
                )
            if (
                self._generation_batch_steps
                and (
                    stream_id != self._generation_batch_stream_id
                    or track != self._generation_batch_track
                    or unit_index != self._generation_batch_unit_index
                )
            ):
                await self._flush_generation_batch()
            if not self._generation_batch_steps:
                self._generation_batch_stream_id = stream_id
                self._generation_batch_track = track
                self._generation_batch_input_id = input_id
                self._generation_batch_unit_index = unit_index
                self._generation_batch_started_at = time.perf_counter()

            self._generation_step_index += 1
            public_step: Dict[str, Any]
            if kind == "text_pending":
                self._pending_text_steps_by_stream.setdefault(
                    stream_id, []
                ).append(self._generation_step_index)
                public_step = {"kind": "pending"}
            else:
                source_step_count = int(
                    output.get("source_step_count", 0) or 0
                )
                pending_steps = self._pending_text_steps_by_stream.setdefault(
                    stream_id, []
                )
                pending_steps.append(self._generation_step_index)
                if source_step_count <= 0 or len(pending_steps) != source_step_count:
                    raise RuntimeError(
                        "FC View text delta source step count mismatch: "
                        f"stream={stream_id}, pending={len(pending_steps)}, "
                        f"source_step_count={source_step_count}"
                    )
                pending_steps.clear()
                public_step = {
                    "kind": "text",
                    "text": str(output.get("text") or ""),
                }
            self._generation_batch_steps.append(public_step)
            elapsed = (
                time.perf_counter() - self._generation_batch_started_at
                if self._generation_batch_started_at is not None
                else 0.0
            )
            if (
                len(self._generation_batch_steps) >= 16
                or elapsed >= 0.05
            ):
                await self._flush_generation_batch()

    async def _flush_generation_batch(self) -> None:
        """Send one semantic delta batch without redundant IDs or text copies."""

        if not self._generation_batch_steps:
            return
        fields: Dict[str, Any] = {
            "unit_index": self._generation_batch_unit_index,
            "steps": list(self._generation_batch_steps),
        }
        if self._generation_batch_track == "spoken":
            event_type = "response.spoken.delta"
        elif self._current_block_kind == "think":
            event_type = "response.think.delta"
        elif self._current_block_kind == "tool_call":
            event_type = "response.tool_call.delta"
            fields["tool_call_id"] = self._current_tool_call_id
        else:
            raise RuntimeError(
                "non-spoken text delta has no active semantic message"
            )
        await self._send(event_type, **fields)
        self._generation_batch_index += 1
        self._generation_batch_stream_id = None
        self._generation_batch_track = None
        self._generation_batch_input_id = None
        self._generation_batch_unit_index = None
        self._generation_batch_started_at = None
        self._generation_batch_steps = []

    async def _emit_generation_warnings(
        self,
        warnings: List[Any],
        *,
        unit_index: int,
        input_id: Optional[str],
    ) -> None:
        """Emit non-fatal View boundary warnings without exposing token IDs."""

        for warning in warnings:
            fields = (
                warning.model_dump()
                if hasattr(warning, "model_dump")
                else dict(warning)
            )
            fields.pop("stream_id", None)
            await self._send(
                "response.warning",
                unit_index=unit_index,
                **fields,
            )

    async def _emit_spoken(
        self,
        spoken: Any,
        *,
        input_id: Optional[str],
        unit_index: int,
    ) -> None:
        is_listen = bool(getattr(spoken, "is_listen", False))
        is_speaking = bool(getattr(spoken, "is_speaking", False))
        text = str(getattr(spoken, "spoken_text_delta", "") or "")
        generation_steps = list(
            getattr(spoken, "generation_steps", None) or []
        )
        await self._record_generation_steps(
            generation_steps,
            unit_index=unit_index,
            input_id=input_id,
        )
        await self._emit_generation_warnings(
            list(getattr(spoken, "warnings", None) or []),
            unit_index=unit_index,
            input_id=input_id,
        )
        await self._flush_generation_batch()
        waveform = getattr(spoken, "audio_waveform", None)
        logger.info(
            "fc_spoken input_id=%s listen=%s speaking=%s text=%r turn_eos=%s",
            input_id,
            is_listen,
            is_speaking,
            _short(text),
            bool(getattr(spoken, "spoken_turn_eos", False)),
        )

        if is_speaking and waveform is not None:
            await self._send(
                "response.spoken.delta",
                unit_index=unit_index,
                audio=_audio_waveform_to_float32_base64(waveform),
                sample_rate=int(
                    getattr(spoken, "audio_sample_rate", None) or 24000
                ),
            )
        elif is_speaking and not any(
            getattr(getattr(step, "output", None), "kind", None)
            in {"text_pending", "text_delta"}
            for step in generation_steps
        ):
            await self._send(
                "response.spoken.delta",
                unit_index=unit_index,
                steps=[],
            )
        spoken_ids = list(getattr(spoken, "spoken_token_ids", None) or [])
        fc_capability = _resolve_fc_duplex_capability(self.backend)
        protocol_ids = getattr(fc_capability, "ids", None) or {}
        if is_listen:
            reason = "listen"
        elif bool(getattr(spoken, "spoken_turn_eos", False)):
            reason = "turn_eos"
        elif protocol_ids.get("tts_pad") in spoken_ids:
            reason = "tts_pad"
        elif protocol_ids.get("spoken_slot_eos") in spoken_ids:
            reason = "slot_eos"
        else:
            reason = "slot_end"
        end_fields: Dict[str, Any] = {
            "unit_index": unit_index,
            "reason": reason,
        }
        if reason == "turn_eos":
            end_fields["full_text"] = str(
                getattr(spoken, "spoken_full_text", "") or ""
            )
        await self._send("response.spoken.end", **end_fields)

    async def _emit_step_events(
        self,
        step: Any,
        *,
        input_id: Optional[str],
        unit_index: int,
    ) -> None:
        """Convert one FcNonSpokenGenerateResult into API events.

        This intentionally mirrors audio_duplex_board.session._emit_step_events:
        token/text opens an unknown block immediately, every generated step is
        streamed, matching end tokens close completed spans, and budget markers
        terminate the current transport text stream without emitting business done.
        """

        token_ids = list(getattr(step, "token_ids", None) or [])
        step_text = str(getattr(step, "text_delta", "") or "")
        close_reason = getattr(step, "close_reason", None)
        span_started = getattr(step, "span_started", None)
        if self._current_block_id is None and span_started:
            await self._begin_block(
                str(span_started),
                unit_index=unit_index,
            )
        await self._record_generation_steps(
            list(getattr(step, "generation_steps", None) or []),
            unit_index=unit_index,
            input_id=input_id,
        )
        generation_warnings = list(getattr(step, "warnings", None) or [])
        await self._emit_generation_warnings(
            generation_warnings,
            unit_index=unit_index,
            input_id=input_id,
        )

        logger.info(
            "fc_non_spoken_step input_id=%s close=%s text=%r spans=%s",
            input_id,
            close_reason,
            _short(step_text),
            [getattr(span, "type", None) for span in list(getattr(step, "closed_spans", None) or [])],
        )

        suppress_fallback_text = any(
            getattr(warning, "code", None)
            == "incomplete_bpe_at_stream_end"
            for warning in generation_warnings
        )
        for span in list(getattr(step, "closed_spans", None) or []):
            await self._emit_close_for_span(
                span,
                unit_index=unit_index,
                suppress_fallback_text=suppress_fallback_text,
            )
        if close_reason:
            await self._emit_non_spoken_end(
                reason=str(close_reason),
                unit_index=unit_index,
            )

    async def _emit_non_spoken_end(
        self,
        *,
        reason: str,
        unit_index: int,
    ) -> None:
        """发送当前 Unit 唯一的 non-spoken slot 结束事件。

        参数:
            reason: 当前已实现的 ``eos``、``no_action`` 或
                ``budget_reached``。
            unit_index: 终止信号所属 Unit。

        返回:
            无返回值。

        异常:
            RuntimeError: reason 尚未实现，或同一 Unit 重复结束。
        """

        supported_reasons = {"eos", "no_action", "budget_reached"}
        if reason not in supported_reasons:
            raise RuntimeError(f"unsupported non-spoken end reason: {reason}")
        if self._unit_non_spoken_end is not None:
            raise RuntimeError(
                "duplicate response.non_spoken.end: "
                f"unit={unit_index}, previous={self._unit_non_spoken_end}, "
                f"next={reason}"
            )
        self._unit_non_spoken_end = reason
        await self._send(
            "response.non_spoken.end",
            unit_index=unit_index,
            reason=reason,
        )

    async def _begin_block(self, kind: str, *, unit_index: int) -> None:
        self._block_seq += 1
        self._current_block_id = f"nsb_{self._block_seq:06d}"
        self._current_block_kind = kind
        self._current_tool_call_id = None
        self._current_block_streamed = False
        self._block_started_sent = False
        if kind == "think":
            self._block_started_sent = True
            await self._send(
                "response.think.begin",
                unit_index=unit_index,
            )
        elif kind == "tool_call":
            self._current_tool_call_id = self._api_id_for_internal(None)
            self._block_started_sent = True
            await self._send(
                "response.tool_call.begin",
                tool_call_id=self._current_tool_call_id,
                unit_index=unit_index,
            )
        else:
            raise RuntimeError(f"unsupported semantic message kind: {kind}")

    async def _abort_current_block(self, *, unit_index: int) -> None:
        if self._current_block_kind == "tool_call" and self._current_tool_call_id:
            await self._send(
                "response.tool_call.done",
                tool_call_id=self._current_tool_call_id,
                unit_index=unit_index,
                error="aborted",
            )
        self._current_block_id = None
        self._current_block_kind = None
        self._current_tool_call_id = None
        self._current_block_streamed = False
        self._block_started_sent = False

    async def _emit_close_for_span(
        self,
        span: Any,
        *,
        unit_index: int,
        suppress_fallback_text: bool = False,
    ) -> None:
        """Close the current non-spoken block and emit final API events.

        Mirrors audio_duplex_board.session._emit_close_for_span, replacing board
        events with formal API events and leaving tool execution to the client.
        """

        span_type = getattr(span, "type", None)
        if span_type == "think":
            if self._current_block_kind != "think":
                raise RuntimeError("think end without active think")
            text = str(getattr(span, "text", "") or "")
            fields: Dict[str, Any] = {"unit_index": unit_index}
            if not suppress_fallback_text:
                fields["full_text"] = text
            await self._send("response.think.end", **fields)
            self._clear_current_block()
            return
        if span_type != "tool_call":
            return

        if self._current_block_kind != "tool_call":
            raise RuntimeError("tool-call end without active tool-call")

        internal_id = getattr(span, "tool_call_id", None)
        if self._current_tool_call_id:
            api_id = self._current_tool_call_id
            if internal_id:
                self._internal_to_api[str(internal_id)] = api_id
                self._api_to_internal[api_id] = str(internal_id)
        else:
            api_id = self._api_id_for_internal(str(internal_id) if internal_id else None)
        wire = getattr(span, "wire", None) or ""
        if not self._current_tool_call_id:
            self._current_tool_call_id = api_id
        done_fields: Dict[str, Any] = {
            "tool_call_id": api_id,
            "unit_index": unit_index,
        }
        if suppress_fallback_text:
            done_fields["error"] = "incomplete_bpe_at_stream_end"
            await self._send("response.tool_call.done", **done_fields)
            self._clear_current_block()
            return
        raw = _tool_call_raw(span)
        done_fields["full_text"] = wire
        if raw.get("error"):
            done_fields["error"] = raw["error"]
        else:
            arguments = raw.get("arguments")
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                pass
            done_fields["call"] = {
                "name": raw.get("name"),
                "arguments": arguments,
            }
        await self._send("response.tool_call.done", **done_fields)
        self._clear_current_block()

    def _clear_current_block(self) -> None:
        self._current_block_id = None
        self._current_block_kind = None
        self._current_tool_call_id = None
        self._current_block_streamed = False
        self._block_started_sent = False

    def _api_id_for_internal(self, internal_id: Optional[str]) -> str:
        if internal_id and internal_id in self._internal_to_api:
            return self._internal_to_api[internal_id]
        self._tool_seq += 1
        api_id = f"tc_{self._tool_seq:06d}"
        if internal_id:
            self._internal_to_api[internal_id] = api_id
            self._api_to_internal[api_id] = internal_id
        return api_id


def fc_duplex_enabled(params: Dict[str, Any]) -> bool:
    config = _first_dict(params.get("config"), params.get("duplex"), params.get("fc_duplex"))
    value = _coalesce(params.get("fc_duplex"), params.get("runtime"), config.get("runtime"), config.get("enabled"))
    if isinstance(value, str):
        return value in {"fc", "fc_duplex", "true", "1", "yes"}
    return bool(value)


def _non_spoken_close_reason_to_sp_token(reason: str) -> Optional[str]:
    if reason == "eos":
        return "non_spoken_eos"
    if reason == "no_action":
        return "no_action"
    if reason == "budget_reached":
        return "non_spoken_budget_reached"
    if reason == "hold":
        return "non_spoken_hold"
    if reason == "abort":
        return "non_spoken_abort"
    return None


def _tool_call_raw(span: Any) -> Dict[str, Any]:
    error = getattr(span, "error", None)
    tool_call = getattr(span, "tool_call", None)
    if error:
        return {"error": str(error)}
    if not isinstance(tool_call, dict):
        return {"error": "missing parsed tool call"}
    name = tool_call.get("name")
    if not name:
        return {"error": "missing tool call name"}
    arguments = tool_call.get("arguments") or {}
    return {
        "type": "function_call",
        "name": name,
        "arguments": (
            arguments
            if isinstance(arguments, str)
            else json.dumps(arguments, ensure_ascii=False)
        ),
    }


def _parse_arguments(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _contents_to_text(contents: Any) -> str:
    if isinstance(contents, str):
        return contents
    if not isinstance(contents, list):
        return json.dumps(contents, ensure_ascii=False)
    parts: List[str] = []
    for item in contents:
        if isinstance(item, dict):
            if item.get("kind") == "text" or "text" in item:
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        else:
            parts.append(str(item))
    return "".join(parts)


def _resolve_fc_duplex_capability(backend: Any) -> Optional[Any]:
    """Reach the model's initialized ``FcDuplexCapability`` instance from ``backend``.

    ``backend`` here is a ``core.processors.pytorch_backend.PyTorchBackend``
    (or an equivalent backend implementing the same fc_duplex_* facade); the
    model instance holding the initialized ``FcDuplexCapability`` sits at
    ``backend.processor.model.fc_duplex``, not directly on ``backend``. Any
    other backend implementation (e.g. a C++ backend without this Python
    model layer) simply won't have this attribute chain, in which case this
    returns None and callers degrade gracefully for exact-match kind detection.
    """

    processor = getattr(backend, "processor", None)
    model = getattr(processor, "model", None)
    return getattr(model, "fc_duplex", None)


def _guess_block_kind_from_token_ids(token_ids: List[int], backend: Any) -> Optional[str]:
    """Detect a new block's tentative kind by exact match against known opener ids.

    Replaces the former ``_guess_block_kind_from_tokens(token_strs)`` (mirrored
    from ``audio_duplex_board.session``), which guessed the kind by substring-
    matching ``tokenizer.convert_ids_to_tokens`` output. That reverse lookup is
    structurally unreliable for byte-level BPE: ordinary content tokens decode
    to byte-surrogate representations that are not valid UTF-8 (100% garbled,
    regardless of whether a character was split across multiple tokens), and
    protocol special tokens can resolve to stale pre-training names or fail to
    resolve entirely for ids added by embedding resize. ``token_ids`` (the raw
    sampled ids) has always been reliable; this uses it correctly by comparing
    against the exact opener ids the model's own ``FcDuplexCapability``
    already resolved once via ``minicpm_o5_sdk.O5SpecialTokenRegistry``
    (``fc_duplex.ids["think_start"/"tool_call_start"]``), so no new SDK
    dependency is needed at this layer.

    Returns "think" or "tool_call" when this step's sampled ids contain the
    corresponding opener token id, otherwise None (caller renders as
    `unknown` until kind becomes clear from the closed span). See
    docs/fc-duplex/o45-fc-merge-audit-2026-07-10.md for the full comparison
    against the o45-fc board MVP this behavior is ported from.
    """

    fc_duplex = _resolve_fc_duplex_capability(backend)
    opener_ids = getattr(fc_duplex, "ids", None) or {}
    think_start_id = opener_ids.get("think_start")
    tool_call_start_id = opener_ids.get("tool_call_start")
    for tid in token_ids:
        if tid == think_start_id:
            return "think"
        if tid == tool_call_start_id:
            return "tool_call"
    return None


def _kind_for_event(kind: Optional[str]) -> str:
    if kind in ("think", "tool_call"):
        return kind
    return "unknown"


def _estimate_remaining_budget_1s(
    step_durations_ms: List[float],
    *,
    pre_non_spoken_elapsed_ms: float,
) -> Optional[int]:
    if not step_durations_ms:
        return None
    sorted_durations = sorted(duration for duration in step_durations_ms if duration > 0)
    if not sorted_durations:
        return None
    index = min(len(sorted_durations) - 1, int(0.95 * (len(sorted_durations) - 1)))
    p95_ms = max(sorted_durations[index], 1.0)
    available_ms = max(0.0, 1000.0 - pre_non_spoken_elapsed_ms)
    return int(available_ms // p95_ms)


def _short(value: str, limit: int = 300) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def _audio_waveform_to_float32_base64(audio_waveform: Any) -> str:
    array = np.asarray(audio_waveform, dtype=np.float32).reshape(-1)
    return base64.b64encode(array.tobytes()).decode("utf-8")


def _model_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def _first_dict(*values: Any) -> Dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _coalesce(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _optional_positive_int(value: Any, *, field_name: str) -> Optional[int]:
    """解析一个可选正整数配置。

    参数:
        value: 来自 Session config 或进程环境的原始值。
        field_name: 报错时使用的配置字段名。

    返回:
        None 或解析后的正整数。

    异常:
        RuntimeError: 值不是正整数。
    """

    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise RuntimeError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise RuntimeError(f"{field_name} must be a positive integer")
    return parsed


def _resolve_profile_bound_budget(
    *,
    requested: Optional[int],
    registered: Optional[int],
    field_name: str,
) -> Optional[int]:
    """合并 Session 参数和 launcher 注入的 Profile budget。

    参数:
        requested: Session 显式请求的 budget。
        registered: launcher 通过环境变量注入的 Profile budget。
        field_name: 报错时使用的字段名。

    返回:
        唯一有效 budget；两侧都缺失时返回 None。

    异常:
        RuntimeError: Session 参数与已注册 Profile 不一致。
    """

    if registered is None:
        return requested
    if requested is None:
        return registered
    if requested != registered:
        raise RuntimeError(
            f"{field_name}={requested} conflicts with Checkpoint Profile value "
            f"{registered}"
        )
    return registered


def _optional_scheduling(value: Any, *, field_name: str) -> Optional[str]:
    """解析可选 non-spoken 调度模式。

    参数:
        value: Session 或 Profile 注入的原始模式。
        field_name: 报错时使用的字段名。

    返回:
        None、``quality`` 或 ``latency``。

    异常:
        RuntimeError: 值不属于已支持模式。
    """

    if value is None or value == "":
        return None
    scheduling = str(value).lower()
    if scheduling not in {"quality", "latency"}:
        raise RuntimeError(f"{field_name} must be 'latency' or 'quality'")
    return scheduling


def _resolve_profile_bound_choice(
    *,
    requested: Optional[str],
    registered: Optional[str],
    field_name: str,
) -> Optional[str]:
    """合并 Session 选择和 Profile 注册选择。

    参数:
        requested: Session 显式值。
        registered: launcher 注入的 Profile 值。
        field_name: 冲突报错字段名。

    返回:
        唯一有效值；两侧都缺失时返回 None。

    异常:
        RuntimeError: Session 与 Profile 选择冲突。
    """

    if registered is None:
        return requested
    if requested is None:
        return registered
    if requested != registered:
        raise RuntimeError(
            f"{field_name}={requested} conflicts with Checkpoint Profile value "
            f"{registered}"
        )
    return registered


def _extract_frame_base64_list(payload: Dict[str, Any]) -> Optional[list[str]]:
    direct = payload.get("frame_base64_list") or payload.get("video_frames")
    if direct:
        return list(direct)
    frames = payload.get("frames")
    if not frames:
        return None
    out: List[str] = []
    for frame in frames:
        if isinstance(frame, str):
            out.append(frame)
        elif isinstance(frame, dict):
            data = frame.get("data") or frame.get("base64")
            if data:
                out.append(data)
    return out or None


def _extract_audio_base64(payload: Dict[str, Any]) -> Optional[str]:
    for key in ("audio_base64", "audio_data"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    audio = payload.get("audio")
    if isinstance(audio, str) and audio:
        return audio
    if isinstance(audio, dict):
        value = audio.get("data") or audio.get("base64") or audio.get("audio_base64")
        if isinstance(value, str) and value:
            return value
    return None


def _safe_decode_audio_base64(audio_base64: Optional[str]) -> Optional[np.ndarray]:
    """Decode base64 float32 PCM audio for diagnostics; never raises.

    The audio dump is opt-in diagnostics (FC_DUPLEX_AUDIO_DUMP_DIR), so a
    decode failure must not take down the main prefill/generate flow.
    """

    if not audio_base64:
        return None
    try:
        return decode_audio_base64(audio_base64)
    except Exception as exc:  # noqa: BLE001
        logger.warning("audio dump: failed to decode audio_base64 for dump: %s", exc)
        return None
