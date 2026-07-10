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
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np
import soundfile as sf

from core.schemas.fc_duplex import FcToolResponse, NonSpokenStepGenerationFlag
from py_backend.media import decode_audio_base64, decode_frame_base64_list


SendEvent = Callable[[str], Awaitable[None]]
logger = logging.getLogger(__name__)


def _deferred_budget_reached_step() -> SimpleNamespace:
    return SimpleNamespace(
        token_ids=[],
        terminated=True,
        close_reason="budget_reached",
        generation_flag="continue_non_spoken_generation",
        closed_spans=[],
        text="",
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


class FcDuplexSessionRuntime:
    """Small per-session scheduler for FC duplex protocol input/output."""

    def __init__(
        self,
        *,
        session_id: str,
        backend: Any,
        send: Callable[[str, Any], Awaitable[None]],
    ) -> None:
        self.session_id = session_id
        self.backend = backend
        self._send = send
        self._response_id: Optional[str] = None
        self._tools: List[Dict[str, Any]] = []
        self._pending_tool_responses: List[FcToolResponse] = []
        self._streaming_tool_results: Dict[str, List[Any]] = {}
        self._api_to_internal: Dict[str, str] = {}
        self._internal_to_api: Dict[str, str] = {}
        self._tool_seq = 0
        self._max_spoken_tokens = 24
        self._non_spoken_budget_per_unit = 12
        self._non_spoken_scheduling = "latency"
        self._decode_mode = "greedy"
        self._sample_rate = 16000
        self._input_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
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
        self._non_spoken_budget_per_unit = int(
            config.get("non_spoken_budget_per_unit", params.get("non_spoken_budget_per_unit", 12)) or 12
        )
        requested_scheduling = str(
            config.get("non_spoken_scheduling", params.get("non_spoken_scheduling", "latency")) or "latency"
        ).lower()
        if requested_scheduling not in {"latency", "quality"}:
            raise RuntimeError("fc_duplex non_spoken_scheduling must be 'latency' or 'quality'")
        self._non_spoken_scheduling = requested_scheduling
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
            FcToolResponse(call_id=internal_id, content=_contents_to_text(payload.get("contents")))
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
            except Exception:
                logger.exception("FC runtime failed to process audio payload")
                raise
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

        if self._audio_dump_session_dir is not None:
            self._maybe_dump_user_wav(
                unit_index=unit_index,
                samples=_safe_decode_audio_base64(audio_base64),
                sample_rate=sample_rate,
            )

        await asyncio.to_thread(
            self.backend.fc_duplex_prefill,
            audio_data=audio_base64,
            frame_list=frame_list,
            tool_responses=tool_responses or None,
            sample_rate=sample_rate,
        )

        spoken = await asyncio.to_thread(
            self.backend.fc_duplex_spoken_generate,
            max_tokens=self._max_spoken_tokens,
            decode_mode=self._decode_mode,
        )
        await self._emit_spoken(spoken, input_id=input_id)
        if self._audio_dump_session_dir is not None:
            self._maybe_dump_speak_wav(unit_index=unit_index, spoken=spoken)
        spoken_done_elapsed_ms = (time.perf_counter() - unit_t0) * 1000

        await self._run_non_spoken_loop(
            input_id=input_id,
            pre_non_spoken_elapsed_ms=spoken_done_elapsed_ms,
        )

        await asyncio.to_thread(self.backend.fc_duplex_finalize)

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
        trace_dir = os.environ.get("FC_DUPLEX_TRACE_DIR", "/user/weihongliang/fc_trace_logs")
        session = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.session_id or "session")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(trace_dir, f"fc_trace_{session}_{stamp}.json")
        try:
            info = await asyncio.to_thread(dump, path=path, session_id=self.session_id, reason=reason)
            logger.info("fc_model_trace_dumped session=%s path=%s info=%s", self.session_id, path, info)
        except Exception:
            logger.exception("failed to dump fc model trace: session=%s path=%s", self.session_id, path)

    async def _run_non_spoken_loop(self, *, input_id: Optional[str], pre_non_spoken_elapsed_ms: float) -> None:
        used = 0
        step_durations_ms: List[float] = []
        for _ in range(max(0, self._non_spoken_budget_per_unit)):
            if self._non_spoken_scheduling == "latency" and self._next_input_event.is_set():
                step = _deferred_budget_reached_step()
                await self._emit_step_events(step, input_id=input_id)
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
            await self._emit_step_events(step, input_id=input_id)
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
        step = _deferred_budget_reached_step()
        await self._emit_step_events(step, input_id=input_id)
        await self._emit_budget_debug(
            input_id=input_id,
            used=used,
            step_durations_ms=step_durations_ms,
            pre_non_spoken_elapsed_ms=pre_non_spoken_elapsed_ms,
        )

    async def _emit_budget_debug(
        self,
        *,
        input_id: Optional[str],
        used: int,
        step_durations_ms: List[float],
        pre_non_spoken_elapsed_ms: float,
    ) -> None:
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

    async def _emit_spoken(self, spoken: Any, *, input_id: Optional[str]) -> None:
        is_listen = bool(getattr(spoken, "is_listen", False))
        is_speaking = bool(getattr(spoken, "is_speaking", False))
        text = str(getattr(spoken, "spoken_text", "") or "")
        waveform = getattr(spoken, "audio_waveform", None)
        metadata = _model_to_dict(spoken)
        metadata.pop("audio_waveform", None)
        logger.info(
            "fc_spoken input_id=%s listen=%s speaking=%s text=%r turn_eos=%s",
            input_id,
            is_listen,
            is_speaking,
            _short(text),
            bool(getattr(spoken, "spoken_turn_eos", False)),
        )

        if is_listen:
            await self._send_sp_token("listen", input_id=input_id)
            await self._send(
                "response.output.delta",
                kind="listen",
                session_id=self.session_id,
                response_id=self._response_id,
                input_id=input_id,
                metrics=metadata,
            )
            return
        if is_speaking:
            await self._send_sp_token("speak", input_id=input_id)
        if text:
            await self._send(
                "response.output.delta",
                kind="text",
                session_id=self.session_id,
                response_id=self._response_id,
                input_id=input_id,
                text=text,
                metrics=metadata,
            )
        if is_speaking and waveform is not None:
            await self._send(
                "response.output.delta",
                kind="audio",
                session_id=self.session_id,
                response_id=self._response_id,
                input_id=input_id,
                audio=_audio_waveform_to_float32_base64(waveform),
                sample_rate=int(getattr(spoken, "audio_sample_rate", None) or 24000),
                metrics=metadata,
            )
        if bool(getattr(spoken, "spoken_turn_eos", False)):
            await self._send_sp_token("spoken_turn_eos", input_id=input_id)

    async def _emit_step_events(self, step: Any, *, input_id: Optional[str]) -> None:
        """Convert one FcNonSpokenGenerateResult into API events.

        This intentionally mirrors audio_duplex_board.session._emit_step_events:
        token/text opens an unknown block immediately, every generated step is
        streamed, and only closed_spans close the current block. Budget markers
        never close an open block.
        """

        token_ids = list(getattr(step, "token_ids", None) or [])
        step_text = str(getattr(step, "text", "") or "")
        close_reason = getattr(step, "close_reason", None)

        logger.info(
            "fc_non_spoken_step input_id=%s close=%s text=%r spans=%s",
            input_id,
            close_reason,
            _short(step_text),
            [getattr(span, "type", None) for span in list(getattr(step, "closed_spans", None) or [])],
        )

        # Same exception as MVP: a lone no_action terminator is not content and
        # should not open an unknown block.
        is_no_action_marker = (
            bool(getattr(step, "terminated", False)) and str(close_reason or "") == "no_action"
        )
        should_open = (
            self._current_block_id is None
            and (token_ids or step_text)
            and not is_no_action_marker
        )
        if should_open:
            tentative_kind = _guess_block_kind_from_token_ids(token_ids, self.backend)
            await self._begin_block(tentative_kind, input_id=input_id)

        if (token_ids or step_text) and self._current_block_id is not None:
            await self._send_block_delta(
                token_ids=token_ids,
                step_text=step_text,
                input_id=input_id,
            )

        if close_reason:
            token = _non_spoken_close_reason_to_sp_token(str(close_reason))
            if token:
                await self._send_sp_token(token, input_id=input_id)
            if str(close_reason) == "abort":
                await self._abort_current_block(input_id=input_id)

        for span in list(getattr(step, "closed_spans", None) or []):
            await self._emit_close_for_span(span, input_id=input_id)

    async def _send_block_delta(
        self,
        *,
        token_ids: List[int],
        step_text: str,
        input_id: Optional[str],
    ) -> None:
        if self._current_block_kind is None:
            kind = _guess_block_kind_from_token_ids(token_ids, self.backend)
            if kind:
                await self._upgrade_block_kind(kind, input_id=input_id)

        if self._current_block_kind == "think":
            if step_text:
                self._current_block_streamed = True
                await self._send(
                    "response.think.delta",
                    session_id=self.session_id,
                    response_id=self._response_id,
                    input_id=input_id,
                    delta=step_text,
                    token_observations=_token_observations(token_ids, self.backend),
                )
            return

        if self._current_block_kind == "tool_call":
            tool_call_id = self._current_tool_call_id or self._current_block_id
            if step_text:
                self._current_block_streamed = True
                await self._send(
                    "response.tool_call.args.delta",
                    session_id=self.session_id,
                    response_id=self._response_id,
                    input_id=input_id,
                    tool_call_id=tool_call_id,
                    delta=step_text,
                    token_observations=_token_observations(token_ids, self.backend),
                )
            return

        if step_text or token_ids:
            await self._send(
                "debug.fc_non_spoken.delta",
                session_id=self.session_id,
                response_id=self._response_id,
                input_id=input_id,
                text=step_text,
                token_ids=token_ids,
                token_observations=_token_observations(token_ids, self.backend),
                block_id=self._current_block_id,
                block_kind=_kind_for_event(self._current_block_kind),
            )

    async def _begin_block(self, kind: Optional[str], *, input_id: Optional[str]) -> None:
        self._block_seq += 1
        self._current_block_id = f"nsb_{self._block_seq:06d}"
        self._current_block_kind = kind
        self._current_tool_call_id = None
        self._current_block_streamed = False
        self._block_started_sent = False
        if kind == "think":
            self._block_started_sent = True
            await self._send("response.think.begin", session_id=self.session_id, response_id=self._response_id, input_id=input_id)
        elif kind == "tool_call":
            self._current_tool_call_id = self._api_id_for_internal(None)
            self._block_started_sent = True
            await self._send(
                "response.tool_call.args.begin",
                session_id=self.session_id,
                response_id=self._response_id,
                input_id=input_id,
                tool_call_id=self._current_tool_call_id,
            )

    async def _upgrade_block_kind(self, kind: str, *, input_id: Optional[str]) -> None:
        if kind == self._current_block_kind:
            return
        self._current_block_kind = kind
        if kind == "think":
            if not self._block_started_sent:
                self._block_started_sent = True
                await self._send("response.think.begin", session_id=self.session_id, response_id=self._response_id, input_id=input_id)
        elif kind == "tool_call":
            self._current_tool_call_id = self._api_id_for_internal(None)
            if not self._block_started_sent:
                self._block_started_sent = True
                await self._send(
                    "response.tool_call.args.begin",
                    session_id=self.session_id,
                    response_id=self._response_id,
                    input_id=input_id,
                    tool_call_id=self._current_tool_call_id,
                )

    async def _abort_current_block(self, *, input_id: Optional[str]) -> None:
        if self._current_block_kind == "tool_call" and self._current_tool_call_id:
            await self._send(
                "response.tool_call.abort",
                session_id=self.session_id,
                response_id=self._response_id,
                input_id=input_id,
                tool_call_id=self._current_tool_call_id,
            )
        self._current_block_id = None
        self._current_block_kind = None
        self._current_tool_call_id = None
        self._current_block_streamed = False
        self._block_started_sent = False

    async def _send_sp_token(self, token: str, *, input_id: Optional[str]) -> None:
        await self._send(
            "response.output.sp_tokens",
            session_id=self.session_id,
            response_id=self._response_id,
            input_id=input_id,
            token=token,
        )

    async def _emit_close_for_span(self, span: Any, *, input_id: Optional[str]) -> None:
        """Close the current non-spoken block and emit final API events.

        Mirrors audio_duplex_board.session._emit_close_for_span, replacing board
        events with formal API events and leaving tool execution to the client.
        """

        span_type = getattr(span, "type", None)
        if span_type == "think":
            if self._current_block_id is None:
                await self._begin_block("think", input_id=input_id)
            elif self._current_block_kind != "think":
                await self._upgrade_block_kind("think", input_id=input_id)
            text = str(getattr(span, "text", "") or "")
            if text and not self._current_block_streamed:
                await self._send(
                    "response.think.delta",
                    session_id=self.session_id,
                    response_id=self._response_id,
                    input_id=input_id,
                    delta=text,
                )
            await self._send("response.think.end", session_id=self.session_id, response_id=self._response_id, input_id=input_id)
            self._clear_current_block()
            return
        if span_type != "tool_call":
            return

        if self._current_block_id is None:
            await self._begin_block("tool_call", input_id=input_id)
        elif self._current_block_kind != "tool_call":
            await self._upgrade_block_kind("tool_call", input_id=input_id)

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
        if not self._block_started_sent:
            self._block_started_sent = True
            await self._send(
                "response.tool_call.args.begin",
                session_id=self.session_id,
                response_id=self._response_id,
                input_id=input_id,
                tool_call_id=api_id,
            )
        if wire and not self._current_block_streamed:
            await self._send(
                "response.tool_call.args.delta",
                session_id=self.session_id,
                response_id=self._response_id,
                input_id=input_id,
                tool_call_id=api_id,
                delta=wire,
            )
        await self._send(
            "response.tool_call.args.end",
            session_id=self.session_id,
            response_id=self._response_id,
            input_id=input_id,
            tool_call_id=api_id,
        )
        raw = _tool_call_raw(span)
        await self._send(
            "response.tool_call.args.raw",
            session_id=self.session_id,
            response_id=self._response_id,
            input_id=input_id,
            tool_call_id=api_id,
            raw=raw,
        )
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
    return {
        "type": "function_call",
        "name": name,
        "arguments": json.dumps(tool_call.get("arguments") or {}, ensure_ascii=False),
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
    returns None and callers degrade gracefully (no exact-match kind
    detection / no token_observations, never a crash).
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


def _token_observations(token_ids: List[int], backend: Any) -> Optional[List[Dict[str, Any]]]:
    """Per-token ``{id, text}`` observations for protocol §8 ``token_observations``.

    Previously sourced ``text`` from ``token_strs`` (a
    ``tokenizer.convert_ids_to_tokens`` reverse lookup that is structurally
    unreliable for byte-level BPE — it exposes the internal byte-to-surrogate
    vocab mapping table, not real decoded text, so a token that's half of a
    multi-byte character comes back as unrelated-looking garbage rather than
    any recognizable placeholder). That field no longer exists; this instead
    decodes each token individually through the SDK's own decoder
    (``FcDuplexCapability.decode_text``, i.e. ``minicpm_o5_sdk``'s
    ``decode_ordinary``), the same method already used to produce the
    reliable ``text``/``step_text`` field. Empirically verified: for a token
    that's a complete character/word on its own, this returns the correct
    readable text; for a token that's only half of a multi-byte character
    split across a token boundary, the SDK decoder's standard UTF-8
    decode-with-replacement semantics return a single U+FFFD ``'�'`` --
    an honest, universally-recognized "encoding boundary" marker, not
    misleading garbage. Special tokens (``is_special``) use the protocol's
    ``id2name`` display name instead of attempting a text decode. See
    docs/fc-duplex/o45-fc-merge-audit-2026-07-10.md for the verification
    transcript.

    Returns None (omit the field) when the model's FcDuplexCapability isn't
    reachable through ``backend`` (e.g. a non-Python backend) -- per protocol
    §8 this field is optional ("backend MAY"), so omitting it entirely is
    protocol-compliant when there is no reliable source.
    """

    if not token_ids:
        return None
    fc_duplex = _resolve_fc_duplex_capability(backend)
    if fc_duplex is None:
        return None
    observations = []
    for tid in token_ids:
        tid_int = int(tid)
        try:
            if fc_duplex.is_special(tid_int):
                text = fc_duplex.id2name.get(tid_int)
            else:
                text = fc_duplex.decode_text([tid_int])
        except Exception:  # noqa: BLE001 - 观测是可选调试字段，解码失败不能挂主流程
            text = None
        observations.append({"id": tid_int, "text": text})
    return observations


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
