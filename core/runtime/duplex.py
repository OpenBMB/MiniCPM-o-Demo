"""Duplex session runtime.

The current worker protocol still speaks the legacy duplex WebSocket messages,
but this module starts separating session lifecycle from transport handling.

Key boundary:
- worker.py owns WebSocket parsing, recording, and response transport
- DuplexSessionRuntime owns per-session inference lifecycle
- Backend adapter owns backend-specific prefill/generate/finalize mechanics

This keeps PyTorch's deferred finalize as an internal runtime detail instead of
an operation that worker.py must manually pair with every generate call.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

import numpy as np

from core.runtime.backends import DuplexBackendAdapter
from core.runtime.events import RuntimeControl, RuntimeEvent
from core.runtime.metrics import BackendMetrics

logger = logging.getLogger(__name__)


@dataclass
class DuplexFrameResult:
    """A completed duplex frame ready to be emitted by the transport layer."""

    result: Any
    result_dict: Dict[str, Any]
    prefill_ms: float
    prefill_result: Dict[str, Any]
    metrics: Dict[str, Any]
    wall_clock_ms: float
    n_vision_images: int
    vision_tokens: int

    @property
    def kv_cache_len(self) -> int:
        return int(self.metrics.get("kv_cache_length", 0) or 0)

    def to_runtime_event(self) -> RuntimeEvent:
        return RuntimeEvent(
            channel="output.duplex_result",
            payload={
                "frame": self,
                "result": self.result,
                "result_dict": self.result_dict,
                "prefill_ms": self.prefill_ms,
                "prefill_result": self.prefill_result,
                "metrics": self.metrics,
                "wall_clock_ms": self.wall_clock_ms,
                "n_vision_images": self.n_vision_images,
                "vision_tokens": self.vision_tokens,
            },
        )


@dataclass
class DuplexPrepareParams:
    """Backend-facing prepare parameters for one duplex session."""

    system_prompt_text: Optional[str]
    ref_audio_path: Optional[str]
    prompt_wav_path: Optional[str]
    config: Optional[Dict[str, Any]] = None


@dataclass
class DuplexInputFrame:
    """Backend-facing input frame for one duplex unit."""

    audio_waveform: np.ndarray
    frame_list: Optional[list]
    max_slice_nums: int = 1
    force_listen: bool = False
    chunk_start: Optional[float] = None


EmitRuntimeEvent = Callable[[RuntimeEvent], Awaitable[None]]


class DuplexSessionRuntime:
    """Runtime lifecycle wrapper for a single duplex session."""

    def __init__(
        self,
        backend: DuplexBackendAdapter,
        *,
        finalize_timeout_s: float = 5.0,
    ) -> None:
        self.backend = backend
        self.finalize_timeout_s = finalize_timeout_s
        self._finalize_done = asyncio.Event()
        self._finalize_done.set()
        self._finalize_task: Optional[asyncio.Task[None]] = None
        self._closed = False
        self._paused = False
        self._emit: Optional[EmitRuntimeEvent] = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._run_task: Optional[asyncio.Task[None]] = None

    async def configure(self, config: Optional[Dict[str, Any]]) -> None:
        await asyncio.to_thread(self.backend.configure, config)

    async def prepare(
        self,
        params: DuplexPrepareParams,
    ) -> str:
        await self.wait_for_finalize()
        if params.config:
            await self.configure(params.config)
        return await asyncio.to_thread(
            self.backend.prepare,
            system_prompt_text=params.system_prompt_text,
            ref_audio_path=params.ref_audio_path,
            prompt_wav_path=params.prompt_wav_path,
        )

    async def start(self, emit: EmitRuntimeEvent) -> None:
        """Start the runtime machine.

        After this point callers should push frames/controls into the runtime
        instead of directly driving frame processing.
        """

        if self._closed:
            raise RuntimeError("duplex runtime is already closed")
        self._emit = emit
        if self._run_task is None or self._run_task.done():
            self._run_task = asyncio.create_task(self._run_loop())

    async def push_frame(self, frame: DuplexInputFrame) -> None:
        """Queue an input frame for the runtime-owned processing loop."""

        if self._run_task is None:
            raise RuntimeError("duplex runtime has not been started")
        await self._queue.put(("frame", frame, None))

    async def push_control(self, command: RuntimeControl) -> RuntimeEvent:
        """Queue a control command and wait for its state event."""

        if self._run_task is None:
            raise RuntimeError("duplex runtime has not been started")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[RuntimeEvent] = loop.create_future()
        await self._queue.put(("control", command, future))
        return await future

    async def _run_loop(self) -> None:
        emit = self._emit
        if emit is None:
            raise RuntimeError("duplex runtime emit handler is not configured")

        while not self._closed:
            kind, payload, future = await self._queue.get()

            if kind == "frame":
                if self._paused:
                    await emit(RuntimeEvent(
                        channel="session",
                        payload={"state": "input_ignored", "reason": "paused"},
                    ))
                    continue
                try:
                    await self.process_frame(frame=payload, emit=emit)
                except Exception as exc:
                    logger.exception("Duplex runtime frame processing failed")
                    await emit(RuntimeEvent(
                        channel="session",
                        payload={"state": "error", "error": str(exc)},
                    ))
                continue

            if kind == "control":
                try:
                    event = await self.control(payload)
                except Exception as exc:
                    if future is not None and not future.done():
                        future.set_exception(exc)
                    raise
                await emit(event)
                if future is not None and not future.done():
                    future.set_result(event)
                if payload.type == "session.close":
                    break

    async def process_frame(
        self,
        *,
        frame: DuplexInputFrame,
        emit: EmitRuntimeEvent,
        use_deferred_finalize: bool = True,
    ) -> DuplexFrameResult:
        """Run one duplex frame and emit the transport-facing result.

        Finalize is always managed by the runtime:
        - deferred mode: emit first, then finalize in the background
        - sync mode: finalize before emit
        """

        if self._closed:
            raise RuntimeError("duplex runtime is already closed")
        if self._paused:
            raise RuntimeError("duplex runtime is paused")

        await self.wait_for_finalize()
        chunk_t0 = frame.chunk_start if frame.chunk_start is not None else time.perf_counter()

        def _duplex_step():
            t0 = time.perf_counter()
            prefill_result = self.backend.prefill(
                audio_waveform=frame.audio_waveform,
                frame_list=frame.frame_list,
                max_slice_nums=frame.max_slice_nums,
            )
            t_prefill = time.perf_counter()
            gen_result = self.backend.generate(force_listen=frame.force_listen)

            prefill_ms = (t_prefill - t0) * 1000
            backend_metrics = self.backend.metrics()
            return gen_result, prefill_ms, prefill_result, backend_metrics

        result, prefill_ms, prefill_result, backend_metrics = await asyncio.to_thread(_duplex_step)
        result.server_send_ts = time.time()

        wall_clock_ms = (time.perf_counter() - chunk_t0) * 1000
        n_vision_images = (
            prefill_result.get("n_vision_images", 0)
            if isinstance(prefill_result, dict)
            else 0
        )
        vision_tokens = n_vision_images * 64

        metrics = BackendMetrics.from_mapping(backend_metrics).to_dict()
        metrics.update({
            "prefill_ms": round(prefill_ms, 1),
            "generate_ms": result.cost_all_ms,
            "wall_clock_ms": round(wall_clock_ms, 1),
            "cost_llm_ms": result.cost_llm_ms,
            "cost_tts_prep_ms": result.cost_tts_prep_ms,
            "cost_tts_ms": result.cost_tts_ms,
            "cost_token2wav_ms": result.cost_token2wav_ms,
            "n_tokens": result.n_tokens,
            "n_tts_tokens": result.n_tts_tokens,
            "vision_slices": n_vision_images,
            "vision_tokens": vision_tokens,
        })
        metrics = {key: value for key, value in metrics.items() if value is not None}

        result_dict = result.model_dump()

        frame_result = DuplexFrameResult(
            result=result,
            result_dict=result_dict,
            prefill_ms=prefill_ms,
            prefill_result=prefill_result if isinstance(prefill_result, dict) else {},
            metrics=metrics,
            wall_clock_ms=wall_clock_ms,
            n_vision_images=n_vision_images,
            vision_tokens=vision_tokens,
        )

        event = frame_result.to_runtime_event()

        if use_deferred_finalize:
            try:
                await emit(event)
            finally:
                self._schedule_finalize()
        else:
            await self._run_finalize_sync()
            await emit(event)

        return frame_result

    async def control(self, command: RuntimeControl) -> RuntimeEvent:
        """Apply a runtime control command and return a state event.

        Control events share transport with input events, but are not model
        observations.  They update runtime/session state.
        """

        if command.type == "session.pause":
            await self.wait_for_finalize()
            self._paused = True
            return RuntimeEvent(
                channel="session",
                payload={"state": "paused", **command.payload},
            )

        if command.type == "session.resume":
            self._paused = False
            return RuntimeEvent(
                channel="session",
                payload={"state": "active"},
            )

        if command.type == "legacy.interrupt":
            return RuntimeEvent(
                channel="session",
                payload={"state": "interrupted", "deprecated": True},
            )

        if command.type == "response.cancel":
            await self.wait_for_finalize()
            try:
                await asyncio.to_thread(self.backend.stop)
            except Exception:
                logger.exception("Duplex response cancel failed")
            return RuntimeEvent(
                channel="session",
                payload={"state": "cancelled"},
            )

        if command.type == "session.close":
            await self.close()
            return RuntimeEvent(
                channel="session",
                payload={"state": "closed", **command.payload},
            )

        raise ValueError(f"unsupported runtime control: {command.type}")

    async def wait_for_finalize(self) -> None:
        await self._finalize_done.wait()
        task = self._finalize_task
        if task is not None and task.done():
            # Surface unexpected task exceptions close to the next operation.
            task.result()
            self._finalize_task = None

    def _schedule_finalize(self) -> None:
        if self._finalize_task is not None and not self._finalize_task.done():
            raise RuntimeError("duplex finalize already in flight")

        self._finalize_done.clear()

        async def _do_finalize() -> None:
            try:
                await asyncio.to_thread(self.backend.finalize)
            except Exception:
                logger.exception("Duplex finalize failed")
                raise
            finally:
                self._finalize_done.set()

        self._finalize_task = asyncio.create_task(_do_finalize())

    async def _run_finalize_sync(self) -> None:
        self._finalize_done.clear()
        try:
            await asyncio.to_thread(self.backend.finalize)
        finally:
            self._finalize_done.set()
            self._finalize_task = None

    async def close(self) -> None:
        """Drain finalize, then stop and cleanup backend resources."""

        if self._closed:
            return

        current_task = asyncio.current_task()
        if self._run_task is not None and self._run_task is not current_task and not self._run_task.done():
            self._run_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._run_task
            self._run_task = None

        try:
            await self._drain_finalize_for_close()
        except Exception:
            logger.exception("Duplex finalize failed before runtime close; continuing cleanup")

        try:
            await asyncio.to_thread(self.backend.stop)
        except Exception:
            logger.exception("Duplex stop failed during runtime close")

        try:
            await self._drain_finalize_for_close()
        except Exception:
            logger.exception("Duplex finalize failed after stop; continuing cleanup")

        await asyncio.to_thread(self.backend.cleanup)
        self._closed = True

    async def _drain_finalize_for_close(self) -> None:
        task = self._finalize_task
        if task is None:
            return
        if task.done():
            task.result()
            self._finalize_task = None
            return
        try:
            await asyncio.wait_for(task, timeout=self.finalize_timeout_s)
        finally:
            self._finalize_task = None

