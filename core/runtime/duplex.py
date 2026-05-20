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
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

import numpy as np

from core.runtime.backends import DuplexBackendAdapter

logger = logging.getLogger(__name__)


@dataclass
class DuplexFrameResult:
    """A completed duplex frame ready to be emitted by the transport layer."""

    result: Any
    result_dict: Dict[str, Any]
    prefill_ms: float
    prefill_result: Dict[str, Any]
    kv_cache_len: int
    wall_clock_ms: float
    n_vision_images: int
    vision_tokens: int


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


EmitDuplexFrame = Callable[[DuplexFrameResult], Awaitable[None]]


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

    async def process_frame(
        self,
        *,
        frame: DuplexInputFrame,
        emit: EmitDuplexFrame,
        use_deferred_finalize: bool = True,
    ) -> DuplexFrameResult:
        """Run one duplex frame and emit the transport-facing result.

        Finalize is always managed by the runtime:
        - deferred mode: emit first, then finalize in the background
        - sync mode: finalize before emit
        """

        if self._closed:
            raise RuntimeError("duplex runtime is already closed")

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
            kv_len = self.backend.kv_cache_length()
            return gen_result, prefill_ms, prefill_result, kv_len

        result, prefill_ms, prefill_result, kv_cache_len = await asyncio.to_thread(_duplex_step)
        result.server_send_ts = time.time()

        wall_clock_ms = (time.perf_counter() - chunk_t0) * 1000
        n_vision_images = (
            prefill_result.get("n_vision_images", 0)
            if isinstance(prefill_result, dict)
            else 0
        )
        vision_tokens = n_vision_images * 64

        result_dict = result.model_dump()
        result_dict["wall_clock_ms"] = round(wall_clock_ms, 1)
        result_dict["kv_cache_length"] = kv_cache_len
        result_dict["vision_slices"] = n_vision_images
        result_dict["vision_tokens"] = vision_tokens

        frame_result = DuplexFrameResult(
            result=result,
            result_dict=result_dict,
            prefill_ms=prefill_ms,
            prefill_result=prefill_result if isinstance(prefill_result, dict) else {},
            kv_cache_len=kv_cache_len,
            wall_clock_ms=wall_clock_ms,
            n_vision_images=n_vision_images,
            vision_tokens=vision_tokens,
        )

        if use_deferred_finalize:
            try:
                await emit(frame_result)
            finally:
                self._schedule_finalize()
        else:
            await self._run_finalize_sync()
            await emit(frame_result)

        return frame_result

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

