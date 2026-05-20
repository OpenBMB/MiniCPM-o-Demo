"""Backend adapter interfaces for session runtimes.

SessionRuntime owns session semantics and lifecycle.  Backend adapters own how a
particular inference implementation (PyTorch, C++, SGLang, etc.) executes those
semantics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

import numpy as np


class DuplexBackendAdapter(Protocol):
    """Minimal backend contract required by DuplexSessionRuntime."""

    def configure(self, config: Optional[Dict[str, Any]]) -> None:
        ...

    def prepare(
        self,
        *,
        system_prompt_text: Optional[str],
        ref_audio_path: Optional[str],
        prompt_wav_path: Optional[str],
    ) -> str:
        ...

    def prefill(
        self,
        *,
        audio_waveform: Optional[np.ndarray],
        frame_list: Optional[list],
        max_slice_nums: int,
    ) -> Dict[str, Any]:
        ...

    def generate(self, *, force_listen: bool) -> Any:
        ...

    def finalize(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def cleanup(self) -> None:
        ...

    def kv_cache_length(self) -> int:
        ...


class WorkerDuplexBackendAdapter:
    """Adapter over the current MiniCPMOWorker duplex methods.

    This keeps existing worker methods as the compatibility surface while the
    runtime layer becomes independent of worker.py implementation details.
    """

    def __init__(self, worker: Any):
        self.worker = worker

    def configure(self, config: Optional[Dict[str, Any]]) -> None:
        if not config:
            return

        # Keep existing PyTorch behavior: config is written to the duplex view.
        processor = getattr(self.worker, "processor", None)
        if processor is not None:
            from core.schemas.duplex import DuplexConfig

            duplex_view = processor.set_duplex_mode()
            duplex_view.config = DuplexConfig(**config)

        # Backends such as C++ may expose their own config ingestion hook.
        if hasattr(self.worker, "set_duplex_config"):
            self.worker.set_duplex_config(config)

    def prepare(
        self,
        *,
        system_prompt_text: Optional[str],
        ref_audio_path: Optional[str],
        prompt_wav_path: Optional[str],
    ) -> str:
        return self.worker.duplex_prepare(
            system_prompt_text=system_prompt_text,
            ref_audio_path=ref_audio_path,
            prompt_wav_path=prompt_wav_path,
        )

    def prefill(
        self,
        *,
        audio_waveform: Optional[np.ndarray],
        frame_list: Optional[list],
        max_slice_nums: int,
    ) -> Dict[str, Any]:
        return self.worker.duplex_prefill(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            max_slice_nums=max_slice_nums,
        )

    def generate(self, *, force_listen: bool) -> Any:
        return self.worker.duplex_generate(force_listen=force_listen)

    def finalize(self) -> None:
        self.worker.duplex_finalize()

    def stop(self) -> None:
        self.worker.duplex_stop()

    def cleanup(self) -> None:
        self.worker.duplex_cleanup()

    def kv_cache_length(self) -> int:
        processor = getattr(self.worker, "processor", None)
        if processor is not None:
            return int(getattr(processor, "kv_cache_length", 0) or 0)
        return int(getattr(self.worker, "kv_cache_length", 0) or 0)

