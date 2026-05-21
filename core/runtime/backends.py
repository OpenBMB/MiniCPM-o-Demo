"""Runtime-facing backend contracts and views.

SessionRuntime owns session semantics and lifecycle. Backend implementations
own how a concrete inference engine (PyTorch, C++, SGLang, etc.) executes those
semantics. The view classes below expose only the narrow chat/duplex surfaces
that each runtime needs from a full backend worker object.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Protocol

import numpy as np

from core.schemas.streaming import StreamingChunk
from core.runtime.metrics import BackendMetrics


def _coerce_backend_metrics(data: Any, *, backend: Optional[str] = None) -> Dict[str, Any]:
    if isinstance(data, BackendMetrics):
        metrics = data.to_dict()
    elif isinstance(data, dict):
        metrics = BackendMetrics.from_mapping(data).to_dict()
    else:
        metrics = BackendMetrics().to_dict()
    if backend and not metrics.get("backend"):
        metrics["backend"] = backend
    return metrics


class DuplexRuntimeBackend(Protocol):
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

    def metrics(self) -> Dict[str, Any]:
        ...


class DuplexBackendView:
    """Runtime duplex view over a full backend implementation."""

    def __init__(self, worker: Any):
        self.worker = worker
        self._config: Optional[Dict[str, Any]] = None

    def configure(self, config: Optional[Dict[str, Any]]) -> None:
        self._config = config
        self.worker.set_duplex_config(config)

    def prepare(
        self,
        *,
        system_prompt_text: Optional[str],
        ref_audio_path: Optional[str],
        prompt_wav_path: Optional[str],
    ) -> str:
        cfg = self._config or {}
        return self.worker.duplex_prepare(
            system_prompt_text=system_prompt_text,
            ref_audio_path=ref_audio_path,
            prompt_wav_path=prompt_wav_path,
            length_penalty=cfg.get("length_penalty", 1.1),
            sampling=cfg,
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

    def metrics(self) -> Dict[str, Any]:
        return _coerce_backend_metrics(self.worker.metrics())


class ChatRuntimeBackend(Protocol):
    """Minimal backend contract required by ChatSessionRuntime."""

    def prefill(
        self,
        *,
        session_id: str,
        msgs: list,
        omni_mode: bool,
        max_slice_nums: Optional[int],
        use_tts_template: bool,
        enable_thinking: bool,
    ) -> str:
        ...

    def metrics(self) -> Dict[str, Any]:
        ...

    def init_tts(self, ref_audio: Optional[np.ndarray]) -> None:
        ...

    def streaming_generate(
        self,
        *,
        session_id: str,
        generate_audio: bool,
        max_new_tokens: int,
        length_penalty: float,
    ) -> Iterator[StreamingChunk]:
        ...

    def non_streaming_generate(
        self,
        *,
        session_id: str,
        max_new_tokens: int,
        generate_audio: bool,
        use_tts_template: bool,
        enable_thinking: bool,
        tts_ref_audio: Optional[np.ndarray],
        length_penalty: float,
    ) -> Any:
        ...


class ChatBackendView:
    """Runtime chat view over a full backend implementation."""

    def __init__(self, worker: Any):
        self.worker = worker

    def prefill(
        self,
        *,
        session_id: str,
        msgs: list,
        omni_mode: bool,
        max_slice_nums: Optional[int],
        use_tts_template: bool,
        enable_thinking: bool,
    ) -> str:
        return self.worker.chat_prefill(
            session_id=session_id,
            msgs=msgs,
            omni_mode=omni_mode,
            max_slice_nums=max_slice_nums,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
        )

    def metrics(self) -> Dict[str, Any]:
        return _coerce_backend_metrics(self.worker.metrics())

    def init_tts(self, ref_audio: Optional[np.ndarray]) -> None:
        self.worker.chat_init_tts(ref_audio)

    def streaming_generate(
        self,
        *,
        session_id: str,
        generate_audio: bool,
        max_new_tokens: int,
        length_penalty: float,
    ) -> Iterator[StreamingChunk]:
        yield from self.worker.chat_streaming_generate(
            session_id=session_id,
            generate_audio=generate_audio,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        )

    def non_streaming_generate(
        self,
        *,
        session_id: str,
        max_new_tokens: int,
        generate_audio: bool,
        use_tts_template: bool,
        enable_thinking: bool,
        tts_ref_audio: Optional[np.ndarray],
        length_penalty: float,
    ) -> Any:
        return self.worker.chat_non_streaming_generate(
            session_id=session_id,
            max_new_tokens=max_new_tokens,
            generate_audio=generate_audio,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
            tts_ref_audio=tts_ref_audio,
            length_penalty=length_penalty,
        )

