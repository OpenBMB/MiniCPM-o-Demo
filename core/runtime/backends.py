"""Backend adapter interfaces for session runtimes.

SessionRuntime owns session semantics and lifecycle.  Backend adapters own how a
particular inference implementation (PyTorch, C++, SGLang, etc.) executes those
semantics.
"""

from __future__ import annotations

import gc
from typing import Any, Dict, Iterator, Optional, Protocol

import numpy as np
import torch

from core.schemas.streaming import StreamingChunk


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
    """Adapter over the current PyTorch duplex processor view."""

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
        duplex_view = self.worker.processor.set_duplex_mode()
        return duplex_view.prepare(
            system_prompt_text=system_prompt_text,
            ref_audio_path=ref_audio_path or self.worker.ref_audio_path,
            prompt_wav_path=prompt_wav_path,
        )

    def prefill(
        self,
        *,
        audio_waveform: Optional[np.ndarray],
        frame_list: Optional[list],
        max_slice_nums: int,
    ) -> Dict[str, Any]:
        duplex_view = self.worker.processor.set_duplex_mode()
        return duplex_view.prefill(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            max_slice_nums=max_slice_nums,
        )

    def generate(self, *, force_listen: bool) -> Any:
        duplex_view = self.worker.processor.set_duplex_mode()
        return duplex_view.generate(force_listen=force_listen)

    def finalize(self) -> None:
        duplex_view = self.worker.processor.set_duplex_mode()
        duplex_view.finalize()

    def stop(self) -> None:
        duplex_view = self.worker.processor.set_duplex_mode()
        duplex_view.stop()

    def cleanup(self) -> None:
        if self.worker.processor is None:
            return
        duplex_view = self.worker.processor.set_duplex_mode()
        duplex_view.cleanup()
        gc.collect()
        torch.cuda.empty_cache()

    def kv_cache_length(self) -> int:
        processor = getattr(self.worker, "processor", None)
        if processor is not None:
            return int(getattr(processor, "kv_cache_length", 0) or 0)
        return int(getattr(self.worker, "kv_cache_length", 0) or 0)


class ChatBackendAdapter(Protocol):
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

    def kv_cache_length(self) -> int:
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


class WorkerChatBackendAdapter:
    """Adapter over the current PyTorch chat processor view."""

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
        chat_view = self.worker.processor.set_chat_mode()
        return chat_view.prefill(
            session_id=session_id,
            msgs=msgs,
            omni_mode=omni_mode,
            max_slice_nums=max_slice_nums,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
        )

    def kv_cache_length(self) -> int:
        processor = getattr(self.worker, "processor", None)
        if processor is not None:
            return int(getattr(processor, "kv_cache_length", 0) or 0)
        return 0

    def init_tts(self, ref_audio: Optional[np.ndarray]) -> None:
        if ref_audio is not None:
            self.worker.processor.model.init_token2wav_cache(prompt_speech_16k=ref_audio)
            return

        ref_audio_path = getattr(self.worker, "ref_audio_path", None)
        if ref_audio_path:
            import librosa

            loaded_ref, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
            self.worker.processor.model.init_token2wav_cache(prompt_speech_16k=loaded_ref)

    def streaming_generate(
        self,
        *,
        session_id: str,
        generate_audio: bool,
        max_new_tokens: int,
        length_penalty: float,
    ) -> Iterator[StreamingChunk]:
        chat_view = self.worker.processor.set_chat_mode()
        yield from chat_view.streaming_generate(
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
        chat_view = self.worker.processor.set_chat_mode()
        return chat_view.generate(
            session_id=session_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            generate_audio=generate_audio,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
            tts_ref_audio=tts_ref_audio,
            tts_sampling_params=None,
            length_penalty=length_penalty,
        )

