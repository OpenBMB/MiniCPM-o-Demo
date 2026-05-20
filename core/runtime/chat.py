"""Turn-based chat session runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np

from core.runtime.backends import ChatBackendAdapter
from core.schemas.streaming import StreamingChunk


@dataclass
class ChatPrefillParams:
    session_id: str
    msgs: list
    omni_mode: bool = False
    max_slice_nums: Optional[int] = None
    use_tts_template: bool = False
    enable_thinking: bool = False


@dataclass
class ChatGenerateParams:
    session_id: str
    generate_audio: bool = False
    max_new_tokens: int = 256
    use_tts_template: bool = False
    enable_thinking: bool = False
    tts_ref_audio: Optional[np.ndarray] = None
    length_penalty: float = 1.1


class ChatSessionRuntime:
    """Runtime wrapper for the legacy /ws/chat worker flow."""

    def __init__(self, backend: ChatBackendAdapter):
        self.backend = backend

    async def prefill(self, params: ChatPrefillParams) -> int:
        await asyncio.to_thread(
            self.backend.prefill,
            session_id=params.session_id,
            msgs=params.msgs,
            omni_mode=params.omni_mode,
            max_slice_nums=params.max_slice_nums,
            use_tts_template=params.use_tts_template,
            enable_thinking=params.enable_thinking,
        )
        return self.backend.kv_cache_length()

    async def init_tts(self, ref_audio: Optional[np.ndarray]) -> None:
        await asyncio.to_thread(self.backend.init_tts, ref_audio)

    def streaming_generate(self, params: ChatGenerateParams) -> Iterator[StreamingChunk]:
        yield from self.backend.streaming_generate(
            session_id=params.session_id,
            generate_audio=params.generate_audio,
            max_new_tokens=params.max_new_tokens,
            length_penalty=params.length_penalty,
        )

    async def non_streaming_generate(self, params: ChatGenerateParams) -> Any:
        return await asyncio.to_thread(
            self.backend.non_streaming_generate,
            session_id=params.session_id,
            max_new_tokens=params.max_new_tokens,
            generate_audio=params.generate_audio,
            use_tts_template=params.use_tts_template,
            enable_thinking=params.enable_thinking,
            tts_ref_audio=params.tts_ref_audio,
            length_penalty=params.length_penalty,
        )

