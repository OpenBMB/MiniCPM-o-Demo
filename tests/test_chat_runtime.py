import asyncio

import numpy as np

from core.runtime.chat import ChatGenerateParams, ChatPrefillParams, ChatSessionRuntime
from core.runtime.backends import WorkerChatBackendAdapter
from core.schemas.streaming import StreamingChunk


class _FakeProcessor:
    kv_cache_length = 0


class _FakeWorker:
    ref_audio_path = None

    def __init__(self):
        self.calls = []
        self.processor = _FakeProcessor()

    def chat_prefill(self, **kwargs):
        self.calls.append(("prefill", kwargs))
        self.processor.kv_cache_length = 123
        return "prompt"

    def chat_streaming_generate(self, **kwargs):
        self.calls.append(("streaming_generate", kwargs))
        yield StreamingChunk(chunk_index=0, text_delta="hi", is_final=False)

    def chat_non_streaming_generate(self, **kwargs):
        self.calls.append(("non_streaming_generate", kwargs))
        return "done"


def test_chat_runtime_prefill_returns_kv_length():
    async def _run():
        worker = _FakeWorker()
        runtime = ChatSessionRuntime(WorkerChatBackendAdapter(worker))

        kv = await runtime.prefill(ChatPrefillParams(session_id="s1", msgs=[]))

        assert kv == 123
        assert worker.calls[0][0] == "prefill"

    asyncio.run(_run())


def test_chat_runtime_streaming_and_non_streaming_generation():
    async def _run():
        worker = _FakeWorker()
        runtime = ChatSessionRuntime(WorkerChatBackendAdapter(worker))

        chunks = list(runtime.streaming_generate(ChatGenerateParams(session_id="s1", generate_audio=True)))
        result = await runtime.non_streaming_generate(ChatGenerateParams(
            session_id="s1",
            generate_audio=True,
            tts_ref_audio=np.zeros(10, dtype=np.float32),
        ))

        assert chunks[0].text_delta == "hi"
        assert result == "done"
        assert [c[0] for c in worker.calls] == ["streaming_generate", "non_streaming_generate"]

    asyncio.run(_run())

