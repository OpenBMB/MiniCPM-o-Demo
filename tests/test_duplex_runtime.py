import asyncio

import numpy as np

from core.runtime.backends import WorkerDuplexBackendAdapter
from core.runtime.duplex import DuplexInputFrame, DuplexSessionRuntime


class _FakeResult:
    is_listen = True
    text = ""
    audio_data = None
    current_time = 1
    cost_all_ms = 1.0
    cost_llm_ms = None
    cost_tts_prep_ms = None
    cost_tts_ms = None
    cost_token2wav_ms = None
    n_tokens = None
    n_tts_tokens = None

    def model_dump(self):
        return {
            "is_listen": self.is_listen,
            "text": self.text,
            "audio_data": self.audio_data,
            "current_time": self.current_time,
            "cost_all_ms": self.cost_all_ms,
        }


class _FakeWorker:
    gpu_id = 0
    kv_cache_length = 0

    def __init__(self):
        self.calls = []

    def duplex_prepare(self, **_kwargs):
        self.calls.append("prepare")
        return "prompt"

    def duplex_prefill(self, **_kwargs):
        self.calls.append("prefill")
        self.kv_cache_length += 10
        return {"n_vision_images": 1}

    def duplex_generate(self, **_kwargs):
        self.calls.append("generate")
        return _FakeResult()

    def duplex_finalize(self):
        self.calls.append("finalize")

    def duplex_stop(self):
        self.calls.append("stop")

    def duplex_cleanup(self):
        self.calls.append("cleanup")


def test_deferred_finalize_is_runtime_managed():
    async def _run():
        worker = _FakeWorker()
        runtime = DuplexSessionRuntime(WorkerDuplexBackendAdapter(worker))
        emitted = []

        await runtime.process_frame(
            frame=DuplexInputFrame(
                audio_waveform=np.zeros(16000, dtype=np.float32),
                frame_list=[],
                max_slice_nums=1,
                force_listen=False,
            ),
            emit=lambda frame: emitted.append(frame) or asyncio.sleep(0),
            use_deferred_finalize=True,
        )
        await runtime.wait_for_finalize()

        assert [c for c in worker.calls if c in ("prefill", "generate", "finalize")] == [
            "prefill",
            "generate",
            "finalize",
        ]
        assert emitted[0].result_dict["vision_slices"] == 1

    asyncio.run(_run())


def test_close_drains_finalize_before_cleanup():
    async def _run():
        worker = _FakeWorker()
        runtime = DuplexSessionRuntime(WorkerDuplexBackendAdapter(worker))

        await runtime.process_frame(
            frame=DuplexInputFrame(
                audio_waveform=np.zeros(16000, dtype=np.float32),
                frame_list=[],
                max_slice_nums=1,
                force_listen=False,
            ),
            emit=lambda _frame: asyncio.sleep(0),
            use_deferred_finalize=True,
        )
        await runtime.close()

        assert worker.calls[-3:] == ["finalize", "stop", "cleanup"]

    asyncio.run(_run())

