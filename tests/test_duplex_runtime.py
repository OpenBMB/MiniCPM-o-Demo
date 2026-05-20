import asyncio

import numpy as np

from core.runtime.backends import WorkerDuplexBackendAdapter
from core.runtime.events import RuntimeControl
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


class _FakeDuplexView:
    def __init__(self, worker):
        self.worker = worker

    def prepare(self, **_kwargs):
        self.worker.calls.append("prepare")
        return "prompt"

    def prefill(self, **_kwargs):
        self.worker.calls.append("prefill")
        self.worker.processor.kv_cache_length += 10
        return {"n_vision_images": 1}

    def generate(self, **_kwargs):
        self.worker.calls.append("generate")
        return _FakeResult()

    def finalize(self):
        self.worker.calls.append("finalize")

    def stop(self):
        self.worker.calls.append("stop")

    def cleanup(self):
        self.worker.calls.append("cleanup")


class _FakeProcessor:
    def __init__(self, worker):
        self.worker = worker
        self.kv_cache_length = 0

    def set_duplex_mode(self):
        return _FakeDuplexView(self.worker)


class _FakeWorker:
    gpu_id = 0
    ref_audio_path = None

    def __init__(self):
        self.calls = []
        self.processor = _FakeProcessor(self)

    def set_duplex_config(self, _config):
        return

    def duplex_prepare(self, **kwargs):
        return self.processor.set_duplex_mode().prepare(**kwargs)

    def duplex_prefill(self, **kwargs):
        return self.processor.set_duplex_mode().prefill(**kwargs)

    def duplex_generate(self, **kwargs):
        return self.processor.set_duplex_mode().generate(**kwargs)

    def duplex_finalize(self):
        self.processor.set_duplex_mode().finalize()

    def duplex_stop(self):
        self.processor.set_duplex_mode().stop()

    def duplex_cleanup(self):
        self.processor.set_duplex_mode().cleanup()

    def metrics(self):
        return {"backend": "fake", "kv_cache_length": self.processor.kv_cache_length}


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
            emit=lambda event: emitted.append(event) or asyncio.sleep(0),
            use_deferred_finalize=True,
        )
        await runtime.wait_for_finalize()

        assert [c for c in worker.calls if c in ("prefill", "generate", "finalize")] == [
            "prefill",
            "generate",
            "finalize",
        ]
        event = emitted[0]
        assert event.channel == "output.duplex_result"
        assert "vision_slices" not in event.payload["result_dict"]
        assert event.payload["metrics"]["vision_slices"] == 1
        assert event.payload["frame"].metrics["vision_slices"] == 1
        assert event.payload["metrics"]["kv_cache_length"] == 10

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
            emit=lambda _event: asyncio.sleep(0),
            use_deferred_finalize=True,
        )
        await runtime.close()

        assert worker.calls[-3:] == ["finalize", "stop", "cleanup"]

    asyncio.run(_run())


def test_pause_blocks_processing_until_resume():
    async def _run():
        worker = _FakeWorker()
        runtime = DuplexSessionRuntime(WorkerDuplexBackendAdapter(worker))

        paused = await runtime.control(RuntimeControl(type="session.pause", payload={"timeout": 60}))
        assert paused.payload["state"] == "paused"

        try:
            await runtime.process_frame(
                frame=DuplexInputFrame(
                    audio_waveform=np.zeros(16000, dtype=np.float32),
                    frame_list=[],
                ),
                emit=lambda _event: asyncio.sleep(0),
            )
        except RuntimeError as exc:
            assert "paused" in str(exc)
        else:
            raise AssertionError("expected paused runtime to reject frames")

        resumed = await runtime.control(RuntimeControl(type="session.resume"))
        assert resumed.payload["state"] == "active"

        await runtime.process_frame(
            frame=DuplexInputFrame(
                audio_waveform=np.zeros(16000, dtype=np.float32),
                frame_list=[],
            ),
            emit=lambda _event: asyncio.sleep(0),
        )

    asyncio.run(_run())


def test_runtime_machine_owns_frame_processing_loop():
    async def _run():
        worker = _FakeWorker()
        runtime = DuplexSessionRuntime(WorkerDuplexBackendAdapter(worker))
        emitted = []
        emitted_event = asyncio.Event()

        async def _emit(event):
            emitted.append(event)
            emitted_event.set()

        await runtime.start(_emit)
        await runtime.push_frame(DuplexInputFrame(
            audio_waveform=np.zeros(16000, dtype=np.float32),
            frame_list=[],
        ))
        await asyncio.wait_for(emitted_event.wait(), timeout=1.0)
        await runtime.wait_for_finalize()
        await runtime.close()

        assert emitted[0].channel == "output.duplex_result"
        assert [c for c in worker.calls if c in ("prefill", "generate", "finalize")] == [
            "prefill",
            "generate",
            "finalize",
        ]

    asyncio.run(_run())


def test_runtime_machine_ignores_frames_while_paused():
    async def _run():
        worker = _FakeWorker()
        runtime = DuplexSessionRuntime(WorkerDuplexBackendAdapter(worker))
        emitted = []
        emitted_event = asyncio.Event()

        async def _emit(event):
            emitted.append(event)
            emitted_event.set()

        await runtime.start(_emit)
        paused = await runtime.push_control(RuntimeControl(type="session.pause", payload={}))
        assert paused.payload["state"] == "paused"

        emitted_event.clear()
        await runtime.push_frame(DuplexInputFrame(
            audio_waveform=np.zeros(16000, dtype=np.float32),
            frame_list=[],
        ))
        await asyncio.wait_for(emitted_event.wait(), timeout=1.0)
        await runtime.close()

        assert emitted[-1].payload["state"] == "input_ignored"
        assert "prefill" not in worker.calls

    asyncio.run(_run())

