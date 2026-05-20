import asyncio
import base64

import numpy as np

from core.runtime.events import RuntimeEvent
from core.runtime.legacy_duplex import parse_audio_chunk_message
from core.runtime.sinks import CompositeSink, DuplexRecordingSink, LegacyDuplexWebSocketSink
from tests.test_duplex_runtime import _FakeResult


class _FakeWs:
    def __init__(self):
        self.messages = []

    async def send_json(self, msg):
        self.messages.append(msg)


class _CollectSink:
    def __init__(self):
        self.events = []

    async def emit(self, event):
        self.events.append(event)


def test_legacy_duplex_websocket_sink_sends_legacy_result_payload():
    async def _run():
        ws = _FakeWs()
        sink = LegacyDuplexWebSocketSink(ws)

        await sink.emit(RuntimeEvent(
            channel="output.duplex_result",
            payload={"result_dict": {"is_listen": False, "text": "hi"}},
        ))

        assert ws.messages == [{"type": "result", "is_listen": False, "text": "hi"}]

    asyncio.run(_run())


def test_composite_sink_fans_out_events_in_order():
    async def _run():
        a = _CollectSink()
        b = _CollectSink()
        sink = CompositeSink([a, b])
        event = RuntimeEvent(channel="metrics", payload={"x": 1})

        await sink.emit(event)

        assert a.events == [event]
        assert b.events == [event]

    asyncio.run(_run())


class _FakeRecorder:
    turn_index = 0

    def __init__(self):
        self.calls = []

    def save_user_audio(self, chunk_index, audio):
        self.calls.append(("save_user_audio", chunk_index, len(audio)))
        return f"user_audio/{chunk_index:03d}.wav"

    def save_user_frame(self, chunk_index, frame_bytes):
        self.calls.append(("save_user_frame", chunk_index, len(frame_bytes)))
        return f"user_frames/{chunk_index:03d}.jpg"

    def save_ai_audio(self, turn_index, chunk_index, audio):
        self.calls.append(("save_ai_audio", turn_index, chunk_index, len(audio)))
        return f"ai_audio/{turn_index:03d}_{chunk_index:03d}.wav"

    def record_chunk(self, **kwargs):
        self.calls.append(("record_chunk", kwargs))


def test_duplex_recording_sink_persists_input_and_output():
    async def _run():
        recorder = _FakeRecorder()
        sink = DuplexRecordingSink(recorder)
        audio = np.zeros(8, dtype=np.float32)
        legacy_input = parse_audio_chunk_message(
            {
                "audio_base64": base64.b64encode(audio.tobytes()).decode("utf-8"),
            },
            session_max_slice_nums=1,
            chunk_start=1.0,
        )

        sink.capture_input(legacy_input, receive_ts_ms=123.4)

        result = _FakeResult()
        result.is_listen = False
        result.audio_data = base64.b64encode(np.ones(6, dtype=np.float32).tobytes()).decode("utf-8")
        event = RuntimeEvent(
            channel="output.duplex_result",
            payload={
                "frame": type("Frame", (), {
                    "result": result,
                    "result_dict": result.model_dump(),
                    "prefill_ms": 12.3,
                })(),
            },
        )
        await sink.emit(event)

        assert recorder.calls[0] == ("save_user_audio", 0, 8)
        assert recorder.calls[1] == ("save_ai_audio", 0, 0, 6)
        assert recorder.calls[2][0] == "record_chunk"
        recorded = recorder.calls[2][1]
        assert recorded["index"] == 0
        assert recorded["receive_ts_ms"] == 123.4
        assert recorded["user_audio_rel"] == "user_audio/000.wav"
        assert recorded["ai_audio_rel"] == "ai_audio/000_000.wav"
        assert sink.chunk_index == 1

    asyncio.run(_run())

