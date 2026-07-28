import base64
from types import SimpleNamespace

import numpy as np
import pytest

from py_backend.server import BackendProtocolSession, BackendServerState
from py_backend.usage import SessionUsage, TokenUsage


def test_duplex_chunk_usage_keeps_modalities_separate():
    usage = TokenUsage.from_duplex_chunk(
        {
            "input_text_tokens": 5,
            "input_audio_tokens": 50,
            "input_vision_tokens": 128,
        },
        SimpleNamespace(n_tokens=7, n_tts_tokens=25),
        has_video=True,
    ).to_dict()

    assert usage == {
        "total_tokens": 215,
        "input_tokens": 183,
        "output_tokens": 32,
        "input_token_details": {
            "text_tokens": 5,
            "audio_tokens": 50,
            "image_tokens": 0,
            "video_tokens": 128,
            "cached_tokens": 0,
        },
        "output_token_details": {
            "text_tokens": 7,
            "audio_tokens": 25,
        },
    }


def test_session_usage_accumulates_from_session_start():
    session = SessionUsage()
    session.add(TokenUsage(input_text_tokens=10, input_audio_tokens=20))

    current = TokenUsage(
        input_text_tokens=1,
        input_audio_tokens=50,
        output_text_tokens=4,
        output_audio_tokens=25,
    )
    cumulative = session.add(current)

    assert current.to_dict()["total_tokens"] == 80
    assert cumulative["total_tokens"] == 110
    assert cumulative["input_token_details"]["text_tokens"] == 11
    assert cumulative["input_token_details"]["audio_tokens"] == 70
    assert cumulative["output_token_details"]["text_tokens"] == 4
    assert cumulative["output_token_details"]["audio_tokens"] == 25
    assert cumulative["input_token_details"]["cached_tokens"] == 0


def test_invalid_backend_counts_fall_back_to_zero():
    usage = TokenUsage.from_counts({
        "input_text_tokens": None,
        "input_audio_tokens": -3,
        "output_text_tokens": "bad",
    })

    assert usage.to_dict()["total_tokens"] == 0


@pytest.mark.asyncio
async def test_duplex_output_events_include_chunk_and_cumulative_usage():
    class FakeWebSocket:
        def __init__(self):
            self.events = []

        async def send_json(self, event):
            self.events.append(event)

    class FakeBackend:
        def duplex_prefill(self, **_kwargs):
            return {
                "input_text_tokens": 1,
                "input_audio_tokens": 50,
                "input_vision_tokens": 0,
                "n_vision_slices": 0,
            }

        def duplex_generate(self, **_kwargs):
            return SimpleNamespace(
                is_listen=False,
                text="你好",
                audio_data="audio",
                end_of_turn=False,
                n_tokens=3,
                n_tts_tokens=25,
            )

        def duplex_finalize(self):
            return None

        def metrics(self):
            return {}

    backend = FakeBackend()
    websocket = FakeWebSocket()
    session = BackendProtocolSession(
        session_id="test",
        mode="full_duplex",
        backend=backend,
        ws=websocket,
        state=BackendServerState(backend),
    )
    session.initialized = True
    audio = base64.b64encode(np.zeros(16000, dtype=np.float32).tobytes()).decode()

    await session.push({"type": "input.append", "input": {"audio": audio}})
    await session._drain_finalize()

    assert len(websocket.events) == 2
    text_event, audio_event = websocket.events
    assert text_event["kind"] == "text"
    assert audio_event["kind"] == "audio"
    assert text_event["chunk_index"] == audio_event["chunk_index"] == 0
    assert text_event["chunk_usage"] == audio_event["chunk_usage"]
    assert text_event["usage"] == audio_event["usage"]
    assert text_event["chunk_usage"]["total_tokens"] == 79
    assert text_event["usage"]["total_tokens"] == 79
