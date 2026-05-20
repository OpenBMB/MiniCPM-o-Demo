import base64

import numpy as np

from core.runtime.events import RuntimeEvent
from core.runtime.worker_protocol import (
    parse_worker_control_message,
    parse_worker_input_message,
    parse_worker_prepare_message,
    runtime_event_to_worker_message,
    runtime_event_to_worker_messages,
)


def _pcm_b64(samples: int = 1600) -> str:
    return base64.b64encode(np.zeros(samples, dtype=np.float32).tobytes()).decode("utf-8")


def test_parse_worker_prepare_message():
    params, voice_refs = parse_worker_prepare_message({
        "type": "duplex.session.prepare",
        "payload": {
            "system_prompt": "hello",
            "config": {"max_slice_nums": 2},
            "voice": {"ref_audio_base64": _pcm_b64()},
        },
    })
    try:
        assert params.system_prompt_text == "hello"
        assert params.config == {"max_slice_nums": 2}
        assert params.ref_audio_path
        assert params.prompt_wav_path == params.ref_audio_path
    finally:
        voice_refs.cleanup()


def test_parse_worker_input_message():
    frame = parse_worker_input_message({
        "type": "duplex.input.audio.append",
        "payload": {
            "audio_base64": _pcm_b64(400),
            "force_listen": True,
            "max_slice_nums": 3,
        },
    })

    assert frame.audio_waveform.shape[0] == 400
    assert frame.force_listen is True
    assert frame.max_slice_nums == 3


def test_parse_worker_input_message_uses_default_for_null_max_slice_nums():
    frame = parse_worker_input_message({
        "type": "duplex.input.audio.append",
        "payload": {
            "audio_base64": _pcm_b64(400),
            "max_slice_nums": None,
        },
    }, default_max_slice_nums=5)

    assert frame.max_slice_nums == 5


def test_parse_worker_control_message():
    control = parse_worker_control_message({
        "type": "duplex.control.close",
        "payload": {"reason": "test"},
    })

    assert control.type == "session.close"
    assert control.payload["reason"] == "test"


def test_runtime_event_to_worker_message():
    msg = runtime_event_to_worker_message(RuntimeEvent(
        channel="output.duplex_result",
        payload={
            "result_dict": {"text": "hi"},
            "prefill_ms": 1,
            "metrics": {"kv_cache_length": 2, "prefill_ms": 1},
            "wall_clock_ms": 3,
            "n_vision_images": 4,
            "vision_tokens": 256,
        },
    ))

    assert msg["type"] == "duplex.output.result"
    assert msg["payload"]["result"] == {"text": "hi"}
    assert msg["payload"]["metrics"]["kv_cache_length"] == 2


def test_runtime_event_to_worker_messages_splits_duplex_output():
    messages = runtime_event_to_worker_messages(RuntimeEvent(
        channel="output.duplex_result",
        payload={
            "result_dict": {
                "is_listen": False,
                "text": "hi",
                "audio_data": "pcm",
                "end_of_turn": True,
            },
            "prefill_ms": 1,
            "metrics": {"kv_cache_length": 2, "prefill_ms": 1},
        },
    ))

    assert [msg["type"] for msg in messages] == [
        "duplex.metrics.frame",
        "duplex.output.text.delta",
        "duplex.output.audio.delta",
        "duplex.output.turn.done",
    ]
    assert messages[0]["payload"]["kv_cache_length"] == 2
    assert "kv_cache_length" not in messages[1]["payload"]
    assert messages[2]["payload"]["audio_base64"] == "pcm"

