import base64

import numpy as np

from core.runtime.events import RuntimeEvent
from core.runtime.worker_protocol import (
    parse_worker_control_message,
    parse_worker_input_message,
    parse_worker_prepare_message,
    runtime_event_to_worker_message,
)


def _pcm_b64(samples: int = 1600) -> str:
    return base64.b64encode(np.zeros(samples, dtype=np.float32).tobytes()).decode("utf-8")


def test_parse_worker_prepare_message():
    params, voice_refs = parse_worker_prepare_message({
        "type": "session.prepare",
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
        "type": "input.append",
        "payload": {
            "audio_base64": _pcm_b64(400),
            "force_listen": True,
            "max_slice_nums": 3,
        },
    })

    assert frame.audio_waveform.shape[0] == 400
    assert frame.force_listen is True
    assert frame.max_slice_nums == 3


def test_parse_worker_control_message():
    control = parse_worker_control_message({
        "type": "control",
        "payload": {"command": "session.close", "reason": "test"},
    })

    assert control.type == "session.close"
    assert control.payload["reason"] == "test"


def test_runtime_event_to_worker_message():
    msg = runtime_event_to_worker_message(RuntimeEvent(
        channel="output.duplex_result",
        payload={
            "result_dict": {"text": "hi"},
            "prefill_ms": 1,
            "kv_cache_len": 2,
            "wall_clock_ms": 3,
            "n_vision_images": 4,
            "vision_tokens": 256,
        },
    ))

    assert msg["type"] == "runtime.event"
    assert msg["channel"] == "output.duplex_result"
    assert msg["payload"]["result"] == {"text": "hi"}
    assert msg["payload"]["metrics"]["kv_cache_len"] == 2

