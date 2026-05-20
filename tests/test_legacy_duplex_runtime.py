import base64
import io

import numpy as np
from PIL import Image

from core.runtime.legacy_duplex import (
    parse_audio_chunk_message,
    parse_control_message,
    parse_prepare_message,
)


def _pcm_b64(samples: int = 1600) -> str:
    return base64.b64encode(np.zeros(samples, dtype=np.float32).tobytes()).decode("utf-8")


def _jpg_b64() -> str:
    img = Image.new("RGB", (3, 4), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_parse_prepare_message_resolves_runtime_params():
    msg = {
        "type": "prepare",
        "system_prompt": "hello",
        "ref_audio_base64": _pcm_b64(800),
        "config": {"max_slice_nums": 4, "temperature": 0.7},
        "deferred_finalize": False,
    }

    parsed = parse_prepare_message(msg)
    try:
        assert parsed.system_prompt == "hello"
        assert parsed.use_deferred_finalize is False
        assert parsed.session_max_slice_nums == 4
        assert parsed.params.config == msg["config"]
        assert parsed.params.ref_audio_path
        assert parsed.params.prompt_wav_path == parsed.params.ref_audio_path
    finally:
        parsed.cleanup()


def test_parse_audio_chunk_message_decodes_input_frame():
    msg = {
        "type": "audio_chunk",
        "audio_base64": _pcm_b64(400),
        "frame_base64_list": [_jpg_b64()],
        "max_slice_nums": 2,
        "force_listen": True,
    }

    parsed = parse_audio_chunk_message(msg, session_max_slice_nums=1, chunk_start=123.0)

    assert parsed.frame.audio_waveform.shape[0] == 400
    assert parsed.frame.max_slice_nums == 2
    assert parsed.frame.force_listen is True
    assert parsed.frame.chunk_start == 123.0
    assert parsed.frame.frame_list is not None
    assert parsed.frame.frame_list[0].size == (3, 4)
    assert parsed.first_frame_bytes is not None


def test_parse_audio_chunk_message_requires_audio():
    try:
        parse_audio_chunk_message({}, session_max_slice_nums=1)
    except ValueError as exc:
        assert "audio_base64" in str(exc)
    else:
        raise AssertionError("expected missing audio to fail")


def test_parse_control_message_maps_legacy_controls():
    pause = parse_control_message({"type": "pause", "timeout": 12})
    assert pause is not None
    assert pause.type == "session.pause"
    assert pause.payload["timeout"] == 12

    resume = parse_control_message({"type": "resume"})
    assert resume is not None
    assert resume.type == "session.resume"

    stop = parse_control_message({"type": "stop"})
    assert stop is not None
    assert stop.type == "session.close"

    interrupt = parse_control_message({"type": "interrupt"})
    assert interrupt is not None
    assert interrupt.type == "legacy.interrupt"

    assert parse_control_message({"type": "audio_chunk"}) is None

