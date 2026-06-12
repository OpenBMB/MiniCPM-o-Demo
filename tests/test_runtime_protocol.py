import base64

import numpy as np

from runtime.protocol import DEFAULT_WORKER_CAPABILITIES, capability_for_request
from py_backend.chat_util import parse_worker_chat_request_message


def test_default_worker_capabilities_cover_existing_modes():
    assert "chat" in DEFAULT_WORKER_CAPABILITIES
    assert "streaming" in DEFAULT_WORKER_CAPABILITIES
    assert "half_duplex_audio" in DEFAULT_WORKER_CAPABILITIES
    assert "audio_duplex" in DEFAULT_WORKER_CAPABILITIES
    assert "omni_duplex" in DEFAULT_WORKER_CAPABILITIES


def test_capability_for_request_maps_legacy_aliases():
    assert capability_for_request("chat_ws") == "streaming"
    assert capability_for_request("duplex") == "omni_duplex"
    assert capability_for_request("audio_duplex") == "audio_duplex"


def test_parse_worker_chat_request_message():
    ref_audio = np.array([0.1, -0.2], dtype=np.float32)
    msg = {
        "type": "chat.request",
        "payload": {
            "messages": [{"role": "user", "content": "hi"}],
            "streaming": False,
            "generation": {"max_new_tokens": 32, "length_penalty": 0.9},
            "image": {"max_slice_nums": 2},
            "tts": {
                "enabled": True,
                "ref_audio_data": base64.b64encode(ref_audio.tobytes()).decode("utf-8"),
            },
            "omni_mode": True,
            "enable_thinking": True,
        },
    }

    parsed = parse_worker_chat_request_message(msg)

    assert parsed.messages == [{"role": "user", "content": "hi"}]
    assert parsed.streaming is False
    assert parsed.max_new_tokens == 32
    assert parsed.length_penalty == 0.9
    assert parsed.max_slice_nums == 2
    assert parsed.generate_audio is True
    assert parsed.use_tts_template is True
    assert parsed.omni_mode is True
    assert parsed.enable_thinking is True
    np.testing.assert_allclose(parsed.tts_ref_audio, ref_audio)

