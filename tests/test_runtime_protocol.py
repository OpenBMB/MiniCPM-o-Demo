from core.runtime.protocol import DEFAULT_WORKER_CAPABILITIES, capability_for_request


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

