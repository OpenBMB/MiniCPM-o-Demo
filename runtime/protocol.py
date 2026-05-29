"""Worker/runtime protocol constants shared by worker and gateway layers."""

from __future__ import annotations

from typing import Dict, List


DEFAULT_WORKER_CAPABILITIES: List[str] = [
    "chat",
    "streaming",
    "half_duplex_audio",
    "audio_duplex",
    "omni_duplex",
]

REQUEST_CAPABILITY_MAP: Dict[str, str] = {
    "chat": "chat",
    "streaming": "streaming",
    "chat_ws": "streaming",
    "half_duplex_audio": "half_duplex_audio",
    "audio_duplex": "audio_duplex",
    "omni_duplex": "omni_duplex",
    "duplex": "omni_duplex",
}


def capability_for_request(request_type: str) -> str:
    return REQUEST_CAPABILITY_MAP.get(request_type, request_type)

