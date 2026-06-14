"""Media decoding helpers for worker/runtime boundaries."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DecodedFrames:
    frame_list: Optional[list]
    first_frame_bytes: Optional[bytes] = None


def decode_audio_base64(audio_b64: str) -> np.ndarray:
    """Decode base64 float32 PCM audio into a numpy waveform."""

    audio_bytes = base64.b64decode(audio_b64)
    return np.frombuffer(audio_bytes, dtype=np.float32)


def decode_frame_base64_list(frame_b64_list: Optional[list[str]]) -> DecodedFrames:
    """Decode optional JPEG base64 frame list into PIL images.

    The first raw JPEG bytes are returned so persistence/recording code can save
    the original compressed frame without re-encoding the PIL image.
    """

    if not frame_b64_list:
        return DecodedFrames(frame_list=None)

    from PIL import Image

    frame_list = []
    first_frame_bytes: Optional[bytes] = None
    for fb64 in frame_b64_list:
        frame_bytes = base64.b64decode(fb64)
        if first_frame_bytes is None:
            first_frame_bytes = frame_bytes
        frame_list.append(Image.open(io.BytesIO(frame_bytes)))
    return DecodedFrames(frame_list=frame_list, first_frame_bytes=first_frame_bytes)

