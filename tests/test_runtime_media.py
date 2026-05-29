import base64
import io

import numpy as np
from PIL import Image

from py_backend.media import decode_audio_base64, decode_frame_base64_list


def test_decode_audio_base64_returns_float32_waveform():
    audio = np.arange(8, dtype=np.float32)
    encoded = base64.b64encode(audio.tobytes()).decode("utf-8")

    decoded = decode_audio_base64(encoded)

    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, audio)


def test_decode_frame_base64_list_returns_images_and_first_raw_bytes():
    img = Image.new("RGB", (2, 3), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw = buf.getvalue()

    decoded = decode_frame_base64_list([base64.b64encode(raw).decode("utf-8")])

    assert decoded.first_frame_bytes == raw
    assert decoded.frame_list is not None
    assert decoded.frame_list[0].size == (2, 3)

