import os

import numpy as np
import soundfile as sf

from py_backend.voice import resolve_duplex_voice_refs


def _pcm_b64(samples: int = 1600) -> str:
    import base64

    audio = np.zeros(samples, dtype=np.float32)
    return base64.b64encode(audio.tobytes()).decode("utf-8")


def test_resolve_duplex_voice_refs_reuses_llm_ref_for_tts_when_same():
    refs = resolve_duplex_voice_refs(
        ref_audio_path=None,
        ref_audio_base64=_pcm_b64(),
        tts_ref_audio_base64=None,
    )
    try:
        assert refs.llm_ref_audio_path
        assert refs.tts_ref_audio_path == refs.llm_ref_audio_path
        assert refs.tts_same_as_llm is True
        data, sr = sf.read(refs.llm_ref_audio_path)
        assert sr == 16000
        assert len(data) == 1600
    finally:
        path = refs.llm_ref_audio_path
        refs.cleanup()
        assert path is None or not os.path.exists(path)


def test_resolve_duplex_voice_refs_supports_independent_tts_ref():
    refs = resolve_duplex_voice_refs(
        ref_audio_path=None,
        ref_audio_base64=_pcm_b64(800),
        tts_ref_audio_base64=_pcm_b64(1200),
    )
    try:
        assert refs.llm_ref_audio_path
        assert refs.tts_ref_audio_path
        assert refs.tts_ref_audio_path != refs.llm_ref_audio_path
        assert refs.tts_field_present is True
        assert refs.tts_same_as_llm is False
    finally:
        paths = [refs.llm_ref_audio_path, refs.tts_ref_audio_path]
        refs.cleanup()
        assert all(path is None or not os.path.exists(path) for path in paths)

