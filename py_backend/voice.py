"""Voice reference artifact helpers for session runtimes."""

from __future__ import annotations

import base64
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import soundfile as sf


@dataclass
class DuplexVoiceRefs:
    """Resolved voice references for a duplex session.

    Public/session config may provide reference audio as a path or as base64 PCM.
    Backends currently need filesystem paths, so this helper materializes temp
    WAVs where needed and tracks them for cleanup.
    """

    llm_ref_audio_path: Optional[str]
    tts_ref_audio_path: Optional[str]
    tts_field_present: bool
    tts_same_as_llm: bool
    temp_files: list[str] = field(default_factory=list)

    def cleanup(self) -> None:
        for path in self.temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self.temp_files.clear()


def _base64_pcm_to_temp_wav(audio_b64: str, prefix: str) -> str:
    audio_bytes = base64.b64decode(audio_b64)
    audio = np.frombuffer(audio_bytes, dtype=np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix=prefix)
    sf.write(tmp.name, audio, 16000)
    tmp.close()
    return tmp.name


def resolve_duplex_voice_refs(
    *,
    ref_audio_path: Optional[str],
    ref_audio_base64: Optional[str],
    tts_ref_audio_base64: Optional[str],
) -> DuplexVoiceRefs:
    """Resolve LLM and TTS reference audio into backend-facing file paths."""

    llm_ref_audio_path = ref_audio_path
    tts_ref_audio_path = None
    temp_files: list[str] = []

    if ref_audio_base64 and not ref_audio_path:
        llm_ref_audio_path = _base64_pcm_to_temp_wav(ref_audio_base64, "duplex_llm_ref_")
        temp_files.append(llm_ref_audio_path)

    effective_tts_ref = tts_ref_audio_base64 or ref_audio_base64
    tts_field_present = bool(tts_ref_audio_base64)
    tts_same_as_llm = (effective_tts_ref == ref_audio_base64) if effective_tts_ref else True

    if effective_tts_ref and effective_tts_ref != ref_audio_base64:
        tts_ref_audio_path = _base64_pcm_to_temp_wav(effective_tts_ref, "duplex_tts_ref_")
        temp_files.append(tts_ref_audio_path)
    elif llm_ref_audio_path:
        tts_ref_audio_path = llm_ref_audio_path

    return DuplexVoiceRefs(
        llm_ref_audio_path=llm_ref_audio_path,
        tts_ref_audio_path=tts_ref_audio_path,
        tts_field_present=tts_field_present,
        tts_same_as_llm=tts_same_as_llm,
        temp_files=temp_files,
    )

