"""Legacy /ws/duplex message adapters.

The external worker WebSocket protocol is intentionally kept stable for now.
This module is the translation seam from those legacy message dictionaries into
runtime-facing parameter objects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.runtime.duplex import DuplexInputFrame, DuplexPrepareParams
from core.runtime.events import RuntimeControl, RuntimeEvent
from core.runtime.media import decode_audio_base64, decode_frame_base64_list
from core.runtime.voice import DuplexVoiceRefs, resolve_duplex_voice_refs


@dataclass
class LegacyDuplexPrepare:
    params: DuplexPrepareParams
    voice_refs: DuplexVoiceRefs
    use_deferred_finalize: bool
    session_max_slice_nums: int
    system_prompt: str
    config: Optional[Dict[str, Any]]

    def cleanup(self) -> None:
        self.voice_refs.cleanup()

    @property
    def recording_ref_audio_path(self) -> Optional[str]:
        return self.voice_refs.llm_ref_audio_path


@dataclass
class LegacyDuplexInput:
    frame: DuplexInputFrame
    first_frame_bytes: Optional[bytes]


def parse_prepare_message(msg: Dict[str, Any]) -> LegacyDuplexPrepare:
    """Translate a legacy prepare message into runtime prepare params."""

    system_prompt = msg.get("system_prompt", "You are a helpful assistant.")
    ref_audio_path = msg.get("ref_audio_path")
    ref_audio_b64 = msg.get("ref_audio_base64")
    # TTS ref audio is independent; fallback to ref_audio_base64 for compatibility.
    tts_ref_audio_b64 = msg.get("tts_ref_audio_base64") or ref_audio_b64
    config = msg.get("config")

    use_deferred_finalize = msg.get("deferred_finalize", True)
    session_max_slice_nums = (
        msg.get("max_slice_nums")
        or (config.get("max_slice_nums") if config else None)
        or 1
    )

    voice_refs = resolve_duplex_voice_refs(
        ref_audio_path=ref_audio_path,
        ref_audio_base64=ref_audio_b64,
        tts_ref_audio_base64=tts_ref_audio_b64,
    )

    return LegacyDuplexPrepare(
        params=DuplexPrepareParams(
            system_prompt_text=system_prompt,
            ref_audio_path=voice_refs.llm_ref_audio_path,
            prompt_wav_path=voice_refs.tts_ref_audio_path,
            config=config,
        ),
        voice_refs=voice_refs,
        use_deferred_finalize=use_deferred_finalize,
        session_max_slice_nums=int(session_max_slice_nums),
        system_prompt=system_prompt,
        config=config,
    )


def parse_audio_chunk_message(
    msg: Dict[str, Any],
    *,
    session_max_slice_nums: int,
    chunk_start: Optional[float] = None,
) -> LegacyDuplexInput:
    """Translate a legacy audio_chunk message into a runtime input frame."""

    audio_b64 = msg.get("audio_base64")
    if not audio_b64:
        raise ValueError("Missing audio_base64")

    decoded_frames = decode_frame_base64_list(msg.get("frame_base64_list"))
    frame = DuplexInputFrame(
        audio_waveform=decode_audio_base64(audio_b64),
        frame_list=decoded_frames.frame_list,
        max_slice_nums=int(msg.get("max_slice_nums", session_max_slice_nums)),
        force_listen=bool(msg.get("force_listen", False)),
        chunk_start=chunk_start if chunk_start is not None else time.perf_counter(),
    )
    return LegacyDuplexInput(
        frame=frame,
        first_frame_bytes=decoded_frames.first_frame_bytes,
    )


def parse_control_message(msg: Dict[str, Any]) -> Optional[RuntimeControl]:
    """Translate legacy control messages into runtime control commands."""

    msg_type = msg.get("type")
    if msg_type == "pause":
        return RuntimeControl(
            type="session.pause",
            payload={"timeout": msg.get("timeout")},
        )
    if msg_type == "resume":
        return RuntimeControl(type="session.resume")
    if msg_type == "stop":
        return RuntimeControl(type="session.close", payload={"reason": "client_stop"})
    if msg_type == "interrupt":
        return RuntimeControl(type="legacy.interrupt")
    return None


def legacy_result_payload_from_event(event: RuntimeEvent) -> Dict[str, Any]:
    """Translate a runtime duplex output event to the legacy WS result payload."""

    if event.channel != "output.duplex_result":
        raise ValueError(f"unsupported event channel for legacy duplex result: {event.channel}")
    result_dict = event.payload.get("result_dict")
    if not isinstance(result_dict, dict):
        raise ValueError("runtime event missing result_dict payload")
    return {"type": "result", **result_dict}

