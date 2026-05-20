"""Worker-internal runtime protocol helpers.

This module defines the first version of a gateway-worker runtime protocol.  It
is intentionally separate from legacy `/ws/duplex` payloads so gateway code can
eventually talk to workers without inheriting page/demo message shapes.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from core.runtime.duplex import DuplexInputFrame, DuplexPrepareParams
from core.runtime.events import RuntimeControl, RuntimeEvent
from core.runtime.media import decode_audio_base64, decode_frame_base64_list
from core.runtime.voice import DuplexVoiceRefs, resolve_duplex_voice_refs


class WorkerProtocolError(ValueError):
    pass


def _coalesce_int(value: Any, default: int) -> int:
    return int(default if value is None else value)


def parse_worker_prepare_message(msg: Dict[str, Any]) -> tuple[DuplexPrepareParams, DuplexVoiceRefs]:
    """Parse a `session.prepare` worker protocol message."""

    payload = msg.get("payload") or {}
    if not isinstance(payload, dict):
        raise WorkerProtocolError("session.prepare payload must be an object")

    voice = payload.get("voice") or {}
    if not isinstance(voice, dict):
        raise WorkerProtocolError("session.prepare voice must be an object")

    voice_refs = resolve_duplex_voice_refs(
        ref_audio_path=voice.get("ref_audio_path"),
        ref_audio_base64=voice.get("ref_audio_base64"),
        tts_ref_audio_base64=voice.get("tts_ref_audio_base64"),
    )
    params = DuplexPrepareParams(
        system_prompt_text=payload.get("system_prompt", "You are a helpful assistant."),
        ref_audio_path=voice_refs.llm_ref_audio_path,
        prompt_wav_path=voice_refs.tts_ref_audio_path,
        config=payload.get("config"),
    )
    return params, voice_refs


def parse_worker_input_message(
    msg: Dict[str, Any],
    *,
    default_max_slice_nums: int = 1,
    chunk_start: Optional[float] = None,
) -> DuplexInputFrame:
    """Parse an `input.append` worker protocol message."""

    payload = msg.get("payload") or {}
    if not isinstance(payload, dict):
        raise WorkerProtocolError("input.append payload must be an object")

    audio_b64 = payload.get("audio_base64")
    if not audio_b64:
        raise WorkerProtocolError("input.append payload.audio_base64 is required")

    decoded_frames = decode_frame_base64_list(payload.get("frame_base64_list"))
    return DuplexInputFrame(
        audio_waveform=decode_audio_base64(audio_b64),
        frame_list=decoded_frames.frame_list,
        max_slice_nums=_coalesce_int(payload.get("max_slice_nums"), default_max_slice_nums),
        force_listen=bool(payload.get("force_listen", False)),
        chunk_start=chunk_start if chunk_start is not None else time.perf_counter(),
    )


def parse_worker_control_message(msg: Dict[str, Any]) -> RuntimeControl:
    """Parse a `control` worker protocol message."""

    payload = msg.get("payload") or {}
    if not isinstance(payload, dict):
        raise WorkerProtocolError("control payload must be an object")
    command = payload.get("command")
    if not command:
        raise WorkerProtocolError("control payload.command is required")
    return RuntimeControl(type=command, payload={k: v for k, v in payload.items() if k != "command"})


def runtime_event_to_worker_message(event: RuntimeEvent) -> Dict[str, Any]:
    """Serialize a RuntimeEvent for the worker protocol."""

    if event.channel == "output.duplex_result":
        frame = event.payload.get("frame")
        payload = {
            "result": event.payload.get("result_dict", {}),
            "metrics": {
                "prefill_ms": event.payload.get("prefill_ms"),
                "kv_cache_len": event.payload.get("kv_cache_len"),
                "wall_clock_ms": event.payload.get("wall_clock_ms"),
                "vision_slices": event.payload.get("n_vision_images"),
                "vision_tokens": event.payload.get("vision_tokens"),
            },
        }
        if frame is not None:
            payload["metrics"]["prefill_ms"] = getattr(frame, "prefill_ms", payload["metrics"]["prefill_ms"])
        return {
            "type": "runtime.event",
            "channel": event.channel,
            "payload": payload,
        }

    return {
        "type": "runtime.event",
        "channel": event.channel,
        "payload": event.payload,
    }

