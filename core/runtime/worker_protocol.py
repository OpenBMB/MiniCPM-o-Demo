"""Worker-internal runtime protocol helpers.

This module defines the first version of a gateway-worker runtime protocol.  It
is intentionally separate from legacy `/ws/duplex` payloads so gateway code can
eventually talk to workers without inheriting page/demo message shapes.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from core.runtime.events import RuntimeControl, RuntimeEvent
from core.runtime.media import decode_audio_base64, decode_frame_base64_list
from core.runtime.voice import DuplexVoiceRefs, resolve_duplex_voice_refs


@dataclass
class DuplexFrameResult:
    """A completed duplex frame ready to be emitted by the transport layer."""

    result: Any
    result_dict: Dict[str, Any]
    prefill_ms: float
    prefill_result: Dict[str, Any]
    metrics: Dict[str, Any]
    wall_clock_ms: float
    n_vision_images: int
    vision_tokens: int

    @property
    def kv_cache_len(self) -> int:
        return int(self.metrics.get("kv_cache_length", 0) or 0)

    def to_runtime_event(self) -> RuntimeEvent:
        return RuntimeEvent(
            channel="output.duplex_result",
            payload={
                "frame": self,
                "result": self.result,
                "result_dict": self.result_dict,
                "prefill_ms": self.prefill_ms,
                "prefill_result": self.prefill_result,
                "metrics": self.metrics,
                "wall_clock_ms": self.wall_clock_ms,
                "n_vision_images": self.n_vision_images,
                "vision_tokens": self.vision_tokens,
            },
        )


@dataclass
class DuplexPrepareParams:
    """Backend-facing prepare parameters for one duplex session."""

    system_prompt_text: Optional[str]
    ref_audio_path: Optional[str]
    prompt_wav_path: Optional[str]
    config: Optional[Dict[str, Any]] = None


@dataclass
class DuplexInputFrame:
    """Backend-facing input frame for one duplex unit."""

    audio_waveform: np.ndarray
    frame_list: Optional[list]
    max_slice_nums: int = 1
    force_listen: bool = False
    chunk_start: Optional[float] = None


class WorkerProtocolError(ValueError):
    pass


def _coalesce_int(value: Any, default: int) -> int:
    return int(default if value is None else value)


@dataclass
class WorkerChatRequest:
    messages: list
    streaming: bool
    max_new_tokens: int
    length_penalty: float
    max_slice_nums: Optional[int]
    generate_audio: bool
    tts_ref_audio: Optional[np.ndarray]
    use_tts_template: bool
    omni_mode: bool
    enable_thinking: bool


def parse_worker_chat_request_message(msg: Dict[str, Any]) -> WorkerChatRequest:
    """Parse a `chat.request` worker protocol message."""

    if msg.get("type") != "chat.request":
        raise WorkerProtocolError("expected chat.request message")

    payload = msg.get("payload") or {}
    if not isinstance(payload, dict):
        raise WorkerProtocolError("chat.request payload must be an object")

    generation = payload.get("generation") or {}
    if not isinstance(generation, dict):
        raise WorkerProtocolError("chat.request generation must be an object")

    image = payload.get("image") or {}
    if not isinstance(image, dict):
        raise WorkerProtocolError("chat.request image must be an object")

    tts = payload.get("tts") or {}
    if not isinstance(tts, dict):
        raise WorkerProtocolError("chat.request tts must be an object")

    max_slice_nums = None
    if image.get("max_slice_nums") is not None:
        max_slice_nums = int(image["max_slice_nums"])

    generate_audio = bool(tts.get("enabled", False))
    tts_ref_audio = None
    ref_b64 = tts.get("ref_audio_data")
    if generate_audio and ref_b64:
        tts_ref_audio = np.frombuffer(base64.b64decode(ref_b64), dtype=np.float32)

    return WorkerChatRequest(
        messages=payload.get("messages", []),
        streaming=bool(payload.get("streaming", True)),
        max_new_tokens=int(generation.get("max_new_tokens", 256)),
        length_penalty=float(generation.get("length_penalty", 1.1)),
        max_slice_nums=max_slice_nums,
        generate_audio=generate_audio,
        tts_ref_audio=tts_ref_audio,
        use_tts_template=bool(payload.get("use_tts_template", False) or generate_audio),
        omni_mode=bool(payload.get("omni_mode", False)),
        enable_thinking=bool(payload.get("enable_thinking", False)),
    )


def parse_worker_prepare_message(msg: Dict[str, Any]) -> tuple[DuplexPrepareParams, DuplexVoiceRefs]:
    """Parse a `duplex.session.prepare` worker protocol message."""

    payload = msg.get("payload") or {}
    if not isinstance(payload, dict):
        raise WorkerProtocolError("duplex.session.prepare payload must be an object")

    voice = payload.get("voice") or {}
    if not isinstance(voice, dict):
        raise WorkerProtocolError("duplex.session.prepare voice must be an object")

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
    """Parse a `duplex.input.audio.append` worker protocol message."""

    payload = msg.get("payload") or {}
    if not isinstance(payload, dict):
        raise WorkerProtocolError("duplex.input.audio.append payload must be an object")

    audio_b64 = payload.get("audio_base64")
    if not audio_b64:
        raise WorkerProtocolError("duplex.input.audio.append payload.audio_base64 is required")

    decoded_frames = decode_frame_base64_list(payload.get("frame_base64_list"))
    return DuplexInputFrame(
        audio_waveform=decode_audio_base64(audio_b64),
        frame_list=decoded_frames.frame_list,
        max_slice_nums=_coalesce_int(payload.get("max_slice_nums"), default_max_slice_nums),
        force_listen=bool(payload.get("force_listen", False)),
        chunk_start=chunk_start if chunk_start is not None else time.perf_counter(),
    )


def parse_worker_control_message(msg: Dict[str, Any]) -> RuntimeControl:
    """Parse a `duplex.control.*` worker protocol message."""

    msg_type = msg.get("type")
    payload = msg.get("payload") or {}
    if not isinstance(payload, dict):
        raise WorkerProtocolError("duplex control payload must be an object")

    command_by_type = {
        "duplex.control.pause": "session.pause",
        "duplex.control.resume": "session.resume",
        "duplex.control.cancel": "response.cancel",
        "duplex.control.close": "session.close",
    }
    command = command_by_type.get(msg_type)
    if command is None:
        raise WorkerProtocolError(f"unknown duplex control message type: {msg_type}")
    return RuntimeControl(type=command, payload=payload)


def runtime_event_to_worker_messages(event: RuntimeEvent) -> list[Dict[str, Any]]:
    """Serialize a RuntimeEvent into fine-grained worker protocol messages."""

    if event.channel == "output.duplex_result":
        frame = event.payload.get("frame")
        result = event.payload.get("result_dict", {})
        metrics = dict(event.payload.get("metrics") or {})
        if frame is not None:
            metrics["prefill_ms"] = getattr(frame, "prefill_ms", metrics.get("prefill_ms"))

        messages: list[Dict[str, Any]] = [{
            "type": "duplex.metrics.frame",
            "payload": metrics,
        }]

        if result.get("is_listen"):
            messages.append({
                "type": "duplex.output.listen",
                "payload": {},
            })
            return messages

        text = result.get("text", "") or ""
        if text:
            messages.append({
                "type": "duplex.output.text.delta",
                "payload": {
                    "text": text,
                },
            })

        messages.append({
            "type": "duplex.output.audio.delta",
            "payload": {
                "audio_base64": result.get("audio_data"),
                # Keep text on the audio event while the public UI still uses
                # one speak event for text and audio display.
                "text": text,
                "end_of_turn": result.get("end_of_turn", False),
            },
        })
        if result.get("end_of_turn", False):
            messages.append({
                "type": "duplex.output.turn.done",
                "payload": {},
            })
        return messages

    if event.channel == "session":
        state = event.payload.get("state")
        if state == "closed":
            return [{"type": "duplex.session.closed", "payload": event.payload}]
        if state == "cancelled":
            return [{"type": "duplex.output.cancelled", "payload": event.payload}]
        if state == "paused":
            return [{"type": "duplex.session.paused", "payload": event.payload}]
        if state == "active":
            return [{"type": "duplex.session.resumed", "payload": event.payload}]

    return [{
        "type": "duplex.event",
        "channel": event.channel,
        "payload": event.payload,
    }]


def runtime_event_to_worker_message(event: RuntimeEvent) -> Dict[str, Any]:
    """Serialize a RuntimeEvent for callers that expect one message."""

    messages = runtime_event_to_worker_messages(event)
    if len(messages) == 1:
        return messages[0]
    if event.channel == "output.duplex_result":
        result = event.payload.get("result_dict", {})
        metrics = messages[0].get("payload", {})
        payload = {
            "result": result,
            "metrics": metrics,
        }
        return {
            "type": "duplex.output.result",
            "payload": payload,
        }
    return messages[0]

