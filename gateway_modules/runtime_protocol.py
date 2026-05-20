"""Gateway-side translators for the worker runtime protocol."""

from __future__ import annotations

from typing import Any, Dict


def chat_client_to_worker_runtime(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a public turn-based chat request for the worker runtime protocol."""

    return {
        "type": "chat.request",
        "payload": msg,
    }


def worker_runtime_to_response_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Translate worker chat runtime messages to public response API events."""

    msg_type = msg.get("type")
    payload = msg.get("payload") or {}

    if msg_type == "chat.prefill_done":
        return {
            "type": "response.prefill.done",
            "input_tokens": payload.get("input_tokens", 0),
        }

    if msg_type == "chat.chunk":
        if payload.get("audio_base64"):
            return {
                "type": "response.output_audio.delta",
                "audio": payload.get("audio_base64"),
                "text": payload.get("text_delta", ""),
            }
        return {
            "type": "response.output_text.delta",
            "text": payload.get("text_delta", ""),
        }

    if msg_type == "chat.done":
        out = {
            "type": "response.completed",
            "text": payload.get("text", ""),
            "audio": payload.get("audio_base64"),
            "generated_tokens": payload.get("generated_tokens", 0),
            "input_tokens": payload.get("input_tokens", 0),
        }
        if payload.get("recording_session_id"):
            out["recording_session_id"] = payload["recording_session_id"]
        return out

    return msg


def realtime_client_to_worker_runtime(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Translate public /v1/realtime client events to worker runtime protocol."""

    msg_type = msg.get("type")
    if msg_type == "session.update":
        session_cfg = msg.get("session") or {}
        config = dict(session_cfg.get("voice_config") or {})
        if session_cfg.get("max_slice_nums") is not None:
            config["max_slice_nums"] = session_cfg.get("max_slice_nums")
        return {
            "type": "duplex.session.prepare",
            "payload": {
                "system_prompt": session_cfg.get("instructions", "You are a helpful assistant."),
                "config": config or None,
                "voice": {
                    "ref_audio_base64": session_cfg.get("ref_audio"),
                    "tts_ref_audio_base64": session_cfg.get("tts_ref_audio"),
                },
            },
        }

    if msg_type == "input_audio_buffer.append":
        return {
            "type": "duplex.input.audio.append",
            "payload": {
                "audio_base64": msg.get("audio", ""),
                "frame_base64_list": msg.get("video_frames"),
                "force_listen": msg.get("force_listen", False),
                "max_slice_nums": msg.get("max_slice_nums"),
            },
        }

    if msg_type == "session.close":
        return {
            "type": "duplex.control.close",
            "payload": {"reason": msg.get("reason", "client_closed")},
        }

    if msg_type == "response.cancel":
        return {"type": "duplex.control.cancel", "payload": {}}

    return msg


def worker_runtime_to_realtime(msg: Dict[str, Any], *, session_id: str) -> Dict[str, Any]:
    """Translate worker runtime protocol events to public /v1/realtime events."""

    msg_type = msg.get("type")
    if msg_type == "duplex.session.ready":
        return {
            "type": "session.created",
            "session_id": session_id,
            "prompt_length": msg.get("prompt_length", 0),
        }

    payload = msg.get("payload") or {}
    if msg_type == "duplex.output.listen":
        return {"type": "response.listen"}

    if msg_type == "duplex.output.text.delta":
        return {
            "type": "response.output_text.delta",
            "text": payload.get("text", ""),
        }

    if msg_type == "duplex.output.audio.delta":
        return {
            "type": "response.output_audio.delta",
            "text": payload.get("text", ""),
            "audio": payload.get("audio_base64"),
            "end_of_turn": payload.get("end_of_turn", False),
        }

    if msg_type == "duplex.output.turn.done":
        return {"type": "response.done"}

    if msg_type == "duplex.session.closed":
        return {"type": "session.closed", "reason": payload.get("reason", "stopped")}
    if msg_type == "duplex.output.cancelled":
        return {"type": "response.cancelled"}
    if msg_type == "duplex.session.paused":
        return {"type": "session.paused", "timeout": payload.get("timeout")}
    if msg_type == "duplex.session.resumed":
        return {"type": "session.resumed"}
    if msg_type == "duplex.metrics.frame":
        return {"type": "response.metrics", **payload}

    return msg

