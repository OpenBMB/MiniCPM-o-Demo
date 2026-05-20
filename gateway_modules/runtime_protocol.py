"""Gateway-side translators for the worker runtime protocol."""

from __future__ import annotations

from typing import Any, Dict


def chat_client_to_worker_runtime(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a public turn-based chat request for the worker runtime protocol."""

    return {
        "type": "chat.request",
        "payload": msg,
    }


def worker_runtime_to_legacy_chat(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Translate worker chat runtime messages back to the existing page protocol."""

    msg_type = msg.get("type")
    payload = msg.get("payload") or {}

    if msg_type == "chat.prefill_done":
        return {
            "type": "prefill_done",
            "input_tokens": payload.get("input_tokens", 0),
        }

    if msg_type == "chat.chunk":
        out = {"type": "chunk"}
        if payload.get("text_delta"):
            out["text_delta"] = payload["text_delta"]
        if payload.get("audio_base64"):
            out["audio_data"] = payload["audio_base64"]
        return out

    if msg_type == "chat.done":
        out = {
            "type": "done",
            "text": payload.get("text", ""),
            "audio_data": payload.get("audio_base64"),
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
            "type": "session.prepare",
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
            "type": "input.append",
            "payload": {
                "audio_base64": msg.get("audio", ""),
                "frame_base64_list": msg.get("video_frames"),
                "force_listen": msg.get("force_listen", False),
                "max_slice_nums": msg.get("max_slice_nums"),
            },
        }

    if msg_type == "session.close":
        return {
            "type": "control",
            "payload": {"command": "session.close", "reason": msg.get("reason", "client_closed")},
        }

    if msg_type == "response.cancel":
        return {"type": "control", "payload": {"command": "response.cancel"}}

    return msg


def worker_runtime_to_realtime(msg: Dict[str, Any], *, session_id: str) -> Dict[str, Any]:
    """Translate worker runtime protocol events to public /v1/realtime events."""

    msg_type = msg.get("type")
    if msg_type == "session.ready":
        return {
            "type": "session.created",
            "session_id": session_id,
            "prompt_length": msg.get("prompt_length", 0),
        }

    if msg_type == "runtime.event":
        channel = msg.get("channel")
        payload = msg.get("payload") or {}

        if channel == "output.duplex_result":
            result = payload.get("result") or {}
            kv_len = result.get("kv_cache_length", 0)
            if result.get("is_listen"):
                return {
                    "type": "response.listen",
                    "kv_cache_length": kv_len,
                }
            return {
                "type": "response.output_audio.delta",
                "text": result.get("text", ""),
                "audio": result.get("audio_data"),
                "end_of_turn": result.get("end_of_turn", False),
                "kv_cache_length": kv_len,
            }

        if channel == "session":
            state = payload.get("state")
            if state == "closed":
                return {"type": "session.closed", "reason": payload.get("reason", "stopped")}
            if state == "cancelled":
                return {"type": "response.cancelled"}
            if state == "paused":
                return {"type": "session.paused", "timeout": payload.get("timeout")}
            if state == "active":
                return {"type": "session.resumed"}

    return msg

