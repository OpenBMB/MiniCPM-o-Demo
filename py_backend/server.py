"""Experimental backend protocol server.

This process exposes the draft backend-server protocol while reusing the
current Python/C++ backend methods.  It is intentionally an adapter layer:
strong request/response schemas can replace the loose parsing here later
without changing inference code.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

from core.processors.backend_factory import create_backend
from py_backend.media import decode_audio_base64, decode_frame_base64_list
from py_backend.voice import resolve_duplex_voice_refs
from py_backend.chat_util import (
    convert_to_model_msgs,
    parse_raw_messages,
    parse_worker_chat_request_message,
)


logger = logging.getLogger("backend_server")

SERVER_CONFIG: Dict[str, Any] = {}
_backend: Any = None
_server_state: Optional["BackendServerState"] = None


def _payload(message: Dict[str, Any]) -> Dict[str, Any]:
    payload = message.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError("message must carry an object `payload`")
    return payload


def _message_type(message: Dict[str, Any]) -> str:
    return str(message.get("type") or "")


def _first_dict(*values: Any) -> Dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _coalesce(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _get_input_payload(message: Dict[str, Any]) -> Dict[str, Any]:
    value = message.get("input")
    if not isinstance(value, dict):
        raise RuntimeError("input.append must carry an object `input`")
    return value


def _extract_frame_base64_list(payload: Dict[str, Any]) -> Optional[list[str]]:
    direct = payload.get("frame_base64_list") or payload.get("video_frames")
    if direct:
        return list(direct)

    frames = payload.get("frames")
    if not frames:
        return None
    out: list[str] = []
    for frame in frames:
        if isinstance(frame, str):
            out.append(frame)
        elif isinstance(frame, dict):
            data = frame.get("data") or frame.get("base64")
            if data:
                out.append(data)
    return out or None


def _extract_audio_base64(payload: Dict[str, Any]) -> Optional[str]:
    for key in ("audio_base64", "audio_data"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value

    audio = payload.get("audio")
    if isinstance(audio, str) and audio:
        return audio
    if isinstance(audio, dict):
        value = audio.get("data") or audio.get("base64") or audio.get("audio_base64")
        if isinstance(value, str) and value:
            return value
    return None


def _result_metrics(result: Any, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metrics = dict(base or {})
    for attr, key in (
        ("cost_all_ms", "generate_ms"),
        ("cost_llm_ms", "cost_llm_ms"),
        ("cost_tts_prep_ms", "cost_tts_prep_ms"),
        ("cost_tts_ms", "cost_tts_ms"),
        ("cost_token2wav_ms", "cost_token2wav_ms"),
        ("n_tokens", "n_tokens"),
        ("n_tts_tokens", "n_tts_tokens"),
    ):
        value = getattr(result, attr, None)
        if value is not None:
            metrics[key] = value
    return {key: value for key, value in metrics.items() if value is not None}


@dataclass
class BackendServerState:
    backend: Any
    sessions: Dict[str, "BackendProtocolSession"] = field(default_factory=dict)
    active_session_id: Optional[str] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def register(self, session: "BackendProtocolSession") -> None:
        async with self.lock:
            if self.active_session_id is not None:
                raise RuntimeError("backend already has an active session")
            self.active_session_id = session.session_id
            self.sessions[session.session_id] = session

    async def forget(self, session_id: str) -> None:
        async with self.lock:
            self.sessions.pop(session_id, None)
            if self.active_session_id == session_id:
                self.active_session_id = None


class BackendProtocolSession:
    def __init__(
        self,
        *,
        session_id: str,
        mode: str,
        backend: Any,
        ws: WebSocket,
        state: BackendServerState,
    ) -> None:
        self.session_id = session_id
        self.mode = mode
        self.backend = backend
        self.ws = ws
        self.state = state
        self.closed = False
        self.initialized = False
        self._finalize_done = asyncio.Event()
        self._finalize_done.set()
        self._finalize_task: Optional[asyncio.Task[None]] = None
        self._op_lock = asyncio.Lock()
        self._active_response_id: Optional[str] = None

    async def send(self, event_type: str, **fields: Any) -> None:
        data = {"type": event_type, **{k: v for k, v in fields.items() if v is not None}}
        data["server_send_ts"] = time.time()
        await self.ws.send_json(data)

    async def send_output_delta(self, kind: str, **fields: Any) -> None:
        await self.send("response.output.delta", kind=kind, **fields)

    async def init(self, params: Dict[str, Any]) -> None:
        if self.initialized:
            raise RuntimeError("session is already initialized")
        if self.mode == "full_duplex":
            await self._init_duplex(params)
        self.initialized = True
        await self.send(
            "session.created",
            session_id=self.session_id,
            mode=self.mode,
            metrics=self._safe_metrics(),
        )

    async def push(self, message: Dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("session is closed")
        if not self.initialized:
            raise RuntimeError("session is not initialized")
        payload = _get_input_payload(message)
        if self.mode == "turn_based":
            await self._push_turn_based(payload)
            return
        if self.mode == "full_duplex":
            await self._push_full_duplex(payload)
            return
        raise RuntimeError(f"unsupported mode: {self.mode}")

    async def close(self, *, reason: str = "client_closed", emit_event: bool = True) -> None:
        if self.closed:
            return
        self.closed = True
        await self._drain_finalize()

        if self.mode == "full_duplex":
            with suppress(Exception):
                await asyncio.to_thread(self.backend.duplex_stop)
            await self._drain_finalize()
            await asyncio.to_thread(self.backend.duplex_cleanup)

        if emit_event:
            with suppress(Exception):
                await self.send("session.closed", session_id=self.session_id, reason=reason)
        with suppress(Exception):
            await self.ws.close(code=1000, reason=reason)
        await self.state.forget(self.session_id)

    async def fatal(self, reason: str, *, message: Optional[str] = None) -> None:
        logger.error("fatal backend session termination: session=%s reason=%s message=%s", self.session_id, reason, message)
        self.closed = True
        with suppress(Exception):
            await self.send(
                "session.closed",
                session_id=self.session_id,
                reason=reason,
                diagnostic={"message": message} if message else None,
            )
        with suppress(Exception):
            await self.ws.close(code=1011, reason=reason)
        with suppress(Exception):
            if self.mode == "full_duplex":
                await asyncio.to_thread(self.backend.duplex_stop)
                await self._drain_finalize()
                await asyncio.to_thread(self.backend.duplex_cleanup)
        await self.state.forget(self.session_id)

    async def _init_duplex(self, params: Dict[str, Any]) -> None:
        config = _first_dict(params.get("config"), params.get("duplex"))
        if config:
            await asyncio.to_thread(self.backend.set_duplex_config, config)

        voice = _first_dict(params.get("voice"), params.get("defaults"))
        refs = resolve_duplex_voice_refs(
            ref_audio_path=_coalesce(params.get("ref_audio_path"), voice.get("ref_audio_path")),
            ref_audio_base64=_coalesce(
                params.get("ref_audio_base64"),
                voice.get("ref_audio_base64"),
                voice.get("ref_audio"),
            ),
            tts_ref_audio_base64=_coalesce(
                params.get("tts_ref_audio_base64"),
                voice.get("tts_ref_audio_base64"),
                voice.get("tts_ref_audio"),
            ),
        )
        try:
            await asyncio.to_thread(
                self.backend.duplex_prepare,
                system_prompt_text=_coalesce(
                    params.get("system_prompt"),
                    params.get("instructions"),
                    default="You are a helpful assistant.",
                ),
                ref_audio_path=refs.llm_ref_audio_path,
                prompt_wav_path=refs.tts_ref_audio_path,
                length_penalty=float(config.get("length_penalty", 1.1) if config else 1.1),
                sampling=config or None,
            )
        finally:
            refs.cleanup()

    async def _push_turn_based(self, payload: Dict[str, Any]) -> None:
        async with self._op_lock:
            request = parse_worker_chat_request_message({"type": "chat.request", "payload": payload})
            response_id = str(payload.get("response_id") or f"resp_{uuid.uuid4().hex[:12]}")
            input_id = payload.get("input_id")

            messages = parse_raw_messages(request.messages)
            model_msgs = convert_to_model_msgs(messages)

            await asyncio.to_thread(
                self.backend.chat_prefill,
                session_id=self.session_id,
                msgs=model_msgs,
                omni_mode=request.omni_mode,
                max_slice_nums=request.max_slice_nums,
                use_tts_template=request.use_tts_template,
                enable_thinking=request.enable_thinking,
            )

            if request.generate_audio and request.streaming:
                await asyncio.to_thread(self.backend.chat_init_tts, request.tts_ref_audio)

            if request.streaming:
                await self._stream_turn_based(request, response_id=response_id, input_id=input_id)
            else:
                await self._non_stream_turn_based(request, response_id=response_id, input_id=input_id)

    async def _stream_turn_based(self, request: Any, *, response_id: str, input_id: Optional[str]) -> None:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run_generate() -> None:
            try:
                for chunk in self.backend.chat_streaming_generate(
                    session_id=self.session_id,
                    generate_audio=request.generate_audio,
                    max_new_tokens=request.max_new_tokens,
                    length_penalty=request.length_penalty,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

        task = loop.run_in_executor(None, _run_generate)
        full_text = ""
        try:
            while True:
                tag, payload = await queue.get()
                if tag == "chunk":
                    if payload.text_delta:
                        full_text += payload.text_delta
                        await self.send_output_delta(
                            "text",
                            session_id=self.session_id,
                            response_id=response_id,
                            input_id=input_id,
                            text=payload.text_delta,
                        )
                    if payload.audio_data:
                        await self.send_output_delta(
                            "audio",
                            session_id=self.session_id,
                            response_id=response_id,
                            input_id=input_id,
                            audio=payload.audio_data,
                        )
                    continue
                if tag == "done":
                    await self.send(
                        "response.done",
                        session_id=self.session_id,
                        response_id=response_id,
                        input_id=input_id,
                        text=full_text,
                        reason="turn_end",
                        metrics=self._safe_metrics(),
                    )
                    return
                if tag == "error":
                    raise payload
        finally:
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(task, timeout=5.0)

    async def _non_stream_turn_based(self, request: Any, *, response_id: str, input_id: Optional[str]) -> None:
        result = await asyncio.to_thread(
            self.backend.chat_non_streaming_generate,
            session_id=self.session_id,
            max_new_tokens=request.max_new_tokens,
            generate_audio=request.generate_audio,
            use_tts_template=request.use_tts_template,
            enable_thinking=request.enable_thinking,
            tts_ref_audio=request.tts_ref_audio,
            length_penalty=request.length_penalty,
        )

        text = result
        waveform = None
        if isinstance(result, tuple):
            text, waveform = result

        if waveform is not None:
            audio_base64 = base64.b64encode(waveform.astype(np.float32).tobytes()).decode("utf-8")
        else:
            audio_base64 = None

        await self.send(
            "response.done",
            session_id=self.session_id,
            response_id=response_id,
            input_id=input_id,
            text=text or "",
            audio=audio_base64,
            reason="turn_end",
            metrics=self._safe_metrics(),
        )

    async def _push_full_duplex(self, payload: Dict[str, Any]) -> None:
        async with self._op_lock:
            await self._wait_finalize()
            input_id = payload.get("input_id")
            audio_base64 = _extract_audio_base64(payload)
            if not audio_base64:
                raise RuntimeError("full_duplex input requires audio")

            audio_waveform = decode_audio_base64(audio_base64)
            decoded_frames = decode_frame_base64_list(_extract_frame_base64_list(payload))
            hints = _first_dict(payload.get("hints"))
            force_listen = bool(_coalesce(payload.get("force_listen"), hints.get("force_listen"), default=False))
            max_slice_nums = int(_coalesce(payload.get("max_slice_nums"), hints.get("max_slice_nums"), default=1))

            t0 = time.perf_counter()

            def _duplex_step() -> tuple[Any, float, Dict[str, Any], Dict[str, Any]]:
                prefill_t0 = time.perf_counter()
                prefill_result = self.backend.duplex_prefill(
                    audio_waveform=audio_waveform,
                    frame_list=decoded_frames.frame_list,
                    max_slice_nums=max_slice_nums,
                )
                prefill_ms = (time.perf_counter() - prefill_t0) * 1000
                result = self.backend.duplex_generate(force_listen=force_listen)
                return result, prefill_ms, prefill_result, self._safe_metrics()

            result, prefill_ms, prefill_result, backend_metrics = await asyncio.to_thread(_duplex_step)
            wall_clock_ms = (time.perf_counter() - t0) * 1000
            metrics = _result_metrics(result, backend_metrics)
            metrics["prefill_ms"] = round(prefill_ms, 1)
            metrics["wall_clock_ms"] = round(wall_clock_ms, 1)
            if isinstance(prefill_result, dict):
                n_vision_images = prefill_result.get("n_vision_images")
                if n_vision_images is not None:
                    metrics["vision_slices"] = n_vision_images
                    metrics["vision_tokens"] = int(n_vision_images) * 64

            if result.is_listen:
                await self.send_output_delta(
                    "listen",
                    session_id=self.session_id,
                    response_id=self._active_response_id,
                    input_id=input_id,
                    metrics=metrics,
                )
                self._active_response_id = None
                self._schedule_finalize()
                return

            if self._active_response_id is None:
                self._active_response_id = str(payload.get("response_id") or f"resp_{uuid.uuid4().hex[:12]}")

            if result.text:
                await self.send_output_delta(
                    "text",
                    session_id=self.session_id,
                    response_id=self._active_response_id,
                    input_id=input_id,
                    text=result.text,
                    metrics=metrics,
                )
            if result.audio_data:
                await self.send_output_delta(
                    "audio",
                    session_id=self.session_id,
                    response_id=self._active_response_id,
                    input_id=input_id,
                    audio=result.audio_data,
                    metrics=metrics,
                )
            if result.end_of_turn:
                await self.send_output_delta(
                    "listen",
                    session_id=self.session_id,
                    response_id=self._active_response_id,
                    input_id=input_id,
                    metrics=metrics,
                )
                self._active_response_id = None

            self._schedule_finalize()

    def _safe_metrics(self) -> Dict[str, Any]:
        try:
            metrics = self.backend.metrics()
            return dict(metrics or {})
        except Exception:
            logger.exception("backend metrics failed")
            return {}

    async def _wait_finalize(self) -> None:
        await self._finalize_done.wait()
        if self._finalize_task is not None and self._finalize_task.done():
            self._finalize_task.result()
            self._finalize_task = None

    def _schedule_finalize(self) -> None:
        if self._finalize_task is not None and not self._finalize_task.done():
            raise RuntimeError("duplex finalize already in flight")

        self._finalize_done.clear()

        async def _run() -> None:
            try:
                await asyncio.to_thread(self.backend.duplex_finalize)
            finally:
                self._finalize_done.set()

        self._finalize_task = asyncio.create_task(_run())

    async def _drain_finalize(self) -> None:
        task = self._finalize_task
        if task is None:
            return
        if task.done():
            task.result()
            self._finalize_task = None
            return
        try:
            await asyncio.wait_for(task, timeout=5.0)
        finally:
            self._finalize_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _backend, _server_state
    logging.basicConfig(level=logging.INFO)
    logger.info("Loading backend server backend: pytorch")
    _backend = create_backend(SERVER_CONFIG)
    await asyncio.to_thread(_backend.load_model)
    _server_state = BackendServerState(_backend)
    logger.info("Backend server ready")
    try:
        yield
    finally:
        for session in list((_server_state.sessions if _server_state else {}).values()):
            with suppress(Exception):
                await session.close(reason="server_shutdown")
        if _backend is not None and hasattr(_backend, "shutdown"):
            await asyncio.to_thread(_backend.shutdown)


app = FastAPI(title="MiniCPMO45 Backend Protocol Server", lifespan=lifespan)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ready" if _backend is not None else "loading",
        "backend": "pytorch",
        "worker_status": getattr(_backend, "status", None),
        "active_session_id": _server_state.active_session_id if _server_state else None,
    }


@app.websocket("/backend")
async def backend_ws(ws: WebSocket) -> None:
    if _server_state is None or _backend is None:
        await ws.close(code=1013, reason="backend not ready")
        return

    await ws.accept()
    session: Optional[BackendProtocolSession] = None
    try:
        first = json.loads(await ws.receive_text())
        if _message_type(first) != "session.init":
            raise RuntimeError("first message must be session.init")

        params = _payload(first)
        mode = str(params.get("mode") or "full_duplex")
        # session identity 由 backend 分配，不接受客户端建议的 session_id（见协议 schema §3.1）
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        session = BackendProtocolSession(
            session_id=session_id,
            mode=mode,
            backend=_backend,
            ws=ws,
            state=_server_state,
        )
        await _server_state.register(session)
        await session.init(params)

        while not session.closed:
            message = json.loads(await ws.receive_text())
            msg_type = _message_type(message)
            if msg_type == "input.append":
                await session.push(message)
                continue
            # close 只走 HTTP unary 控制通道（见协议 network §3.2），WS 上不接受 close
            raise RuntimeError(f"unsupported message type: {msg_type}")

    except WebSocketDisconnect:
        if session is not None:
            await session.close(reason="client_disconnected", emit_event=False)
    except Exception as exc:
        if session is not None:
            await session.fatal("backend_error", message=str(exc))
        else:
            with suppress(Exception):
                await ws.close(code=1011, reason="backend_error")


@app.post("/sessions/{session_id}/close")
async def close_session(session_id: str, request: Request) -> Dict[str, Any]:
    if _server_state is None:
        raise HTTPException(status_code=503, detail="backend not ready")

    session = _server_state.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    reason = "client_closed"
    with suppress(Exception):
        body = await request.json()
        if isinstance(body, dict) and body.get("reason"):
            reason = str(body["reason"])

    await session.close(reason=reason)
    return {"ok": True, "session_id": session_id, "closed": True}


def main() -> None:
    from config import get_config
    import uvicorn

    cfg = get_config()
    parser = argparse.ArgumentParser(description="MiniCPMO45 backend protocol server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=22500)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--pt-path", default=None)
    parser.add_argument("--ref-audio-path", default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--duplex-pause-timeout", type=float, default=None)
    args = parser.parse_args()

    model_path = args.model_path or cfg.model.model_path
    if not model_path:
        parser.error(
            "model path is required for the backend: pass --model-path <dir> "
            "or set model.model_path in config.json"
        )

    SERVER_CONFIG.update({
        "model_path": model_path,
        "gpu_id": args.gpu_id,
        "pt_path": args.pt_path or cfg.model.pt_path,
        "ref_audio_path": args.ref_audio_path or cfg.ref_audio_path,
        "duplex_pause_timeout": args.duplex_pause_timeout or cfg.duplex_pause_timeout,
        "compile": cfg.compile,
        "chat_vocoder": cfg.chat_vocoder,
        "attn_implementation": cfg.attn_implementation,
    })

    uvicorn.run(app, host=args.host, port=args.port, ws_max_size=128 * 1024 * 1024)


if __name__ == "__main__":
    main()
