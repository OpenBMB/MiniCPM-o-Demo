"""Worker runtime WebSocket handlers.

These handlers keep transport loops out of ``worker.py`` while still operating
against the current worker/backend adapter contracts.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from core.runtime.backends import ChatBackendView, DuplexBackendView
from core.runtime.chat import ChatGenerateParams, ChatPrefillParams, ChatSessionRuntime
from core.runtime.events import RuntimeEvent
from core.runtime.manager import RuntimeManager
from core.runtime.metrics import log_duplex_frame
from core.runtime.worker_protocol import (
    WorkerProtocolError,
    parse_worker_chat_request_message,
    parse_worker_control_message,
    parse_worker_input_message,
    parse_worker_prepare_message,
    runtime_event_to_worker_messages,
)
from core.schemas.common import (
    AudioContent,
    ContentItem,
    ImageContent,
    Message,
    Role,
    TextContent,
    VideoContent,
)
from session_recorder import TurnBasedSessionRecorder, generate_session_id


def _is_worker_ready(worker: Any, *, required_method: str) -> bool:
    if worker is None:
        return False
    if getattr(worker, "processor", None) is not None:
        return True
    return hasattr(worker, required_method)


def _parse_raw_messages(raw_messages: List[dict]) -> List[Message]:
    """Parse frontend raw messages into schema messages."""

    messages: List[Message] = []
    for raw_message in raw_messages:
        role = Role(raw_message["role"])
        content = raw_message["content"]
        if isinstance(content, list):
            content_items: List[ContentItem] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and item.get("text"):
                    content_items.append(TextContent(text=item["text"]))
                elif item.get("type") == "audio" and item.get("data"):
                    content_items.append(AudioContent(data=item["data"]))
                elif item.get("type") == "image" and item.get("data"):
                    content_items.append(ImageContent(data=item["data"]))
                elif item.get("type") == "video" and item.get("data"):
                    content_items.append(VideoContent(
                        data=item["data"],
                        stack_frames=item.get("stack_frames", 1),
                    ))
            if content_items:
                messages.append(Message(role=role, content=content_items))
        else:
            messages.append(Message(role=role, content=content))
    return messages


def _convert_to_model_msgs(schema_messages: List[Message]) -> list:
    """Convert schema messages into the current model message format."""

    from core.processors.base import MiniCPMOProcessorMixin

    mixin = MiniCPMOProcessorMixin()
    model_msgs = []
    for message in schema_messages:
        content = mixin._convert_content_to_model_format(message.content)
        if len(content) == 1 and isinstance(content[0], str):
            content = content[0]
        model_msgs.append({"role": message.role.value, "content": content})
    return model_msgs


def _ws_client_info(ws: WebSocket) -> Dict[str, Any]:
    return {
        "client_id": ws.query_params.get("client_id"),
        "page_session_id": ws.query_params.get("page_session_id"),
        "ip": ws.query_params.get("client_ip"),
        "user_agent": ws.query_params.get("user_agent"),
        "origin": ws.query_params.get("origin"),
    }


def _ws_source_info(ws: WebSocket, default_channel: str, default_mode: Optional[str] = None) -> Dict[str, Any]:
    return {
        "channel": ws.query_params.get("source_channel") or default_channel,
        "mode": ws.query_params.get("source_mode") or default_mode,
        "gateway_session_id": ws.query_params.get("gateway_session_id") or ws.query_params.get("session_id"),
        "path": ws.query_params.get("source_path"),
        "page_route": ws.query_params.get("page_route"),
        "client_surface": ws.query_params.get("client_surface"),
    }


def _record_chat_input_summary(
    raw_messages: List[Dict[str, Any]],
    recorder: Optional[TurnBasedSessionRecorder],
) -> Dict[str, Any]:
    """Build recording input summary and persist user media when recording is enabled."""

    input_summary: Dict[str, Any] = {}
    audio_idx = 0
    video_idx = 0
    for raw_message in raw_messages:
        if raw_message.get("role") != "user":
            continue
        content = raw_message.get("content", "")
        if isinstance(content, str):
            input_summary["text"] = content
            continue
        if not isinstance(content, list):
            continue

        texts = [
            item["text"]
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
        ]
        if texts:
            input_summary["text"] = " ".join(texts)
        if recorder is None:
            continue

        saved_imgs = []
        saved_videos = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image" and item.get("data"):
                try:
                    img_data = base64.b64decode(item["data"])
                    idx = recorder.next_image_index()
                    saved_imgs.append(recorder.save_user_image(idx, img_data))
                except Exception:
                    pass
            elif item.get("type") == "audio" and item.get("data"):
                try:
                    audio_bytes = base64.b64decode(item["data"])
                    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                    input_summary["audio"] = recorder.save_user_audio(audio_idx, audio_np)
                    audio_idx += 1
                except Exception:
                    pass
            elif item.get("type") == "video" and item.get("data"):
                try:
                    video_bytes = base64.b64decode(item["data"])
                    saved_videos.append(recorder.save_user_video(video_idx, video_bytes))
                    video_idx += 1
                except Exception:
                    pass
        if saved_imgs:
            input_summary["images"] = saved_imgs
        if saved_videos:
            input_summary["videos"] = saved_videos

    return input_summary


async def handle_worker_chat_runtime_ws(
    ws: WebSocket,
    *,
    session_id: str,
    worker: Any,
    busy_status: Any,
    idle_status: Any,
    logger: Any,
) -> None:
    """Serve one worker-internal turn-based chat runtime WebSocket."""

    if not _is_worker_ready(worker, required_method="chat_prefill"):
        await ws.close(code=1013, reason="Worker not ready")
        return

    if not worker.state.is_idle:
        await ws.close(code=1013, reason=f"Worker busy: {worker.state.status.value}")
        return

    await ws.accept()
    worker.state.status = busy_status
    worker.state.current_session_id = session_id

    chat_runtime = ChatSessionRuntime(ChatBackendView(worker))
    chat_recorder: Optional[TurnBasedSessionRecorder] = None
    recording_session_id: Optional[str] = None

    try:
        raw = await ws.receive_text()
        msg = json.loads(raw)
        request = parse_worker_chat_request_message(msg)

        messages = _parse_raw_messages(request.messages)
        model_msgs = _convert_to_model_msgs(messages)

        from config import get_config

        cfg = get_config()
        if cfg.recording.enabled:
            recording_session_id = generate_session_id("chat")
            sys_prompt = ""
            for raw_message in request.messages:
                if raw_message.get("role") == "system":
                    content = raw_message.get("content", "")
                    sys_prompt = content if isinstance(content, str) else str(content)
                    break
            chat_recorder = TurnBasedSessionRecorder(
                session_id=recording_session_id,
                app_type="chat",
                worker_id=worker.gpu_id,
                config_snapshot={
                    "system_prompt": sys_prompt,
                    "streaming": request.streaming,
                    "ref_audio": cfg.ref_audio_path,
                },
                client_info=_ws_client_info(ws),
                source_info=_ws_source_info(ws, "demo_turnbased", "chat"),
                data_dir=cfg.data_dir,
            )

        pre_kv = await chat_runtime.prefill(ChatPrefillParams(
            session_id=session_id,
            msgs=model_msgs,
            omni_mode=request.omni_mode,
            max_slice_nums=request.max_slice_nums,
            use_tts_template=request.use_tts_template,
            enable_thinking=request.enable_thinking,
        ))
        await ws.send_json({
            "type": "chat.prefill_done",
            "payload": {"input_tokens": pre_kv},
        })

        if request.generate_audio:
            await chat_runtime.init_tts(request.tts_ref_audio)

        input_summary = _record_chat_input_summary(request.messages, chat_recorder)
        gen_start = time.perf_counter()

        if request.streaming:
            await _stream_chat_response(
                ws,
                worker=worker,
                chat_runtime=chat_runtime,
                request=request,
                session_id=session_id,
                pre_kv=pre_kv,
                gen_start=gen_start,
                chat_recorder=chat_recorder,
                recording_session_id=recording_session_id,
                input_summary=input_summary,
            )
        else:
            await _send_non_streaming_chat_response(
                ws,
                worker=worker,
                chat_runtime=chat_runtime,
                request=request,
                session_id=session_id,
                pre_kv=pre_kv,
                gen_start=gen_start,
                chat_recorder=chat_recorder,
                recording_session_id=recording_session_id,
                input_summary=input_summary,
            )

    except WebSocketDisconnect:
        logger.info("Worker runtime chat WebSocket disconnected")
    except WorkerProtocolError as exc:
        await ws.send_json({"type": "error", "error": str(exc)})
    except Exception as exc:
        logger.error("Worker runtime chat WebSocket error: %s", exc, exc_info=True)
        try:
            await ws.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        if chat_recorder:
            try:
                chat_recorder.finalize()
            except Exception as exc:
                logger.error("[WorkerChat] recorder finalize failed: %s", exc, exc_info=True)
        worker.state.status = idle_status
        worker.state.current_session_id = None
        try:
            await ws.close()
        except Exception:
            pass


async def _stream_chat_response(
    ws: WebSocket,
    *,
    worker: Any,
    chat_runtime: ChatSessionRuntime,
    request: Any,
    session_id: str,
    pre_kv: int,
    gen_start: float,
    chat_recorder: Optional[TurnBasedSessionRecorder],
    recording_session_id: Optional[str],
    input_summary: Dict[str, Any],
) -> None:
    if chat_recorder:
        chat_recorder.start_turn(turn_index=0, request_ts_ms=0.0, input_summary=input_summary)
    chunk_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _run_generate():
        try:
            for chunk in chat_runtime.streaming_generate(ChatGenerateParams(
                session_id=session_id,
                generate_audio=request.generate_audio,
                max_new_tokens=request.max_new_tokens,
                length_penalty=request.length_penalty,
            )):
                loop.call_soon_threadsafe(chunk_queue.put_nowait, ("chunk", chunk))
            loop.call_soon_threadsafe(chunk_queue.put_nowait, ("done", None))
        except Exception as exc:
            loop.call_soon_threadsafe(chunk_queue.put_nowait, ("error", exc))

    gen_task = loop.run_in_executor(None, _run_generate)
    full_text = ""
    chunk_count = 0

    while True:
        tag, payload = await chunk_queue.get()
        if tag == "chunk":
            chunk_payload: Dict[str, Any] = {}
            if payload.text_delta:
                chunk_payload["text_delta"] = payload.text_delta
                full_text += payload.text_delta
            if payload.audio_data:
                chunk_payload["audio_base64"] = payload.audio_data
            if chunk_payload:
                await ws.send_json({"type": "chat.chunk", "payload": chunk_payload})
            if chat_recorder:
                chat_recorder.add_streaming_chunk(
                    text_delta=payload.text_delta,
                    audio_base64=payload.audio_data,
                )
            chunk_count += 1
        elif tag == "done":
            model = getattr(getattr(worker, "processor", None), "model", None)
            gen_ids = getattr(model, "_streaming_generated_token_ids", None)
            generated_tokens = len(gen_ids) if gen_ids else chunk_count
            elapsed = round((time.perf_counter() - gen_start) * 1000, 1)
            if chat_recorder:
                chat_recorder.end_turn(timing={
                    "elapsed_ms": elapsed,
                    "tokens": generated_tokens,
                    "input_tokens": pre_kv,
                })
            await ws.send_json({
                "type": "chat.done",
                "payload": {
                    "text": full_text,
                    "generated_tokens": generated_tokens,
                    "input_tokens": pre_kv,
                    **({"recording_session_id": recording_session_id} if recording_session_id else {}),
                },
            })
            break
        elif tag == "error":
            await ws.send_json({"type": "error", "error": str(payload)})
            break

    try:
        await asyncio.wait_for(gen_task, timeout=5.0)
    except asyncio.TimeoutError:
        pass


async def _send_non_streaming_chat_response(
    ws: WebSocket,
    *,
    worker: Any,
    chat_runtime: ChatSessionRuntime,
    request: Any,
    session_id: str,
    pre_kv: int,
    gen_start: float,
    chat_recorder: Optional[TurnBasedSessionRecorder],
    recording_session_id: Optional[str],
    input_summary: Dict[str, Any],
) -> None:
    result = await chat_runtime.non_streaming_generate(ChatGenerateParams(
        session_id=session_id,
        max_new_tokens=request.max_new_tokens,
        generate_audio=request.generate_audio,
        use_tts_template=request.use_tts_template,
        enable_thinking=request.enable_thinking,
        tts_ref_audio=request.tts_ref_audio,
        length_penalty=request.length_penalty,
    ))

    text = result
    audio_base64 = None
    output_audio_np = None
    if isinstance(result, tuple):
        text, waveform = result
        if waveform is not None:
            output_audio_np = waveform.astype(np.float32)
            audio_base64 = base64.b64encode(output_audio_np.tobytes()).decode("utf-8")

    model = getattr(getattr(worker, "processor", None), "model", None)
    stats = getattr(model, "_last_chat_token_stats", {})
    elapsed = round((time.perf_counter() - gen_start) * 1000, 1)

    if chat_recorder:
        chat_recorder.record_chat_turn(
            turn_index=0,
            request_ts_ms=0.0,
            input_summary=input_summary,
            output_text=text or "",
            output_audio=output_audio_np,
            timing={
                "elapsed_ms": elapsed,
                "tokens": stats.get("generated_tokens", 0),
                "input_tokens": pre_kv,
            },
        )

    await ws.send_json({
        "type": "chat.done",
        "payload": {
            "text": text or "",
            "audio_base64": audio_base64,
            "generated_tokens": stats.get("generated_tokens", 0),
            "input_tokens": pre_kv,
            **({"recording_session_id": recording_session_id} if recording_session_id else {}),
        },
    })


async def handle_worker_duplex_runtime_ws(
    ws: WebSocket,
    *,
    session_id: str,
    worker: Any,
    runtime_manager: RuntimeManager,
    active_status: Any,
    idle_status: Any,
    logger: Any,
) -> None:
    """Serve one worker-internal duplex runtime WebSocket."""

    if not _is_worker_ready(worker, required_method="duplex_prepare"):
        await ws.close(code=1013, reason="Worker not ready")
        return

    if not worker.state.is_idle:
        await ws.close(code=1013, reason=f"Worker busy: {worker.state.status.value}")
        return

    await ws.accept()
    worker.state.status = active_status
    worker.state.current_session_id = session_id

    runtime = runtime_manager.create_duplex(
        session_id,
        DuplexBackendView(worker),
    )
    session_max_slice_nums = 1

    async def _emit_event(event: RuntimeEvent) -> None:
        log_duplex_frame(logger, event.payload["frame"], gpu_id=worker.gpu_id)
        for out_msg in runtime_event_to_worker_messages(event):
            await ws.send_json(out_msg)

    try:
        while True:
            msg = json.loads(await ws.receive_text())
            msg_type = msg.get("type")

            if msg_type == "duplex.session.prepare":
                voice_refs = None
                try:
                    params, voice_refs = parse_worker_prepare_message(msg)
                    prompt = await runtime.prepare(params)
                    session_max_slice_nums = int((params.config or {}).get("max_slice_nums", session_max_slice_nums))
                    await ws.send_json({
                        "type": "duplex.session.ready",
                        "session_id": session_id,
                        "prompt_length": len(prompt),
                    })
                    await runtime.start(_emit_event)
                finally:
                    if voice_refs is not None:
                        voice_refs.cleanup()

            elif msg_type == "duplex.input.audio.append":
                frame = parse_worker_input_message(
                    msg,
                    default_max_slice_nums=session_max_slice_nums,
                    chunk_start=time.perf_counter(),
                )
                await runtime.push_frame(frame)

            elif msg_type in {
                "duplex.control.pause",
                "duplex.control.resume",
                "duplex.control.cancel",
                "duplex.control.close",
            }:
                control = parse_worker_control_message(msg)
                await runtime.push_control(control)
                if control.type == "session.close":
                    runtime_manager.forget_duplex(session_id)
                    break

            else:
                raise WorkerProtocolError(f"unknown worker runtime message type: {msg_type}")

    except WebSocketDisconnect:
        logger.info("Worker runtime duplex WebSocket disconnected")
    except WorkerProtocolError as exc:
        await ws.send_json({"type": "error", "error": str(exc)})
    except Exception as exc:
        logger.error("Worker runtime duplex WebSocket error: %s", exc, exc_info=True)
        try:
            await ws.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await runtime_manager.close_duplex(session_id)
        except Exception as exc:
            logger.error("Worker runtime duplex cleanup failed: %s", exc, exc_info=True)
        worker.state.status = idle_status
        worker.state.current_session_id = None
        worker.state.duplex_pause_time = None
