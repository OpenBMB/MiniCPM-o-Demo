"""MiniCPMO45 推理 Worker

每个 Worker 占用一张 GPU，持有一个 UnifiedProcessor 实例，
提供 Chat (HTTP) / Streaming (WebSocket) / Duplex (WebSocket) 三种推理 API。

启动方式：
    cd /user/sunweiyue/lib/swy-dev/minicpmo45_service
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/base/bin/python worker.py \\
        --port 10031 \\
        --model-path /path/to/base_model \\
        --pt-path /path/to/custom.pt \\
        --ref-audio-path /path/to/ref.wav
"""

import re
import json
import time
import uuid
import asyncio
import argparse
import logging
import base64
import threading
from typing import Optional, List, Dict, Any

import numpy as np
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from core.internal.worker_state import WorkerStatus
from core.processors.pytorch_backend import MiniCPMOWorker
from core.schemas.common import Message, Role, TextContent, AudioContent, ContentItem
from core.schemas.chat import ChatRequest, ChatResponse
from core.schemas.streaming import StreamingRequest
from core.runtime.manager import RuntimeManager
from core.runtime.protocol import DEFAULT_WORKER_CAPABILITIES
from core.runtime.worker_handlers import (
    handle_worker_chat_runtime_ws,
    handle_worker_duplex_runtime_ws,
)
from session_recorder import TurnBasedSessionRecorder, generate_session_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("worker")

# ============ 请求/响应模型 ============

class WorkerHealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    worker_status: WorkerStatus
    gpu_id: int
    model_loaded: bool
    current_session_id: Optional[str] = None
    total_requests: int = 0
    avg_inference_time_ms: float = 0.0
    kv_cache_length: int = 0  # 当前 LLM KV cache token 总数
    capabilities: List[str] = Field(default_factory=list)


# ============ FastAPI 应用 ============

worker: Optional[Any] = None
runtime_manager = RuntimeManager()

# 启动参数（通过 main() 传入）
WORKER_CONFIG: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载模型"""
    global worker
    config = WORKER_CONFIG

    if config.get("backend") == "cpp":
        from core.processors.cpp_backend import CppBackendWorker

        worker = CppBackendWorker(
            gpu_id=config["gpu_id"],
            ref_audio_path=config.get("ref_audio_path"),
            duplex_pause_timeout=config.get("duplex_pause_timeout", 60.0),
            **config.get("cpp_backend", {}),
        )
    else:
        worker = MiniCPMOWorker(
            model_path=config["model_path"],
            gpu_id=config["gpu_id"],
            pt_path=config.get("pt_path"),
            ref_audio_path=config.get("ref_audio_path"),
            duplex_pause_timeout=config.get("duplex_pause_timeout", 60.0),
            compile=config.get("compile", False),
            chat_vocoder=config.get("chat_vocoder", "token2wav"),
            attn_implementation=config.get("attn_implementation", "auto"),
        )

    # 模型加载是同步操作（~15s），在线程中执行避免阻塞
    await asyncio.to_thread(worker.load_model)

    yield

    logger.info("Worker shutting down")
    await runtime_manager.close_all()


app = FastAPI(title="MiniCPMO45 Worker", lifespan=lifespan)


# ========== 健康检查 ==========

@app.get("/health", response_model=WorkerHealthResponse)
async def health():
    """健康检查"""
    if worker is None:
        return WorkerHealthResponse(
            status="initializing",
            worker_status=WorkerStatus.LOADING,
            gpu_id=0,
            model_loaded=False,
            capabilities=[],
        )

    avg_time = 0.0
    if worker.state.total_requests > 0:
        avg_time = worker.state.total_inference_time_ms / worker.state.total_requests

    worker_metrics = worker.metrics()
    kv_len = int(worker_metrics.get("kv_cache_length", 0) or 0)
    model_loaded = worker.processor is not None or bool(getattr(worker.state, "is_idle", False))
    return WorkerHealthResponse(
        status="healthy" if model_loaded else "error",
        worker_status=worker.state.status,
        gpu_id=worker.gpu_id,
        model_loaded=model_loaded,
        current_session_id=worker.state.current_session_id,
        total_requests=worker.state.total_requests,
        avg_inference_time_ms=avg_time,
        kv_cache_length=kv_len,
        capabilities=DEFAULT_WORKER_CAPABILITIES,
    )


# ========== Chat API ==========

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat 推理（无状态）"""
    if worker is None or (worker.processor is None and not hasattr(worker, "chat")):
        raise HTTPException(status_code=503, detail="Worker not ready")

    if not worker.state.is_idle:
        # Gateway 排队机制已保证并发安全，但 Worker 可能还在 cleanup 上一个任务
        # （如 Duplex WS close 后 GPU 资源释放），短暂等待而非立即拒绝
        for _ in range(10):
            await asyncio.sleep(0.5)
            if worker.state.is_idle:
                break
        else:
            raise HTTPException(
                status_code=429,
                detail=f"Worker busy after waiting 5s, status: {worker.state.status.value}",
            )

    # 录制：创建 TurnBasedSessionRecorder
    chat_recorder: Optional[TurnBasedSessionRecorder] = None
    chat_session_id: Optional[str] = None
    from config import get_config
    chat_cfg = get_config()
    if chat_cfg.recording.enabled:
        chat_session_id = generate_session_id("chat")
        sys_prompt = ""
        for m in request.messages:
            if m.role == "system":
                c = m.content
                sys_prompt = c if isinstance(c, str) else str(c)
                break
        chat_recorder = TurnBasedSessionRecorder(
            session_id=chat_session_id,
            app_type="chat",
            worker_id=worker.gpu_id,
            config_snapshot={
                "system_prompt": sys_prompt,
                "ref_audio": chat_cfg.ref_audio_path,
            },
            data_dir=chat_cfg.data_dir,
        )

    try:
        response = await asyncio.to_thread(worker.chat, request)

        # 录制：记录 chat turn
        if chat_recorder and response.success:
            input_summary: Dict[str, Any] = {}
            for m in request.messages:
                if m.role == "user":
                    c = m.content
                    if isinstance(c, str):
                        input_summary["text"] = c
                    elif isinstance(c, list):
                        texts = [it.text for it in c if hasattr(it, "text") and it.text]
                        if texts:
                            input_summary["text"] = " ".join(texts)
            output_audio: Optional[np.ndarray] = None
            if response.audio_data:
                try:
                    audio_bytes = base64.b64decode(response.audio_data)
                    output_audio = np.frombuffer(audio_bytes, dtype=np.float32)
                except Exception:
                    pass
            chat_recorder.record_chat_turn(
                turn_index=0,
                request_ts_ms=0.0,
                input_summary=input_summary,
                output_text=response.text,
                output_audio=output_audio,
                timing={
                    "elapsed_ms": round(response.duration_ms, 1) if response.duration_ms else 0,
                    "tokens": response.tokens_generated or 0,
                },
            )

        if chat_recorder:
            chat_recorder.finalize()

        if chat_session_id and response.success:
            response.recording_session_id = chat_session_id

        return response
    except Exception as e:
        if chat_recorder:
            try:
                chat_recorder.finalize()
            except Exception:
                pass
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/v1/worker/sessions/{session_id}/chat")
async def worker_chat_runtime_ws(ws: WebSocket, session_id: str):
    """Worker-internal turn-based chat runtime protocol."""
    await handle_worker_chat_runtime_ws(
        ws,
        session_id=_sanitize_session_id(session_id),
        worker=worker,
        busy_status=WorkerStatus.BUSY_CHAT,
        idle_status=WorkerStatus.IDLE,
        logger=logger,
    )


# ========== Half-Duplex Stop 信号（每连接独立） ==========
# 每个 WS 连接创建独立的 threading.Event()，按 session_id 索引。
# HTTP POST /half_duplex/stop 广播到所有活跃 session。
# 安全性：
#   - dict 操作在 asyncio 单线程事件循环中，无并发写入
#   - threading.Event 本身线程安全（asyncio 线程 ↔ generate 工作线程）
_SESSION_ID_RE = re.compile(r'^[a-zA-Z0-9_\-]+$')


def _sanitize_session_id(session_id: str) -> str:
    """校验 session_id 只含安全字符，防止 path traversal"""
    if not _SESSION_ID_RE.match(session_id):
        safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', session_id)
        return safe
    return session_id


def _ws_client_info(ws: WebSocket) -> Dict[str, Any]:
    """Best-effort client/page identity captured from gateway-forwarded query params."""
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


_half_duplex_stop_events: Dict[str, threading.Event] = {}

_half_duplex_ref_audio_cache: Dict[str, np.ndarray] = {}


@app.post("/half_duplex/stop")
async def half_duplex_stop():
    """停止所有正在进行的 Half-Duplex 生成"""
    if not _half_duplex_stop_events:
        return {"success": False, "message": "No active half-duplex session"}
    for sid, evt in _half_duplex_stop_events.items():
        evt.set()
        logger.info(f"Half-Duplex stop signal sent to session {sid}")
    return {"success": True, "message": f"Stop signal sent to {len(_half_duplex_stop_events)} session(s)"}


# ========== Half-Duplex WebSocket ==========

@app.websocket("/ws/half_duplex")
async def half_duplex_ws(ws: WebSocket):
    """Half-Duplex Audio WebSocket

    协议：
    1. Client → {"type": "prepare", "system_content": [...], "config": {...}}
       system_content 格式与 turn-based 相同: [{type:"text",text:...}, {type:"audio",data:...}, ...]
    2. Client → {"type": "audio_chunk", "audio_base64": "..."} (连续)
    3. Server → {"type": "vad_state", "speaking": true/false}
    4. Server → {"type": "generating"}
    5. Server → {"type": "chunk", ...} (流式)
    6. Server → {"type": "turn_done", ...}
    7. Client → {"type": "stop"} / Server → {"type": "timeout"}
    """
    if worker is None or (worker.processor is None and not hasattr(worker, "half_duplex_prefill")):
        await ws.close(code=1013, reason="Worker not ready")
        return

    await ws.accept()
    conn_id = uuid.uuid4().hex[:8]
    session_id = ws.query_params.get("session_id", f"hdx_{conn_id}")
    logger.info(f"Half-Duplex WS connected (conn={conn_id}, session={session_id})")

    worker.state.status = WorkerStatus.BUSY_HALF_DUPLEX
    worker.state.current_session_id = session_id

    from vad import StreamingVAD, VadOptions

    vad: Optional[StreamingVAD] = None
    turn_index = 0
    session_start = time.perf_counter()
    timeout_s = 300
    generate_audio = True
    max_new_tokens = 256
    length_penalty = 1.1
    temperature = 0.7
    is_generating = False
    stop_event = threading.Event()

    vad_armed_at: float = 0.0
    INITIAL_GUARD_S = 0.5
    _half_duplex_stop_events[conn_id] = stop_event

    hdx_recorder: Optional[TurnBasedSessionRecorder] = None
    hdx_session_start_perf: float = 0.0

    try:
        while True:
            elapsed_s = time.perf_counter() - session_start
            if elapsed_s > timeout_s:
                await ws.send_json({"type": "timeout", "elapsed_s": round(elapsed_s, 1)})
                logger.info(f"Half-Duplex session timeout after {elapsed_s:.0f}s")
                break

            remaining = timeout_s - elapsed_s
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=min(remaining, 5.0))
            except asyncio.TimeoutError:
                continue

            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "prepare":
                config = msg.get("config", {})

                vad_cfg = config.get("vad", {})
                vad_options = VadOptions(
                    threshold=vad_cfg.get("threshold", 0.7),
                    min_speech_duration_ms=vad_cfg.get("min_speech_duration_ms", 128),
                    min_silence_duration_ms=vad_cfg.get("min_silence_duration_ms", 500),
                    speech_pad_ms=vad_cfg.get("speech_pad_ms", 30),
                )
                vad = StreamingVAD(options=vad_options)

                gen_cfg = config.get("generation", {})
                max_new_tokens = gen_cfg.get("max_new_tokens", 256)
                length_penalty = gen_cfg.get("length_penalty", 1.1)
                temperature = gen_cfg.get("temperature", 0.7)

                tts_cfg = config.get("tts", {})
                generate_audio = tts_cfg.get("enabled", True)

                session_cfg = config.get("session", {})
                timeout_s = session_cfg.get("timeout_s", 300)
                session_start = time.perf_counter()

                worker.reset_half_duplex_session()

                # 解析 system_content 列表（与 turn-based 相同的 schema）
                ref_audio_ndarray: Optional[np.ndarray] = None
                system_content_items = msg.get("system_content", [])
                content_items: List[ContentItem] = []
                for item in system_content_items:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text" and item.get("text"):
                        content_items.append(TextContent(text=item["text"]))
                    elif item.get("type") == "audio" and item.get("data"):
                        content_items.append(AudioContent(data=item["data"]))
                        if ref_audio_ndarray is None:
                            try:
                                audio_bytes = base64.b64decode(item["data"])
                                ref_audio_ndarray = np.frombuffer(audio_bytes, dtype=np.float32)
                            except Exception:
                                pass
                if content_items:
                    sys_msg = Message(role=Role.SYSTEM, content=content_items)
                else:
                    sys_msg = Message(role=Role.SYSTEM, content="You are a helpful assistant.")
                logger.info(f"[HalfDuplex] system_content: {len(content_items)} items")

                request = StreamingRequest(
                    session_id=session_id,
                    messages=[sys_msg],
                    is_last_chunk=True,
                    use_tts_template=generate_audio,
                )
                await asyncio.to_thread(worker.half_duplex_prefill, request)

                if generate_audio:
                    if ref_audio_ndarray is not None:
                        await asyncio.to_thread(worker.half_duplex_init_tts, ref_audio_ndarray)
                    elif worker.ref_audio_path:
                        await asyncio.to_thread(worker.half_duplex_init_tts)

                vad_armed_at = time.perf_counter()
                hdx_session_start_perf = time.perf_counter()

                from config import get_config
                hdx_cfg = get_config()
                if hdx_cfg.recording.enabled:
                    hdx_recorder = TurnBasedSessionRecorder(
                        session_id=session_id,
                        app_type="half_duplex_audio",
                        worker_id=worker.gpu_id,
                        config_snapshot={
                            "system_content_count": len(content_items),
                            "vad": vad_cfg,
                            "generation": gen_cfg,
                            "tts_enabled": generate_audio,
                            "timeout_s": timeout_s,
                        },
                        client_info=_ws_client_info(ws),
                        source_info=_ws_source_info(ws, "demo_half_duplex", "audio"),
                        data_dir=hdx_cfg.data_dir,
                    )

                rec_sid = hdx_recorder.session_id if hdx_recorder else None
                await ws.send_json({
                    "type": "prepared",
                    "session_id": session_id,
                    "timeout_s": timeout_s,
                    **({"recording_session_id": rec_sid} if rec_sid else {}),
                })
                logger.info(f"[HalfDuplex] prepared: timeout={timeout_s}s, vad_threshold={vad_options.threshold}")

            elif msg_type == "audio_chunk":
                if vad is None:
                    await ws.send_json({"type": "error", "error": "Not prepared yet"})
                    continue
                if is_generating:
                    continue

                if time.perf_counter() - vad_armed_at < INITIAL_GUARD_S:
                    continue

                audio_b64 = msg.get("audio_base64", "")
                if not audio_b64:
                    continue

                audio_bytes = base64.b64decode(audio_b64)
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)

                was_speaking = vad.is_speaking
                speech_segment = vad.feed(audio_chunk)

                if vad.is_speaking and not was_speaking:
                    await ws.send_json({"type": "vad_state", "speaking": True})
                elif not vad.is_speaking and was_speaking and speech_segment is None:
                    await ws.send_json({"type": "vad_state", "speaking": False})

                if speech_segment is not None:
                    await ws.send_json({"type": "vad_state", "speaking": False})
                    await ws.send_json({
                        "type": "generating",
                        "speech_duration_ms": round(len(speech_segment) / 16000 * 1000),
                    })
                    is_generating = True

                    try:
                        speech_duration_ms = round(len(speech_segment) / 16000 * 1000)
                        request_ts_ms = (time.perf_counter() - hdx_session_start_perf) * 1000

                        if hdx_recorder:
                            user_audio_rel = hdx_recorder.save_user_audio(turn_index, speech_segment)
                            hdx_recorder.start_turn(
                                turn_index=turn_index,
                                request_ts_ms=request_ts_ms,
                                input_summary={
                                    "type": "voice",
                                    "duration_ms": speech_duration_ms,
                                    "audio": user_audio_rel,
                                },
                            )

                        audio_b64_data = base64.b64encode(speech_segment.tobytes()).decode('utf-8')
                        user_msg = Message(
                            role=Role.USER,
                            content=[AudioContent(data=audio_b64_data)],
                        )
                        prefill_request = StreamingRequest(
                            session_id=session_id,
                            messages=[user_msg],
                            is_last_chunk=True,
                            use_tts_template=generate_audio,
                        )
                        await asyncio.to_thread(worker.half_duplex_prefill, prefill_request)

                        chunk_queue: asyncio.Queue = asyncio.Queue()
                        loop = asyncio.get_event_loop()
                        stop_event.clear()
                        gen_start = time.perf_counter()

                        def _run_generate():
                            try:
                                for chunk in worker.half_duplex_generate(
                                    session_id=session_id,
                                    generate_audio=generate_audio,
                                    max_new_tokens=max_new_tokens,
                                    length_penalty=length_penalty,
                                ):
                                    loop.call_soon_threadsafe(chunk_queue.put_nowait, ("chunk", chunk))
                                    if stop_event.is_set():
                                        break
                                loop.call_soon_threadsafe(chunk_queue.put_nowait, ("done", None))
                            except Exception as e:
                                loop.call_soon_threadsafe(chunk_queue.put_nowait, ("error", e))

                        gen_task = loop.run_in_executor(None, _run_generate)

                        full_text = ""
                        while True:
                            item_type, payload = await chunk_queue.get()
                            if item_type == "chunk":
                                await ws.send_json({"type": "chunk", **payload.model_dump()})
                                if payload.text_delta:
                                    full_text += payload.text_delta
                                if hdx_recorder:
                                    hdx_recorder.add_streaming_chunk(
                                        text_delta=payload.text_delta,
                                        audio_base64=payload.audio_data,
                                    )
                            elif item_type == "done":
                                break
                            elif item_type == "error":
                                raise payload

                        await gen_task

                        gen_elapsed_ms = (time.perf_counter() - gen_start) * 1000
                        if hdx_recorder:
                            hdx_recorder.end_turn(timing={
                                "elapsed_ms": round(gen_elapsed_ms, 1),
                                "speech_input_ms": speech_duration_ms,
                            })

                        turn_index += 1
                        await ws.send_json({
                            "type": "turn_done",
                            "turn_index": turn_index,
                            "text": full_text,
                        })

                        vad.reset()
                        logger.info(f"[HalfDuplex] turn {turn_index} done, VAD reset")

                    except Exception as e:
                        logger.error(f"[HalfDuplex] generate failed: {e}", exc_info=True)
                        await ws.send_json({"type": "error", "error": str(e)})
                    finally:
                        is_generating = False

            elif msg_type == "stop":
                stop_event.set()
                logger.info(f"Half-Duplex stop requested (conn={conn_id})")
                break

            else:
                await ws.send_json({"type": "error", "error": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info(f"Half-Duplex WS disconnected (conn={conn_id})")
    except Exception as e:
        logger.error(f"Half-Duplex WS error (conn={conn_id}): {e}", exc_info=True)
    finally:
        if hdx_recorder:
            try:
                hdx_recorder.finalize()
            except Exception as e:
                logger.error(f"[HalfDuplex] recorder finalize failed: {e}", exc_info=True)

        stop_event.set()
        _half_duplex_stop_events.pop(conn_id, None)
        if worker.state.status == WorkerStatus.BUSY_HALF_DUPLEX:
            worker.state.status = WorkerStatus.IDLE
            worker.state.current_session_id = None
            logger.info(f"Half-Duplex session ended (conn={conn_id}, turns={turn_index})")


# ========== Duplex WebSocket ==========

@app.websocket("/v1/worker/sessions/{session_id}/duplex")
async def worker_duplex_runtime_ws(ws: WebSocket, session_id: str):
    """Worker-internal duplex runtime protocol.

    This endpoint is meant for gateway-worker communication and uses runtime
    event payloads instead of page/demo-shaped result messages.
    """
    await handle_worker_duplex_runtime_ws(
        ws,
        session_id=_sanitize_session_id(session_id),
        worker=worker,
        runtime_manager=runtime_manager,
        active_status=WorkerStatus.DUPLEX_ACTIVE,
        idle_status=WorkerStatus.IDLE,
        logger=logger,
    )


# ============ 缓存状态查询 ==========

@app.get("/cache_info")
async def cache_info():
    """查询当前 Worker 的 KV Cache 状态"""
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not ready")

    return {
        "status": worker.state.status.value,
        "note": "KV cache state is now tracked by Gateway (cached_hash on WorkerConnection)",
    }


@app.post("/clear_cache")
async def clear_cache():
    """手动清除 KV Cache（重置 Streaming 模型 session）"""
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not ready")

    worker.reset_half_duplex_session()
    return {"success": True, "message": "Cache cleared"}


# ============ 入口 ============

def main():
    from config import get_config
    cfg = get_config()

    parser = argparse.ArgumentParser(description="MiniCPMO45 Worker")
    parser.add_argument("--port", type=int, default=None, help=f"Worker port (default: from config, base={cfg.worker_base_port})")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--model-path", type=str, default=None, help="Base model path")
    parser.add_argument("--pt-path", type=str, default=None, help="Custom weights path (.pt)")
    parser.add_argument("--ref-audio-path", type=str, default=None, help="Default ref audio path")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU ID (inferred from port if not set)")
    parser.add_argument("--worker-index", type=int, default=0, help="Worker index (0, 1, 2, ...)")
    parser.add_argument("--duplex-pause-timeout", type=float, default=None, help="Duplex pause timeout (s)")
    parser.add_argument("--backend", choices=("pytorch", "cpp"), default=None, help="Override inference backend")
    args = parser.parse_args()

    port = args.port or cfg.worker_port(args.worker_index)
    gpu_id = args.gpu_id if args.gpu_id is not None else args.worker_index
    backend = args.backend or cfg.backend

    WORKER_CONFIG.update({
        "backend": backend,
        "model_path": args.model_path or cfg.model.model_path,
        "gpu_id": gpu_id,
        "pt_path": args.pt_path or cfg.model.pt_path,
        "ref_audio_path": args.ref_audio_path or cfg.ref_audio_path,
        "duplex_pause_timeout": args.duplex_pause_timeout or cfg.duplex_pause_timeout,
        "compile": cfg.compile,
        "chat_vocoder": cfg.chat_vocoder,
        "attn_implementation": cfg.attn_implementation,
        "cpp_backend": {
            **cfg.cpp_backend.model_dump(),
            "cpp_server_port": (
                (cfg.cpp_backend.cpp_server_port + args.worker_index)
                if cfg.cpp_backend.cpp_server_port is not None
                else None
            ),
        },
    })

    logger.info(f"Starting Worker on port {port}, GPU {gpu_id}")
    # Bump WS max payload from uvicorn's 16 MiB default to 128 MiB so that
    # base64-encoded video attachments (commonly 30-60 MiB after inflation)
    # can be received without the connection being torn down with code 1009.
    uvicorn.run(app, host=args.host, port=port, ws_max_size=128 * 1024 * 1024)


if __name__ == "__main__":
    main()
