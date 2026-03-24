"""MiniCPMO45 推理 Gateway

请求分发网关，不加载模型，负责：
- 路由 Chat/Streaming/Duplex 请求到 Worker
- 会话映射和 KV Cache LRU 命中路由
- 统一 FIFO 请求排队（容量 1000，位置追踪 + ETA 估算）
- Worker 健康检查

启动方式：
    cd /user/sunweiyue/lib/swy-dev/minicpmo45_service
    PYTHONPATH=. .venv/base/bin/python gateway.py \\
        --port 10024 \\
        --workers localhost:22400,localhost:22401
"""

import os
import re
import json
import asyncio
import argparse
import logging
import time
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

import zipfile
from io import BytesIO

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles

from gateway_modules.models import (
    GatewayWorkerStatus,
    ServiceStatus,
    WorkersResponse,
    QueueStatus,
    EtaConfig,
    EtaStatus,
)
from gateway_modules.worker_pool import WorkerPool, WorkerConnection
from gateway_modules.ref_audio_registry import (
    RefAudioRegistry,
    RefAudioListResponse,
    UploadRefAudioRequest,
    RefAudioResponse,
)
from gateway_modules.app_registry import (
    AppRegistry,
    AppToggleRequest,
    AppsPublicResponse,
    AppsAdminResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gateway")


_SESSION_ID_RE = re.compile(r'^[a-zA-Z0-9_\-]+$')


def _sanitize_session_id(session_id: str) -> str:
    """校验 session_id 只含安全字符，防止 path traversal"""
    if not _SESSION_ID_RE.match(session_id):
        safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', session_id)
        return safe
    return session_id


# ============ 全局变量 ============

worker_pool: Optional[WorkerPool] = None
ref_audio_registry: Optional[RefAudioRegistry] = None
app_registry: AppRegistry = AppRegistry()

# 配置（通过 main() 传入）
GATEWAY_CONFIG: Dict[str, Any] = {}


# ============ 应用初始化 ============

_cleanup_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    global worker_pool, ref_audio_registry, _cleanup_task

    workers = GATEWAY_CONFIG.get("workers", ["localhost:10031"])
    max_queue = GATEWAY_CONFIG.get("max_queue_size", 1000)
    timeout = GATEWAY_CONFIG.get("timeout", 300.0)

    # 从 config 读取 ETA 参数
    eta_config_data = GATEWAY_CONFIG.get("eta_config")
    eta_config = EtaConfig(**eta_config_data) if eta_config_data else EtaConfig()

    worker_pool = WorkerPool(
        worker_addresses=workers,
        max_queue_size=max_queue,
        request_timeout=timeout,
        eta_config=eta_config,
        ema_alpha=GATEWAY_CONFIG.get("eta_ema_alpha", 0.3),
        ema_min_samples=GATEWAY_CONFIG.get("eta_ema_min_samples", 3),
    )
    await worker_pool.start()

    # 初始化参考音频注册表
    data_dir = os.path.join(os.path.dirname(__file__), "data", "assets", "ref_audio")
    ref_audio_registry = RefAudioRegistry(storage_dir=data_dir)

    # 启动 session 清理后台任务（每天一次）
    _cleanup_task = asyncio.create_task(_session_cleanup_loop())

    logger.info(f"Gateway started, {len(worker_pool.workers)} workers, {ref_audio_registry.count} ref audios")

    yield

    if _cleanup_task:
        _cleanup_task.cancel()
    await worker_pool.stop()
    logger.info("Gateway stopped")


async def _session_cleanup_loop() -> None:
    """每天执行一次 session 清理（retention_days 和 max_storage_gb 都为 -1 时不执行）"""
    from session_cleanup import cleanup_sessions
    from config import get_config

    await asyncio.sleep(60)
    while True:
        try:
            cfg = get_config()
            days = cfg.recording.session_retention_days
            gb = cfg.recording.max_storage_gb
            if days < 0 and gb < 0:
                logger.info("[Cleanup] Disabled (retention_days=-1, max_storage_gb=-1), sleeping")
            else:
                report = await asyncio.to_thread(
                    cleanup_sessions, cfg.data_dir, days, gb,
                )
                logger.info(f"[Cleanup] {report}")
        except Exception as e:
            logger.error(f"[Cleanup] Failed: {e}", exc_info=True)
        await asyncio.sleep(86400)


app = FastAPI(
    title="MiniCPMO45 Gateway",
    description="MiniCPMO45 多模态推理网关",
    version="1.0.0-alpha.2",
    lifespan=lifespan,
)


# ============ 健康检查 ============

@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/status", response_model=ServiceStatus)
async def status():
    """服务状态"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    return ServiceStatus(
        gateway_healthy=True,
        total_workers=len(worker_pool.workers),
        idle_workers=worker_pool.idle_count,
        busy_workers=worker_pool.busy_count,
        duplex_workers=worker_pool.duplex_count,
        loading_workers=worker_pool.loading_count,
        error_workers=worker_pool.error_count,
        offline_workers=worker_pool.offline_count,
        queue_length=worker_pool.queue_length,
        max_queue_size=worker_pool.max_queue_size,
        running_tasks=worker_pool._get_running_tasks(),
    )


@app.get("/workers", response_model=WorkersResponse)
async def list_workers():
    """Worker 列表"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    return WorkersResponse(
        total=len(worker_pool.workers),
        workers=worker_pool.get_all_workers(),
    )


# ============ Chat API（无状态，HTTP 代理到 Worker） ============

@app.post("/api/chat")
async def chat(request: Request):
    """Chat 推理

    无状态，路由到任意空闲 Worker。
    如果无空闲 Worker，入 FIFO 队列等待。
    """
    if not app_registry.is_enabled("turnbased"):
        raise HTTPException(status_code=403, detail="Turn-based Chat is currently disabled")
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    request_body = await request.json()
    queue_start = datetime.now()

    # 入队（如果有空闲 Worker 会立即分配）
    try:
        ticket, future = worker_pool.enqueue("chat")
    except WorkerPool.QueueFullError:
        raise HTTPException(
            status_code=503,
            detail=f"Queue full ({worker_pool.max_queue_size} requests)",
        )

    # 等待 Worker 分配（同时检测客户端断开）
    worker: Optional[WorkerConnection] = None
    try:
        if future.done():
            worker = future.result()
        else:
            # 排队等待，定期检查客户端是否断开
            while not future.done():
                if await request.is_disconnected():
                    worker_pool.cancel(ticket.ticket_id)
                    return  # 客户端已断开
                try:
                    worker = await asyncio.wait_for(
                        asyncio.shield(future), timeout=2.0
                    )
                    break
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    raise HTTPException(status_code=503, detail="Request cancelled")
            if worker is None and future.done():
                worker = future.result()
    except asyncio.CancelledError:
        raise HTTPException(status_code=503, detail="Request cancelled")

    if worker is None:
        raise HTTPException(status_code=503, detail="No worker available")

    # 标记 Worker 为 BUSY
    queue_done_time = datetime.now()
    queue_wait_ms = (queue_done_time - queue_start).total_seconds() * 1000
    estimated_queue_s = ticket.estimated_wait_s

    worker.mark_busy(GatewayWorkerStatus.BUSY_CHAT, "chat")
    task_start = datetime.now()

    try:
        async with httpx.AsyncClient(timeout=worker_pool.request_timeout) as client:
            resp = await client.post(
                f"{worker.url}/chat",
                json=request_body,
                timeout=worker_pool.request_timeout,
            )

        worker.total_requests += 1
        worker.last_heartbeat = datetime.now()

        result = resp.json()
        result["queue_wait_ms"] = round(queue_wait_ms)
        result["estimated_queue_wait_s"] = round(estimated_queue_s, 1)
        return result

    except Exception as e:
        logger.error(f"[{ticket.ticket_id}] Chat request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        duration = (datetime.now() - task_start).total_seconds()
        worker_pool.release_worker(worker, request_type="chat", duration_s=duration)


# ============ Chat WebSocket 代理 ============

@app.websocket("/ws/chat")
async def chat_ws_proxy(ws: WebSocket):
    """Chat WebSocket 代理 — 排队后透传到 Worker /ws/chat"""
    if not app_registry.is_enabled("turnbased"):
        await ws.close(code=1008, reason="Turn-based Chat is currently disabled")
        return
    if worker_pool is None:
        await ws.close(code=1013, reason="Service not ready")
        return

    await ws.accept()

    assigned_worker: Optional[WorkerConnection] = None
    worker_ws = None
    task_start: Optional[datetime] = None

    try:
        # 收到前端发来的请求消息
        raw = await ws.receive_text()

        # 排队获取 Worker
        try:
            ticket, future = worker_pool.enqueue("chat")
        except WorkerPool.QueueFullError:
            await ws.send_json({"type": "error", "error": "Queue full"})
            return

        # 等待 Worker 分配
        while not future.done():
            try:
                assigned_worker = await asyncio.wait_for(asyncio.shield(future), timeout=2.0)
                break
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                await ws.send_json({"type": "error", "error": "Cancelled"})
                return
        if assigned_worker is None and future.done():
            assigned_worker = future.result()

        if assigned_worker is None:
            await ws.send_json({"type": "error", "error": "No worker available"})
            return

        assigned_worker.mark_busy(GatewayWorkerStatus.BUSY_CHAT, "chat_ws")
        task_start = datetime.now()

        # 连接 Worker WebSocket
        import websockets
        ws_url = f"ws://{assigned_worker.host}:{assigned_worker.port}/ws/chat"
        worker_ws = await websockets.connect(ws_url)

        # 转发请求
        await worker_ws.send(raw)

        # 透传 Worker 的所有响应到前端
        async for msg_data in worker_ws:
            await ws.send_text(msg_data)

    except WebSocketDisconnect:
        logger.info("Chat WS proxy: client disconnected")
    except Exception as e:
        logger.error(f"Chat WS proxy error: {e}", exc_info=True)
        try:
            await ws.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        if worker_ws:
            try:
                await worker_ws.close()
            except Exception:
                pass
        if assigned_worker and task_start:
            duration = (datetime.now() - task_start).total_seconds()
            worker_pool.release_worker(assigned_worker, request_type="chat_ws", duration_s=duration)
        try:
            await ws.close()
        except Exception:
            pass


# ============ Half-Duplex WebSocket（独占 Worker，FIFO 排队 + 代理到 Worker） ============

@app.websocket("/ws/half_duplex/{session_id}")
async def half_duplex_ws(ws: WebSocket, session_id: str):
    """Half-Duplex WebSocket 代理

    独占一个 Worker，直到用户停止或会话超时（默认 3 分钟）。
    前端发送音频 chunk，Worker 用 VAD 检测语音后 prefill + generate。
    """
    if not app_registry.is_enabled("half_duplex_audio"):
        await ws.close(code=1008, reason="Half-Duplex Audio is currently disabled")
        return
    if worker_pool is None:
        await ws.close(code=1013, reason="Service not ready")
        return

    session_id = _sanitize_session_id(session_id)
    await ws.accept()

    try:
        ticket, future = worker_pool.enqueue("half_duplex_audio", session_id=session_id)
    except WorkerPool.QueueFullError:
        await ws.send_json({
            "type": "error",
            "error": f"Queue full ({worker_pool.max_queue_size} requests)",
        })
        await ws.close(code=1013, reason="Queue full")
        return

    worker: Optional[WorkerConnection] = None
    if future.done():
        worker = future.result()
    else:
        try:
            await ws.send_json({
                "type": "queued",
                "position": ticket.position,
                "estimated_wait_s": ticket.estimated_wait_s,
                "ticket_id": ticket.ticket_id,
                "queue_length": worker_pool.queue_length,
            })
            while not future.done():
                try:
                    worker = await asyncio.wait_for(
                        asyncio.shield(future), timeout=3.0
                    )
                    break
                except asyncio.TimeoutError:
                    updated = worker_pool.get_ticket(ticket.ticket_id)
                    if updated:
                        await ws.send_json({
                            "type": "queue_update",
                            "position": updated.position,
                            "estimated_wait_s": updated.estimated_wait_s,
                            "queue_length": worker_pool.queue_length,
                        })
                except asyncio.CancelledError:
                    worker_pool.cancel(ticket.ticket_id)
                    return
        except (WebSocketDisconnect, Exception) as e:
            logger.info(f"Half-Duplex WS disconnected during queue: session={session_id} ({e})")
            worker_pool.cancel(ticket.ticket_id)
            return
        if worker is None and future.done():
            worker = future.result()

    if worker is None:
        await ws.send_json({"type": "error", "error": "No worker available"})
        await ws.close(code=1013, reason="No worker available")
        return

    await ws.send_json({"type": "queue_done"})
    logger.info(f"Half-Duplex WS connected: session={session_id} → {worker.worker_id}")

    worker.mark_busy(GatewayWorkerStatus.BUSY_HALF_DUPLEX, "half_duplex_audio", session_id=session_id)
    task_start = datetime.now()

    worker_ws = None

    try:
        import websockets
        ws_url = f"ws://{worker.host}:{worker.port}/ws/half_duplex?session_id={session_id}"

        max_retries = 5
        for attempt in range(max_retries):
            try:
                worker_ws = await websockets.connect(ws_url, open_timeout=5)
                break
            except Exception as conn_err:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Half-Duplex WS connect to {worker.worker_id} failed (attempt {attempt + 1}): "
                        f"{conn_err}, retrying in 1s..."
                    )
                    await asyncio.sleep(1.0)
                else:
                    raise

        async def client_to_worker():
            try:
                async for raw in ws.iter_text():
                    await worker_ws.send(raw)
            except WebSocketDisconnect:
                pass

        async def worker_to_client():
            try:
                async for raw in worker_ws:
                    await ws.send_text(raw)
            except Exception:
                pass

        done, pending = await asyncio.wait(
            [
                asyncio.create_task(client_to_worker()),
                asyncio.create_task(worker_to_client()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

    except Exception as e:
        logger.error(f"Half-Duplex WS error: {e}", exc_info=True)
    finally:
        if worker_ws:
            try:
                await worker_ws.close()
            except Exception:
                pass

        if worker:
            duration = (datetime.now() - task_start).total_seconds() if task_start else 0
            worker_pool.release_worker(worker, request_type="half_duplex_audio", duration_s=duration)
            logger.info(f"Half-Duplex WS ended: session={session_id}, Worker released ({duration:.1f}s)")


# ============ 前端诊断日志写入 ============

async def _write_diagnostic(path: str, msg: dict) -> None:
    """将前端上报的诊断数据追加写入 JSONL 文件（异步，不阻塞事件循环）"""
    msg["_server_recv_ts"] = time.time()
    line = json.dumps(msg, ensure_ascii=False) + "\n"
    try:
        await asyncio.to_thread(_sync_append, path, line)
    except Exception as e:
        logger.warning(f"Failed to write diagnostic: {e}")


def _sync_append(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


# ============ Duplex WebSocket（有状态，FIFO 排队 + 代理到 Worker） ============

@app.websocket("/ws/duplex/{session_id}")
async def duplex_ws(ws: WebSocket, session_id: str):
    """Duplex WebSocket 代理

    先 accept WS（以便推送排队状态），然后入 FIFO 队列等待 Worker。
    Duplex 独占一个 Worker，直到用户挂断或暂停超时。
    """
    duplex_app = "audio_duplex" if session_id.startswith("adx_") else "omni"
    if not app_registry.is_enabled(duplex_app):
        await ws.close(code=1008, reason=f"{duplex_app} is currently disabled")
        return

    if worker_pool is None:
        await ws.close(code=1013, reason="Service not ready")
        return

    session_id = _sanitize_session_id(session_id)

    # 先 accept，这样排队期间可以推送状态
    await ws.accept()

    # 入队
    try:
        duplex_type = "audio_duplex" if session_id.startswith("adx_") else "omni_duplex"
        ticket, future = worker_pool.enqueue(duplex_type, session_id=session_id)
    except WorkerPool.QueueFullError:
        await ws.send_json({
            "type": "error",
            "error": f"Queue full ({worker_pool.max_queue_size} requests)",
        })
        await ws.close(code=1013, reason="Queue full")
        return

    # 等待 Worker 分配（排队期间检测前端断连，断连时取消 ticket）
    worker: Optional[WorkerConnection] = None
    if future.done():
        worker = future.result()
    else:
        try:
            await ws.send_json({
                "type": "queued",
                "position": ticket.position,
                "estimated_wait_s": ticket.estimated_wait_s,
                "ticket_id": ticket.ticket_id,
                "queue_length": worker_pool.queue_length,
            })
            while not future.done():
                try:
                    worker = await asyncio.wait_for(
                        asyncio.shield(future), timeout=3.0
                    )
                    break
                except asyncio.TimeoutError:
                    updated = worker_pool.get_ticket(ticket.ticket_id)
                    if updated:
                        await ws.send_json({
                            "type": "queue_update",
                            "position": updated.position,
                            "estimated_wait_s": updated.estimated_wait_s,
                            "queue_length": worker_pool.queue_length,
                        })
                except asyncio.CancelledError:
                    worker_pool.cancel(ticket.ticket_id)
                    return
        except (WebSocketDisconnect, Exception) as e:
            logger.info(f"Duplex WS disconnected during queue wait: session={session_id}, cancelling ticket {ticket.ticket_id} ({e})")
            worker_pool.cancel(ticket.ticket_id)
            return
        if worker is None and future.done():
            worker = future.result()

    if worker is None:
        await ws.send_json({"type": "error", "error": "No worker available"})
        await ws.close(code=1013, reason="No worker available")
        return

    # 通知前端排队完成
    await ws.send_json({"type": "queue_done"})
    logger.info(f"Duplex WS connected: session={session_id} → {worker.worker_id}")

    worker.mark_busy(GatewayWorkerStatus.DUPLEX_ACTIVE, duplex_type, session_id=session_id)
    task_start = datetime.now()

    worker_ws = None

    try:
        import websockets
        ws_url = f"ws://{worker.host}:{worker.port}/ws/duplex?session_id={session_id}"

        # Worker 可能在清理上一个 Duplex session（GPU 显存释放等），
        # 短暂重试确保 Worker 准备就绪
        max_retries = 5
        for attempt in range(max_retries):
            try:
                worker_ws = await websockets.connect(ws_url, open_timeout=5)
                break
            except Exception as conn_err:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Duplex WS connect to {worker.worker_id} failed (attempt {attempt + 1}): "
                        f"{conn_err}, retrying in 1s..."
                    )
                    await asyncio.sleep(1.0)
                else:
                    raise

        diag_log_path = os.path.join("tmp", f"diag_{session_id}.jsonl")

        async def client_to_worker():
            """Client → Worker"""
            try:
                async for raw in ws.iter_text():
                    msg = json.loads(raw)

                    if msg.get("type") == "client_diagnostic":
                        await _write_diagnostic(diag_log_path, msg)
                        continue

                    if msg.get("type") == "pause":
                        worker.update_duplex_status(GatewayWorkerStatus.DUPLEX_PAUSED)
                    elif msg.get("type") == "resume":
                        worker.update_duplex_status(GatewayWorkerStatus.DUPLEX_ACTIVE)
                    elif msg.get("type") == "stop":
                        pass

                    await worker_ws.send(raw)
            except WebSocketDisconnect:
                pass

        async def worker_to_client():
            """Worker → Client"""
            try:
                async for raw in worker_ws:
                    await ws.send_text(raw)
            except Exception:
                pass

        done, pending = await asyncio.wait(
            [
                asyncio.create_task(client_to_worker()),
                asyncio.create_task(worker_to_client()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

    except Exception as e:
        logger.error(f"Duplex WS error: {e}", exc_info=True)
    finally:
        if worker_ws:
            try:
                await worker_ws.close()
            except Exception:
                pass

        if worker:
            duration = (datetime.now() - task_start).total_seconds() if task_start else 0
            # Duplex 结束后 Worker 端会执行 full_reinit，期间不可被重新分配。
            worker_pool.release_worker(
                worker,
                request_type=duplex_type,
                duration_s=duration,
                post_release_status=GatewayWorkerStatus.LOADING,
            )
            logger.info(f"Duplex WS ended: session={session_id}, type={duplex_type}, Worker released ({duration:.1f}s)")


# ============ 默认 Ref Audio 分发 ============

@app.get("/api/frontend_defaults")
async def get_frontend_defaults():
    """返回前端页面需要的默认配置

    前端页面加载时调用此接口获取 playback_delay_ms 等可配置的默认值，
    避免前端硬编码。返回值来自 config.json。
    """
    from config import get_config
    return get_config().frontend_defaults()


# ============ System Prompt 预设 ============

_presets_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None


def _get_audio_meta(rel_path: str, project_root: str) -> Dict[str, Any]:
    """获取音频文件的元数据（不加载 base64），用于预设列表"""
    import librosa

    if not rel_path:
        return {"name": "", "duration": 0}

    abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(project_root, rel_path)
    name = os.path.basename(abs_path)

    if not os.path.exists(abs_path):
        return {"name": name, "duration": 0}

    try:
        audio, sr = librosa.load(abs_path, sr=16000, mono=True)
        return {"name": name, "duration": round(len(audio) / sr, 1)}
    except Exception:
        return {"name": name, "duration": 0}


def _load_audio_base64(rel_path: str, project_root: str) -> Optional[Dict[str, Any]]:
    """加载音频文件为 base64（按需调用）"""
    import librosa

    if not rel_path:
        return None

    abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(project_root, rel_path)
    if not os.path.exists(abs_path):
        return None

    try:
        audio, sr = librosa.load(abs_path, sr=16000, mono=True)
        audio_bytes = audio.astype(np.float32).tobytes()
        import base64 as b64mod
        return {
            "data": b64mod.b64encode(audio_bytes).decode("ascii"),
            "name": os.path.basename(abs_path),
            "duration": round(len(audio) / sr, 1),
        }
    except Exception as e:
        logger.error(f"Failed to load audio {abs_path}: {e}")
        return None


def _load_presets_from_dir(project_root: str) -> Dict[str, List[Dict[str, Any]]]:
    """扫描 assets/presets/<mode>/*.yaml，返回元数据（不含音频 base64）"""
    import yaml

    presets_root = os.path.join(project_root, "assets", "presets")
    result: Dict[str, List[Dict[str, Any]]] = {}

    if not os.path.isdir(presets_root):
        return result

    for mode_dir in sorted(os.listdir(presets_root)):
        mode_path = os.path.join(presets_root, mode_dir)
        if not os.path.isdir(mode_path):
            continue

        mode_presets = []
        for fname in sorted(os.listdir(mode_path)):
            if not fname.endswith((".yaml", ".yml")):
                continue
            fpath = os.path.join(mode_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    preset = yaml.safe_load(f)
                if not preset or not isinstance(preset, dict):
                    continue

                if "system_content" in preset:
                    resolved = []
                    for item in preset["system_content"]:
                        if item.get("type") == "audio" and item.get("path"):
                            meta = _get_audio_meta(item["path"], project_root)
                            resolved.append({
                                "type": "audio",
                                "data": None,
                                "path": item["path"],
                                "name": meta["name"],
                                "duration": meta["duration"],
                            })
                        else:
                            resolved.append(item)
                    preset["system_content"] = resolved

                if "ref_audio_path" in preset:
                    meta = _get_audio_meta(preset["ref_audio_path"], project_root)
                    preset["ref_audio"] = {
                        "data": None,
                        "path": preset["ref_audio_path"],
                        "name": meta["name"],
                        "duration": meta["duration"],
                    }
                    del preset["ref_audio_path"]

                mode_presets.append(preset)
            except Exception as e:
                logger.error(f"Failed to load preset {fpath}: {e}")

        if mode_presets:
            mode_presets.sort(key=lambda p: p.get("order", 999))
            result[mode_dir] = mode_presets

    total = sum(len(v) for v in result.values())
    logger.info(f"Loaded {total} presets (metadata only) across {len(result)} modes")
    return result


@app.get("/api/presets")
async def get_presets():
    """返回预设元数据（不含音频 base64，音频通过 /api/presets/{mode}/{id}/audio 按需加载）"""
    global _presets_cache

    if _presets_cache is not None:
        return _presets_cache

    project_root = os.path.dirname(__file__)
    _presets_cache = _load_presets_from_dir(project_root)
    return _presets_cache


@app.get("/api/presets/{mode}/{preset_id}/audio")
async def get_preset_audio(mode: str, preset_id: str):
    """按需加载单个 preset 的音频数据"""
    global _presets_cache
    if _presets_cache is None:
        project_root = os.path.dirname(__file__)
        _presets_cache = _load_presets_from_dir(project_root)

    mode_presets = _presets_cache.get(mode, [])
    preset = next((p for p in mode_presets if p.get("id") == preset_id), None)
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset not found: {mode}/{preset_id}")

    project_root = os.path.dirname(__file__)
    result: Dict[str, Any] = {}

    if "system_content" in preset:
        audio_items = []
        for item in preset["system_content"]:
            if item.get("type") == "audio" and item.get("path"):
                loaded = _load_audio_base64(item["path"], project_root)
                audio_items.append(loaded or {"data": None, "name": item.get("name", ""), "duration": 0})
        result["system_content_audio"] = audio_items

    if preset.get("ref_audio") and preset["ref_audio"].get("path"):
        loaded = _load_audio_base64(preset["ref_audio"]["path"], project_root)
        result["ref_audio"] = loaded or {"data": None, "name": preset["ref_audio"].get("name", ""), "duration": 0}

    return result


# 缓存：启动后首次请求时加载，之后直接返回
_default_ref_audio_cache: Optional[Dict[str, Any]] = None


@app.get("/api/default_ref_audio")
async def get_default_ref_audio():
    """返回默认参考音频（PCM float32 16kHz mono base64）

    前端页面加载时调用此接口获取默认 ref audio，
    之后所有请求统一通过 ref_audio_base64 传递音频数据。
    """
    global _default_ref_audio_cache

    if _default_ref_audio_cache is not None:
        return _default_ref_audio_cache

    from config import get_config
    cfg = get_config()

    if not cfg.ref_audio_path:
        raise HTTPException(status_code=404, detail="No default ref audio configured")

    # 解析路径（支持相对路径，相对于 minicpmo45_service/）
    ref_path = cfg.ref_audio_path
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(os.path.dirname(__file__), ref_path)

    if not os.path.exists(ref_path):
        raise HTTPException(status_code=404, detail=f"Default ref audio not found: {cfg.ref_audio_path}")

    try:
        import base64
        import librosa
        import numpy as np

        # 加载并重采样为 16kHz mono float32（与前端上传格式一致）
        audio, sr = librosa.load(ref_path, sr=16000, mono=True)
        duration = len(audio) / 16000

        # 转换为 base64（PCM float32）
        audio_bytes = audio.astype(np.float32).tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        _default_ref_audio_cache = {
            "name": os.path.basename(cfg.ref_audio_path),
            "duration": round(duration, 1),
            "sample_rate": 16000,
            "samples": len(audio),
            "base64": audio_b64,
        }
        logger.info(
            f"Default ref audio loaded: {_default_ref_audio_cache['name']} "
            f"({duration:.1f}s, {len(audio)} samples)"
        )
        return _default_ref_audio_cache

    except Exception as e:
        logger.error(f"Failed to load default ref audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load ref audio: {e}")


# ============ 素材管理 API ============

@app.get("/api/assets/ref_audio", response_model=RefAudioListResponse)
async def list_ref_audios():
    """列出参考音频"""
    if ref_audio_registry is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    return RefAudioListResponse(
        total=ref_audio_registry.count,
        ref_audios=ref_audio_registry.list_all(),
    )


@app.post("/api/assets/ref_audio", response_model=RefAudioResponse)
async def upload_ref_audio(request: UploadRefAudioRequest):
    """上传参考音频"""
    if ref_audio_registry is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        info = ref_audio_registry.upload(
            name=request.name,
            audio_base64=request.audio_base64,
        )
        return RefAudioResponse(
            success=True,
            id=info.id,
            name=info.name,
            message=f"Uploaded successfully, duration={info.duration_ms}ms",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload ref audio failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/assets/ref_audio/{ref_id}", response_model=RefAudioResponse)
async def delete_ref_audio(ref_id: str):
    """删除参考音频"""
    if ref_audio_registry is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    if not ref_audio_registry.exists(ref_id):
        raise HTTPException(status_code=404, detail=f"Ref audio not found: {ref_id}")

    success = ref_audio_registry.delete(ref_id)
    return RefAudioResponse(
        success=success,
        id=ref_id,
        message="Deleted" if success else "Failed to delete",
    )


# ============ 队列状态 API ============

@app.get("/api/queue", response_model=QueueStatus)
async def get_queue():
    """获取当前队列状态"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return worker_pool.get_queue_status()


@app.get("/api/queue/{ticket_id}")
async def get_queue_ticket(ticket_id: str):
    """获取指定排队项的状态（前端轮询用）"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    ticket = worker_pool.get_ticket(ticket_id)
    if ticket is None:
        return {"found": False, "message": "Ticket not in queue (may have been assigned or cancelled)"}
    return {"found": True, "ticket": ticket.model_dump()}


@app.delete("/api/queue/{ticket_id}")
async def cancel_queue_item(ticket_id: str):
    """取消排队项（Admin 用）"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    ok = worker_pool.cancel(ticket_id)
    return {"success": ok}


# ============ ETA 配置 API ============

@app.get("/api/config/eta", response_model=EtaStatus)
async def get_eta_config():
    """获取 ETA 配置和 EMA 状态"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return worker_pool.eta_tracker.get_status()


@app.put("/api/config/eta", response_model=EtaStatus)
async def update_eta_config(new_config: EtaConfig):
    """更新 ETA 基准配置（运行时生效，无需重启）"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    worker_pool.eta_tracker.update_config(new_config)
    alpha_str = f", ema_alpha={new_config.ema_alpha}" if new_config.ema_alpha is not None else ""
    logger.info(
        f"ETA config updated: chat={new_config.eta_chat_s}s, "
        f"streaming={new_config.eta_streaming_s}s, "
        f"half_duplex={new_config.eta_half_duplex_s}s, "
        f"audio_duplex={new_config.eta_audio_duplex_s}s, "
        f"omni_duplex={new_config.eta_omni_duplex_s}s, "
        f"duplex={new_config.eta_duplex_s}s{alpha_str}"
    )
    return worker_pool.eta_tracker.get_status()


# ============ 缓存状态 API ============

@app.get("/cache")
async def list_cache():
    """查看各 Worker 的 KV cache 状态"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    return {
        "workers": [
            {
                "worker_id": w.worker_id,
                "status": w.status.value,
            }
            for w in worker_pool.workers.values()
        ],
    }


# ============ Session API ============

_BASE_DIR = os.path.dirname(__file__)


def _list_active_sessions_payload() -> Dict[str, Any]:
    """Gateway 当前占用 Worker 的会话列表（Admin / 测试共用）。"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    sessions: List[Dict[str, Any]] = []
    for w in worker_pool.workers.values():
        if not w.current_session_id:
            continue
        last_active = w.last_heartbeat or w.task_started_at or datetime.now()
        sessions.append(
            {
                "session_id": w.current_session_id,
                "worker_id": w.worker_id,
                "messages_hash": w.cached_hash or "",
                "last_active": last_active.isoformat()
                if hasattr(last_active, "isoformat")
                else str(last_active),
            }
        )
    return {"total": len(sessions), "sessions": sessions}


@app.get("/sessions")
async def list_active_sessions():
    """列出活跃会话（与 admin.html `GET /sessions` 一致）。"""
    return _list_active_sessions_payload()


@app.get("/api/sessions")
async def list_active_sessions_api():
    """同 list_active_sessions，REST 风格路径便于与 `/api/sessions/{id}` 并列。"""
    return _list_active_sessions_payload()


def _session_dir(session_id: str) -> str:
    """获取 session 目录的绝对路径（含路径安全校验）"""
    safe_id = _sanitize_session_id(session_id)
    from config import get_config
    cfg = get_config()
    base = os.path.join(_BASE_DIR, cfg.data_dir, "sessions", safe_id)
    resolved = os.path.realpath(base)
    sessions_root = os.path.realpath(os.path.join(_BASE_DIR, cfg.data_dir, "sessions"))
    if not resolved.startswith(sessions_root):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    return resolved


@app.get("/api/sessions/{session_id}")
async def get_session_meta(session_id: str):
    """获取 session 元数据 (meta.json)"""
    sdir = _session_dir(session_id)
    meta_path = os.path.join(sdir, "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/sessions/{session_id}/recording")
async def get_session_recording(session_id: str):
    """获取录制 timeline (recording.json)"""
    sdir = _session_dir(session_id)
    rec_path = os.path.join(sdir, "recording.json")
    if not os.path.exists(rec_path):
        raise HTTPException(status_code=404, detail=f"Recording not found for session: {session_id}")
    with open(rec_path, "r", encoding="utf-8") as f:
        return json.load(f)


_MIME_MAP = {
    ".wav": "audio/wav",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webm": "video/webm",
    ".mp4": "video/mp4",
    ".mp3": "audio/mpeg",
}


@app.api_route("/api/sessions/{session_id}/assets/{asset_path:path}", methods=["GET", "HEAD"])
async def get_session_asset(session_id: str, asset_path: str):
    """获取 session 资源文件（音频 WAV / 图片）

    音频已在录制时存为 16-bit PCM WAV，直接 serve（支持 Range 请求）。
    """
    sdir = _session_dir(session_id)

    if ".." in asset_path or asset_path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid asset path")

    full_path = os.path.realpath(os.path.join(sdir, asset_path))
    if not full_path.startswith(os.path.realpath(sdir)):
        raise HTTPException(status_code=400, detail="Path traversal detected")

    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}")

    ext = os.path.splitext(full_path)[1].lower()
    mime = _MIME_MAP.get(ext)
    if mime:
        return FileResponse(full_path, media_type=mime)
    return FileResponse(full_path)


@app.get("/api/sessions/{session_id}/download")
async def download_session(session_id: str):
    """打包下载整个 session (zip)"""
    sdir = _session_dir(session_id)
    if not os.path.isdir(sdir):
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    def _build_zip() -> BytesIO:
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(sdir):
                for fname in files:
                    abs_path = os.path.join(root, fname)
                    arc_name = os.path.join(session_id, os.path.relpath(abs_path, sdir))
                    zf.write(abs_path, arc_name)
        buf.seek(0)
        return buf

    zip_buf = await asyncio.to_thread(_build_zip)
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={session_id}.zip"},
    )


_UPLOAD_MAX_BYTES = 200 * 1024 * 1024  # 200 MB
_ALLOWED_UPLOAD_TYPES = {"video/webm", "video/mp4", "audio/wav", "audio/webm"}
_EXT_FROM_MIME = {"video/webm": ".webm", "video/mp4": ".mp4", "audio/wav": ".wav", "audio/webm": ".webm"}


@app.post("/api/sessions/{session_id}/upload-recording")
async def upload_session_recording(session_id: str, file: UploadFile = File(...)):
    """上传前端录制文件（视频/音频）到 session 目录

    存储为 data/sessions/{session_id}/frontend_replay.{ext}
    """
    sdir = _session_dir(session_id)
    if not os.path.isdir(sdir):
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    content_type = (file.content_type or "").split(";")[0].strip()
    if content_type not in _ALLOWED_UPLOAD_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")

    ext = _EXT_FROM_MIME.get(content_type, ".bin")
    dest = os.path.join(sdir, f"frontend_replay{ext}")

    total = 0
    with open(dest, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _UPLOAD_MAX_BYTES:
                f.close()
                os.remove(dest)
                raise HTTPException(status_code=413, detail="File too large (max 200MB)")
            f.write(chunk)

    logger.info(f"[Session] uploaded frontend recording: {dest} ({total / 1024 / 1024:.1f} MB)")
    return {"status": "ok", "path": f"frontend_replay{ext}", "size_bytes": total}


@app.get("/s/{session_id}", response_class=HTMLResponse)
async def session_viewer(session_id: str):
    """Session 回看页面"""
    sdir = _session_dir(session_id)
    meta_path = os.path.join(sdir, "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    viewer_path = os.path.join(os.path.dirname(__file__), "static", "session-viewer.html")
    if os.path.exists(viewer_path):
        return FileResponse(viewer_path)
    return HTMLResponse(f"<h1>Session {session_id}</h1><p>Viewer page not found</p>")


# ============ APP 管理 ============

@app.get("/api/apps", response_model=AppsPublicResponse)
async def get_enabled_apps():
    """返回当前启用的 APP 列表（前端导航栏和主页卡片使用）"""
    return AppsPublicResponse(apps=app_registry.get_enabled_apps())


@app.get("/api/admin/apps", response_model=AppsAdminResponse)
async def get_all_apps():
    """返回所有 APP 列表（含 enabled 状态，Admin 页面使用）"""
    return AppsAdminResponse(apps=app_registry.get_all_apps())


@app.put("/api/admin/apps/{app_id}")
async def toggle_app(app_id: str, req: AppToggleRequest):
    """切换 APP 启用/禁用状态（Admin 操作，运行时生效）"""
    result = app_registry.set_enabled(app_id, req.enabled)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unknown app: {app_id}")
    action = "enabled" if req.enabled else "disabled"
    logger.info(f"[AppRegistry] App '{app_id}' {action}")
    return {"success": True, "app": result}


# ============ 静态文件 ============

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """首页：模式选择（Turn-based / Omni Duplex / Audio Duplex）"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse(
        "<h1>MiniCPMO45 Service</h1>"
        "<p>API docs: <a href='/docs'>/docs</a></p>"
    )


@app.get("/turnbased", response_class=HTMLResponse)
async def turnbased():
    """Turn-based Chat Demo 页面"""
    if not app_registry.is_enabled("turnbased"):
        return RedirectResponse(url="/", status_code=302)
    page_path = os.path.join(static_dir, "turnbased.html")
    if os.path.exists(page_path):
        return FileResponse(page_path)
    return HTMLResponse("<h1>Turn-based Chat</h1><p>Page not found</p>")


@app.get("/omni", response_class=HTMLResponse)
async def omni():
    """Omni Duplex Demo 页面"""
    if not app_registry.is_enabled("omni"):
        return RedirectResponse(url="/", status_code=302)
    omni_path = os.path.join(static_dir, "omni", "omni.html")
    if os.path.exists(omni_path):
        return FileResponse(omni_path)
    return HTMLResponse("<h1>Omni</h1><p>Omni page not found</p>")


@app.get("/half_duplex", response_class=HTMLResponse)
async def half_duplex():
    """Half-Duplex Audio Demo 页面"""
    if not app_registry.is_enabled("half_duplex_audio"):
        return RedirectResponse(url="/", status_code=302)
    page_path = os.path.join(static_dir, "half-duplex", "half_duplex.html")
    if os.path.exists(page_path):
        return FileResponse(page_path)
    return HTMLResponse("<h1>Half-Duplex Audio</h1><p>Page not found</p>")


@app.get("/audio_duplex", response_class=HTMLResponse)
async def audio_duplex():
    """语音双工 Demo 页面（简化版 Omni，无视频）"""
    if not app_registry.is_enabled("audio_duplex"):
        return RedirectResponse(url="/", status_code=302)
    page_path = os.path.join(static_dir, "audio-duplex", "audio_duplex.html")
    if os.path.exists(page_path):
        return FileResponse(page_path)
    return HTMLResponse("<h1>Audio Duplex</h1><p>Page not found</p>")


@app.get("/admin", response_class=HTMLResponse)
async def admin():
    """Admin Dashboard"""
    admin_path = os.path.join(static_dir, "admin.html")
    if os.path.exists(admin_path):
        return FileResponse(admin_path)
    return HTMLResponse("<h1>Admin</h1><p>Admin page not found</p>")


# ============ 入口 ============

def main():
    from config import get_config
    cfg = get_config()

    parser = argparse.ArgumentParser(description="MiniCPMO45 Gateway")
    parser.add_argument("--port", type=int, default=None, help=f"Gateway port (default: {cfg.gateway_port})")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--workers", type=str, default=None, help="Worker addresses, comma-separated")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of workers (auto-generate addresses)")
    parser.add_argument("--max-queue-size", type=int, default=None, help="Max queue size")
    parser.add_argument("--timeout", type=float, default=None, help="Request timeout (s)")
    # 协议选择：默认 HTTPS，--http 可降级为 HTTP
    proto_group = parser.add_mutually_exclusive_group()
    proto_group.add_argument("--https", action="store_true", default=True,
                             help="启用 HTTPS（默认，自签名证书）")
    proto_group.add_argument("--http", action="store_true",
                             help="降级为 HTTP（不推荐，麦克风等浏览器 API 需要 HTTPS）")
    parser.add_argument("--ssl-certfile", type=str, default="certs/cert.pem", help="SSL cert file path")
    parser.add_argument("--ssl-keyfile", type=str, default="certs/key.pem", help="SSL key file path")
    args = parser.parse_args()

    # --http 被指定时，关闭 HTTPS
    use_https = not args.http

    port = args.port or cfg.gateway_port

    # Worker 地址：优先命令行，否则根据 num_workers 自动生成
    if args.workers:
        worker_list = args.workers.split(",")
    elif args.num_workers is not None:
        worker_list = cfg.worker_addresses(args.num_workers)
    else:
        worker_list = cfg.worker_addresses(cfg.num_workers)

    GATEWAY_CONFIG.update({
        "workers": worker_list,
        "max_queue_size": args.max_queue_size or cfg.max_queue_size,
        "timeout": args.timeout or cfg.request_timeout,
        "eta_config": {
            "eta_chat_s": cfg.eta_chat_s,
            "eta_streaming_s": cfg.eta_streaming_s,
            "eta_half_duplex_s": cfg.eta_half_duplex_s,
            "eta_audio_duplex_s": cfg.eta_audio_duplex_s,
            "eta_omni_duplex_s": cfg.eta_omni_duplex_s,
            "eta_duplex_s": cfg.eta_duplex_s,
        },
        "eta_ema_alpha": cfg.eta_ema_alpha,
        "eta_ema_min_samples": cfg.eta_ema_min_samples,
    })

    proto_label = "HTTPS" if use_https else "HTTP"
    logger.info(f"Starting Gateway on port {port} ({proto_label})")
    logger.info(f"Workers: {worker_list}")

    ssl_kwargs = {}
    if use_https:
        cert = args.ssl_certfile
        key = args.ssl_keyfile
        if not os.path.exists(cert) or not os.path.exists(key):
            logger.error(f"SSL cert/key not found: {cert}, {key}")
            logger.error("Generate with: openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem -out certs/cert.pem -days 365 -nodes -subj '/CN=dev'")
            logger.error("Or use --http to start without HTTPS")
            return
        ssl_kwargs = {"ssl_certfile": cert, "ssl_keyfile": key}
        logger.info(f"HTTPS enabled: cert={cert}, key={key}")
    else:
        logger.warning("Running in HTTP mode (no TLS). Browser microphone/camera APIs may not work.")

    uvicorn.run(app, host=args.host, port=port, **ssl_kwargs)


if __name__ == "__main__":
    main()
