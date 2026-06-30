"""GPU-side model server: thin FastAPI wrapper around `FcDuplexView`.

Runs inside a Cybertron GPU job. Holds the 9B MiniCPM-o 4.5 model in memory
and exposes the 7 FC duplex primitives over plain HTTP so that the business
server (run_server.py) can live on the devbox and iterate quickly without
reloading the model on every change.

Endpoints
=========

GET  /health
    Liveness + reports which checkpoint is loaded.

POST /prepare                       body: FcDuplexPrepareRequest
    Returns FcDuplexPrepareResult.

POST /streaming_prefill             body: FcDuplexPrefillRequest
    Returns FcDuplexPrefillResult.

POST /streaming_spoken_generate     body: FcSpokenGenerateRequest
    Returns FcSpokenGenerateResult. audio_waveform (numpy array) is encoded
    as `{"__np__": True, "dtype": ..., "shape": [...], "base64": "..."}`.

POST /streaming_non_spoken_generate body: FcNonSpokenGenerateRequest
    Returns FcNonSpokenGenerateResult.

POST /finalize_unit                 body: {}
    Returns FcDuplexUnitInfo.

POST /cleanup                       body: {}
    Returns {"ok": true}.

POST /offline_inference_from_train_data  body: FcDuplexTrainDataRequest
    Returns FcDuplexTrainDataResult (used by the replay debug path).

Concurrency
===========

FcDuplexView is a single stateful object — only ONE caller at a time.
We protect every endpoint with an asyncio.Lock so business-side concurrency
bugs cannot corrupt model state. If you need parallel sessions, run another
model server on another GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# Forward-declare types so the file is importable without GPU stack for typing.
# All real instantiation happens inside `create_app()`.
from core.schemas.fc_duplex import (
    FcDuplexPrefillRequest,
    FcDuplexPrefillResult,
    FcDuplexPrepareRequest,
    FcDuplexPrepareResult,
    FcDuplexUnitInfo,
    FcNonSpokenGenerateRequest,
    FcNonSpokenGenerateResult,
    FcSpokenGenerateRequest,
    FcSpokenGenerateResult,
    FcDuplexTrainDataRequest,
)


# ============================================================ helpers


_NP_KIND = "__np__"


def _encode_np(arr: Any) -> Any:
    """Replace numpy arrays in a return payload with a JSON-safe envelope.

    The only field that ships numpy in current FC duplex is
    `audio_waveform`. We special-case that. Other fields are JSON-native.
    """

    if arr is None:
        return None
    np_arr = np.asarray(arr)
    return {
        _NP_KIND: True,
        "dtype": str(np_arr.dtype),
        "shape": list(np_arr.shape),
        "base64": base64.b64encode(np_arr.tobytes()).decode("ascii"),
    }


def _spoken_result_to_payload(result: FcSpokenGenerateResult) -> dict[str, Any]:
    payload = result.model_dump(mode="json")
    # `audio_waveform` was excluded from model_dump for numpy reasons; re-inject.
    payload["audio_waveform"] = _encode_np(result.audio_waveform)
    return payload


def _non_spoken_result_to_payload(result: FcNonSpokenGenerateResult) -> dict[str, Any]:
    payload = result.model_dump(mode="json")
    payload["audio_waveform"] = _encode_np(result.audio_waveform)
    return payload


# ============================================================ app factory


def create_app(
    *,
    model_path: str,
    pt_path: str | None,
    sdk_src: str | None,
) -> FastAPI:
    """Create the GPU-side model server FastAPI app.

    Args:
        model_path: HF model dir (MiniCPM-o 4.5 base).
        pt_path: Optional fine-tuned checkpoint overlay (.pt).
        sdk_src: Optional local SDK src to prepend on sys.path before any
            tokenize call inside the view.
    """

    if sdk_src:
        sdk_src_path = str(Path(sdk_src))
        if sdk_src_path not in sys.path:
            sys.path.insert(0, sdk_src_path)

    from core.processors import UnifiedProcessor

    processor = UnifiedProcessor(
        model_path=model_path,
        pt_path=pt_path,
        device="cuda",
        compile=False,
        attn_implementation="sdpa",
    )
    view = processor.fc_duplex
    lock = asyncio.Lock()

    app = FastAPI(title="Audio Duplex Board Model Server")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "model_path": model_path,
            "pt_path": pt_path,
            "sdk_src": sdk_src,
        }

    @app.post("/prepare")
    async def prepare(req: FcDuplexPrepareRequest) -> Any:
        async with lock:
            try:
                result = await asyncio.to_thread(view.prepare, req)
            except Exception as exc:  # noqa: BLE001
                _log_exc("/prepare", exc)
                raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
        return result.model_dump(mode="json")

    @app.post("/streaming_prefill")
    async def streaming_prefill(req: FcDuplexPrefillRequest) -> Any:
        async with lock:
            try:
                result = await asyncio.to_thread(view.streaming_prefill, req)
            except Exception as exc:  # noqa: BLE001
                _log_exc("/streaming_prefill", exc)
                raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
        return result.model_dump(mode="json")

    @app.post("/streaming_spoken_generate")
    async def streaming_spoken_generate(req: FcSpokenGenerateRequest) -> Any:
        async with lock:
            try:
                result = await asyncio.to_thread(view.streaming_spoken_generate, req)
            except Exception as exc:  # noqa: BLE001
                _log_exc("/streaming_spoken_generate", exc)
                raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
        return _spoken_result_to_payload(result)

    @app.post("/streaming_non_spoken_generate")
    async def streaming_non_spoken_generate(req: FcNonSpokenGenerateRequest) -> Any:
        async with lock:
            try:
                result = await asyncio.to_thread(view.streaming_non_spoken_generate, req)
            except Exception as exc:  # noqa: BLE001
                _log_exc("/streaming_non_spoken_generate", exc)
                raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
        return _non_spoken_result_to_payload(result)

    @app.post("/finalize_unit")
    async def finalize_unit() -> Any:
        async with lock:
            try:
                result = await asyncio.to_thread(view.finalize_unit)
            except Exception as exc:  # noqa: BLE001
                _log_exc("/finalize_unit", exc)
                raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
        return result.model_dump(mode="json")

    @app.post("/cleanup")
    async def cleanup() -> Any:
        async with lock:
            try:
                await asyncio.to_thread(view.cleanup)
            except Exception as exc:  # noqa: BLE001
                _log_exc("/cleanup", exc)
                raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
        return {"ok": True}

    @app.post("/offline_inference_from_train_data")
    async def offline_inference_from_train_data(req: FcDuplexTrainDataRequest) -> Any:
        # This call is long-running (whole offline replay). Still holds the lock
        # for its full duration; that is intentional because the model is
        # single-tenant.
        async with lock:
            try:
                result = await asyncio.to_thread(
                    view.offline_inference_from_train_data, req
                )
            except Exception as exc:  # noqa: BLE001
                _log_exc("/offline_inference_from_train_data", exc)
                raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
        return result.model_dump(mode="json")

    # =====================================================================
    # WebSocket control channel
    # =====================================================================
    #
    # Persistent ws between business and model_server. Multiplexes:
    #
    #   client → server (RPC request/response by `id`):
    #     {"type": "rpc_request", "id": <int>, "method": "prepare"
    #         | "streaming_prefill" | "streaming_spoken_generate"
    #         | "finalize_unit" | "cleanup",
    #      "params": {...}}
    #     → server replies
    #     {"type": "rpc_response", "id": <int>, "ok": true, "result": {...}}
    #     or {"type": "rpc_response", "id": <int>, "ok": false, "error": "..."}
    #
    #   client → server (server-side decode loop):
    #     {"type": "decode_start", "id": <int>,
    #      "params": {"decode_mode": "greedy"}}
    #     → server streams back many
    #       {"type": "decode_step", "id": <int>, "step": {...}}
    #     → server eventually sends one
    #       {"type": "decode_end", "id": <int>, "reason": "<...>", "n_steps": N}
    #
    #   client → server (cooperative stop):
    #     {"type": "decode_stop", "id": <int>}
    #     → server's loop checks the stop flag at the TOP of each iteration.
    #       That gives ~half-token average extra latency: any in-flight
    #       streaming_non_spoken_generate(max_tokens=1) call completes, the
    #       result is still sent as decode_step, THEN the next iteration's
    #       check trips and decode_end with reason=stopped_by_client fires.
    #
    # Single tenant: at most one ws connection is processed at a time
    # (FcDuplexView is stateful). A second concurrent connection gets 1008.
    ws_active = asyncio.Lock()

    @app.websocket("/ws/control")
    async def ws_control(ws: WebSocket) -> None:
        await ws.accept()
        if ws_active.locked():
            await ws.close(code=1008, reason="another control session is active")
            return
        async with ws_active:
            await _run_ws_control_session(ws=ws, view=view, view_lock=lock)

    return app


async def _run_ws_control_session(*, ws: WebSocket, view: Any, view_lock: asyncio.Lock) -> None:
    """One ws control session lifecycle.

    Maintains:
      - active_decodes: dict[req_id, asyncio.Task] for in-flight decode loops
      - stop_events:    dict[req_id, asyncio.Event] mirroring the client-side
                        stop signal; loop checks at top of every iteration.
    """

    active_decodes: dict[int, asyncio.Task[None]] = {}
    stop_events: dict[int, asyncio.Event] = {}
    send_lock = asyncio.Lock()

    async def safe_send(msg: dict[str, Any]) -> None:
        async with send_lock:
            try:
                await ws.send_json(msg)
            except Exception:  # noqa: BLE001
                # ws likely closed; swallow so background tasks don't crash
                return

    async def handle_rpc(msg: dict[str, Any]) -> None:
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}
        try:
            async with view_lock:
                if method == "prepare":
                    req = FcDuplexPrepareRequest.model_validate(params)
                    result = await asyncio.to_thread(view.prepare, req)
                    payload = result.model_dump(mode="json")
                elif method == "streaming_prefill":
                    req = FcDuplexPrefillRequest.model_validate(params)
                    result = await asyncio.to_thread(view.streaming_prefill, req)
                    payload = result.model_dump(mode="json")
                elif method == "streaming_spoken_generate":
                    req = FcSpokenGenerateRequest.model_validate(params)
                    result = await asyncio.to_thread(view.streaming_spoken_generate, req)
                    payload = _spoken_result_to_payload(result)
                elif method == "finalize_unit":
                    result = await asyncio.to_thread(view.finalize_unit)
                    payload = result.model_dump(mode="json")
                elif method == "cleanup":
                    await asyncio.to_thread(view.cleanup)
                    payload = {"ok": True}
                else:
                    await safe_send({
                        "type": "rpc_response",
                        "id": req_id,
                        "ok": False,
                        "error": f"unknown method: {method}",
                    })
                    return
            await safe_send({
                "type": "rpc_response",
                "id": req_id,
                "ok": True,
                "result": payload,
            })
        except Exception as exc:  # noqa: BLE001
            _log_exc(f"ws_rpc/{method}", exc)
            await safe_send({
                "type": "rpc_response",
                "id": req_id,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            })

    async def handle_decode_start(msg: dict[str, Any]) -> None:
        req_id = msg.get("id")
        params = msg.get("params") or {}
        decode_mode = str(params.get("decode_mode") or "greedy")
        stop_event = asyncio.Event()
        stop_events[req_id] = stop_event

        async def decode_loop() -> None:
            n_steps = 0
            try:
                async with view_lock:
                    for _ in range(_MAX_DECODE_STEPS_PER_UNIT):
                        # === The check that gives the "half-token average
                        # extra latency" semantics: any in-flight step is
                        # completed and sent as a decode_step before the next
                        # iteration trips this and exits cleanly.
                        if stop_event.is_set():
                            await safe_send({
                                "type": "decode_end",
                                "id": req_id,
                                "reason": "stopped_by_client",
                                "n_steps": n_steps,
                            })
                            return

                        try:
                            step = await asyncio.to_thread(
                                view.streaming_non_spoken_generate,
                                FcNonSpokenGenerateRequest(
                                    max_tokens=1, decode_mode=decode_mode
                                ),
                            )
                        except Exception as exc:  # noqa: BLE001
                            _log_exc("ws_decode/step", exc)
                            await safe_send({
                                "type": "decode_end",
                                "id": req_id,
                                "reason": "error",
                                "n_steps": n_steps,
                                "error": f"{type(exc).__name__}: {exc}",
                            })
                            return

                        n_steps += 1
                        await safe_send({
                            "type": "decode_step",
                            "id": req_id,
                            "step": _non_spoken_result_to_payload(step),
                        })
                        if step.terminated:
                            await safe_send({
                                "type": "decode_end",
                                "id": req_id,
                                "reason": step.close_reason or "view_terminated",
                                "n_steps": n_steps,
                            })
                            return

                    # Hard cap — should NEVER hit in healthy real-time pacing
                    # because client signals stop well before.
                    close = await asyncio.to_thread(
                        view.streaming_non_spoken_generate,
                        FcNonSpokenGenerateRequest(
                            max_tokens=0,
                            decode_mode=decode_mode,
                            close_reason="budget_reached",
                        ),
                    )
                    await safe_send({
                        "type": "decode_step",
                        "id": req_id,
                        "step": _non_spoken_result_to_payload(close),
                    })
                    await safe_send({
                        "type": "decode_end",
                        "id": req_id,
                        "reason": "hard_cap_budget_reached",
                        "n_steps": n_steps,
                    })
            finally:
                stop_events.pop(req_id, None)
                active_decodes.pop(req_id, None)

        task = asyncio.create_task(decode_loop())
        active_decodes[req_id] = task

    try:
        while True:
            try:
                msg = await ws.receive_json()
            except WebSocketDisconnect:
                return
            msg_type = msg.get("type")
            if msg_type == "rpc_request":
                # rpc_request can run concurrently with active decode tasks,
                # but each will independently acquire view_lock so view state
                # is serialized. The expected client behavior is to wait for
                # decode_end before calling streaming_prefill / etc anyway.
                asyncio.create_task(handle_rpc(msg))
            elif msg_type == "decode_start":
                await handle_decode_start(msg)
            elif msg_type == "decode_stop":
                target_id = msg.get("id")
                ev = stop_events.get(target_id)
                if ev is not None:
                    ev.set()
            else:
                await safe_send({
                    "type": "error",
                    "error": f"unknown message type: {msg_type}",
                })
    finally:
        for task in active_decodes.values():
            task.cancel()


# Hard cap on per-unit decode steps. Real-time operation should never hit this
# because client-side next-prefill signal arrives ~1s in and stops the loop.
_MAX_DECODE_STEPS_PER_UNIT = 200


def _log_exc(endpoint: str, exc: BaseException) -> None:
    print(
        f"[model_server] {endpoint} failed: {type(exc).__name__}: {exc}\n"
        f"{traceback.format_exc()}",
        flush=True,
    )


# ============================================================ CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-side model server for audio_duplex_board")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind host (default 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=18081,
        help="Bind port (default 18081, business server uses 18080).",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local HF model dir (MiniCPM-o 4.5).",
    )
    parser.add_argument(
        "--pt-path",
        default=None,
        help="Optional fine-tuned checkpoint overlay (.pt).",
    )
    parser.add_argument(
        "--sdk-src",
        default=None,
        help="Optional local SDK src dir to prepend on sys.path.",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Optional TLS private key (PEM) for HTTPS model server.",
    )
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="Optional TLS cert chain (PEM) for HTTPS model server.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        model_path=args.model_path,
        pt_path=args.pt_path,
        sdk_src=args.sdk_src,
    )
    kwargs: dict[str, Any] = {"host": args.host, "port": args.port}
    if args.ssl_keyfile and args.ssl_certfile:
        kwargs["ssl_keyfile"] = args.ssl_keyfile
        kwargs["ssl_certfile"] = args.ssl_certfile
    uvicorn.run(app, **kwargs)


if __name__ == "__main__":
    main()
