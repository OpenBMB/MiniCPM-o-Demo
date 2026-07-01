"""Persistent WebSocket client that duck-types `FcDuplexView` and streams
non-spoken decode tokens from the model_server.

Why a separate file from `remote_view.py`:

- `RemoteFcDuplexView` (in remote_view.py) is sync over HTTP — kept for
  backward compatibility, simpler code, and use cases that don't care about
  per-token streaming.
- `WsRemoteFcDuplexView` is async over WebSocket, multiplexed RPC + streaming
  non_spoken decode. The chatty per-token path lives entirely server-side
  inside the model_server's decode loop; the client only sees per-token
  `decode_step` events and a single `decode_end`. Stops are cooperative:
  client sends `decode_stop`, server's loop checks at the TOP of each
  iteration. Average extra latency on stop ≈ half a token's decode time.

Public method surface mirrors what `AudioDuplexBoardSession` actually uses
on `FcDuplexView`, but every method is `async`:

    await prepare(FcDuplexPrepareRequest) -> FcDuplexPrepareResult
    await streaming_prefill(FcDuplexPrefillRequest) -> FcDuplexPrefillResult
    await streaming_spoken_generate(FcSpokenGenerateRequest) -> FcSpokenGenerateResult
    await finalize_unit() -> FcDuplexUnitInfo
    await cleanup()
    await offline_inference_from_train_data(FcDuplexTrainDataRequest)  # via HTTP, long-running

Plus the new streaming primitive:

    async for step in stream_non_spoken_decode(decode_mode, stop_event):
        # step: FcNonSpokenGenerateResult (one per server-side decode step)
        # `stop_event` is an external asyncio.Event; setting it makes the
        # client send `decode_stop` to the server. The server's loop will
        # exit before the NEXT decode step (so the last in-flight step
        # still arrives — that's the "half-token average latency" property).
    # After the loop, last_decode_reason / last_decode_n_steps are populated
    # on the stream object (use stream.last_reason / stream.last_n_steps).

Connection lifecycle: lazy. First `_ensure_connected()` opens the ws and
spawns a permanent recv loop. On disconnect, all pending RPC futures fail
and active decode queues receive a synthetic `decode_end(reason="ws_disconnected")`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import ssl
from typing import Any, AsyncIterator, Optional

import numpy as np
import requests
import urllib3
import websockets

from core.schemas.fc_duplex import (
    FcDuplexPrefillRequest,
    FcDuplexPrefillResult,
    FcDuplexPrepareRequest,
    FcDuplexPrepareResult,
    FcDuplexUnitInfo,
    FcNonSpokenGenerateResult,
    FcSpokenGenerateRequest,
    FcSpokenGenerateResult,
    FcDuplexTrainDataRequest,
    FcDuplexTrainDataResult,
)


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


_NP_KIND = "__np__"


def _decode_np(value: Any) -> Any:
    """Decode the {__np__,dtype,shape,base64} envelope produced by model_server."""

    if not isinstance(value, dict) or not value.get(_NP_KIND):
        return value
    dtype = np.dtype(value["dtype"])
    shape = tuple(value["shape"])
    raw = base64.b64decode(value["base64"])
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


class NonSpokenDecodeStream:
    """Async iterator yielding `FcNonSpokenGenerateResult` per server step.

    Lifecycle:
      1. Caller `async for step in stream:` consumes steps as they arrive.
      2. On every loop iteration the server checks `stop_event`; setting it
         from the caller side triggers `decode_stop` to be sent over ws.
         The server's loop will exit at its next check, but any in-flight
         decode call completes first — that's the half-token average extra
         latency property the caller asked for.
      3. After the loop returns (normally or via stop), `self.last_reason`
         and `self.last_n_steps` describe how the server-side loop ended.
    """

    def __init__(
        self,
        client: "WsRemoteFcDuplexView",
        request_id: int,
        decode_mode: str,
        stop_event: asyncio.Event,
    ) -> None:
        self._client = client
        self._id = request_id
        self._decode_mode = decode_mode
        self._stop_event = stop_event
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._started = False
        self._stop_watcher: Optional[asyncio.Task[None]] = None
        self.last_reason: Optional[str] = None
        self.last_n_steps: int = 0
        self.last_error: Optional[str] = None

    async def _on_message(self, msg: dict[str, Any]) -> None:
        """Called by the client's recv loop when a message for our id arrives."""
        await self._queue.put(msg)

    async def _start(self) -> None:
        if self._started:
            return
        self._started = True
        # Register queue with client first, so any race-y `decode_step` can land
        # before the start command return.
        self._client._register_decode_stream(self._id, self)
        await self._client._send_raw({
            "type": "decode_start",
            "id": self._id,
            "params": {"decode_mode": self._decode_mode},
        })
        # Spawn stop-watcher: on stop_event set, send decode_stop. Server's
        # next-iteration check exits the loop and sends decode_end.
        async def watch_stop() -> None:
            try:
                await self._stop_event.wait()
                await self._client._send_raw({
                    "type": "decode_stop",
                    "id": self._id,
                })
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: BLE001
                return
        self._stop_watcher = asyncio.create_task(watch_stop())

    def __aiter__(self) -> AsyncIterator[FcNonSpokenGenerateResult]:
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[FcNonSpokenGenerateResult]:
        await self._start()
        try:
            while True:
                msg = await self._queue.get()
                msg_type = msg.get("type")
                if msg_type == "decode_step":
                    step_data = dict(msg.get("step") or {})
                    if "audio_waveform" in step_data:
                        step_data["audio_waveform"] = _decode_np(step_data["audio_waveform"])
                    yield FcNonSpokenGenerateResult.model_validate(step_data)
                elif msg_type == "decode_end":
                    self.last_reason = msg.get("reason")
                    self.last_n_steps = int(msg.get("n_steps") or 0)
                    self.last_error = msg.get("error")
                    return
                else:
                    # ignore stray
                    continue
        finally:
            self._client._unregister_decode_stream(self._id)
            if self._stop_watcher and not self._stop_watcher.done():
                self._stop_watcher.cancel()


class WsRemoteFcDuplexView:
    """Persistent-WS client mimicking FcDuplexView.

    Args:
        base_url: e.g. `https://10.156.9.171:18081` (https or http; we map
            to wss/ws for the /ws/control endpoint).
        verify_tls: pass False for in-cluster self-signed cert.
        offline_http_url: optional override for the HTTP base used by
            `offline_inference_from_train_data`. Defaults to base_url.
    """

    def __init__(
        self,
        base_url: str,
        *,
        verify_tls: bool = False,
        offline_http_url: Optional[str] = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._verify_tls = verify_tls
        self._ws_url = (
            self._base.replace("https://", "wss://").replace("http://", "ws://")
            + "/ws/control"
        )
        self._offline_http_base = (offline_http_url or self._base).rstrip("/")
        self._http_session = requests.Session()
        # ws state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connect_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._recv_task: Optional[asyncio.Task[None]] = None
        self._id_counter = 0
        self._rpc_futures: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._decode_streams: dict[int, NonSpokenDecodeStream] = {}

    # ----------------------------------------------- connection management

    async def _ensure_connected(self) -> None:
        async with self._connect_lock:
            if self._ws is not None and not getattr(self._ws, "closed", False):
                return
            ssl_ctx: Optional[ssl.SSLContext] = None
            if self._ws_url.startswith("wss://"):
                ssl_ctx = ssl.create_default_context()
                if not self._verify_tls:
                    ssl_ctx.check_hostname = False
                    ssl_ctx.verify_mode = ssl.CERT_NONE
            self._ws = await websockets.connect(
                self._ws_url,
                ssl=ssl_ctx,
                max_size=64 * 1024 * 1024,  # 64MB safety for audio_waveform payloads
                # 之前 ping_interval/timeout=20s，但模型服务器第一次跑 TTS
                # token2wav 时 GPU 前端会阻塞 asyncio 事件循环 20s 以上（
                # torch.compile 的 kernel launch + token2wav mel/vocoder），
                # 结果客户端把 ws 判死了。业务与模型都在内网跑，健康监控靠
                # RPC 本身即可，这里干脆关掉 keepalive。
                ping_interval=None,
                ping_timeout=None,
                close_timeout=5,
            )
            self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:  # noqa: BLE001
                    continue
                msg_type = msg.get("type")
                if msg_type == "rpc_response":
                    rid = msg.get("id")
                    fut = self._rpc_futures.pop(rid, None)
                    if fut is not None and not fut.done():
                        fut.set_result(msg)
                elif msg_type in ("decode_step", "decode_end"):
                    rid = msg.get("id")
                    stream = self._decode_streams.get(rid)
                    if stream is not None:
                        await stream._on_message(msg)
                else:
                    # error or stray; ignore for now
                    continue
        except Exception:  # noqa: BLE001
            pass
        finally:
            # Connection lost: fail all pending RPCs and decode streams
            for fut in list(self._rpc_futures.values()):
                if not fut.done():
                    fut.set_exception(RuntimeError("ws connection lost"))
            self._rpc_futures.clear()
            for stream in list(self._decode_streams.values()):
                await stream._on_message({
                    "type": "decode_end",
                    "id": stream._id,
                    "reason": "ws_disconnected",
                    "n_steps": stream.last_n_steps,
                })
            self._ws = None

    async def _send_raw(self, msg: dict[str, Any]) -> None:
        await self._ensure_connected()
        async with self._send_lock:
            assert self._ws is not None
            await self._ws.send(json.dumps(msg))

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def _register_decode_stream(self, req_id: int, stream: NonSpokenDecodeStream) -> None:
        self._decode_streams[req_id] = stream

    def _unregister_decode_stream(self, req_id: int) -> None:
        self._decode_streams.pop(req_id, None)

    # ----------------------------------------------- RPC primitives

    async def _call_rpc(self, method: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        await self._ensure_connected()
        rid = self._next_id()
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._rpc_futures[rid] = fut
        await self._send_raw({
            "type": "rpc_request",
            "id": rid,
            "method": method,
            "params": params or {},
        })
        msg = await fut
        if not msg.get("ok"):
            raise RuntimeError(f"ws rpc {method} failed: {msg.get('error')}")
        return msg.get("result") or {}

    async def prepare(self, req: FcDuplexPrepareRequest) -> FcDuplexPrepareResult:
        data = await self._call_rpc("prepare", req.model_dump(mode="json"))
        return FcDuplexPrepareResult.model_validate(data)

    async def streaming_prefill(self, req: FcDuplexPrefillRequest) -> FcDuplexPrefillResult:
        data = await self._call_rpc("streaming_prefill", req.model_dump(mode="json"))
        return FcDuplexPrefillResult.model_validate(data)

    async def streaming_spoken_generate(
        self, req: FcSpokenGenerateRequest
    ) -> FcSpokenGenerateResult:
        data = await self._call_rpc("streaming_spoken_generate", req.model_dump(mode="json"))
        if "audio_waveform" in data:
            data["audio_waveform"] = _decode_np(data["audio_waveform"])
        return FcSpokenGenerateResult.model_validate(data)

    async def finalize_unit(self) -> FcDuplexUnitInfo:
        data = await self._call_rpc("finalize_unit", {})
        return FcDuplexUnitInfo.model_validate(data)

    async def cleanup(self) -> None:
        await self._call_rpc("cleanup", {})

    # ----------------------------------------------- streaming non_spoken

    def stream_non_spoken_decode(
        self,
        *,
        decode_mode: str = "greedy",
        stop_event: asyncio.Event,
    ) -> NonSpokenDecodeStream:
        """Return an async iterator over server-decoded non_spoken steps.

        The server keeps a tight decode loop and streams each step back.
        Caller can set `stop_event` to ask the server to bail out at the
        TOP of the next loop iteration — any in-flight step completes
        first and is delivered as a final `decode_step`, then `decode_end`
        with reason=`stopped_by_client` arrives.

        Usage:

            stop = asyncio.Event()
            stream = view.stream_non_spoken_decode(stop_event=stop)
            async for step in stream:
                ...
                if some_condition:
                    stop.set()  # cooperative stop, server completes one more step
            # after the loop, stream.last_reason / stream.last_n_steps are filled
        """

        rid = self._next_id()
        return NonSpokenDecodeStream(self, rid, decode_mode, stop_event)

    # ----------------------------------------------- legacy HTTP fallback

    async def offline_inference_from_train_data(
        self, req: FcDuplexTrainDataRequest
    ) -> FcDuplexTrainDataResult:
        """Long-running offline replay; still goes over HTTP because it is
        called only by the /api/replay-case debug endpoint and benefits
        from the per-request timeout / connection model of plain HTTP.
        """

        def _sync_post() -> dict[str, Any]:
            url = f"{self._offline_http_base}/offline_inference_from_train_data"
            resp = self._http_session.post(
                url,
                json=req.model_dump(mode="json"),
                timeout=3600.0,
                verify=self._verify_tls,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"POST /offline_inference_from_train_data failed: "
                    f"HTTP {resp.status_code} body={resp.text[:500]}"
                )
            return resp.json()

        data = await asyncio.to_thread(_sync_post)
        return FcDuplexTrainDataResult.model_validate(data)


class WsRemoteProcessor:
    """Tiny shim exposing `.fc_duplex` so run_server can pass it where
    `UnifiedProcessor`/`RemoteProcessor` was previously expected."""

    def __init__(self, fc_duplex: WsRemoteFcDuplexView) -> None:
        self.fc_duplex = fc_duplex
