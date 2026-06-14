"""Thin client for the backend-server protocol."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx
import websockets


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _ws_url_for(base_url: str, path: str) -> str:
    parsed = urlsplit(_join_url(base_url, path))
    if parsed.scheme == "http":
        scheme = "ws"
    elif parsed.scheme == "https":
        scheme = "wss"
    else:
        scheme = parsed.scheme
    return urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment))


class RemoteBackendSession:
    """One remote backend session using init/push/pull/close primitives."""

    def __init__(
        self,
        *,
        base_url: str,
        mode: str,
        session_id: Optional[str] = None,
        ws_path: str = "/backend",
        close_timeout_s: float = 30.0,
        max_ws_size: int = 128 * 1024 * 1024,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.mode = mode
        self.session_id = session_id
        self.ws_path = ws_path
        self.close_timeout_s = close_timeout_s
        self.max_ws_size = max_ws_size
        self._ws: Any = None
        self._closed = False

    @property
    def initialized(self) -> bool:
        return self._ws is not None and self.session_id is not None and not self._closed

    async def init(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._ws is not None:
            raise RuntimeError("backend session is already initialized")

        init_params = dict(params or {})
        init_params.setdefault("mode", self.mode)
        # session identity 由 backend 分配：不向 init 发送建议的 session_id（见协议 schema §3.1）。
        # backend 在 session.created 中回传真实 id，下面 pull 后读回。
        init_params.pop("session_id", None)

        self._ws = await websockets.connect(
            _ws_url_for(self.base_url, self.ws_path),
            max_size=self.max_ws_size,
        )
        await self._ws.send(json.dumps({
            "type": "session.init",
            "payload": init_params,
        }))

        event = await self.pull()
        if event.get("type") not in {"session.created", "initialized"}:
            raise RuntimeError(f"backend init returned unexpected event: {event.get('type')}")
        if event.get("session_id"):
            self.session_id = str(event["session_id"])
        return event

    async def push(self, input_payload: Dict[str, Any]) -> None:
        if self._ws is None or self._closed:
            raise RuntimeError("backend session is not active")
        await self._ws.send(json.dumps({
            "type": "input.append",
            "input": input_payload,
        }))

    async def pull(self) -> Dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("backend session WebSocket is not connected")
        raw = await self._ws.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        event = json.loads(raw)
        if not isinstance(event, dict):
            raise RuntimeError("backend event must be a JSON object")
        if event.get("type") == "session.created" and event.get("session_id"):
            self.session_id = str(event["session_id"])
        if event.get("type") == "session.closed":
            self._closed = True
        return event

    async def unary(self, method: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generic one-shot RPC to the backend. The only method today is "close"."""
        payload = payload or {}
        if method in {"close", "backend.close", "session.close"}:
            return await self.close(reason=str(payload.get("reason") or "client_closed"))
        raise ValueError(f"unsupported backend unary method: {method}")

    async def close(self, *, reason: str = "client_closed") -> Dict[str, Any]:
        if self._closed:
            if self._ws is not None:
                await self._ws.close()
                self._ws = None
            return {"ok": True, "session_id": self.session_id, "closed": True}

        result: Dict[str, Any]
        if self.session_id is not None:
            async with httpx.AsyncClient(timeout=self.close_timeout_s) as client:
                response = await client.post(
                    _join_url(self.base_url, f"/sessions/{self.session_id}/close"),
                    json={"reason": reason},
                )
                response.raise_for_status()
                result = response.json()
        else:
            result = {"ok": True, "session_id": None, "closed": True}

        self._closed = True
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        return result

    async def aclose(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        self._closed = True
