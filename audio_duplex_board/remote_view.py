"""Business-side HTTP client that duck-types `FcDuplexView`.

The business server (run_server.py) and the GPU model server (model_server.py)
talk over plain HTTP. `RemoteFcDuplexView` exposes the exact same method
surface that `core.processors.unified.FcDuplexView` does (the subset the
board session uses), so session.py can stay model-server-agnostic.

Method surface used by `AudioDuplexBoardSession`:

    prepare(FcDuplexPrepareRequest) -> FcDuplexPrepareResult
    streaming_prefill(FcDuplexPrefillRequest) -> FcDuplexPrefillResult
    streaming_spoken_generate(FcSpokenGenerateRequest) -> FcSpokenGenerateResult
    streaming_non_spoken_generate(FcNonSpokenGenerateRequest) -> FcNonSpokenGenerateResult
    finalize_unit() -> FcDuplexUnitInfo
    cleanup() -> None
    offline_inference_from_train_data(FcDuplexTrainDataRequest)
        -> FcDuplexTrainDataResult  # used by replay endpoint only

All methods are synchronous (blocking HTTP via `requests`) because session.py
already wraps them in `asyncio.to_thread`. Adding async httpx here would
not buy anything and would force session.py to know whether it's remote
or local.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import requests

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
    FcDuplexTrainDataResult,
)


_NP_KIND = "__np__"


def _decode_np(value: Any) -> Any:
    """Reverse of model_server._encode_np."""

    if not isinstance(value, dict) or not value.get(_NP_KIND):
        return value
    dtype = np.dtype(value["dtype"])
    shape = tuple(value["shape"])
    raw = base64.b64decode(value["base64"])
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


class RemoteFcDuplexView:
    """HTTP client mimicking FcDuplexView's primitive surface.

    Args:
        base_url: e.g. `https://10.156.9.151:18081`.
        verify_tls: pass False for self-signed model server cert (devbox
            controls both sides, in-cluster traffic is private).
        timeout_sec: per-request HTTP timeout. prefill/spoken/non_spoken
            on a 9B model usually complete in <2s, but the very first
            prepare can take several seconds.
    """

    def __init__(
        self,
        base_url: str,
        *,
        verify_tls: bool = False,
        timeout_sec: float = 120.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._verify = verify_tls
        self._timeout = timeout_sec
        self._session = requests.Session()

    # -------- primitives consumed by AudioDuplexBoardSession --------

    def prepare(self, req: FcDuplexPrepareRequest) -> FcDuplexPrepareResult:
        data = self._post("/prepare", req)
        return FcDuplexPrepareResult.model_validate(data)

    def streaming_prefill(self, req: FcDuplexPrefillRequest) -> FcDuplexPrefillResult:
        data = self._post("/streaming_prefill", req)
        return FcDuplexPrefillResult.model_validate(data)

    def streaming_spoken_generate(
        self, req: FcSpokenGenerateRequest
    ) -> FcSpokenGenerateResult:
        data = self._post("/streaming_spoken_generate", req)
        if "audio_waveform" in data:
            data["audio_waveform"] = _decode_np(data["audio_waveform"])
        return FcSpokenGenerateResult.model_validate(data)

    def streaming_non_spoken_generate(
        self, req: FcNonSpokenGenerateRequest
    ) -> FcNonSpokenGenerateResult:
        data = self._post("/streaming_non_spoken_generate", req)
        if "audio_waveform" in data:
            data["audio_waveform"] = _decode_np(data["audio_waveform"])
        return FcNonSpokenGenerateResult.model_validate(data)

    def finalize_unit(self) -> FcDuplexUnitInfo:
        data = self._post("/finalize_unit", None)
        return FcDuplexUnitInfo.model_validate(data)

    def cleanup(self) -> None:
        self._post("/cleanup", None)

    # -------- additional primitive used by the replay debug path --------

    def offline_inference_from_train_data(
        self, req: FcDuplexTrainDataRequest
    ) -> FcDuplexTrainDataResult:
        data = self._post("/offline_inference_from_train_data", req)
        return FcDuplexTrainDataResult.model_validate(data)

    # -------- internals --------

    def _post(self, path: str, body: Any) -> Any:
        url = f"{self._base}{path}"
        if body is None:
            payload: dict[str, Any] = {}
        elif hasattr(body, "model_dump"):
            payload = body.model_dump(mode="json")
        elif isinstance(body, dict):
            payload = body
        else:
            raise TypeError(f"unsupported body type: {type(body)}")
        resp = self._session.post(
            url, json=payload, timeout=self._timeout, verify=self._verify
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"POST {path} failed: HTTP {resp.status_code} body={resp.text[:500]}"
            )
        return resp.json()


class RemoteProcessor:
    """Tiny shim exposing `.fc_duplex` so session/run_server can pass it where
    `UnifiedProcessor` was previously expected."""

    def __init__(self, fc_duplex: RemoteFcDuplexView) -> None:
        self.fc_duplex = fc_duplex
