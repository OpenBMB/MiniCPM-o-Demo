"""Backend metrics schema.

BackendMetrics is the observable metric snapshot produced by the inference
backend (kv cache length, timings, token counts). It lives in schemas because
it is a shared data contract consumed by the backend and the protocol layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class BackendMetrics:
    """Observable backend/runtime state sampled by session runtimes."""

    backend: Optional[str] = None
    kv_cache_length: int = 0
    n_past_max: Optional[int] = None

    prefill_ms: Optional[float] = None
    generate_ms: Optional[float] = None
    wall_clock_ms: Optional[float] = None
    cost_llm_ms: Optional[float] = None
    cost_tts_prep_ms: Optional[float] = None
    cost_tts_ms: Optional[float] = None
    cost_token2wav_ms: Optional[float] = None

    n_tokens: Optional[int] = None
    n_tts_tokens: Optional[int] = None
    vision_slices: Optional[int] = None
    vision_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]]) -> "BackendMetrics":
        if not data:
            return cls()
        allowed = cls.__dataclass_fields__.keys()
        values = {key: value for key, value in data.items() if key in allowed}
        return cls(**values)
