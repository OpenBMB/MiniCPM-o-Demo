"""Runtime metrics/log formatting helpers."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.runtime.worker_protocol import DuplexFrameResult


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


def log_duplex_frame(
    logger: logging.Logger,
    frame: DuplexFrameResult,
    *,
    gpu_id: int,
) -> None:
    """Log one duplex frame using the legacy worker log format."""

    result = frame.result
    if not result.is_listen:
        llm = result.cost_llm_ms or 0
        tts_prep = result.cost_tts_prep_ms or 0
        tts = result.cost_tts_ms or 0
        t2w = result.cost_token2wav_ms or 0
        total = result.cost_all_ms or 0
        n_tok = result.n_tokens or 0
        n_tts_tok = result.n_tts_tokens or 0
        logger.info(
            f"[GPU {gpu_id}] SPEAK t={result.current_time} wall={frame.wall_clock_ms:.0f}ms | "
            f"prefill={frame.prefill_ms:.0f} llm={llm:.0f} tts_prep={tts_prep:.0f} "
            f"tts={tts:.0f} t2w={t2w:.0f} total={total:.0f}ms | "
            f"tokens={n_tok} tts_tokens={n_tts_tok} kv={frame.kv_cache_len} | "
            f"vimg={frame.n_vision_images} vtok={frame.vision_tokens} | "
            f"text='{(result.text or '')[:20]}'"
        )
        return

    total = result.cost_all_ms or 0
    logger.info(
        f"[GPU {gpu_id}] LISTEN t={result.current_time} wall={frame.wall_clock_ms:.0f}ms | "
        f"prefill={frame.prefill_ms:.0f} generate={total:.0f}ms kv={frame.kv_cache_len} | "
        f"vimg={frame.n_vision_images} vtok={frame.vision_tokens}"
    )

