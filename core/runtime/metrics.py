"""Runtime metrics/log formatting helpers."""

from __future__ import annotations

import logging

from core.runtime.duplex import DuplexFrameResult


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

