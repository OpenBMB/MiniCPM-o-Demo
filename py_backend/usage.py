"""Realtime token usage accounting.

The public payload mirrors the familiar ``usage`` shape while keeping
multimodal token details. ``SessionUsage`` owns the session-wide accumulator;
callers keep the returned per-chunk value alongside the cumulative snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


def _token_count(value: Any) -> int:
    """Convert backend counters to a non-negative integer."""

    if value is None:
        return 0
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


@dataclass
class TokenUsage:
    input_text_tokens: int = 0
    input_audio_tokens: int = 0
    input_image_tokens: int = 0
    input_video_tokens: int = 0
    output_text_tokens: int = 0
    output_audio_tokens: int = 0

    @classmethod
    def from_counts(cls, counts: Mapping[str, Any]) -> "TokenUsage":
        return cls(
            input_text_tokens=_token_count(counts.get("input_text_tokens")),
            input_audio_tokens=_token_count(counts.get("input_audio_tokens")),
            input_image_tokens=_token_count(counts.get("input_image_tokens")),
            input_video_tokens=_token_count(counts.get("input_video_tokens")),
            output_text_tokens=_token_count(counts.get("output_text_tokens")),
            output_audio_tokens=_token_count(counts.get("output_audio_tokens")),
        )

    @classmethod
    def from_duplex_chunk(
        cls,
        prefill_result: Any,
        generate_result: Any,
        *,
        has_video: bool,
    ) -> "TokenUsage":
        prefill = prefill_result if isinstance(prefill_result, Mapping) else {}
        vision_tokens = _token_count(prefill.get("input_vision_tokens"))
        return cls(
            input_text_tokens=_token_count(prefill.get("input_text_tokens")),
            input_audio_tokens=_token_count(prefill.get("input_audio_tokens")),
            input_image_tokens=0 if has_video else vision_tokens,
            input_video_tokens=vision_tokens if has_video else 0,
            output_text_tokens=_token_count(getattr(generate_result, "n_tokens", None)),
            output_audio_tokens=_token_count(getattr(generate_result, "n_tts_tokens", None)),
        )

    def add(self, other: "TokenUsage") -> None:
        self.input_text_tokens += other.input_text_tokens
        self.input_audio_tokens += other.input_audio_tokens
        self.input_image_tokens += other.input_image_tokens
        self.input_video_tokens += other.input_video_tokens
        self.output_text_tokens += other.output_text_tokens
        self.output_audio_tokens += other.output_audio_tokens

    def to_dict(self) -> Dict[str, Any]:
        input_tokens = (
            self.input_text_tokens
            + self.input_audio_tokens
            + self.input_image_tokens
            + self.input_video_tokens
        )
        output_tokens = self.output_text_tokens + self.output_audio_tokens
        return {
            "total_tokens": input_tokens + output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_token_details": {
                "text_tokens": self.input_text_tokens,
                "audio_tokens": self.input_audio_tokens,
                "image_tokens": self.input_image_tokens,
                "video_tokens": self.input_video_tokens,
                "cached_tokens": 0,
            },
            "output_token_details": {
                "text_tokens": self.output_text_tokens,
                "audio_tokens": self.output_audio_tokens,
            },
        }


class SessionUsage:
    """Accumulate token counts for one realtime session."""

    def __init__(self) -> None:
        self._total = TokenUsage()

    def add(self, usage: TokenUsage) -> Dict[str, Any]:
        self._total.add(usage)
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        return self._total.to_dict()
