"""Realtime token usage accounting.

The model reports per-operation token deltas. This module converts those
deltas into the public multimodal usage shape and owns session accumulation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict


def _count(value: Any) -> int:
    if value is None:
        return 0
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


@dataclass
class TokenUsage:
    input_text_tokens: int = 0
    input_audio_tokens: int = 0
    input_vision_tokens: int = 0
    output_text_tokens: int = 0
    output_audio_tokens: int = 0

    @classmethod
    def from_counts(cls, counts: Mapping[str, Any] | None) -> "TokenUsage":
        counts = counts or {}
        return cls(
            input_text_tokens=_count(counts.get("input_text_tokens")),
            input_audio_tokens=_count(counts.get("input_audio_tokens")),
            input_vision_tokens=_count(counts.get("input_vision_tokens")),
            output_text_tokens=_count(counts.get("output_text_tokens")),
            output_audio_tokens=_count(counts.get("output_audio_tokens")),
        )

    @classmethod
    def from_duplex_chunk(
        cls,
        prefill_result: Any,
        generate_result: Any,
    ) -> "TokenUsage":
        prefill = _mapping(prefill_result)
        prefill_usage = _mapping(prefill.get("usage", prefill))
        generated_usage = _mapping(getattr(generate_result, "usage", None))
        vision_tokens = _count(prefill_usage.get("input_vision_tokens"))
        return cls(
            input_text_tokens=_count(prefill_usage.get("input_text_tokens")),
            input_audio_tokens=_count(prefill_usage.get("input_audio_tokens")),
            input_vision_tokens=vision_tokens,
            output_text_tokens=_count(
                generated_usage.get("output_text_tokens", getattr(generate_result, "n_tokens", 0))
            ),
            output_audio_tokens=_count(
                generated_usage.get("output_audio_tokens", getattr(generate_result, "n_tts_tokens", 0))
            ),
        )

    def add(self, other: "TokenUsage") -> None:
        self.input_text_tokens += other.input_text_tokens
        self.input_audio_tokens += other.input_audio_tokens
        self.input_vision_tokens += other.input_vision_tokens
        self.output_text_tokens += other.output_text_tokens
        self.output_audio_tokens += other.output_audio_tokens

    def to_dict(self) -> Dict[str, Any]:
        input_tokens = (
            self.input_text_tokens
            + self.input_audio_tokens
            + self.input_vision_tokens
        )
        output_tokens = self.output_text_tokens + self.output_audio_tokens
        return {
            "total_tokens": input_tokens + output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_token_details": {
                "text_tokens": self.input_text_tokens,
                "audio_tokens": self.input_audio_tokens,
                "vision_tokens": self.input_vision_tokens,
            },
            "output_token_details": {
                "text_tokens": self.output_text_tokens,
                "audio_tokens": self.output_audio_tokens,
            },
        }


class SessionUsage:
    """Cumulative usage for one backend session."""

    def __init__(self) -> None:
        self._total = TokenUsage()

    def add(self, delta: TokenUsage) -> Dict[str, Any]:
        self._total.add(delta)
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        return self._total.to_dict()
