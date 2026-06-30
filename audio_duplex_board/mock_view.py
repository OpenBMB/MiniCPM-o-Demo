"""GPU-free mock FcDuplexView for board UI validation.

The mock implements the same method names used by `AudioDuplexBoardSession`.
It detects loud user audio chunks by RMS energy and emits deterministic
display_object_on_board tool calls in sequence, allowing the frontend and tool
pipeline to be tested without loading a model.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field

import numpy as np

from core.schemas.fc_duplex import (
    FcClosedSpan,
    FcDuplexPrefillResult,
    FcDuplexUnitInfo,
    FcNonSpokenGenerateResult,
    FcSpokenGenerateResult,
)


@dataclass
class MockFcDuplexView:
    """A deterministic mock compatible with the board session's FcDuplex calls.

    Args:
        energy_threshold: RMS threshold above which an audio chunk triggers a
            tool call.
        tool_names: Ordered objects emitted on loud chunks.
        generated_audio_sample_rate: Sample rate for synthetic AI waveform.
    """

    energy_threshold: float = 0.012
    tool_names: list[str] = field(default_factory=lambda: ["苹果", "番茄", "腌白菜"])
    generated_audio_sample_rate: int = 24000

    def __post_init__(self) -> None:
        self.unit_index = 0
        self._prepared = False
        self._pending_tool_name: str | None = None
        self._next_tool_index = 0
        self._last_energy = 0.0
        self._last_was_loud = False

    def prepare(self, request) -> object:
        self._prepared = True
        self.unit_index = 0
        self._pending_tool_name = None
        self._next_tool_index = 0
        self._last_energy = 0.0
        self._last_was_loud = False
        return request

    def streaming_prefill(self, request) -> FcDuplexPrefillResult:
        samples = _decode_float32_base64(request.audio_data)
        self._last_energy = _rms(samples)
        is_loud = self._last_energy >= self.energy_threshold
        # Rising-edge trigger avoids emitting a card for every loud second while
        # the user keeps speaking.
        if is_loud and not self._last_was_loud:
            self._pending_tool_name = self.tool_names[self._next_tool_index % len(self.tool_names)]
            self._next_tool_index += 1
        self._last_was_loud = is_loud
        return FcDuplexPrefillResult(
            unit_index=self.unit_index,
            n_audio_placeholders=10 if samples.size else 0,
            has_input_event=bool(request.tool_responses),
        )

    def streaming_spoken_generate(self, request) -> FcSpokenGenerateResult:
        if self._pending_tool_name:
            text = f"听到了，我把{self._pending_tool_name}放到画板上。"
            waveform = _sine_wave(
                duration_sec=0.45,
                sample_rate=self.generated_audio_sample_rate,
                frequency_hz=440.0 + 30.0 * ((self._next_tool_index - 1) % len(self.tool_names)),
            )
            return FcSpokenGenerateResult(
                is_listen=False,
                is_speaking=True,
                spoken_token_ids=[151706, 151718],
                spoken_text=text,
                spoken_turn_eos=False,
                audio_waveform=waveform,
                audio_sample_rate=self.generated_audio_sample_rate,
                n_audio_samples=int(waveform.shape[0]),
            )
        return FcSpokenGenerateResult(
            is_listen=True,
            is_speaking=False,
            spoken_token_ids=[151705],
        )

    def streaming_non_spoken_generate(self, request) -> FcNonSpokenGenerateResult:
        if self._pending_tool_name:
            tool_name = self._pending_tool_name
            self._pending_tool_name = None
            call_id = f"mock_call_{self.unit_index:04d}"
            return FcNonSpokenGenerateResult(
                token_ids=[],
                terminated=True,
                close_reason="eos",
                closed_spans=[
                    FcClosedSpan(
                        type="tool_call",
                        tool_call_id=call_id,
                        wire=(
                            '<function name="display_object_on_board">'
                            f'<param name="name">{tool_name}</param>'
                            "</function>"
                        ),
                        tool_call={
                            "name": "display_object_on_board",
                            "arguments": {"name": tool_name},
                            "tool_call_id": call_id,
                        },
                    )
                ],
            )
        return FcNonSpokenGenerateResult(
            token_ids=[],
            terminated=True,
            close_reason="no_action",
            closed_spans=[],
        )

    def finalize_unit(self) -> FcDuplexUnitInfo:
        unit = self.unit_index
        self.unit_index += 1
        return FcDuplexUnitInfo(
            unit=unit,
            n_audio=10,
            is_listen=not self._last_was_loud,
            is_speaking=self._last_was_loud,
            non_spoken_terminator="eos",
        )


@dataclass
class MockUnifiedProcessor:
    """Tiny processor shim exposing `.fc_duplex` like UnifiedProcessor."""

    energy_threshold: float = 0.012

    def __post_init__(self) -> None:
        self.fc_duplex = MockFcDuplexView(energy_threshold=self.energy_threshold)


def _decode_float32_base64(audio_base64: str | None) -> np.ndarray:
    if not audio_base64:
        return np.zeros((0,), dtype=np.float32)
    raw = base64.b64decode(audio_base64)
    if not raw:
        return np.zeros((0,), dtype=np.float32)
    return np.frombuffer(raw, dtype=np.float32)


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(np.square(samples.astype(np.float64))))))


def _sine_wave(
    *,
    duration_sec: float,
    sample_rate: int,
    frequency_hz: float,
) -> np.ndarray:
    t = np.arange(int(duration_sec * sample_rate), dtype=np.float32) / float(sample_rate)
    envelope = np.linspace(0.3, 0.0, t.shape[0], dtype=np.float32)
    return (0.15 * envelope * np.sin(2.0 * np.pi * frequency_hz * t)).astype(np.float32)
