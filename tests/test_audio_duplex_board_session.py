"""Unit tests for the standalone audio_duplex_board session.

These tests use a fake FC duplex view and do not load GPU models.
"""

from __future__ import annotations

from audio_duplex_board.config import AudioDuplexBoardConfig
from audio_duplex_board.schemas import StreamAudioChunkRequest, StreamPrepareRequest
from audio_duplex_board.session import AudioDuplexBoardSession
from core.schemas.fc_duplex import (
    FcClosedSpan,
    FcDuplexPrefillResult,
    FcDuplexUnitInfo,
    FcNonSpokenGenerateResult,
    FcSpokenGenerateResult,
)


class _FakeProcessor:
    def __init__(self) -> None:
        self.fc_duplex = _FakeFcDuplex()


class _FakeFcDuplex:
    def __init__(self) -> None:
        self.prepare_calls = []
        self.prefill_tool_response_counts = []
        self.unit = 0

    def prepare(self, request):
        self.prepare_calls.append(request)

    def streaming_prefill(self, request):
        self.prefill_tool_response_counts.append(len(request.tool_responses or []))
        return FcDuplexPrefillResult(
            unit_index=self.unit,
            n_audio_placeholders=10,
            has_input_event=bool(request.tool_responses),
        )

    def streaming_spoken_generate(self, request):
        return FcSpokenGenerateResult(
            is_listen=True,
            is_speaking=False,
            spoken_token_ids=[151705],
        )

    def streaming_non_spoken_generate(self, request):
        if self.unit == 0:
            return FcNonSpokenGenerateResult(
                token_ids=[],
                terminated=True,
                close_reason="eos",
                closed_spans=[
                    FcClosedSpan(
                        type="tool_call",
                        tool_call_id="call_1",
                        wire='<function name="display_object_on_board"><param name="name">刷毛硬毛刷</param></function>',
                        tool_call={
                            "name": "display_object_on_board",
                            "arguments": {"name": "刷毛硬毛刷"},
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

    def finalize_unit(self):
        current = self.unit
        self.unit += 1
        return FcDuplexUnitInfo(
            unit=current,
            n_audio=10,
            is_listen=True,
            is_speaking=False,
            non_spoken_terminator="eos",
        )


def test_streaming_session_creates_board_card_and_queues_tool_response() -> None:
    session = AudioDuplexBoardSession(
        config=AudioDuplexBoardConfig(model_path="/tmp/model"),
        processor=_FakeProcessor(),  # type: ignore[arg-type]
    )

    prepare_events = session.prepare_stream(StreamPrepareRequest(system_prompt="test"))
    assert [event.type for event in prepare_events] == ["session_started"]

    first_events = session.process_audio_chunk(
        StreamAudioChunkRequest(audio_base64="AAAA", sample_rate=16000)
    )
    first_types = [event.type for event in first_events]
    assert "tool_call_final" in first_types
    assert "board_card_created" in first_types
    assert "board_card_updated" in first_types

    second_events = session.process_audio_chunk(
        StreamAudioChunkRequest(audio_base64="AAAA", sample_rate=16000)
    )
    assert second_events[0].payload["tool_response_count"] == 1
