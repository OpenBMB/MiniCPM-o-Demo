"""FcDuplexView 的安全增量文本与逐 generation step 投影测试。"""

from __future__ import annotations

from typing import Any

from core.processors.unified import FcDuplexView
from core.schemas.fc_duplex import (
    FcDuplexPrepareRequest,
    FcNonSpokenGenerateRequest,
    FcSpokenGenerateRequest,
)
from minicpm_o5_sdk import O5TokenizerID, load_builtin_tokenizer


class _FakeCapability:
    """只暴露 View 所需 SDK tokenizer 的 capability stub。"""

    def __init__(self) -> None:
        self.protocol_tokenizer = load_builtin_tokenizer(O5TokenizerID.O45_FC)


class _FakeFcModel:
    """按预设结果返回 FC primitive dict 的 CPU fake model。"""

    def __init__(self) -> None:
        self.fc_duplex = _FakeCapability()
        self.non_spoken_results: list[dict[str, Any]] = []
        self.spoken_results: list[dict[str, Any]] = []

    def fc_duplex_prepare(self, **_: Any) -> dict[str, Any]:
        return {}

    def fc_duplex_streaming_non_spoken_generate(self, **_: Any) -> dict[str, Any]:
        return self.non_spoken_results.pop(0)

    def fc_duplex_streaming_spoken_generate(self, **_: Any) -> dict[str, Any]:
        return self.spoken_results.pop(0)

    def fc_duplex_cleanup(self) -> None:
        return


def _step_kinds(result: Any) -> list[str]:
    return [step.output.kind for step in result.generation_steps]


def test_non_spoken_text_delta_crosses_primitive_calls_without_replacement() -> None:
    """Think 字符跨 token/step 时应保留 pending，并在完整后输出一个 safe delta。"""

    model = _FakeFcModel()
    tokenizer = model.fc_duplex.protocol_tokenizer
    think_start = tokenizer.token_to_id("<think>")
    think_end = tokenizer.token_to_id("</think>")
    text_ids = tokenizer.encode_ordinary("龘")
    assert len(text_ids) == 2
    model.non_spoken_results = [
        {"token_ids": [think_start]},
        {"token_ids": [text_ids[0]]},
        {"token_ids": [text_ids[1]]},
        {
            "token_ids": [think_end],
            "closed_spans": [{"type": "think", "text": "龘"}],
        },
    ]
    view = FcDuplexView(model)  # type: ignore[arg-type]
    view.prepare(FcDuplexPrepareRequest())

    started = view.streaming_non_spoken_generate(FcNonSpokenGenerateRequest())
    pending = view.streaming_non_spoken_generate(FcNonSpokenGenerateRequest())
    emitted = view.streaming_non_spoken_generate(FcNonSpokenGenerateRequest())
    closed = view.streaming_non_spoken_generate(FcNonSpokenGenerateRequest())

    assert _step_kinds(started) == ["protocol"]
    assert _step_kinds(pending) == ["text_pending"]
    assert pending.text_delta == ""
    assert _step_kinds(emitted) == ["text_delta"]
    assert emitted.text_delta == "龘"
    assert emitted.generation_steps[0].output.source_step_count == 2
    assert _step_kinds(closed) == ["protocol"]
    assert "\ufffd" not in emitted.text_delta
    assert view.resume_boundary_status() == {"status": "available"}


def test_spoken_stream_keeps_pending_text_across_units_until_turn_eos() -> None:
    """Spoken turn 的 decoder 应跨 Unit 保留，turn_eos 后回到可恢复边界。"""

    model = _FakeFcModel()
    tokenizer = model.fc_duplex.protocol_tokenizer
    speak = tokenizer.token_to_id("<|speak|>")
    turn_eos = tokenizer.token_to_id("<|spoken_turn_eos|>")
    text_ids = tokenizer.encode_ordinary("龘")
    assert len(text_ids) == 2
    model.spoken_results = [
        {
            "is_speaking": True,
            "spoken_ids": [speak, text_ids[0]],
        },
        {
            "is_speaking": True,
            "spoken_ids": [text_ids[1], turn_eos],
            "spoken_turn_eos": True,
        },
    ]
    view = FcDuplexView(model)  # type: ignore[arg-type]
    view.prepare(FcDuplexPrepareRequest())

    first = view.streaming_spoken_generate(FcSpokenGenerateRequest())
    assert _step_kinds(first) == ["protocol", "text_pending"]
    assert first.spoken_text_delta == ""
    assert view.resume_boundary_status()["reason"] == "pending_text_delta"

    second = view.streaming_spoken_generate(FcSpokenGenerateRequest())
    assert _step_kinds(second) == ["text_delta", "protocol", "protocol"]
    assert second.spoken_text_delta == "龘"
    assert second.generation_steps[0].output.source_step_count == 2
    assert view.resume_boundary_status() == {"status": "available"}


def test_view_marks_non_roundtrippable_safe_delta_as_non_resumable() -> None:
    """可显示文本若不能精确 encode 回原 token IDs，checkpoint 必须不可恢复。"""

    model = _FakeFcModel()
    tokenizer = model.fc_duplex.protocol_tokenizer
    speak = tokenizer.token_to_id("<|speak|>")
    turn_eos = tokenizer.token_to_id("<|spoken_turn_eos|>")
    live_text_ids = [124100, 141319]
    decoded_text = tokenizer.decode_ordinary(live_text_ids)
    assert tokenizer.encode_ordinary(decoded_text) != live_text_ids
    model.spoken_results = [
        {
            "is_speaking": True,
            "spoken_ids": [speak, *live_text_ids, turn_eos],
            "spoken_turn_eos": True,
        }
    ]
    view = FcDuplexView(model)  # type: ignore[arg-type]
    view.prepare(FcDuplexPrepareRequest())

    result = view.streaming_spoken_generate(FcSpokenGenerateRequest())

    assert result.spoken_text_delta == decoded_text
    assert view.resume_boundary_status() == {
        "status": "unavailable",
        "reason": "text_delta_roundtrip_mismatch",
        "stream_id": result.generation_steps[1].stream_id,
    }
