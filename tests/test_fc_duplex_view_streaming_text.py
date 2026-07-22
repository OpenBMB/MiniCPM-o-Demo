"""FcDuplexView 的安全增量文本与逐 generation step 投影测试。"""

from __future__ import annotations

from typing import Any

import pytest

from core.processors.unified import FcDuplexView, ToolCallStateManager
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


def test_spoken_repeated_speak_across_units_reuses_turn_stream() -> None:
    """每个 Unit 的 speak 是 continuation marker，不应重建 turn decoder。"""

    model = _FakeFcModel()
    tokenizer = model.fc_duplex.protocol_tokenizer
    speak = tokenizer.token_to_id("<|speak|>")
    slot_eos = tokenizer.token_to_id("<|spoken_slot_eos|>")
    turn_eos = tokenizer.token_to_id("<|spoken_turn_eos|>")
    text_ids = tokenizer.encode_ordinary("龘")
    assert len(text_ids) == 2
    model.spoken_results = [
        {
            "is_speaking": True,
            "spoken_ids": [speak, text_ids[0], slot_eos],
        },
        {
            "is_speaking": True,
            "spoken_ids": [speak, text_ids[1], turn_eos],
            "spoken_turn_eos": True,
        },
    ]
    view = FcDuplexView(model)  # type: ignore[arg-type]
    view.prepare(FcDuplexPrepareRequest())

    first = view.streaming_spoken_generate(FcSpokenGenerateRequest())
    second = view.streaming_spoken_generate(FcSpokenGenerateRequest())

    assert first.generation_steps[1].stream_id == second.generation_steps[1].stream_id
    assert second.spoken_text_delta == "龘"
    assert view.resume_boundary_status() == {"status": "available"}


def test_spoken_listen_before_turn_eos_is_runtime_error() -> None:
    """Active spoken turn 尚未 turn_eos 时出现 listen 属于协议错误。"""

    model = _FakeFcModel()
    tokenizer = model.fc_duplex.protocol_tokenizer
    speak = tokenizer.token_to_id("<|speak|>")
    slot_eos = tokenizer.token_to_id("<|spoken_slot_eos|>")
    listen = tokenizer.token_to_id("<|listen|>")
    model.spoken_results = [
        {
            "is_speaking": True,
            "spoken_ids": [speak, *tokenizer.encode_ordinary("继续"), slot_eos],
        },
        {
            "is_listen": True,
            "spoken_ids": [listen],
        },
    ]
    view = FcDuplexView(model)  # type: ignore[arg-type]
    view.prepare(FcDuplexPrepareRequest())
    view.streaming_spoken_generate(FcSpokenGenerateRequest())

    with pytest.raises(RuntimeError, match="listen before spoken_turn_eos"):
        view.streaming_spoken_generate(FcSpokenGenerateRequest())


def test_budget_reached_terminates_think_stream_and_next_opener_is_new_stream() -> None:
    """Budget close 应销毁 decoder；下一 Unit opener 创建新 stream。"""

    model = _FakeFcModel()
    tokenizer = model.fc_duplex.protocol_tokenizer
    think_start = tokenizer.token_to_id("<think>")
    think_end = tokenizer.token_to_id("</think>")
    model.non_spoken_results = [
        {"token_ids": [think_start, *tokenizer.encode_ordinary("完整")]},
        {
            "token_ids": [*tokenizer.encode_ordinary("后续"), think_end],
            "closed_spans": [{"type": "think", "text": "完整后续"}],
        },
    ]
    view = FcDuplexView(model)  # type: ignore[arg-type]
    view.prepare(FcDuplexPrepareRequest())

    first = view.streaming_non_spoken_generate(FcNonSpokenGenerateRequest())
    terminated = view.terminate_non_spoken_text_stream("budget_reached")
    second = view.streaming_non_spoken_generate(FcNonSpokenGenerateRequest())

    assert terminated.warnings == []
    assert terminated.generation_steps[0].output.semantic_key == "non_spoken_budget_reached"
    assert terminated.generation_steps[0].output.deferred_model_feed is True
    assert second.span_started == "think"
    assert first.generation_steps[0].stream_id != second.generation_steps[0].stream_id


def test_pending_bpe_at_budget_close_emits_warning_and_marks_resume_unsafe() -> None:
    """Budget 终止 incomplete BPE 时应 warning，不应抛异常或补替换字符。"""

    model = _FakeFcModel()
    tokenizer = model.fc_duplex.protocol_tokenizer
    think_start = tokenizer.token_to_id("<think>")
    text_ids = tokenizer.encode_ordinary("龘")
    assert len(text_ids) == 2
    model.non_spoken_results = [{"token_ids": [think_start, text_ids[0]]}]
    view = FcDuplexView(model)  # type: ignore[arg-type]
    view.prepare(FcDuplexPrepareRequest())
    view.streaming_non_spoken_generate(FcNonSpokenGenerateRequest())

    terminated = view.terminate_non_spoken_text_stream("budget_reached")

    assert len(terminated.warnings) == 1
    warning = terminated.warnings[0]
    assert warning.code == "incomplete_bpe_at_stream_end"
    assert warning.reason == "budget_reached"
    assert "\ufffd" not in warning.message
    assert view.resume_boundary_status()["reason"] == "pending_text_delta"


def test_pending_bpe_at_explicit_end_emits_warning_instead_of_runtime_error() -> None:
    """Matching end 遇到 incomplete BPE 时关闭 stream 并返回 warning。"""

    model = _FakeFcModel()
    tokenizer = model.fc_duplex.protocol_tokenizer
    think_start = tokenizer.token_to_id("<think>")
    think_end = tokenizer.token_to_id("</think>")
    text_ids = tokenizer.encode_ordinary("龘")
    assert len(text_ids) == 2
    model.non_spoken_results = [
        {"token_ids": [think_start, text_ids[0]]},
        {
            "token_ids": [think_end],
            "closed_spans": [{"type": "think", "text": "\ufffd"}],
        },
    ]
    view = FcDuplexView(model)  # type: ignore[arg-type]
    view.prepare(FcDuplexPrepareRequest())
    view.streaming_non_spoken_generate(FcNonSpokenGenerateRequest())

    closed = view.streaming_non_spoken_generate(FcNonSpokenGenerateRequest())

    assert len(closed.warnings) == 1
    assert closed.warnings[0].reason == "think_end"


def test_tool_state_blocks_resume_only_while_valid_call_waits_for_result() -> None:
    """Parse error 和已回填调用不应永久触发 unsupported_tool_state。"""

    manager = ToolCallStateManager()
    manager.register_parse_error("malformed")
    assert manager.has_state is True
    manager.consume_pending_started_events()
    manager.consume_pending_error_responses()
    assert manager.has_state is False

    call_id = manager.register_tool_call(
        {
            "name": "display_object_on_board",
            "arguments": '{"name":"小白兔"}',
        }
    )
    assert manager.has_state is True
    manager.consume_pending_started_events()

    manager.validate_and_mark_responses(
        [{"call_id": call_id, "content": "displayed"}]
    )
    assert manager.has_state is False

    replayed_manager = ToolCallStateManager()
    replayed_manager.restore_completed_sequence(1)
    assert replayed_manager.register_tool_call({"name": "next"}) == "fc_call_000002"
