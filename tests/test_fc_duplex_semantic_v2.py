"""Minimal FC Duplex semantic realtime API v2 contract tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from core.fc_duplex_resume import build_fc_duplex_resume_plan
from core.schemas.fc_duplex import (
    FcGenerationProtocolOutput,
    FcGenerationTextDeltaOutput,
    FcViewGenerationStep,
    NonSpokenStepGenerationFlag,
)
from minicpm_o5_sdk import O5TokenizerID, load_builtin_tokenizer
from py_backend.fc_duplex_runtime import FcDuplexSessionRuntime


class _SemanticV2Backend:
    """One-Unit listen/no-action backend stub."""

    def fc_duplex_prepare(self, **_: Any) -> None:
        """接受 Runtime 的通用 Session 初始化。"""

        return

    def fc_duplex_prefill(self, **_: Any) -> Any:
        return SimpleNamespace(tool_events=[])

    def fc_duplex_spoken_generate(self, **_: Any) -> Any:
        return SimpleNamespace(
            is_listen=True,
            is_speaking=False,
            spoken_token_ids=[1],
            spoken_text_delta="",
            spoken_full_text=None,
            spoken_turn_eos=False,
            audio_waveform=None,
            generation_steps=[
                FcViewGenerationStep(
                    token_id=1,
                    stream_id="spoken_protocol",
                    track="spoken",
                    output=FcGenerationProtocolOutput(
                        semantic_key="listen"
                    ),
                )
            ],
            warnings=[],
        )

    def fc_duplex_non_spoken_generate(self, **_: Any) -> Any:
        return SimpleNamespace(
            token_ids=[2],
            text_delta="",
            span_started=None,
            close_reason="no_action",
            terminated=True,
            closed_spans=[],
            generation_flag=NonSpokenStepGenerationFlag.no_action,
            generation_steps=[
                FcViewGenerationStep(
                    token_id=2,
                    stream_id="non_spoken_protocol",
                    track="non_spoken",
                    output=FcGenerationProtocolOutput(
                        semantic_key="no_action"
                    ),
                )
            ],
            warnings=[],
        )

    def fc_duplex_finalize(self) -> None:
        return

    def fc_duplex_resume_boundary_status(self) -> dict[str, str]:
        return {"status": "available"}


@pytest.mark.asyncio
async def test_runtime_emits_minimal_semantic_v2_events_only() -> None:
    """Default public wire must not emit v1 batch/sp-token/block envelopes."""

    events: list[dict[str, Any]] = []

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_v2",
        backend=_SemanticV2Backend(),
        send=send,
    )
    await runtime.prepare(
        {
            "checkpoint_profile_id": "profile_test",
            "config": {
                "non_spoken_scheduling": "quality",
                "non_spoken_budget_while_listening": 30,
                "non_spoken_budget_while_speaking": 15,
            },
        }
    )
    await runtime._process_audio_payload(
        {
            "input_id": "input_000000",
            "audio_base64": "",
            "sample_rate": 16000,
        }
    )

    event_types = [event["type"] for event in events]
    assert event_types == [
        "response.unit.started",
        "response.spoken.end",
        "response.non_spoken.end",
        "response.unit.committed",
    ]
    assert events[0] == {
        "type": "response.unit.started",
        "unit_index": 0,
        "input_id": "input_000000",
        "tool_events": [],
    }
    assert events[1] == {
        "type": "response.spoken.end",
        "unit_index": 0,
        "reason": "listen",
    }
    assert events[2] == {
        "type": "response.non_spoken.end",
        "unit_index": 0,
        "reason": "no_action",
    }
    assert events[3] == {
        "type": "response.unit.committed",
        "unit_index": 0,
        "resume": {"status": "available"},
    }
    assert all("block_id" not in event for event in events)
    assert all("response_id" not in event for event in events)
    assert all("session_id" not in event for event in events)


def test_semantic_v2_resume_rejects_missing_non_spoken_end() -> None:
    """Unit committed 前缺少 slot-level non_spoken.end 时必须拒绝。"""

    tokenizer = load_builtin_tokenizer(O5TokenizerID.O45_FC)
    history = [
        {
            "type": "session.init",
            "payload": {
                "mode": "full_duplex",
                "fc_duplex": True,
                "tokenizer_target": "o45_fc",
            },
        },
        {
            "type": "input.append",
            "input": {"input_id": "u0", "audio_base64": "AAAAAA=="},
        },
        {
            "type": "response.unit.started",
            "unit_index": 0,
            "input_id": "u0",
            "tool_events": [],
        },
        {"type": "response.spoken.end", "unit_index": 0, "reason": "listen"},
        {
            "type": "response.unit.committed",
            "unit_index": 0,
            "resume": {"status": "available"},
        },
    ]

    with pytest.raises(Exception, match="non_spoken.end"):
        build_fc_duplex_resume_plan(
            protocol_version="fc-duplex-semantic-v2",
            model="minicpm-o-4.5",
            tokenizer_target="o45_fc",
            tokenizer_fingerprint={
                "vocab_hash": tokenizer.fingerprint.vocab_hash,
                "merges_hash": tokenizer.fingerprint.merges_hash,
            },
            through_unit_index=0,
            history=history,
        )


def test_semantic_v2_resume_rejects_duplicate_non_spoken_end() -> None:
    """同一 Unit 重复发送 non_spoken.end 时必须拒绝。"""

    tokenizer = load_builtin_tokenizer(O5TokenizerID.O45_FC)
    end = {
        "type": "response.non_spoken.end",
        "unit_index": 0,
        "reason": "no_action",
    }
    history = [
        {
            "type": "session.init",
            "payload": {
                "mode": "full_duplex",
                "fc_duplex": True,
                "tokenizer_target": "o45_fc",
            },
        },
        {
            "type": "input.append",
            "input": {"input_id": "u0", "audio_base64": "AAAAAA=="},
        },
        {
            "type": "response.unit.started",
            "unit_index": 0,
            "input_id": "u0",
            "tool_events": [],
        },
        {"type": "response.spoken.end", "unit_index": 0, "reason": "listen"},
        end,
        dict(end),
    ]

    with pytest.raises(Exception, match="重复 non_spoken.end"):
        build_fc_duplex_resume_plan(
            protocol_version="fc-duplex-semantic-v2",
            model="minicpm-o-4.5",
            tokenizer_target="o45_fc",
            tokenizer_fingerprint={
                "vocab_hash": tokenizer.fingerprint.vocab_hash,
                "merges_hash": tokenizer.fingerprint.merges_hash,
            },
            through_unit_index=0,
            history=history,
        )


def test_semantic_v2_history_replays_think_stream_without_ids() -> None:
    """Ordered semantic events must be enough for stateless token replay."""

    tokenizer = load_builtin_tokenizer(O5TokenizerID.O45_FC)
    fingerprint = {
        "vocab_hash": tokenizer.fingerprint.vocab_hash,
        "merges_hash": tokenizer.fingerprint.merges_hash,
    }
    text_ids = tokenizer.encode_ordinary("用户在思考")
    history = [
        {
            "type": "session.init",
            "payload": {
                "mode": "full_duplex",
                "fc_duplex": True,
                "tokenizer_target": "o45_fc",
            },
        },
        {
            "type": "input.append",
            "input": {
                "input_id": "input_000000",
                "audio_base64": "AAAAAA==",
            },
        },
        {
            "type": "response.unit.started",
            "unit_index": 0,
            "input_id": "input_000000",
            "tool_events": [],
        },
        {
            "type": "response.spoken.end",
            "unit_index": 0,
            "reason": "listen",
        },
        {"type": "response.think.begin", "unit_index": 0},
        {
            "type": "response.think.delta",
            "unit_index": 0,
            "steps": [
                *[
                    {"kind": "pending"}
                    for _ in range(len(text_ids) - 1)
                ],
                {
                    "kind": "text",
                    "text": "用户在思考",
                }
            ],
        },
        {
            "type": "response.think.end",
            "unit_index": 0,
            "full_text": "用户在思考",
        },
        {
            "type": "response.non_spoken.end",
            "unit_index": 0,
            "reason": "eos",
        },
        {
            "type": "response.unit.committed",
            "unit_index": 0,
            "resume": {"status": "available"},
        },
    ]

    plan = build_fc_duplex_resume_plan(
        protocol_version="fc-duplex-semantic-v2",
        model="minicpm-o-4.5",
        tokenizer_target="o45_fc",
        tokenizer_fingerprint=fingerprint,
        through_unit_index=0,
        history=history,
    )

    assert plan.protocol_version == "fc-duplex-semantic-v2"
    assert text_ids[0] in plan.units[0].non_spoken_token_ids


def test_semantic_v2_replays_completed_tool_result_by_unit_marker() -> None:
    """Tool result content comes from input; Unit marker only records attribution."""

    tokenizer = load_builtin_tokenizer(O5TokenizerID.O45_FC)
    fingerprint = {
        "vocab_hash": tokenizer.fingerprint.vocab_hash,
        "merges_hash": tokenizer.fingerprint.merges_hash,
    }
    wire = (
        '<function name="display_object_on_board">'
        '<param name="name">老鼠</param></function>'
    )
    wire_ids = tokenizer.encode_ordinary(wire)
    history = [
        {
            "type": "session.init",
            "payload": {
                "mode": "full_duplex",
                "fc_duplex": True,
                "tokenizer_target": "o45_fc",
            },
        },
        {
            "type": "input.append",
            "input": {
                "input_id": "u0",
                "audio_base64": "AAAAAA==",
            },
        },
        {
            "type": "response.unit.started",
            "unit_index": 0,
            "input_id": "u0",
            "tool_events": [],
        },
        {"type": "response.spoken.end", "unit_index": 0, "reason": "listen"},
        {
            "type": "response.tool_call.begin",
            "tool_call_id": "tc_000001",
            "unit_index": 0,
        },
        {
            "type": "response.tool_call.delta",
            "tool_call_id": "tc_000001",
            "unit_index": 0,
            "steps": [
                *[{"kind": "pending"} for _ in range(len(wire_ids) - 1)],
                {
                    "kind": "text",
                    "text": wire,
                },
            ],
        },
        {
            "type": "response.tool_call.done",
            "tool_call_id": "tc_000001",
            "unit_index": 0,
            "full_text": wire,
            "call": {
                "name": "display_object_on_board",
                "arguments": {"name": "老鼠"},
            },
        },
        {
            "type": "response.non_spoken.end",
            "unit_index": 0,
            "reason": "eos",
        },
        {
            "type": "response.unit.committed",
            "unit_index": 0,
            "resume": {
                "status": "unavailable",
                "reason": "pending_tool_result",
            },
        },
        {
            "type": "input.tool_result",
            "tool_call_id": "tc_000001",
            "content": {"status": "displayed", "name": "老鼠"},
        },
        {
            "type": "input.append",
            "input": {
                "input_id": "u1",
                "audio_base64": "AAAAAA==",
            },
        },
        {
            "type": "response.unit.started",
            "unit_index": 1,
            "input_id": "u1",
            "tool_events": [
                {"type": "tool_started", "tool_call_id": "tc_000001"},
                {"type": "tool_result", "tool_call_id": "tc_000001"},
            ],
        },
        {"type": "response.spoken.end", "unit_index": 1, "reason": "listen"},
        {
            "type": "response.non_spoken.end",
            "unit_index": 1,
            "reason": "no_action",
        },
        {
            "type": "response.unit.committed",
            "unit_index": 1,
            "resume": {"status": "available"},
        },
    ]

    plan = build_fc_duplex_resume_plan(
        protocol_version="fc-duplex-semantic-v2",
        model="minicpm-o-4.5",
        tokenizer_target="o45_fc",
        tokenizer_fingerprint=fingerprint,
        through_unit_index=1,
        history=history,
    )

    assert plan.tool_call_count == 1
    assert plan.units[1].tool_events[1]["content"].startswith("{")


@pytest.mark.asyncio
async def test_runtime_merges_tool_end_and_raw_into_one_done_event() -> None:
    """Tool wire closes with one executable done event and no redundant envelopes."""

    events: list[dict[str, Any]] = []

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_tool_v2",
        backend=_SemanticV2Backend(),
        send=send,
    )
    await runtime._begin_block("tool_call", unit_index=3)
    step = FcViewGenerationStep(
        token_id=100,
        stream_id="tool_call_1",
        track="non_spoken",
        output=FcGenerationTextDeltaOutput(
            text="<function></function>",
            source_step_count=1,
        ),
    )
    await runtime._record_generation_steps(
        [step],
        unit_index=3,
        input_id="u3",
    )
    await runtime._flush_generation_batch()
    await runtime._emit_close_for_span(
        SimpleNamespace(
            type="tool_call",
            tool_call_id="fc_call_000001",
            wire="<function></function>",
            tool_call={
                "name": "display_object_on_board",
                "arguments": '{"name":"猫"}',
            },
            error=None,
        ),
        unit_index=3,
    )

    assert [event["type"] for event in events] == [
        "response.tool_call.begin",
        "response.tool_call.delta",
        "response.tool_call.done",
    ]
    done = events[-1]
    assert done["call"] == {
        "name": "display_object_on_board",
        "arguments": {"name": "猫"},
    }
    assert "raw" not in done
    assert "block_id" not in done


def test_semantic_v2_spoken_pending_crosses_unit_slot_end() -> None:
    """Spoken decoder pending survives template slot_end and resolves next Unit."""

    tokenizer = load_builtin_tokenizer(O5TokenizerID.O45_FC)
    fingerprint = {
        "vocab_hash": tokenizer.fingerprint.vocab_hash,
        "merges_hash": tokenizer.fingerprint.merges_hash,
    }
    text_ids = tokenizer.encode_ordinary("龘")
    assert len(text_ids) == 2
    history = [
        {
            "type": "session.init",
            "payload": {
                "mode": "full_duplex",
                "fc_duplex": True,
                "tokenizer_target": "o45_fc",
            },
        },
        {
            "type": "input.append",
            "input": {"input_id": "u0", "audio_base64": "AAAAAA=="},
        },
        {
            "type": "response.unit.started",
            "unit_index": 0,
            "input_id": "u0",
            "tool_events": [],
        },
        {
            "type": "response.spoken.delta",
            "unit_index": 0,
            "steps": [{"kind": "pending"}],
        },
        {"type": "response.spoken.end", "unit_index": 0, "reason": "slot_end"},
        {
            "type": "response.non_spoken.end",
            "unit_index": 0,
            "reason": "no_action",
        },
        {
            "type": "response.unit.committed",
            "unit_index": 0,
            "resume": {
                "status": "unavailable",
                "reason": "unsupported_spoken_turn_state",
            },
        },
        {
            "type": "input.append",
            "input": {"input_id": "u1", "audio_base64": "AAAAAA=="},
        },
        {
            "type": "response.unit.started",
            "unit_index": 1,
            "input_id": "u1",
            "tool_events": [],
        },
        {
            "type": "response.spoken.delta",
            "unit_index": 1,
            "steps": [
                {
                    "kind": "text",
                    "text": "龘",
                }
            ],
        },
        {
            "type": "response.spoken.end",
            "unit_index": 1,
            "reason": "turn_eos",
            "full_text": "龘",
        },
        {
            "type": "response.non_spoken.end",
            "unit_index": 1,
            "reason": "no_action",
        },
        {
            "type": "response.unit.committed",
            "unit_index": 1,
            "resume": {"status": "available"},
        },
    ]

    plan = build_fc_duplex_resume_plan(
        protocol_version="fc-duplex-semantic-v2",
        model="minicpm-o-4.5",
        tokenizer_target="o45_fc",
        tokenizer_fingerprint=fingerprint,
        through_unit_index=1,
        history=history,
    )

    assert text_ids[0] in plan.units[0].spoken_token_ids
    assert text_ids[1] in plan.units[1].spoken_token_ids


def test_semantic_v2_rejects_output_for_non_active_unit() -> None:
    """Semantic output cannot claim a future Unit that has not started."""

    tokenizer = load_builtin_tokenizer(O5TokenizerID.O45_FC)
    fingerprint = {
        "vocab_hash": tokenizer.fingerprint.vocab_hash,
        "merges_hash": tokenizer.fingerprint.merges_hash,
    }
    history = [
        {
            "type": "session.init",
            "payload": {
                "mode": "full_duplex",
                "fc_duplex": True,
                "tokenizer_target": "o45_fc",
            },
        },
        {
            "type": "input.append",
            "input": {"input_id": "u0", "audio_base64": "AAAAAA=="},
        },
        {
            "type": "response.unit.started",
            "unit_index": 0,
            "input_id": "u0",
            "tool_events": [],
        },
        {"type": "response.think.begin", "unit_index": 1},
    ]

    with pytest.raises(Exception, match="active Unit"):
        build_fc_duplex_resume_plan(
            protocol_version="fc-duplex-semantic-v2",
            model="minicpm-o-4.5",
            tokenizer_target="o45_fc",
            tokenizer_fingerprint=fingerprint,
            through_unit_index=0,
            history=history,
        )


@pytest.mark.asyncio
async def test_protocol_only_speak_emits_empty_spoken_delta() -> None:
    """SPEAK without text/audio still needs a semantic event for exact replay."""

    events: list[dict[str, Any]] = []

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_empty_speak",
        backend=_SemanticV2Backend(),
        send=send,
    )
    await runtime._emit_spoken(
        SimpleNamespace(
            is_listen=False,
            is_speaking=True,
            spoken_token_ids=[2],
            spoken_text_delta="",
            spoken_full_text=None,
            spoken_turn_eos=False,
            audio_waveform=None,
            generation_steps=[
                FcViewGenerationStep(
                    token_id=2,
                    stream_id="spoken_1",
                    track="spoken",
                    output=FcGenerationProtocolOutput(
                        semantic_key="speak"
                    ),
                )
            ],
            warnings=[],
        ),
        input_id="u0",
        unit_index=0,
    )

    assert events[0] == {
        "type": "response.spoken.delta",
        "unit_index": 0,
        "steps": [],
    }
    assert events[1]["type"] == "response.spoken.end"
