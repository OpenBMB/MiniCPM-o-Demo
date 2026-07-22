"""FC Duplex runtime canonical generation batch 与 Unit checkpoint 测试。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from core.schemas.fc_duplex import (
    FcClosedSpan,
    FcGenerationProtocolOutput,
    FcGenerationStreamTerminationResult,
    FcGenerationTextDeltaOutput,
    FcGenerationTextPendingOutput,
    FcGenerationWarning,
    FcViewGenerationStep,
    NonSpokenStepGenerationFlag,
)
from minicpm_o5_sdk import O5TokenizerID, load_builtin_tokenizer
from py_backend.fc_duplex_runtime import FcDuplexSessionRuntime


class _FakeRuntimeBackend:
    """提供一个可完成单 Unit 的 backend stub。"""

    def __init__(self) -> None:
        self.replayed_units: list[dict[str, Any]] = []
        self.next_stream_sequence: int | None = None

    def fc_duplex_prepare(self, **_: Any) -> None:
        return

    def fc_duplex_prefill(self, **_: Any) -> None:
        return

    def fc_duplex_spoken_generate(self, **_: Any) -> Any:
        return SimpleNamespace(
            is_listen=True,
            is_speaking=False,
            spoken_text="",
            spoken_text_delta="",
            spoken_turn_eos=False,
            audio_waveform=None,
            audio_sample_rate=None,
            generation_steps=[
                FcViewGenerationStep(
                    token_id=1,
                    stream_id="spoken_protocol",
                    track="spoken",
                    output=FcGenerationProtocolOutput(semantic_key="listen"),
                )
            ],
        )

    def fc_duplex_non_spoken_generate(self, **_: Any) -> Any:
        return SimpleNamespace(
            token_ids=[2],
            text="",
            text_delta="",
            close_reason="no_action",
            terminated=True,
            closed_spans=[],
            generation_flag=NonSpokenStepGenerationFlag.no_action,
            generation_steps=[
                FcViewGenerationStep(
                    token_id=2,
                    stream_id="non_spoken_protocol",
                    track="non_spoken",
                    output=FcGenerationProtocolOutput(semantic_key="no_action"),
                )
            ],
        )

    def fc_duplex_finalize(self) -> None:
        return

    def fc_duplex_resume_boundary_status(self) -> dict[str, str]:
        return {"status": "available"}

    def fc_duplex_replay_completed_unit(self, **fields: Any) -> None:
        self.replayed_units.append(fields)

    def fc_duplex_restore_generation_stream_sequence(
        self,
        *,
        next_stream_sequence: int,
    ) -> None:
        self.next_stream_sequence = next_stream_sequence

    def fc_duplex_cleanup(self) -> None:
        return

    def fc_duplex_terminate_non_spoken_text_stream(
        self,
        *,
        reason: str,
    ) -> FcGenerationStreamTerminationResult:
        return FcGenerationStreamTerminationResult(
            generation_steps=[
                FcViewGenerationStep(
                    token_id=99,
                    stream_id="think_1",
                    track="non_spoken",
                    output=FcGenerationProtocolOutput(
                        semantic_key="non_spoken_budget_reached",
                        deferred_model_feed=True,
                    ),
                )
            ],
            warnings=[
                FcGenerationWarning(
                    code="incomplete_bpe_at_stream_end",
                    stream_id="think_1",
                    track="non_spoken",
                    reason=reason,
                    message="文本边界包含未完成 BPE，公共 API 历史无法保证精确复现",
                )
            ],
        )


@pytest.mark.asyncio
async def test_runtime_batches_safe_text_steps_without_exposing_token_ids() -> None:
    """Batch 应保留 pending/delta 次级边界，但公共 step 不包含 token_id。"""

    events: list[dict[str, Any]] = []

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_test",
        backend=_FakeRuntimeBackend(),
        send=send,
    )
    steps = [
        FcViewGenerationStep(
            token_id=100,
            stream_id="think_1",
            track="non_spoken",
            output=FcGenerationTextPendingOutput(),
        ),
        FcViewGenerationStep(
            token_id=101,
            stream_id="think_1",
            track="non_spoken",
            output=FcGenerationTextDeltaOutput(
                text="龘",
                source_step_count=2,
            ),
        ),
    ]

    await runtime._record_generation_steps(steps, unit_index=0, input_id="u0")
    await runtime._flush_generation_batch()

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "response.generation.step_batch"
    assert event["event_index"] == 0
    assert [step["output"]["kind"] for step in event["steps"]] == [
        "text_pending",
        "text_delta",
    ]
    assert event["steps"][1]["output"]["source_step_indices"] == [0, 1]
    assert all("token_id" not in step for step in event["steps"])


@pytest.mark.asyncio
async def test_runtime_emits_available_unit_checkpoint_after_canonical_steps() -> None:
    """安全 Unit 应在 step batch 后发送带连续 event_index 的 checkpoint。"""

    events: list[dict[str, Any]] = []

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_test",
        backend=_FakeRuntimeBackend(),
        send=send,
    )
    await runtime._process_audio_payload(
        {
            "input_id": "u0",
            "audio_base64": "",
            "sample_rate": 16000,
        }
    )

    canonical_events = [
        event
        for event in events
        if event["type"]
        in {
            "response.unit.input_events",
            "response.generation.step_batch",
            "response.unit.committed",
        }
    ]
    assert [event["event_index"] for event in canonical_events] == [0, 1, 2, 3]
    assert canonical_events[0]["events"] == []
    checkpoint = canonical_events[-1]
    assert checkpoint["type"] == "response.unit.committed"
    assert checkpoint["unit_index"] == 0
    assert checkpoint["last_step_index"] == 1
    assert checkpoint["resume"] == {"status": "available"}


@pytest.mark.asyncio
async def test_runtime_resume_replays_public_history_and_continues_indices() -> None:
    """session.resume 应重建安全 Unit，并从 checkpoint 后的序号继续。"""

    events: list[dict[str, Any]] = []
    backend = _FakeRuntimeBackend()

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_resumed",
        backend=backend,
        send=send,
    )
    tokenizer = load_builtin_tokenizer(O5TokenizerID.O45_FC)
    history = [
        {
            "type": "session.init",
            "payload": {
                "mode": "full_duplex",
                "fc_duplex": True,
                "tokenizer_target": "o45_fc",
                "generate_audio": False,
            },
        },
        {"type": "input.append", "input": {"input_id": "u0", "audio_base64": "AAAAAA=="}},
        {
            "type": "response.generation.step_batch",
            "event_index": 0,
            "batch_index": 0,
            "stream_id": "spoken_protocol",
            "track": "spoken",
            "steps": [
                {
                    "step_index": 0,
                    "unit_index": 0,
                    "output": {"kind": "protocol", "semantic_key": "listen"},
                }
            ],
        },
        {
            "type": "response.generation.step_batch",
            "event_index": 1,
            "batch_index": 1,
            "stream_id": "non_spoken_protocol",
            "track": "non_spoken",
            "steps": [
                {
                    "step_index": 1,
                    "unit_index": 0,
                    "output": {"kind": "protocol", "semantic_key": "no_action"},
                }
            ],
        },
        {
            "type": "response.unit.committed",
            "event_index": 2,
            "unit_index": 0,
            "input_id": "u0",
            "last_step_index": 1,
            "resume": {"status": "available"},
        },
    ]

    await runtime.resume(
        {
            "protocol_version": "fc-duplex-resume-v1",
            "model": "minicpm-o-4.5",
            "tokenizer_target": "o45_fc",
            "tokenizer_fingerprint": {
                "vocab_hash": tokenizer.fingerprint.vocab_hash,
                "merges_hash": tokenizer.fingerprint.merges_hash,
            },
            "through_unit_index": 0,
            "history": history,
        }
    )

    assert len(backend.replayed_units) == 1
    replayed = backend.replayed_units[0]
    assert len(replayed["spoken_token_ids"]) == 1
    assert len(replayed["non_spoken_token_ids"]) == 1
    assert events[-1] == {
        "type": "session.resumed",
        "session_id": "sess_resumed",
        "through_unit_index": 0,
        "next_unit_index": 1,
    }
    assert runtime._generation_event_index == 3
    assert runtime._generation_step_index == 2
    assert runtime._generation_batch_index == 2
    assert backend.next_stream_sequence == 1


@pytest.mark.asyncio
async def test_runtime_rejects_missing_or_duplicate_input_id() -> None:
    """Live FC 输入必须有唯一 input_id，确保 checkpoint 能绑定真实处理 payload。"""

    async def send(_: str, **__: Any) -> None:
        return

    runtime = FcDuplexSessionRuntime(
        session_id="sess_input_ids",
        backend=_FakeRuntimeBackend(),
        send=send,
    )
    with pytest.raises(RuntimeError, match="requires input_id"):
        await runtime.enqueue_audio_input({"audio_base64": "AAAAAA=="})

    payload = {"input_id": "u0", "audio_base64": "AAAAAA=="}
    await runtime.enqueue_audio_input(payload)
    with pytest.raises(RuntimeError, match="duplicate"):
        await runtime.enqueue_audio_input(payload)
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_emits_budget_protocol_step_and_incomplete_bpe_warning() -> None:
    """Budget close 必须进入 canonical batch，并传输非致命 warning。"""

    events: list[dict[str, Any]] = []

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_warning",
        backend=_FakeRuntimeBackend(),
        send=send,
    )
    step = await runtime._build_deferred_budget_reached_step()
    await runtime._emit_step_events(step, input_id="u0", unit_index=0)

    batch = next(
        event
        for event in events
        if event["type"] == "response.generation.step_batch"
    )
    output = batch["steps"][0]["output"]
    assert output == {
        "kind": "protocol",
        "semantic_key": "non_spoken_budget_reached",
        "deferred_model_feed": True,
    }
    warning = next(
        event for event in events if event["type"] == "response.warning"
    )
    assert warning["code"] == "incomplete_bpe_at_stream_end"
    assert warning["reason"] == "budget_reached"


@pytest.mark.asyncio
async def test_runtime_does_not_leak_replacement_text_after_incomplete_bpe_warning() -> None:
    """Closed-span fallback 不得在 warning 后重新输出 lossy U+FFFD。"""

    events: list[dict[str, Any]] = []

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_lossy",
        backend=_FakeRuntimeBackend(),
        send=send,
    )
    await runtime._begin_block("think", input_id="u0")
    step = SimpleNamespace(
        token_ids=[],
        text_delta="",
        generation_steps=[],
        warnings=[
            FcGenerationWarning(
                code="incomplete_bpe_at_stream_end",
                stream_id="think_1",
                track="non_spoken",
                reason="think_end",
                message="文本边界包含未完成 BPE，公共 API 历史无法保证精确复现",
            )
        ],
        close_reason=None,
        terminated=False,
        closed_spans=[FcClosedSpan(type="think", text="\ufffd")],
    )

    await runtime._emit_step_events(step, input_id="u0", unit_index=0)

    assert any(event["type"] == "response.warning" for event in events)
    assert any(event["type"] == "response.think.end" for event in events)
    assert not any(
        event["type"] == "response.think.delta"
        and "\ufffd" in str(event.get("delta") or "")
        for event in events
    )


@pytest.mark.asyncio
async def test_detached_queue_runtime_error_invokes_session_fatal_callback() -> None:
    """后台 Unit 处理异常必须通知并关闭所属 Session，不能只留 task exception。"""

    fatal_errors: list[Exception] = []
    backend = _FakeRuntimeBackend()

    def fail_spoken(**_: Any) -> Any:
        raise RuntimeError("listen before spoken_turn_eos")

    backend.fc_duplex_spoken_generate = fail_spoken  # type: ignore[method-assign]

    async def send(_: str, **__: Any) -> None:
        return

    async def on_fatal(error: Exception) -> None:
        fatal_errors.append(error)

    runtime = FcDuplexSessionRuntime(
        session_id="sess_fatal",
        backend=backend,
        send=send,
        on_fatal=on_fatal,
    )
    await runtime.enqueue_audio_input(
        {
            "input_id": "u0",
            "audio_base64": "AAAAAA==",
            "sample_rate": 16000,
        }
    )
    assert runtime._queue_worker is not None
    await runtime._queue_worker

    assert len(fatal_errors) == 1
    assert "listen before spoken_turn_eos" in str(fatal_errors[0])
    assert runtime._closed is True


@pytest.mark.asyncio
async def test_runtime_emits_explicit_processed_unit_tool_event_attribution() -> None:
    """Tool events 必须由 backend 显式绑定到实际处理 Unit，不能靠时序推断。"""

    events: list[dict[str, Any]] = []

    async def send(event_type: str, **fields: Any) -> None:
        events.append({"type": event_type, **fields})

    runtime = FcDuplexSessionRuntime(
        session_id="sess_tool_events",
        backend=_FakeRuntimeBackend(),
        send=send,
    )
    runtime._internal_to_api["fc_call_000001"] = "tc_000002"
    prefill = SimpleNamespace(
        tool_events=[
            {"type": "tool_started", "call_id": "fc_call_000001"},
            {
                "type": "tool_response",
                "call_id": "fc_call_000001",
                "content": "displayed",
            },
        ]
    )

    await runtime._emit_unit_input_events(
        prefill,
        unit_index=7,
        input_id="actual_processed_input",
    )

    assert events == [
        {
            "type": "response.unit.input_events",
            "session_id": "sess_tool_events",
            "response_id": None,
            "input_id": "actual_processed_input",
            "event_index": 0,
            "unit_index": 7,
            "events": [
                {
                    "type": "tool_started",
                    "tool_call_id": "tc_000002",
                },
                {
                    "type": "tool_response",
                    "tool_call_id": "tc_000002",
                    "content": "displayed",
                },
            ],
        }
    ]
