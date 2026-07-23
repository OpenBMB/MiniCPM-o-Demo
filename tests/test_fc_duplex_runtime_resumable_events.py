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
async def test_runtime_uses_explicit_profile_budgets_by_spoken_state() -> None:
    """Runtime 应按当前 Unit 的 spoken 决策选择 Profile 中对应 budget。"""

    async def send(_: str, **__: Any) -> None:
        return

    runtime = FcDuplexSessionRuntime(
        session_id="sess_profile_budget",
        backend=_FakeRuntimeBackend(),
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

    assert runtime._select_non_spoken_budget(SimpleNamespace(is_speaking=False)) == 30
    assert runtime._select_non_spoken_budget(SimpleNamespace(is_speaking=True)) == 15
    assert runtime.resume_identity["checkpoint_profile_id"] == "profile_test"
    assert runtime.resume_identity["non_spoken_budget_while_listening"] == 30
    assert runtime.resume_identity["non_spoken_budget_while_speaking"] == 15


@pytest.mark.asyncio
async def test_runtime_rejects_missing_checkpoint_profile_budget() -> None:
    """未显式提供 Profile budget 时不能回退到某个 checkpoint 专属默认值。"""

    async def send(_: str, **__: Any) -> None:
        return

    runtime = FcDuplexSessionRuntime(
        session_id="sess_missing_profile_budget",
        backend=_FakeRuntimeBackend(),
        send=send,
    )

    with pytest.raises(RuntimeError, match="non-spoken budgets"):
        await runtime.prepare({})


@pytest.mark.asyncio
async def test_runtime_uses_launcher_profile_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session 未覆盖时应使用 launcher 注入的 Profile 身份和两类 budget。"""

    async def send(_: str, **__: Any) -> None:
        return

    monkeypatch.setenv("CHECKPOINT_PROFILE_ID", "profile_env")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_SCHEDULING", "quality")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING", "30")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING", "15")
    runtime = FcDuplexSessionRuntime(
        session_id="sess_profile_env",
        backend=_FakeRuntimeBackend(),
        send=send,
    )

    await runtime.prepare({})

    assert runtime.resume_identity["checkpoint_profile_id"] == "profile_env"
    assert runtime._select_non_spoken_budget(SimpleNamespace(is_speaking=False)) == 30
    assert runtime._select_non_spoken_budget(SimpleNamespace(is_speaking=True)) == 15


@pytest.mark.asyncio
async def test_runtime_rejects_session_budget_conflicting_with_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session 参数不能静默覆盖 launcher 已绑定的 Checkpoint Profile。"""

    async def send(_: str, **__: Any) -> None:
        return

    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING", "30")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING", "15")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_SCHEDULING", "quality")
    runtime = FcDuplexSessionRuntime(
        session_id="sess_profile_conflict",
        backend=_FakeRuntimeBackend(),
        send=send,
    )

    with pytest.raises(RuntimeError, match="conflicts with Checkpoint Profile"):
        await runtime.prepare(
            {
                "config": {
                    "non_spoken_budget_while_listening": 12,
                    "non_spoken_budget_while_speaking": 12,
                }
            }
        )


@pytest.mark.asyncio
async def test_runtime_rejects_scheduling_conflicting_with_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session 不能把 Profile 绑定的 quality 静默切成 latency。"""

    async def send(_: str, **__: Any) -> None:
        return

    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_SCHEDULING", "quality")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING", "30")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING", "15")
    runtime = FcDuplexSessionRuntime(
        session_id="sess_scheduling_conflict",
        backend=_FakeRuntimeBackend(),
        send=send,
    )

    with pytest.raises(RuntimeError, match="conflicts with Checkpoint Profile"):
        await runtime.prepare({"config": {"non_spoken_scheduling": "latency"}})


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

    await runtime._begin_block("think", unit_index=0)
    events.clear()
    await runtime._record_generation_steps(steps, unit_index=0, input_id="u0")
    await runtime._flush_generation_batch()

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "response.think.delta"
    assert event["unit_index"] == 0
    assert event["steps"] == [
        {"kind": "pending"},
        {"kind": "text", "text": "龘", "source_steps": 2},
    ]
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
            "input_id": "u0",
            "audio_base64": "",
            "sample_rate": 16000,
        }
    )

    assert [event["type"] for event in events] == [
        "response.unit.started",
        "response.spoken.end",
        "response.unit.committed",
    ]
    assert events[0]["tool_events"] == []
    checkpoint = events[-1]
    assert checkpoint["type"] == "response.unit.committed"
    assert checkpoint["unit_index"] == 0
    assert checkpoint["non_spoken_end"] == "no_action"
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
                "checkpoint_profile_id": "profile_test",
                "tokenizer_target": "o45_fc",
                "generate_audio": False,
                "config": {
                    "non_spoken_scheduling": "quality",
                    "non_spoken_budget_while_listening": 30,
                    "non_spoken_budget_while_speaking": 15,
                },
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

    assert runtime._unit_non_spoken_end == "budget_reached"
    assert not any(
        event["type"] in {
            "response.generation.step_batch",
            "response.output.sp_tokens",
        }
        for event in events
    )
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
    await runtime._begin_block("think", unit_index=0)
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
        event["type"] == "response.think.end"
        and "\ufffd" in str(event.get("full_text") or "")
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
            "type": "response.unit.started",
            "input_id": "actual_processed_input",
            "unit_index": 7,
            "tool_events": [
                {
                    "type": "tool_started",
                    "tool_call_id": "tc_000002",
                },
                {
                    "type": "tool_result",
                    "tool_call_id": "tc_000002",
                },
            ],
        }
    ]
