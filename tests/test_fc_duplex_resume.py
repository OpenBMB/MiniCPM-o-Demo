"""FC Duplex generation step 日志与 stateless resume 测试。

测试只使用真实 SDK tokenizer bundle 和纯 CPU canonicalizer，不加载模型。
"""

from __future__ import annotations

import pytest

from core.fc_duplex_resume import (
    FcDuplexResumeError,
    build_fc_duplex_resume_plan,
)
from minicpm_o5_sdk import O5TokenizerID, load_builtin_tokenizer


def _fingerprint(tokenizer_target: str) -> dict[str, str]:
    """返回测试 target 的 public resume fingerprint。"""

    tokenizer = load_builtin_tokenizer(O5TokenizerID(tokenizer_target))
    return {
        "vocab_hash": tokenizer.fingerprint.vocab_hash,
        "merges_hash": tokenizer.fingerprint.merges_hash,
    }


def _history_for_cross_unit_text(*, checkpoint_status: str = "available") -> tuple[list[dict], list[int]]:
    """构造一个普通字符跨两个 Unit 才完成的最小公共历史。"""

    tokenizer = load_builtin_tokenizer(O5TokenizerID.O45_FC)
    text = "龘"
    token_ids = tokenizer.encode_ordinary(text)
    assert len(token_ids) == 2
    history = [
        {
            "type": "session.init",
            "payload": {
                "mode": "full_duplex",
                "fc_duplex": True,
                "tokenizer_target": "o45_fc",
            },
        },
        {"type": "input.append", "input": {"input_id": "u0", "audio_base64": "AAAAAA=="}},
        {
            "type": "response.generation.step_batch",
            "event_index": 0,
            "batch_index": 0,
            "stream_id": "think_1",
            "track": "non_spoken",
            "steps": [
                {
                    "step_index": 0,
                    "unit_index": 0,
                    "output": {"kind": "protocol", "semantic_key": "think_start"},
                }
            ],
        },
        {
            "type": "response.generation.step_batch",
            "event_index": 1,
            "batch_index": 1,
            "stream_id": "think_1",
            "track": "non_spoken",
            "steps": [
                {
                    "step_index": 1,
                    "unit_index": 0,
                    "output": {"kind": "text_pending"},
                }
            ],
        },
        {
            "type": "response.unit.committed",
            "event_index": 2,
            "unit_index": 0,
            "input_id": "u0",
            "last_step_index": 1,
            "resume": {
                "status": "unavailable",
                "reason": "pending_text_delta",
                "stream_id": "think_1",
                "pending_from_step": 1,
            },
        },
        {
            "type": "input.append",
            "input": {
                "input_id": "dropped_by_latency_queue",
                "audio_base64": "AQAAAA==",
            },
        },
        {"type": "input.append", "input": {"input_id": "u1", "audio_base64": "AAAAAA=="}},
        {
            "type": "response.generation.step_batch",
            "event_index": 3,
            "batch_index": 2,
            "stream_id": "spoken_protocol",
            "track": "spoken",
            "steps": [
                {
                    "step_index": 2,
                    "unit_index": 1,
                    "output": {"kind": "protocol", "semantic_key": "listen"},
                }
            ],
        },
        {
            "type": "response.generation.step_batch",
            "event_index": 4,
            "batch_index": 3,
            "stream_id": "think_1",
            "track": "non_spoken",
            "steps": [
                {
                    "step_index": 3,
                    "unit_index": 1,
                    "output": {
                        "kind": "text_delta",
                        "delta_index": 0,
                        "text": text,
                        "source_step_indices": [1, 3],
                    },
                }
            ],
        },
        {
            "type": "response.generation.step_batch",
            "event_index": 5,
            "batch_index": 4,
            "stream_id": "think_1",
            "track": "non_spoken",
            "steps": [
                {
                    "step_index": 4,
                    "unit_index": 1,
                    "output": {"kind": "protocol", "semantic_key": "think_end"},
                }
            ],
        },
        {
            "type": "response.unit.committed",
            "event_index": 6,
            "unit_index": 1,
            "input_id": "u1",
            "last_step_index": 4,
            "resume": {"status": checkpoint_status},
        },
    ]
    return history, token_ids


def test_resume_plan_reencodes_cross_unit_delta_to_original_token_steps() -> None:
    """安全 delta 应恢复原 token IDs，并保持它们各自所属 Unit。"""

    history, token_ids = _history_for_cross_unit_text()

    plan = build_fc_duplex_resume_plan(
        protocol_version="fc-duplex-resume-v1",
        model="minicpm-o-4.5",
        tokenizer_target="o45_fc",
        tokenizer_fingerprint=_fingerprint("o45_fc"),
        through_unit_index=1,
        history=history,
    )

    assert plan.through_unit_index == 1
    assert len(plan.units) == 2
    assert plan.units[1].input_payload["input_id"] == "u1"
    assert plan.seen_input_ids == ["u0", "dropped_by_latency_queue", "u1"]
    assert plan.units[0].non_spoken_token_ids[-1] == token_ids[0]
    assert token_ids[1] in plan.units[1].non_spoken_token_ids


def test_resume_plan_rejects_pending_checkpoint() -> None:
    """请求停在尚无安全文本的 Unit 边界时必须明确失败。"""

    history, _ = _history_for_cross_unit_text()

    with pytest.raises(FcDuplexResumeError) as exc_info:
        build_fc_duplex_resume_plan(
            protocol_version="fc-duplex-resume-v1",
            model="minicpm-o-4.5",
            tokenizer_target="o45_fc",
            tokenizer_fingerprint=_fingerprint("o45_fc"),
            through_unit_index=0,
            history=history,
        )

    assert exc_info.value.code == "non_resumable_text_boundary"
    assert exc_info.value.unit_index == 0


def test_resume_plan_rejects_target_mismatch() -> None:
    """历史 tokenizer target 与请求 target 不一致时不得尝试恢复。"""

    history, _ = _history_for_cross_unit_text()

    with pytest.raises(FcDuplexResumeError) as exc_info:
        build_fc_duplex_resume_plan(
            protocol_version="fc-duplex-resume-v1",
            model="minicpm-o-4.5",
            tokenizer_target="o5",
            tokenizer_fingerprint={"vocab_hash": "unused", "merges_hash": "unused"},
            through_unit_index=1,
            history=history,
        )

    assert exc_info.value.code == "model_or_tokenizer_mismatch"


def test_resume_plan_rejects_missing_step_history() -> None:
    """safe delta 引用缺失 generation step 时必须失败，不能猜测 token 归属。"""

    history, _ = _history_for_cross_unit_text()
    first_batch_index = next(
        index
        for index, event in enumerate(history)
        if event.get("type") == "response.generation.step_batch"
    )
    history.pop(first_batch_index)

    with pytest.raises(FcDuplexResumeError) as exc_info:
        build_fc_duplex_resume_plan(
            protocol_version="fc-duplex-resume-v1",
            model="minicpm-o-4.5",
            tokenizer_target="o45_fc",
            tokenizer_fingerprint=_fingerprint("o45_fc"),
            through_unit_index=1,
            history=history,
        )

    assert exc_info.value.code == "incomplete_event_history"


def test_resume_plan_rejects_malformed_audio_instead_of_replaying_silence() -> None:
    """声明了 audio 但没有合法 payload 时不得静默按无音频 Unit 重放。"""

    history, _ = _history_for_cross_unit_text()
    history[0 + 1]["input"] = {
        "input_id": "u0",
        "audio": {"garbage": "not-audio"},
    }

    with pytest.raises(FcDuplexResumeError) as exc_info:
        build_fc_duplex_resume_plan(
            protocol_version="fc-duplex-resume-v1",
            model="minicpm-o-4.5",
            tokenizer_target="o45_fc",
            tokenizer_fingerprint=_fingerprint("o45_fc"),
            through_unit_index=1,
            history=history,
        )

    assert exc_info.value.code == "incomplete_event_history"


def test_resume_plan_rejects_spoken_after_non_spoken_in_same_unit() -> None:
    """Public history 的 Unit 内 track 顺序必须与 replay skeleton 一致。"""

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
            "type": "response.generation.step_batch",
            "event_index": 0,
            "batch_index": 0,
            "stream_id": "non_spoken_protocol",
            "track": "non_spoken",
            "steps": [
                {
                    "step_index": 0,
                    "unit_index": 0,
                    "output": {"kind": "protocol", "semantic_key": "no_action"},
                }
            ],
        },
        {
            "type": "response.generation.step_batch",
            "event_index": 1,
            "batch_index": 1,
            "stream_id": "spoken_protocol",
            "track": "spoken",
            "steps": [
                {
                    "step_index": 1,
                    "unit_index": 0,
                    "output": {"kind": "protocol", "semantic_key": "listen"},
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

    with pytest.raises(FcDuplexResumeError) as exc_info:
        build_fc_duplex_resume_plan(
            protocol_version="fc-duplex-resume-v1",
            model="minicpm-o-4.5",
            tokenizer_target="o45_fc",
            tokenizer_fingerprint=_fingerprint("o45_fc"),
            through_unit_index=0,
            history=history,
        )

    assert exc_info.value.code == "incomplete_event_history"


def test_resume_plan_preserves_deferred_budget_feed_for_later_safe_checkpoint() -> None:
    """早期 budget close 应按 next-prefill 时序 replay，不永久污染后续 checkpoint。"""

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
                    "output": {
                        "kind": "protocol",
                        "semantic_key": "non_spoken_budget_reached",
                        "deferred_model_feed": True,
                    },
                }
            ],
        },
        {
            "type": "response.unit.committed",
            "event_index": 2,
            "unit_index": 0,
            "input_id": "u0",
            "last_step_index": 1,
            "resume": {
                "status": "unavailable",
                "reason": "unsupported_deferred_close",
            },
        },
        {
            "type": "input.append",
            "input": {"input_id": "u1", "audio_base64": "AAAAAA=="},
        },
        {
            "type": "response.generation.step_batch",
            "event_index": 3,
            "batch_index": 2,
            "stream_id": "spoken_protocol",
            "track": "spoken",
            "steps": [
                {
                    "step_index": 2,
                    "unit_index": 1,
                    "output": {"kind": "protocol", "semantic_key": "listen"},
                }
            ],
        },
        {
            "type": "response.generation.step_batch",
            "event_index": 4,
            "batch_index": 3,
            "stream_id": "non_spoken_protocol",
            "track": "non_spoken",
            "steps": [
                {
                    "step_index": 3,
                    "unit_index": 1,
                    "output": {"kind": "protocol", "semantic_key": "no_action"},
                }
            ],
        },
        {
            "type": "response.unit.committed",
            "event_index": 5,
            "unit_index": 1,
            "input_id": "u1",
            "last_step_index": 3,
            "resume": {"status": "available"},
        },
    ]

    plan = build_fc_duplex_resume_plan(
        protocol_version="fc-duplex-resume-v1",
        model="minicpm-o-4.5",
        tokenizer_target="o45_fc",
        tokenizer_fingerprint=_fingerprint("o45_fc"),
        through_unit_index=1,
        history=history,
    )

    assert plan.units[0].deferred_non_spoken_close is True
    assert plan.units[1].deferred_non_spoken_close is False


def test_resume_plan_rejects_ordinary_text_without_semantic_opener() -> None:
    """Replay 不得接受 live View 会拒绝的无 opener ordinary stream。"""

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
            "stream_id": "think_1",
            "track": "non_spoken",
            "steps": [
                {
                    "step_index": 1,
                    "unit_index": 0,
                    "output": {
                        "kind": "text_delta",
                        "delta_index": 0,
                        "text": "a",
                        "source_step_indices": [1],
                    },
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

    with pytest.raises(FcDuplexResumeError) as exc_info:
        build_fc_duplex_resume_plan(
            protocol_version="fc-duplex-resume-v1",
            model="minicpm-o-4.5",
            tokenizer_target="o45_fc",
            tokenizer_fingerprint=_fingerprint("o45_fc"),
            through_unit_index=0,
            history=history,
        )

    assert exc_info.value.code == "incomplete_event_history"


def test_resume_plan_accepts_turn_eos_followed_by_synthetic_slot_eos() -> None:
    """Live turn_eos→synthetic slot_eos 顺序必须可 canonicalize/replay。"""

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
    ]
    semantic_keys = [
        ("spoken", "spoken_1", "speak"),
        ("spoken", "spoken_1", "spoken_turn_eos"),
        ("spoken", "spoken_1", "spoken_slot_eos"),
        ("non_spoken", "non_spoken_protocol", "no_action"),
    ]
    for index, (track, stream_id, semantic_key) in enumerate(semantic_keys):
        history.append(
            {
                "type": "response.generation.step_batch",
                "event_index": index,
                "batch_index": index,
                "stream_id": stream_id,
                "track": track,
                "steps": [
                    {
                        "step_index": index,
                        "unit_index": 0,
                        "output": {
                            "kind": "protocol",
                            "semantic_key": semantic_key,
                        },
                    }
                ],
            }
        )
    history.append(
        {
            "type": "response.unit.committed",
            "event_index": 4,
            "unit_index": 0,
            "input_id": "u0",
            "last_step_index": 3,
            "resume": {"status": "available"},
        }
    )

    plan = build_fc_duplex_resume_plan(
        protocol_version="fc-duplex-resume-v1",
        model="minicpm-o-4.5",
        tokenizer_target="o45_fc",
        tokenizer_fingerprint=_fingerprint("o45_fc"),
        through_unit_index=0,
        history=history,
    )

    assert plan.through_unit_index == 0
