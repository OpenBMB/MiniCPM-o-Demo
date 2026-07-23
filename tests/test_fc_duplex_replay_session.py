"""Recorded FC Session 按当前 Checkpoint Profile 重放的测试。"""

from __future__ import annotations

from types import SimpleNamespace

from scripts.fc_duplex_replay_session import apply_checkpoint_profile_to_init


def test_replay_rebinds_recorded_init_to_current_profile() -> None:
    """历史 Session 的旧 budget 必须被当前 Profile 参数完整替换。"""

    recorded = {
        "type": "session.init",
        "payload": {
            "config": {
                "non_spoken_scheduling": "latency",
                "non_spoken_budget_per_unit": 12,
            }
        },
    }
    args = SimpleNamespace(
        checkpoint_profile_id="profile_current",
        non_spoken_scheduling="quality",
        non_spoken_budget_while_listening=30,
        non_spoken_budget_while_speaking=15,
    )

    rebound = apply_checkpoint_profile_to_init(recorded, args)

    assert recorded["payload"]["config"]["non_spoken_budget_per_unit"] == 12
    assert rebound["payload"]["checkpoint_profile_id"] == "profile_current"
    assert rebound["payload"]["config"] == {
        "non_spoken_scheduling": "quality",
        "non_spoken_budget_while_listening": 30,
        "non_spoken_budget_while_speaking": 15,
    }
