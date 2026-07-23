"""FC Board 从通用进程环境暴露 Checkpoint Profile 默认值的测试。"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_fc_board_defaults_expose_profile_runtime_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gateway defaults 应把 launcher 注入值透传给通用应用壳。"""

    import gateway

    monkeypatch.setenv("CHECKPOINT_PROFILE_ID", "profile_test")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_SCHEDULING", "quality")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING", "30")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING", "15")
    monkeypatch.setenv("FC_DUPLEX_UNIT_SEC", "1.0")
    monkeypatch.setattr(gateway, "_fc_board_case_folder", lambda: None)

    defaults = await gateway.fc_board_defaults()

    assert defaults["checkpoint_profile_id"] == "profile_test"
    assert defaults["non_spoken_scheduling"] == "quality"
    assert defaults["non_spoken_budget_while_listening"] == 30
    assert defaults["non_spoken_budget_while_speaking"] == 15
    assert defaults["unit_sec"] == 1.0
