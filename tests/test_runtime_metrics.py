import logging

from core.runtime.duplex import DuplexFrameResult
from core.runtime.metrics import log_duplex_frame
from tests.test_duplex_runtime import _FakeResult


def test_log_duplex_frame_listen(caplog):
    frame = DuplexFrameResult(
        result=_FakeResult(),
        result_dict={},
        prefill_ms=12,
        prefill_result={},
        metrics={"kv_cache_length": 34},
        wall_clock_ms=56,
        n_vision_images=1,
        vision_tokens=64,
    )

    with caplog.at_level(logging.INFO):
        log_duplex_frame(logging.getLogger("test_runtime_metrics"), frame, gpu_id=0)

    assert "LISTEN" in caplog.text
    assert "kv=34" in caplog.text


def test_log_duplex_frame_speak(caplog):
    result = _FakeResult()
    result.is_listen = False
    result.text = "hello world"
    frame = DuplexFrameResult(
        result=result,
        result_dict={},
        prefill_ms=12,
        prefill_result={},
        metrics={"kv_cache_length": 34},
        wall_clock_ms=56,
        n_vision_images=0,
        vision_tokens=0,
    )

    with caplog.at_level(logging.INFO):
        log_duplex_frame(logging.getLogger("test_runtime_metrics"), frame, gpu_id=1)

    assert "SPEAK" in caplog.text
    assert "hello world" in caplog.text

