import logging

from core.runtime.worker_protocol import DuplexFrameResult
from core.runtime.metrics import log_duplex_frame


class _FakeResult:
    is_listen = True
    text = ""
    audio_data = None
    current_time = 1
    cost_all_ms = 1.0
    cost_llm_ms = None
    cost_tts_prep_ms = None
    cost_tts_ms = None
    cost_token2wav_ms = None
    n_tokens = None
    n_tts_tokens = None


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

