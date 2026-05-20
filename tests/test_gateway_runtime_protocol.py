from gateway_modules.runtime_protocol import (
    realtime_client_to_worker_runtime,
    worker_runtime_to_realtime,
)


def test_realtime_session_update_to_worker_prepare():
    msg = realtime_client_to_worker_runtime({
        "type": "session.update",
        "session": {
            "instructions": "hello",
            "max_slice_nums": 2,
            "ref_audio": "ref",
            "tts_ref_audio": "tts",
            "voice_config": {"temperature": 0.7},
        },
    })

    assert msg["type"] == "session.prepare"
    assert msg["payload"]["system_prompt"] == "hello"
    assert msg["payload"]["config"]["temperature"] == 0.7
    assert msg["payload"]["config"]["max_slice_nums"] == 2
    assert msg["payload"]["voice"]["ref_audio_base64"] == "ref"
    assert msg["payload"]["voice"]["tts_ref_audio_base64"] == "tts"


def test_realtime_input_to_worker_input_append():
    msg = realtime_client_to_worker_runtime({
        "type": "input_audio_buffer.append",
        "audio": "a",
        "video_frames": ["f"],
        "force_listen": True,
    })

    assert msg == {
        "type": "input.append",
        "payload": {
            "audio_base64": "a",
            "frame_base64_list": ["f"],
            "force_listen": True,
            "max_slice_nums": None,
        },
    }


def test_worker_runtime_output_to_realtime_events():
    listen = worker_runtime_to_realtime({
        "type": "runtime.event",
        "channel": "output.duplex_result",
        "payload": {"result": {"is_listen": True, "kv_cache_length": 3}},
    }, session_id="rt_1")
    speak = worker_runtime_to_realtime({
        "type": "runtime.event",
        "channel": "output.duplex_result",
        "payload": {
            "result": {
                "is_listen": False,
                "text": "hi",
                "audio_data": "pcm",
                "end_of_turn": False,
                "kv_cache_length": 4,
            }
        },
    }, session_id="rt_1")

    assert listen == {"type": "response.listen", "kv_cache_length": 3}
    assert speak == {
        "type": "response.output_audio.delta",
        "text": "hi",
        "audio": "pcm",
        "end_of_turn": False,
        "kv_cache_length": 4,
    }

