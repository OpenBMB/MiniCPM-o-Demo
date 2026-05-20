from gateway_modules.runtime_protocol import (
    chat_client_to_worker_runtime,
    realtime_client_to_worker_runtime,
    worker_runtime_to_response_api,
    worker_runtime_to_realtime,
)


def test_chat_client_request_wraps_for_worker_runtime():
    request = {"messages": [{"role": "user", "content": "hi"}], "streaming": True}

    msg = chat_client_to_worker_runtime(request)

    assert msg == {
        "type": "chat.request",
        "payload": request,
    }


def test_worker_chat_runtime_messages_to_response_api():
    prefill = worker_runtime_to_response_api({
        "type": "chat.prefill_done",
        "payload": {"input_tokens": 3},
    })
    text = worker_runtime_to_response_api({
        "type": "chat.chunk",
        "payload": {"text_delta": "hi"},
    })
    audio = worker_runtime_to_response_api({
        "type": "chat.chunk",
        "payload": {"text_delta": "hi", "audio_base64": "pcm"},
    })
    done = worker_runtime_to_response_api({
        "type": "chat.done",
        "payload": {
            "text": "hello",
            "audio_base64": "wav",
            "generated_tokens": 2,
            "input_tokens": 3,
            "recording_session_id": "rec1",
        },
    })

    assert prefill == {"type": "response.prefill.done", "input_tokens": 3}
    assert text == {"type": "response.output_text.delta", "text": "hi"}
    assert audio == {"type": "response.output_audio.delta", "audio": "pcm", "text": "hi"}
    assert done == {
        "type": "response.completed",
        "text": "hello",
        "audio": "wav",
        "generated_tokens": 2,
        "input_tokens": 3,
        "recording_session_id": "rec1",
    }


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

    assert msg["type"] == "duplex.session.prepare"
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
        "type": "duplex.input.audio.append",
        "payload": {
            "audio_base64": "a",
            "frame_base64_list": ["f"],
            "force_listen": True,
            "max_slice_nums": None,
        },
    }


def test_worker_runtime_output_to_realtime_events():
    listen = worker_runtime_to_realtime({
        "type": "duplex.output.listen",
        "payload": {},
    }, session_id="rt_1")
    speak = worker_runtime_to_realtime({
        "type": "duplex.output.audio.delta",
        "payload": {
            "text": "hi",
            "audio_base64": "pcm",
            "end_of_turn": False,
        },
    }, session_id="rt_1")
    text = worker_runtime_to_realtime({
        "type": "duplex.output.text.delta",
        "payload": {"text": "hi"},
    }, session_id="rt_1")
    metrics = worker_runtime_to_realtime({
        "type": "duplex.metrics.frame",
        "payload": {"kv_cache_length": 4, "prefill_ms": 1.0},
    }, session_id="rt_1")

    assert listen == {"type": "response.listen"}
    assert speak == {
        "type": "response.output_audio.delta",
        "text": "hi",
        "audio": "pcm",
        "end_of_turn": False,
    }
    assert text == {"type": "response.output_text.delta", "text": "hi"}
    assert metrics == {"type": "response.metrics", "kv_cache_length": 4, "prefill_ms": 1.0}

