"""Tool-calling (tools / tool_choice) protocol tests — issue #21.

These tests are MODEL-FREE: they exercise the real protocol/plumbing code
(parse, schema, message conversion, the ``_push_turn_based`` server path and
the tool-call output parser) without loading the 18 GB MiniCPM-o weights or a
GPU. The only optional, network-dependent check renders the prompt through the
real MiniCPM-o 4.5 tokenizer + chat template and is skipped when that
tokenizer is not available locally.

Run (no model, no network needed for the core checks)::

    python -m pytest tests/test_toolchoice.py -v

The last test (``test_real_template_renders_tools``) additionally passes when
a MiniCPM-o tokenizer is reachable, via the ``MINICPM_O_TOK`` env var or a
local clone. It is ``pytest.skip``-ped otherwise so CI stays green offline.

Coverage
--------
1. ``parse_worker_chat_request_message`` carries ``tools`` + ``tool_choice``
   (previously silently dropped — the root cause of #21).
2. ``tool_choice="none"`` withholds the tool block (``effective_tools is None``).
3. ``parse_raw_messages`` + ``convert_to_model_msgs`` carry ``role:"tool"`` and
   assistant ``tool_calls`` through to the model messages.
4. The real MiniCPM-o 4.5 tokenizer + chat template render the ``# Tools``
   block and a ``
```
 tool-result round-trip (optional / network).
5. ``_extract_tool_calls`` parses the fenced ``tool_call`` block (and the XML
   ``<tool_call>`` form) into OpenAI-style ``tool_calls`` + clean text.
6. End-to-end: the real ``_push_turn_based`` server method threads
   ``tools`` into ``chat_prefill``, emits ``tool_calls`` on ``response.done``,
   feeds a tool result back on turn 2, and honours ``tool_choice="none"``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

# Make the repo root importable when pytest runs from the tests/ dir.
_TESTS_DIR = Path(__file__).parent
_ROOT = _TESTS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from py_backend.chat_util import (  # noqa: E402
    _extract_tool_calls,
    convert_to_model_msgs,
    parse_raw_messages,
    parse_worker_chat_request_message,
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}

# The fenced form the chat template instructs the model to emit.
_BT = chr(96) * 3
_FENCE_OPEN = "\n" + _BT + "tool_call"
_FENCE_CLOSE = "\n" + _BT + "tool_call"


def _payload(**over):
    base = {
        "messages": [{"role": "user", "content": "What's the weather in Barcelona?"}],
        "streaming": False,
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "generation": {"max_new_tokens": 256},
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# 1 + 2. Protocol parse
# ---------------------------------------------------------------------------

def test_parse_carries_tools_and_tool_choice():
    req = parse_worker_chat_request_message(
        {"type": "chat.request", "payload": _payload()}
    )
    assert req.tools is not None and len(req.tools) == 1
    assert req.tools[0]["function"]["name"] == "get_weather"
    assert req.tool_choice == "auto"
    # effective_tools honours auto (passes tools through).
    assert req.effective_tools is not None and len(req.effective_tools) == 1


def test_parse_tool_choice_none_withholds_tools():
    req = parse_worker_chat_request_message(
        {"type": "chat.request", "payload": _payload(tool_choice="none")}
    )
    assert req.tools is not None  # raw tools are still captured...
    assert req.effective_tools is None  # ...but the tool block is withheld.


def test_parse_validates_tools_is_list_of_dicts():
    from py_backend.chat_util import ChatRequestError

    with pytest.raises(ChatRequestError):
        parse_worker_chat_request_message(
            {"type": "chat.request", "payload": _payload(tools="not-a-list")}
        )
    with pytest.raises(ChatRequestError):
        parse_worker_chat_request_message(
            {"type": "chat.request", "payload": _payload(tools=["also-not-a-dict"])}
        )


def test_parse_without_tools_is_backwards_compatible():
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "streaming": False,
        "generation": {},
    }
    req = parse_worker_chat_request_message({"type": "chat.request", "payload": payload})
    assert req.tools is None
    assert req.tool_choice is None
    assert req.effective_tools is None


# ---------------------------------------------------------------------------
# 3. Message conversion
# ---------------------------------------------------------------------------

def test_convert_carries_tool_role_and_tool_calls():
    raw = [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
        ]},
        {"role": "tool", "content": "{\"temp_c\": 18}"},
        {"role": "user", "content": "Thanks"},
    ]
    model_msgs = convert_to_model_msgs(parse_raw_messages(raw))
    roles = [m["role"] for m in model_msgs]
    assert "tool" in roles
    assistant = next(m for m in model_msgs if m["role"] == "assistant")
    assert assistant.get("tool_calls"), "assistant tool_calls must survive"
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"


def test_convert_normal_turn_unchanged():
    raw = [{"role": "user", "content": "hello"}]
    model_msgs = convert_to_model_msgs(parse_raw_messages(raw))
    assert model_msgs == [{"role": "user", "content": "hello"}]
    assert "tool_calls" not in model_msgs[0]


# ---------------------------------------------------------------------------
# 4. Real template render (optional / network)
# ---------------------------------------------------------------------------

def _try_load_tokenizer():
    candidates = []
    env = os.environ.get("MINICPM_O_TOK")
    if env:
        candidates.append(env)
    candidates += ["/tmp/minicpmo-tok", str(_ROOT / "MiniCPMO45")]
    from transformers import AutoTokenizer
    for path in candidates:
        if not path or not os.path.isdir(path):
            continue
        try:
            # The custom tokenizer class lives in MiniCPMO45/.
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            return tok
        except Exception:
            continue
    return None


def test_real_template_renders_tools():
    tok = _try_load_tokenizer()
    if tok is None:
        pytest.skip(
            "MiniCPM-o tokenizer not available locally; set MINICPM_O_TOK to run "
            "the real-template render check."
        )
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Weather in Barcelona?"}],
        tools=[WEATHER_TOOL], tokenize=False, add_generation_prompt=True,
    )
    assert "# Tools" in prompt
    assert "get_weather" in prompt
    # The template tells the model to emit a tool_call block.
    assert "tool_call" in prompt

    # Round-trip: feed a tool result back and confirm it renders.
    roundtrip = tok.apply_chat_template(
        [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
            ]},
            {"role": "tool", "content": "{\"temp_c\": 18}"},
            {"role": "user", "content": "Thanks"},
        ],
        tools=[WEATHER_TOOL], tokenize=False, add_generation_prompt=True,
    )
    assert "<tool_response>" in roundtrip
    assert '{"temp_c": 18}' in roundtrip
    assert "get_weather" in roundtrip


# ---------------------------------------------------------------------------
# 5. Output parser
# ---------------------------------------------------------------------------

def test_extract_fenced_tool_call():
    model_out = (
        "Here you go." + _FENCE_OPEN
        + '{"name": "get_weather", "arguments": {"city": "Barcelona"}}'
        + _FENCE_CLOSE
    )
    clean, calls = _extract_tool_calls(model_out)
    assert len(calls) == 1
    c = calls[0]
    assert c["type"] == "function"
    assert c["id"].startswith("call_")
    assert c["function"]["name"] == "get_weather"
    assert c["function"]["arguments"] == {"city": "Barcelona"}
    # Visible text does not leak the raw fence.
    assert "tool_call" not in clean
    assert clean.strip() == "Here you go."


def test_extract_xml_tool_call():
    open_tag = chr(60) + "tool_call" + chr(62)
    close_tag = chr(60) + "/tool_call" + chr(62)
    model_out = "checking" + "\n" + open_tag + \
        '{"name": "get_weather", "arguments": {"city": "Madrid"}}' + "\n" + close_tag
    clean, calls = _extract_tool_calls(model_out)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert calls[0]["function"]["arguments"] == {"city": "Madrid"}


def test_extract_arguments_as_string():
    # Double-encoded arguments: the model emitted the arguments object as a
    # JSON string inside the tool_call block. json.dumps keeps the escaping
    # correct.
    inner = json.dumps({"city": "Lisbon"})  # '{"city": "Lisbon"}'
    payload = json.dumps({"name": "get_weather", "arguments": inner})
    model_out = _FENCE_OPEN + payload + _FENCE_CLOSE
    _, calls = _extract_tool_calls(model_out)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    # A JSON-string argument is parsed into an object.
    assert calls[0]["function"]["arguments"] == {"city": "Lisbon"}


def test_extract_no_tool_call_is_passthrough():
    text = "Just a normal answer."
    clean, calls = _extract_tool_calls(text)
    assert calls == []
    assert clean == text


def test_extract_malformed_block_does_not_crash():
    model_out = "hi" + _FENCE_OPEN + "not json at all" + _FENCE_CLOSE
    clean, calls = _extract_tool_calls(model_out)
    assert calls == []  # malformed block skipped, not raised
    assert "not json" not in clean


# ---------------------------------------------------------------------------
# 6. End-to-end through the real _push_turn_based server path
# ---------------------------------------------------------------------------

class _FakeWS:
    def __init__(self):
        self.events = []

    async def send_json(self, data):
        self.events.append(data)


class _StubBackend:
    """Stands in ONLY for model inference; everything else is real code."""

    def __init__(self, first_output):
        self.prefill_kwargs = None
        self._first = first_output
        self._n = 0

    def metrics(self):
        return {"backend": "stub"}

    def chat_prefill(self, **kw):
        self.prefill_kwargs = kw
        return "prompt"

    def chat_non_streaming_generate(self, **kw):
        self._n += 1
        if self._n == 1:
            return self._first
        return "It is 18C in Barcelona."


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_e2e_tools_roundtrip_and_none():
    from py_backend.server import BackendProtocolSession

    model_out = (
        "Let me check.\n" + _FENCE_OPEN
        + '{"name": "get_weather", "arguments": {"city": "Barcelona"}}'
        + _FENCE_CLOSE
    )

    async def scenario():
        ws = _FakeWS()
        backend = _StubBackend(model_out)
        session = BackendProtocolSession(
            session_id="sess_e2e", mode="turn_based",
            backend=backend, ws=ws, state=None,
        )

        # TURN 1: ask about the weather with tools attached.
        await session._push_turn_based(_payload(response_id="resp1", input_id="in1"))

        # tools reached the prefill (the fix's core claim).
        assert backend.prefill_kwargs is not None
        assert backend.prefill_kwargs.get("tools")
        assert backend.prefill_kwargs["tools"][0]["function"]["name"] == "get_weather"

        done1 = next(e for e in ws.events if e.get("type") == "response.done")
        assert done1.get("tool_calls") and len(done1["tool_calls"]) == 1
        c = done1["tool_calls"][0]
        assert c["function"]["name"] == "get_weather"
        assert c["function"]["arguments"] == {"city": "Barcelona"}
        assert c["id"].startswith("call_")
        assert "tool_call" not in done1.get("text", "")

        # TURN 2: feed the tool result back.
        ws.events.clear()
        turn2_msgs = [
            {"role": "user", "content": "What's the weather in Barcelona?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_abc", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city": "Barcelona"}'}}
            ]},
            {"role": "tool", "content": '{"temp_c": 18, "condition": "sunny"}'},
            {"role": "user", "content": "Thanks, got it?"},
        ]
        await session._push_turn_based(
            _payload(messages=turn2_msgs, response_id="resp2", input_id="in2")
        )
        msgs2 = backend.prefill_kwargs.get("msgs", [])
        assert any(m.get("role") == "tool" for m in msgs2)
        assert any(m.get("tool_calls") for m in msgs2)
        done2 = next(e for e in ws.events if e.get("type") == "response.done")
        assert "tool_calls" not in done2  # final answer: no tool calls
        assert done2.get("text") == "It is 18C in Barcelona."

        # tool_choice="none" withholds the tool block.
        ws.events.clear()
        await session._push_turn_based(
            _payload(response_id="resp3", input_id="in3", tool_choice="none")
        )
        assert backend.prefill_kwargs.get("tools") is None

    _run(scenario())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
