"""
Repro for OpenBMB/MiniCPM-o-Demo#21 — "tool_choice=auto doesn't work even in text-only mode".

Runs the repo's REAL chat-request parser (py_backend/chat_util.parse_worker_chat_request_message,
called on every turn at py_backend/server.py:292) with an OpenAI-style request that carries
`tools` + `tool_choice=auto`. No model / no GPU needed: the tool fields are handled (or not)
purely at the wire-protocol layer, before any model call.

Run:  cd <repo> && python3 repro_toolchoice.py
"""
import sys, json, dataclasses
sys.path.insert(0, ".")

from py_backend.chat_util import parse_worker_chat_request_message
from core.schemas.common import Message, GenerationConfig
from py_backend.server import app  # noqa: F403  (import to confirm no OpenAI HTTP route)

# ---- What an OpenAI-compatible client actually sends ---------------------
openai_payload = {
    "messages": [
        {"role": "user", "content": "What's the weather in Barcelona today?"},
    ],
    "streaming": False,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ],
    "tool_choice": "auto",
    "generation": {"max_new_tokens": 256},
}

# ---- 1) Feed the REAL parser the client's full payload --------------------
parsed = parse_worker_chat_request_message({"type": "chat.request", "payload": openai_payload})
print("=== [1] Fields the real WorkerChatRequest actually carries ===")
fields = [f.name for f in dataclasses.fields(parsed)]
print("carried fields :", fields)
print("has 'tools'     :", "tools" in fields)
print("has 'tool_choice':", "tool_choice" in fields)

tools_seen = getattr(parsed, "tools", None)
tool_choice_seen = getattr(parsed, "tool_choice", None)
print("parsed.tools    :", tools_seen)
print("parsed.tool_choice:", tool_choice_seen)

dropped = {k: v for k, v in openai_payload.items() if k not in fields and k not in ("messages",)}
print("\n=== [2] Client fields SILENTLY DROPPED by the protocol (not passed to the model) ===")
print(json.dumps(dropped, indent=2))

# ---- 3) Prove the schema layers have no tool concept ----------------------
print("\n=== [3] GenerationConfig fields (sampling layer — no tool fields) ===")
print([f for f in GenerationConfig.model_fields.keys()])

print("\n=== [4] Message schema fields (message layer — no tool_calls / role=tool) ===")
print([f for f in Message.model_fields.keys()])
import core.schemas.common as c
roles = [r.value for r in c.Role]
print("allowed roles  :", roles, "| 'tool' role allowed:", "tool" in roles)

# ---- 5) Does the FastAPI app expose any OpenAI /v1/chat/completions route? --
routes = sorted({getattr(r, "path", "?") for r in app.routes})
print("\n=== [5] Registered HTTP/WS routes (is there an OpenAI-compat chat endpoint?) ===")
print(routes)
print("any '/v1/chat/completions' route:", any("chat/completions" in p for p in routes))
