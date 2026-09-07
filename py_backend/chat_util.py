"""Chat request/message parsing utilities for the backend protocol server.

These helpers parse a `chat.request` protocol packet into a structured request,
and translate frontend raw messages into the model message format. They are
transport-agnostic and reused by the backend protocol server before inference.
"""

from __future__ import annotations

import base64
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.schemas.common import (
    AudioContent,
    ContentItem,
    ImageContent,
    Message,
    Role,
    TextContent,
    VideoContent,
)


class ChatRequestError(ValueError):
    pass


# --- Function calling (issue #21) ------------------------------------------
# The MiniCPM-o 4.5 chat template tells the model to return, inside a fenced
# tool_call block, a JSON object {"name": <fn>, "arguments": {...}}. The tag
# strings are built with chr(96) so this source file never contains raw
# triple-backtick sequences (which some editors/linters mis-render as markdown).
_BT = chr(96) * 3
# The chat template opens AND closes the block the same way: a newline, then
# ```tool_call. So both delimiters are "\n```tool_call".
_TC_OPEN = "\n" + _BT + "tool_call"
_TC_CLOSE = "\n" + _BT + "tool_call"
_TC_RE = re.compile(re.escape(_TC_OPEN) + r"(.*?)" + re.escape(_TC_CLOSE), re.DOTALL)
# Defensive: some models emit the XML-style tool_call tags referenced in
# the template's prose instead of the fenced form. Match those too.
_BT2 = chr(60) + "tool_call" + chr(62)
_ET2 = chr(60) + "/tool_call" + chr(62)
_TC_XML_RE = re.compile(re.escape(_BT2) + r"(.*?)" + re.escape(_ET2), re.DOTALL)


def _extract_tool_calls(text):
    """Parse model output into (clean_text, tool_calls).

    Returns a tuple:
      clean_text  - visible text with any tool_call block removed.
      tool_calls  - OpenAI-style list the client can execute:
                    [{"id": "call_<uuid>", "type": "function",
                      "function": {"name": <fn>, "arguments": <object>}}]

    When no tool_call block is present, returns (text, []). A malformed block is
    skipped (dropped from tool_calls, still stripped from clean_text) rather than
    raising, so a single bad block cannot crash the turn.
    """
    if not text:
        return text or "", []

    # Prefer the fenced form (what the template's example output uses); fall
    # back to the XML-tag form referenced in the template's prose.
    pat = _TC_RE if _TC_OPEN in text else _TC_XML_RE
    if _TC_OPEN not in text and _BT2 not in text:
        return text or "", []

    tool_calls = []
    last_end = 0
    clean_parts = []
    for m in pat.finditer(text):
        clean_parts.append(text[last_end:m.start()])
        block = m.group(1).strip()
        payload = _first_json_object(block)
        if payload is not None:
            name = payload.get("name")
            arguments = payload.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (ValueError, TypeError):
                    pass
            tool_calls.append({
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {
                    "name": name if name is not None else "unknown",
                    "arguments": arguments if arguments is not None else {},
                },
            })
        last_end = m.end()
    clean_parts.append(text[last_end:])

    clean_text = "".join(clean_parts).strip()
    clean_text = re.sub(re.escape(_TC_OPEN), "", clean_text).strip()
    clean_text = re.sub(re.escape(_BT2), "", clean_text).strip()
    return clean_text, tool_calls


def _first_json_object(block):
    """Return the first balanced {...} object in block as a dict, else None.

    Dependency-free scanner: finds the first "{" that closes at a matching "}"
    and json.loads that slice. Tracks strings + escapes so braces inside string
    values do not confuse the depth count.
    """
    start = block.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        end = -1
        for i in range(start, len(block)):
            ch = block[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
        if end != -1:
            try:
                return json.loads(block[start:end + 1])
            except (ValueError, TypeError):
                pass
        start = block.find("{", start + 1)
    return None


@dataclass
class WorkerChatRequest:
    messages: list
    streaming: bool
    max_new_tokens: int
    length_penalty: float
    max_slice_nums: Optional[int]
    generate_audio: bool
    tts_ref_audio: Optional[np.ndarray]
    use_tts_template: bool
    omni_mode: bool
    enable_thinking: bool
    # --- OpenAI-compatible function calling (issue #21 fix) -----------------
    # Raw `tools` (list of function schemas) and `tool_choice` as sent by the
    # client. These were previously silently dropped by the protocol, so the
    # model never saw the tools and could not call them.
    tools: Optional[list] = None
    tool_choice: Any = None

    @property
    def effective_tools(self) -> Optional[list]:
        """Tools to pass to the chat template.

        Returns `None` (no tool block rendered) when the client explicitly
        asked for `tool_choice="none"`, or when no tools were provided.
        `tool_choice="auto"` / a specific function name / omitted -> tools
        pass through and the model decides per the template's instruction.
        Note: the MiniCPM-o 4.5 chat template does not itself branch on
        `tool_choice` (it always lets the model choose), so "auto" and
        "specific" are honoured the same way: tools are exposed and the model
        selects. "none" is the only one we enforce by withholding tools.
        """
        if not self.tools:
            return None
        if isinstance(self.tool_choice, str) and self.tool_choice.lower() == "none":
            return None
        return self.tools


def parse_worker_chat_request_message(msg: Dict[str, Any]) -> WorkerChatRequest:
    """Parse a `chat.request` protocol message into a structured request."""

    if msg.get("type") != "chat.request":
        raise ChatRequestError("expected chat.request message")

    payload = msg.get("payload") or {}
    if not isinstance(payload, dict):
        raise ChatRequestError("chat.request payload must be an object")

    generation = payload.get("generation") or {}
    if not isinstance(generation, dict):
        raise ChatRequestError("chat.request generation must be an object")

    image = payload.get("image") or {}
    if not isinstance(image, dict):
        raise ChatRequestError("chat.request image must be an object")

    tts = payload.get("tts") or {}
    if not isinstance(tts, dict):
        raise ChatRequestError("chat.request tts must be an object")

    max_slice_nums = None
    if image.get("max_slice_nums") is not None:
        max_slice_nums = int(image["max_slice_nums"])

    generate_audio = bool(tts.get("enabled", False))
    tts_ref_audio = None
    ref_b64 = tts.get("ref_audio_data")
    if generate_audio and ref_b64:
        tts_ref_audio = np.frombuffer(base64.b64decode(ref_b64), dtype=np.float32)

    # --- Function calling: thread tools / tool_choice through (issue #21) ---
    # Previously these keys were never read, so OpenAI-style clients sending
    # `tools` + `tool_choice` had them silently dropped before reaching the model.
    tools_raw = payload.get("tools")
    tools: Optional[list] = None
    if tools_raw is not None:
        if not isinstance(tools_raw, list):
            raise ChatRequestError("chat.request tools must be a list")
        for t in tools_raw:
            if not isinstance(t, dict):
                raise ChatRequestError("each tool must be an object")
        tools = tools_raw
    tool_choice = payload.get("tool_choice")

    return WorkerChatRequest(
        messages=payload.get("messages", []),
        streaming=bool(payload.get("streaming", True)),
        max_new_tokens=int(generation.get("max_new_tokens", 256)),
        length_penalty=float(generation.get("length_penalty", 1.1)),
        max_slice_nums=max_slice_nums,
        generate_audio=generate_audio,
        tts_ref_audio=tts_ref_audio,
        use_tts_template=bool(payload.get("use_tts_template", False) or generate_audio),
        omni_mode=bool(payload.get("omni_mode", False)),
        enable_thinking=bool(payload.get("enable_thinking", False)),
        tools=tools,
        tool_choice=tool_choice,
    )


def parse_raw_messages(raw_messages: List[dict]) -> List[Message]:
    """Parse frontend raw messages into schema messages."""

    messages: List[Message] = []
    for raw_message in raw_messages:
        role = Role(raw_message["role"])
        content = raw_message.get("content", "")
        tool_calls = raw_message.get("tool_calls") or None
        # tool_calls only make sense on assistant turns; keep them only there
        # so the chat template renders them in the right place.
        if not isinstance(tool_calls, list):
            tool_calls = None
        if isinstance(content, list):
            content_items: List[ContentItem] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and item.get("text"):
                    content_items.append(TextContent(text=item["text"]))
                elif item.get("type") == "audio" and item.get("data"):
                    content_items.append(AudioContent(data=item["data"]))
                elif item.get("type") == "image" and item.get("data"):
                    content_items.append(ImageContent(data=item["data"]))
                elif item.get("type") == "video" and item.get("data"):
                    content_items.append(VideoContent(
                        data=item["data"],
                        stack_frames=item.get("stack_frames", 1),
                    ))
            if content_items:
                messages.append(Message(role=role, content=content_items, tool_calls=tool_calls))
            else:
                # A tool-call assistant turn may carry no text content — give it
                # an empty string so the template's `if content` guard is clean.
                content_val = "" if tool_calls else content
                messages.append(Message(role=role, content=content_val, tool_calls=tool_calls))
        else:
            messages.append(Message(role=role, content=content, tool_calls=tool_calls))
    return messages


def convert_to_model_msgs(schema_messages: List[Message]) -> list:
    """Convert schema messages into the current model message format."""

    from core.processors.base import MiniCPMOProcessorMixin

    mixin = MiniCPMOProcessorMixin()
    model_msgs = []
    for message in schema_messages:
        content = mixin._convert_content_to_model_format(message.content)
        if len(content) == 1 and isinstance(content[0], str):
            content = content[0]
        model_msgs.append({"role": message.role.value, "content": content})
        # Function calling: forward assistant tool_calls so the chat template's
        # <tools> block can render the <tool_call> turn. Absent for normal turns.
        if message.tool_calls:
            model_msgs[-1]["tool_calls"] = message.tool_calls
    return model_msgs
