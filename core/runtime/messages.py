"""Message parsing/conversion shared between the backend server and runtime.

These helpers translate frontend raw messages into schema messages and then into
the model message format. They are transport-agnostic and reused by the backend
protocol server.
"""

from __future__ import annotations

from typing import List

from core.schemas.common import (
    AudioContent,
    ContentItem,
    ImageContent,
    Message,
    Role,
    TextContent,
    VideoContent,
)


def parse_raw_messages(raw_messages: List[dict]) -> List[Message]:
    """Parse frontend raw messages into schema messages."""

    messages: List[Message] = []
    for raw_message in raw_messages:
        role = Role(raw_message["role"])
        content = raw_message["content"]
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
                messages.append(Message(role=role, content=content_items))
        else:
            messages.append(Message(role=role, content=content))
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
    return model_msgs
