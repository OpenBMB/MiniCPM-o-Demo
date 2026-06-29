"""MiniCPMO45 处理器模块

本模块提供统一处理器，支持 Chat/Streaming/Duplex 三种模式热切换。

使用方式：
=========

```python
from core.processors import UnifiedProcessor

processor = UnifiedProcessor(model_path=..., pt_path=...)

# Chat 模式
chat = processor.set_chat_mode()
response = chat.chat(request)

# Streaming 模式（毫秒级切换）
half_duplex = processor.set_half_duplex_mode()
for chunk in streaming.generate(session_id):
    print(chunk.text_delta, end="")

# Duplex 模式（毫秒级切换）
duplex = processor.set_duplex_mode()
duplex.prepare(...)
result = duplex.generate()
```

核心优势：
=========

- 模型只加载一次，显存高效
- Chat/Streaming/Duplex 毫秒级切换（< 1ms）
- 一个 Worker 支持所有模式
- 类型安全的 View API
"""

from core.processors.base import BaseProcessor, MiniCPMOProcessorMixin
from core.processors.unified import UnifiedProcessor, ChatView, HalfDuplexView, DuplexView

__all__ = [
    # 基类
    "BaseProcessor",
    "MiniCPMOProcessorMixin",
    # 统一处理器
    "UnifiedProcessor",
    "ChatView",
    "HalfDuplexView",
    "DuplexView",
]
