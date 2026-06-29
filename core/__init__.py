"""MiniCPMO45 核心模块

本模块提供 MiniCPMO45 模型的完整封装，使调用者无需阅读模型代码即可使用所有功能。

模块组织：
=========

```
core/
├── schemas/           # 数据类型定义
│   ├── common.py      # 通用类型（Message, Role, TTSConfig 等）
│   ├── chat.py        # 单工对话（ChatRequest/Response）
│   ├── streaming.py   # 流式对话（StreamingRequest/Chunk）
│   └── duplex.py      # 双工对话（DuplexConfig/Result）
│
├── processors/        # 处理器实现
│   ├── base.py        # 基类
│   └── unified.py     # UnifiedProcessor（统一处理器）
│
├── capabilities.py    # 能力声明
└── factory.py         # 工厂模式
```

快速入门：
=========

**使用 UnifiedProcessor（推荐）**

```python
from core.processors import UnifiedProcessor
from core.schemas import ChatRequest, Message, Role

# 创建统一处理器（一次加载，支持所有模式）
processor = UnifiedProcessor(
    model_path="/path/to/base_model",  # HuggingFace 格式基础模型
    pt_path="/path/to/custom.pt",      # 可选：覆盖权重
)

# Chat 模式
chat = processor.set_chat_mode()
response = chat.chat(ChatRequest(
    messages=[Message(role=Role.USER, content="你好")]
))
print(response.text)

# Half-Duplex 模式（毫秒级切换）
half_duplex = processor.set_half_duplex_mode()
half_duplex.prefill(StreamingRequest(...))
for chunk in half_duplex.generate(session_id="user_001"):
    print(chunk.text_delta, end="")

# Duplex 模式（毫秒级切换）
duplex = processor.set_duplex_mode()
duplex.prepare(system_prompt_text="你是助手")
for audio_chunk in audio_stream:
    duplex.prefill(audio_waveform=audio_chunk)
    result = duplex.generate()
    if not result.is_listen:
        print(result.text)
```

三种模式对比：
=============

| 特性 | Chat | Streaming | Duplex |
|------|------|-----------|--------|
| 返回方式 | 完整 | 流式 | 实时 |
| 打断支持 | ❌ | ❌ | ✅ |
| 回溯支持 | ✅ | ✅ | ❌ |
| 切换延迟 | < 1ms | < 1ms | < 1ms |

关键发现（来自开发实践）：
========================

1. **TTS 模式**：mode="default" 会忽略 ref_audio！必须用 AUDIO_ASSISTANT
2. **双工 System Prompt**：必须使用特殊 token 格式，否则输出乱码
3. **force_listen_count**：双工模式的启动保护期，固定值（如 3）
4. **音频格式**：输入必须 16kHz 单声道，输出是 24kHz
"""

# 能力声明
from core.capabilities import (
    ProcessorMode,
    ProcessorCapabilities,
    CAPABILITIES,
    CHAT_CAPABILITIES,
    HALF_DUPLEX_CAPABILITIES,
    DUPLEX_CAPABILITIES,
    get_capabilities,
    supports_feature,
)

# 处理器
from core.processors import (
    BaseProcessor,
    UnifiedProcessor,
    ChatView,
    HalfDuplexView,
    DuplexView,
)

# 工厂
from core.factory import (
    ProcessorFactory,
    create_processor,
)

# Schema - 通用类型
from core.schemas import (
    # 枚举
    Role,
    TTSMode,
    ContentType,
    
    # 内容类型
    TextContent,
    ImageContent,
    AudioContent,
    VideoContent,
    ContentItem,
    
    # 消息
    Message,
    
    # 配置
    TTSSamplingParams,
    TTSConfig,
    ImageConfig,
    GenerationConfig,
    
    # 单工对话
    ChatRequest,
    ChatResponse,
    
    # 流式对话
    StreamingConfig,
    StreamingRequest,
    StreamingChunk,
    StreamingResponse,
    RollbackResult,
    
    # 双工对话
    DuplexConfig,
    DuplexPrepareRequest,
    DuplexPrefillRequest,
    DuplexGenerateResult,
    DuplexOfflineInput,
    DuplexChunkResult,
    DuplexOfflineOutput,
)

__all__ = [
    # 能力声明
    "ProcessorMode",
    "ProcessorCapabilities",
    "CAPABILITIES",
    "CHAT_CAPABILITIES",
    "HALF_DUPLEX_CAPABILITIES",
    "DUPLEX_CAPABILITIES",
    "get_capabilities",
    "supports_feature",
    
    # 处理器
    "BaseProcessor",
    "UnifiedProcessor",
    "ChatView",
    "HalfDuplexView",
    "DuplexView",
    
    # 工厂
    "ProcessorFactory",
    "create_processor",
    
    # Schema - 枚举
    "Role",
    "TTSMode",
    "ContentType",
    
    # Schema - 内容类型
    "TextContent",
    "ImageContent",
    "AudioContent",
    "VideoContent",
    "ContentItem",
    
    # Schema - 消息
    "Message",
    
    # Schema - 配置
    "TTSSamplingParams",
    "TTSConfig",
    "ImageConfig",
    "GenerationConfig",
    
    # Schema - 单工对话
    "ChatRequest",
    "ChatResponse",
    
    # Schema - 流式对话
    "StreamingConfig",
    "StreamingRequest",
    "StreamingChunk",
    "StreamingResponse",
    "RollbackResult",
    
    # Schema - 双工对话
    "DuplexConfig",
    "DuplexPrepareRequest",
    "DuplexPrefillRequest",
    "DuplexGenerateResult",
    "DuplexOfflineInput",
    "DuplexChunkResult",
    "DuplexOfflineOutput",
]
