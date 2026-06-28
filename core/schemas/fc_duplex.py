"""FC slot duplex schemas.

This module defines API contracts for the new FC slot duplex protocol.
It intentionally contains no tokenizer, protocol state machine, or model
execution logic.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


FcNonSpokenCloseReason = Literal["eos", "no_action", "budget_reached", "hold", "abort"]


class FcDuplexConfig(BaseModel):
    """Configuration for FC slot duplex inference."""

    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    tool_format: str = Field("minicpm4_xml", description="SDK tool serialization format")
    unit_sec: float = Field(1.0, gt=0.0, description="Seconds per duplex unit")
    sample_rate: int = Field(16000, gt=0, description="Input audio sample rate")
    max_spoken_tokens: int = Field(24, ge=1, description="Max spoken tokens per unit")
    non_spoken_budget_per_unit: int = Field(12, ge=0, description="Offline non-spoken budget per unit")
    extra_response_units: int = Field(4, ge=0, description="Extra silent units after input audio")
    decode_mode: str = Field("greedy", description="Decode mode: greedy or sampling")


class FcToolResponse(BaseModel):
    """Tool response injected into an input event slot."""

    call_id: str = Field(..., description="Tool call id to which this response belongs")
    content: Any = Field(..., description="Raw tool response content")


class FcDuplexPrepareRequest(BaseModel):
    """Prepare an FC duplex session."""

    system_prompt: str = Field("", description="System prompt text")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Tool definitions")


class FcDuplexPrefillRequest(BaseModel):
    """Prefill one FC duplex unit with user input and external events."""

    audio_path: Optional[str] = Field(None, description="Audio file path")
    audio_data: Optional[str] = Field(None, description="Base64 float32 PCM audio")
    frame_list: Optional[List[Any]] = Field(None, description="Optional video/image frames")
    tool_responses: Optional[List[FcToolResponse]] = Field(None, description="Tool responses for this unit")
    sample_rate: int = Field(16000, gt=0, description="Input audio sample rate")


class FcSpokenGenerateRequest(BaseModel):
    """Generate the ai_spoken slot for the current unit."""

    max_tokens: int = Field(24, ge=1, description="Max spoken tokens for this unit")
    decode_mode: str = Field("greedy", description="Decode mode: greedy or sampling")


class FcNonSpokenGenerateRequest(BaseModel):
    """Generate or close the ai_non_spoken slot for the current unit."""

    max_tokens: int = Field(1, ge=0, description="Max non-spoken tokens to sample")
    decode_mode: str = Field("greedy", description="Decode mode: greedy or sampling")
    close_reason: Optional[FcNonSpokenCloseReason] = Field(None, description="Optional forced close reason")


class FcFinalizeUnitRequest(BaseModel):
    """Finalize the current FC duplex unit."""

    pass


class FcClosedSpan(BaseModel):
    """A think/tool_call span that closed during non-spoken generation."""

    type: Literal["think", "tool_call"] = Field(..., description="Closed span type")
    tool_call_id: Optional[str] = Field(None, description="Framework assigned tool call id")
    text: Optional[str] = Field(None, description="Decoded think text")
    wire: Optional[str] = Field(None, description="Raw tool call wire text")
    tool_call: Optional[Dict[str, Any]] = Field(None, description="Parsed tool call")
    error: Optional[str] = Field(None, description="Tool call parse or management error")


class FcDuplexStepResult(BaseModel):
    """Result of one FC duplex primitive call."""

    token_ids: List[int] = Field(default_factory=list, description="Generated or inserted token ids")
    terminated: bool = Field(False, description="Whether the current slot naturally or forcibly terminated")
    close_reason: Optional[str] = Field(None, description="Close reason if the slot was closed")
    closed_spans: List[FcClosedSpan] = Field(default_factory=list, description="Spans closed by this step")
    text: str = Field("", description="Decoded text produced by this step")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional implementation details")


class FcDuplexUnitInfo(BaseModel):
    """Summary for one FC duplex unit."""

    unit: int = Field(..., description="Unit index")
    n_audio: int = Field(0, description="Number of audio placeholder embeddings")
    has_event: bool = Field(False, description="Whether input_event_slot was present")
    is_listen: Optional[bool] = Field(None, description="Whether this unit chose listen")
    is_speaking: bool = Field(False, description="Whether this unit chose speak")
    spoken_ids: List[int] = Field(default_factory=list, description="Spoken slot token ids")
    non_spoken_ids: List[int] = Field(default_factory=list, description="Non-spoken slot token ids")
    non_spoken_terminator: Optional[str] = Field(None, description="Non-spoken close reason")
    closed_spans: List[FcClosedSpan] = Field(default_factory=list, description="Spans closed in this unit")


class FcDuplexOutput(BaseModel):
    """Structured FC duplex output."""

    success: bool = Field(..., description="Whether inference succeeded")
    error: Optional[str] = Field(None, description="Error message")
    output_ids: List[int] = Field(default_factory=list, description="Full FC output token ids")
    output_render: str = Field("", description="Human-readable token stream")
    spoken_text: str = Field("", description="Decoded spoken text")
    think_text: str = Field("", description="Decoded think text")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Decoded tool calls")
    units_info: List[FcDuplexUnitInfo] = Field(default_factory=list, description="Per-unit summaries")
    total_units: int = Field(0, description="Total units")
    total_duration_ms: float = Field(0.0, description="Wall-clock duration")


class FcDuplexOfflineInput(BaseModel):
    """Offline FC duplex input."""

    system_prompt: str = Field("", description="System prompt text")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Tool definitions")
    user_audio_path: Optional[str] = Field(None, description="Input audio file path")
    audio_data: Optional[str] = Field(None, description="Base64 float32 PCM audio")
    image_paths: Optional[List[str]] = Field(None, description="Optional image path per unit")
    tool_responses_by_unit: Dict[int, List[FcToolResponse]] = Field(
        default_factory=dict,
        description="Tool responses keyed by unit index",
    )
    tool_call_ids: Optional[List[str]] = Field(
        None,
        description="Optional deterministic tool call ids for offline consistency tests",
    )
    config: FcDuplexConfig = Field(default_factory=FcDuplexConfig, description="Offline config")


class FcDuplexOfflineOutput(FcDuplexOutput):
    """Offline FC duplex output."""

    n_audio_units: int = Field(0, description="Number of units driven by input audio")
