"""FC slot duplex schemas.

This module defines API contracts for the new FC slot duplex protocol.
It intentionally contains no tokenizer, protocol state machine, or model
execution logic.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


FcNonSpokenCloseReason = Literal["eos", "no_action", "budget_reached", "hold", "abort"]


class NonSpokenStepGenerationFlag(str, Enum):
    """Loop-control signal for one non-spoken generation step."""

    no_action = "no_action"
    non_spoken_slot_eos = "non_spoken_slot_eos"
    continue_non_spoken_generation = "continue_non_spoken_generation"


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
    ref_audio_path: Optional[str] = Field(None, description="Reference audio path for FC TTS conditioning")
    prompt_wav_path: Optional[str] = Field(None, description="Prompt wav path for Token2Wav streaming cache")
    generate_audio: bool = Field(False, description="Whether to generate audio waveform from spoken text tokens")


class FcDuplexPrepareResult(BaseModel):
    """Result of preparing an FC duplex session."""

    prefill_ids: List[int] = Field(default_factory=list, description="System/tool prefill token ids")
    output_render: str = Field("", description="Rendered prefill token stream")
    resized: bool = Field(False, description="Whether the LLM embedding table was resized")
    old_vocab_size: Optional[int] = Field(None, description="Vocabulary size before resize")
    new_vocab_size: Optional[int] = Field(None, description="Vocabulary size after resize")
    required_vocab_size: Optional[int] = Field(None, description="Minimum vocabulary size required by FC special tokens")
    generate_audio: bool = Field(False, description="Whether TTS audio generation is enabled")
    has_ref_audio: bool = Field(False, description="Whether reference audio was loaded and fed")
    prompt_wav_path: Optional[str] = Field(None, description="Prompt wav path used by Token2Wav")


class FcDuplexPrefillRequest(BaseModel):
    """Prefill one FC duplex unit with user input and external events."""

    audio_path: Optional[str] = Field(None, description="Audio file path")
    audio_data: Optional[str] = Field(None, description="Base64 float32 PCM audio")
    frame_list: Optional[List[Any]] = Field(None, description="Optional video/image frames")
    tool_responses: Optional[List[FcToolResponse]] = Field(None, description="Tool responses for this unit")
    sample_rate: int = Field(16000, gt=0, description="Input audio sample rate")


class FcDuplexPrefillResult(BaseModel):
    """Result of pre-filling one FC duplex unit."""

    unit_index: int = Field(..., description="Current unit index")
    n_audio_placeholders: int = Field(0, description="Number of user audio placeholder embeddings")
    has_input_event: bool = Field(False, description="Whether this prefill inserted input events")
    is_listen: Optional[bool] = Field(None, description="Current unit listen state if already known")
    is_speaking: bool = Field(False, description="Current unit speaking state if already known")
    inserted_token_ids: List[int] = Field(default_factory=list, description="Token ids inserted by this prefill, if tracked")


class FcSpokenGenerateRequest(BaseModel):
    """Generate the ai_spoken slot for the current unit."""

    max_tokens: int = Field(24, ge=1, description="Max spoken tokens for this unit")
    decode_mode: str = Field("greedy", description="Decode mode: greedy or sampling")


class FcNonSpokenGenerateRequest(BaseModel):
    """Generate or close the ai_non_spoken slot for the current unit."""

    max_tokens: int = Field(1, ge=0, description="Max non-spoken tokens to sample")
    decode_mode: str = Field("greedy", description="Decode mode: greedy or sampling")
    close_reason: Optional[FcNonSpokenCloseReason] = Field(None, description="Optional forced close reason")


class FcDecodeOutputRequest(BaseModel):
    """Decode a token stream into structured FC duplex output."""

    output_ids: Optional[List[int]] = Field(None, description="Token ids to decode; defaults to current session output")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Tool definitions used to deserialize tool calls")


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
    audio_waveform: Optional[Any] = Field(None, description="Generated 24kHz audio waveform, if requested")
    audio_sample_rate: Optional[int] = Field(None, description="Sample rate of audio_waveform")
    n_tts_tokens: int = Field(0, description="Number of generated TTS audio tokens")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional implementation details")


class FcSpokenGenerateResult(BaseModel):
    """Typed result of generating the ai_spoken slot."""

    is_listen: bool = Field(False, description="Whether the model chose listen")
    is_speaking: bool = Field(False, description="Whether the model chose speak")
    spoken_token_ids: List[int] = Field(default_factory=list, description="Spoken slot token ids")
    spoken_text: str = Field("", description="Decoded spoken text")
    spoken_turn_eos: bool = Field(False, description="Whether this unit ended the spoken turn")
    audio_waveform: Optional[Any] = Field(None, description="Generated 24kHz waveform, if requested")
    audio_sample_rate: Optional[int] = Field(None, description="Sample rate of audio_waveform")
    n_audio_samples: int = Field(0, description="Number of samples in audio_waveform")
    n_tts_tokens: int = Field(0, description="Number of generated TTS audio tokens")
    cost_llm: float = Field(0.0, description="LLM generation cost in seconds")
    cost_tts_prep: float = Field(0.0, description="TTS condition preparation cost in seconds")
    cost_tts: float = Field(0.0, description="TTS token generation cost in seconds")
    cost_token2wav: float = Field(0.0, description="Token2Wav cost in seconds")


class FcNonSpokenGenerateResult(FcDuplexStepResult):
    """Typed result of generating the ai_non_spoken slot."""

    generation_flag: NonSpokenStepGenerationFlag = Field(
        NonSpokenStepGenerationFlag.continue_non_spoken_generation,
        description="Explicit caller loop-control signal for the non-spoken slot",
    )


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
    audio_sample_rate: Optional[int] = Field(None, description="Sample rate of generated spoken audio")
    n_audio_samples: int = Field(0, description="Number of generated spoken audio samples")


class FcDecodedUnit(BaseModel):
    """Decoded view of one FC duplex unit."""

    unit_index: int = Field(..., description="Decoded unit index")
    is_listen: Optional[bool] = Field(None, description="Whether this decoded unit listened")
    spoken_text: str = Field("", description="Decoded spoken text in this unit")
    non_spoken_terminator: Optional[str] = Field(None, description="Decoded non-spoken terminator")
    raw_non_spoken: str = Field("", description="Rendered raw non-spoken content for this unit")


class FcDecodedToolCall(BaseModel):
    """Decoded tool call from the generated token stream."""

    tool_call_id: Optional[str] = Field(None, description="Framework assigned tool call id, if available")
    name: Optional[str] = Field(None, description="Tool/function name")
    arguments: Any = Field(None, description="Tool/function arguments")
    error: Optional[str] = Field(None, description="Tool call parse error")
    wire: Optional[str] = Field(None, description="Raw tool call wire text")


class FcDecodeOutputResult(BaseModel):
    """Structured decoded token-stream output."""

    units: List[FcDecodedUnit] = Field(default_factory=list, description="Decoded per-unit summaries")
    spoken_text: str = Field("", description="All decoded spoken text")
    think_text: str = Field("", description="All decoded think text")
    tool_calls: List[FcDecodedToolCall] = Field(default_factory=list, description="Decoded tool calls")
    output_ids: List[int] = Field(default_factory=list, description="Decoded token ids")
    output_render: str = Field("", description="Rendered token stream")


class FcDuplexOutput(BaseModel):
    """Structured FC duplex output."""

    success: bool = Field(..., description="Whether inference succeeded")
    error: Optional[str] = Field(None, description="Error message")
    output_ids: List[int] = Field(default_factory=list, description="Full FC output token ids")
    output_render: str = Field("", description="Human-readable token stream")
    spoken_text: str = Field("", description="Decoded spoken text")
    think_text: str = Field("", description="Decoded think text")
    tool_calls: List[FcDecodedToolCall] = Field(default_factory=list, description="Decoded tool calls")
    units_info: List[FcDuplexUnitInfo] = Field(default_factory=list, description="Per-unit summaries")
    audio_waveforms: List[Any] = Field(default_factory=list, description="Generated 24kHz audio waveforms per spoken unit")
    total_units: int = Field(0, description="Total units")
    total_duration_ms: float = Field(0.0, description="Wall-clock duration")


class FcDuplexOfflineInput(BaseModel):
    """Offline FC duplex input."""

    system_prompt: str = Field("", description="System prompt text")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Tool definitions")
    user_audio_path: Optional[str] = Field(None, description="Input audio file path")
    audio_data: Optional[str] = Field(None, description="Base64 float32 PCM audio")
    unit_audio_chunks: Optional[List[Any]] = Field(
        None,
        description="Optional pre-scheduled per-unit float32 audio chunks; used by train-data arrangement scheduling",
    )
    ref_audio_path: Optional[str] = Field(None, description="Reference audio path for FC TTS conditioning")
    prompt_wav_path: Optional[str] = Field(None, description="Prompt wav path for Token2Wav streaming cache")
    generate_audio: bool = Field(False, description="Whether to generate audio waveform from spoken text tokens")
    image_paths: Optional[List[str]] = Field(None, description="Optional image path per unit")
    tool_responses_by_unit: Dict[int, List[FcToolResponse]] = Field(
        default_factory=dict,
        description="Tool responses keyed by unit index",
    )
    tool_responses_by_call_id: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool responses keyed by tool call id; offline inference sends them in the next unit after a call closes",
    )
    non_spoken_budgets_while_listening: Optional[List[Optional[int]]] = Field(
        None,
        description="Optional per-unit non-spoken budgets for listening units; None entries mean no sample-level limit",
    )
    non_spoken_budgets_while_speaking: Optional[List[Optional[int]]] = Field(
        None,
        description="Optional per-unit non-spoken budgets for speaking units; None entries mean no sample-level limit",
    )
    tool_call_ids: Optional[List[str]] = Field(
        None,
        description="Optional deterministic tool call ids for offline consistency tests",
    )
    config: FcDuplexConfig = Field(default_factory=FcDuplexConfig, description="Offline config")


class FcDuplexOfflineOutput(FcDuplexOutput):
    """Offline FC duplex output."""

    n_audio_units: int = Field(0, description="Number of units driven by input audio")


class FcDuplexAudioArtifact(BaseModel):
    """Audio files written during FC duplex evaluation."""

    sample_rate: int = Field(24000, description="Audio artifact sample rate")
    unit_audio_paths: List[str] = Field(default_factory=list, description="Per-unit generated audio wav paths")
    full_audio_path: Optional[str] = Field(None, description="Concatenated generated audio wav path")
    n_audio_units: int = Field(0, description="Number of generated audio units written to disk")


class FcTokenStreamDiff(BaseModel):
    """First differing region between two rendered token streams."""

    index: int = Field(..., description="First differing character index")
    gt_context: str = Field("", description="Ground-truth context around the first diff")
    pred_context: str = Field("", description="Prediction context around the first diff")


class FcDuplexComparisonResult(BaseModel):
    """GT vs prediction comparison for FC duplex train-data evaluation."""

    token_ids_exact: bool = Field(False, description="Whether GT and prediction token ids are exactly equal")
    rendered_token_stream_exact: bool = Field(False, description="Whether rendered token streams are exactly equal")
    spoken_text_exact: bool = Field(False, description="Whether decoded spoken text is exactly equal")
    think_text_exact: bool = Field(False, description="Whether decoded think text is exactly equal")
    tool_calls_semantic_exact: bool = Field(False, description="Whether decoded tool calls match ignoring ids")
    tool_call_ids_exact: bool = Field(False, description="Whether generated tool call ids match train data")
    first_rendered_token_stream_diff: Optional[FcTokenStreamDiff] = Field(None, description="First rendered token stream diff")


class FcDuplexTrainDataRequest(BaseModel):
    """Run offline inference and comparison from an FC duplex training sample."""

    train_data_path: Optional[str] = Field(None, description="Path to a training data JSON file")
    train_data: Optional[Any] = Field(None, description="Already loaded training data structure or SDK object")
    data_root: Optional[str] = Field(None, description="Directory that contains media files for this training sample")
    config: FcDuplexConfig = Field(default_factory=FcDuplexConfig, description="Offline inference config")
    non_spoken_budget_per_unit: Optional[int] = Field(None, description="Override non-spoken budget per unit")
    generate_audio: bool = Field(False, description="Whether to generate and optionally save TTS audio")
    ref_audio_path: Optional[str] = Field(None, description="Reference audio path; defaults to sample user audio")
    prompt_wav_path: Optional[str] = Field(None, description="Token2Wav prompt path; defaults to ref_audio_path")
    output_artifact_dir: Optional[str] = Field(None, description="Directory to write source, streams, and audio artifacts")
    use_train_tool_call_ids: bool = Field(True, description="Use GT tool call ids for deterministic evaluation")
    inject_train_tool_responses: bool = Field(True, description="Inject GT tool responses after matching tool calls close")


class FcDuplexTrainDataResult(BaseModel):
    """Typed output of train-data offline inference and comparison."""

    sample_id: str = Field("", description="Training sample id")
    success: bool = Field(..., description="Whether inference and comparison completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    source_path: Optional[str] = Field(None, description="Source training JSON path")
    user_audio_path: Optional[str] = Field(None, description="User audio path used for inference")
    gt_output_ids: List[int] = Field(default_factory=list, description="Ground-truth token ids")
    pred_output_ids: List[int] = Field(default_factory=list, description="Predicted token ids")
    gt_output_render: str = Field("", description="Rendered ground-truth token stream")
    pred_output_render: str = Field("", description="Rendered predicted token stream")
    gt_spoken_text: str = Field("", description="Ground-truth spoken text")
    pred_spoken_text: str = Field("", description="Predicted spoken text")
    gt_think_text: str = Field("", description="Ground-truth think text")
    pred_think_text: str = Field("", description="Predicted think text")
    gt_tool_calls: List[FcDecodedToolCall] = Field(default_factory=list, description="Ground-truth decoded tool calls")
    pred_tool_calls: List[FcDecodedToolCall] = Field(default_factory=list, description="Predicted decoded tool calls")
    tool_call_ids: List[str] = Field(default_factory=list, description="GT tool call ids used by the fixed generator")
    tool_responses_by_call_id: Dict[str, Any] = Field(default_factory=dict, description="GT tool responses keyed by call id")
    units_info: List[FcDuplexUnitInfo] = Field(default_factory=list, description="Predicted per-unit info")
    comparison: Optional[FcDuplexComparisonResult] = Field(None, description="GT vs prediction comparison")
    audio_artifact: Optional[FcDuplexAudioArtifact] = Field(None, description="Generated TTS audio artifact")
    total_duration_ms: float = Field(0.0, description="Wall-clock duration in milliseconds")
