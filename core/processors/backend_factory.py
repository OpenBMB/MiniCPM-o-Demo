"""Factory for concrete backend implementations."""

from __future__ import annotations

from typing import Any, Dict


def create_backend(config: Dict[str, Any]) -> Any:
    """Create the configured backend without leaking the concrete choice into host code."""

    if config.get("backend") == "cpp":
        from core.processors.cpp_backend import CppBackend

        return CppBackend(
            gpu_id=config["gpu_id"],
            ref_audio_path=config.get("ref_audio_path"),
            duplex_pause_timeout=config.get("duplex_pause_timeout", 60.0),
            **config.get("cpp_backend", {}),
        )

    from core.processors.pytorch_backend import PyTorchBackend

    return PyTorchBackend(
        model_path=config["model_path"],
        gpu_id=config["gpu_id"],
        pt_path=config.get("pt_path"),
        ref_audio_path=config.get("ref_audio_path"),
        duplex_pause_timeout=config.get("duplex_pause_timeout", 60.0),
        compile=config.get("compile", False),
        chat_vocoder=config.get("chat_vocoder", "token2wav"),
        attn_implementation=config.get("attn_implementation", "auto"),
    )
