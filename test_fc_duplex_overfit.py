#!/usr/bin/env python3
"""Run FC slot duplex inference on the overfit dob_dev_plan_0001 sample."""

import argparse
import json
import os
from pathlib import Path

from minicpm_o5_sdk import OpenAIToolDefinition

from core.processors import UnifiedProcessor
from core.schemas.fc_duplex import FcDuplexConfig, FcDuplexOfflineInput


DEFAULT_BASE_MODEL = "/user/heweiquan/project/MiniCPM-o-4_5"
DEFAULT_PT_PATH = "/user/heweiquan/models/minicpm-o45-fc-overfit/minicpm-v_100.pt"
DEFAULT_DATA_PATH = (
    "/user/heweiquan/project/MiniCPM-o-4_5-fc_duplex_infer/"
    "data/training_data/dob_dev_plan_0001.json"
)


def load_case(data_path: str):
    with open(data_path, "r", encoding="utf-8") as f:
        structure = json.load(f)

    system_prompt = "\n".join(
        seg["text"]
        for seg in structure.get("system", {}).get("segments", [])
        if seg.get("kind") == "text"
    )
    raw_tools = structure.get("system", {}).get("tools") or []
    tools = [OpenAIToolDefinition.model_validate(t).model_dump() for t in raw_tools] or None
    tool_call_ids = []
    ai_non_spoken = ((structure.get("tracks") or {}).get("ai_non_spoken") or {}).get("segments") or []
    for segment in ai_non_spoken:
        content = segment.get("content") or {}
        if content.get("kind") == "tool_call" and content.get("tool_call_id"):
            tool_call_ids.append(content["tool_call_id"])

    sample_id = Path(data_path).stem
    media_dir = Path(data_path).parent.parent / "media" / sample_id
    audio_path = media_dir / "user_audio_0.opus"
    if not audio_path.exists():
        raise FileNotFoundError(f"missing user audio: {audio_path}")

    return system_prompt, tools, str(audio_path), tool_call_ids


def main():
    parser = argparse.ArgumentParser(description="FC duplex overfit smoke test")
    parser.add_argument("--model-path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--pt-path", default=DEFAULT_PT_PATH)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--extra-response-units", type=int, default=4)
    parser.add_argument("--decode-mode", default="greedy")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    system_prompt, tools, audio_path, tool_call_ids = load_case(args.data)

    print("[paths]")
    print(f"  base model : {args.model_path}")
    print(f"  pt path    : {args.pt_path}")
    print(f"  data       : {args.data}")
    print(f"  audio      : {audio_path}")
    print(f"  tools      : {len(tools or [])}")
    print(f"  gt call ids: {tool_call_ids}")

    processor = UnifiedProcessor(
        model_path=args.model_path,
        pt_path=args.pt_path,
        device=args.device,
        compile=False,
        attn_implementation=args.attn_implementation,
    )
    fc = processor.fc_duplex

    result = fc.offline_inference(
        FcDuplexOfflineInput(
            system_prompt=system_prompt,
            tools=tools,
            user_audio_path=audio_path,
            tool_call_ids=tool_call_ids,
            config=FcDuplexConfig(
                decode_mode=args.decode_mode,
                non_spoken_budget_per_unit=args.budget,
                extra_response_units=args.extra_response_units,
            ),
        ),
        non_spoken_budget_per_unit=args.budget,
    )

    print("\n[result]")
    print(f"  success       : {result.success}")
    print(f"  error         : {result.error}")
    print(f"  total_units   : {result.total_units}")
    print(f"  n_audio_units : {result.n_audio_units}")
    print(f"  duration_ms   : {result.total_duration_ms:.1f}")
    print(f"  spoken_text   : {result.spoken_text!r}")
    print(f"  think_text    : {result.think_text!r}")
    print(f"  tool_calls    : {result.tool_calls}")

    print("\n[units]")
    for unit in result.units_info:
        print(
            f"  unit={unit.unit:02d} "
            f"listen={unit.is_listen} speaking={unit.is_speaking} "
            f"n_audio={unit.n_audio} "
            f"spoken={len(unit.spoken_ids)} non_spoken={len(unit.non_spoken_ids)} "
            f"term={unit.non_spoken_terminator} spans={len(unit.closed_spans)}"
        )

    print("\n[render head]")
    print(result.output_render)


if __name__ == "__main__":
    main()
