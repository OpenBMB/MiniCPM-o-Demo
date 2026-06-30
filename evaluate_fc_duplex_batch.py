#!/usr/bin/env python3
"""Batch evaluate FC duplex inference against training-data token streams."""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


from core.processors import UnifiedProcessor
from core.schemas.fc_duplex import FcDuplexConfig, FcDuplexTrainDataRequest


DEFAULT_BASE_MODEL = "/user/heweiquan/project/MiniCPM-o-4_5"
DEFAULT_PT_PATH = "/user/heweiquan/models/minicpm-o45-fc-overfit/20260630/v4/minicpm-v_400.pt"
DEFAULT_DATA_DIR = "/user/heweiquan/dataset/DuplexFcTest/delivery_train_data"
DEFAULT_OUTPUT_DIR = "/user/heweiquan/dataset/DuplexFcTest/test_res/20260630/v4/fc_duplex_test_results"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_one(
    fc,
    data_path: Path,
    output_dir: Path,
    budget: Optional[int],
    extra_response_units: int,
    decode_mode: str,
    generate_audio: bool,
    ref_audio_path: Optional[str],
    prompt_wav_path: Optional[str],
) -> Dict[str, Any]:
    sample_id = data_path.stem
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    config_kwargs = {
        "decode_mode": decode_mode,
        "extra_response_units": extra_response_units,
    }
    if budget is not None:
        config_kwargs["non_spoken_budget_per_unit"] = budget

    result = fc.offline_inference_from_train_data(
        FcDuplexTrainDataRequest(
            train_data_path=str(data_path),
            config=FcDuplexConfig(**config_kwargs),
            non_spoken_budget_per_unit=budget,
            generate_audio=generate_audio,
            ref_audio_path=ref_audio_path,
            prompt_wav_path=prompt_wav_path,
            output_artifact_dir=str(sample_dir),
        )
    )
    dumped = result.model_dump()
    if not dumped.get("success") and dumped.get("pred_output_render"):
        (sample_dir / "pred_prefix_token_stream.txt").write_text(dumped["pred_output_render"], encoding="utf-8")
        write_json(sample_dir / "pred_prefix_token_ids.json", dumped.get("pred_output_ids") or [])
    write_json(sample_dir / "comparison.json", dumped)
    return dumped


def make_mutated_inputs(data_paths: List[Path], output_dir: Path) -> List[Path]:
    mutated_dir = output_dir / "modified_inputs"
    mutated_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for idx, src in enumerate(data_paths[:2], start=1):
        structure = read_json(src)
        structure["data_id"] = f"{structure.get('data_id', src.stem)}:modified_{idx}"
        for segment in structure.get("system", {}).get("segments", []):
            if segment.get("kind") == "text":
                segment["text"] = (
                    segment.get("text", "")
                    + f"\n\n轻微改动测试 {idx}：保持原规则不变，但表达更偏保守，只有非常明确的可见物体才展示。"
                )
                break
        if structure.get("system", {}).get("tools"):
            tool = structure["system"]["tools"][0].get("function", {})
            tool["description"] = (tool.get("description") or "") + f"（轻微改动测试 {idx}）"
        dst = mutated_dir / f"{src.stem}_modified_{idx}.json"
        write_json(dst, structure)

        media_src = src.parent.parent / "media" / src.stem
        media_dst = dst.parent.parent / "media" / dst.stem
        if media_dst.exists():
            shutil.rmtree(media_dst)
        shutil.copytree(media_src, media_dst)
        out.append(dst)
    return out


def summarize(comparisons: List[Dict[str, Any]], output_dir: Path, group: str) -> Dict[str, Any]:
    def matched(item: Dict[str, Any], key: str) -> bool:
        return bool((item.get("comparison") or {}).get(key))

    summary = {
        "group": group,
        "total": len(comparisons),
        "success": sum(1 for c in comparisons if c.get("success")),
        "token_ids_exact": sum(1 for c in comparisons if matched(c, "token_ids_exact")),
        "rendered_token_stream_exact": sum(1 for c in comparisons if matched(c, "rendered_token_stream_exact")),
        "spoken_text_exact": sum(1 for c in comparisons if matched(c, "spoken_text_exact")),
        "think_text_exact": sum(1 for c in comparisons if matched(c, "think_text_exact")),
        "tool_calls_semantic_exact": sum(1 for c in comparisons if matched(c, "tool_calls_semantic_exact")),
        "tool_call_ids_exact": sum(1 for c in comparisons if matched(c, "tool_call_ids_exact")),
        "failed_samples": [c["sample_id"] for c in comparisons if not c.get("success")],
        "token_diff_samples": [
            c["sample_id"]
            for c in comparisons
            if not (matched(c, "token_ids_exact") and matched(c, "rendered_token_stream_exact"))
        ],
        "semantic_diff_samples": [
            c["sample_id"]
            for c in comparisons
            if not (
                matched(c, "spoken_text_exact")
                and matched(c, "think_text_exact")
                and matched(c, "tool_calls_semantic_exact")
                and matched(c, "tool_call_ids_exact")
            )
        ],
    }
    write_json(output_dir / f"{group}_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Batch FC duplex train/infer consistency evaluation")
    parser.add_argument("--model-path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--pt-path", default=DEFAULT_PT_PATH)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Debug override for non-spoken budget. Omit to use SDK train-data per-unit budgets.",
    )
    parser.add_argument("--extra-response-units", type=int, default=0)
    parser.add_argument("--decode-mode", default="greedy")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-mutated", action="store_true")
    parser.add_argument(
        "--ref-audio-path",
        default=None,
        help="Reference audio path for FC TTS. Required together with --tts-prompt-path to enable audio generation.",
    )
    parser.add_argument(
        "--tts-prompt-path",
        default=None,
        help="Short Token2Wav prompt audio path. Required together with --ref-audio-path to enable audio generation.",
    )
    args = parser.parse_args()

    if bool(args.ref_audio_path) != bool(args.tts_prompt_path):
        parser.error("--ref-audio-path and --tts-prompt-path must be provided together to enable TTS audio generation")
    generate_audio = bool(args.ref_audio_path and args.tts_prompt_path)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_paths = sorted(data_dir.glob("*.json"))
    if args.limit is not None:
        data_paths = data_paths[:args.limit]

    print(f"[load] model={args.model_path}")
    print(f"[load] pt={args.pt_path}")
    processor = UnifiedProcessor(
        model_path=args.model_path,
        pt_path=args.pt_path,
        device=args.device,
        compile=False,
        attn_implementation=args.attn_implementation,
    )
    fc = processor.fc_duplex

    original_results = []
    original_dir = output_dir / "original"
    for index, path in enumerate(data_paths, start=1):
        print(f"[original {index:03d}/{len(data_paths):03d}] {path.name}", flush=True)
        case_start = time.perf_counter()
        try:
            comparison = run_one(
                fc,
                path,
                original_dir,
                args.budget,
                args.extra_response_units,
                args.decode_mode,
                generate_audio,
                args.ref_audio_path,
                args.tts_prompt_path,
            )
        except Exception as exc:
            comparison = {"sample_id": path.stem, "data_path": str(path), "success": False, "error": repr(exc), "matches": {}}
            sample_dir = original_dir / path.stem
            write_json(sample_dir / "comparison.json", comparison)
            print(f"  ERROR: {exc}", flush=True)
        elapsed = time.perf_counter() - case_start
        print(f"  elapsed: {elapsed:.2f}s", flush=True)
        original_results.append(comparison)
    original_summary = summarize(original_results, output_dir, "original")
    print(f"[original summary] {original_summary}", flush=True)

    mutated_summary = None
    if not args.skip_mutated:
        mutated_paths = make_mutated_inputs(data_paths, output_dir)
        mutated_results = []
        mutated_dir = output_dir / "modified"
        for index, path in enumerate(mutated_paths, start=1):
            print(f"[modified {index:03d}/{len(mutated_paths):03d}] {path.name}", flush=True)
            case_start = time.perf_counter()
            try:
                comparison = run_one(
                    fc,
                    path,
                    mutated_dir,
                    args.budget,
                    args.extra_response_units,
                    args.decode_mode,
                    generate_audio,
                    args.ref_audio_path,
                    args.tts_prompt_path,
                )
            except Exception as exc:
                comparison = {"sample_id": path.stem, "data_path": str(path), "success": False, "error": repr(exc), "matches": {}}
                sample_dir = mutated_dir / path.stem
                write_json(sample_dir / "comparison.json", comparison)
                print(f"  ERROR: {exc}", flush=True)
            elapsed = time.perf_counter() - case_start
            print(f"  elapsed: {elapsed:.2f}s", flush=True)
            mutated_results.append(comparison)
        mutated_summary = summarize(mutated_results, output_dir, "modified")
        print(f"[modified summary] {mutated_summary}", flush=True)

    write_json(output_dir / "run_config.json", vars(args))
    write_json(output_dir / "summary.json", {"original": original_summary, "modified": mutated_summary})
    print(f"[done] results written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

    