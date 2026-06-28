#!/usr/bin/env python3
"""Batch evaluate FC duplex inference against training-data token streams."""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from minicpm_o5_sdk import O5DuplexTrainingData, O5TokenizerID, OpenAIToolDefinition

from core.processors import UnifiedProcessor
from core.schemas.fc_duplex import FcDuplexConfig, FcDuplexOfflineInput


DEFAULT_BASE_MODEL = "/user/heweiquan/project/MiniCPM-o-4_5"
DEFAULT_PT_PATH = "/user/heweiquan/models/minicpm-o45-fc-overfit/minicpm-v_100.pt"
DEFAULT_DATA_DIR = "/user/heweiquan/project/MiniCPM-o-4_5-fc_duplex_infer/data/training_data"
DEFAULT_OUTPUT_DIR = "/user/heweiquan/project/MiniCPM-o-Demo-FC/fc_duplex_test_results"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_case(data_path: Path) -> Tuple[Dict[str, Any], str, Optional[List[Dict[str, Any]]], str, List[str]]:
    structure = read_json(data_path)
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

    sample_id = data_path.stem
    media_dir = data_path.parent.parent / "media" / sample_id
    audio_path = media_dir / "user_audio_0.opus"
    if not audio_path.exists():
        raise FileNotFoundError(f"missing user audio: {audio_path}")
    return structure, system_prompt, tools, str(audio_path), tool_call_ids


def tokenize_training_data(data_path: Path) -> List[int]:
    structure = read_json(data_path)
    media_dir = data_path.parent.parent / "media" / data_path.stem
    tokenized = O5DuplexTrainingData.load_structure(
        structure,
        data_root=str(media_dir),
    ).tokenize(tokenizer_id=O5TokenizerID.O45_FC).tokenized_data
    return list(tokenized.input_ids)


def normalize_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for call in tool_calls or []:
        normalized.append({
            "tool_call_id": call.get("tool_call_id"),
            "name": call.get("name"),
            "arguments": call.get("arguments"),
            "error": call.get("error"),
        })
    return normalized


def comparable_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "name": call.get("name"),
            "arguments": call.get("arguments"),
            "error": call.get("error"),
        }
        for call in (tool_calls or [])
    ]


def first_diff(a: str, b: str) -> Optional[Dict[str, Any]]:
    if a == b:
        return None
    n = min(len(a), len(b))
    index = next((i for i in range(n) if a[i] != b[i]), n)
    return {
        "index": index,
        "gt_context": a[max(0, index - 80): index + 160],
        "pred_context": b[max(0, index - 80): index + 160],
    }


def run_one(
    fc,
    data_path: Path,
    output_dir: Path,
    budget: int,
    extra_response_units: int,
    decode_mode: str,
) -> Dict[str, Any]:
    sample_id = data_path.stem
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    structure, system_prompt, tools, audio_path, tool_call_ids = load_case(data_path)
    gt_ids = tokenize_training_data(data_path)
    gt_decoded = fc.decode_output(output_ids=gt_ids, tools=tools)
    gt_render = gt_decoded.get("output_render", "")

    start = time.time()
    pred = fc.offline_inference(
        FcDuplexOfflineInput(
            system_prompt=system_prompt,
            tools=tools,
            user_audio_path=audio_path,
            tool_call_ids=tool_call_ids,
            config=FcDuplexConfig(
                decode_mode=decode_mode,
                non_spoken_budget_per_unit=budget,
                extra_response_units=extra_response_units,
            ),
        ),
        non_spoken_budget_per_unit=budget,
    )
    elapsed = time.time() - start

    pred_render = pred.output_render or ""
    gt_tool_calls = normalize_tool_calls(gt_decoded.get("tool_calls", []))
    pred_tool_calls = normalize_tool_calls(pred.tool_calls)
    comparison = {
        "sample_id": sample_id,
        "data_path": str(data_path),
        "audio_path": audio_path,
        "success": pred.success,
        "error": pred.error,
        "elapsed_sec": elapsed,
        "gt": {
            "n_tokens": len(gt_ids),
            "spoken_text": gt_decoded.get("spoken_text", ""),
            "think_text": gt_decoded.get("think_text", ""),
            "tool_calls": gt_tool_calls,
            "tool_call_ids": tool_call_ids,
        },
        "prediction": {
            "n_tokens": len(pred.output_ids),
            "spoken_text": pred.spoken_text,
            "think_text": pred.think_text,
            "tool_calls": pred_tool_calls,
            "total_units": pred.total_units,
            "n_audio_units": pred.n_audio_units,
        },
        "matches": {
            "token_stream_exact": gt_render == pred_render,
            "spoken_text_exact": gt_decoded.get("spoken_text", "") == pred.spoken_text,
            "think_text_exact": gt_decoded.get("think_text", "") == pred.think_text,
            "tool_calls_semantic_exact": comparable_tool_calls(gt_tool_calls) == comparable_tool_calls(pred_tool_calls),
            "tool_call_ids_exact": tool_call_ids == [c.get("tool_call_id") for c in pred_tool_calls],
        },
        "first_token_stream_diff": first_diff(gt_render, pred_render),
    }

    write_json(sample_dir / "source.json", structure)
    write_text(sample_dir / "gt_token_stream.txt", gt_render)
    write_text(sample_dir / "pred_token_stream.txt", pred_render)
    write_json(sample_dir / "comparison.json", comparison)
    write_json(sample_dir / "units_info.json", [u.model_dump() for u in pred.units_info])
    return comparison


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
        return bool((item.get("matches") or {}).get(key))

    summary = {
        "group": group,
        "total": len(comparisons),
        "success": sum(1 for c in comparisons if c.get("success")),
        "token_stream_exact": sum(1 for c in comparisons if matched(c, "token_stream_exact")),
        "spoken_text_exact": sum(1 for c in comparisons if matched(c, "spoken_text_exact")),
        "think_text_exact": sum(1 for c in comparisons if matched(c, "think_text_exact")),
        "tool_calls_semantic_exact": sum(1 for c in comparisons if matched(c, "tool_calls_semantic_exact")),
        "tool_call_ids_exact": sum(1 for c in comparisons if matched(c, "tool_call_ids_exact")),
        "failed_samples": [c["sample_id"] for c in comparisons if not c.get("success")],
        "token_diff_samples": [c["sample_id"] for c in comparisons if not matched(c, "token_stream_exact")],
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
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--extra-response-units", type=int, default=4)
    parser.add_argument("--decode-mode", default="greedy")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-mutated", action="store_true")
    args = parser.parse_args()

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
        try:
            comparison = run_one(fc, path, original_dir, args.budget, args.extra_response_units, args.decode_mode)
        except Exception as exc:
            comparison = {"sample_id": path.stem, "data_path": str(path), "success": False, "error": repr(exc), "matches": {}}
            sample_dir = original_dir / path.stem
            write_json(sample_dir / "comparison.json", comparison)
            print(f"  ERROR: {exc}", flush=True)
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
            try:
                comparison = run_one(fc, path, mutated_dir, args.budget, args.extra_response_units, args.decode_mode)
            except Exception as exc:
                comparison = {"sample_id": path.stem, "data_path": str(path), "success": False, "error": repr(exc), "matches": {}}
                sample_dir = mutated_dir / path.stem
                write_json(sample_dir / "comparison.json", comparison)
                print(f"  ERROR: {exc}", flush=True)
            mutated_results.append(comparison)
        mutated_summary = summarize(mutated_results, output_dir, "modified")
        print(f"[modified summary] {mutated_summary}", flush=True)

    write_json(output_dir / "run_config.json", vars(args))
    write_json(output_dir / "summary.json", {"original": original_summary, "modified": mutated_summary})
    print(f"[done] results written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
