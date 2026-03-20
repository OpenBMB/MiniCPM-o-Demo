#!/usr/bin/env python3
"""Benchmark compiled vs eager duplex inference on the same video.

Loads the model once (with compile=True), then runs the same omni full-duplex
session twice: once with compiled modules, once with eager modules.
Prints a side-by-side timing comparison at the end.

Usage:
    CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=./torch_compile_cache \
        PYTHONPATH=. .venv/base/bin/python test_compile_bench.py
"""

import os
import sys
import time
import logging
import torch
from config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("compile_bench")

VIDEO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets", "samples", "compile.mp4",
)
MAX_CHUNKS = 8


def module_type_label(mod) -> str:
    cls = type(mod).__name__
    if cls == "OptimizedModule":
        return f"OptimizedModule (compiled)"
    return f"{cls} (eager)"


def print_header(label: str, model):
    active = getattr(model, "_compile_active", "N/A")
    llm_label = module_type_label(model.llm.model)
    tts_label = module_type_label(model.tts.model) if hasattr(model.tts, "model") else "N/A"
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  _compile_active = {active}")
    print(f"  llm.model = {llm_label}")
    print(f"  tts.model = {tts_label}")
    print(f"{'='*70}")


def run_bench(model, label: str) -> dict:
    print_header(label, model)
    t0 = time.time()
    result = model.benchmark(
        video_paths=[VIDEO_PATH],
        max_chunks_per_video=MAX_CHUNKS,
    )
    elapsed = time.time() - t0
    print(f"  [{label}] done in {elapsed:.1f}s, "
          f"units={result.get('num_units', 0)}, "
          f"listen={result.get('listen_count', 0)}, "
          f"speak={result.get('speak_count', 0)}")
    return result


def format_stats(stats: dict, key_path: str) -> str:
    keys = key_path.split(".")
    d = stats
    for k in keys:
        d = d.get(k, {})
    if not d:
        return "N/A"
    return f"avg={d.get('avg', 0):.0f}ms  min={d.get('min', 0):.0f}ms  max={d.get('max', 0):.0f}ms"


def print_comparison(compiled_result: dict, eager_result: dict):
    print("\n")
    print("=" * 70)
    print("  Compiled vs Eager 对比")
    print("=" * 70)

    rows = [
        ("总用时",            "total_time",   "s",  True),
    ]

    # top-level
    for label, key, unit, is_time in rows:
        cv = compiled_result.get(key, 0)
        ev = eager_result.get(key, 0)
        if is_time:
            diff_pct = ((ev - cv) / cv * 100) if cv > 0 else 0
            print(f"  {label:20s}  compiled={cv:.1f}{unit}  eager={ev:.1f}{unit}  "
                  f"差异={diff_pct:+.1f}%")
        else:
            print(f"  {label:20s}  compiled={cv}  eager={ev}")

    print(f"  {'单位数':20s}  compiled={compiled_result.get('num_units',0)}  "
          f"eager={eager_result.get('num_units',0)}")

    # per-decision-type stats
    for decision in ("listen", "speak"):
        cs = compiled_result.get(f"{decision}_stats", {})
        es = eager_result.get(f"{decision}_stats", {})
        cc = cs.get("count", 0)
        ec = es.get("count", 0)
        if cc == 0 and ec == 0:
            continue

        print(f"\n  ── {decision.upper()} (compiled n={cc}, eager n={ec}) ──")

        metric_paths = [
            ("prefill total",    "prefill.total"),
            ("  vision_process", "prefill.vision_process"),
            ("  vision_embed",   "prefill.vision_embed"),
            ("  vision_feed",    "prefill.vision_feed"),
            ("  audio_process",  "prefill.audio_process"),
            ("  audio_embed",    "prefill.audio_embed"),
            ("  audio_feed",     "prefill.audio_feed"),
            ("generate total",   "generate.total"),
            ("  llm",            "generate.llm"),
            ("  tts_prep",       "generate.tts_prep"),
            ("  tts",            "generate.tts"),
            ("  token2wav",      "generate.token2wav"),
            ("unit_total",       "unit_total"),
        ]

        for metric_label, path in metric_paths:
            keys = path.split(".")
            cd = cs
            for k in keys:
                cd = cd.get(k, {}) if isinstance(cd, dict) else {}
            ed = es
            for k in keys:
                ed = ed.get(k, {}) if isinstance(ed, dict) else {}

            c_avg = cd.get("avg", 0) if isinstance(cd, dict) else 0
            e_avg = ed.get("avg", 0) if isinstance(ed, dict) else 0

            if c_avg == 0 and e_avg == 0:
                continue

            diff_pct = ((e_avg - c_avg) / c_avg * 100) if c_avg > 0 else 0
            arrow = "↑ slower" if diff_pct > 2 else ("↓ faster" if diff_pct < -2 else "≈")
            print(f"    {metric_label:18s}  compiled={c_avg:6.0f}ms  eager={e_avg:6.0f}ms  "
                  f"{diff_pct:+6.1f}% {arrow}")

    print("=" * 70)


def main():
    cfg = get_config()

    print("=" * 70)
    print("  Compiled vs Eager Duplex Benchmark")
    print("=" * 70)
    print(f"  Model:      {cfg.model.model_path}")
    print(f"  Video:      {VIDEO_PATH}")
    print(f"  Max chunks: {MAX_CHUNKS}")
    print()

    from core.processors.unified import UnifiedProcessor

    logger.info("加载模型 (compile=True)...")
    t0 = time.time()
    processor = UnifiedProcessor(
        model_path=cfg.model.model_path,
        pt_path=cfg.model.pt_path,
        ref_audio_path=cfg.ref_audio_path,
        compile=True,
        chat_vocoder=cfg.chat_vocoder,
        attn_implementation=cfg.attn_implementation,
    )
    logger.info(f"模型加载完成 ({time.time() - t0:.1f}s)")

    model = processor.model

    # ── Round 1: Compiled ──
    model.set_compile_enabled(True)
    compiled_result = run_bench(model, "COMPILED")

    # ── Reset state between runs ──
    torch.cuda.empty_cache()

    # ── Round 2: Eager ──
    model.set_compile_enabled(False)
    eager_result = run_bench(model, "EAGER")

    # ── Comparison ──
    print_comparison(compiled_result, eager_result)


if __name__ == "__main__":
    main()
