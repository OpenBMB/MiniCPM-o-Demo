#!/usr/bin/env python3
"""Omni duplex benchmark script.

Runs the duplex inference pipeline on video(s) and reports per-module
timing breakdown, separately for LISTEN and SPEAK decisions.

Usage:
    # Single video (defaults)
    CUDA_VISIBLE_DEVICES=0 .venv/base/bin/python benchmark.py

    # Custom video + ref audio
    CUDA_VISIBLE_DEVICES=0 .venv/base/bin/python benchmark.py \
        --video assets/samples/compile.mp4 \
        --ref-audio assets/ref_audio/ref_en_dlc_1.wav

    # Directory of videos
    CUDA_VISIBLE_DEVICES=0 .venv/base/bin/python benchmark.py \
        --video-dir /path/to/videos/

    # With torch.compile
    CUDA_VISIBLE_DEVICES=0 .venv/base/bin/python benchmark.py --compile
"""

import argparse
import gc
import json
import logging
import os
import subprocess
import time
from datetime import datetime

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")


def _collect_gpu_info(gpu_id: int = 0) -> dict:
    """Collect NVIDIA GPU information via torch.cuda and nvidia-smi."""
    info: dict = {}
    if not torch.cuda.is_available():
        info["available"] = False
        return info

    info["available"] = True
    info["device_count"] = torch.cuda.device_count()
    info["device_name"] = torch.cuda.get_device_name(gpu_id)
    info["cuda_version"] = torch.version.cuda
    props = torch.cuda.get_device_properties(gpu_id)
    info["total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
    info["major"] = props.major
    info["minor"] = props.minor
    info["multi_processor_count"] = props.multi_processor_count

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.max.sm",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            parts = [p.strip() for p in out.stdout.strip().split(",")]
            if len(parts) >= 8:
                info["driver_version"] = parts[0]
                info["memory_used_mb"] = float(parts[1])
                info["memory_total_mb"] = float(parts[2])
                info["temperature_c"] = int(parts[3])
                info["power_draw_w"] = float(parts[4])
                info["power_limit_w"] = float(parts[5])
                info["sm_clock_mhz"] = int(parts[6])
                info["sm_clock_max_mhz"] = int(parts[7])
    except Exception:
        pass

    return info


def main():
    from config import get_config
    cfg = get_config()

    parser = argparse.ArgumentParser(
        description="Omni duplex benchmark: per-module timing for LISTEN and SPEAK",
    )
    parser.add_argument("--model-path", type=str, default=None, help="Base model path")
    parser.add_argument("--pt-path", type=str, default=None, help="Extra weights path (.pt)")
    parser.add_argument("--ref-audio", type=str, default=None, help="Reference audio path for TTS")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID (default: 0)")
    parser.add_argument("--video", type=str, nargs="+", default=None, help="MP4 video path(s)")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory containing MP4 videos")
    parser.add_argument(
        "--system-prompt", type=str, default="Streaming Omni Conversation.",
        help="System prompt content",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=0,
        help="Max 1-second chunks per video (0 = all)",
    )
    parser.add_argument("--compile", action="store_true", help="Apply torch.compile before benchmark")
    parser.add_argument(
        "--compile-mode", type=str, default="default",
        help="torch.compile mode (default/reduce-overhead/max-autotune)",
    )
    parser.add_argument(
        "--attn-implementation", type=str, default=None,
        help="Attention impl (auto/flash_attention_2/sdpa)",
    )
    args = parser.parse_args()

    model_path = args.model_path or cfg.model.model_path
    pt_path = args.pt_path or cfg.model.pt_path
    attn_impl = args.attn_implementation or cfg.attn_implementation
    gpu_id = args.gpu_id

    def _is_quantized(path: str) -> bool:
        cfg_file = os.path.join(path, "config.json")
        if not os.path.isfile(cfg_file):
            return False
        try:
            import json as _json
            with open(cfg_file, "r", encoding="utf-8") as f:
                c = _json.load(f)
            qcfg = c.get("quantization_config")
            return bool(qcfg and qcfg.get("quant_method"))
        except Exception:
            return False

    is_quantized = _is_quantized(model_path)

    project_root = os.path.dirname(os.path.abspath(__file__))
    ref_audio_path = args.ref_audio or os.path.join(
        project_root, "assets", "ref_audio", "ref_en_dlc_1.wav"
    )

    video_paths = args.video
    video_dir = args.video_dir
    if video_paths is None and video_dir is None:
        video_paths = [os.path.join(project_root, "assets", "samples", "compile.mp4")]

    logger.info("=" * 60)
    logger.info("Omni Duplex Benchmark")
    logger.info("=" * 60)
    logger.info(f"PyTorch:       {torch.__version__}")
    logger.info(f"CUDA:          {torch.version.cuda}")
    if torch.cuda.is_available():
        logger.info(f"GPU:           {torch.cuda.get_device_name(gpu_id)}")
    logger.info(f"Model:         {model_path}")
    logger.info(f"PT path:       {pt_path}")
    logger.info(f"Ref audio:     {ref_audio_path}")
    logger.info(f"Attn impl:     {attn_impl}")
    logger.info(f"Quantized:     {is_quantized}")
    logger.info(f"Videos:        {video_paths or video_dir}")
    logger.info(f"System prompt: {args.system_prompt}")
    logger.info(f"Max chunks:    {args.max_chunks or 'all'}")
    logger.info(f"Compile:       {args.compile}")
    logger.info("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    total_start = time.time()

    # ── 1. Load model ──
    logger.info("[1/3] Loading model...")
    t0 = time.time()

    from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO

    resolved_attn = attn_impl
    if resolved_attn == "auto":
        try:
            from transformers.utils import is_flash_attn_2_available
            if is_flash_attn_2_available():
                resolved_attn = "flash_attention_2"
            else:
                resolved_attn = "sdpa"
        except ImportError:
            resolved_attn = "sdpa"
        logger.info(f"attn_implementation: auto -> {resolved_attn}")

    model = MiniCPMO.from_pretrained(
        model_path,
        trust_remote_code=True,
        _attn_implementation=resolved_attn,
    )
    if is_quantized:
        model.eval().cuda()
        logger.info("Quantized model detected — skipping .bfloat16() cast")
    else:
        model.bfloat16().eval().cuda()
    load_time = time.time() - t0
    logger.info(f"[1/3] Model loaded ({load_time:.1f}s)")

    # ── 2. Unified initialization ──
    logger.info("[2/3] init_unified...")
    t0 = time.time()
    model.init_unified(
        pt_path=pt_path,
        preload_both_tts=True,
        device="cuda",
        chat_vocoder=cfg.chat_vocoder,
    )
    init_time = time.time() - t0
    logger.info(f"[2/3] init_unified done ({init_time:.1f}s)")

    # ── 2.5 Optional torch.compile ──
    if args.compile:
        skip_modules = ["llm.model"] if is_quantized else None
        logger.info("[2.5/3] apply_torch_compile (mode=%s, skip=%s)...", args.compile_mode, skip_modules)
        t0 = time.time()
        model.apply_torch_compile(mode=args.compile_mode, dynamic=True, skip_modules=skip_modules)
        logger.info(f"[2.5/3] apply_torch_compile done ({time.time() - t0:.1f}s)")

    # ── 3. Benchmark ──
    logger.info("[3/3] Running benchmark...")
    t0 = time.time()
    results = model.benchmark(
        video_paths=video_paths,
        video_dir=video_dir,
        ref_audio_path=ref_audio_path,
        system_prompt=args.system_prompt,
        max_chunks_per_video=args.max_chunks,
    )
    benchmark_time = time.time() - t0
    logger.info(f"[3/3] Benchmark done ({benchmark_time:.1f}s)")

    # ── Collect GPU info (after inference, before cleanup) ──
    gpu_info = _collect_gpu_info(gpu_id)

    # ── Cleanup ──
    del model
    gc.collect()
    torch.cuda.empty_cache()

    total = time.time() - total_start

    # ── Build and save benchmark.json ──
    output = {
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": gpu_info,
        },
        "config": {
            "model_path": model_path,
            "pt_path": pt_path,
            "attn_implementation": resolved_attn,
            "quantized": is_quantized,
            "chat_vocoder": cfg.chat_vocoder,
        },
        "parameters": {
            "video_paths": video_paths,
            "video_dir": video_dir,
            "ref_audio_path": ref_audio_path,
            "system_prompt": args.system_prompt,
            "max_chunks_per_video": args.max_chunks,
            "compile": args.compile,
            "compile_mode": args.compile_mode if args.compile else None,
            "gpu_id": gpu_id,
        },
        "timing": {
            "model_load_s": round(load_time, 2),
            "init_unified_s": round(init_time, 2),
            "benchmark_s": round(benchmark_time, 2),
            "total_script_s": round(total, 2),
        },
        "results": results,
    }

    output_path = os.path.join(project_root, "benchmark.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"Total script time: {total:.1f}s")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 60)

    print("\n" + json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
