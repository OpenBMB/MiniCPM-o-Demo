#!/usr/bin/env python3
"""torch.compile pre-compilation script.

Runs torch.compile + warmup ahead of time, writing Triton kernel caches to
disk so that subsequent worker starts can skip the expensive code-generation
and compilation steps.

Usage:
    # Use settings from config.json
    CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=./torch_compile_cache .venv/base/bin/python precompile.py
"""

import argparse
import gc
import logging
import os
import sys
import time

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("precompile")


def main():
    from config import get_config
    cfg = get_config()

    parser = argparse.ArgumentParser(
        description="torch.compile pre-compilation: generate Triton kernel caches ahead of time",
    )
    parser.add_argument("--model-path", type=str, default=None, help="Base model path")
    parser.add_argument("--pt-path", type=str, default=None, help="Extra weights path (.pt)")
    parser.add_argument("--ref-audio-path", type=str, default=None, help="Reference audio path for TTS")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID (default: 0)")
    parser.add_argument("--mode", type=str, default="default", help="Compile mode (default/reduce-overhead/max-autotune)")
    parser.add_argument("--max-warmup-chunks", type=int, default=10, help="Max 1s chunks to process during warmup")
    parser.add_argument("--warmup-video", type=str, default=None, help="MP4 video path for warmup")
    parser.add_argument("--attn-implementation", type=str, default=None, help="Attention impl (auto/flash_attention_2/sdpa)")
    args = parser.parse_args()

    model_path = args.model_path or cfg.model.model_path
    pt_path = args.pt_path or cfg.model.pt_path
    ref_audio_path = args.ref_audio_path or cfg.ref_audio_path
    attn_impl = args.attn_implementation or cfg.attn_implementation
    gpu_id = args.gpu_id

    logger.info("=" * 60)
    logger.info("torch.compile pre-compilation")
    logger.info("=" * 60)
    logger.info(f"PyTorch:       {torch.__version__}")
    logger.info(f"CUDA:          {torch.version.cuda}")
    if torch.cuda.is_available():
        logger.info(f"GPU:           {torch.cuda.get_device_name(gpu_id)}")
    logger.info(f"Model:         {model_path}")
    logger.info(f"PT path:       {pt_path}")
    logger.info(f"Ref audio:     {ref_audio_path}")
    logger.info(f"Attn impl:     {attn_impl}")
    logger.info(f"Compile mode:  {args.mode}")

    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "./torch_compile_cache")
    autograd_cache = os.environ.get("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    logger.info(f"Cache dir:     {cache_dir}")
    logger.info(f"AOTAutograd:   {autograd_cache}")
    logger.info("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    total_start = time.time()

    # ── 1. Load model ──
    logger.info("[1/4] Loading model...")
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
    model.bfloat16().eval().cuda()
    logger.info(f"[1/4] Model loaded ({time.time() - t0:.1f}s)")

    # ── 2. Unified initialization ──
    logger.info("[2/4] init_unified...")
    t0 = time.time()
    model.init_unified(
        pt_path=pt_path,
        preload_both_tts=True,
        device="cuda",
        chat_vocoder=cfg.chat_vocoder,
    )
    logger.info(f"[2/4] init_unified done ({time.time() - t0:.1f}s)")

    # ── 3. Wrap sub-modules with torch.compile ──
    logger.info("[3/4] apply_torch_compile...")
    t0 = time.time()
    model.apply_torch_compile(mode=args.mode, dynamic=True)
    logger.info(f"[3/4] apply_torch_compile done ({time.time() - t0:.1f}s)")

    # ── 4. Warmup (triggers actual Triton compilation) ──
    logger.info("[4/4] warmup_compile (real duplex inference to trigger compilation)...")
    t0 = time.time()
    model.warmup_compile(
        warmup_video_path=args.warmup_video,
        ref_audio_path=ref_audio_path,
        max_warmup_chunks=args.max_warmup_chunks,
        total_estimate_seconds=1000,
    )
    logger.info(f"[4/4] warmup_compile done ({time.time() - t0:.1f}s)")

    # ── Cleanup ──
    del model
    gc.collect()
    torch.cuda.empty_cache()

    total = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"Pre-compilation finished in {total:.1f}s")
    logger.info(f"Cache written to: {cache_dir}")
    logger.info("=" * 60)
    logger.info("Compilation done!")


if __name__ == "__main__":
    main()
