#!/usr/bin/env python3
"""Test script for torch.compile dynamic switching between modes.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/base/bin/python test_compile_switch.py
"""

import time
import torch
from config import get_config


def module_type_label(mod) -> str:
    cls = type(mod).__name__
    if cls == "OptimizedModule":
        return f"OptimizedModule (compiled) -> wraps {type(mod._orig_mod).__name__}"
    return f"{cls} (eager)"


def print_module_state(processor, label: str):
    model = processor.model
    llm_mod = model.llm.model
    tts_mod = model.tts.model if hasattr(model.tts, "model") else None

    compile_active = getattr(model, "_compile_active", "N/A")

    print(f"\n{'='*60}")
    print(f"  [{label}]")
    print(f"  _compile_active = {compile_active}")
    print(f"  llm.model       = {module_type_label(llm_mod)}")
    if tts_mod is not None:
        print(f"  tts.model       = {module_type_label(tts_mod)}")
    print(f"{'='*60}")


def main():
    cfg = get_config()

    print("=" * 60)
    print("  torch.compile 动态切换测试")
    print("=" * 60)
    print(f"  Model: {cfg.model.model_path}")
    print(f"  compile: True (强制开启以测试切换)")
    print()

    from core.processors.unified import UnifiedProcessor

    print("[1/2] 加载模型 (compile=True)...")
    t0 = time.time()
    processor = UnifiedProcessor(
        model_path=cfg.model.model_path,
        pt_path=cfg.model.pt_path,
        ref_audio_path=cfg.ref_audio_path,
        compile=True,
        chat_vocoder=cfg.chat_vocoder,
        attn_implementation=cfg.attn_implementation,
    )
    print(f"      模型加载完成 ({time.time() - t0:.1f}s)")

    print_module_state(processor, "初始状态 (compile 完成后)")

    # ── 测试切换 ──
    print("\n\n[2/2] 开始切换测试...\n")

    steps = [
        ("set_chat_mode",        "Chat 模式 → 期望: eager",        False),
        ("set_duplex_mode",      "Duplex 模式 → 期望: compiled",   True),
        ("set_half_duplex_mode", "Half-Duplex 模式 → 期望: eager", False),
        ("set_duplex_mode",      "Duplex 模式 → 期望: compiled",   True),
        ("set_chat_mode",        "Chat 模式 → 期望: eager",        False),
    ]

    all_pass = True
    for method_name, desc, expect_compiled in steps:
        t = time.time()
        getattr(processor, method_name)()
        elapsed_ms = (time.time() - t) * 1000

        model = processor.model
        is_compiled = getattr(model, "_compile_active", None)

        llm_is_optimized = type(model.llm.model).__name__ == "OptimizedModule"
        tts_is_optimized = (
            type(model.tts.model).__name__ == "OptimizedModule"
            if hasattr(model.tts, "model") else None
        )

        ok = (is_compiled == expect_compiled)
        if llm_is_optimized != expect_compiled:
            ok = False
        if tts_is_optimized is not None and tts_is_optimized != expect_compiled:
            ok = False

        status = "PASS ✓" if ok else "FAIL ✗"
        if not ok:
            all_pass = False

        print(f"  {status}  {desc}")
        print(f"         _compile_active={is_compiled}, "
              f"llm.model={module_type_label(model.llm.model)}, "
              f"tts.model={module_type_label(model.tts.model) if hasattr(model.tts, 'model') else 'N/A'}, "
              f"耗时={elapsed_ms:.1f}ms")
        print()

    # ── 权重一致性验证 ──
    print("-" * 60)
    print("  权重共享验证 (eager 和 compiled 应共享同一份权重):")
    model = processor.model

    processor.set_duplex_mode()
    compiled_llm = model.llm.model
    compiled_ptr = next(compiled_llm._orig_mod.parameters()).data_ptr()

    processor.set_chat_mode()
    eager_llm = model.llm.model
    eager_ptr = next(eager_llm.parameters()).data_ptr()

    ptrs_match = compiled_ptr == eager_ptr
    print(f"  llm.model 权重 data_ptr 一致: {ptrs_match} "
          f"({'PASS ✓' if ptrs_match else 'FAIL ✗'})")
    if not ptrs_match:
        all_pass = False

    print()
    print("=" * 60)
    if all_pass:
        print("  所有测试通过 ✓")
    else:
        print("  存在失败项 ✗")
    print("=" * 60)


if __name__ == "__main__":
    main()
