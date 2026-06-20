"""Duplex worker-runtime WebSocket A/B 对比基准测试（多轮统计版）

直接连接 Worker 的 /v1/worker/duplex 端点，发送真实音频，
对 normal 和 compile 模式各跑 N_ROUNDS 轮，汇总 LISTEN/SPEAK 稳态统计。

用法:
    cd /user/sunweiyue/lib/swy-dev/minicpmo45_service
    PYTHONPATH=. .venv/base/bin/python tests/bench_duplex_ws.py
"""

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import websockets

# ========== 配置 ==========

TESTS_DIR = Path(__file__).parent
REF_AUDIO_PATH = str(TESTS_DIR / "cases" / "common" / "ref_audio" / "BH-Ref-HT-F224-Ref06_82_U001_话题_3_348s-355s.wav")
USER_AUDIO_PATH = str(TESTS_DIR / "cases" / "common" / "user_audio" / "000_user_audio0.wav")

WORKERS = {
    "normal": "ws://localhost:22400",
    "compile": "ws://localhost:22401",
}

SYSTEM_PROMPT = "You are a helpful assistant."
CHUNK_DURATION_S = 1.0
SAMPLE_RATE = 16000
NUM_CHUNKS = 30       # 每轮发 30 chunk
N_ROUNDS = 5          # 每个 worker 跑 5 轮


@dataclass
class ChunkResult:
    """单个 chunk 结果"""
    turn: int
    status: str  # LISTEN / SPEAK
    wall_ms: float
    prefill_ms: float = 0
    llm_ms: float = 0
    tts_ms: float = 0
    t2w_ms: float = 0
    total_ms: float = 0
    n_tokens: int = 0
    n_tts_tokens: int = 0
    text: str = ""


@dataclass
class SessionResult:
    """一轮 session"""
    worker_name: str
    round_idx: int
    prepare_ms: float = 0
    chunks: List[ChunkResult] = field(default_factory=list)
    error: Optional[str] = None


async def run_duplex_session(
    worker_name: str,
    worker_base_url: str,
    user_audio: np.ndarray,
    ref_audio: np.ndarray,
    round_idx: int,
    verbose: bool = True,
) -> SessionResult:
    """连接 worker 并执行一轮 duplex session"""
    result = SessionResult(worker_name=worker_name, round_idx=round_idx)

    try:
        ws_url = f"{worker_base_url}/v1/worker/duplex"
        async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
            ref_audio_b64 = base64.b64encode(ref_audio.astype(np.float32).tobytes()).decode()

            t0 = time.perf_counter()
            await ws.send(json.dumps({
                "type": "duplex.session.prepare",
                "payload": {
                    "system_prompt": SYSTEM_PROMPT,
                    "voice": {
                        "ref_audio_base64": ref_audio_b64,
                        "tts_ref_audio_base64": ref_audio_b64,
                    },
                },
            }))
            resp = json.loads(await ws.recv())
            result.prepare_ms = (time.perf_counter() - t0) * 1000

            if resp.get("type") != "duplex.session.ready":
                result.error = f"prepare failed: {resp}"
                return result

            if verbose:
                print(f"  [R{round_idx}] prepared {result.prepare_ms:.0f}ms")

            chunk_size = int(CHUNK_DURATION_S * SAMPLE_RATE)
            total_samples = len(user_audio)

            for i in range(NUM_CHUNKS):
                start = (i * chunk_size) % total_samples
                end = min(start + chunk_size, total_samples)
                chunk = user_audio[start:end]

                audio_b64 = base64.b64encode(chunk.astype(np.float32).tobytes()).decode()

                t_send = time.perf_counter()
                await ws.send(json.dumps({
                    "type": "duplex.input.audio.append",
                    "payload": {
                        "audio_base64": audio_b64,
                    },
                }))

                metrics = {}
                result_payload = {}
                while True:
                    raw_resp = await ws.recv()
                    resp = json.loads(raw_resp)
                    if resp.get("type") == "error":
                        continue
                    payload = resp.get("payload") or {}
                    if resp.get("type") == "duplex.metrics.frame":
                        metrics = payload
                        continue
                    if resp.get("type") == "duplex.output.text.delta":
                        result_payload["text"] = payload.get("text", "")
                        continue
                    if resp.get("type") == "duplex.output.listen":
                        result_payload = {"is_listen": True, "kv_cache_length": payload.get("kv_cache_length")}
                        break
                    if resp.get("type") == "duplex.output.audio.delta":
                        result_payload.update({
                            "is_listen": False,
                            "text": payload.get("text", result_payload.get("text", "")),
                            "audio_data": payload.get("audio_base64"),
                            "end_of_turn": payload.get("end_of_turn", False),
                            "kv_cache_length": payload.get("kv_cache_length"),
                        })
                        break
                    result.error = f"unexpected runtime response: {resp}"
                    return result
                wall_ms = (time.perf_counter() - t_send) * 1000
                is_listen = result_payload.get("is_listen", True)
                status = "LISTEN" if is_listen else "SPEAK"

                cr = ChunkResult(
                    turn=result_payload.get("current_time", i + 1),
                    status=status,
                    wall_ms=wall_ms,
                    prefill_ms=metrics.get("prefill_ms", 0) or 0,
                )
                if not is_listen:
                    cr.llm_ms = result_payload.get("cost_llm_ms", 0) or 0
                    cr.tts_ms = result_payload.get("cost_tts_ms", 0) or 0
                    cr.t2w_ms = result_payload.get("cost_token2wav_ms", 0) or 0
                    cr.total_ms = result_payload.get("cost_all_ms", 0) or 0
                    cr.n_tokens = result_payload.get("n_tokens", 0) or 0
                    cr.n_tts_tokens = result_payload.get("n_tts_tokens", 0) or 0
                    cr.text = result_payload.get("text", "") or ""

                result.chunks.append(cr)

            # stop
            await ws.send(json.dumps({
                "type": "duplex.control.close",
                "payload": {"reason": "benchmark_done"},
            }))
            try:
                await asyncio.wait_for(ws.recv(), timeout=2.0)
            except Exception:
                pass

            # 本轮摘要
            n_listen = sum(1 for c in result.chunks if c.status == "LISTEN")
            n_speak = sum(1 for c in result.chunks if c.status == "SPEAK")
            speak_walls = [c.wall_ms for c in result.chunks if c.status == "SPEAK"]
            listen_walls = [c.wall_ms for c in result.chunks if c.status == "LISTEN"]
            if verbose:
                speak_avg = f"{np.mean(speak_walls):.0f}" if speak_walls else "N/A"
                listen_avg = f"{np.mean(listen_walls):.0f}" if listen_walls else "N/A"
                print(f"  [R{round_idx}] LISTEN={n_listen} SPEAK={n_speak} | "
                      f"listen_avg={listen_avg}ms speak_avg={speak_avg}ms")

    except Exception as e:
        result.error = str(e)
        print(f"  [R{round_idx}] ERROR: {e}")

    return result


def print_aggregate(all_results: Dict[str, List[SessionResult]]) -> None:
    """汇总多轮结果，打印对比"""
    print("\n" + "=" * 90)
    print("  汇总统计（所有轮次合并，排除每轮前 3 chunk）")
    print("=" * 90)

    summaries: Dict[str, Dict] = {}

    for name, sessions in all_results.items():
        all_listen: List[float] = []
        all_speak_wall: List[float] = []
        all_speak_tts: List[float] = []
        all_speak_llm: List[float] = []
        all_speak_t2w: List[float] = []
        all_prepare: List[float] = []
        total_listen_count = 0
        total_speak_count = 0

        for sess in sessions:
            if sess.error:
                continue
            all_prepare.append(sess.prepare_ms)
            # 排除每轮前 3 个 chunk（可能含初始 LISTEN + 首次 SPEAK 编译）
            stable_chunks = sess.chunks[3:]
            for c in stable_chunks:
                if c.status == "LISTEN":
                    all_listen.append(c.wall_ms)
                    total_listen_count += 1
                else:
                    all_speak_wall.append(c.wall_ms)
                    all_speak_tts.append(c.tts_ms)
                    all_speak_llm.append(c.llm_ms)
                    all_speak_t2w.append(c.t2w_ms)
                    total_speak_count += 1

        summaries[name] = {
            "n_sessions": len([s for s in sessions if not s.error]),
            "n_listen": total_listen_count,
            "n_speak": total_speak_count,
            "prepare_avg": np.mean(all_prepare) if all_prepare else 0,
            "listen_avg": np.mean(all_listen) if all_listen else 0,
            "listen_p50": np.median(all_listen) if all_listen else 0,
            "speak_wall_avg": np.mean(all_speak_wall) if all_speak_wall else 0,
            "speak_wall_p50": np.median(all_speak_wall) if all_speak_wall else 0,
            "speak_wall_min": np.min(all_speak_wall) if all_speak_wall else 0,
            "speak_wall_max": np.max(all_speak_wall) if all_speak_wall else 0,
            "tts_avg": np.mean(all_speak_tts) if all_speak_tts else 0,
            "llm_avg": np.mean(all_speak_llm) if all_speak_llm else 0,
            "t2w_avg": np.mean(all_speak_t2w) if all_speak_t2w else 0,
        }

    # 输出各 worker 统计
    for name, s in summaries.items():
        print(f"\n  [{name}] ({s['n_sessions']} rounds, "
              f"LISTEN={s['n_listen']}, SPEAK={s['n_speak']})")
        print(f"    prepare:       avg={s['prepare_avg']:.0f}ms")
        print(f"    LISTEN wall:   avg={s['listen_avg']:.0f}ms  p50={s['listen_p50']:.0f}ms")
        print(f"    SPEAK wall:    avg={s['speak_wall_avg']:.0f}ms  p50={s['speak_wall_p50']:.0f}ms  "
              f"min={s['speak_wall_min']:.0f}ms  max={s['speak_wall_max']:.0f}ms")
        print(f"      tts:         avg={s['tts_avg']:.0f}ms")
        print(f"      llm:         avg={s['llm_avg']:.0f}ms")
        print(f"      t2w:         avg={s['t2w_avg']:.0f}ms")

    # 对比表
    if len(summaries) == 2:
        names = list(summaries.keys())
        a, b = summaries[names[0]], summaries[names[1]]
        print(f"\n{'─' * 90}")
        print(f"  对比: {names[0]} vs {names[1]}")
        print(f"{'─' * 90}")

        def _delta(va: float, vb: float) -> str:
            d = vb - va
            pct = (d / va * 100) if va else 0
            return f"{d:>+7.1f}ms ({pct:>+5.1f}%)"

        rows = [
            ("LISTEN wall avg", a["listen_avg"], b["listen_avg"]),
            ("SPEAK wall avg", a["speak_wall_avg"], b["speak_wall_avg"]),
            ("  tts avg", a["tts_avg"], b["tts_avg"]),
            ("  llm avg", a["llm_avg"], b["llm_avg"]),
            ("  t2w avg", a["t2w_avg"], b["t2w_avg"]),
            ("prepare avg", a["prepare_avg"], b["prepare_avg"]),
            ("SPEAK count", a["n_speak"], b["n_speak"]),
            ("LISTEN count", a["n_listen"], b["n_listen"]),
        ]
        print(f"  {'metric':<20} {names[0]:>10} {names[1]:>10}   {'delta':>20}")
        for label, va, vb in rows:
            if "count" in label:
                print(f"  {label:<20} {int(va):>10} {int(vb):>10}   {int(vb)-int(va):>+10}")
            else:
                print(f"  {label:<20} {va:>9.0f}ms {vb:>9.0f}ms   {_delta(va, vb)}")


async def main():
    """多轮 A/B 对比"""
    user_audio, sr = sf.read(USER_AUDIO_PATH)
    if sr != SAMPLE_RATE:
        import librosa
        user_audio = librosa.resample(user_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    user_audio = user_audio.astype(np.float32)

    ref_audio, sr = sf.read(REF_AUDIO_PATH)
    if sr != SAMPLE_RATE:
        import librosa
        ref_audio = librosa.resample(ref_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    ref_audio = ref_audio.astype(np.float32)

    print(f"User audio: {len(user_audio)} samples ({len(user_audio)/SAMPLE_RATE:.1f}s)")
    print(f"Ref audio:  {len(ref_audio)} samples ({len(ref_audio)/SAMPLE_RATE:.1f}s)")
    print(f"Config: {NUM_CHUNKS} chunks × {CHUNK_DURATION_S}s, {N_ROUNDS} rounds per worker\n")

    all_results: Dict[str, List[SessionResult]] = {name: [] for name in WORKERS}

    # 交替测试：R1(normal) -> R1(compile) -> R2(normal) -> R2(compile) -> ...
    # 减少 GPU 热状态、内存碎片等系统性偏差
    for r in range(N_ROUNDS):
        for name, url in WORKERS.items():
            print(f"\n--- Round {r+1}/{N_ROUNDS}: {name} ---")
            sess = await run_duplex_session(name, url, user_audio, ref_audio, r + 1)
            all_results[name].append(sess)
            if sess.error:
                print(f"  ERROR: {sess.error}")

    # 汇总
    print_aggregate(all_results)


if __name__ == "__main__":
    asyncio.run(main())
