#!/usr/bin/env python3
"""
launch.py — Cross-platform Python launcher (replaces start_all.sh)

Usage:
    # Use all GPUs (reads CUDA_VISIBLE_DEVICES, defaults to GPU 0)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python launch.py

    # Specify GPUs directly
    python launch.py --gpus 0,1,2,3

    # Fall back to HTTP (not recommended; browser mic/camera require HTTPS)
    python launch.py --http

    # Specify a custom Python interpreter (default: auto-detects .venv/base/bin/python)
    python launch.py --python /path/to/python

    # Stop all running services
    python launch.py --stop
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import json
import threading
import http.client
import shutil
from pathlib import Path

# ──────────────────────────── Color output ────────────────────────────

COLORS = {
    "gateway": "\033[36m",   # cyan
    "worker0": "\033[32m",   # green
    "worker1": "\033[33m",   # yellow
    "worker2": "\033[34m",   # blue
    "worker3": "\033[35m",   # magenta
    "info":    "\033[1;37m", # bold white
    "ok":      "\033[1;32m", # bold green
    "warn":    "\033[1;33m", # bold yellow
    "err":     "\033[1;31m", # bold red
    "reset":   "\033[0m",
}

def cprint(tag: str, msg: str):
    color = COLORS.get(tag, COLORS["info"])
    reset = COLORS["reset"]
    prefix = f"[{tag.upper():>8}]"
    print(f"{color}{prefix}{reset} {msg}", flush=True)

# ──────────────────────────── Helpers ────────────────────────────

ROOT = Path(__file__).parent.resolve()
TMP_DIR = ROOT / "tmp"
PID_FILE = TMP_DIR / "launch.pids"

def find_python() -> str:
    """Locate a Python interpreter, in priority order."""
    candidates = [
        ROOT / ".venv" / "base" / "bin" / "python",
        ROOT / ".venv" / "base" / "bin" / "python3",
        Path(sys.executable),  # the interpreter running this script
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # Fall back to PATH
    found = shutil.which("python3") or shutil.which("python")
    if found:
        return found
    raise RuntimeError("Cannot find a Python interpreter. Use --python to specify one.")

def load_config() -> dict:
    """Load config.json; return empty dict on failure."""
    cfg_path = ROOT / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def get_config_value(cfg: dict, *keys, default=None):
    """Safely read a nested config field."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
    return node if node is not None else default

def wait_for_worker(port: int, timeout: int = 600, tag: str = "worker") -> bool:
    """
    Poll a worker's HTTP health endpoint until model_loaded is True or timeout expires.
    Prints progress every 15 seconds.
    Returns True on success, False on timeout.
    """
    deadline   = time.time() + timeout
    start      = time.time()
    next_print = start + 5
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
                if body.get("model_loaded"):
                    return True
        except Exception:
            pass
        now = time.time()
        if now >= next_print:
            elapsed   = int(now - start)
            remaining = int(deadline - now)
            cprint(tag, f"Model loading… waited {elapsed}s, up to {remaining}s remaining "
                        f"(a 22 GB model may take 3-5 min on first load)")
            next_print = now + 15
        time.sleep(2)
    return False

def stream_output(proc: subprocess.Popen, tag: str):
    """Background thread: forward subprocess stdout + stderr in real time."""
    def _read(stream):
        try:
            for line in iter(stream.readline, b""):
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    cprint(tag, text)
        except Exception:
            pass
    t1 = threading.Thread(target=_read, args=(proc.stdout,), daemon=True)
    t2 = threading.Thread(target=_read, args=(proc.stderr,), daemon=True)
    t1.start()
    t2.start()

def save_pids(pids: dict):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with open(PID_FILE, "w") as f:
        json.dump(pids, f)

def load_pids() -> dict:
    if PID_FILE.exists():
        try:
            with open(PID_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def stop_services():
    """Stop all services recorded in the PID file."""
    pids = load_pids()
    if not pids:
        cprint("warn", "No PID file found. Trying to kill processes by name...")
        os.system("pkill -f 'gateway.py|worker.py' 2>/dev/null")
        return
    for name, pid in pids.items():
        try:
            os.kill(pid, signal.SIGTERM)
            cprint("info", f"Sent SIGTERM to {name} (PID {pid})")
        except ProcessLookupError:
            cprint("warn", f"{name} (PID {pid}) is no longer running")
        except Exception as e:
            cprint("err", f"Could not stop {name}: {e}")
    PID_FILE.unlink(missing_ok=True)
    cprint("ok", "All services stopped.")

# ──────────────────────────── Main startup ────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MiniCPM-o-Demo launcher — cross-platform alternative to start_all.sh"
    )
    parser.add_argument("--gpus",    type=str, default=None,
                        help="Comma-separated GPU IDs, e.g. 0,1,2,3 "
                             "(default: reads CUDA_VISIBLE_DEVICES, falls back to 0)")
    parser.add_argument("--http",    action="store_true",
                        help="Use HTTP instead of HTTPS (not recommended)")
    parser.add_argument("--python",  type=str, default=None,
                        help="Path to the Python interpreter to use")
    parser.add_argument("--stop",    action="store_true",
                        help="Stop all services launched by a previous run")
    parser.add_argument("--worker-timeout", type=int, default=600,
                        help="Seconds to wait for each worker to become ready (default: 600)")
    args = parser.parse_args()

    # ── Stop mode ──
    if args.stop:
        stop_services()
        return

    # ── Load config ──
    cfg = load_config()
    gateway_port     = get_config_value(cfg, "service", "gateway_port",     default=8006)
    worker_base_port = get_config_value(cfg, "service", "worker_base_port", default=22400)

    # ── Resolve GPU list ──
    raw_gpus = args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_ids  = [g.strip() for g in raw_gpus.split(",") if g.strip()]
    if not gpu_ids:
        gpu_ids = ["0"]

    # ── Resolve Python interpreter ──
    python_exe = args.python or find_python()
    cprint("info", f"Python      : {python_exe}")
    cprint("info", f"Project root: {ROOT}")
    cprint("info", f"GPUs        : {gpu_ids}")
    cprint("info", f"Gateway port: {gateway_port}")
    cprint("info", f"Worker base port: {worker_base_port}")
    print()

    # ── Base environment ──
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(ROOT)
    if "TORCHINDUCTOR_CACHE_DIR" not in base_env:
        base_env["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / "torch_compile_cache")

    processes: dict[str, subprocess.Popen] = {}
    pids: dict[str, int] = {}
    worker_addresses = []

    # ─────────────── Start workers ───────────────
    for idx, gpu_id in enumerate(gpu_ids):
        worker_port = worker_base_port + idx
        worker_addresses.append(f"localhost:{worker_port}")
        tag = f"worker{idx}"

        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        cmd = [
            python_exe, "-u",           # -u: unbuffered output
            str(ROOT / "worker.py"),
            "--worker-index", str(idx),
            "--gpu-id", gpu_id,
            "--port", str(worker_port),
        ]

        cprint("info", f"Starting Worker {idx} (GPU {gpu_id}, port {worker_port})...")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes[tag] = proc
        pids[tag] = proc.pid
        stream_output(proc, tag)
        cprint("ok", f"Worker {idx} PID={proc.pid}")

    save_pids(pids)

    # ─────────────── Wait for all workers to be ready ───────────────
    print()
    cprint("info", f"Waiting for {len(gpu_ids)} worker(s) to be ready "
                   f"(timeout: {args.worker_timeout}s)...")
    for idx, addr in enumerate(worker_addresses):
        port = worker_base_port + idx
        tag  = f"worker{idx}"
        cprint("info", f"Waiting for {tag} (:{port})...")

        proc = processes[tag]
        ok   = wait_for_worker(port, timeout=args.worker_timeout, tag=tag)

        if proc.poll() is not None:
            cprint("err", f"{tag} exited early (exit code {proc.returncode}). Check logs above.")
            _cleanup(processes)
            sys.exit(1)
        if not ok:
            cprint("err", f"{tag} did not become ready within {args.worker_timeout}s. Check logs above.")
            _cleanup(processes)
            sys.exit(1)
        cprint("ok", f"{tag} is ready")

    # ─────────────── Start gateway ───────────────
    print()
    cprint("info", "Starting Gateway...")
    gateway_cmd = [
        python_exe, "-u",
        str(ROOT / "gateway.py"),
        "--port", str(gateway_port),
        "--workers", ",".join(worker_addresses),
    ]
    if args.http:
        gateway_cmd.append("--http")

    gw_proc = subprocess.Popen(
        gateway_cmd,
        cwd=str(ROOT),
        env=base_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    processes["gateway"] = gw_proc
    pids["gateway"] = gw_proc.pid
    save_pids(pids)
    stream_output(gw_proc, "gateway")
    cprint("ok", f"Gateway PID={gw_proc.pid}")

    # ─────────────── Print access URLs ───────────────
    print()
    proto = "http" if args.http else "https"
    cprint("ok", "=" * 55)
    cprint("ok", f"  Services started! Open: {proto}://localhost:{gateway_port}")
    cprint("ok", f"  Turn-based Chat : {proto}://localhost:{gateway_port}/")
    cprint("ok", f"  Half-Duplex     : {proto}://localhost:{gateway_port}/half_duplex")
    cprint("ok", f"  Omni Full-Duplex: {proto}://localhost:{gateway_port}/omni")
    cprint("ok", f"  Audio Full-Duplex:{proto}://localhost:{gateway_port}/audio_duplex")
    cprint("ok", f"  Dashboard       : {proto}://localhost:{gateway_port}/admin")
    cprint("ok", f"  API Docs        : {proto}://localhost:{gateway_port}/docs")
    if not args.http:
        cprint("warn", "  Your browser will warn about an untrusted certificate.")
        cprint("warn", "  Click 'Advanced' -> 'Proceed' to continue.")
    cprint("ok", "=" * 55)
    cprint("info", "Press Ctrl+C to stop all services.")
    print()

    # ─────────────── Monitor: stop everything if any process exits ───────────────
    def _handle_signal(signum, frame):
        cprint("warn", "Termination signal received. Stopping all services...")
        _cleanup(processes)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    try:
        while True:
            for name, proc in list(processes.items()):
                rc = proc.poll()
                if rc is not None:
                    cprint("err", f"{name} exited unexpectedly (exit code {rc}). "
                                  "Stopping all services...")
                    _cleanup(processes)
                    sys.exit(1)
            time.sleep(3)
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup(processes)


def _cleanup(processes: dict):
    """Gracefully terminate all child processes."""
    for name, proc in processes.items():
        if proc.poll() is None:
            cprint("info", f"Stopping {name} (PID {proc.pid})...")
            try:
                proc.terminate()
            except Exception:
                pass
    # Wait up to 10 seconds for clean shutdown
    deadline = time.time() + 10
    for name, proc in processes.items():
        remaining = max(0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            cprint("warn", f"{name} did not respond to SIGTERM, sending SIGKILL...")
            try:
                proc.kill()
            except Exception:
                pass
    PID_FILE.unlink(missing_ok=True)
    cprint("ok", "All services stopped.")


if __name__ == "__main__":
    main()
