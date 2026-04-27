# C++ Backend Deployment

This guide walks through running the demo on top of the
[`llama.cpp-omni`](https://github.com/tc-mb/llama.cpp-omni) C++ inference engine
instead of the PyTorch backend.

Use the C++ backend when you need:

- Lower VRAM (Q4_K_M needs roughly 9 GB instead of ~22 GB)
- Lower TTFT and faster decode on the same hardware
- A self-contained `llama-server` process that you can also call directly via HTTP

The Python service in this repository (`gateway.py` + `worker.py` + the static
frontend) stays the same — only the inference backend swaps out.

---

## 1. Components You Need

| Piece | Where it comes from | Notes |
|---|---|---|
| This demo (Python service + frontend) | The current branch you are reading | gateway / worker / static pages / mobile React app |
| `llama.cpp-omni` (C++ engine) | [`tc-mb/llama.cpp-omni`](https://github.com/tc-mb/llama.cpp-omni), branch `feat/web-demo` | We compile `llama-server` from this; `worker.py` will spawn it as a subprocess |
| MiniCPM-o-4_5 GGUF weights | See the [llama.cpp-omni README](https://github.com/tc-mb/llama.cpp-omni/tree/feat/web-demo#prerequisites) | Same weights for both PyTorch and C++ paths; downloaded separately |

---

## 2. Build `llama-server`

```bash
git clone https://github.com/tc-mb/llama.cpp-omni.git
cd llama.cpp-omni
git checkout feat/web-demo

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

CMake auto-detects CUDA (Linux + NVIDIA) and Metal (macOS). After the build
finishes, `build/bin/llama-server` is the binary that `worker.py` will launch.

You do **not** need to start `llama-server` yourself — the Python worker spawns
it on demand.

---

## 3. Install Python Dependencies

Same as the PyTorch path — see the standard
[deployment guide](deployment.md) for `install.sh` / venv setup.

The C++ backend does not require PyTorch CUDA at runtime, but the worker
itself is still a Python process, so the venv is needed.

---

## 4. Configure `config.json`

Set `backend` to `cpp` and point `cpp_backend.llamacpp_root` and `model_dir` at
your local checkouts:

```json
{
    "backend": "cpp",

    "cpp_backend": {
        "llamacpp_root": "/abs/path/to/llama.cpp-omni",
        "model_dir":     "/abs/path/to/MiniCPM-o-4_5-gguf",
        "llm_model":     "MiniCPM-o-4_5-Q4_K_M.gguf",
        "cpp_server_port": 19080,
        "ctx_size": 8192,
        "n_gpu_layers": 99
    },

    "audio": {
        "ref_audio_path": "assets/ref_audio/ref_minicpm_signature.wav",
        "playback_delay_ms": 200
    },

    "service": {
        "gateway_port": 8040,
        "worker_base_port": 22440,
        "num_workers": 1,
        "max_queue_size": 1000,
        "request_timeout": 300.0,
        "data_dir": "data"
    },

    "duplex": {
        "pause_timeout": 60.0
    }
}
```

| Field | What it controls |
|---|---|
| `cpp_backend.llamacpp_root` | Absolute path of the `llama.cpp-omni` checkout. `worker.py` runs `llama-server` from `${llamacpp_root}/build/bin/llama-server` and uses `${llamacpp_root}/tools/omni/output_<port>` as TTS WAV output. |
| `cpp_backend.model_dir` | Absolute path to the GGUF directory (LLM, TTS, vision, audio, token2wav-gguf). |
| `cpp_backend.llm_model` | Filename inside `model_dir`. Use the quantization you downloaded (`MiniCPM-o-4_5-Q4_K_M.gguf`, `-Q8_0.gguf`, or `-F16.gguf`). |
| `cpp_backend.cpp_server_port` | HTTP port `worker.py` will start `llama-server` on. Each worker needs its own port if you scale to multiple GPUs. |
| `cpp_backend.ctx_size` / `n_gpu_layers` | Forwarded to `llama-server` flags `--ctx-size` and `--n-gpu-layers`. |

---

## 5. Start the Stack

```bash
CUDA_VISIBLE_DEVICES=0 bash start_all.sh
```

What that produces:

```
gateway.py        :8040 (HTTPS)        ─┐
                                        │  HTTP / WS  (internal)
worker.py         :22440  GPU 0        ─┘
    │
    │  spawns + HTTP-calls
    ▼
llama-server      :19080  GPU 0
    /v1/stream/omni_init
    /v1/stream/update_session_config
    /v1/stream/prefill
    /v1/stream/decode  (SSE)
    /v1/stream/break
```

The first `llama-server` boot loads the GGUF modules (VPM, APM, LLM, TTS,
Token2Wav) and takes 10–60 s. The worker’s `/health` endpoint will start
returning `worker_status: "idle"` once `omni_init` finishes.

After that, open:

- `https://localhost:8040/` — desktop entry (Home / Omni / Audio-Duplex / Turnbased / Half-Duplex)
- `https://localhost:8040/mobile/` — mobile React frontend
- `https://localhost:8040/mobile-omni/` — mobile-adapted Omni page (DOM bridge over the desktop omni-app.js)

> Camera and microphone require HTTPS. The shipped self-signed certs in
> `certs/` work locally — accept the browser warning. Falling back to HTTP
> (`bash start_all.sh --http`) will only let text input through.

---

## 6. Mobile Frontend Build

The `/mobile/` route is served from `static/mobile/`, which is **gitignored**.
It is the build output of the React + Vite project under `frontend/mobile/`.
Build it once after you clone:

```bash
cd frontend/mobile
bun install                 # or `npm install`, requires Node ≥ 20.19
bun run --bun build:static  # publishes to ../../static/mobile/
```

See [`frontend/mobile/README.md`](../../frontend/mobile/README.md) for dev
proxy / npm-only / hot-reload details.

---

## 7. Stop

```bash
pkill -f "gateway.py|worker.py|llama-server"
```

`worker.py` already restarts `llama-server` after each session
(`full_reinit`) to keep KV cache state clean across runs.

---

## Multi-GPU

To run multiple workers on the same box:

```json
"service": {
    "gateway_port": 8040,
    "worker_base_port": 22440,
    "num_workers": 2,
    ...
},
"cpp_backend": {
    "cpp_server_port": 19080,
    ...
}
```

Each worker gets its own GPU (`CUDA_VISIBLE_DEVICES=0,1 bash start_all.sh`)
and spawns its own `llama-server` on `cpp_server_port + worker_index`.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `llama-server not found` in worker log | `cpp_backend.llamacpp_root` is wrong, or `cmake --build … --target llama-server` was not run. |
| Worker `/health` says `worker_status: "loading"` for a long time | `omni_init` is still loading GGUF modules. Check `tmp/worker_<i>.log` for the C++ side: lines tagged `[CPP]`. |
| WAV files appear under `${llamacpp_root}/tools/omni/output_<port>/round_XXX/` but the browser plays nothing | Check that the gateway is HTTPS — many browsers block `Audio` / `MediaDevices` on insecure origins. |
| `kv_cache_length` keeps shrinking mid-conversation | This is the C++ side sliding-window pruning kicking in. The desktop and mobile UIs expose a "Stop on KV pruning" toggle (default on) that ends the session cleanly when this happens. |
