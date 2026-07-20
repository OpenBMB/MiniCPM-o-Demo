# Configuration & Deployment

## System Requirements

| Requirement | Minimum |
|------|---------|
| GPU | NVIDIA GPU with VRAM > 28GB |
| OS | Linux |
| Python | 3.10 |
| CUDA | Compatible with PyTorch 2.8.0 |
| FFmpeg | For video frame extraction and inference result visualization |

### Resource Consumption Reference

| Resource | Token2Wav (Default) |
|------|-------------------|
| VRAM (per Worker, after initialization) | ~21.5 GB |
| Model loading time | ~16s |
| Mode switching latency | < 0.1ms |

> torch.compile mode incurs an additional ~60s compilation time on first inference.

---

## Dependency Installation

### Using install.sh (Recommended)

```bash
# 1. Install Python 3.10 (miniconda recommended)
mkdir -p ./miniconda3_install_tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_25.11.1-1-Linux-x86_64.sh \
    -O ./miniconda3_install_tmp/miniconda.sh
bash ./miniconda3_install_tmp/miniconda.sh -b -u -p ./miniconda3
source ./miniconda3/bin/activate

# 2. One-command installation
bash ./install.sh
```

`install.sh` automatically performs the following steps:
1. Creates a Python venv virtual environment at `.venv/base`
2. Installs PyTorch 2.8.0 + torchaudio
3. Installs all dependencies from `requirements.txt`
4. Verifies the installation

### Manual Installation

```bash
source ./miniconda3/bin/activate
python -m venv .venv/base
source .venv/base/bin/activate

pip install "torch==2.8.0" "torchaudio==2.8.0"
pip install -r requirements.txt
```

### Python Dependency List

| Category | Package | Version |
|------|-----|------|
| **Core ML** | transformers | 4.51.0 |
| | accelerate | 1.12.0 |
| | safetensors | >= 0.7.0 |
| **MiniCPM-o** | minicpmo-utils[all] | >= 1.0.5 |
| **Web Service** | fastapi | >= 0.128.0 |
| | uvicorn | >= 0.40.0 |
| | httpx | >= 0.28.0 |
| | websockets | >= 16.0 |
| | python-multipart | — |
| **Data** | pydantic | >= 2.11.0 |
| | numpy | >= 2.2.0 |
| **Utilities** | tqdm | >= 4.67.0 |
| **Testing** | pytest | >= 9.0.0 |
| | pytest-asyncio | >= 1.3.0 |

---

## Configuration

### config.json

All configuration is centralized in the `config.json` file at the project root. Copy from `config.example.json` for first-time setup:

```bash
cp config.example.json config.json
```

`config.json` is listed in `.gitignore` and will not be committed.

### Configuration Priority

```
CLI arguments > config.json > Pydantic defaults
```

### Complete Field Reference

#### model — Model Configuration

| Field | Type | Default | Description |
|------|------|--------|------|
| `model_path` | str | _(required)_ | HuggingFace format model directory or Hub ID |
| `pt_path` | str | null | Optional .pt weight override path |
| `attn_implementation` | str | `"auto"` | Attention implementation method |

#### audio — Audio Configuration

| Field | Type | Default | Description |
|------|------|--------|------|
| `ref_audio_path` | str | `assets/ref_audio/ref_minicpm_signature.wav` | Default TTS reference audio path |
| `playback_delay_ms` | int | 200 | Frontend audio playback delay (ms); higher values are smoother but add latency |

#### service — Service Configuration

| Field | Type | Default | Description |
|------|------|--------|------|
| `gateway_port` | int | 8006 | Gateway listening port |
| `worker_base_port` | int | 22400 | Worker base port (Worker N = base + N) |
| `max_queue_size` | int | 1000 | Maximum queued requests |
| `request_timeout` | float | 300.0 | Request timeout (seconds) |
| `compile` | bool | false | Enable torch.compile acceleration |
| `data_dir` | str | `"data"` | Data storage directory |
| `eta_chat_s` | float | 15.0 | Chat task baseline ETA (seconds) |
| `eta_streaming_s` | float | 20.0 | Streaming task baseline ETA (seconds) |
| `eta_audio_duplex_s` | float | 120.0 | Audio Duplex task baseline ETA (seconds) |
| `eta_omni_duplex_s` | float | 90.0 | Omni Duplex task baseline ETA (seconds) |
| `eta_ema_alpha` | float | 0.3 | ETA EMA smoothing coefficient |
| `eta_ema_min_samples` | int | 3 | ETA EMA minimum sample count |

#### duplex — Duplex Configuration

| Field | Type | Default | Description |
|------|------|--------|------|
| `pause_timeout` | float | 60.0 | Duplex pause timeout (seconds); the Worker is automatically released after timeout |

### Minimal Configuration

```json
{
  "model": {
    "model_path": "openbmb/MiniCPM-o-4_5"
  }
}
```

### Full Configuration Example

```json
{
  "model": {
    "model_path": "openbmb/MiniCPM-o-4_5",
    "pt_path": null,
    "attn_implementation": "auto"
  },
  "audio": {
    "ref_audio_path": "assets/ref_audio/ref_minicpm_signature.wav",
    "playback_delay_ms": 200,
    "chat_vocoder": "token2wav"
  },
  "service": {
    "gateway_port": 8006,
    "worker_base_port": 22400,
    "max_queue_size": 1000,
    "request_timeout": 300.0,
    "compile": false,
    "data_dir": "data",
    "eta_chat_s": 15.0,
    "eta_streaming_s": 20.0,
    "eta_audio_duplex_s": 120.0,
    "eta_omni_duplex_s": 90.0,
    "eta_ema_alpha": 0.3,
    "eta_ema_min_samples": 3
  },
  "duplex": {
    "pause_timeout": 60.0
  }
}
```

---

## Attention Backend

Controls the Attention implementation used for model inference, configured via the `attn_implementation` field.

| Value | Behavior | Use Case |
|----|------|---------|
| `"auto"` (default) | Detects flash-attn → `flash_attention_2`; otherwise → `sdpa` | Recommended |
| `"flash_attention_2"` | Forces Flash Attention 2 | When flash-attn is confirmed installed |
| `"sdpa"` | PyTorch built-in SDPA | When flash-attn cannot be compiled |
| `"eager"` | Naive Attention | Debug only |

**Performance Comparison** (A100): `flash_attention_2` is ~5-15% faster than `sdpa`; `sdpa` is several times faster than `eager`.

**Note**: The Audio (Whisper) submodule always uses SDPA (incompatible with flash_attention_2). Vision / LLM / TTS follow the configuration.

---

## Starting & Stopping

### One-Command Start (start_all.sh)

```bash
# Use all available GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash start_all.sh

# Specify GPUs
CUDA_VISIBLE_DEVICES=0,1 bash start_all.sh

# Enable torch.compile (experimental)
bash start_all.sh --compile

# Downgrade to HTTP (not recommended; microphone/camera APIs require HTTPS)
bash start_all.sh --http
```

`start_all.sh` execution flow:
1. Parses command-line arguments (`--http`, `--compile`)
2. Reads port configuration from `config.py`
3. Detects the number of available GPUs
4. Launches one Worker process per GPU (`nohup`)
5. Waits for all Workers to pass health checks
6. Starts the Gateway process
7. Outputs the access URL and log paths

### Manual Start

```bash
# Worker (one per GPU)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/base/bin/python worker.py \
    --worker-index 0 --gpu-id 0

# Gateway
PYTHONPATH=. .venv/base/bin/python gateway.py \
    --port 8006 --internal-port 8007

curl -X PUT http://127.0.0.1:8007/internal/workers/local-worker \
    -H 'content-type: application/json' \
    --data '{"endpoint":"127.0.0.1:22400","gpu_group":"gpu-0"}'
```

### CLI Arguments

**Worker Arguments**:
```bash
python worker.py \
    --model-path /path/to/model \
    --pt-path /path/to/weights.pt \
    --ref-audio-path /path/to/ref.wav \
    --worker-index 0 \
    --gpu-id 0 \
    --compile
```

**Gateway Arguments**:
```bash
python gateway.py \
    --port 8006 \
    --internal-port 8007 \
    --http
```

### Stopping the Service

```bash
pkill -f "gateway.py|worker.py"
```

---

## Model Download

### Automatic Download (Default)

When `model_path` is set to `openbmb/MiniCPM-o-4_5`, the model is automatically downloaded from HuggingFace on first startup.

### Manual Download

**HuggingFace CLI**:
```bash
pip install -U huggingface_hub
huggingface-cli download openbmb/MiniCPM-o-4_5 --local-dir /path/to/MiniCPM-o-4_5
```

**hf-mirror (China)**:
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download openbmb/MiniCPM-o-4_5 --local-dir /path/to/MiniCPM-o-4_5
```

**ModelScope (China)**:
```bash
pip install modelscope
modelscope download --model OpenBMB/MiniCPM-o-4_5 --local_dir /path/to/MiniCPM-o-4_5
```

---

## Testing

### Schema Unit Tests (No GPU Required)

```bash
PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_schemas.py -v
```

### Processor Tests (GPU Required)

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/base/bin/python -m pytest \
    tests/test_chat.py tests/test_streaming.py tests/test_duplex.py -v -s
```

### API Integration Tests (Service Must Be Running)

```bash
PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_api.py -v -s
```

### Test File Reference

| File | Description |
|------|------|
| `test_schemas.py` | Schema unit tests |
| `test_chat.py` | Chat inference tests |
| `test_streaming.py` | Streaming inference tests |
| `test_duplex.py` | Duplex inference tests |
| `test_api.py` | API integration tests |
| `test_queue.py` | Queue logic tests |
| `test_queue_stress.py` | Queue stress tests |
| `test_integration.py` | Integration tests |
| `test_e2e.py` | End-to-end tests |
| `bench_duplex_ws.py` | Duplex WebSocket performance benchmark |
| `mock_worker.py` | Mock Worker (for GPU-free testing) |
| `js/queue-scenario.test.js` | Frontend queue scenario tests (Vitest) |
| `js/countdown-timer.test.js` | Countdown component tests (Vitest) |

---

## Runtime Directory Structure

```
data/
├── sessions/                  # Session recording data
│   ├── omni_abc123/
│   │   ├── meta.json
│   │   ├── recording.json
│   │   ├── user_audio/
│   │   ├── ai_audio/
│   │   └── ...
│   └── ...
└── ref_audio/                 # Uploaded reference audios
    ├── registry.json
    └── *.wav

tmp/
├── gateway.pid                # Gateway process PID
├── gateway.log                # Gateway log
├── worker_0.pid               # Worker 0 PID
├── worker_0.log               # Worker 0 log
└── diag_omni_*.jsonl          # Diagnostic logs
```
