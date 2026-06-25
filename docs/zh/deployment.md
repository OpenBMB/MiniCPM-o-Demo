# 配置与部署

## 系统要求

| 要求 | 最低配置 |
|------|---------|
| GPU | NVIDIA GPU，显存 > 28GB |
| 操作系统 | Linux |
| Python | 3.10 |
| CUDA | 与 PyTorch 2.8.0 兼容 |
| FFmpeg | 用于视频帧提取和推理结果可视化 |

### 资源消耗参考

| 资源 | Token2Wav（默认） |
|------|-------------------|
| 显存（每 Worker，初始化后） | ~21.5 GB |
| 模型加载时间 | ~16s |
| 模式切换延迟 | < 0.1ms |

> torch.compile 模式首次推理额外 ~60s 编译耗时。

---

## 依赖安装

### 使用 install.sh（推荐）

```bash
# 1. 安装 Python 3.10（推荐 miniconda）
mkdir -p ./miniconda3_install_tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_25.11.1-1-Linux-x86_64.sh \
    -O ./miniconda3_install_tmp/miniconda.sh
bash ./miniconda3_install_tmp/miniconda.sh -b -u -p ./miniconda3
source ./miniconda3/bin/activate

# 2. 一键安装
bash ./install.sh
```

`install.sh` 自动完成以下步骤：
1. 在 `.venv/base` 创建 Python venv 虚拟环境
2. 安装 PyTorch 2.8.0 + torchaudio
3. 安装 `requirements.txt` 中的所有依赖
4. 验证安装结果

### 手动安装

```bash
source ./miniconda3/bin/activate
python -m venv .venv/base
source .venv/base/bin/activate

pip install "torch==2.8.0" "torchaudio==2.8.0"
pip install -r requirements.txt
```

### Python 依赖清单

| 类别 | 包 | 版本 |
|------|-----|------|
| **核心 ML** | transformers | 4.51.0 |
| | accelerate | 1.12.0 |
| | safetensors | >= 0.7.0 |
| **MiniCPM-o** | minicpmo-utils[all] | >= 1.0.5 |
| **Web 服务** | fastapi | >= 0.128.0 |
| | uvicorn | >= 0.40.0 |
| | httpx | >= 0.28.0 |
| | websockets | >= 16.0 |
| | python-multipart | — |
| **数据** | pydantic | >= 2.11.0 |
| | numpy | >= 2.2.0 |
| **工具** | tqdm | >= 4.67.0 |
| **测试** | pytest | >= 9.0.0 |
| | pytest-asyncio | >= 1.3.0 |

---

## 配置说明

### config.json

所有配置集中在项目根目录的 `config.json` 文件中。首次使用时从 `config.example.json` 复制：

```bash
cp config.example.json config.json
```

`config.json` 已在 `.gitignore` 中，不会被提交。

### 配置优先级

```
CLI 参数 > config.json > Pydantic 默认值
```

### 完整字段说明

#### model — 模型配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | str | _(必填)_ | HuggingFace 格式模型目录或 Hub ID |
| `pt_path` | str | null | 额外 .pt 权重覆盖路径 |
| `attn_implementation` | str | `"auto"` | Attention 实现方式 |

#### audio — 音频配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ref_audio_path` | str | `assets/ref_audio/ref_minicpm_signature.wav` | 默认 TTS 参考音频路径 |
| `playback_delay_ms` | int | 200 | 前端音频播放延迟（ms），越大越平滑但延迟越高 |

#### service — 服务配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gateway_port` | int | 8006 | Gateway 监听端口 |
| `worker_base_port` | int | 22400 | Worker 起始端口（Worker N = base + N） |
| `max_queue_size` | int | 1000 | 最大排队请求数 |
| `request_timeout` | float | 300.0 | 请求超时（秒） |
| `compile` | bool | false | 启用 torch.compile 加速 |
| `data_dir` | str | `"data"` | 数据存储目录 |
| `eta_chat_s` | float | 15.0 | Chat 任务基准 ETA（秒） |
| `eta_streaming_s` | float | 20.0 | Streaming 任务基准 ETA（秒） |
| `eta_audio_duplex_s` | float | 120.0 | Audio Duplex 任务基准 ETA（秒） |
| `eta_omni_duplex_s` | float | 90.0 | Omni Duplex 任务基准 ETA（秒） |
| `eta_ema_alpha` | float | 0.3 | ETA EMA 平滑系数 |
| `eta_ema_min_samples` | int | 3 | ETA EMA 最少样本数 |

#### duplex — 双工配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pause_timeout` | float | 60.0 | Duplex 暂停超时（秒），超时后自动释放 Worker |

### 最小配置

```json
{
  "model": {
    "model_path": "openbmb/MiniCPM-o-4_5"
  }
}
```

### 完整配置示例

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

控制模型推理使用的 Attention 实现，通过 `attn_implementation` 字段配置。

| 值 | 行为 | 适用场景 |
|----|------|---------|
| `"auto"`（默认） | 检测到 flash-attn → `flash_attention_2`；否则 → `sdpa` | 推荐 |
| `"flash_attention_2"` | 强制使用 Flash Attention 2 | 确认已安装 flash-attn |
| `"sdpa"` | PyTorch 内置 SDPA | 无法编译 flash-attn |
| `"eager"` | 朴素 Attention | 仅 debug |

**性能对比**（A100）：`flash_attention_2` 比 `sdpa` 快约 5-15%，`sdpa` 比 `eager` 快数倍。

**注意**：Audio（Whisper）子模块始终使用 SDPA（与 flash_attention_2 不兼容）。Vision / LLM / TTS 遵循配置。

---

## 启动与停止

### 一键启动（start_all.sh）

```bash
# 使用所有可用 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 bash start_all.sh

# 指定 GPU
CUDA_VISIBLE_DEVICES=0,1 bash start_all.sh

# 启用 torch.compile（实验性）
bash start_all.sh --compile

# 降级为 HTTP（不推荐，麦克风/摄像头 API 需要 HTTPS）
bash start_all.sh --http
```

`start_all.sh` 执行流程：
1. 解析命令行参数（`--http`, `--compile`）
2. 从 `config.py` 读取端口配置
3. 检测可用 GPU 数量
4. 为每个 GPU 启动一个 Worker 进程（`nohup`）
5. 等待所有 Worker 健康检查通过
6. 启动 Gateway 进程
7. 输出访问地址和日志路径

### 手动启动

```bash
# Worker（每张 GPU 一个）
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/base/bin/python worker.py \
    --worker-index 0 --gpu-id 0

# Gateway
PYTHONPATH=. .venv/base/bin/python gateway.py \
    --port 8006 --internal-port 8007

curl -X PUT http://127.0.0.1:8007/internal/workers/local-worker \
    -H 'content-type: application/json' \
    --data '{"endpoint":"127.0.0.1:22400","gpu_group":"gpu-0"}'
```

### CLI 参数

**Worker 参数**：
```bash
python worker.py \
    --model-path /path/to/model \
    --pt-path /path/to/weights.pt \
    --ref-audio-path /path/to/ref.wav \
    --worker-index 0 \
    --gpu-id 0 \
    --compile
```

**Gateway 参数**：
```bash
python gateway.py \
    --port 8006 \
    --internal-port 8007 \
    --http
```

### 停止服务

```bash
pkill -f "gateway.py|worker.py"
```

---

## 模型下载

### 自动下载（默认）

`model_path` 设为 `openbmb/MiniCPM-o-4_5` 时，首次启动自动从 HuggingFace 下载。

### 手动下载

**HuggingFace CLI**：
```bash
pip install -U huggingface_hub
huggingface-cli download openbmb/MiniCPM-o-4_5 --local-dir /path/to/MiniCPM-o-4_5
```

**hf-mirror（中国）**：
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download openbmb/MiniCPM-o-4_5 --local-dir /path/to/MiniCPM-o-4_5
```

**ModelScope（中国）**：
```bash
pip install modelscope
modelscope download --model OpenBMB/MiniCPM-o-4_5 --local_dir /path/to/MiniCPM-o-4_5
```

---

## 测试

### Schema 单元测试（无需 GPU）

```bash
PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_schemas.py -v
```

### Processor 测试（需要 GPU）

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/base/bin/python -m pytest \
    tests/test_chat.py tests/test_streaming.py tests/test_duplex.py -v -s
```

### API 集成测试（需要先启动服务）

```bash
PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_api.py -v -s
```

### 测试文件说明

| 文件 | 说明 |
|------|------|
| `test_schemas.py` | Schema 单元测试 |
| `test_chat.py` | Chat 推理测试 |
| `test_streaming.py` | Streaming 推理测试 |
| `test_duplex.py` | Duplex 推理测试 |
| `test_api.py` | API 集成测试 |
| `test_queue.py` | 队列逻辑测试 |
| `test_queue_stress.py` | 队列压力测试 |
| `test_integration.py` | 集成测试 |
| `test_e2e.py` | 端到端测试 |
| `bench_duplex_ws.py` | Duplex WebSocket 性能基准测试 |
| `mock_worker.py` | Mock Worker（用于无 GPU 测试） |
| `js/queue-scenario.test.js` | 前端队列场景测试（Vitest） |
| `js/countdown-timer.test.js` | 倒计时组件测试（Vitest） |

---

## 运行时目录结构

```
data/
├── sessions/                  # 会话录制数据
│   ├── omni_abc123/
│   │   ├── meta.json
│   │   ├── recording.json
│   │   ├── user_audio/
│   │   ├── ai_audio/
│   │   └── ...
│   └── ...
└── ref_audio/                 # 上传的参考音频
    ├── registry.json
    └── *.wav

tmp/
├── gateway.pid                # Gateway 进程 PID
├── gateway.log                # Gateway 日志
├── worker_0.pid               # Worker 0 PID
├── worker_0.log               # Worker 0 日志
└── diag_omni_*.jsonl          # 诊断日志
```
