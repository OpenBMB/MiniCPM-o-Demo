# MiniCPM-o 4.5 PyTorch 简易演示系统

[English Documentation](README.md) | [详细文档](https://minicpmo45.modelbest.cn/docs/zh/) | [Realtime API 文档](https://minicpmo45.modelbest.cn/docs/zh/realtime-api/overview/)

[可直接使用的在线演示系统](https://minicpmo45.modelbest.cn/) | [Discord](https://discord.gg/UTbTeCQe) | [飞书群](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=228m5ca0-dfa1-464c-9406-b8b2f86d76ea)

本演示系统为 `MiniCPM-o 4.5` 模型训练团队官方提供的演示系统。本演示系统使用 PyTorch + CUDA 推理后端，结合简易的前后端设计，旨在以透明、简洁、无性能损失的方式，全面地演示 MiniCPM-o 4.5 的音视频全模态全双工能力。

## 关于 MiniCPM-o 4.5

MiniCPM-o 4.5 是 MiniCPM-o 系列中最新、能力最强的模型。该模型基于 SigLip2、Whisper-medium、CosyVoice2 和 Qwen3-8B，以端到端方式构建，总参数量为 9B。它在性能上取得了显著提升，并引入了全双工多模态实时流式交互的新特性。MiniCPM-o 4.5 的主要亮点包括：

- 🔥 **领先的视觉能力。** MiniCPM-o 4.5 在 OpenCompass 上取得了 77.6 的平均分，该评测涵盖 8 个主流基准测试。仅凭 9B 参数，它超越了 GPT-4o、Gemini 2.0 Pro 等广泛使用的商业模型，并接近 Gemini 2.5 Flash 的视觉语言能力。它在单一模型中同时支持 instruct 和 thinking 模式，更好地覆盖了不同用户场景下的效率与性能权衡。

- 🎙 **强大的语音能力。** MiniCPM-o 4.5 支持中英双语实时语音对话，并可配置不同的音色。它具有更自然、更富表现力且更稳定的语音对话效果。模型还支持通过简单的参考音频片段实现声音克隆和角色扮演等趣味功能，克隆性能超越了 CosyVoice2 等强大的 TTS 工具。

- 🎬 **全新的全双工主动式多模态实时流式交互能力。** 作为新特性，MiniCPM-o 4.5 可以同时处理实时、连续的视频和音频输入流，同时以端到端方式生成并发的文本和语音输出流，互不阻塞。这使得 MiniCPM-o 4.5 能够同时看、听、说，创造流畅的实时全模态对话体验。除了被动响应外，模型还能进行主动交互，例如基于对实时场景的持续理解主动发起提醒或评论。

- 💪 **强大的 OCR 能力、高效率及其他。** 延续 MiniCPM-V 系列的优势视觉能力，MiniCPM-o 4.5 可以高效处理高分辨率图像（最高 180 万像素）和高帧率视频（最高 10fps），支持任意宽高比。它在 OmniDocBench 端到端英文文档解析中达到了最先进的性能，超越了 Gemini-3 Flash 和 GPT-5 等商业模型，以及 DeepSeek-OCR 2 等专业工具。它还具有可信赖的行为，在 MMHal-Bench 上匹配 Gemini 2.5 Flash，并支持超过 30 种语言的多语言能力。

- 💫 **易于使用。** MiniCPM-o 4.5 可以通过多种方式轻松使用：基本用法推荐以 100% 精度使用 PyTorch + Nvidia GPU 推理。其他端侧适配包括：(1) llama.cpp 和 Ollama 支持在本地设备上进行高效 CPU 推理；(2) 提供 16 种尺寸的 int4 和 GGUF 格式量化模型；(3) vLLM 和 SGLang 支持高吞吐和内存高效推理；(4) FlagOS 支持统一多芯片后端插件。我们还开源了 Web 演示系统，可在 GPU、PC（如 MacBook）等本地设备上体验全双工多模态实时流式交互。

<details>
<summary><b>模型架构</b></summary>

- **端到端全模态架构。** 模态编码器/解码器与 LLM 通过隐藏状态以端到端方式密集连接。这实现了更好的信息流动和控制，也有助于在训练过程中充分利用丰富的多模态知识。

- **全双工全模态实时流式机制。** (1) 我们将离线的模态编码器/解码器转变为在线和全双工模式，用于流式输入/输出。语音 token 解码器以交错方式建模文本和语音 token，以支持全双工语音生成（即与新输入及时同步）。这也有助于更稳定的长语音生成（例如 > 1 分钟）。(2) 我们以毫秒为单位在时间线上同步所有输入和输出流，通过时分复用（TDM）机制在 LLM 骨干网络中进行全模态流式处理。它将并行的全模态流在小的周期性时间片内划分为顺序信息组。

- **主动交互机制。** LLM 持续监控输入的视频和音频流，并以 1Hz 的频率决定是否说话。这种高频率的决策机制结合全双工特性，是实现主动交互能力的关键。

- **可配置的语音建模设计。** 我们继承了 MiniCPM-o 2.6 的多模态系统提示词设计，包括传统的文本系统提示词和新的音频系统提示词来确定助手音色。这使得在推理时可以克隆新音色并进行语音对话中的角色扮演。

</details>

---

| 模式 | 特点 | 输入输出模态 | 范式
|------|------|------|------
| **Turn-based Chat (轮次对话)** | 低延迟流式交互，按钮触发回复，支持离线视频、音频理解分析，回复正确性好，基础能力强 | 音频+文本+视频输入，音频+文本输出 | 轮次对话范式
| **Half-Duplex Audio (半双工语音)** | VAD 自动检测语音边界，无需按钮即可进行语音通话，语音生成质量更高，回复准确性强，用户获得感好 | 语音输入，文本+语音输出 | 半双工范式
| **Omnimodal Full-Duplex (全模态全双工)** | 全模态全双工实时交互，视觉语音输入、语音输出同时发生，模型完全自主决定说话时机，前沿能力强大 | 视觉+语音输入，文本+语音输出 | 全双工范式
| **Audio Full-Duplex (语音全双工)** | 语音全双工实时交互，语音输入和语音输出同时发生，模型完全自主决定说话时机，前沿能力强大 | 语音输入，文本+语音输出 | 全双工范式

目前支持的 4 种模式共享同一个模型实例，支持毫秒级热切换（< 0.1ms）。

**其他特性：**

- 可自定义系统提示词
- 可自定义参考音频
- 代码简洁易读，便于二次开发
- 可作为 API 后端供第三方应用调用


![Demo Preview](assets/images/demo_preview.png)


## 架构

```
Frontend (HTML/JS)
    |  HTTPS / WSS
Gateway (:8006, HTTPS)
    |  HTTP / WS (internal)
Worker Pool (:22400+)
    +-- Worker 0 (GPU 0)
    +-- Worker 1 (GPU 1)
    +-- ...
```

- **Frontend** — 模式选择首页、Turn-based Chat 轮次对话、Omni / Audio Duplex 全双工交互、Admin Dashboard 监控面板
- **Gateway** — 请求路由与分发、WebSocket 代理、请求排队与会话亲和
- **Worker** — 每 Worker 独占一张 GPU，支持 Turn-based Chat / Duplex 协议，Duplex 支持暂停/恢复（超时自动释放）



## 快速开始

### 检查系统要求
1. 确保你有一张显存大于 28GB 的 NVIDIA GPU。
2. 确保你的机器安装了 Linux 操作系统。

### 安装 FFmpeg

FFmpeg 用于视频帧提取 和 推理结果可视化。更多信息请访问 [FFmpeg 官网](https://ffmpeg.org/)。

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**验证安装:**
```bash
ffmpeg -version
```

### 部署步骤
**1. 安装Python 3.10**

推荐使用 miniconda 安装 Python 3.10。

```bash
mkdir -p ./miniconda3_install_tmp

# 下载 miniconda3 安装脚本
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_25.11.1-1-Linux-x86_64.sh -O ./miniconda3_install_tmp/miniconda.sh 

# 将 miniconda3 安装到项目目录下
bash ./miniconda3_install_tmp/miniconda.sh -b -u -p ./miniconda3 
```

安装完成后，会得到一个空的 base 环境，激活这个 base 环境，base 环境中默认为 Python 3.10。

```bash
source ./miniconda3/bin/activate
python --version # 应显示 3.10.x
```

**2. 安装 MiniCPM-o 4.5 所需的依赖**

使用项目目录下的 `install.sh` 安装依赖是最快的，它会在项目目录下的 .venv 中创建一个名为 `base` 的venv虚拟环境，并在其中安装所有的依赖。

```bash
source ./miniconda3/bin/activate
bash ./install.sh
```

如果网络良好，整个安装过程大约花费 5 分钟。如果你处在中国，可以考虑使用第三方 PyPi 镜像源，例如清华镜像源。

<details>
<summary>点击展开手动安装步骤</summary>

您也可以手动安装依赖，分 2 步：

```bash
# 首先准备好一个空的 python 3.10 环境
source ./miniconda3/bin/activate
python -m venv .venv/base
source .venv/base/bin/activate

# 安装 PyTorch。
pip install "torch==2.8.0" "torchaudio==2.8.0"

# 安装其余依赖。
pip install -r requirements.txt
```

</details>

**3. 创建配置文件**

将项目目录下的 `config.example.json` 复制为 `config.json`。

```bash
cp config.example.json config.json
```

模型路径（`model_path`），默认使用 `openbmb/MiniCPM-o-4_5`，如果你可以访问 huggingface，无需修改，将会自动从 huggingface 拉取模型。

<details>
<summary>点击展开关于模型路径的详细说明</summary>

(可选) 如果你习惯于下载模型权重到固定位置，或无法访问 huggingface，可以修改 model_path 为你的模型路径。
```bash
# 安装huggingface cli
pip install -U huggingface_hub

# 下载模型
huggingface-cli download openbmb/MiniCPM-o-4_5 --local-dir /path/to/your/MiniCPM-o-4_5

```

如果无法访问 huggingface，可以使用以下两种方式提前下载模型。

- 使用 hf-mirror 提前下载模型

```bash
pip install -U huggingface_hub

export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download openbmb/MiniCPM-o-4_5 --local-dir /path/to/your/MiniCPM-o-4_5
```

- 使用 modelscope 提前下载模型

```bash
pip install modelscope

modelscope download --model OpenBMB/MiniCPM-o-4_5 --local_dir /path/to/your/MiniCPM-o-4_5
```


</details>

<br/>

修改 `"gateway_port": 8006` 即可改变部署的端口，默认为 8006。


**4. 部署架构**

当前部署拆成三个运行角色：

```text
Browser -> Gateway -> Python Worker -> Backend
```

- **Gateway** 是对外的 HTTPS/WebSocket 入口，不加载模型，负责路由、排队、session 录制和 worker 健康检查。
- **Python Worker** 暴露 worker WebSocket/health API，维护 worker 状态，并把 runtime protocol 消息转发给 backend server。
- **Backend** 负责实际模型推理。Backend 可以是 PyTorch 实现（`py_backend/server.py`），也可以是 C++ 实现（`llama.cpp-omni` 的 `llama-omni-server`）。

**5. Docker 部署（推荐）**

Docker 是当前仓库的部署权威来源。Dockerfile 和 entrypoint 定义了受支持的进程拓扑、依赖版本、端口、健康检查、模型挂载和 backend 启动参数。裸机部署只建议作为高级调试方式，并应与 Dockerfile / entrypoint 保持等价。

**前置条件：**
- Docker 和 Compose v2 插件
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- 每个 worker-backend 实例独占一张 NVIDIA GPU
- 模型权重从宿主机挂载，镜像内不包含模型权重

**PyTorch backend（Compose）：**

```bash
mkdir -p certs data
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout certs/key.pem -out certs/cert.pem -subj "/CN=minicpm-o"

MODEL_HOST_PATH=/path/to/MiniCPM-o-4_5 docker compose up -d --build
docker compose logs -f gateway
docker compose logs -f worker-backend-0
```

`docker-compose.yml` 默认启动一个 gateway 和两个绑定 GPU 0 / GPU 1 的 `worker-backend` 容器。请按机器 GPU 数量显式增删 worker service，并同步修改 gateway 的 `--workers` 列表。如果确实需要单张 GPU 跑多个 worker 实例，可以参考 `docker-compose.multi.yml`。

**C++ backend（Compose）：**

```bash
mkdir -p certs data
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout certs/key.pem -out certs/cert.pem -subj "/CN=minicpm-o"

GGUF_MODEL_HOST_PATH=/path/to/MiniCPM-o-4_5-gguf \
GATEWAY_HOST_PORT=8006 \
CPP_GPU_ID=0 \
docker compose -f docker-compose.cpp.yml up -d --build

docker compose -f docker-compose.cpp.yml logs -f gateway
docker compose -f docker-compose.cpp.yml logs -f cpp-worker-backend
```

`docker-compose.cpp.yml` 默认启动一个 gateway 和一个 `cpp-worker-backend` 容器。C++ worker-backend 容器内部同时运行 `llama-omni-server` 和 `worker.py`；gateway 通过 Compose 网络访问 worker。C++ 镜像默认构建 `tc-mb/llama.cpp-omni` 的 `master` 分支；如果需要固定上游版本，可以在构建时传入 `LLAMA_OMNI_REFSPEC=<branch-or-ref>` 和 `LLAMA_OMNI_REF=<ref-or-commit>`。compose 文件默认通过 `LLAMA_SERVER_EXTRA_ARGS` 给 `llama-omni-server` 传入 `-c 8192`，用于显式设置 backend context size；只有在确认目标 context 配置时才建议覆盖 `LLAMA_SERVER_EXTRA_ARGS`。

**裸机部署：**

裸机命令不是当前主安装路径，因为不同机器的 CUDA、Python、编译器和模型目录差异很大。如果需要裸机调试，请以 Dockerfile 和 entrypoint 作为参考实现：

- `docker/Dockerfile.gateway`
- `docker/Dockerfile.worker-backend`
- `docker/entrypoint-worker-backend.sh`
- `docker/Dockerfile.cpp-worker-backend`
- `docker/entrypoint-cpp-worker-backend.sh`

请保持同样的运行拓扑：Gateway -> Python Worker -> Backend。C++ backend 的 backend 进程是 `llama-omni-server`；PyTorch backend 的 backend 进程是 `py_backend/server.py`。

**停止 Docker 服务：**

```bash
docker compose down                      # PyTorch backend compose
docker compose -f docker-compose.cpp.yml down  # C++ backend compose
```

<br/>
<br/>


## C++ 后端（llama.cpp）

本 Demo 同时支持基于 llama.cpp-omni 的 **C++ 推理后端**，可以通过 `llama-omni-server` 运行 MiniCPM-o 4.5。请以上方 Docker 部署章节作为权威安装路径；该路径会构建 `tc-mb/llama.cpp-omni` 的 `master` 分支，并按 Gateway -> Python Worker -> C++ Backend 拓扑接入匹配的 realtime backend protocol。

### 桌面端应用（Windows & macOS）

提供 Windows 和 macOS 的开箱即用安装包，前往 [llama.cpp-omni Releases](https://github.com/tc-mb/llama.cpp-omni/releases/) 下载。

---

## 已知问题和改进计划

- 轮次对话模式下，图片输入暂时不可用，仅支持音频和文本输入，近期会拆分出图片问答模式。
- 半双工的语音通话（无需按钮触发回复）正在开发中，近期合入。
- 语音全双工模式下，回声消除目前存在问题，影响到打断成功率，推荐使用耳机进行交互，近期将修复。
- 语音模式下，由于模型的训练策略，中文和英文通话下，需要使用对应语言的系统提示词。

<br/>

## 项目结构

**项目代码结构**
```
minicpmo45_service/
├── config.json               # 服务配置（从 config.example.json 复制，gitignored）
├── config.example.json       # 配置示例（完整字段 + 默认值）
├── config.py                 # 配置加载逻辑（Pydantic 定义 + JSON 加载）
├── requirements.txt          # Python 依赖
├── docker-compose.yml        # 推荐的 PyTorch backend 部署
├── docker-compose.cpp.yml    # 推荐的 C++ backend 部署
├── docker-compose.multi.yml  # 单卡多 worker 部署变体
├── docker/                   # Dockerfile 和容器 entrypoint
│
├── gateway.py                # Gateway（路由、排队、WS 代理）
├── worker.py                 # Worker（runtime protocol 转发层）
├── gateway_modules/          # Gateway 业务模块
├── py_backend/               # PyTorch backend server
├── runtime/                  # Backend protocol client/session 层
│
├── core/                     # 核心封装
│   ├── schemas/              # Pydantic Schema（请求/响应）
│   └── processors/           # 推理处理器（UnifiedProcessor）
│
├── MiniCPMO45/               # 模型核心推理代码
├── static/                   # 前端页面
├── resources/                # 资源文件（参考音频等）
├── tests/                    # 测试
└── tmp/                      # 运行时日志和 PID 文件
```

**前端路由设定**

| 页面 | URL |
|------|-----|
| 轮次对话 | https://localhost:8006 |
| 半双工语音 | https://localhost:8006/half_duplex |
| 全模态全双工 | https://localhost:8006/omni |
| 语音全双工 | https://localhost:8006/audio_duplex |
| 仪表盘 | https://localhost:8006/admin |
| 文档 / Realtime API 文档 | https://localhost:8006/docs |

<br/>
<br/>

## 配置说明

### config.json — 统一配置文件

所有配置集中在 `config.json`（从 `config.example.json` 复制）。
`config.json` 已 gitignore，不会被提交。

**配置优先级**：CLI 参数 > config.json > Pydantic 默认值

| 分组 | 字段 | 默认值 | 说明 |
|------|------|--------|------|
| **model** | `model_path` | _(必填)_ | HuggingFace 格式模型目录 |
| model | `pt_path` | null | 额外 .pt 权重覆盖 |
| model | `attn_implementation` | `"auto"` | Attention 实现：`"auto"`/`"flash_attention_2"`/`"sdpa"`/`"eager"` |
| **audio** | `ref_audio_path` | `assets/ref_audio/ref_minicpm_signature.wav` | 默认 TTS 参考音频 |
| audio | `playback_delay_ms` | 200 | 前端音频播放延迟（ms），越大越平滑但延迟越高 |
| audio | `chat_vocoder` | `"token2wav"` | Chat 模式 vocoder：`"token2wav"`（默认）或 `"cosyvoice2"` |
| **service** | `gateway_port` | 8006 | Gateway 端口 |
| service | `worker_base_port` | 22400 | Worker 起始端口 |
| service | `max_queue_size` | 100 | 最大排队请求数 |
| service | `request_timeout` | 300.0 | 请求超时（秒） |
| service | `compile` | false | torch.compile 加速 |
| service | `data_dir` | "data" | 数据目录 |
| **duplex** | `pause_timeout` | 60.0 | Duplex 暂停超时（秒） |

**最小配置**（只需模型路径）：
```json
{"model": {"model_path": "/path/to/model"}}
```

## CLI 参数覆盖

```bash
# Worker
python worker.py --model-path /alt/model --pt-path /alt/weights.pt --ref-audio-path /alt/ref.wav

# Gateway
python gateway.py --port 10025 --workers localhost:22400,localhost:22401 --http
```


## 资源消耗

| 资源 | Token2Wav（默认） | + torch.compile |
|------|-------------------|-----------------|
| 显存（每 Worker，初始化完成后） | ~21.5 GB | ~21.5 GB |
| 模型加载时间 | ~16s | ~16s + ~5 min（有缓存）/ ~15 min（无缓存）|
| 模式切换延迟 | < 0.1ms | < 0.1ms |
| Omni Full-Duplex 单 unit 延迟（A100） | ~0.9s | **~0.5s** |

## 测试

```bash

# Schema 单元测试（无需 GPU）
PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_schemas.py -v

# Processor 测试（需要 GPU）
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_chat.py tests/test_streaming.py tests/test_duplex.py -v -s

# API 集成测试（需要先启动服务）
PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_api.py -v -s
```
