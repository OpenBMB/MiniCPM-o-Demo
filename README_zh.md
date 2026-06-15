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
| **Omnimodal Full-Duplex (全模态全双工)** | 全模态全双工实时交互，视觉语音输入、语音输出同时发生，模型完全自主决定说话时机，前沿能力强大 | 视觉+语音输入，文本+语音输出 | 全双工范式
| **Audio Full-Duplex (语音全双工)** | 语音全双工实时交互，语音输入和语音输出同时发生，模型完全自主决定说话时机，前沿能力强大 | 语音输入，文本+语音输出 | 全双工范式

目前支持的 3 种模式共享同一个模型实例，支持毫秒级热切换（< 0.1ms）。

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

### 部署步骤
快速部署方式是 Docker Compose。裸机部署请参考 Dockerfile 和 entrypoint 来确定依赖与启动方式，并保持 Gateway、Python Worker、Backend 三个启动环节一致。

**部署架构**

当前部署拆成三个运行角色：

```text
Browser -> Gateway -> Python Worker -> Backend
```

- **Gateway** 是对外的 HTTPS/WebSocket 入口，不加载模型，负责路由、排队、session 录制和 worker 健康检查。
- **Python Worker** 暴露 worker WebSocket/health API，维护 worker 状态，并把 runtime protocol 消息转发给 backend server。
- **Backend** 负责实际模型推理。Backend 可以是 PyTorch 实现（`py_backend/server.py`），也可以是 C++ 实现（`llama.cpp-omni` 的 `llama-omni-server`）。

**Docker 部署（推荐）**

Docker Compose 是当前维护的快速部署方式。部署请使用 Compose 文件；具体启动流程、端口、挂载、健康检查和 backend 参数，请直接查看 `docker-compose*.yml`、`docker/Dockerfile.*` 和 `docker/entrypoint-*.sh`。

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

请按机器 GPU 数量修改 `docker-compose.yml`。如果确实需要单张 GPU 跑多个 worker 实例，可以参考 `docker-compose.multi.yml`。

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

C++ backend 推荐入口是 `docker-compose.cpp.yml`。它使用 `docker/Dockerfile.cpp-worker-backend` 和 `docker/entrypoint-cpp-worker-backend.sh` 定义的 C++ worker 镜像；llama.cpp-omni ref、backend 启动命令和默认 `LLAMA_SERVER_EXTRA_ARGS` 以这些文件为准。

**裸机部署：**

裸机部署时，请把 Docker 里的依赖和 entrypoint 启动命令映射到宿主机环境，并保持 Gateway、Python Worker、Backend 三段启动流程一致。

**停止 Docker 服务：**

```bash
docker compose down                      # PyTorch backend compose
docker compose -f docker-compose.cpp.yml down  # C++ backend compose
```

<br/>
<br/>


## C++ 后端（llama.cpp）

本 Demo 同时支持基于 llama.cpp-omni 的 **C++ 推理后端**。请以上方 Docker 部署章节作为权威安装路径；启动细节直接查看 `docker-compose.cpp.yml` 和 `docker/Dockerfile.cpp-worker-backend`。

### 桌面端应用（Windows & macOS）

提供 Windows 和 macOS 的开箱即用安装包，前往 [llama.cpp-omni Releases](https://github.com/tc-mb/llama.cpp-omni/releases/) 下载。

---

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
└── tmp/                      # 运行时日志和 PID 文件
```

## 配置说明

`config.json` 为裸机直接启动进程提供默认配置；如果在容器里显式挂载该文件，也可以作为容器内默认值。Docker 部署默认不会把宿主机的 `config.json` 拷进镜像；部署行为以 Compose、entrypoint、环境变量和 CLI 参数为准。

如果需要裸机调试，请从 `config.example.json` 和 `config.py` 开始看。CLI 参数优先级高于 `config.json`，缺省字段会回落到 Pydantic 默认值。


## 资源消耗

| 资源 | Token2Wav（默认） | + torch.compile |
|------|-------------------|-----------------|
| 显存（每 Worker，初始化完成后） | ~21.5 GB | ~21.5 GB |
| 模型加载时间 | ~16s | ~16s + ~5 min（有缓存）/ ~15 min（无缓存）|
| 模式切换延迟 | < 0.1ms | < 0.1ms |
| Omni Full-Duplex 单 unit 延迟（A100） | ~0.9s | **~0.5s** |
