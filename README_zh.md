# MiniCPM-o 4.5 Demo —— C++ 后端

[English README](README.md)

本文介绍如何让本仓库（Python gateway + worker + 静态前端）跑在
[`llama.cpp-omni`](https://github.com/tc-mb/llama.cpp-omni) C++ 推理引擎之上，
而不是默认的 PyTorch 后端。

什么时候用 C++ 后端：

- 显存吃紧
- 想要更低的 TTFT 与更快的解码
- 需要一个独立的 `llama-server` 进程，可单独通过 HTTP 调用

切到 C++ 后端时，仓库里的 `gateway.py` / `worker.py` / 前端都不变，只是底层
推理换了实现。

---

## TL;DR —— 5 步从零跑起来

GGUF 权重已下载好（参见
[llama.cpp-omni README](https://github.com/tc-mb/llama.cpp-omni/tree/feat/web-demo#prerequisites)）的话，整个流程就是这些命令：

```bash
# 1. 编 C++ 引擎
git clone https://github.com/tc-mb/llama.cpp-omni.git
cd llama.cpp-omni && git checkout feat/web-demo \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target llama-server -j
cd ..

# 2. 准备本仓库（Python venv + 移动端前端构建）
git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git
cd MiniCPM-o-Demo && git checkout Comni
bash install.sh
( cd frontend/mobile && bun install && bun run --bun build:static )   # 或 npm

# 3. 配置（绝对路径）
cp config.example.json config.json
# 编辑 config.json:
#   "backend": "cpp"
#   "cpp_backend.llamacpp_root" = ../llama.cpp-omni 的绝对路径
#   "cpp_backend.model_dir"     = MiniCPM-o-4_5-gguf 的绝对路径

# 4. 启动
CUDA_VISIBLE_DEVICES=0 bash start_all.sh

# 5. 浏览器打开
#    https://localhost:8040/         (桌面)
#    https://localhost:8040/mobile/  (移动 React)
```

> 第一次启动会加载所有 GGUF 模块，大概 10–60 秒。等 worker 的 `/health` 返回
> `worker_status: "idle"` 即可。

下面是逐步详细说明。

---

## 1. 你需要的三块东西

| 组件 | 来源 | 说明 |
|---|---|---|
| 本 demo（Python 服务 + 前端） | 你正在阅读的这个分支 | gateway / worker / 静态页面 / mobile React 工程 |
| `llama.cpp-omni`（C++ 引擎） | [`tc-mb/llama.cpp-omni`](https://github.com/tc-mb/llama.cpp-omni)，分支 `feat/web-demo` | 我们从这里编出 `llama-server`；`worker.py` 会以子进程方式拉起它 |
| MiniCPM-o-4_5 GGUF 权重 | 见 [llama.cpp-omni README](https://github.com/tc-mb/llama.cpp-omni/tree/feat/web-demo#prerequisites) | PyTorch 路径与 C++ 路径共用同一份权重，需自行下载 |

---

## 2. 编译 `llama-server`

> 不想自己编译？`tc-mb/llama.cpp-omni` 在 [Releases 页面](https://github.com/tc-mb/llama.cpp-omni/releases/latest)
> 提供了一键安装包（Comni for Windows / macOS），下载即用。
> 下面这一节是给走源码路径（也就是本仓库这套部署方式）的人看的。

```bash
git clone https://github.com/tc-mb/llama.cpp-omni.git
cd llama.cpp-omni
git checkout feat/web-demo

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

CMake 会自动检测 CUDA（Linux + NVIDIA）和 Metal（macOS）。编完之后，
`build/bin/llama-server` 就是后面 `worker.py` 要拉起的二进制。

你**不需要**自己手动启动 `llama-server`，Python worker 会在收到第一个会话时
拉起它。

---

## 3. 安装 Python 依赖

和 PyTorch 路径一样，跑 [`install.sh`](install.sh) 即可：

```bash
bash install.sh
```

脚本会创建 `.venv/base/`（Python 3.10），升级 `pip`，安装
`torch==2.8.0` + `torchaudio==2.8.0`，最后装
[`requirements.txt`](requirements.txt) 里其它依赖。

C++ 后端运行时不依赖 PyTorch CUDA，但 worker 本身仍然是 Python 进程，所以
venv 还是要装。当前 PyTorch 是无条件装的；后续会加一个 `cpp-only` 的安装模式
跳过它。

想用其它 Python 解释器：

```bash
PYTHON=python3.11 bash install.sh
```

---

## 4. 配置 `config.json`

把 `backend` 设为 `cpp`，并把 `cpp_backend.llamacpp_root` 和 `model_dir`
指到本地目录：

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

| 字段 | 含义 |
|---|---|
| `cpp_backend.llamacpp_root` | `llama.cpp-omni` 仓库的绝对路径。`worker.py` 会执行 `${llamacpp_root}/build/bin/llama-server`，并把 `${llamacpp_root}/tools/omni/output_<port>` 当成 TTS WAV 输出目录。 |
| `cpp_backend.model_dir` | GGUF 文件夹绝对路径（包含 LLM、TTS、vision、audio、token2wav-gguf）。 |
| `cpp_backend.llm_model` | `model_dir` 下的文件名。按你下载的量化版本填（`MiniCPM-o-4_5-Q4_K_M.gguf` / `-Q8_0.gguf` / `-F16.gguf`）。 |
| `cpp_backend.cpp_server_port` | `worker.py` 启动 `llama-server` 用的 HTTP 端口。多 worker 时每个 worker 需要独立端口。 |
| `cpp_backend.ctx_size` / `n_gpu_layers` | 直接透传到 `llama-server` 的 `--ctx-size` / `--n-gpu-layers`。 |

---

## 5. 启动整套服务

```bash
CUDA_VISIBLE_DEVICES=0 bash start_all.sh
```

启动后的拓扑：

```
gateway.py        :8040 (HTTPS)        ─┐
                                        │  HTTP / WS  (内部)
worker.py         :22440  GPU 0        ─┘
    │
    │  以子进程拉起 + HTTP 调用
    ▼
llama-server      :19080  GPU 0
    /v1/stream/omni_init
    /v1/stream/update_session_config
    /v1/stream/prefill
    /v1/stream/decode  (SSE)
    /v1/stream/break
```

第一次拉起 `llama-server` 会加载 GGUF 各模块（VPM / APM / LLM / TTS /
Token2Wav），通常 10–60 秒。worker 的 `/health` 在 `omni_init` 完成后会返回
`worker_status: "idle"`。

之后用浏览器访问：

- `https://localhost:8040/` — 桌面端总入口（Home / Omni / Audio-Duplex / Turnbased / Half-Duplex）
- `https://localhost:8040/mobile/` — 移动端 React 前端
- `https://localhost:8040/mobile-omni/` — 桌面 Omni 页面的移动端适配版（DOM 桥接到 omni-app.js）

> 摄像头和麦克风权限要求 HTTPS。仓库里 `certs/` 下自带的自签名证书可以本地
> 直接用，浏览器报警告时点继续即可。回退到 HTTP（`bash start_all.sh --http`）
> 只能用文字交互。

---

## 6. 移动端前端构建

`/mobile/` 这个路由实际是 `static/mobile/`，**不在 git 里**。它是
`frontend/mobile/` 下 React + Vite 工程的产物，第一次拉代码后构建一次就好：

```bash
cd frontend/mobile
bun install                 # 或 npm install，要求 Node ≥ 20.19
bun run --bun build:static  # 输出到 ../../static/mobile/
```

代理调试 / 仅 npm / 热更新等更多说明见
[`frontend/mobile/README.md`](frontend/mobile/README.md)。

---

## 7. 停止

```bash
pkill -f "gateway.py|worker.py|llama-server"
```

`worker.py` 在每个会话结束后会重启 `llama-server`（`full_reinit`），保证
KV cache 状态干净。

---

## 多 GPU

同机多 worker：

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

每个 worker 占一张卡（`CUDA_VISIBLE_DEVICES=0,1 bash start_all.sh`），各自在
`cpp_server_port + worker_index` 上拉起独立的 `llama-server`。

---

## 常见问题

| 现象 | 多半的原因 |
|---|---|
| worker 日志里报 `llama-server not found` | `cpp_backend.llamacpp_root` 不对，或者 `cmake --build … --target llama-server` 没跑成功。 |
| worker 的 `/health` 长时间停在 `worker_status: "loading"` | `omni_init` 还在加载 GGUF。看 `tmp/worker_<i>.log` 里 `[CPP]` 开头的行就是 C++ 侧输出。 |
| `${llamacpp_root}/tools/omni/output_<port>/round_XXX/` 下能看到 WAV，但浏览器没声音 | 检查是不是 HTTPS。很多浏览器对 `Audio` / `MediaDevices` 在非安全源下都会拒绝。 |
| 对话中途 `kv_cache_length` 一直在缩 | C++ 侧滑动窗口在裁 KV。桌面和移动端设置抽屉都有「Stop on KV pruning」开关（默认开），命中时会干净地结束会话。 |
