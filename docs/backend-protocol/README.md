## 文档结构

| 文档 | 内容 | 何时看 |
|------|------|--------|
| [network.md](./network.md) | **过程 / 原语 / 完成语义。** 四原语（init/push/pull/unary）、session 生命周期、输入与下行事件语义、背压、断线、fail-fast。规范层。 | 先读这份，理解协议形状与状态机。 |
| [schema.md](./schema.md) | **消息 schema / 字段 / 编码。** 消息封套、契约字段 vs 透传字段、音频/图像编码、各消息的字段表、metrics 字段。用 RFC 2119 MUST/SHOULD/MAY。 | 实现具体收发与编解码时对照。 |
| [sequences.md](./sequences.md) | **时序图 + 示例数据包。** full_duplex、turn_based 流式、turn_based 一次性三条交互的 mermaid 时序图与真实抓包示例。 | 想看一次完整交互长什么样时。 |

## 参考实现

仓库里已有一份**可跑通的 Python backend 实现**（协议的服务端），位于 `py_backend/`。
用别的语言/引擎（如 C++）实现新 backend 时按下表取舍：只需照着“backend 自身”那一层
实现协议表面；**backend 以下（怎么跑推理）完全由你自己决定**，上游与对端都不用碰。

| 文件 | 角色 | 新 backend 实现者 |
|------|------|------------------|
| `py_backend/server.py` | 协议服务端：WS `/backend` + HTTP close 端点、init/push/pull 分发、下行事件发送、生命周期与 fail-fast | **主要参考**（协议表面照它实现） |
| `py_backend/chat_util.py` | turn_based 请求解析（messages/content/generation/tts 字段） | **参考**（字段级解析） |
| `py_backend/media.py` | 音频（float PCM）/ JPEG 帧解码（§1.3 / §1.4 的字节级实现） | **参考**（编解码细节） |
| `py_backend/voice.py` | 参考音频（ref_audio / tts_ref_audio）处理 | **参考** |
| `core/` 全部 + `MiniCPMO45/` | 本实现的**推理层**：模型、框架、引擎，以及 server 驱动引擎的适配层（`pytorch_backend.py`）、推理核心（`unified.py`/`base.py`）、字段类型（`schemas/`）等 | **整体无需参考**——这些都在 backend 以下，用什么框架/模型/引擎跑出协议要求的输出，完全由你决定（cpp 不必用 torch）。字段含义以 schema.md 为准 |
| `runtime/`（backend_client / session / …） | **对端**：驱动你 backend 的 scheduler/runtime 客户端 | **不要实现**——它是协议另一侧；可读它了解对端会发什么、期望收什么 |
| `worker.py`、`gateway.py`、`gateway_modules/` | 更上游的转发层与公网网关 | **与 backend 无关**，忽略 |

## 运行参考实现

三个组件分别启动，顺序：**先 backend（等模型加载好），再 worker，最后 gateway**。
均需 `PYTHONPATH=.`，用仓库 venv `.venv/base/bin/python`。

```bash
# 1) backend 协议服务端（py_backend/server.py，加载模型）
PYTHONPATH=. .venv/base/bin/python -m py_backend.server \
    --host 127.0.0.1 --port 22500 --gpu-id 0 \
    --model-path /user/weihongliang/autoshow_omni/models/MiniCPM-o-4_5

# 2) worker（纯转发，指向上面的 backend）
PYTHONPATH=. .venv/base/bin/python worker.py \
    --host 127.0.0.1 --port 22400 --gpu-id 0 \
    --backend-server-url http://127.0.0.1:22500

# 3) gateway（公网 /v1/realtime 入口）
PYTHONPATH=. .venv/base/bin/python gateway.py \
    --host 0.0.0.0 --port 8006 --http --workers localhost:22400
```

多 worker：每个 worker 各配一个 backend（再起 `--port 22501` 的 backend + `--port 22401
--backend-server-url http://127.0.0.1:22501` 的 worker），gateway `--workers
localhost:22400,localhost:22401`。

健康检查：

```bash
curl http://127.0.0.1:22500/health   # backend
curl http://127.0.0.1:22400/health   # worker
curl http://127.0.0.1:8006/health    # gateway
```

## 端到端测试

`tests/e2e_realtime.py` 打到 gateway 的 `ws://127.0.0.1:8006/v1/realtime`，跑通整条链
（gateway → worker → backend），验证协议而非模型质量。三个服务起好后：

```bash
PYTHONPATH=. .venv/base/bin/python tests/e2e_realtime.py              # chat + video（默认全跑）
PYTHONPATH=. .venv/base/bin/python tests/e2e_realtime.py chat         # 仅 turn_based 非流式
PYTHONPATH=. .venv/base/bin/python tests/e2e_realtime.py chat-stream  # turn_based 流式
PYTHONPATH=. .venv/base/bin/python tests/e2e_realtime.py video        # full_duplex 音视频
```

实现了别的 backend 后，把 worker 的 `--backend-server-url` 指向它、重跑同一个测试即可验证。

