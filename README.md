# MiniCPM-o Demo —— 多 Worker 单 Gateway 部署

基于 Docker Compose 的多实例推理服务部署:**一个 Gateway 统一出口 + N 个 Worker-Backend 各独占一张 GPU**。

---

## 架构

```
                       ┌─────────────────────────────────┐
   前端/客户端 ──WSS──▶ │  Gateway (容器, torch-free)      │  :8006  /v1/realtime
   (浏览器/移动端)      │  统一出口 · 调度 · session 录制   │
                       └───────┬─────────────────┬────────┘
                       compose DNS 静态寻址        compose DNS
                  worker-backend-0:22400     worker-backend-1:22400
                            ▼                       ▼
                ┌────────────────────┐   ┌────────────────────┐
                │ worker-backend-0   │   │ worker-backend-1   │
                │  worker (转发)      │   │  worker (转发)      │
                │    └─ backend       │   │    └─ backend       │
                │       (加载模型)     │   │       (加载模型)     │
                │  GPU 0              │   │  GPU 1              │
                └────────────────────┘   └────────────────────┘
```

### 两类镜像、各司其职

| 组件 | 镜像 | 内容 | GPU | 持久化挂载 |
|------|------|------|-----|-----------|
| **gateway** | `docker/Dockerfile.gateway` | torch-free 控制面:`/v1/realtime` 入口、调度(FIFO 队列 + 负载感知)、协议转发、session 录制 | ❌ 不需要 | `data/`(录制)、`certs/`(TLS) |
| **worker-backend** | `docker/Dockerfile.worker-backend` | 一个容器内 `backend`(加载模型,独占 1 GPU)+ `worker`(纯转发,经 localhost 连 backend) | ✅ 每实例 1 张 | 模型权重(只读) |

- **Gateway** 不加载模型、不依赖 torch/CUDA,镜像轻量;通过 Compose 内部 DNS 静态寻址各 worker。
- **Worker-Backend** 是一个 bundle:`worker` 退化为纯转发隔离层,真正的模型推理在同容器的 `backend` 进程(协议见 `docs/backend-protocol/`)。
- **模型权重不进镜像**(约 19GB),运行时以只读 volume 挂载。

---

## 快速开始(Docker Compose)

### 前置条件

- Docker(含 `docker compose` v2 插件)+ `nvidia-container-toolkit`(让容器能用 GPU)
- NVIDIA 驱动支持 CUDA 12.8(镜像内 torch 为 `+cu128`,自带 CUDA 用户态库;宿主机只需驱动,不需装 CUDA Toolkit)
- 至少 N 张 GPU(每张约 22GB+ 显存供一个 backend 加载模型)
- 模型权重目录(HuggingFace 格式的 `MiniCPM-o-4_5`)

### 启动

```bash
# 1) 准备 TLS 证书(实时音视频页面需 HTTPS — 浏览器仅在 https/localhost 提供麦克风权限)
#    自签即可(浏览器会提示不安全,点继续);也可换成你的真实域名证书。
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -keyout certs/key.pem -out certs/cert.pem -subj "/CN=minicpm-o"

# 2) 指定宿主机上的模型权重路径,一键起全栈(首次会 build 镜像)
MODEL_HOST_PATH=/path/to/MiniCPM-o-4_5 docker compose up -d --build

# 3) 看日志(按实例分流)
docker compose logs -f gateway
docker compose logs -f worker-backend-0
```

启动顺序由 Compose 健康门控保证:**worker-backend 各自加载模型 → healthy → gateway 才启动**(`depends_on: condition: service_healthy`)。单个 backend 加载模型约 30–90s。

### 访问

```
https://<SERVER_IP>:8006/            # 入口
https://<SERVER_IP>:8006/omni        # 实时音视频(需麦克风,必须 https)
https://<SERVER_IP>:8006/turnbased   # 纯文字对话(不需麦克风)
https://<SERVER_IP>:8006/health      # 健康检查
```

> 自签证书:浏览器会警告不安全,点"高级 → 继续访问"。实时页面进入后需"允许使用麦克风"。

### 环境变量

| 变量 | 说明 | 默认 |
|------|------|------|
| `MODEL_HOST_PATH` | 宿主机模型权重目录(必填) | — |
| `DATA_HOST_PATH` | session 录制落盘目录 | `./data` |
| `CERTS_HOST_PATH` | TLS 证书目录 | `./certs` |

---

## 增减 GPU / 扩缩实例

实例是**显式声明**的(不能用 `docker compose --scale` —— scale 出的副本会争抢同一张 GPU)。
增加一个 GPU 实例,在 `docker-compose.yml` 里:

1. 复制一个 `worker-backend-N` service(继承 `x-worker-backend` 锚点),把 `device_ids` 指向新 GPU 编号:
   ```yaml
   worker-backend-2:
     <<: *worker-backend-base
     container_name: minicpm-wb-2
     volumes:
       - ${MODEL_HOST_PATH:?}:/models/MiniCPM-o-4_5:ro
     deploy:
       resources:
         reservations:
           devices:
             - {driver: nvidia, device_ids: ["2"], capabilities: [gpu]}
   ```
2. 把新实例加进 gateway 的 `--workers` 列表:`worker-backend-0:22400,worker-backend-1:22400,worker-backend-2:22400`
3. 在 gateway 的 `depends_on` 加上 `worker-backend-2: {condition: service_healthy}`

> 容器内 GPU 永远是 `cuda:0`:`device_ids` 决定容器看到宿主机哪张卡,容器内部一律编号 0(故 `GPU_ID=0` 不变)。

---

## 一张 GPU 跑多个 Worker(提并发)

当单卡显存远大于一份模型(本模型约 21.8GB/份),可以在一张卡上放多个 worker-backend 实例
来**提高并发 session 数**。做法:多个 service 的 `device_ids` 指向**同一张卡**。

```yaml
# GPU 0 上放两个实例
worker-backend-0a:
  <<: *worker-backend-base
  container_name: minicpm-wb-0a
  volumes:
    - ${MODEL_HOST_PATH:?}:/models/MiniCPM-o-4_5:ro
  deploy:
    resources:
      reservations:
        devices:
          - {driver: nvidia, device_ids: ["0"], capabilities: [gpu]}

worker-backend-0b:
  <<: *worker-backend-base
  container_name: minicpm-wb-0b
  volumes:
    - ${MODEL_HOST_PATH:?}:/models/MiniCPM-o-4_5:ro
  deploy:
    resources:
      reservations:
        devices:
          - {driver: nvidia, device_ids: ["0"], capabilities: [gpu]}   # ← 同样指 0
```

然后把全部实例加进 gateway 的 `--workers` 与 `depends_on`:
`worker-backend-0a:22400,worker-backend-0b:22400,...`

要点:
- **端口不用手动错开**:每个实例是独立容器,容器内 backend(22500)/worker(22400)端口都用默认值,
  容器间靠 compose 网络隔离,不会冲突。`environment` 全部继承 `*worker-backend-base`,无需改。
- **显存核算**:每个实例独立加载一份模型(约 21.8GB,**不共享权重**)。一张 98GB 卡放
  N 个 = N × 21.8GB,**建议每卡 2–3 个**,给运行时 KV cache 增长(单 session 峰值可达 27GB+)留余量,
  不要顶满。

### 代价(务必权衡)

| 维度 | 影响 |
|------|------|
| **显存** | N 份**重复的**模型副本 —— 这是冗余;真正消除需 backend 层多 session 批处理(如 vLLM 式),compose 做不到 |
| **算力** | 同卡多个 backend **共享 GPU 计算单元(SM)**;它们同时推理时算力被瓜分,每个 session 延迟变长 |
| **适用场景** | 适合"并发数高但每个 session 不全程满载"(对话有思考间隙);不适合多个 session 同时持续高强度生成 |
| **启动** | N 份模型同时加载会抢 GPU/IO,启动更慢 |

> 一句话:一卡多 worker 是用"显存冗余 + 算力瓜分"换"更高并发"。卡的显存空、并发是瓶颈时值得;
> 追求单请求低延迟时不要这么做。

---

## 亲和性 / 会话黏性(重要)

这是一个**有状态**服务,部署时必须理解它的会话亲和性约束:

### 1. 一个 Session 绑定一个 Worker,全程不迁移

- 每个 backend **同时只服务一个 session**(并发上限 = worker-backend 实例数 = GPU 数)。
- Gateway 为新 session 挑选一个空闲 worker 后,该 session 的**所有消息全程钉在这个 worker 上**,直到结束。
- 会话状态(KV cache 等)只存在那一个 backend 进程里,**无法迁移到其它实例**;断连即终止,不支持续传。

### 2. Gateway 必须能逐一寻址每个 Worker —— 不要在中间再加负载均衡

- Gateway 自己负责调度(选哪个 worker、FIFO 排队),并**亲自把 session 钉在选定的 worker** 上。
- 因此 Gateway 必须能**直连到每一个具体的 worker 实例**(本部署用 Compose 内部 DNS:`worker-backend-0`、`worker-backend-1` …)。
- ⚠️ **不要在 Gateway 与 worker 之间再插一层 L4/L7 负载均衡**:那会把 Gateway 已经定向的请求随机打散到别的 worker,导致会话状态错乱。
- 若未来跨主机部署(Gateway 与 Worker 不同机),同理:每个 worker 需有**稳定且可被 Gateway 逐一寻址**的地址(如 K8s 的 Headless Service / StatefulSet,而非负载均衡的单一 Service VIP)。

### 3. Gateway 是有状态单点

- worker 池、session→worker 映射都在 Gateway 内存里 → Gateway 单实例,不做多副本;挂掉则在途 session 全部丢失。
- 这是本架构的有意取舍:**调度集中在 Gateway**,避免与外部编排设施的调度冲突。

### 4. 容量 = GPU 数

- N 张 GPU = N 个 worker-backend = 最多 N 个并发 session。超出的请求在 Gateway 侧 FIFO 排队(队列满则拒绝)。

---

## 日志与数据

- **日志**:`docker compose logs [-f] <service>`,按实例分流(gateway / worker-backend-0 / …)。已配置 json-file 滚动(单文件 50MB × 3)。
- **录制**:每个 session 落盘到 `data/sessions/<session_id>/`:
  - `meta.json` —— 会话元信息(client/page 来源、worker、时长等)
  - `stream.jsonl` —— 忠实事件流(每行一个协议帧,音视频二进制以 `@blob/...` 指针引用)
  - `blob/` —— 外置的音视频二进制(`.wav` / `.jpg`)
- 录制开关:`config.json` 的 `recording.enabled`(默认开)。

---

## 协议与组件

- backend 协议(worker↔backend 的 init/push/pull/unary 四原语)规范见 [`docs/backend-protocol/`](docs/backend-protocol/)。
- 端到端冒烟测试:`PYTHONPATH=. python tests/e2e_realtime.py [chat|chat-stream|video]`(打到 Gateway 的 `/v1/realtime`,验证整条链)。

---

## 停止 / 清理

```bash
docker compose down               # 停止并移除容器(保留镜像与 data/)
docker compose down --rmi local   # 连同本地镜像一起删
```

---

## 配置说明

`config.json` 完全可选(所有字段有默认值,文件缺失则全走默认):

- **Gateway / Worker** 的服务参数主要走命令行(端口、`--workers` 等),`config.json` 仅作默认值兜底。
- **Backend** 加载模型需要 `model.model_path`,但本部署由 entrypoint 通过 `--model-path` 指向挂载点,无需写进 `config.json`。
- 即 gateway/worker 无 `config.json` 也能启动;backend 缺模型路径会明确报错。
