# MiniCPM-o Demo —— 多 Worker 单 Gateway 部署

基于 Docker Compose 的多实例推理服务部署:**一个 Gateway 统一出口 + N 个 Worker-Backend 各独占一张 GPU**。

> ⚠️ **本文档(README.md)是当前部署架构的唯一准确说明。** 仓库内其它文档(`README_zh.md`、
> `docs/`、`install.sh` 等)描述的是重构前的旧单体架构(`start_all.sh` 进程内多 worker 那套),
> **均已过时(legacy),请勿参考**。需要了解实现细节时,直接看代码(`gateway.py` / `worker.py` /
> `py_backend/` / `runtime/` / `docker/`)。

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
| **gateway** | `docker/Dockerfile.gateway` | torch-free 控制面:`/v1/realtime` 入口、调度(FIFO 队列 + 负载感知)、转发、session 录制 | ❌ 不需要 | `data/`(录制)、`certs/`(TLS) |
| **worker-backend** | `docker/Dockerfile.worker-backend` | 一个容器内 `backend`(加载模型,独占 1 GPU)+ `worker`(纯转发,经 localhost 连 backend) | ✅ 每实例 1 张 | 模型权重(只读) |

- **Gateway** 不加载模型、不依赖 torch/CUDA,镜像轻量;通过 Compose 内部 DNS 静态寻址各 worker。
- **Worker-Backend** 是一个 bundle:`worker` 退化为纯转发隔离层,真正的模型推理在同容器的 `backend` 进程。
  一个实例是**有状态的长驻进程**:客户端经 Gateway 与它建立一条**持久化 WebSocket 连接**,
  在连接存活期间持续推送输入、流式接收输出(会话状态驻留在该实例,见下文「亲和性」)。
- **模型权重不进镜像**(约 19GB),运行时以只读 volume 挂载。

---

## 快速开始(Docker Compose)

### 前置条件

- Docker(含 `docker compose` v2 插件)+ `nvidia-container-toolkit`(让容器能用 GPU)
- NVIDIA 驱动支持 CUDA 12.8(镜像内 torch 为 `+cu128`,自带 CUDA 用户态库;宿主机只需驱动,不需装 CUDA Toolkit)
- 至少 N 张 GPU(每张约 22GB+ 显存供一个 backend 加载模型)
- 模型权重目录(HuggingFace 格式的 `MiniCPM-o-4_5`)

### 启动

镜像有两种来源,**二选一**:

#### 方式 A:用已发布的现成镜像(免 build,推荐)

镜像已发布到 Docker Hub,直接拉取并打成 compose 期望的本地 tag,即可跳过构建:

```bash
docker pull device0/minicpm-o-worker-backend:dev-20260603
docker pull device0/minicpm-o-gateway:dev-20260603
docker tag  device0/minicpm-o-worker-backend:dev-20260603 minicpm-wb:dev
docker tag  device0/minicpm-o-gateway:dev-20260603        minicpm-gateway:dev
```

之后启动时**不要加 `--build`**(用上面打好 tag 的本地镜像):

```bash
mkdir -p certs && openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -keyout certs/key.pem -out certs/cert.pem -subj "/CN=minicpm-o"
MODEL_HOST_PATH=/path/to/MiniCPM-o-4_5 docker compose up -d   # 无 --build
```

> worker-backend 镜像约 8GB(含 torch/CUDA);gateway 镜像很小。两者均不含模型权重(运行时挂载)。

#### 方式 B:从源码自行 build

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

> 日志查看(两种方式通用):`docker compose logs -f gateway` / `docker compose logs -f worker-backend-0`

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

### 4. 实例是长驻的,不是 serverless

- worker-backend 实例**有状态、长期驻留**:模型常驻显存,会话状态(KV cache 等)驻留在进程内。
- **不应像 serverless 那样按请求拉起 / 用完销毁**——每次重建都要重新加载模型(数十秒)、且会丢掉所有在途会话。实例应保持运行,由 Gateway 调度会话进出。

---

## 日志与数据持久化

容器本身按"无状态"对待——**所有需要留存的东西都挂到宿主机 volume**,容器删除/重建不丢数据。

### 日志

- **按实例分流查看**:`docker compose logs [-f] <service>`(`gateway` / `worker-backend-0` / …),
  每行带 service 名前缀,N 个实例不混淆。
- **滚动**:已配置 json-file 驱动,单文件 50MB × 3,避免长跑撑爆磁盘。
- 容器日志底层落在宿主机 `/var/lib/docker/containers/<id>/*.json.log`(由上面的滚动策略限大小)。
- 如需把应用日志单独挂到宿主机目录,可给 service 加一个 `- ./logs/<name>:/app/logs` volume
  (各实例挂**各自独立**的目录,避免互相覆盖)。

### 录制数据(挂载到宿主机)

Gateway 把每个 session 录制到容器内 `/app/data`,经 volume 持久化到宿主机(默认 `./data`,由
`DATA_HOST_PATH` 指定)。容器重建后历史录制仍在。每个 session 落盘到 `data/sessions/<session_id>/`:

- `meta.json` —— 会话元信息(client/page 来源、所属 worker、起止时间、时长等)
- `stream.jsonl` —— 忠实事件流(每行一个事件,音视频二进制以 `@blob/...` 指针引用)
- `blob/` —— 外置的音视频二进制(`.wav` / `.jpg`)

录制开关:`config.json` 的 `recording.enabled`(默认开)。回看:浏览器打开 `https://<SERVER_IP>:8006/s/<session_id>`。

### 挂载点小结

| 宿主机 | 容器内 | 用途 | 谁挂 |
|--------|--------|------|------|
| `MODEL_HOST_PATH` | `/models/MiniCPM-o-4_5` (ro) | 模型权重(不进镜像) | worker-backend |
| `DATA_HOST_PATH`(默认 `./data`) | `/app/data` | session 录制持久化 | gateway |
| `CERTS_HOST_PATH`(默认 `./certs`) | `/app/certs` (ro) | TLS 证书 | gateway |

---

## 冒烟测试

`PYTHONPATH=. python tests/e2e_realtime.py [chat|chat-stream|video]` —— 打到 Gateway 的 `/v1/realtime`,
跑通整条链(Gateway → Worker → Backend),验证服务可用。

---

## 停止 / 清理

```bash
docker compose down               # 停止并移除容器(保留镜像与挂载的 data/)
docker compose down --rmi local   # 连同本地镜像一起删
```

> `data/`、`certs/` 在宿主机,`docker compose down` **不会删**它们;录制历史得手动清理。

---

## 配置说明

`config.json` 完全可选(所有字段有默认值,文件缺失则全走默认):

- **Gateway / Worker** 的服务参数主要走命令行(端口、`--workers` 等),`config.json` 仅作默认值兜底。
- **Backend** 加载模型需要 `model.model_path`,但本部署由 entrypoint 通过 `--model-path` 指向挂载点,无需写进 `config.json`。
- 即 gateway/worker 无 `config.json` 也能启动;backend 缺模型路径会明确报错。
