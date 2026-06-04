# 34 机器改动清单与恢复方法

> 机器:`weihongliang@34.66.244.63`(GCP GPU 机,2× RTX PRO 6000 Blackwell 98G,Ubuntu 24.04)
> 用途:worker+backend Docker 镜像的构建 + 测试 + 推送到 LIS/SWR
> 记录时间:2026-06-03

---

## 背景:为什么有这些网络改动

34 能访问 LIS API(`lis.modelbest.co`,公网),但**访问不了 SWR 镜像仓**
(`modelbest-kt8zv1.swr-pro.myhuaweicloud.com` → 121.36.48.213,TCP 超时,SWR 是华为云内网/白名单端点)。
为了让 `lisctl image upload` 能把镜像推进 SWR,临时搭了一条
**34 → Mac → SWR** 的中转管道(Mac 能访问 SWR)。

关键障碍:lisctl 是**静态链接 Go 二进制**,既不认 `HTTPS_PROXY` 环境变量,
又躲过 proxychains 的 LD_PRELOAD,所以只能用**网络层透明劫持**(hosts + socat)。

> 若运维把 34 的公网出口 IP 加进 SWR 白名单,以下整套中转管道都不再需要,34 可直接推。

---

## 🔴 临时网络管道(任务完成后全部撤掉,撤掉后 34 网络完全复原)

| # | 改动 | 持久性 | 恢复方法 |
|---|---|---|---|
| A | `/etc/hosts` 加 `127.0.0.1 modelbest-kt8zv1.swr-pro.myhuaweicloud.com` | **持久**（写文件） | `sudo sed -i '/modelbest-kt8zv1.swr-pro/d' /etc/hosts` |
| B | socat 进程：`127.0.0.1:443` → socks(1080) → SWR:443 | 临时（进程） | `sudo pkill -f 'socat.*443'` |
| C | SSH 反向隧道 `:1080`（SOCKS），**由 Mac 端发起** | 临时（Mac 控制） | Mac 上 Ctrl-C 掉 `ssh -N -R 1080 weihongliang@34.66.244.63` |

> A 是唯一持久的网络改动，唯一必须手动删的。B/C 进程/隧道断开即没。
> **dockerd 始终未碰**（没配全局 proxy），docker 平时行为完全正常 —— 这是用 socat 透明劫持
> 而非改 daemon proxy 的好处：零侵入 dockerd。

### 重新搭起管道（下次要推镜像时）

```bash
# 1) Mac 上：起反向 SOCKS 隧道（前台占一个终端窗口，保持不关）
ssh -N -R 1080 weihongliang@34.66.244.63

# 2) 34 上：socat 透明劫持（注意 --experimental，SOCKS5-CONNECT 需要它）
SWR=modelbest-kt8zv1.swr-pro.myhuaweicloud.com
sudo bash -c "setsid socat --experimental TCP4-LISTEN:443,bind=127.0.0.1,reuseaddr,fork \
  SOCKS5-CONNECT:127.0.0.1:1080:${SWR}:443 >/tmp/socat.log 2>&1 < /dev/null &"

# 3) 34 上：hosts 劫持（若已存在可跳过）
grep -q "$SWR" /etc/hosts || echo "127.0.0.1 $SWR" | sudo tee -a /etc/hosts

# 4) 验证全链通（期望 HTTP 401）
curl -sS -o /dev/null -w "%{http_code}\n" --max-time 25 "https://${SWR}/v2/"

# 5) 推镜像（透明劫持，无需任何 proxy 环境变量）
sudo HOME=/home/weihongliang ~/.local/bin/lisctl --format=json \
  image upload <本地镜像:tag> --name <名> --version <版> --env dev
```

### 一键撤掉临时管道（任务完成后）

```bash
# 34 上：
sudo pkill -f 'socat.*443'
sudo sed -i '/modelbest-kt8zv1.swr-pro/d' /etc/hosts
# Mac 上：Ctrl-C 掉 ssh -N -R 1080
```

---

## 🟢 持久安装（对部署有用，建议保留；如需彻底清理见卸载列）

| # | 改动 | 建议 | 卸载方法 |
|---|---|---|---|
| D | `nvidia-container-toolkit` + `nvidia-container-toolkit-base`（GPU 进容器） | **保留** | `sudo apt-get remove nvidia-container-toolkit nvidia-container-toolkit-base` |
| E | `docker-buildx` 0.30.1（BuildKit 缓存挂载） | **保留** | `sudo apt-get remove docker-buildx` |
| F | `socat`、`proxychains4`（中转工具） | 可留可删 | `sudo apt-get remove socat proxychains4` |
| G | `/etc/docker/daemon.json` 加了 nvidia runtime + 重启过 docker daemon | **保留**（GPU 要用） | 删 daemon.json 里 nvidia runtime 段并重启 docker |

> proxychains4 实际对 lisctl 无效（静态 Go），可删；socat 是真正起作用的中转工具。

---

## 🟢 文件 / 登录态

| # | 改动 | 恢复 |
|---|---|---|
| H | `~/.config/lisctl/config.json`（lisctl 登录 token，token_source=manual，从 dev 机搬的 PAT，到期 2026-07-03） | `~/.local/bin/lisctl auth logout` 或删该文件 |
| I | `~/.local/bin/lisctl`（二进制 v1.0.22-dev） | `rm ~/.local/bin/lisctl` |
| J | `/tmp/pc.conf`、`/tmp/socat.log`、`/tmp/lisctl.tar.gz`、拉的 busybox 镜像 | 临时文件无害；`sudo docker rmi busybox` |

---

## 🟢 LIS 平台上的痕迹（验证管道时产生）

| 资产 | 说明 | 清理 |
|---|---|---|
| `probe-min:v1`（inference_image，Completed） | busybox 测试镜像，验证上传管道用 | `sudo HOME=/home/weihongliang ~/.local/bin/lisctl image delete <id> --yes` |
| 4 条 Failed 的 probe-min 上传任务 | 调试过程留痕，未产生实际镜像，一般不用管 | （可不处理） |

---

## ⭐ 镜像发布到 LIS 的两条路径（2026-06-03 验证）

### 路径一（推荐）：经 docker.io 中转，`image register` —— 不需要任何隧道

LIS 的 `image register` 是**后端 Worker Pod 用 skopeo 去拉源镜像并转存进 SWR**，
本地不传镜像层、不碰 SWR。所以只要源镜像在一个 LIS 后端能访问的公网 registry（docker.io）即可：

```bash
# 34 上：build → push docker.io（34 直连 docker.io，0.08s 延迟，快）
sudo docker build -t <dockerhub账号>/minicpm-wb:<tag> -f docker/Dockerfile.worker-backend .
sudo docker login                 # Docker Hub 账号
sudo docker push <dockerhub账号>/minicpm-wb:<tag>

# 让 LIS 后端从 docker.io 拉进 SWR（本地不传镜像层，无需隧道）
sudo HOME=/home/weihongliang ~/.local/bin/lisctl image register \
  docker.io/<dockerhub账号>/minicpm-wb:<tag> \
  --name minicpm-wb --version <tag> --engine vllm --wait
# 私有 repo 加：--registry-username <user> --registry-password <token>
```

优点：不需要 socat/hosts/Mac 隧道，不需要 SWR 白名单，大镜像走 docker.io→华为云数据中心链路（快）。
已用 `docker.io/library/busybox` 验证：LIS 后端 skopeo 自动拉取转存 → Completed。

### 路径二（备份）：`image upload` 直推 SWR —— 需要本节上面那套 socat 隧道

本地 docker 直推 SWR，34 到不了 SWR，所以需要 Mac 中转隧道（见本文件🔴部分）。
大镜像经 Mac 隧道很慢（busybox 4MiB 实测受固定开销影响 ~149KiB/s，大镜像可能数小时）。
**仅在路径一不可用时才用。**

---

## 代码工作区（dev 机，非 34）

- worktree：`/user/weihongliang/MiniCPM-o-Demo-wt-docker-deploy-2026-06-03`，分支 `wt/docker-deploy-2026-06-03`
- 已写文件（未 commit）：`docker/Dockerfile.worker-backend`、`docker/entrypoint-worker-backend.sh`
- 34 上的代码副本：`~/minicpm-docker`（同分支，待 build）
- 模型权重（34，无需下载）：`/root/xubokai/MiniCPM-o-4_5`（19G 完整，4 shard）
