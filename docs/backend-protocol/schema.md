# Backend Server Protocol — Message Schema

本文是 [Backend Server Network Protocol](./network.md) 的配套规范。
上层文档定义过程、原语与完成语义，并将具体 JSON Schema、字段名、音视频编码、
metrics 字段列为 non-goals（见其 §10）。本文定义这些被推迟的部分：消息封套、字段、
编码与约束。

本文用 MUST / MUST NOT / SHOULD / MAY 表示约束强度（含义同 RFC 2119）。

本文不重复上层文档的过程与生命周期语义；章节编号与其对应：本文 §3 对应上层 §4
（Lifecycle），§4 对应 §5（Input），§5 对应 §6（Pull Events）。

---

## 1. 通用约定

### 1.1 消息封套

数据通道（WebSocket）与控制通道（unary）上的每条消息 MUST 是一个 JSON 对象，
且 MUST 含字符串字段 `type`。`type` 标识消息语义。

接收方 MUST 按 `type` 分发，MUST 忽略不认识的顶层字段（向前兼容）。

### 1.2 字段分类

字段分两类，实现者 MUST 区别对待：

- **契约字段（normative）**：本文用 MUST/SHOULD 精确定义其类型、编码与约束。
  取值或编码不符将导致解析失败、数据错位或推理错误。
- **透传字段（opaque pass-through）**：对转发层（worker/runtime）不透明的黑盒。
  转发层 MUST 原样转发，MUST NOT 解析或依赖其内部结构；只有 backend MAY 读取。
  全部可选，缺省时由 backend 取默认值。本文不把其内部字段列为契约——典型例子是
  full_duplex 的 `config`（其字段见 §6，仅供参考）。

### 1.3 音频编码（契约）

本协议中所有音频负载 MUST 为 base64 编码的 **裸 PCM**，样本类型为 **IEEE 754 浮点**；
MUST NOT 是 WAV/容器封装，MUST NOT 是整型 PCM。

适用范围：full_duplex 输入音频、turn_based 输入音频、参考音频、输出音频 delta。

具体的字节序、采样率与声道当前未在协议层钉死（实现现状为 float32 本机字节序 /
16000 Hz / 单声道）；是否显式约定并由 backend 入口校验，见 §8 Open Issues。

### 1.4 图像/视频帧编码（契约）

视频/图像帧 MUST 为 base64 编码的单张 **JPEG** 图片。多帧以 base64 字符串数组表示。

适用于 `image` 内容项（§4.4）与 full_duplex 的 `video_frames`（§4.1）。turn_based
`messages` 中的 `video` 内容项是另一种编码（完整 MP4 容器文件），见 §4.4。

### 1.5 时间戳

backend 发出的每条下行事件 SHOULD 含字段 `server_send_ts`：事件发送时刻的
Unix 时间，单位秒，浮点。接收方用其计算网络漂移；该字段 MUST NOT 影响事件排序
（排序语义见上层 §3.1）。

---

## 2. 传输端点

### 2.1 数据通道

`WebSocket /backend`

第一条消息 MUST 是 init（见 §3.1）。其后上行消息为 push（§4）或 close（§3.3）。
下行为事件流（§5）。

### 2.2 控制通道

`POST /sessions/{session_id}/close`

请求体 MUST 为 JSON 对象，MAY 含 `reason`（字符串）。对存在的 session，响应 MUST 为
JSON 对象，含 `ok`（布尔）、`session_id`（字符串）、`closed`（布尔）。close 后 session 被
遗忘，对同一 id 再次 close 返回 **HTTP 404**（重复语义见 §7.3）。

---

## 3. 生命周期消息

### 3.1 Init 请求

`type` MUST 为 `session.init`。

字段：

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `type` | string = `session.init` | MUST | |
| `payload` | object | MUST | init 参数，结构见下 |
| `payload.mode` | string | MUST | `turn_based` 或 `full_duplex`；缺省 `full_duplex` |
| `payload.voice` | object | MAY | 参考音频，见 §3.1.1（full_duplex） |
| `payload.system_prompt` | string | MAY | 系统提示（full_duplex） |
| `payload.config` | object | MAY | 透传采样/解码参数，见 §6 |

session identity 由 backend 分配，**不接受**客户端指定：init 请求 MUST NOT 含
`session_id`；backend MUST 自行生成 session id，并在 `session.created`（§3.2）中回传。
backend 收到带 `session_id` 的 init MAY 忽略该字段，MUST NOT 采用客户端建议值。

#### 3.1.1 参考音频（voice，契约）

`payload.voice` 为对象，字段均可选：

| 字段 | 别名 | 说明 |
|------|------|------|
| `ref_audio` | `ref_audio_base64` | LLM 参考音频，编码见 §1.3 |
| `tts_ref_audio` | `tts_ref_audio_base64` | TTS 音色参考，编码见 §1.3 |

约束：若 `tts_ref_audio` 缺省而 `ref_audio` 存在，backend MUST 以 `ref_audio` 作为
TTS 参考。参考音频 MUST 以 base64 传输（§1.3），MUST NOT 使用文件路径
（路径是服务端本地配置，不属于本协议）。

### 3.2 Init 确认（session.created）

backend 完成初始化后 MUST 在下行流发送：

| 字段 | 类型 | 约束 |
|------|------|------|
| `type` | string = `session.created` | MUST |
| `session_id` | string | MUST，backend 分配的 session id（§3.1） |
| `mode` | string | MUST |
| `metrics` | object | MAY，见 §5.4 |

### 3.3 Close

close 只走控制通道（HTTP unary，见 §2.2）；数据通道（WebSocket）上 MUST NOT 发送
close 消息（理由见上层 §3.2）。完成语义见上层 §3.2 / §4.3。

---

## 4. 输入消息（push）

push 消息：`type` MUST 为 `input.append`。模型输入 MUST 位于 `input` 字段（对象）。

### 4.1 Full-Duplex 输入

`input` 对象：

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `audio` | string | MUST | base64 音频，编码见 §1.3 |
| `video_frames` | string[] | MAY | JPEG 帧数组，编码见 §1.4；别名 `frame_base64_list` |
| `max_slice_nums` | int \| int[] | MAY | HD 切片数，约束见 §4.3 |
| `force_listen` | bool | MAY | 强制本次为 listen |

一条 push MUST 表示**一个时间片**：1 段音频 + 0..N 视频帧。

### 4.2 Turn-Based 输入

`input` 对象：

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `messages` | object[] | MUST | 见 §4.4 |
| `streaming` | bool | MUST | true=流式 delta；false=单次 response.done |
| `generation` | object | SHOULD | `max_new_tokens`(int)、`length_penalty`(float) |
| `image` | object | MAY | `max_slice_nums`(int)，见 §4.3 |
| `tts` | object | MAY | `enabled`(bool)、`ref_audio_data`(string，§1.3) |
| `omni_mode` | bool | MAY | 透传，缺省 false |
| `use_tts_template` | bool | MAY | 透传；backend MUST 以 `use_tts_template OR tts.enabled` 求值 |
| `enable_thinking` | bool | MAY | 透传，缺省 false |

### 4.3 切片数约束（契约）

`max_slice_nums` 若为整数，应用于每一帧；若为数组，其长度 MUST 等于帧数，
否则为非法输入（按上层 §6.8 终止 session）。

### 4.4 消息内容（messages，契约）

`messages` 为对象数组，每个对象含 `role`（string）与 `content`。
`content` 为字符串，或内容项数组。每个内容项含 `type`，取值与负载：

| `type` | 负载字段 | 编码 |
|--------|----------|------|
| `text` | `text` (string) | — |
| `audio` | `data` (string) | §1.3 |
| `image` | `data` (string) | §1.4（单张 JPEG）|
| `video` | `data` (string)、`stack_frames`(int) | base64 **视频容器文件（MP4）**，见下注 |

> **`video` 内容项的 `data` 与 §1.4 的“JPEG 帧”不同。** 它是 base64 编码的**完整视频
> 容器文件（MP4）**：backend 解码该文件，抽取视频帧与对应音频段一并交给模型；
> `stack_frames` 控制每个采样点堆叠的帧数。这与 full_duplex 的 `video_frames`
> （= JPEG 帧数组，§1.4）是**两条不同的视频输入路径**，不要混用。

---

## 5. 下行事件

事件 `type` 与语义见上层 §6。本节定义字段。所有事件 SHOULD 含 `server_send_ts`（§1.5）。

### 5.1 response.output.delta

| 字段 | 类型 | 约束 |
|------|------|------|
| `type` | string = `response.output.delta` | MUST |
| `kind` | string | MUST，取 `text` \| `audio` \| `listen` |
| `session_id` | string | SHOULD |
| `response_id` | string | SHOULD |
| `input_id` | string | MAY |
| `text` | string | `kind=text` 时 MUST |
| `audio` | string | `kind=audio` 时 MUST，编码见 §1.3 |
| `metrics` | object | MAY，见 §5.4 |

一个 delta MUST 只表达一种 `kind`（不在同一帧混合文字与音频）。

### 5.2 response.done（turn_based）

| 字段 | 类型 | 约束 |
|------|------|------|
| `type` | string = `response.done` | MUST |
| `session_id` / `response_id` | string | SHOULD |
| `input_id` | string | MAY |
| `text` | string | MUST，完整文本 |
| `audio` | string \| null | 非流式且开启 TTS 时为音频（§1.3），否则 null |
| `reason` | string | SHOULD，如 `turn_end` |
| `metrics` | object | MAY |

### 5.3 session.closed

| 字段 | 类型 | 约束 |
|------|------|------|
| `type` | string = `session.closed` | MUST |
| `session_id` | string | SHOULD |
| `reason` | string | SHOULD |
| `diagnostic` | object | MAY，致命错误时含 `message`(string) |

### 5.4 metrics 对象

附着于下行事件（§5.1/§5.2/§3.2）。所有字段可选，有值则附带：

| 字段 | 类型 | 含义 |
|------|------|------|
| `backend` | string | 后端标识 |
| `kv_cache_length` | int | 当前 KV cache token 数 |
| `prefill_ms` `generate_ms` `wall_clock_ms` | float | 各阶段耗时（毫秒） |
| `cost_llm_ms` `cost_tts_prep_ms` `cost_tts_ms` `cost_token2wav_ms` | float | 细分耗时 |
| `n_tokens` `n_tts_tokens` | int | token 计数 |
| `vision_slices` `vision_tokens` | int | 视觉切片/token 数 |

---

## 6. 透传参数（config）

full_duplex 的 `payload.config`（亦称 sampling）为采样/解码参数集合。

- 转发层与 runtime MUST NOT 依赖其字段结构；MUST 原样透传给 backend。
- 全部可选；`config` 缺省时 backend MUST 使用模型默认值。
- 字段集合由 backend 实现决定，不构成本协议的契约面。下列为当前实现接受的字段，
  仅供参考（non-normative）：

  ```
  generate_audio, ls_mode, force_listen_count, max_new_speak_tokens_per_chunk,
  decode_mode, temperature, top_k, top_p, text_repetition_penalty,
  text_repetition_window_size, length_penalty, listen_prob_scale, listen_top_k,
  tts_temperature, chunk_ms, sample_rate
  ```

---

## 7. 状态与错误处理

本协议是 scheduler 与 backend 之间的**受信任内部协议**（见上层 §6.8）。`close` 是唯一的
正常终止操作（走 HTTP unary 控制通道）；除此之外，任何重复、乱序或非法的状态转移都视为
对端实现 bug，backend MUST **fail-fast**：立即终止 session 并关闭 WebSocket，MUST NOT
提供可恢复的错误分支或 reject 事件。

### 7.1 会话状态

一个 WebSocket 上的 session 有三个状态：`uninitialized` → `active` → `closed`。

### 7.2 操作合法性

| 操作 | uninitialized | active | closed |
|------|---------------|--------|--------|
| init | → active | **非法（fail-fast）** | — |
| push | **非法（fail-fast）** | 处理 | — |
| close（仅 HTTP unary） | 404（尚无此 session） | → closed，session 被遗忘 | 404（session 已遗忘） |
| 其它/未知 `type` | **非法（fail-fast）** | **非法（fail-fast）** | — |

### 7.3 close 的重复语义

- **close 后 backend 立即遗忘该 session**（释放资源并从会话表移除），不保留 `closed`
  状态记录。
- 因此对同一 `session_id` **再次** HTTP close 返回 **404**。backend **不区分**“从未存在”
  与“已关闭并遗忘”——两者都返回 404。
- **协议层不保证 close 幂等。** 重复 close 的安全性由调用方（runtime/scheduler）在自己
  一侧保证：记录 session 已关闭、不再重复发起 close。参考实现中 runtime 用本地 `_closed`
  标志短路，第二次 close 直接本地返回成功、不打到 backend（故正常使用下重复 close 无副作用）。
- WS 数据通道在 close 后即关闭，不再接收任何消息。

### 7.4 MUST fail-fast 的情形（非穷举）

backend 收到下列情形之一，MUST 立即终止 session 并关闭 WebSocket：

- 第一条消息不是 init。
- 已 `active` 后再次收到 init（重复初始化）。
- `uninitialized` 状态下收到 push（init 前 push）。
- 并发建立第二个 session（同一 backend 同时只允许一个 active session）。
- 未知或与当前模式不匹配的消息 `type` / 输入形状。
- 推理引擎自身异常。

### 7.5 fail-fast 的表现

终止时，backend：

- MUST 立即停止处理该 session 的输入，MUST NOT 再发送模型内容事件。
- SHOULD 在关闭前 best-effort 发送一个 `session.closed`（带 `reason`，致命错误可带
  `diagnostic.message`）作为诊断；发送失败则直接关闭连接。
- MUST 关闭 WebSocket。

接收方（scheduler/runtime）收到 `session.closed`（无论来自主动 close 还是 fail-fast）
MUST 视该 session 为终止，丢弃之；MUST NOT 尝试在同一连接上恢复（见上层 §9）。

---

## 8. Open Issues

以下为尚未最终确认的字段/编码设计点，可能在后续版本调整：

- **音频的字节序、采样率与声道。** §1.3 只把"base64 裸 float PCM、非容器、非整型"
  定为契约，未在协议层钉死字节序/采样率/声道。当前 Python 实现按本机字节序读写
  （在常见小端平台上即 float32 LE），采样率 16000 Hz、单声道，且不显式校验，依赖双方
  遵守约定。待决策：是否在协议层显式钉死这三项，并要求 backend 在入口校验、对违例
  fail-fast。

---

## 附录 A. 参考实现（non-normative）

当前 Python 实现的对应位置，仅供实现者对照，不构成契约：

- `py_backend/server.py` —— WS/HTTP 服务端，init/push/close 分发，事件发送。
- `py_backend/chat_util.py` —— turn_based 请求解析。
- `py_backend/media.py` —— 音频/JPEG 解码（§1.3 / §1.4 的字节级实现）。
- `py_backend/voice.py` —— 参考音频处理。
- `core/schemas/{common,duplex,metrics}.py` —— 字段含义参考。

实现现状偏差（non-normative）：

- §5.4 的 `vision_slices` / `vision_tokens`：当前模型层不返回图像计数，故该字段
  当前不被填充。
