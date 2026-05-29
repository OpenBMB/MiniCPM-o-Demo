# Backend Server Protocol

scheduler/runtime 与远端 inference backend server 之间的下层网络协议。它**不是**公网
`/v1/realtime` 协议，也不描述 UI 事件——它描述调度层如何初始化一个有状态 backend
session、持续提交模型输入，并从 backend 读取模型输出、状态与指标。

新实现一侧（如 C++/llama-server backend）请按本目录三份文档实现。术语约定（backend vs
scheduler/runtime）见 [network.md §1](./network.md#1-terminology)。

## 文档结构

| 文档 | 内容 | 何时看 |
|------|------|--------|
| [network.md](./network.md) | **过程 / 原语 / 完成语义。** 四原语（init/push/pull/unary）、session 生命周期、输入与下行事件语义、背压、断线、fail-fast。规范层。 | 先读这份，理解协议形状与状态机。 |
| [schema.md](./schema.md) | **消息 schema / 字段 / 编码。** 消息封套、契约字段 vs 透传字段、音频/图像编码、各消息的字段表、metrics 字段。用 RFC 2119 MUST/SHOULD/MAY。 | 实现具体收发与编解码时对照。 |
| [sequences.md](./sequences.md) | **时序图 + 示例数据包。** full_duplex、turn_based 流式、turn_based 一次性三条交互的 mermaid 时序图与真实抓包示例。 | 想看一次完整交互长什么样时。 |

## 约定

- 规范强度用 MUST / MUST NOT / SHOULD / MAY（含义同 RFC 2119）。
- 标注 **(契约 / normative)** 的字段与编码是硬约束；**透传 / opaque** 字段转发层原样转发、
  不解析（见 [schema.md §1.2](./schema.md#12-字段分类)）。
- 尚未定稿的设计点集中在各文档的 **Open Issues** 节
  （[network.md §11](./network.md#11-open-issues)、[schema.md §8](./schema.md#8-open-issues)），
  实现可参考当前倾向，但不应视为稳定契约。
