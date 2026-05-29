## 文档结构

| 文档 | 内容 | 何时看 |
|------|------|--------|
| [network.md](./network.md) | **过程 / 原语 / 完成语义。** 四原语（init/push/pull/unary）、session 生命周期、输入与下行事件语义、背压、断线、fail-fast。规范层。 | 先读这份，理解协议形状与状态机。 |
| [schema.md](./schema.md) | **消息 schema / 字段 / 编码。** 消息封套、契约字段 vs 透传字段、音频/图像编码、各消息的字段表、metrics 字段。用 RFC 2119 MUST/SHOULD/MAY。 | 实现具体收发与编解码时对照。 |
| [sequences.md](./sequences.md) | **时序图 + 示例数据包。** full_duplex、turn_based 流式、turn_based 一次性三条交互的 mermaid 时序图与真实抓包示例。 | 想看一次完整交互长什么样时。 |

