# MiniCPM-o 4.5 文档

MiniCPM-o 4.5 PyTorch 简易演示系统是由模型训练团队官方提供的推理与演示系统，使用 PyTorch + CUDA 推理后端，以透明、简洁的方式全面展示 MiniCPM-o 4.5 的音视频全模态全双工能力。

系统支持四种交互模式（Turn-based Chat、Half-Duplex、Omnimodal Full-Duplex、Audio Full-Duplex），共享同一模型实例，毫秒级热切换。

## 快速开始

- [系统架构](architecture/index.md) — 整体架构、模式拓扑与请求流转
- [模型模块](model.md) — 多模态模型的内部结构与数据流
- [API 参考](api.md) — 接口定义与调用方式
- [配置与部署](deployment.md) — 环境要求、配置说明与部署步骤
