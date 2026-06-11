---
title: "前端模块概述"
---

前端采用纯 **HTML + JavaScript + CSS** 构建，无框架依赖，通过模块化设计组织代码。支持实时 WebSocket 通信、Web Audio API 音频处理和 MediaRecorder 视频录制。

## 模块结构

```
static/
├── index.html                          # 首页（模式选择 + 最近会话）
├── turnbased.html                      # 轮次对话页面
├── admin.html                          # 管理面板
├── session-viewer.html                 # 会话回放查看器
│
├── omni/                               # Omni 全双工页面
│   ├── omni.html / omni-app.js / omni.css
│
├── audio-duplex/                       # 音频全双工页面
│   ├── audio_duplex.html / audio-duplex-app.js / audio-duplex.css
│
├── duplex/                             # 双工共享库
│   ├── duplex-shared.css               #   共享样式
│   ├── lib/                            #   核心库（10+ 模块）
│   └── ui/                             #   UI 组件
│
├── shared/                             # 跨页面共享组件
│   ├── app-nav.js / preset-selector.js / save-share.js
│
├── lib/                                # 通用工具库
│   ├── chat-eta-estimator.js / countdown-timer.js
│
├── ref-audio-player.js                 # 参考音频播放器
├── system-content-editor.js            # 系统内容编辑器
└── user-content-editor.js              # 用户消息编辑器
```

## 子文档导航

| 文档 | 内容 |
|------|------|
| [页面与路由](./pages/) | 各页面功能详解、路由系统、Turn-based Chat 状态管理 |
| [音频处理](./audio/) | AudioWorklet 采集、AudioPlayer 播放、LUFS 测量、混音器 |
| [双工会话](./duplex-session/) | DuplexSession 类、WebSocket 协议、状态机、录制系统 |
| [UI 组件](./components/) | 共享组件库、内容编辑器、预设选择器、导航系统 |

## 页面路由

| 页面 | URL | 说明 |
|------|-----|------|
| 首页 | `/` | 模式选择卡片、最近会话列表 |
| 轮次对话 | `/turnbased` | Turn-based Chat 交互 |
| Omni 全双工 | `/omni` | 视觉 + 语音全双工 |
| 音频全双工 | `/audio_duplex` | 纯音频全双工 |
| 管理面板 | `/admin` | Worker 状态、队列管理、应用开关 |
| 会话回放 | `/s/{session_id}` | 会话录制回放 |
| API 文档 | `/docs` | FastAPI 自动生成 |

## 技术栈

- 无框架：纯 HTML + ES Module JavaScript
- 实时通信：WebSocket（全双工、流式对话）
- 音频处理：Web Audio API + AudioWorklet
- 视频采集：getUserMedia + Canvas
- 视频录制：MediaRecorder API
- 状态持久化：localStorage
