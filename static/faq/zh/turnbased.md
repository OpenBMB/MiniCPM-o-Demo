# Turn-based Chat 常见问题

## 什么是 Turn-based Chat 模式？

Turn-based Chat 是经典的一问一答对话模式。你可以发送文本、音频输入、视频输入，模型会生成文本和语音回复。适合离线测试、调试提示词使用。

## 支持哪些输入类型？

- **纯文本**：直接输入文字提问
- **语音**：上传音频文件作为输入
- **视频**：上传视频文件作为输入
- 支持多模态混合输入

## 关于 System Prompt

MiniCPM-o 4.5 支持多模态 System Prompt。当前这套 preset / system prompt 仍然会影响回复文本风格，以及语音回复的表达方式、节奏感和韵律。当前部署默认使用预设/默认参考音频，但没有开放单独替换 TTS 参考音频的入口。若要真正替换 C++ 侧的 TTS 合成音色，需要参照[换音色指南](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp-omni/%E6%8D%A2%E9%9F%B3%E8%89%B2%E6%8C%87%E5%8D%97.md)自行替换 `prompt_cache.gguf` 和相关资源。

在顶部的配置卡片中，展开 **System Prompt** 区域即可编辑。System Prompt 会在每次对话开始时发送给模型，用于设定角色和行为。

- 关于语音风格

部分预设会包含内置音频内容，同时 System Prompt 文本也会影响语音回复效果。普通使用场景下，可以通过切换 preset 或调整 System Prompt 中的文字指令来改变风格；但当前 UI 不开放单独替换 TTS 参考音频。

## 模式切换（语音、视频理解、文本对话）

可以通过System Prompt的不同预设切换，也可以自定义系统提示词。

## 什么情况下开启语音回复？

对于口语对话，可以开启Voice Response开关；对于书面对话，例如离线视频分析、markdown格式的回复，关闭Voice Response开关。关闭Voice Response开关后，模型会生成纯文本回复。

## 开启流式回复会影响效果吗？

会有影响，如果认为流式效果不佳，可以尝试使用非流式生成，等待时间较高。

## 连接状态显示 Offline 怎么办？

- 确认服务是否已启动（`bash start_all.sh`）
- 检查 Gateway 和 Worker 进程是否正常运行
- 查看浏览器控制台是否有 WebSocket 错误
- 确认访问地址和端口是否正确

