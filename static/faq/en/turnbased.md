# Turn-based Chat FAQ

## What is Turn-based Chat mode?

Turn-based Chat is the classic question-and-answer conversation mode. You can send text, audio input, or video input, and the model will generate text and voice responses. Suitable for offline testing and prompt debugging.

## What input types are supported?

- **Text**: Type your questions directly
- **Audio**: Upload audio files as input
- **Video**: Upload video files as input
- Multimodal mixed input is supported

## About System Prompt

MiniCPM-o 4.5 supports multimodal system prompts. In the current deployment, the preset/system prompt still affects text style as well as voice expression, rhythm, and prosody. The deployment uses preset/default reference audio, but does not expose a separate TTS reference-audio replacement control. To fully replace the C++ TTS synthesis voice, follow the [voice guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp-omni/%E6%8D%A2%E9%9F%B3%E8%89%B2%E6%8C%87%E5%8D%97.md) and replace `prompt_cache.gguf` plus the related assets yourself.

In the configuration card at the top, expand the **System Prompt** section to edit it. The System Prompt is sent to the model at the beginning of each conversation to define the role and behavior.

- About Voice Style

Some presets include a built-in audio item, and the System Prompt text also affects the final voice reply. In normal use, you can change style by switching presets or editing the System Prompt text, but the UI does not expose a separate TTS reference-audio replacement control.

## Mode Switching (Voice, Video Understanding, Text Chat)

You can switch between different presets via the System Prompt, or customize the system prompt directly.

## When should I enable voice response?

For spoken conversations, enable the Voice Response toggle. For written conversations, such as offline video analysis or markdown-formatted responses, disable the Voice Response toggle. When Voice Response is disabled, the model will generate text-only responses.

## Does enabling streaming affect quality?

Yes, it can. If you find the streaming quality unsatisfactory, try using non-streaming generation instead, which has a longer wait time.

## What should I do if the connection status shows Offline?

- Confirm the service has been started (`bash start_all.sh`)
- Check that the Gateway and Worker processes are running normally
- Look for WebSocket errors in the browser console
- Verify the access address and port are correct
