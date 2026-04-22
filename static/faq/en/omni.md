# Omni Full-Duplex FAQ

## What is Omni Full-Duplex mode?

In Omni Full-Duplex mode, you can speak and show visuals at the same time. The model can simultaneously see, listen, and respond in real time without missing any information, while maintaining low-latency feedback. This is the most powerful interaction mode of MiniCPM-o 4.5.

## What devices are required?

- **Camera**: For real-time video input (built-in or external)
- **Microphone**: For voice input
- **Headphones (optional)**: To prevent the model's reply audio from being picked up by the microphone, causing echo

## Responses are too short?

Due to limitations in the model's training data, responses may be brief. **You can increase the Length Penalty parameter in the left sidebar to 1.3 to achieve longer responses and better empathy.** The default value of 1.05 produces shorter responses. However, a known issue is that at Length Penalty = 1.3, voice interruption may become difficult. This will be a focus of improvement in the next model version.

## Can the preset/system prompt affect voice tone and prosody?

Yes. The preset and system prompt text still influence response language, speaking style, rhythm, and prosody. This deployment uses preset/default reference audio, but does not expose a separate TTS reference-audio replacement control. To fully replace the C++ TTS synthesis voice, follow the [voice guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp-omni/%E6%8D%A2%E9%9F%B3%E8%89%B2%E6%8C%87%E5%8D%97.md) and replace `prompt_cache.gguf` plus the related assets yourself.

## Can I customize the system prompt?

Yes. Click the **Advanced** button under Preset System Prompt to edit the system prompt text. This changes the assistant's role, behavior, response style, and voice expression, but does not provide a separate TTS reference-audio replacement control.

## What should I do if it feels laggy?

- If deployed on previous-generation GPUs such as A100/4090, you can enable `torch.compile` acceleration. Refer to [torch.compile Acceleration](https://openbmb.github.io/minicpm-o-4_5-pytorch-simple-demo/site/zh/compile.html) for details. This can significantly improve perceived smoothness. The inference time for each duplex unit (1s) can be reduced by half to 0.5s, achieving zero lag.
- Check the server GPU load on the `/admin` page to view Worker status

## What is the difference between Live mode and File mode?

- **Live**: Real-time camera + microphone input
- **File**: Upload a pre-recorded video file as input, with options to select the audio source (video audio track, microphone, or a mix of both microphone and video audio track)

## What is Vision HD?

When Vision HD is enabled, each frame uses high-definition visual encoding (192 tok/frame vs. the default 64 tok/frame), suitable for scenarios that require text recognition or fine details. However, it adds approximately 100ms of latency.

## What does the MaxKV parameter mean?

MaxKV limits the KV cache size (i.e., the context window length). The default is 8192 tokens, which is the model's training length. Exceeding this length may cause issues. Changing this value is not recommended.

## Why is the video not displaying?

- Confirm that the browser has been granted **camera permissions**
- Some browsers (especially Firefox) require HTTPS — make sure you are accessing via HTTPS
- Click the flip button in the top-right corner of the video to try switching between front and rear cameras
- Check whether another application is occupying the camera

## How do I use fullscreen mode?

Click the **fullscreen button** in the bottom-right corner of the video to enter fullscreen mode. In fullscreen, you can view conversation content through the bottom subtitles. The height and opacity of the subtitles can be adjusted in the **Fullscreen Subtitle** configuration.

## What does the Delay parameter in the left sidebar do?

Playback delay buffer — higher values are **more stable** but increase latency. Default is 200ms. If you experience audio stuttering, try increasing it. Under good network conditions, it can be reduced to 100ms.
