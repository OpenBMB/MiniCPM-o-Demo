# Half-Duplex Audio FAQ

## What is Half-Duplex mode?

Half-Duplex mode supports voice conversation using VAD (Voice Activity Detection) to detect when the user finishes speaking, then triggers model inference. Only one party speaks at a time — the model listens while you speak, and you wait while the model replies. Interruption is not currently supported, but is theoretically feasible.

The model's response accuracy, speech naturalness, and typo rate are at a satisfactory level. Suitable for scenarios that require high performance.

## How do I adjust the VAD parameters?

- **Threshold**: Detection sensitivity for determining when the user starts speaking. Higher values make it less likely to be triggered by noise. Set 0.6-0.7 for quiet environments, and 0.8-0.95 for noisy environments.
- **Min Silence**: How long the silence should last before considering the user has finished speaking. Default is 800ms. If you tend to pause while speaking, increase it to 1200ms.

## Why is the model not responding?

- Check whether the service status in the top-right corner shows **Online**
- Confirm that microphone permissions have been granted to the browser
- The VAD threshold may be set too high, causing speech to go undetected
- Check the **State** in the left panel: it should cycle through `listening → processing → speaking`

## How do I select audio devices?

In the **Audio Devices** configuration area on the left panel:
- **Mic**: Select the input microphone
- **Speaker**: Select the audio output device
- Click **Refresh** to refresh the device list (use after plugging/unplugging devices)

## What is Length Penalty?

Length Penalty controls the length of model responses. Values greater than 1 encourage longer responses, while values less than 1 favor shorter responses. The default value of 1.1 produces moderately-length responses. At 1.1, the model exhibits better empathy.

## Can I replace the TTS voice from the web UI?

No. The system prompt and preset still affect voice expression and style, but the web UI does not provide a separate control for replacing the C++ TTS synthesis cache. To fully replace the C++ TTS voice, follow the [voice guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp-omni/%E6%8D%A2%E9%9F%B3%E8%89%B2%E6%8C%87%E5%8D%97.md) and replace `prompt_cache.gguf` plus the related assets yourself.

## What does Session Timeout mean?

**Session Timeout** sets the exclusive lock duration for the Worker. After this time, the session will automatically end and the Worker will be released for other users. Default is 300 seconds (5 minutes). This can be adjusted.

## How do I save conversation recordings?

- Make sure the **Rec** checkbox in the bottom control bar is checked
- After the conversation ends, click the **Download Rec** button to download the recording
- The recording format is stereo WAV: left channel is your microphone, right channel is the model's reply
