# Audio Full-Duplex FAQ

## What is Audio Full-Duplex mode?

Audio Full-Duplex mode supports **real-time bidirectional audio-only conversation**. You and the model can speak simultaneously without blocking. It is similar to a real phone call experience, but without video.

> Note: This feature is currently experimental. The model may not immediately respond to new questions while it is speaking. Using headphones is recommended for the best experience. This will be improved in the next model version.

## Responses are too short?

Due to limitations in the model's training data, responses may be brief. **You can increase the Length Penalty parameter in the left sidebar to 1.3 to achieve longer responses and better empathy.** The default value of 1.05 produces shorter responses. However, a known issue is that at Length Penalty = 1.3, voice interruption may become difficult. This will be a focus of improvement in the next model version.

## What does Preset System Prompt control?

It mainly affects response language, assistant behavior, and conversational style. The current deployment uses a preset default voice configuration instead of separate voice customization controls.

## Can I customize the system prompt?

Yes. Click the **Advanced** button under Preset System Prompt to edit the system prompt text. This changes the assistant's role, behavior, and response style, but does not change the default voice configuration.


## What is the difference from Half-Duplex?

| Feature | Half-Duplex | Audio Full-Duplex |
|---------|------------|-------------------|
| Communication | Turn-based | Simultaneous speaking |
| Interruption | Must wait for model to finish | Can interrupt anytime |
| Latency | Lower | Slightly higher |
| Stability | More stable | Experimental |

## What is the difference between Live mode and File mode?

- **Live**: Real-time voice input using the microphone
- **File**: Upload an audio file as input, with the option to use only the file audio or mix it with the microphone

## Why are headphones recommended?

In full-duplex mode, the model's reply audio may be picked up by the microphone through the speakers, causing echo. Using headphones effectively avoids this issue and allows the model to hear your voice more accurately.

## What should I set the Delay to?

- **Default 200ms**: Suitable for most scenarios
- **50-100ms**: Low latency but may cause audio stuttering
- **300-500ms**: More stable, suitable for poor network conditions

## What does the Force Listen button do?

Clicking **Force Listen** forces the model into listening mode, interrupting its current response. Use this when you want to interrupt the model while it is speaking.

## What does the Pause button do?

Pauses the current session. No audio data is sent during the pause. You can resume the conversation afterward. Suitable for temporary interruptions.
