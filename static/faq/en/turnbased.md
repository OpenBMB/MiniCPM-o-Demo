# Turn-based Chat FAQ

## What is Turn-based Chat mode?

Turn-based Chat is the classic question-and-answer conversation mode. You can send text, audio input, or video input, and the model will generate text and voice responses. Suitable for offline testing and prompt debugging.

## What input types are supported?

- **Text**: Type your questions directly
- **Audio**: Upload audio files as input
- **Video**: Upload video files as input
- Multimodal mixed input is supported

## About System Prompt

The System Prompt mainly defines the assistant's role, behavior, and response style. In the current deployment, voice replies use the preset's default voice configuration rather than a voice-customization workflow.

In the configuration card at the top, expand the **System Prompt** section to edit it. The System Prompt is sent to the model at the beginning of each conversation to define the role and behavior.

- About Voice Replies

Some presets include a built-in audio item used to initialize the default voice reply. In normal use, you usually only need to edit the text instructions in the System Prompt.

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
