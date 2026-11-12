# vLLM-Omni backend

The desktop Audio Full-Duplex and Omni Full-Duplex pages can connect to an
existing MiniCPM-o 4.5 vLLM-Omni server. Apple Silicon currently requires the
deployment profile in [vLLM-Metal #1235](https://github.com/vllm-project/vllm-metal/pull/1235);
follow its [launch instructions](https://github.com/vllm-project/vllm-metal/blob/101f4584108ba57fb4ba4dce4f798f6a2a04c6b1/docs/omni.md#minicpm-o-45).

In an environment with the gateway dependencies installed (see
[docker/Dockerfile.gateway](../../docker/Dockerfile.gateway)), run from this repository:

```bash
python gateway.py --host 127.0.0.1 --port 8006 --http \
  --vllm-omni-url ws://127.0.0.1:8097/v1/realtime
```

Open http://127.0.0.1:8006/audio_duplex for voice or
http://127.0.0.1:8006/omni for camera and voice. The browser connects directly
to the given WebSocket URL; use a browser-accessible `wss://` URL when serving
the page over HTTPS. Set `--vllm-omni-model` if the server uses a different
served model name. No Python workers or additional model copies are needed.
Session capacity is controlled by the Omni deployment.

This mode supports streaming audio/video, system prompts, a shared reference
voice, force-listen, pause/resume, and local recording downloads. Deployment
settings, independent TTS reference audio, and server-side save/share are not
exposed in this mode. Turn-based and mobile clients use the provider protocol.
Playback delay can absorb short delivery gaps; it does not increase inference
speed. Omit `--vllm-omni-url` to use the normal provider backend.
