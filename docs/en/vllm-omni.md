# vLLM-Omni backend

The desktop Audio Full-Duplex and Omni Full-Duplex pages can connect to an
existing MiniCPM-o 4.5 vLLM-Omni server. Apple Silicon currently requires the
deployment profile in [vLLM-Metal #1235](https://github.com/vllm-project/vllm-metal/pull/1235);
follow its [launch instructions](https://github.com/vllm-project/vllm-metal/blob/101f4584108ba57fb4ba4dce4f798f6a2a04c6b1/docs/omni.md#minicpm-o-45).

Voice interruption uses Omni's server VAD. Install `onnxruntime` in the serving
environment and cache its Silero model before starting the server:

```bash
hf download istupakov/silero-vad-onnx silero_vad.onnx \
  --revision 8b14476858ef240c50b3884bb38cc67290c1cc70
```

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
served model name. The Omni server manages models and session capacity.

Independent TTS reference audio and server-side save/share are unavailable.
Turn-based and mobile clients use the provider backend.
