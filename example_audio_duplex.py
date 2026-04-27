"""
MiniCPM-o Realtime API — Audio Duplex Example

纯粹根据协议文档编写，未参考任何实现代码。
用项目自带的音频文件做输入，连接后完成一次完整对话。

用法:
    pip install websockets
    python example_audio_duplex.py
"""

import asyncio
import base64
import json
import ssl
import struct
import wave

import websockets

# ── 配置 ──────────────────────────────────────────────────────────

HOST = "127.0.0.1:8060"
WS_URL = f"wss://{HOST}/v1/realtime?mode=audio"

# 自签名证书，跳过校验
SSL_CTX = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

REF_AUDIO_PATH = "assets/ref_audio/ref_minicpm_signature.wav"
USER_AUDIO_PATH = "tests/cases/common/user_audio/当出现植物大战僵尸的时候提醒我.wav"  # 3.7s 有语音内容
# USER_AUDIO_PATH = "tests/cases/common/user_audio/000_user_audio0.wav"          # 11.3s
OUTPUT_WAV_PATH = "example_output.wav"
USE_REF_AUDIO = False  # 带 ref_audio 时服务端处理耗时长，可能断连；先关闭

INPUT_RATE = 16000   # 上行 16kHz
OUTPUT_RATE = 24000  # 下行 24kHz
CHUNK_SAMPLES = 16000  # 1 秒 = 16000 samples


# ── 音频工具 ──────────────────────────────────────────────────────

def load_wav_as_float32_chunks(path: str, chunk_samples: int = CHUNK_SAMPLES) -> list[bytes]:
    """读取 WAV 文件，转为 float32 PCM，按 1 秒切片。"""
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1, "需要单声道"
        assert wf.getframerate() == INPUT_RATE, f"需要 {INPUT_RATE}Hz"
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    # int16 → float32
    n = len(raw) // sw
    if sw == 2:
        samples = struct.unpack(f"<{n}h", raw)
        floats = [s / 32768.0 for s in samples]
    else:
        raise ValueError(f"不支持的 sample width: {sw}")

    chunks = []
    for i in range(0, len(floats), chunk_samples):
        seg = floats[i : i + chunk_samples]
        if len(seg) < chunk_samples:
            seg += [0.0] * (chunk_samples - len(seg))
        chunks.append(struct.pack(f"<{chunk_samples}f", *seg))
    return chunks


def encode_pcm(pcm_bytes: bytes) -> str:
    return base64.b64encode(pcm_bytes).decode()


def load_wav_file_base64(path: str) -> str:
    """整个 WAV 文件 base64 编码（用于 ref_audio）。"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def save_output_wav(pcm_chunks: list[bytes], path: str):
    """把收到的 float32 PCM 拼接后存为 int16 WAV。"""
    raw = b"".join(pcm_chunks)
    n = len(raw) // 4
    floats = struct.unpack(f"<{n}f", raw)
    int16 = struct.pack(f"<{n}h", *[int(max(-1, min(1, s)) * 32767) for s in floats])
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(OUTPUT_RATE)
        wf.writeframes(int16)


# ── 主流程 ────────────────────────────────────────────────────────

async def main():
    # 准备音频
    print(f"加载用户音频: {USER_AUDIO_PATH}")
    audio_chunks = load_wav_as_float32_chunks(USER_AUDIO_PATH)
    print(f"  → {len(audio_chunks)} 个 1s chunk")

    ref_audio_b64 = None
    if USE_REF_AUDIO:
        print(f"加载参考音色: {REF_AUDIO_PATH}")
        ref_audio_b64 = load_wav_file_base64(REF_AUDIO_PATH)

    # 收集服务端返回
    received_texts: list[str] = []
    received_audio: list[bytes] = []

    print(f"\n连接 {WS_URL} ...")
    async with websockets.connect(WS_URL, ssl=SSL_CTX) as ws:
        print("✓ WebSocket 已连接")

        # ── Phase 1: 等待排队（可能跳过） ──

        # 文档说排队阶段客户端只被动接收，不发消息。
        # 但如果没有排队，不会收到任何 queue 事件。
        # ⚠️ 文档没有明确说"无排队时应直接发 session.update"，
        #    我这里先收一条消息看看是不是 queue 事件，超时就直接继续。
        queue_done = False
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                msg = json.loads(raw)
                t = msg["type"]
                if t == "session.queued":
                    print(f"  排队中: 第 {msg['position']} 位, 预计 {msg.get('estimated_wait_s', '?')}s")
                elif t == "session.queue_update":
                    print(f"  排队更新: 第 {msg['position']} 位")
                elif t == "session.queue_done":
                    print("  排队结束")
                    queue_done = True
                    break
                elif t == "error":
                    err = msg["error"]
                    print(f"  ✗ 错误: [{err['code']}] {err['message']}")
                    return
                else:
                    # 不是排队事件，说明没有排队，退出等待
                    print(f"  无排队（收到 {t}）")
                    break
        except asyncio.TimeoutError:
            print("  无排队事件（超时），直接继续")

        # ── Phase 2: 发 session.update → 收 session.created ──

        session_config = {
            "instructions": "你是一个友好的中文助手。请用简短的中文回复。",
        }
        if ref_audio_b64:
            session_config["ref_audio"] = ref_audio_b64
        await ws.send(json.dumps({
            "type": "session.update",
            "session": session_config,
        }))
        print("→ 已发送 session.update")

        # 等 session.created
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            msg = json.loads(raw)
            if msg["type"] == "session.created":
                print(f"✓ session.created: id={msg['session_id']}, prompt_tokens={msg.get('prompt_length')}")
                break
            elif msg["type"] == "error":
                print(f"✗ 错误: {msg['error']}")
                return

        # ── Phase 3: 全双工对话 ──
        # 上行：每秒发一个 audio chunk
        # 下行：同时接收 response.output_audio.delta / response.listen
        # 文档说：客户端始终在发 append，不管服务端是在听还是在说。

        stop = asyncio.Event()

        async def send_audio():
            """上行：逐个发送音频 chunk，发完后补发静音保持连接。"""
            silence = struct.pack(f"<{CHUNK_SAMPLES}f", *([0.0] * CHUNK_SAMPLES))
            for i, chunk in enumerate(audio_chunks):
                if stop.is_set():
                    return
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": encode_pcm(chunk),
                }))
                print(f"  → chunk {i+1}/{len(audio_chunks)}")
                await asyncio.sleep(1.0)

            # 用户音频发完，继续发静音等模型说完
            print("  → 用户音频发完，发送静音等待回复...")
            for _ in range(15):
                if stop.is_set():
                    return
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": encode_pcm(silence),
                }))
                await asyncio.sleep(1.0)

            # 关闭
            print("→ 发送 session.close")
            await ws.send(json.dumps({
                "type": "session.close",
                "reason": "user_stop"
            }))

        async def receive_events():
            """下行：处理服务端推送。"""
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    t = msg["type"]

                    if t == "response.output_audio.delta":
                        text = msg.get("text", "")
                        audio_b64 = msg.get("audio", "")
                        eot = msg.get("end_of_turn", False)
                        kv = msg.get("kv_cache_length", 0)

                        if text:
                            received_texts.append(text)
                            print(f"  ← 说: \"{text}\"  (kv={kv})", end="")
                            if eot:
                                print("  [回合结束]")
                            else:
                                print()
                        if audio_b64:
                            received_audio.append(base64.b64decode(audio_b64))

                    elif t == "response.listen":
                        kv = msg.get("kv_cache_length", 0)
                        print(f"  ← 听  (kv={kv})")

                    elif t == "session.go_away":
                        print(f"  ← go_away: {msg.get('reason')}, 剩余 {msg.get('time_left_ms')}ms")

                    elif t == "session.closed":
                        print(f"✓ 会话关闭: {msg['reason']}")
                        # ⚠️ 文档里客户端发 reason="user_stop"，
                        #    服务端回 reason="stopped"（不是 "user_stop"）
                        stop.set()
                        return

                    elif t == "error":
                        err = msg.get("error", {})
                        print(f"  ✗ 错误 [{err.get('type')}]: {err.get('code')} — {err.get('message')}")

            except websockets.ConnectionClosed:
                stop.set()

        await asyncio.gather(send_audio(), receive_events())

    # ── 保存结果 ──

    if received_audio:
        save_output_wav(received_audio, OUTPUT_WAV_PATH)
        print(f"\n音频已保存: {OUTPUT_WAV_PATH} ({len(received_audio)} 个 chunk)")

    full_text = "".join(received_texts)
    print(f"完整文本: {full_text or '（无）'}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
