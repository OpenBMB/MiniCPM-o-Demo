/**
 * realtime-app.js — Realtime API page entry
 *
 * Uses RealtimeSession (OpenAI Realtime protocol) with:
 *   - Camera capture → video_frames
 *   - Mic capture → input_audio_buffer.append
 *   - Chat bubbles for model responses
 *   - Protocol Data Flow panel showing all events
 */

import { RealtimeSession } from '../duplex/lib/realtime-session.js';
import { resampleAudio as downsample, arrayBufferToBase64 } from '../duplex/lib/duplex-utils.js';

const SAMPLE_RATE_IN = 16000;
const CHUNK_MS = 1000;
const FRAME_W = 448;
const FRAME_H = 336;
const JPEG_QUALITY = 0.6;

// ── DOM refs ────────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

const btnStart = $('btnStart');
const btnStop = $('btnStop');
const btnPause = $('btnPause');
const btnForce = $('btnForce');
const btnClearLog = $('btnClearLog');

const videoEl = $('videoEl');
const frameCanvas = $('frameCanvas');
const videoPlaceholder = $('videoPlaceholder');
const waveformCanvas = $('waveformCanvas');
const chatArea = $('chatArea');
const flowLog = $('flowLog');

// ── State ───────────────────────────────────────────────────────────────────

let session = null;
let audioCtx = null;
let mediaStream = null;
let processorNode = null;
let analyserNode = null;
let chunkBuffer = [];
let waveformRunning = false;
let speakCount = 0;

// ── Session setup ───────────────────────────────────────────────────────────

function createSession() {
    const maxKv = parseInt($('cfgMaxKv').value) || 8192;
    const s = new RealtimeSession('rt', {
        getMaxKvTokens: () => parseInt($('cfgMaxKv').value) || 8192,
        getPlaybackDelayMs: () => 200,
        getStopOnSlidingWindow: () => false,
        getWsUrl: (sessionId) => {
            const proto = location.protocol === 'https:' ? 'wss' : 'ws';
            return `${proto}://${location.host}/v1/realtime?session_id=${sessionId}`;
        },
    });

    s.onSystemLog = (text) => addBubble(text, 'system');

    s.onSpeakStart = (text) => {
        speakCount++;
        $('statSpeaks').textContent = speakCount;
        const div = document.createElement('div');
        div.className = 'bubble ai';
        div.textContent = text;
        chatArea.appendChild(div);
        chatArea.scrollTop = chatArea.scrollHeight;
        return div;
    };

    s.onSpeakUpdate = (handle, text) => {
        if (handle) handle.textContent = text;
    };

    s.onSpeakEnd = () => {};

    s.onListenResult = () => {};

    s.onMetrics = (data) => {
        if (data.type === 'result') {
            $('statChunks').textContent = data.chunksSent || 0;
            if (data.kvCacheLength) {
                $('statKV').textContent = data.kvCacheLength;
                $('stKv').textContent = data.kvCacheLength.toLocaleString();
            }
            if (data.ttfsMs) $('statTTFS').textContent = Math.round(data.ttfsMs) + 'ms';
            if (data.latencyMs) $('stWall').textContent = Math.round(data.latencyMs) + 'ms';
            $('stChunks').textContent = data.chunksSent || 0;
            $('stSpeaks').textContent = speakCount;

            if (data.modelState) {
                const ms = $('modelState');
                ms.textContent = data.modelState;
                ms.className = 'model-state ' + (data.modelState === 'listening' ? 'listening' : 'speaking');
                $('stModel').textContent = data.modelState;
            }
            if (data.visionSlices !== undefined) {
                $('stVision').textContent = `${data.visionSlices || 0} slices / ${data.visionTokens || 0} tok`;
            }
        } else if (data.type === 'state') {
            if (data.sessionState) $('stSession').textContent = data.sessionState;
            if (data.sessionId) $('stSid').textContent = data.sessionId;
        }
    };

    s.onProtocolEvent = (entry) => {
        const div = document.createElement('div');
        const cls = entry.type.includes('listen') ? ' is-listen'
            : entry.type.includes('audio.delta') || entry.type.includes('speak') ? ' is-speak'
            : entry.type.includes('error') ? ' is-error' : '';
        div.className = `flow-entry${cls}`;

        const ts = new Date(entry.ts).toLocaleTimeString('en', { hour12: false, fractionalSecondDigits: 1 });
        const arrow = entry.dir === 'client' ? '→' : '←';

        div.innerHTML = `
            <span class="flow-ts">${ts}</span>
            <span class="flow-arrow ${entry.dir}">${arrow}</span>
            <span class="flow-type">${entry.type}</span>
            <span class="flow-summary">${entry.summary}</span>
        `;
        flowLog.appendChild(div);
        flowLog.scrollTop = flowLog.scrollHeight;
    };

    s.onRunningChange = (running) => {
        btnStart.disabled = running;
        btnStop.disabled = !running;
        btnPause.disabled = !running;
        btnForce.disabled = !running;
        if (!running) {
            $('stSession').textContent = 'stopped';
            stopMedia();
        }
    };

    s.onPauseStateChange = (state) => {
        btnPause.textContent = state === 'paused' ? 'Resume' : state === 'pausing' ? 'Pausing...' : 'Pause';
    };

    s.onForceListenChange = (active) => {
        btnForce.textContent = `Force Listen: ${active ? 'ON' : 'OFF'}`;
        btnForce.style.borderColor = active ? '#f59e0b' : '';
    };

    s.onPrepared = async () => {
        addBubble('Session created — model ready', 'system');
    };

    s.onCleanup = () => {
        stopMedia();
    };

    return s;
}

// ── Media capture ───────────────────────────────────────────────────────────

async function startMedia() {
    // Camera
    try {
        const camStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: FRAME_W }, height: { ideal: FRAME_H }, facingMode: 'user' }
        });
        videoEl.srcObject = camStream;
        videoPlaceholder.style.display = 'none';
        mediaStream = camStream;
        addBubble('Camera started → video_frames', 'system');
    } catch (e) {
        addBubble(`Camera unavailable: ${e.message}`, 'system');
    }

    // Mic
    try {
        const micStream = await navigator.mediaDevices.getUserMedia({
            audio: { sampleRate: SAMPLE_RATE_IN, channelCount: 1, echoCancellation: true, noiseSuppression: true }
        });
        if (mediaStream) {
            micStream.getAudioTracks().forEach(t => mediaStream.addTrack(t));
        } else {
            mediaStream = micStream;
        }

        audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE_IN });
        const source = audioCtx.createMediaStreamSource(micStream);

        // Analyser for waveform
        analyserNode = audioCtx.createAnalyser();
        analyserNode.fftSize = 2048;
        source.connect(analyserNode);

        // Processor for chunking
        processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
        chunkBuffer = [];

        processorNode.onaudioprocess = (e) => {
            if (!session || !session.running) return;
            const data = e.inputBuffer.getChannelData(0);
            chunkBuffer.push(new Float32Array(data));

            const totalSamples = chunkBuffer.reduce((s, b) => s + b.length, 0);
            if (totalSamples >= SAMPLE_RATE_IN * (CHUNK_MS / 1000)) {
                sendAudioChunk();
            }
        };

        source.connect(processorNode);
        processorNode.connect(audioCtx.destination);

        waveformRunning = true;
        drawWaveform();
        addBubble('Microphone started → input_audio_buffer.append', 'system');
    } catch (e) {
        addBubble(`Mic error: ${e.message}`, 'system');
    }
}

function stopMedia() {
    waveformRunning = false;
    if (processorNode) { processorNode.disconnect(); processorNode = null; }
    if (audioCtx) { audioCtx.close().catch(() => {}); audioCtx = null; }
    if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
    }
    videoEl.srcObject = null;
    videoPlaceholder.style.display = '';
    chunkBuffer = [];
}

function sendAudioChunk() {
    const total = chunkBuffer.reduce((s, b) => s + b.length, 0);
    const samples = SAMPLE_RATE_IN * (CHUNK_MS / 1000);
    const merged = new Float32Array(Math.min(total, samples));
    let offset = 0;
    for (const buf of chunkBuffer) {
        const copy = Math.min(buf.length, merged.length - offset);
        merged.set(buf.subarray(0, copy), offset);
        offset += copy;
        if (offset >= merged.length) break;
    }
    chunkBuffer = [];

    const audioB64 = arrayBufferToBase64(merged.buffer);

    const msg = {
        type: 'audio_chunk',
        audio_base64: audioB64,
    };

    // Capture video frame
    const frame = captureFrame();
    if (frame) {
        msg.frame_base64_list = [frame];
    }

    msg.max_slice_nums = parseInt($('cfgSlices').value) || 1;

    session.sendChunk(msg);
}

function captureFrame() {
    if (!videoEl.srcObject || videoEl.readyState < 2) return null;
    frameCanvas.width = FRAME_W;
    frameCanvas.height = FRAME_H;
    const ctx = frameCanvas.getContext('2d');
    ctx.drawImage(videoEl, 0, 0, FRAME_W, FRAME_H);
    const dataUrl = frameCanvas.toDataURL('image/jpeg', JPEG_QUALITY);
    return dataUrl.split(',')[1];
}

// ── Waveform ────────────────────────────────────────────────────────────────

function drawWaveform() {
    if (!waveformRunning || !analyserNode) return;
    requestAnimationFrame(drawWaveform);

    const canvas = waveformCanvas;
    const container = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    const h = 48;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const bufLen = analyserNode.frequencyBinCount;
    const data = new Float32Array(bufLen);
    analyserNode.getFloatTimeDomainData(data);

    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, w, h);
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = '#6366f1';
    ctx.beginPath();
    const sliceW = w / bufLen;
    let x = 0;
    for (let i = 0; i < bufLen; i++) {
        const y = (data[i] * 0.5 + 0.5) * h;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        x += sliceW;
    }
    ctx.stroke();
}

// ── Chat bubbles ────────────────────────────────────────────────────────────

function addBubble(text, type) {
    const div = document.createElement('div');
    div.className = `bubble ${type}`;
    div.textContent = text;
    chatArea.appendChild(div);
    chatArea.scrollTop = chatArea.scrollHeight;
}

// ── Button handlers ─────────────────────────────────────────────────────────

btnStart.onclick = async () => {
    btnStart.disabled = true;
    chatArea.innerHTML = '';
    flowLog.innerHTML = '';
    speakCount = 0;
    $('statChunks').textContent = '0';
    $('statSpeaks').textContent = '0';
    $('statKV').textContent = '0';
    $('statTTFS').textContent = '—';

    session = createSession();

    const prompt = $('cfgPrompt').value;
    const preparePayload = {
        deferred_finalize: true,
        max_slice_nums: parseInt($('cfgSlices').value) || 1,
    };

    try {
        await session.start(prompt, preparePayload, startMedia);
        $('stSession').textContent = 'active';
        $('serviceStatus').textContent = 'Connected';
        $('serviceStatus').style.background = '#22c55e';
    } catch (e) {
        addBubble(`Failed to start: ${e.message}`, 'system');
        btnStart.disabled = false;
        $('serviceStatus').textContent = 'Error';
        $('serviceStatus').style.background = '#ef4444';
    }
};

btnStop.onclick = () => {
    if (session) session.stop();
    $('serviceStatus').textContent = 'Disconnected';
    $('serviceStatus').style.background = '';
};

btnPause.onclick = () => {
    if (session) session.pauseToggle();
};

btnForce.onclick = () => {
    if (session) session.toggleForceListen();
};

btnClearLog.onclick = () => {
    flowLog.innerHTML = '';
};

// ── Health check ────────────────────────────────────────────────────────────

async function checkHealth() {
    try {
        const resp = await fetch('/health');
        const data = await resp.json();
        if (data.status === 'ok') {
            $('serviceStatus').textContent = 'Service OK';
            $('serviceStatus').style.background = '#22c55e';
        }
    } catch {
        $('serviceStatus').textContent = 'Offline';
        $('serviceStatus').style.background = '#ef4444';
    }
}

checkHealth();
setInterval(checkHealth, 30000);
