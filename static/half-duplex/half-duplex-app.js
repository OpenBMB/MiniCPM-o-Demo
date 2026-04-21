/**
 * half-duplex-app.js — Half-Duplex Audio page entry
 *
 * Mic audio → WebSocket → Server-side VAD → Model inference → Audio response
 * Uses duplex-shared.css design language.
 */

import { AudioDeviceSelector } from '../lib/audio-device-selector.js';
import { SessionRecorder } from '../duplex/lib/session-recorder.js';
import { initDataTipTooltips } from '../duplex/ui/duplex-ui.js';

const SAMPLE_RATE = 16000;
const SAMPLE_RATE_OUT = 24000;
const CHUNK_DURATION_S = 0.5;
const CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION_S;
const STORAGE_KEY = 'half_duplex_settings';

// DOM refs
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const btnResetSettings = document.getElementById('btnResetSettings');
const statusLamp = document.getElementById('statusLamp');
const lampTimer = document.getElementById('lampTimer');
const waveformOverlay = document.getElementById('waveformOverlay');
const waveformPlaceholder = document.getElementById('waveformPlaceholder');
const waveformCanvas = document.getElementById('waveformCanvas');
const chatLog = document.getElementById('chatLog');
const chatEmpty = document.getElementById('chatEmpty');
const chatSessionInfo = document.getElementById('chatSessionInfo');
const serviceStatus = document.getElementById('serviceStatus');
const stateValue = document.getElementById('stateValue');
const turnValue = document.getElementById('turnValue');
const remainingValue = document.getElementById('remainingValue');
const queueValue = document.getElementById('queueValue');

// State
let ws = null;
let audioCtx = null;
let captureNode = null;
let audioStream = null;
let audioSource = null;
let analyserNode = null;
let sessionStartTime = null;
let timerInterval = null;
let audioPlayer = null;
let aiSpeaking = false;
let waveformRunning = false;
let turnIndex = 0;

// Recording
let sessionRecorder = null;
let lastRecordingBlob = null;
const _saveShareUI = typeof SaveShareUI !== 'undefined'
    ? new SaveShareUI({ containerId: 'save-share-container', appType: 'half_duplex_audio' })
    : null;

// ============================================================
// Settings persistence (localStorage)
// ============================================================

const DEFAULTS = {
    vadThreshold: 0.8,
    vadMinSpeech: 128,
    vadMinSilence: 800,
    vadSpeechPad: 30,
    genMaxTokens: 256,
    genLengthPenalty: 1.1,
    genTemperature: 0.7,
    ttsEnabled: true,
    sessionTimeout: 300,
};

function saveSettings() {
    const data = {};
    for (const key of Object.keys(DEFAULTS)) {
        const el = document.getElementById(key);
        if (!el) continue;
        if (el.type === 'checkbox') data[key] = el.checked;
        else if (el.tagName === 'TEXTAREA') data[key] = el.value;
        else data[key] = parseFloat(el.value);
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

function loadSettings() {
    let data;
    try { data = JSON.parse(localStorage.getItem(STORAGE_KEY)); } catch (_) { /* empty */ }
    const vals = { ...DEFAULTS, ...(data || {}) };
    for (const [key, val] of Object.entries(vals)) {
        const el = document.getElementById(key);
        if (!el) continue;
        if (el.type === 'checkbox') el.checked = val;
        else el.value = val;
    }
    updateRangeDisplay();
}

function resetSettings() {
    localStorage.removeItem(STORAGE_KEY);
    loadSettings();
}

function getSettings() {
    return {
        vad: {
            threshold: parseFloat(document.getElementById('vadThreshold').value),
            min_speech_duration_ms: parseInt(document.getElementById('vadMinSpeech').value),
            min_silence_duration_ms: parseInt(document.getElementById('vadMinSilence').value),
            speech_pad_ms: parseInt(document.getElementById('vadSpeechPad').value),
        },
        generation: {
            max_new_tokens: parseInt(document.getElementById('genMaxTokens').value),
            length_penalty: parseFloat(document.getElementById('genLengthPenalty').value),
            temperature: parseFloat(document.getElementById('genTemperature').value),
        },
        tts: {
            enabled: document.getElementById('ttsEnabled').checked,
        },
        session: {
            timeout_s: parseInt(document.getElementById('sessionTimeout').value),
        },
    };
}

function updateRangeDisplay() {
    const el = document.getElementById('vadThreshold');
    const val = document.getElementById('vadThresholdVal');
    if (el && val) val.textContent = el.value;
}

// Auto-save on change
document.querySelectorAll('.panel-config input, .panel-sysconfig textarea').forEach(el => {
    el.addEventListener('input', () => {
        updateRangeDisplay();
        saveSettings();
    });
});

btnResetSettings.addEventListener('click', () => {
    if (confirm('Reset all settings to defaults?')) {
        localStorage.removeItem('half_duplex_preset');
        deviceSelector.clearSaved();
        resetSettings();
        deviceSelector.enumerate();
    }
});

// ============================================================
// Audio device selector (shared component)
// ============================================================

const deviceSelector = new AudioDeviceSelector({
    micSelectEl: document.getElementById('micDevice'),
    speakerSelectEl: document.getElementById('speakerDevice'),
    refreshBtnEl: document.getElementById('btnRefreshDevices'),
    storagePrefix: 'half_duplex',
});

// ============================================================
// System Content Editor (same schema as turn-based)
// ============================================================

/** system_content 列表，与 turnbased 使用相同 schema */
let _systemContentList = [
    { type: 'text', text: '模仿音频样本的音色并生成新的内容。' },
    { type: 'audio', data: null, name: '', duration: 0 },
    { type: 'text', text: '你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。' },
];

let _sceHdx = null;
if (typeof SystemContentEditor !== 'undefined') {
    _sceHdx = new SystemContentEditor(document.getElementById('sysContentEditorHdx'), {
        theme: 'light',
        onChange: (items) => {
            _systemContentList = items;
            saveSettings();
        },
    });
    _sceHdx.setItems(_systemContentList);
}

/** 加载默认参考音频并填充到 system_content 中无数据的 audio 项 */
async function _fetchDefaultRefAudio() {
    try {
        const resp = await fetch('/api/default_ref_audio');
        if (!resp.ok) return;
        const data = await resp.json();
        _systemContentList.forEach(item => {
            if (item.type === 'audio' && !item.data) {
                item.data = data.base64; item.name = data.name; item.duration = data.duration;
            }
        });
        if (_sceHdx) _sceHdx.setItems(_systemContentList);
    } catch (e) { console.warn('Failed to load default ref audio:', e); }
}

/** 应用 preset 到 SystemContentEditor（与 turnbased 相同逻辑） */
function _applyPreset(preset, { audioLoaded } = {}) {
    if (!preset || !preset.system_content) return;
    const items = preset.system_content.map(item => {
        if (item.type === 'audio') {
            return { type: 'audio', data: (audioLoaded ? item.data : null) || null, name: item.name || '', duration: item.duration || 0 };
        }
        return { type: 'text', text: item.text || '' };
    });
    _systemContentList = items;
    if (_sceHdx) _sceHdx.setItems(items);
    saveSettings();
}

const _hdxPreset = typeof PresetSelector !== 'undefined'
    ? new PresetSelector({
        container: document.getElementById('presetSelectorHdx'),
        page: 'half_duplex_audio',
        detailsEl: document.getElementById('hdxSysPromptDetails'),
        onSelect: (preset, opts) => _applyPreset(preset, opts),
        storageKey: 'half_duplex_preset',
    })
    : null;

// ============================================================
// Waveform visualization
// ============================================================

function drawIdleWaveform() {
    const canvas = waveformCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
    canvas.height = canvas.offsetHeight * (window.devicePixelRatio || 1);
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function startWaveformLoop() {
    if (waveformRunning || !analyserNode) return;
    waveformRunning = true;
    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = canvas.offsetHeight * dpr;

    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Float32Array(bufferLength);

    function draw() {
        if (!waveformRunning) return;
        requestAnimationFrame(draw);
        analyserNode.getFloatTimeDomainData(dataArray);

        const w = canvas.width, h = canvas.height;
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, w, h);

        ctx.lineWidth = 1.5 * dpr;
        ctx.strokeStyle = '#4ade80';
        ctx.beginPath();

        const sliceWidth = w / bufferLength;
        let x = 0;
        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i];
            const y = (v * 0.5 + 0.5) * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
            x += sliceWidth;
        }
        ctx.stroke();
    }
    draw();
}

function stopWaveformLoop() {
    waveformRunning = false;
}

// ============================================================
// Audio capture (AudioWorklet, 16kHz)
// ============================================================

async function startCapture() {
    const micId = deviceSelector.getSelectedMicId();
    const audioConstraints = {
        sampleRate: SAMPLE_RATE,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
    };
    if (micId) audioConstraints.deviceId = { exact: micId };
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
    audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
    audioSource = audioCtx.createMediaStreamSource(audioStream);

    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 2048;
    audioSource.connect(analyserNode);

    await audioCtx.audioWorklet.addModule('/static/duplex/lib/capture-processor.js');
    captureNode = new AudioWorkletNode(audioCtx, 'capture-processor', {
        processorOptions: { chunkSize: CHUNK_SIZE },
    });

    captureNode.port.onmessage = (e) => {
        if (e.data.type === 'chunk') {
            const float32 = e.data.audio;
            if (sessionRecorder) sessionRecorder.pushLeft(float32);
            if (ws && ws.readyState === WebSocket.OPEN && !aiSpeaking) {
                const b64 = float32ToBase64(float32);
                ws.send(JSON.stringify({ type: 'audio_chunk', audio_base64: b64 }));
            }
        }
    };

    audioSource.connect(captureNode);
    captureNode.port.postMessage({ command: 'start' });

    waveformPlaceholder.style.display = 'none';
    waveformCanvas.style.display = 'block';
    startWaveformLoop();
}

function stopCapture() {
    stopWaveformLoop();
    if (captureNode) { captureNode.port.postMessage({ command: 'stop' }); captureNode = null; }
    if (audioStream) { audioStream.getTracks().forEach(t => t.stop()); audioStream = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    analyserNode = null;
    audioSource = null;
    waveformPlaceholder.style.display = '';
    waveformCanvas.style.display = '';
}

function float32ToBase64(float32Array) {
    const bytes = new Uint8Array(float32Array.buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
}

// ============================================================
// AI speaking → listening resume (wait for audio to finish)
// ============================================================

function scheduleListeningResume() {
    if (!audioPlayer || !audioPlayer.ctx) {
        aiSpeaking = false;
        setLampState('live', 'Listening');
        updateState('Listening');
        waveformOverlay.classList.remove('visible');
        return;
    }
    const ctx = audioPlayer.ctx;
    const remainingAudio = Math.max(0, audioPlayer.nextTime - ctx.currentTime);
    const delayMs = remainingAudio * 1000 + 800;

    setLampState('generating', 'AI speaking');
    updateState('Playing');
    waveformOverlay.classList.add('visible');
    waveformOverlay.querySelector('span').textContent = 'AI responding...';

    setTimeout(() => {
        aiSpeaking = false;
        setLampState('live', 'Listening');
        updateState('Listening');
        waveformOverlay.classList.remove('visible');
    }, delayMs);
}

// ============================================================
// Audio playback (24kHz model output) — with stop-all support
// ============================================================

function initAudioPlayer() {
    audioPlayer = {
        ctx: new AudioContext({ sampleRate: 24000 }),
        nextTime: 0,
        activeSources: [],
    };
    deviceSelector.applySinkId(audioPlayer.ctx);
}

function playAudioChunk(base64Data) {
    if (!audioPlayer || !base64Data) return;
    const ctx = audioPlayer.ctx;
    const raw = atob(base64Data);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    const float32 = new Float32Array(bytes.buffer);

    if (sessionRecorder) sessionRecorder.pushRight(float32, SAMPLE_RATE_OUT, performance.now());

    const buffer = ctx.createBuffer(1, float32.length, 24000);
    buffer.getChannelData(0).set(float32);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    const now = ctx.currentTime;
    if (audioPlayer.nextTime < now) audioPlayer.nextTime = now + 0.05;
    source.start(audioPlayer.nextTime);
    audioPlayer.nextTime += buffer.duration;

    audioPlayer.activeSources.push(source);
    source.onended = () => {
        const idx = audioPlayer.activeSources.indexOf(source);
        if (idx !== -1) audioPlayer.activeSources.splice(idx, 1);
    };
}

function stopAllAudio() {
    if (!audioPlayer) return;
    for (const src of audioPlayer.activeSources) {
        try { src.stop(); } catch (_) { /* already stopped */ }
    }
    audioPlayer.activeSources = [];
    audioPlayer.nextTime = 0;
}

function resetAudioPlayer() {
    if (audioPlayer) audioPlayer.nextTime = 0;
}

// ============================================================
// WebSocket session
// ============================================================

function generateSessionId() {
    return 'hdx_' + Math.random().toString(36).substring(2, 10);
}

async function startSession() {
    const sessionId = generateSessionId();
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}/ws/half_duplex/${sessionId}`;

    turnIndex = 0;
    turnValue.textContent = '0';

    const recEnabled = document.getElementById('recCheckbox')?.checked;
    if (recEnabled) {
        sessionRecorder = new SessionRecorder(SAMPLE_RATE, SAMPLE_RATE_OUT);
        lastRecordingBlob = null;
        const btn = document.getElementById('btnDownloadRec');
        if (btn) { btn.style.display = 'none'; btn.disabled = true; }
    }

    btnStart.disabled = true;
    setLampState('preparing', 'Preparing');
    updateState('Connecting');

    ws = new WebSocket(url);
    ws.onopen = () => {
        const settings = getSettings();
        ws.send(JSON.stringify({
            type: 'prepare',
            system_content: _sceHdx ? _sceHdx.getItems() : _systemContentList,
            config: settings,
        }));
    };

    ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        handleMessage(msg);
    };

    ws.onclose = () => { endSession(); };
    ws.onerror = (err) => { console.error('WS error:', err); endSession(); };
}

function handleMessage(msg) {
    switch (msg.type) {
        case 'queued':
            queueValue.textContent = `#${msg.position}`;
            updateState('Queued');
            break;

        case 'queue_update':
            queueValue.textContent = `#${msg.position}`;
            break;

        case 'queue_done':
            queueValue.textContent = '—';
            break;

        case 'prepared':
            btnStart.style.display = 'none';
            btnStop.style.display = '';
            btnStop.disabled = false;
            chatSessionInfo.textContent = msg.recording_session_id || msg.session_id;
            if (_saveShareUI) _saveShareUI.setSessionId(msg.recording_session_id || msg.session_id);
            if (sessionRecorder) sessionRecorder.start();
            clearChat();
            setLampState('live', 'Listening');
            updateState('Listening');
            startTimer(msg.timeout_s || 300);
            startCapture();
            initAudioPlayer();
            break;

        case 'vad_state':
            if (msg.speaking) {
                setLampState('speaking', 'Speaking');
                updateState('Speaking');
                waveformOverlay.classList.add('visible');
                waveformOverlay.querySelector('span').textContent = 'Speaking...';
            } else {
                setLampState('live', 'Listening');
                updateState('Listening');
                waveformOverlay.classList.remove('visible');
            }
            break;

        case 'generating':
            aiSpeaking = true;
            stopAllAudio();
            setLampState('generating', 'Thinking');
            updateState('Generating');
            waveformOverlay.classList.add('visible');
            waveformOverlay.querySelector('span').textContent = `Generating (${msg.speech_duration_ms}ms speech)`;
            break;

        case 'chunk':
            if (msg.audio_data) playAudioChunk(msg.audio_data);
            if (msg.text_delta) appendAssistantDelta(msg.text_delta);
            break;

        case 'turn_done':
            turnIndex = msg.turn_index;
            turnValue.textContent = String(turnIndex);
            finalizeAssistantMessage();
            addSpeechIndicator(turnIndex);
            scheduleListeningResume();
            break;

        case 'timeout':
            addSystemMessage('到达最大聊天长度，请重启');
            updateState('Timeout');
            endSession();
            break;

        case 'error':
            console.error('Server error:', msg.error);
            updateState(`Error`);
            addSystemMessage(msg.error);
            break;
    }
}

function endSession() {
    aiSpeaking = false;
    stopCapture();
    stopAllAudio();
    stopTimer();

    if (sessionRecorder && sessionRecorder.recording) {
        const result = sessionRecorder.stop();
        if (result.blob.size > 0) {
            lastRecordingBlob = result.blob;
            addSystemMessage(`Recording: ${result.durationSec.toFixed(1)}s stereo WAV (${(result.blob.size / 1024).toFixed(0)} KB)`);
            const btn = document.getElementById('btnDownloadRec');
            if (btn) { btn.style.display = ''; btn.disabled = false; }
            if (_saveShareUI) _saveShareUI.setRecordingBlob(result.blob, 'wav');
        }
        sessionRecorder = null;
    }

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'stop' }));
        ws.close();
    }
    ws = null;
    btnStart.style.display = '';
    btnStart.disabled = false;
    btnStop.style.display = '';
    btnStop.disabled = true;
    setLampState('stopped', 'Stopped');
    updateState('Idle');
    queueValue.textContent = '—';
    remainingValue.textContent = '—';
    waveformOverlay.classList.remove('visible');
}

// ============================================================
// UI helpers
// ============================================================

function setLampState(state, label) {
    statusLamp.className = 'status-lamp visible ' + state;
    statusLamp.querySelector('.label').textContent = label;
}

function updateState(text) {
    stateValue.textContent = text;
    if (text === 'Listening') {
        stateValue.className = 'status-value listening';
    } else if (text === 'Speaking') {
        stateValue.className = 'status-value speaking';
    } else {
        stateValue.className = 'status-value';
    }
}

function startTimer(timeoutS) {
    sessionStartTime = Date.now();
    const maxMs = timeoutS * 1000;
    timerInterval = setInterval(() => {
        const elapsed = Date.now() - sessionStartTime;
        const remaining = Math.max(0, maxMs - elapsed);
        const min = Math.floor(remaining / 60000);
        const sec = Math.floor((remaining % 60000) / 1000);
        const timeStr = `${min}:${sec.toString().padStart(2, '0')}`;
        lampTimer.textContent = timeStr;
        remainingValue.textContent = timeStr;
        if (remaining <= 0) {
            stopTimer();
            lampTimer.textContent = 'Expired';
            remainingValue.textContent = 'Expired';
        }
    }, 1000);
}

function stopTimer() {
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
    lampTimer.textContent = '';
}

// ============================================================
// Chat log (duplex-style conversation entries)
// ============================================================

let _currentAssistantEntry = null;

function clearChat() {
    chatLog.innerHTML = '';
    _currentAssistantEntry = null;
}

function addSpeechIndicator(turnIdx) {
    const entry = document.createElement('div');
    entry.className = 'conv-entry user';
    entry.innerHTML = `
        <div class="conv-icon"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/></svg></div>
        <div class="conv-text"><span class="speaker user-tag">You</span>Voice input (turn ${turnIdx})</div>
    `;
    chatLog.appendChild(entry);
    chatLog.scrollTop = chatLog.scrollHeight;
}

function appendAssistantDelta(delta) {
    if (!_currentAssistantEntry) {
        _currentAssistantEntry = document.createElement('div');
        _currentAssistantEntry.className = 'conv-entry ai';
        _currentAssistantEntry.innerHTML = `
            <div class="conv-icon">AI</div>
            <div class="conv-text"><span class="speaker ai">AI</span></div>
        `;
        _currentAssistantEntry._textSpan = document.createElement('span');
        _currentAssistantEntry.querySelector('.conv-text').appendChild(_currentAssistantEntry._textSpan);
        chatLog.appendChild(_currentAssistantEntry);
    }
    _currentAssistantEntry._textSpan.textContent += delta;
    chatLog.scrollTop = chatLog.scrollHeight;
}

function finalizeAssistantMessage() {
    _currentAssistantEntry = null;
}

function addSystemMessage(text) {
    const entry = document.createElement('div');
    entry.className = 'conv-entry system';
    entry.innerHTML = `
        <div class="conv-icon">!</div>
        <div class="conv-text">${escapeHtml(text)}</div>
    `;
    chatLog.appendChild(entry);
    chatLog.scrollTop = chatLog.scrollHeight;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ============================================================
// Health check
// ============================================================

async function checkHealth() {
    try {
        const resp = await fetch('/health');
        if (resp.ok) {
            serviceStatus.textContent = 'Online';
            serviceStatus.classList.add('online');
        } else {
            serviceStatus.textContent = 'Error';
            serviceStatus.classList.remove('online');
        }
    } catch {
        serviceStatus.textContent = 'Offline';
        serviceStatus.classList.remove('online');
    }
}

// ============================================================
// Init
// ============================================================

btnStart.addEventListener('click', startSession);
btnStop.addEventListener('click', endSession);
document.getElementById('btnDownloadRec')?.addEventListener('click', () => {
    if (!lastRecordingBlob) return;
    const url = URL.createObjectURL(lastRecordingBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `half-duplex-${new Date().toISOString().slice(0, 19).replace(/:/g, '')}.wav`;
    a.click();
    URL.revokeObjectURL(url);
});

loadSettings();
drawIdleWaveform();
checkHealth();
setInterval(checkHealth, 15000);
deviceSelector.init();

_fetchDefaultRefAudio();
if (_hdxPreset) _hdxPreset.init();
initDataTipTooltips();
