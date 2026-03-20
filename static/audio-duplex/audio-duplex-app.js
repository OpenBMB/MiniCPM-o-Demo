/**
 * audio-duplex-app.js — Audio Full-Duplex page entry (Layer 2 ES Module)
 *
 * Pure audio duplex — microphone or file input with waveform visualization.
 * Supports Live mode (mic only) and File mode (file-only / file+mic).
 */

// Layer 0: Pure logic
import { AudioDeviceSelector } from '../lib/audio-device-selector.js';
import { resampleAudio, arrayBufferToBase64, escapeHtml } from '../duplex/lib/duplex-utils.js';
import { DuplexSession } from '../duplex/lib/duplex-session.js';
import { SessionRecorder } from '../duplex/lib/session-recorder.js';
import { measureLUFS } from '../duplex/lib/lufs.js';
import { MixerController } from '../duplex/lib/mixer-controller.js';

// Layer 1: UI binding
import {
    MetricsPanel,
    getStatusPanelHTML,
    initHealthCheck,
    loadFrontendDefaults,
    setDefaultPauseBtnState,
    setDefaultForceListenBtnState,
    setDuplexButtonStates,
    setQueueButtonStates,
    wireDuplexControls,
    initDataTipTooltips,
    SettingsPersistence,
    getMixerPanelHTML,
} from '../duplex/ui/duplex-ui.js';
import { startDingDongLoop, playAlarmBell, playSessionChime } from '../duplex/lib/queue-chimes.js';
import { createTtsRefController } from '../duplex/ui/tts-ref-controller.js';
import { initRefAudio } from '../duplex/ui/ref-audio-init.js';

// ============================================================================
// Constants & State
// ============================================================================
const SAMPLE_RATE_IN = 16000;
const SAMPLE_RATE_OUT = 24000;
const CHUNK_MS = 1000;
const FILE_MAX_DURATION = 300; // 5 minutes

let currentMode = 'live';
let session = null;
let media = null;          // current MediaProvider

// Save & Share
const _saveShareUI = typeof SaveShareUI !== 'undefined'
    ? new SaveShareUI({ containerId: 'save-share-container', appType: 'audio_duplex' })
    : null;
let selectedFile = null;

// Microphone state (live mode only)
let audioCtxIn = null;
let audioStream = null;
let captureNodeLive = null;
let audioSource = null;
let analyserNode = null;
let waveformRunning = false;

// Session recording
let sessionRecorder = null;

// 排队倒计时（使用共享 CountdownTimer 模块）
import { CountdownTimer } from '../lib/countdown-timer.js';
let _queueCountdownLabel = null;
const _duplexCountdown = new CountdownTimer(({ remaining, position, queueLength }) => {
    if (_queueCountdownLabel) {
        _queueCountdownLabel.textContent = remaining > 0
            ? `Queue ${position}/${queueLength}, ~${remaining}s`
            : `Queue ${position}/${queueLength}, overtime +${Math.abs(remaining)}s`;
    }
});

function setStatusLamp(state) {
    const lamp = document.getElementById('statusLamp');
    if (!lamp) return;
    lamp.className = 'status-lamp';
    if (state === 'hidden') { lamp.classList.remove('visible'); return; }
    lamp.classList.add('visible', state);
    const labels = { live: 'LIVE', preparing: 'Preparing', stopped: 'Stopped' };
    lamp.querySelector('.label').textContent = labels[state] || state;
}

let _queuePhase = null; // null | 'queuing' | 'almost' | 'assigned'
let _stopDingDong = null;
let lastRecordingBlob = null;

// Mixer state
// MixerController instance (created after DOM setup, see bottom of file)
let mixerCtrl = null;

const metricsPanel = new MetricsPanel();

// ============================================================================
// Init: Status panel + health check + defaults + settings persistence
// ============================================================================
document.getElementById('panelStatus').innerHTML = getStatusPanelHTML();
document.getElementById('mixerPanel').innerHTML = getMixerPanelHTML();
let _stopHealthCheck = initHealthCheck('serviceStatus');
initDataTipTooltips();
drawIdleWaveform();

const settingsPersistence = new SettingsPersistence('audio_duplex_settings', [
    // Mode selector
    { type: 'mode', selector: '.mode-btn' },
    // File options
    { type: 'radio', name: 'fileAudioMode' },
    { id: 'padBeforeSec', type: 'number' },
    { id: 'padAfterSec', type: 'number' },
    // Session
    { id: 'playbackDelay', type: 'number' },
    { id: 'maxKvTokens', type: 'number' },
    { id: 'duplexLengthPenalty', type: 'number' },
    // System prompt
    { id: 'systemPrompt', type: 'textarea' },
    // TTS ref mode
    { type: 'radio', name: 'duplexTtsRefMode' },
    // Recording
    { id: 'recCheckbox', type: 'checkbox' },
    // Mixer
    { id: 'mxFileTarget', type: 'number' },
    { id: 'mxFileTrim', type: 'range' },
    { id: 'mxMicTarget', type: 'number' },
    { id: 'mxMicTrim', type: 'range' },
    { id: 'mxMonitor', type: 'range' },
]);

// Priority: HTML defaults → server defaults → localStorage → preset (highest)
loadFrontendDefaults().then(() => {
    settingsPersistence.restore();
    _duplexPreset.init();
});
window._settingsPersistence = settingsPersistence;
const adxDeviceSelector = new AudioDeviceSelector({
    micSelectEl: document.getElementById('adxMicDevice'),
    speakerSelectEl: document.getElementById('adxSpeakerDevice'),
    refreshBtnEl: document.getElementById('adxBtnRefreshDevices'),
    storagePrefix: 'audio_duplex',
    onSpeakerChange: () => {
        if (session && session.audioPlayer && session.audioPlayer._ctx) {
            adxDeviceSelector.applySinkId(session.audioPlayer._ctx);
        }
    },
});
adxDeviceSelector.init();

document.getElementById('btnResetSettings')?.addEventListener('click', () => {
    if (confirm('Reset all settings to defaults?')) {
        localStorage.removeItem('audio_duplex_preset');
        adxDeviceSelector.clearSaved();
        settingsPersistence.clear();
    }
});

// ============================================================================
// Preset Selector
// ============================================================================
// ============================================================================
// Ref Audio (init before preset so preset can update it)
// ============================================================================
const duplexTtsRef = createTtsRefController('duplex', () => refAudio.getBase64());
const refAudio = initRefAudio('refAudioPlayerDuplex', {
    onTtsHintUpdate: () => duplexTtsRef.updateHint(),
});
duplexTtsRef.init();

// ============================================================================
// Preset Selector
// ============================================================================
const _duplexPreset = new PresetSelector({
    container: document.getElementById('presetSelectorDuplex'),
    page: 'audio_duplex',
    detailsEl: document.getElementById('duplexSysPromptDetails'),
    onSelect: (preset, { audioLoaded } = {}) => {
        if (preset && preset.system_prompt) {
            document.getElementById('systemPrompt').value = preset.system_prompt;
            settingsPersistence.save();
        }
        if (audioLoaded && preset && preset.ref_audio && preset.ref_audio.data) {
            refAudio.setAudio(preset.ref_audio.data, preset.ref_audio.name, preset.ref_audio.duration);
        }
    },
    storageKey: 'audio_duplex_preset',
});

// ============================================================================
// Waveform Drawing
// ============================================================================
function drawIdleWaveform() {
    const canvas = document.getElementById('waveformCanvas');
    const container = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(0,255,136,0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
}

function startWaveformDrawing() {
    waveformRunning = true;
    document.getElementById('waveformPlaceholder').style.display = 'none';
    document.getElementById('waveformOverlay').classList.add('visible');
    requestAnimationFrame(drawWaveform);
}

function stopWaveformDrawing() {
    waveformRunning = false;
    document.getElementById('waveformOverlay').classList.remove('visible');
    drawIdleWaveform();
}

function drawWaveform() {
    if (!waveformRunning || !analyserNode) return;
    requestAnimationFrame(drawWaveform);
    const canvas = document.getElementById('waveformCanvas');
    const container = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    const bufLen = analyserNode.frequencyBinCount;
    const data = new Uint8Array(bufLen);
    analyserNode.getByteTimeDomainData(data);
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const sliceWidth = w / bufLen;
    let x = 0;
    for (let i = 0; i < bufLen; i++) {
        const v = data[i] / 128.0;
        const y = v * h / 2;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        x += sliceWidth;
    }
    ctx.stroke();
    ctx.strokeStyle = 'rgba(0,255,136,0.06)';
    ctx.lineWidth = 0.5;
    for (let gy = 0; gy < h; gy += h / 4) {
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    }
}

window.addEventListener('resize', () => { if (!waveformRunning) drawIdleWaveform(); });

// ============================================================================
// Chat Log UI
// ============================================================================
const chatLog = document.getElementById('chatLog');

function addSystemLog(text) {
    document.getElementById('chatEmpty').style.display = 'none';
    const el = document.createElement('div');
    el.className = 'conv-entry system';
    el.innerHTML = `<div class="conv-icon">&#x2699;</div><div class="conv-text">${escapeHtml(text)}</div>`;
    chatLog.appendChild(el);
    scrollChatLog();
}

function addUserLog(text) {
    document.getElementById('chatEmpty').style.display = 'none';
    const el = document.createElement('div');
    el.className = 'conv-entry user';
    el.innerHTML = `<div class="conv-icon">&#x1F464;</div><div class="conv-text"><span class="speaker user-tag">You:</span> ${escapeHtml(text)}</div>`;
    chatLog.appendChild(el);
    scrollChatLog();
}

function addAiLog(text) {
    document.getElementById('chatEmpty').style.display = 'none';
    const el = document.createElement('div');
    el.className = 'conv-entry ai';
    el.innerHTML = `<div class="conv-icon">&#x1F916;</div><div class="conv-text"><span class="speaker ai">AI:</span> <span class="ai-text">${escapeHtml(text)}</span></div>`;
    chatLog.appendChild(el);
    scrollChatLog();
    return el;
}

function scrollChatLog() { chatLog.scrollTop = chatLog.scrollHeight; }

// ============================================================================
// FileAudioProvider — Audio-only file mode
// ============================================================================

class FileAudioProvider {
    /**
     * @param {File} file
     * @param {{audioMode: 'file'|'mixed', padBeforeSec: number, padAfterSec: number}} opts
     */
    constructor(file, opts = {}) {
        this._file = file;
        this._audioMode = opts.audioMode || 'file';
        this._padBefore = Math.max(0, Math.floor(opts.padBeforeSec ?? 0));
        this._padAfter = Math.max(0, Math.floor(opts.padAfterSec ?? 2));

        this._fileAudioChunks = [];   // decoded + normalized file audio
        this._mainChunks = 0;

        // Padded array (file-only mode)
        this._allAudio = [];

        // Phase tracking
        this._chunkIdx = 0;
        this._mainStart = 0;
        this._mainEnd = 0;
        this._grandTotal = 0;
        this._timer = null;

        // Mic pipeline (mixed mode — AudioWorklet graph)
        this._micStream = null;
        this._micCtx = null;
        this._micSource = null;
        this._captureNode = null;
        this._micGainNode = null;
        this._fileGainNode = null;
        this._monitorGainNode = null;
        this._micAnalyserNode = null;
        this._mixAnalyserNode = null;
        this._fileSrcNode = null;
        this._graphConnected = false;

        // Playback element
        this._audioEl = document.getElementById('fileAudioEl');
        this._objectUrl = null;

        this.onChunk = null;
        this.onEnd = null;
        this.running = false;
        this.paused = false;
        this._padBeforeTimer = null;
    }

    async start() {
        addSystemLog('Processing audio file...');

        // 1. Get duration
        const rawDuration = await this._getAudioDuration();
        const cappedDuration = Math.min(rawDuration, FILE_MAX_DURATION);
        this._mainChunks = Math.floor(cappedDuration);
        if (this._mainChunks === 0) throw new Error('Audio too short');
        if (rawDuration > FILE_MAX_DURATION) {
            addSystemLog(`Audio truncated: ${rawDuration.toFixed(1)}s → ${cappedDuration}s`);
        }

        // 2. Decode file audio and normalize
        await this._decodeFileAudio(cappedDuration);

        // 3. Setup mic (mixed mode)
        if (this._audioMode === 'mixed') {
            await this._setupMic();
        }

        // 4. Phase boundaries
        this._mainStart = this._padBefore;
        this._mainEnd = this._padBefore + this._mainChunks;
        this._grandTotal = this._padBefore + this._mainChunks + this._padAfter;

        const parts = [];
        if (this._padBefore > 0) parts.push(`${this._padBefore}s pad`);
        parts.push(`${this._mainChunks}s audio`);
        if (this._padAfter > 0) parts.push(`${this._padAfter}s pad`);
        addSystemLog(`Ready: [${parts.join(' + ')}] = ${this._grandTotal} chunks, mode=${this._audioMode}`);

        // 5. Prepare playback element (user hears the original file audio)
        this._objectUrl = URL.createObjectURL(this._file);
        this._audioEl.src = this._objectUrl;

        // 6. Start feeding
        this._chunkIdx = 0;
        this.running = true;

        if (this._audioMode === 'file') {
            this._buildPaddedArrays();
            this._feedNext();
        } else {
            // mixed: mic runs for entire padded duration; file buffer includes silence padding
            // During padding, file audio = zero → mix = mic only; during main, mix = mic + file
            this._startMainMicPhase();
        }
    }

    // ==================== File-only mode: unified timer ====================

    _buildPaddedArrays() {
        const silence = () => new Float32Array(SAMPLE_RATE_IN);
        this._allAudio = [
            ...Array.from({ length: this._padBefore }, silence),
            ...this._fileAudioChunks,
            ...Array.from({ length: this._padAfter }, silence),
        ];
    }

    _feedNext() {
        if (!this.running || this.paused) return;
        if (this._chunkIdx >= this._grandTotal) {
            this.running = false;
            this._audioEl.pause();
            if (this.onEnd) this.onEnd();
            return;
        }
        // Start playback when entering main phase
        if (this._chunkIdx === this._mainStart) {
            this._audioEl.play();
        }
        const t0 = performance.now();
        const audio = this._allAudio[this._chunkIdx];
        this._chunkIdx++;
        if (this.onChunk) this.onChunk({ audio });
        const elapsed = performance.now() - t0;
        this._timer = setTimeout(() => this._feedNext(), Math.max(0, CHUNK_MS - elapsed));
    }

    // ==================== Mixed mode: phased feeding ====================

    async _setupMic() {
        const _micId = adxDeviceSelector.getSelectedMicId();
        this._micStream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, ...(_micId ? { deviceId: { exact: _micId } } : {}) },
            video: false,
        });

        this._micCtx = new AudioContext({ sampleRate: SAMPLE_RATE_IN });
        if (this._micCtx.state === 'suspended') await this._micCtx.resume();

        await this._micCtx.audioWorklet.addModule('/static/duplex/lib/capture-processor.js');

        this._micSource = this._micCtx.createMediaStreamSource(this._micStream);

        // LUFS-based gain model: effective = auto + trim
        const micTarget = parseFloat(document.getElementById('mxMicTarget')?.value) || -23;
        const micAutoGainDb = micTarget - (mixerCtrl?.micMeasuredLUFS ?? -23);
        const micTrimDb = parseInt(document.getElementById('mxMicTrim')?.value) || 0;
        const fileTrimDb = parseInt(document.getElementById('mxFileTrim')?.value) || 0;
        const monPct = parseInt(document.getElementById('mxMonitor')?.value) || 50;

        this._micGainNode = this._micCtx.createGain();
        this._micGainNode.gain.value = Math.pow(10, (micAutoGainDb + micTrimDb) / 20);

        this._fileGainNode = this._micCtx.createGain();
        this._fileGainNode.gain.value = Math.pow(10, fileTrimDb / 20); // LUFS norm already applied to PCM

        this._captureNode = new AudioWorkletNode(this._micCtx, 'capture-processor', {
            processorOptions: { chunkSize: SAMPLE_RATE_IN },
        });

        this._micAnalyserNode = this._micCtx.createAnalyser();
        this._micAnalyserNode.fftSize = 2048;

        this._mixAnalyserNode = this._micCtx.createAnalyser();
        this._mixAnalyserNode.fftSize = 2048;

        this._fileAnalyserNode = this._micCtx.createAnalyser();
        this._fileAnalyserNode.fftSize = 2048;

        this._monitorGainNode = this._micCtx.createGain();
        this._monitorGainNode.gain.value = monPct / 100;

        this._micChunkCount = 0;
        addSystemLog(`Mic ready: AudioWorklet @${SAMPLE_RATE_IN}Hz, micAutoGain=${micAutoGainDb.toFixed(1)}dB, micTrim=${micTrimDb}dB, fileTrim=${fileTrimDb}dB, monitor=${monPct}%`);
    }

    _connectMic() {
        // Build padded file AudioBuffer: [silence × padBefore] + file + [silence × padAfter]
        // Padding is transparent to the graph — mic captures the entire duration
        const silence = () => new Float32Array(SAMPLE_RATE_IN);
        const paddedChunks = [
            ...Array.from({ length: this._padBefore }, silence),
            ...this._fileAudioChunks,
            ...Array.from({ length: this._padAfter }, silence),
        ];
        const totalSamples = paddedChunks.reduce((a, c) => a + c.length, 0);
        const audioBuf = this._micCtx.createBuffer(1, totalSamples, SAMPLE_RATE_IN);
        const ch = audioBuf.getChannelData(0);
        let pos = 0;
        for (const chunk of paddedChunks) { ch.set(chunk, pos); pos += chunk.length; }
        this._fileSrcNode = this._micCtx.createBufferSource();
        this._fileSrcNode.buffer = audioBuf;

        // Connect graph:
        //   mic → micGain ──→ captureNode → mixAnalyser (waveform)
        //   file → fileGain ─┘
        //   fileGain → monitorGain → speaker (file only, no echo)
        //   micGain → micAnalyser (mic-only meter)
        this._micSource.connect(this._micGainNode);
        this._micGainNode.connect(this._captureNode);
        this._micGainNode.connect(this._micAnalyserNode);

        this._fileSrcNode.connect(this._fileGainNode);
        this._fileGainNode.connect(this._captureNode);
        this._fileGainNode.connect(this._fileAnalyserNode);
        // Speaker output uses HTML audio element (original quality); Web Audio monitor disconnected

        this._captureNode.connect(this._mixAnalyserNode);

        // Wire chunk handler
        this._captureNode.port.onmessage = (e) => {
            if (e.data.type === 'chunk') {
                this._handleMicChunk(e.data.audio);
            }
        };

        this._captureNode.port.postMessage({ command: 'start' });
        this._fileSrcNode.start();
        this._fileSrcNode.onended = () => addSystemLog('File audio in graph completed');

        this._graphConnected = true;

        // Waveform visualization uses the mix analyser
        analyserNode = this._mixAnalyserNode;
        startWaveformDrawing();
        addSystemLog('Mic connected — AudioWorklet graph mixing');
    }

    _disconnectMic() {
        stopWaveformDrawing();
        analyserNode = null;
        this._graphConnected = false;
        if (this._captureNode) {
            this._captureNode.port.postMessage({ command: 'stop' });
            try { this._captureNode.disconnect(); } catch (_) {}
        }
        if (this._fileSrcNode) {
            try { this._fileSrcNode.stop(); } catch (_) {}
            try { this._fileSrcNode.disconnect(); } catch (_) {}
            this._fileSrcNode = null;
        }
        if (this._fileGainNode) try { this._fileGainNode.disconnect(); } catch (_) {}
        if (this._micGainNode) try { this._micGainNode.disconnect(); } catch (_) {}
        if (this._micSource) try { this._micSource.disconnect(); } catch (_) {}
        if (this._monitorGainNode) try { this._monitorGainNode.disconnect(); } catch (_) {}
        if (this._micAnalyserNode) try { this._micAnalyserNode.disconnect(); } catch (_) {}
        if (this._mixAnalyserNode) try { this._mixAnalyserNode.disconnect(); } catch (_) {}
    }

    _startMainMicPhase() {
        // Use native HTML element for speaker output (original quality, not LUFS-normalized 16kHz)
        const monPct = parseInt(document.getElementById('mxMonitor')?.value) || 50;
        this._audioEl.muted = false;
        this._audioEl.volume = monPct / 100;
        // Delay file playback by padBefore — graph plays silence during leading padding
        if (this._padBefore > 0) {
            this._padBeforeTimer = setTimeout(() => {
                if (this.running) this._audioEl.play();
            }, this._padBefore * CHUNK_MS);
        } else {
            this._audioEl.play();
        }
        this._connectMic();
    }

    _handleMicChunk(mixedAudio) {
        // During padding: file audio = zero → mix = mic only
        // During main:    file audio = real → mix = mic + file
        if (this._chunkIdx >= this._grandTotal) {
            this._disconnectMic();
            this._audioEl.pause();
            addSystemLog(`Mixed mode done: ${this._micChunkCount} chunks via AudioWorklet`);
            this.running = false;
            if (this.onEnd) this.onEnd();
            return;
        }
        this._micChunkCount++;
        this._chunkIdx++;
        if (this.onChunk) this.onChunk({ audio: mixedAudio });
    }

    // ==================== Audio graph accessors ====================

    /** Expose AudioWorklet graph nodes for mixer panel and recording */
    get mixerNodes() {
        return {
            micGain: this._micGainNode,
            fileGain: this._fileGainNode,
            monitorGain: this._monitorGainNode,
            monitorEl: this._audioEl,
            micAnalyser: this._micAnalyserNode,
            fileAnalyser: this._fileAnalyserNode,
            mixAnalyser: this._mixAnalyserNode,
            captureNode: this._captureNode,
            connected: this._graphConnected,
        };
    }

    // ==================== Audio processing helpers ====================

    async _decodeFileAudio(cappedDuration) {
        try {
            const arrayBuffer = await this._file.arrayBuffer();
            const audioCtx = new AudioContext();
            const audioBuf = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
            const targetSamples = Math.ceil(cappedDuration * SAMPLE_RATE_IN);
            const offCtx = new OfflineAudioContext(1, targetSamples, SAMPLE_RATE_IN);
            const src = offCtx.createBufferSource();
            src.buffer = audioBuf;
            src.connect(offCtx.destination);
            src.start();
            const resampled = await offCtx.startRendering();
            const pcm = resampled.getChannelData(0);
            await audioCtx.close();

            // LUFS normalization: measure → compute gain → apply
            const fileTargetEl = document.getElementById('mxFileTarget');
            const targetLUFS = fileTargetEl ? parseFloat(fileTargetEl.value) || -33 : -33;
            const srcLUFS = measureLUFS(pcm, SAMPLE_RATE_IN);
            this.measuredLUFS = srcLUFS;

            let fileNormGain;
            if (this._audioMode === 'mixed' && isFinite(srcLUFS)) {
                const autoGainDb = targetLUFS - srcLUFS;
                fileNormGain = Math.pow(10, autoGainDb / 20);
                addSystemLog(`File audio: ${srcLUFS.toFixed(1)} LUFS → target ${targetLUFS} LUFS (auto ${autoGainDb.toFixed(1)} dB)`);
            } else {
                // file-only: normalize to -28 LUFS
                const foTarget = -28;
                const autoDb = isFinite(srcLUFS) ? foTarget - srcLUFS : 0;
                fileNormGain = Math.pow(10, autoDb / 20);
                addSystemLog(`File audio: ${isFinite(srcLUFS) ? srcLUFS.toFixed(1) : '—'} LUFS → ${foTarget} LUFS (gain ${autoDb.toFixed(1)} dB)`);
            }

            this._fileAudioChunks = [];
            for (let i = 0; i < pcm.length; i += SAMPLE_RATE_IN) {
                const chunk = pcm.slice(i, Math.min(i + SAMPLE_RATE_IN, pcm.length));
                for (let j = 0; j < chunk.length; j++) chunk[j] *= fileNormGain;
                this._fileAudioChunks.push(chunk);
            }

            // Update Mixer display
            const measEl = document.getElementById('mxFileMeasured');
            if (measEl) measEl.textContent = isFinite(srcLUFS) ? srcLUFS.toFixed(1) : '—';
            const autoEl = document.getElementById('mxFileAuto');
            if (autoEl && this._audioMode === 'mixed' && isFinite(srcLUFS)) {
                autoEl.textContent = (targetLUFS - srcLUFS).toFixed(1);
            }
        } catch (err) {
            addSystemLog(`Failed to decode audio: ${err.message}`);
            throw err;
        }
    }

    async _getAudioDuration() {
        return new Promise((resolve, reject) => {
            const audio = document.createElement('audio');
            audio.preload = 'metadata';
            const url = URL.createObjectURL(this._file);
            audio.src = url;
            audio.onloadedmetadata = () => {
                const d = audio.duration;
                URL.revokeObjectURL(url);
                resolve(d);
            };
            audio.onerror = () => {
                URL.revokeObjectURL(url);
                reject(new Error('Failed to load audio metadata'));
            };
        });
    }

    pause() {
        if (!this.running || this.paused) return;
        this.paused = true;
        // File-only: stop the timer
        if (this._timer) { clearTimeout(this._timer); this._timer = null; }
        // Cancel pending padBefore timer (delayed HTML play)
        if (this._padBeforeTimer) { clearTimeout(this._padBeforeTimer); this._padBeforeTimer = null; }
        // Mixed: suspend AudioContext (freezes entire graph — mic, file buffer, capture)
        if (this._micCtx && this._micCtx.state === 'running') {
            this._micCtx.suspend();
        }
        // Pause speaker output
        if (!this._audioEl.paused) this._audioEl.pause();
        addSystemLog('File provider paused');
    }

    resume() {
        if (!this.running || !this.paused) return;
        this.paused = false;
        if (this._audioMode === 'file') {
            // File-only: restart the timer and HTML element
            if (this._chunkIdx >= this._mainStart && this._chunkIdx < this._grandTotal) {
                this._audioEl.play();
            }
            this._feedNext();
        } else {
            // Mixed: resume AudioContext (unfreezes entire graph)
            if (this._micCtx && this._micCtx.state === 'suspended') {
                this._micCtx.resume();
            }
            // Resume or schedule HTML audio playback
            if (this._chunkIdx < this._mainStart) {
                // Still in leading padding — schedule delayed play for remaining pad
                const remainingPad = this._mainStart - this._chunkIdx;
                this._padBeforeTimer = setTimeout(() => {
                    if (this.running && !this.paused) this._audioEl.play();
                }, remainingPad * CHUNK_MS);
            } else if (this._chunkIdx < this._mainEnd) {
                // In main phase — resume playback immediately
                this._audioEl.play();
            }
        }
        addSystemLog('File provider resumed');
    }

    stop() {
        this.running = false;
        this.paused = false;
        if (this._timer) { clearTimeout(this._timer); this._timer = null; }
        if (this._padBeforeTimer) { clearTimeout(this._padBeforeTimer); this._padBeforeTimer = null; }
        this._disconnectMic();
        if (this._micCtx) { this._micCtx.close().catch(() => {}); this._micCtx = null; }
        if (this._micStream) { this._micStream.getTracks().forEach(t => t.stop()); this._micStream = null; }
        this._captureNode = null;
        this._micGainNode = null;
        this._fileGainNode = null;
        this._monitorGainNode = null;
        this._micAnalyserNode = null;
        this._mixAnalyserNode = null;
        this._fileSrcNode = null;
        this._micSource = null;
        this._graphConnected = false;
        this._audioEl.muted = false;
        this._audioEl.volume = 1.0;
        this._audioEl.pause();
        this._audioEl.src = '';
        if (this._objectUrl) { URL.revokeObjectURL(this._objectUrl); this._objectUrl = null; }
    }
}

// ============================================================================
// Mode Switching
// ============================================================================
function setMode(mode) {
    if (session) return;
    currentMode = mode;
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.mode === mode));
    const isFile = mode === 'file';
    document.getElementById('fileChooser').classList.toggle('visible', isFile);
    document.getElementById('fileOptions').classList.toggle('visible', isFile);
}

function onFileAudioModeChange() {
    const audioMode = document.querySelector('input[name="fileAudioMode"]:checked')?.value;
    const showMix = audioMode === 'mixed';
    const btn = document.getElementById('btnMixerToggle');
    if (btn) btn.style.display = showMix ? '' : 'none';
    if (!showMix) mixerCtrl?.closeMixer();
}

function onFileSelected(input) {
    if (input.files.length > 0) {
        selectedFile = input.files[0];
        document.getElementById('fileName').textContent = selectedFile.name;
        const audio = document.createElement('audio');
        audio.preload = 'metadata';
        const url = URL.createObjectURL(selectedFile);
        audio.src = url;
        audio.onloadedmetadata = () => {
            const dur = audio.duration;
            const label = dur > FILE_MAX_DURATION
                ? `${dur.toFixed(1)}s (will truncate to ${FILE_MAX_DURATION}s)`
                : `${dur.toFixed(1)}s`;
            document.getElementById('fileDuration').textContent = label;
            URL.revokeObjectURL(url);
        };
        audio.onerror = () => { URL.revokeObjectURL(url); };
        // Measure file LUFS in background
        measureFileLUFS(selectedFile);
    }
}

async function measureFileLUFS(file) {
    try {
        const measEl = document.getElementById('mxFileMeasured');
        const autoEl = document.getElementById('mxFileAuto');
        if (measEl) measEl.textContent = '...';
        if (autoEl) autoEl.textContent = '...';

        const arrayBuffer = await file.arrayBuffer();
        const tmpCtx = new AudioContext();
        const decoded = await tmpCtx.decodeAudioData(arrayBuffer.slice(0));
        const cappedDuration = Math.min(decoded.duration, FILE_MAX_DURATION);
        const targetFrames = Math.ceil(cappedDuration * SAMPLE_RATE_IN);
        const offCtx = new OfflineAudioContext(1, targetFrames, SAMPLE_RATE_IN);
        const src = offCtx.createBufferSource();
        src.buffer = decoded;
        src.connect(offCtx.destination);
        src.start();
        const resampled = await offCtx.startRendering();
        const pcm = resampled.getChannelData(0);
        await tmpCtx.close();

        const srcLUFS = measureLUFS(pcm, SAMPLE_RATE_IN);
        const targetLUFS = parseFloat(document.getElementById('mxFileTarget')?.value) || -33;
        const autoGainDb = isFinite(srcLUFS) ? targetLUFS - srcLUFS : 0;

        if (measEl) measEl.textContent = isFinite(srcLUFS) ? srcLUFS.toFixed(1) : '—';
        if (autoEl) autoEl.textContent = isFinite(srcLUFS) ? autoGainDb.toFixed(1) : '—';

        addSystemLog(`File LUFS: ${isFinite(srcLUFS) ? srcLUFS.toFixed(1) : '—'} → auto ${autoGainDb.toFixed(1)} dB`);
    } catch (err) {
        console.warn('measureFileLUFS failed:', err);
    }
}

// ============================================================================
// Session Control
// ============================================================================
async function startSession() {
    if (session) return;

    if (currentMode === 'file') {
        if (!selectedFile) { alert('Please select an audio file first.'); return; }
        const audioMode = document.querySelector('input[name="fileAudioMode"]:checked')?.value || 'file';
        const _i = (id, def) => { const v = parseInt(document.getElementById(id).value, 10); return Number.isFinite(v) ? v : def; };
        media = new FileAudioProvider(selectedFile, {
            audioMode,
            padBeforeSec: _i('padBeforeSec', 0),
            padAfterSec: _i('padAfterSec', 2),
        });
    }

    session = new DuplexSession('adx', {
        getMaxKvTokens: () => parseInt(document.getElementById('maxKvTokens').value, 10) || 8192,
        getPlaybackDelayMs: () => parseInt(document.getElementById('playbackDelay').value, 10) || 200,
        outputSampleRate: SAMPLE_RATE_OUT,
    });

    setStatusLamp('preparing');
    document.getElementById('lampTimer').textContent = '';

    // Wire hooks
    session.onMetrics = (data) => metricsPanel.update(data);
    session.onSystemLog = addSystemLog;
    session.onSpeakStart = (text) => addAiLog(text);
    session.onSpeakUpdate = (el, text) => {
        const span = el.querySelector('.ai-text');
        if (span) span.textContent = text;
        scrollChatLog();
    };
    session.onSpeakEnd = () => scrollChatLog();
    session.onListenResult = (result) => { if (result.text) addUserLog(result.text); };
    session.onRunningChange = (running) => setDuplexButtonStates(running);
    session.onPauseStateChange = (state) => {
        setDefaultPauseBtnState(state);
        if (session && session.running) {
            if (state === 'active') setStatusLamp('live');
            else if (state === 'paused') setStatusLamp('preparing');
        }
        // Pause/resume media provider and recording to keep timeline aligned
        if (state === 'paused' || state === 'pausing') {
            if (media && media.pause) media.pause();
            if (sessionRecorder && sessionRecorder.pause) sessionRecorder.pause();
        } else if (state === 'active') {
            if (media && media.resume) media.resume();
            if (sessionRecorder && sessionRecorder.resume) sessionRecorder.resume();
        }
    };
    session.onQueueUpdate = (data) => {
        const lamp = document.getElementById('statusLamp');
        if (data) {
            if (_stopHealthCheck) { _stopHealthCheck(); _stopHealthCheck = null; }
            setStatusLamp('preparing');
            _queueCountdownLabel = lamp?.querySelector('.label');
            _duplexCountdown.update(data.estimated_wait_s, data.position, data.queue_length || '?');
            if (data.position === 1 && _queuePhase !== 'almost') {
                _queuePhase = 'almost';
                setQueueButtonStates('almost');
                if (!_stopDingDong) _stopDingDong = startDingDongLoop();
            } else if (data.position !== 1 && _queuePhase !== 'almost') {
                _queuePhase = 'queuing';
                setQueueButtonStates('queuing');
            }
        } else {
            _duplexCountdown.stop();
            if (_stopDingDong) { _stopDingDong(); _stopDingDong = null; }
            if (!_stopHealthCheck) { _stopHealthCheck = initHealthCheck('serviceStatus'); }
        }
    };
    session.onQueueDone = () => {
        _queuePhase = 'assigned';
        if (_stopDingDong) { _stopDingDong(); _stopDingDong = null; }
        setQueueButtonStates('assigned');
        playAlarmBell();
    };
    session.onPrepared = async () => {
        if (session.audioPlayer && session.audioPlayer._ctx) {
            adxDeviceSelector.applySinkId(session.audioPlayer._ctx);
        }
        await playSessionChime();
    };
    session.onForceListenChange = (active) => setDefaultForceListenBtnState(active);
    session.onCleanup = () => {
        _duplexCountdown.stop();
        if (_stopDingDong) { _stopDingDong(); _stopDingDong = null; }
        _queuePhase = null;
        setQueueButtonStates(null);
        setStatusLamp('stopped');
        // Finalize recording
        if (sessionRecorder && sessionRecorder.recording) {
            const result = sessionRecorder.stop();
            if (result.blob.size > 0) {
                lastRecordingBlob = result.blob;
                addSystemLog(`Recording: ${result.durationSec.toFixed(1)}s stereo WAV (${(result.blob.size / 1024).toFixed(0)} KB)`);
                const btn = document.getElementById('btnDownloadRec');
                if (btn) { btn.style.display = ''; btn.disabled = false; }
                if (_saveShareUI) _saveShareUI.setRecordingBlob(result.blob, 'wav');
            }
            sessionRecorder = null;
        }

        if (media) { media.stop(); media = null; }
        stopWaveformDrawing();
        mixerCtrl?.stopMixerMeters();
        if (captureNodeLive) {
            captureNodeLive.port.postMessage({ command: 'stop' });
            try { captureNodeLive.disconnect(); } catch (_) {}
            captureNodeLive = null;
        }
        if (audioSource) { audioSource.disconnect(); audioSource = null; }
        if (analyserNode) { analyserNode.disconnect(); analyserNode = null; }
        if (audioStream) { audioStream.getTracks().forEach(t => t.stop()); audioStream = null; }
        if (audioCtxIn) { audioCtxIn.close().catch(() => {}); audioCtxIn = null; }
        session = null;
        // Restart standalone mixer mic if mixer is still open
        const mixerPanel = document.getElementById('mixerPanel');
        if (mixerPanel && mixerPanel.style.display === 'block') {
            mixerCtrl?.startMixerMic();
            mixerCtrl?.startMixerMeters();
        }
    };

    // Recording setup
    const recEnabled = document.getElementById('recCheckbox')?.checked;
    if (recEnabled) {
        sessionRecorder = new SessionRecorder(SAMPLE_RATE_IN, SAMPLE_RATE_OUT);
        lastRecordingBlob = null;
        const btn = document.getElementById('btnDownloadRec');
        if (btn) { btn.style.display = 'none'; btn.disabled = true; }
    }

    // Pre-start UI
    metricsPanel.reset();
    document.getElementById('chatEmpty').style.display = 'none';
    addSystemLog('Connecting...' + (recEnabled ? ' (recording enabled)' : ''));

    // Build prepare payload
    const preparePayload = {
        config: { length_penalty: parseFloat(document.getElementById('duplexLengthPenalty').value) || 1.05 },
    };
    const refBase64 = refAudio.getBase64();
    if (refBase64) preparePayload.ref_audio_base64 = refBase64;
    const ttsRef = duplexTtsRef.getBase64();
    if (ttsRef && ttsRef !== refBase64) preparePayload.tts_ref_audio_base64 = ttsRef;

    try {
        // Wire AI audio recording hook
        if (sessionRecorder) {
            session.audioPlayer.onRawAudio = (samples, sr, ts) => {
                if (sessionRecorder) sessionRecorder.pushRight(samples, sr, ts);
            };
        }

        await session.start(
            document.getElementById('systemPrompt').value,
            preparePayload,
            currentMode === 'live' ? startMicrophone : async () => {
                media.onChunk = (chunk) => {
                    session.sendChunk({
                        type: 'audio_chunk',
                        audio_base64: arrayBufferToBase64(chunk.audio.buffer),
                    });
                    if (sessionRecorder) sessionRecorder.pushLeft(chunk.audio);
                };
                media.onEnd = () => {
                    addSystemLog('File playback completed (including padding). Auto-stopping session.');
                    stopSession();
                };
                await media.start();
                mixerCtrl?.stopMixerMic();
                mixerCtrl?.startMixerMeters();
            }
        );

        // Start recording after media is ready
        if (sessionRecorder) sessionRecorder.start();

        document.getElementById('chatSessionInfo').textContent = session.sessionId;
        metricsPanel.update({ type: 'state', sessionState: 'Active' });
        setStatusLamp('live');
        addSystemLog('Session active — speak now');

        if (_saveShareUI && session.recordingSessionId) _saveShareUI.setSessionId(session.recordingSessionId);
    } catch (e) {
        const isCancelled = e.message?.includes('cancelled');
        if (!isCancelled) addSystemLog(`Error: ${e.message}`);
        if (session) { try { session.cleanup(); } catch (_) {} }
        session = null;
        media = null;
        setStatusLamp(isCancelled ? 'stopped' : 'hidden');
    }
}

function pauseSession() { if (session) session.pauseToggle(); }
function stopSession() {
    if (!session) return;
    if (_queuePhase) { session.cancelQueue(); } else { session.stop(); }
    session = null;
}
function toggleForceListen() { if (session) session.toggleForceListen(); }

// ============================================================================
// Microphone (Live mode — with Waveform AnalyserNode)
// ============================================================================
async function startMicrophone() {
    audioCtxIn = new AudioContext({ sampleRate: SAMPLE_RATE_IN });
    if (audioCtxIn.state === 'suspended') await audioCtxIn.resume();

    await audioCtxIn.audioWorklet.addModule('/static/duplex/lib/capture-processor.js');

    const _micId = adxDeviceSelector.getSelectedMicId();
    audioStream = await navigator.mediaDevices.getUserMedia({
        audio: _micId ? { deviceId: { exact: _micId } } : true,
    });
    audioSource = audioCtxIn.createMediaStreamSource(audioStream);

    analyserNode = audioCtxIn.createAnalyser();
    analyserNode.fftSize = 2048;

    captureNodeLive = new AudioWorkletNode(audioCtxIn, 'capture-processor', {
        processorOptions: { chunkSize: SAMPLE_RATE_IN },
    });

    // Connect: mic → analyser (waveform) → captureNode (chunk accumulation)
    audioSource.connect(analyserNode);
    analyserNode.connect(captureNodeLive);

    captureNodeLive.port.postMessage({ command: 'start' });
    captureNodeLive.port.onmessage = (e) => {
        if (e.data.type === 'chunk') {
            if (!session || !session.running || session.paused) return;
            const chunk = e.data.audio;
            session.sendChunk({
                type: 'audio_chunk',
                audio_base64: arrayBufferToBase64(chunk.buffer),
            });
            if (sessionRecorder) sessionRecorder.pushLeft(chunk);
        }
    };

    startWaveformDrawing();
}

// ============================================================================
// Wire up event listeners
// ============================================================================
wireDuplexControls({
    onStart: startSession,
    onStop: stopSession,
    onPause: pauseSession,
    onForceListen: toggleForceListen,
});

// Mode buttons
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
});

// File input
document.getElementById('audioFileInput').addEventListener('change', function() {
    onFileSelected(this);
});

// File audio mode radios
document.querySelectorAll('input[name="fileAudioMode"]').forEach(radio => {
    radio.addEventListener('change', onFileAudioModeChange);
});

// Mic calibration is handled by MixerController

// TTS ref mode radios
document.querySelectorAll('input[name="duplexTtsRefMode"]').forEach(radio => {
    radio.addEventListener('change', () => duplexTtsRef.onModeChange());
});

// ============================================================================
// Mixer Controller (shared module)
// ============================================================================
mixerCtrl = new MixerController({
    sampleRate: SAMPLE_RATE_IN,
    addLog: addSystemLog,
    getMedia: () => media,
    monitorElId: 'fileAudioEl',
    isFileMode: () => currentMode === 'file',
    getFileChunks: () => media?._fileAudioChunks,
    getSelectedFile: () => selectedFile,
    getLastRecordingBlob: () => lastRecordingBlob,
    getFallbackNodes: () => ({
        micAnalyser: analyserNode,
        connected: !!captureNodeLive,
    }),
    getDownloadExt: () => 'wav',
    isMicActive: () => !!audioStream,
});

if (document.readyState !== 'loading') {
    mixerCtrl.init();
} else {
    document.addEventListener('DOMContentLoaded', () => mixerCtrl.init());
}

// Cleanup on page unload (release mic, WS, AudioContext)
window.addEventListener('beforeunload', () => {
    if (session?.running) session.stop();
});

/* ---------- end of file ---------- */
