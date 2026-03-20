/**
 * omni-app.js — Omni Full-Duplex page entry (Layer 2 ES Module)
 *
 * Imports Layer 0 (pure logic) and Layer 1 (UI binding),
 * wires everything together for the Omni duplex page.
 */

// Layer 0: Pure logic
import { AudioDeviceSelector } from '../lib/audio-device-selector.js';
import { resampleAudio as downsample, arrayBufferToBase64, escapeHtml } from '../duplex/lib/duplex-utils.js';
import { DuplexSession } from '../duplex/lib/duplex-session.js';
import { SessionVideoRecorder } from '../duplex/lib/session-video-recorder.js';
import { RecordingSettings } from '../duplex/lib/recording-settings.js';
import { measureLUFS } from '../duplex/lib/lufs.js';
import { MixerController } from '../duplex/lib/mixer-controller.js';

// Layer 1: UI binding
import {
    MetricsPanel,
    getStatusPanelHTML,
    initHealthCheck,
    loadFrontendDefaults,
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
// 不在前端做 resize — 直接发摄像头/视频原始分辨率，后端统一处理（scale_resolution=448）

let currentMode = 'live';
let session = null;
let media = null;

// Save & Share
const _saveShareUI = typeof SaveShareUI !== 'undefined'
    ? new SaveShareUI({ containerId: 'save-share-container', appType: 'omni_duplex' })
    : null;
let selectedFile = null;
let cameraPreview = null;

// Session recording
let sessionRecorder = null;
let lastRecordingBlob = null;
/** @type {RecordingSettings|null} */
let recordingSettings = null;

// 排队倒计时（使用共享 CountdownTimer 模块）
import { CountdownTimer } from '../lib/countdown-timer.js';
let _queueCountdownLabel = null;
const _omniCountdown = new CountdownTimer(({ remaining, position, queueLength }) => {
    if (_queueCountdownLabel) {
        _queueCountdownLabel.textContent = remaining > 0
            ? `Queue ${position}/${queueLength}, ~${remaining}s`
            : `Queue ${position}/${queueLength}, overtime +${Math.abs(remaining)}s`;
    }
});

let _queuePhase = null; // null | 'queuing' | 'almost' | 'assigned'
let _stopDingDong = null;

// Mixer state
// MixerController instance (created after DOM setup, see bottom of file)
let mixerCtrl = null;

const metricsPanel = new MetricsPanel();

// Mic waveform visualization
let _omniWaveformRunning = false;
let _omniAnalyserNode = null;

function _omniDrawWaveform() {
    if (!_omniWaveformRunning || !_omniAnalyserNode) return;
    requestAnimationFrame(_omniDrawWaveform);
    const canvas = document.getElementById('omniWaveformCanvas');
    if (!canvas) return;
    const container = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const bufLen = _omniAnalyserNode.frequencyBinCount;
    const data = new Float32Array(bufLen);
    _omniAnalyserNode.getFloatTimeDomainData(data);

    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, w, h);
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = '#4ade80';
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

function _omniStartWaveform(audioCtx, audioSource) {
    _omniAnalyserNode = audioCtx.createAnalyser();
    _omniAnalyserNode.fftSize = 2048;
    audioSource.connect(_omniAnalyserNode);
    _omniWaveformRunning = true;
    const ph = document.getElementById('omniWaveformPlaceholder');
    if (ph) ph.style.display = 'none';
    requestAnimationFrame(_omniDrawWaveform);
}

function _omniStopWaveform() {
    _omniWaveformRunning = false;
    _omniAnalyserNode = null;
    const canvas = document.getElementById('omniWaveformCanvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    const ph = document.getElementById('omniWaveformPlaceholder');
    if (ph) ph.style.display = '';
}

// ============================================================================
// Init: Status panel + health check + defaults + settings persistence
// ============================================================================
document.getElementById('panelStatus').innerHTML =
    `<details class="config-group"><summary>Metrics</summary><div class="cg-body">${getStatusPanelHTML()}</div></details>`;
document.getElementById('mixerPanel').innerHTML = getMixerPanelHTML();
initHealthCheck('serviceStatus');
initDataTipTooltips();

const settingsPersistence = new SettingsPersistence('omni_settings', [
    // Mode selector
    { type: 'mode', selector: '.mode-btn' },
    // File options
    { type: 'radio', name: 'fileAudioMode' },
    { id: 'filePlaybackVol', type: 'range' },
    { id: 'frameOffset', type: 'range' },
    { id: 'padBeforeSec', type: 'number' },
    { id: 'padAfterSec', type: 'number' },
    // Session
    { id: 'playbackDelay', type: 'number' },
    { id: 'maxKvTokens', type: 'number' },
    { id: 'omniLengthPenalty', type: 'number' },
    { id: 'visionHD', type: 'checkbox' },
    // Fullscreen subtitle
    { id: 'fsSubHeight', type: 'number' },
    { id: 'fsAlphaBottom', type: 'number' },
    { id: 'fsAlphaTop', type: 'number' },
    // System prompt
    { id: 'systemPrompt', type: 'textarea' },
    // TTS ref mode
    { type: 'radio', name: 'omniTtsRefMode' },
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
    _omniPreset.init();
});
window._settingsPersistence = settingsPersistence;
const omniDeviceSelector = new AudioDeviceSelector({
    micSelectEl: document.getElementById('omniMicDevice'),
    speakerSelectEl: document.getElementById('omniSpeakerDevice'),
    refreshBtnEl: document.getElementById('omniBtnRefreshDevices'),
    storagePrefix: 'omni',
    onSpeakerChange: () => {
        if (session && session.audioPlayer && session.audioPlayer._ctx) {
            omniDeviceSelector.applySinkId(session.audioPlayer._ctx);
        }
    },
});
omniDeviceSelector.init();

document.getElementById('btnResetSettings')?.addEventListener('click', () => {
    if (confirm('Reset all settings to defaults?')) {
        if (recordingSettings) recordingSettings.clearStorage();
        localStorage.removeItem('omni_preset');
        omniDeviceSelector.clearSaved();
        settingsPersistence.clear();
    }
});

// ============================================================================
// Ref Audio Management (init before preset so preset can update it)
// ============================================================================
const omniTtsRef = createTtsRefController('omni', () => refAudio.getBase64());
const refAudio = initRefAudio('omniRefAudioPlayer', {
    onTtsHintUpdate: () => omniTtsRef.updateHint(),
});
omniTtsRef.init();

// ============================================================================
// Preset Selector
// ============================================================================
const _omniPreset = new PresetSelector({
    container: document.getElementById('presetSelectorOmni'),
    page: 'omni',
    detailsEl: document.getElementById('omniSysPromptDetails'),
    onSelect: (preset, { audioLoaded } = {}) => {
        if (preset && preset.system_prompt) {
            document.getElementById('systemPrompt').value = preset.system_prompt;
            settingsPersistence.save();
        }
        if (audioLoaded && preset && preset.ref_audio && preset.ref_audio.data) {
            refAudio.setAudio(preset.ref_audio.data, preset.ref_audio.name, preset.ref_audio.duration);
        }
    },
    storageKey: 'omni_preset',
});

// ============================================================================
// MediaProvider Classes (Omni-specific)
// ============================================================================

class MediaProvider {
    constructor() {
        this.onChunk = null;
        this.onEnd = null;
        this.running = false;
    }
    async start() { throw new Error('Not implemented'); }
    stop() { this.running = false; }
    getVideoElement() { return null; }
}

let _globalCameraFront = false;
let _globalMirror = false;

class LiveMediaProvider extends MediaProvider {
    constructor() {
        super();
        this._audioStream = null;
        this._videoStream = null;
        this._audioCtx = null;
        this._captureNode = null;
        this._audioSource = null;
        this._canvas = document.getElementById('frameCanvas');
        this._ctx2d = this._canvas.getContext('2d');
        this._videoEl = document.getElementById('videoEl');
        this._useFront = _globalCameraFront;
        this._previewing = false;
        this._aborted = false;
    }

    async startPreview() {
        if (this._previewing || this.running) return;
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('浏览器不支持 getUserMedia，请使用 HTTPS 访问');
        }
        this._aborted = false;
        await this._openVideoStream(this._useFront);
        // Check if stopPreview was called while we were awaiting
        if (this._aborted) {
            if (this._videoStream) {
                this._videoStream.getTracks().forEach(t => t.stop());
                this._videoStream = null;
            }
            this._videoEl.srcObject = null;
            this._videoEl.style.display = 'none';
            return;
        }
        this._previewing = true;
        document.getElementById('camFlipBtn').classList.add('visible');
        document.getElementById('mirrorBtn').classList.add('visible');
        document.getElementById('mirrorBtn').classList.toggle('active', _globalMirror);
        document.getElementById('videoPlaceholder').style.display = 'none';
    }

    stopPreview() {
        this._aborted = true;
        if (this._videoStream) {
            this._videoStream.getTracks().forEach(t => t.stop());
            this._videoStream = null;
        }
        if (this._previewing) {
            this._videoEl.srcObject = null;
            this._videoEl.style.display = 'none';
            document.getElementById('camFlipBtn').classList.remove('visible');
            document.getElementById('mirrorBtn').classList.remove('visible');
            document.getElementById('videoPlaceholder').style.display = 'flex';
        }
        this._previewing = false;
    }

    async start() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('浏览器不支持 getUserMedia。请使用 HTTPS 访问。');
        }
        if (!this._videoStream) {
            await this._openVideoStream(this._useFront);
        }
        const _omniMicId = omniDeviceSelector.getSelectedMicId();
        this._audioStream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, ...(_omniMicId ? { deviceId: { exact: _omniMicId } } : {}) },
            video: false,
        });
        this._audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE_IN });
        if (this._audioCtx.state === 'suspended') await this._audioCtx.resume();
        await this._audioCtx.audioWorklet.addModule('/static/duplex/lib/capture-processor.js');
        this._connectAudioPipeline();
        this.running = true;
        this._previewing = false;
        document.getElementById('camFlipBtn').classList.add('visible');
        document.getElementById('mirrorBtn').classList.add('visible');
        document.getElementById('mirrorBtn').classList.toggle('active', _globalMirror);
        addSystemEntry(`Live: mic=AudioWorklet@${SAMPLE_RATE_IN}Hz, cam=${this._useFront ? 'front' : 'back'}${_globalMirror ? ' [mirror]' : ''}`);
    }

    async _openVideoStream(front) {
        const facing = front ? 'user' : 'environment';
        this._videoStream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: { facingMode: facing },
        });
        // If aborted while waiting for getUserMedia, stop immediately
        if (this._aborted) {
            this._videoStream.getTracks().forEach(t => t.stop());
            this._videoStream = null;
            return;
        }
        this._videoEl.srcObject = this._videoStream;
        this._videoEl.style.display = 'block';
        const displayFlip = front !== _globalMirror;
        this._videoEl.style.transform = displayFlip ? 'scaleX(-1)' : 'none';
    }

    _connectAudioPipeline() {
        if (this._captureNode) { this._captureNode.disconnect(); this._captureNode = null; }
        if (this._audioSource) { this._audioSource.disconnect(); this._audioSource = null; }
        this._audioSource = this._audioCtx.createMediaStreamSource(this._audioStream);

        this._captureNode = new AudioWorkletNode(this._audioCtx, 'capture-processor', {
            processorOptions: { chunkSize: SAMPLE_RATE_IN },
        });

        this._audioSource.connect(this._captureNode);
        this._captureNode.port.postMessage({ command: 'start' });

        _omniStartWaveform(this._audioCtx, this._audioSource);

        // Real-time audio output for recording (CaptureProcessor pass-through, zero delay)
        this._recDest = this._audioCtx.createMediaStreamDestination();
        this._captureNode.connect(this._recDest);

        this._captureNode.port.onmessage = (e) => {
            if (e.data.type === 'chunk') {
                if (!this.running) return;
                const frameB64 = this._captureFrame();
                if (this.onChunk) this.onChunk({ audio: e.data.audio, frameBase64: frameB64 });
            }
        };
    }

    /** Real-time audio output stream for recording (bypasses 1s CaptureProcessor chunking) */
    getAudioOutputStream() { return this._recDest?.stream || null; }

    async flipCamera() {
        const wantFront = !this._useFront;
        if (this._videoStream) {
            this._videoStream.getTracks().forEach(t => t.stop());
            this._videoStream = null;
        }
        try {
            await this._openVideoStream(wantFront);
        } catch (err) {
            addSystemEntry(`Switch failed: ${err.message}`);
            try { await this._openVideoStream(this._useFront); } catch (e2) {
                addSystemEntry(`Recovery failed: ${e2.message}`);
                return;
            }
            return;
        }
        this._useFront = wantFront;
        _globalCameraFront = wantFront;
        addSystemEntry(`Camera: ${wantFront ? '前置' : '后置'}`);
    }

    stop() {
        this.running = false;
        _omniStopWaveform();
        if (this._captureNode) {
            this._captureNode.port.postMessage({ command: 'stop' });
            try { this._captureNode.disconnect(); } catch (_) {}
            this._captureNode = null;
        }
        if (this._audioSource) { this._audioSource.disconnect(); this._audioSource = null; }
        if (this._audioCtx) { this._audioCtx.close().catch(() => {}); this._audioCtx = null; }
        if (this._audioStream) { this._audioStream.getTracks().forEach(t => t.stop()); this._audioStream = null; }
        if (this._videoStream) { this._videoStream.getTracks().forEach(t => t.stop()); this._videoStream = null; }
        this._videoEl.srcObject = null;
        this._videoEl.style.display = 'none';
        this._videoEl.style.transform = 'none';
        document.getElementById('camFlipBtn').classList.remove('visible');
        document.getElementById('mirrorBtn').classList.remove('visible');
    }

    _captureFrame() {
        const v = this._videoEl;
        if (!v.videoWidth) return null;
        const cw = v.videoWidth;
        const ch = v.videoHeight;
        this._canvas.width = cw;
        this._canvas.height = ch;
        if (_globalMirror) {
            this._ctx2d.save();
            this._ctx2d.translate(cw, 0);
            this._ctx2d.scale(-1, 1);
            this._ctx2d.drawImage(v, 0, 0, cw, ch);
            this._ctx2d.restore();
        } else {
            this._ctx2d.drawImage(v, 0, 0, cw, ch);
        }
        return this._canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
    }

    getVideoElement() { return this._videoEl; }
}

const FILE_MAX_DURATION = 120; // 2 minutes

class FileMediaProvider extends MediaProvider {
    /**
     * @param {File} file
     * @param {{audioMode: 'video'|'mic'|'mixed', frameOffset: number,
     *          padBeforeSec: number, padAfterSec: number}} opts
     */
    constructor(file, opts = {}) {
        super();
        this._file = file;
        this._audioMode = opts.audioMode || 'video';
        this._frameOffset = opts.frameOffset ?? 0.5;
        this._padBefore = Math.max(0, Math.floor(opts.padBeforeSec ?? 0));
        this._padAfter = Math.max(0, Math.floor(opts.padAfterSec ?? 2));

        // Pre-processed data (unpadded)
        this._videoAudioChunks = [];
        this._frames = [];
        this._mainChunks = 0; // video content chunk count

        // Padded arrays (video audio mode only)
        this._allAudio = [];
        this._allFrames = [];

        // Phase tracking
        this._chunkIdx = 0;
        this._mainStart = 0;  // first main chunk index
        this._mainEnd = 0;    // first trailing-pad chunk index
        this._grandTotal = 0; // total chunks including pads
        this._firstFrame = null;
        this._lastFrame = null;
        this._timer = null;

        // Mic pipeline (modes: mic, mixed — AudioWorklet graph)
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

        this._videoEl = document.getElementById('videoEl');
        this._canvas = document.getElementById('frameCanvas');
        this._ctx2d = this._canvas.getContext('2d');
        this._objectUrl = null;
        this._padBeforeTimer = null;
        this.paused = false;
    }

    async start() {
        addSystemEntry('Processing video file...');

        // 1. Get duration, cap at 2 minutes
        const rawDuration = await this._getVideoDuration();
        const cappedDuration = Math.min(rawDuration, FILE_MAX_DURATION);
        this._mainChunks = Math.floor(cappedDuration);
        if (this._mainChunks === 0) throw new Error('Video too short');
        if (rawDuration > FILE_MAX_DURATION) {
            addSystemEntry(`Video truncated: ${rawDuration.toFixed(1)}s → ${cappedDuration}s`);
        }

        // 2. Pre-extract frames (with offset k)
        this._frames = await this._extractFrames(this._mainChunks, this._frameOffset);
        this._firstFrame = this._frames[0] || null;
        this._lastFrame = this._frames[this._frames.length - 1] || null;

        // 3. Decode video audio (modes: video, mixed)
        if (this._audioMode === 'video' || this._audioMode === 'mixed') {
            await this._decodeVideoAudio(cappedDuration);
        }

        // 4. Setup mic but don't connect yet (modes: mic, mixed)
        if (this._audioMode === 'mic' || this._audioMode === 'mixed') {
            await this._setupMic();
        }

        // 5. Compute phase boundaries
        this._mainStart = this._padBefore;
        this._mainEnd = this._padBefore + this._mainChunks;
        this._grandTotal = this._padBefore + this._mainChunks + this._padAfter;

        const parts = [];
        if (this._padBefore > 0) parts.push(`${this._padBefore}s pad`);
        parts.push(`${this._mainChunks}s video`);
        if (this._padAfter > 0) parts.push(`${this._padAfter}s pad`);
        addSystemEntry(`Ready: [${parts.join(' + ')}] = ${this._grandTotal} chunks, audio=${this._audioMode}`);

        // 6. Prepare video element (don't play yet — playback timing controlled by phases)
        this._objectUrl = URL.createObjectURL(this._file);
        this._videoEl.autoplay = false;  // override HTML autoplay; we control play timing
        this._videoEl.src = this._objectUrl;
        this._videoEl.pause();           // ensure autoplay doesn't start
        this._videoEl.style.display = 'block';
        // mic-only: mute video; video-only/mixed: play original audio through speaker
        this._videoEl.muted = (this._audioMode === 'mic');
        if (this._audioMode !== 'mic') {
            const volPct = parseInt(document.getElementById('filePlaybackVol')?.value) ?? 30;
            this._videoEl.volume = volPct / 100;
        }

        // 7. Start chunk feeding
        this._chunkIdx = 0;
        this.running = true;

        if (this._audioMode === 'video') {
            this._buildPaddedArrays();
            this._feedNext();
        } else {
            // mic/mixed: mic runs for entire padded duration; file buffer + frames include padding
            // During padding, file audio = zero → mix = mic only; frames = first/last frame
            this._paddedFrames = [
                ...Array(this._padBefore).fill(this._firstFrame),
                ...this._frames,
                ...Array(this._padAfter).fill(this._lastFrame),
            ];
            this._startMainMicPhase();
        }
    }

    // ==================== Video audio mode: unified timer ====================

    _buildPaddedArrays() {
        const silence = () => new Float32Array(SAMPLE_RATE_IN);
        this._allAudio = [
            ...Array.from({ length: this._padBefore }, silence),
            ...this._videoAudioChunks,
            ...Array.from({ length: this._padAfter }, silence),
        ];
        this._allFrames = [
            ...Array(this._padBefore).fill(this._firstFrame),
            ...this._frames,
            ...Array(this._padAfter).fill(this._lastFrame),
        ];
    }

    _feedNext() {
        if (!this.running || this.paused) return;
        if (this._chunkIdx >= this._grandTotal) {
            this.running = false;
            if (this.onEnd) this.onEnd();
            return;
        }
        // Start video playback when entering main phase
        if (this._chunkIdx === this._mainStart) {
            this._videoEl.play();
        }
        const t0 = performance.now();
        const audio = this._allAudio[this._chunkIdx];
        const frame = this._allFrames[this._chunkIdx] || null;
        this._chunkIdx++;
        if (this.onChunk) this.onChunk({ audio, frameBase64: frame });
        const elapsed = performance.now() - t0;
        this._timer = setTimeout(() => this._feedNext(), Math.max(0, CHUNK_MS - elapsed));
    }

    // ==================== Mic/Mixed modes: phased feeding ====================

    async _setupMic() {
        const _omniMicId = omniDeviceSelector.getSelectedMicId();
        this._micStream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, ...(_omniMicId ? { deviceId: { exact: _omniMicId } } : {}) },
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
        this._fileGainNode.gain.value = Math.pow(10, fileTrimDb / 20);

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
        addSystemEntry(`Mic ready: AudioWorklet @${SAMPLE_RATE_IN}Hz, micAutoGain=${micAutoGainDb.toFixed(1)}dB, micTrim=${micTrimDb}dB, fileTrim=${fileTrimDb}dB, monitor=${monPct}%`);
    }

    _connectMic() {
        // mic → micGain → captureNode
        this._micSource.connect(this._micGainNode);
        this._micGainNode.connect(this._captureNode);
        this._micGainNode.connect(this._micAnalyserNode);

        _omniStartWaveform(this._micCtx, this._micSource);

        // mixed mode: connect padded video audio source [silence × padBefore] + video + [silence × padAfter]
        if (this._audioMode === 'mixed') {
            const silence = () => new Float32Array(SAMPLE_RATE_IN);
            const paddedChunks = [
                ...Array.from({ length: this._padBefore }, silence),
                ...this._videoAudioChunks,
                ...Array.from({ length: this._padAfter }, silence),
            ];
            const totalSamples = paddedChunks.reduce((a, c) => a + c.length, 0);
            const audioBuf = this._micCtx.createBuffer(1, totalSamples, SAMPLE_RATE_IN);
            const ch = audioBuf.getChannelData(0);
            let pos = 0;
            for (const chunk of paddedChunks) { ch.set(chunk, pos); pos += chunk.length; }
            this._fileSrcNode = this._micCtx.createBufferSource();
            this._fileSrcNode.buffer = audioBuf;

            this._fileSrcNode.connect(this._fileGainNode);
            this._fileGainNode.connect(this._captureNode);
            this._fileGainNode.connect(this._fileAnalyserNode);
            // Speaker output uses HTML video element (original quality); Web Audio monitor disconnected
            this._fileSrcNode.start();
            this._fileSrcNode.onended = () => addSystemEntry('Video audio in graph completed');
        }

        this._captureNode.connect(this._mixAnalyserNode);

        // Real-time audio output for recording (CaptureProcessor pass-through, zero delay)
        this._recDest = this._micCtx.createMediaStreamDestination();
        this._captureNode.connect(this._recDest);

        this._captureNode.port.onmessage = (e) => {
            if (e.data.type === 'chunk') {
                this._handleMicChunk(e.data.audio);
            }
        };
        this._captureNode.port.postMessage({ command: 'start' });
        this._graphConnected = true;
        addSystemEntry(`Mic connected — AudioWorklet graph (${this._audioMode})`);
    }

    _disconnectMic() {
        _omniStopWaveform();
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
        // Delay video playback by padBefore — graph plays silence during leading padding
        if (this._padBefore > 0) {
            this._padBeforeTimer = setTimeout(() => {
                if (this.running) this._videoEl.play();
            }, this._padBefore * CHUNK_MS);
        } else {
            this._videoEl.play();
        }
        this._connectMic();
    }

    _handleMicChunk(mixedAudio) {
        // During padding: file audio = zero → mix = mic only; frame = first/last
        // During main:    file audio = real → mix = mic + file; frame from video
        if (this._chunkIdx >= this._grandTotal) {
            this._disconnectMic();
            this._videoEl.pause();
            addSystemEntry(`Mic phase done: ${this._micChunkCount} chunks via AudioWorklet`);
            this.running = false;
            if (this.onEnd) this.onEnd();
            return;
        }
        const frame = this._paddedFrames ? this._paddedFrames[this._chunkIdx] : null;
        this._micChunkCount++;
        this._chunkIdx++;
        if (this.onChunk) this.onChunk({ audio: mixedAudio, frameBase64: frame });
    }

    // ==================== Recording audio accessors ====================

    /** Real-time audio output stream for recording (mic/mixed modes, zero delay) */
    getAudioOutputStream() { return this._recDest?.stream || null; }

    /** Concatenated padded audio buffer for recording (video-only mode) */
    getFullAudioBuffer() {
        if (!this._allAudio || this._allAudio.length === 0) return null;
        const total = this._allAudio.reduce((a, c) => a + c.length, 0);
        if (total === 0) return null;
        const buf = new Float32Array(total);
        let pos = 0;
        for (const chunk of this._allAudio) { buf.set(chunk, pos); pos += chunk.length; }
        return buf;
    }

    // ==================== Audio graph accessors ====================

    get mixerNodes() {
        return {
            micGain: this._micGainNode,
            fileGain: this._fileGainNode,
            monitorGain: this._monitorGainNode,
            monitorEl: this._videoEl,
            micAnalyser: this._micAnalyserNode,
            fileAnalyser: this._fileAnalyserNode,
            mixAnalyser: this._mixAnalyserNode,
            captureNode: this._captureNode,
            connected: this._graphConnected,
        };
    }

    // ==================== Audio processing helpers ====================

    async _decodeVideoAudio(cappedDuration) {
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

            // LUFS normalization
            const fileTargetEl = document.getElementById('mxFileTarget');
            const targetLUFS = fileTargetEl ? parseFloat(fileTargetEl.value) || -33 : -33;
            const srcLUFS = measureLUFS(pcm, SAMPLE_RATE_IN);
            this.measuredLUFS = srcLUFS;

            let videoNormGain;
            if (this._audioMode === 'mixed' && isFinite(srcLUFS)) {
                const autoGainDb = targetLUFS - srcLUFS;
                videoNormGain = Math.pow(10, autoGainDb / 20);
                addSystemEntry(`Video audio: ${srcLUFS.toFixed(1)} LUFS → target ${targetLUFS} LUFS (auto ${autoGainDb.toFixed(1)} dB)`);
            } else {
                const foTarget = -28;
                const autoDb = isFinite(srcLUFS) ? foTarget - srcLUFS : 0;
                videoNormGain = Math.pow(10, autoDb / 20);
                addSystemEntry(`Video audio: ${isFinite(srcLUFS) ? srcLUFS.toFixed(1) : '—'} LUFS → ${foTarget} LUFS (gain ${autoDb.toFixed(1)} dB)`);
            }

            this._videoAudioChunks = [];
            for (let i = 0; i < pcm.length; i += SAMPLE_RATE_IN) {
                const chunk = pcm.slice(i, Math.min(i + SAMPLE_RATE_IN, pcm.length));
                for (let j = 0; j < chunk.length; j++) chunk[j] *= videoNormGain;
                this._videoAudioChunks.push(chunk);
            }

            // Update Mixer display
            const measEl = document.getElementById('mxFileMeasured');
            if (measEl) measEl.textContent = isFinite(srcLUFS) ? srcLUFS.toFixed(1) : '—';
            const autoEl = document.getElementById('mxFileAuto');
            if (autoEl && this._audioMode === 'mixed' && isFinite(srcLUFS)) {
                autoEl.textContent = (targetLUFS - srcLUFS).toFixed(1);
            }
        } catch (err) {
            addSystemEntry(`No audio track in video, using silence: ${err.message}`);
            this._videoAudioChunks = [];
            for (let i = 0; i < this._mainChunks; i++) {
                this._videoAudioChunks.push(new Float32Array(SAMPLE_RATE_IN));
            }
        }
    }

    async _getVideoDuration() {
        return new Promise((resolve, reject) => {
            const video = document.createElement('video');
            video.preload = 'metadata';
            const url = URL.createObjectURL(this._file);
            video.src = url;
            video.onloadedmetadata = () => {
                const d = video.duration;
                URL.revokeObjectURL(url);
                resolve(d);
            };
            video.onerror = () => {
                URL.revokeObjectURL(url);
                reject(new Error('Failed to load video metadata'));
            };
        });
    }

    async _extractFrames(count, offset) {
        return new Promise((resolve) => {
            const video = document.createElement('video');
            video.muted = true; video.preload = 'auto';
            const url = URL.createObjectURL(this._file);
            video.src = url;
            const frames = [];
            let idx = 0;
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            video.onloadedmetadata = () => {
                const cw = video.videoWidth;
                const ch = video.videoHeight;
                canvas.width = cw;
                canvas.height = ch;
                const seekNext = () => {
                    const seekTime = idx + offset;
                    if (idx >= count || seekTime >= video.duration) {
                        URL.revokeObjectURL(url);
                        resolve(frames);
                        return;
                    }
                    video.currentTime = seekTime;
                };
                video.onseeked = () => {
                    ctx.drawImage(video, 0, 0, cw, ch);
                    frames.push(canvas.toDataURL('image/jpeg', 0.7).split(',')[1]);
                    idx++;
                    seekNext();
                };
                seekNext();
            };
            video.onerror = () => { URL.revokeObjectURL(url); resolve(frames); };
        });
    }

    pause() {
        if (!this.running || this.paused) return;
        this.paused = true;
        // Video-only: stop the timer
        if (this._timer) { clearTimeout(this._timer); this._timer = null; }
        // Cancel pending padBefore timer (delayed video play)
        if (this._padBeforeTimer) { clearTimeout(this._padBeforeTimer); this._padBeforeTimer = null; }
        // Mic/Mixed: suspend AudioContext (freezes entire graph — mic, file buffer, capture)
        if (this._micCtx && this._micCtx.state === 'running') {
            this._micCtx.suspend();
        }
        // Pause speaker output
        if (!this._videoEl.paused) this._videoEl.pause();
        addSystemEntry('File provider paused');
    }

    resume() {
        if (!this.running || !this.paused) return;
        this.paused = false;
        if (this._audioMode === 'video') {
            // Video-only: restart the timer and video element
            if (this._chunkIdx >= this._mainStart && this._chunkIdx < this._grandTotal) {
                this._videoEl.play();
            }
            this._feedNext();
        } else {
            // Mic/Mixed: resume AudioContext (unfreezes entire graph)
            if (this._micCtx && this._micCtx.state === 'suspended') {
                this._micCtx.resume();
            }
            // Resume or schedule video playback
            if (this._chunkIdx < this._mainStart) {
                // Still in leading padding — schedule delayed play for remaining pad
                const remainingPad = this._mainStart - this._chunkIdx;
                this._padBeforeTimer = setTimeout(() => {
                    if (this.running && !this.paused) this._videoEl.play();
                }, remainingPad * CHUNK_MS);
            } else if (this._chunkIdx < this._mainEnd) {
                // In main phase — resume playback immediately
                this._videoEl.play();
            }
        }
        addSystemEntry('File provider resumed');
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
        this._paddedFrames = null;
        this._videoEl.autoplay = true;   // restore for LiveMediaProvider camera preview
        this._videoEl.muted = false;
        this._videoEl.volume = 1.0;
        this._videoEl.pause();
        this._videoEl.src = '';
        this._videoEl.style.display = 'none';
        if (this._objectUrl) { URL.revokeObjectURL(this._objectUrl); this._objectUrl = null; }
    }
}

// ============================================================================
// Diagnostics
// ============================================================================
let _diagEvents = [];
const _debug = new URLSearchParams(location.search).has('debug');

function _sendDiagnostic(payload) {
    if (session && session.ws && session.ws.readyState === WebSocket.OPEN) {
        try {
            session.ws.send(JSON.stringify({
                type: 'client_diagnostic',
                ts: performance.now(),
                session_elapsed_s: session._sessionStartTime
                    ? ((performance.now() - session._sessionStartTime) / 1000) : 0,
                ...payload,
            }));
        } catch (_) {}
    }
}

function _sendSessionSummary() {
    if (_diagEvents.length === 0) return;
    const speakChunks = _diagEvents.filter(e => e.ev === 'chunk');
    const p = (arr) => {
        if (arr.length === 0) return { min: 0, max: 0, avg: 0 };
        const sorted = [...arr].sort((a, b) => a - b);
        return { min: sorted[0], max: sorted[sorted.length - 1],
            avg: Math.round(sorted.reduce((a, b) => a + b, 0) / sorted.length) };
    };
    _sendDiagnostic({
        event: 'session_summary',
        total_results: session ? session._resultCount : 0,
        total_speak_chunks: speakChunks.length,
        total_gaps: session ? session.audioPlayer.gapCount : 0,
        total_shift_ms: session ? session.audioPlayer.totalShiftMs : 0,
        model_stats: p(speakChunks.map(c => c.model_ms).filter(v => v > 0)),
        drift_stats: p(speakChunks.map(c => c.drift_ms).filter(v => v !== null)),
    });
}

// ============================================================================
// Conversation UI
// ============================================================================
const conversationLog = document.getElementById('conversationLog');
const convEmpty = document.getElementById('convEmpty');
const fsChatInner = document.getElementById('fsChatInner');
const FS_CHAT_MAX = 12;
let _fsSpeakEl = null;
let _fsSpeakMsgEl = null;

function _fsPushAiMsg(text) {
    const el = document.createElement('div');
    el.className = 'fs-chat-msg';
    el.innerHTML = '<span class="fs-msg-icon">\u{1F916}</span><span class="fs-msg-text"></span>';
    const textSpan = el.querySelector('.fs-msg-text');
    textSpan.textContent = text;
    fsChatInner.appendChild(el);
    while (fsChatInner.children.length > FS_CHAT_MAX) fsChatInner.removeChild(fsChatInner.firstChild);
    return textSpan;
}

function clearConversation() {
    conversationLog.innerHTML = '';
    convEmpty.style.display = 'flex';
    conversationLog.appendChild(convEmpty);
    fsChatInner.innerHTML = '';
    _fsSpeakEl = null; _fsSpeakMsgEl = null;
}

function addSystemEntry(text) {
    convEmpty.style.display = 'none';
    const el = document.createElement('div');
    el.className = 'conv-entry system';
    el.innerHTML = `<div class="conv-icon">&#x2699;</div><div class="conv-text">${escapeHtml(text)}</div>`;
    conversationLog.appendChild(el);
    conversationLog.scrollTop = conversationLog.scrollHeight;
}

function addSpeakEntry(text) {
    convEmpty.style.display = 'none';
    const el = document.createElement('div');
    el.className = 'conv-entry speak';
    const textEl = document.createElement('div');
    textEl.className = 'conv-text';
    textEl.textContent = text;
    el.innerHTML = `<div class="conv-icon">&#x1F916;</div>`;
    el.appendChild(textEl);
    conversationLog.appendChild(el);
    conversationLog.scrollTop = conversationLog.scrollHeight;
    _fsSpeakEl = _fsPushAiMsg(text);
    return textEl;
}

function updateFsSpeakText(text) { if (_fsSpeakEl) _fsSpeakEl.textContent = text; }
function finishFsSpeak() { _fsSpeakEl = null; _fsSpeakMsgEl = null; }

function updateTimeBadge(chunks) {
    const m = Math.floor(chunks / 60);
    const s = chunks % 60;
    document.getElementById('lampTimer').textContent = `${m}:${s.toString().padStart(2, '0')}`;
}

// ============================================================================
// Button State Management
// ============================================================================
function setButtonStates(running) {
    const start = document.getElementById('btnStart');
    const fsStart = document.getElementById('fsBtnStart');
    start.disabled = running;
    document.getElementById('btnStop').disabled = !running;
    document.getElementById('btnForceListen').disabled = !running;
    document.getElementById('fsBtnForceListen').disabled = !running;
    document.getElementById('btnHD').disabled = !running;
    document.getElementById('fsBtnHD').disabled = !running;
    if (!running) {
        start.textContent = 'Start';
        start.classList.remove('live');
        if (fsStart) { fsStart.textContent = 'Start'; fsStart.classList.remove('live'); }
        setForceListenBtnState(false);
        setHDBtnState(false);
        setPauseBtnState('active');
        document.getElementById('btnPause').disabled = true;
        document.getElementById('fsBtnPause').disabled = true;
    } else {
        start.textContent = '● Live';
        start.classList.add('live');
        if (fsStart) { fsStart.textContent = '● Live'; fsStart.classList.add('live'); }
        document.getElementById('btnPause').disabled = false;
        document.getElementById('fsBtnPause').disabled = false;
    }
    syncFullscreenButtons(running);
}

function setPauseBtnState(state) {
    const btn = document.getElementById('btnPause');
    const fsBtn = document.getElementById('fsBtnPause');
    switch (state) {
        case 'active': btn.textContent = 'Pause'; btn.disabled = false; fsBtn.textContent = 'Pause'; fsBtn.disabled = false; break;
        case 'pausing': btn.textContent = 'Pausing...'; btn.disabled = true; fsBtn.textContent = 'Pausing...'; fsBtn.disabled = true; break;
        case 'paused': btn.textContent = 'Resume'; btn.disabled = false; fsBtn.textContent = 'Resume'; fsBtn.disabled = false; break;
    }
}

function setForceListenBtnState(active) {
    const btn = document.getElementById('btnForceListen');
    const fsBtn = document.getElementById('fsBtnForceListen');
    btn.textContent = active ? 'Release' : 'Force Listen';
    btn.classList.toggle('force-listen-active', active);
    fsBtn.textContent = active ? 'Release' : 'Force Listen';
    fsBtn.classList.toggle('force-listen-active', active);
}

// ============================================================================
// HD Vision Toggle
// ============================================================================
let _hdToggleActive = false;

function toggleHD() {
    _hdToggleActive = !_hdToggleActive;
    setHDBtnState(_hdToggleActive);
}

function setHDBtnState(active) {
    _hdToggleActive = active;
    const btn = document.getElementById('btnHD');
    const fsBtn = document.getElementById('fsBtnHD');
    btn.classList.toggle('force-listen-active', active);
    fsBtn.classList.toggle('force-listen-active', active);
    const cb = document.getElementById('visionHD');
    if (cb) cb.checked = active;
    const hint = document.getElementById('visionHint');
    if (hint) hint.textContent = active ? '192 tok' : '64 tok';
}

function getEffectiveMaxSliceNums() {
    return _hdToggleActive ? 2 : 1;
}

document.getElementById('visionHD')?.addEventListener('change', function () {
    setHDBtnState(this.checked);
});

// ============================================================================
// Video / Camera / Fullscreen / Subtitle
// ============================================================================
function flipCamera() {
    if (media && media instanceof LiveMediaProvider) {
        addSystemEntry('Switching camera...');
        media.flipCamera().then(() => addSystemEntry('Camera switched \u2713')).catch(err => addSystemEntry(`Camera switch failed: ${err.message}`));
        return;
    }
    if (cameraPreview && cameraPreview._previewing) {
        cameraPreview.flipCamera().then(() => addSystemEntry(`Preview: ${cameraPreview._useFront ? '前置' : '后置'}`)).catch(err => addSystemEntry(`Preview switch failed: ${err.message}`));
    }
}

(function() {
    const btn = document.getElementById('camFlipBtn');
    let lastFlipTime = 0;
    function doFlip(e) { e.preventDefault(); const now = Date.now(); if (now - lastFlipTime < 500) return; lastFlipTime = now; flipCamera(); }
    btn.addEventListener('touchend', doFlip, { passive: false });
    btn.addEventListener('click', doFlip);
})();

function toggleMirror() {
    _globalMirror = !_globalMirror;
    document.getElementById('mirrorBtn').classList.toggle('active', _globalMirror);
    const videoEl = document.getElementById('videoEl');
    const provider = (media && media instanceof LiveMediaProvider) ? media : (cameraPreview && cameraPreview._previewing) ? cameraPreview : null;
    if (provider) {
        const displayFlip = provider._useFront !== _globalMirror;
        videoEl.style.transform = displayFlip ? 'scaleX(-1)' : 'none';
    }
    addSystemEntry(`Mirror: ${_globalMirror ? 'ON' : 'OFF'}`);
}

(function() {
    const btn = document.getElementById('mirrorBtn');
    let last = 0;
    function handler(e) { e.preventDefault(); if (Date.now() - last < 400) return; last = Date.now(); toggleMirror(); }
    btn.addEventListener('touchend', handler, { passive: false });
    btn.addEventListener('click', handler);
})();

function updateVideoBorderPosition() {
    const video = document.getElementById('videoEl');
    const container = document.getElementById('videoContainer');
    const overlay = document.getElementById('videoBorderOverlay');
    if (!video.videoWidth || !video.videoHeight) { overlay.style.display = 'none'; return; }
    const cW = container.clientWidth, cH = container.clientHeight;
    const vAR = video.videoWidth / video.videoHeight, cAR = cW / cH;
    let rW, rH, oX, oY;
    if (vAR > cAR) { rW = cW; rH = cW / vAR; oX = 0; oY = (cH - rH) / 2; }
    else { rH = cH; rW = cH * vAR; oX = (cW - rW) / 2; oY = 0; }
    overlay.style.left = oX + 'px'; overlay.style.top = oY + 'px';
    overlay.style.width = rW + 'px'; overlay.style.height = rH + 'px';
    overlay.style.display = 'block';
}
document.getElementById('videoEl').addEventListener('loadedmetadata', updateVideoBorderPosition);
window.addEventListener('resize', updateVideoBorderPosition);

function setStatusLamp(state) {
    const lamp = document.getElementById('statusLamp');
    const overlay = document.getElementById('videoBorderOverlay');
    lamp.className = 'status-lamp';
    if (state === 'hidden') { lamp.classList.remove('visible'); overlay.classList.remove('active'); return; }
    lamp.classList.add('visible', state);
    const labels = { live: 'LIVE', preparing: 'Preparing', stopped: 'Stopped' };
    lamp.querySelector('.label').textContent = labels[state] || state;
    if (state === 'live') { updateVideoBorderPosition(); overlay.classList.add('active'); }
    else { overlay.classList.remove('active'); }
}

let _isVideoFullscreen = false;
function toggleVideoFullscreen() {
    _isVideoFullscreen = !_isVideoFullscreen;
    document.body.classList.toggle('video-fullscreen', _isVideoFullscreen);
    const icon = document.getElementById('fullscreenIcon');
    icon.innerHTML = _isVideoFullscreen
        ? '<path d="M8 3v3a2 2 0 0 1-2 2H3"/><path d="M21 8h-3a2 2 0 0 1-2-2V3"/><path d="M3 16h3a2 2 0 0 1 2 2v3"/><path d="M16 21v-3a2 2 0 0 1 2-2h3"/>'
        : '<path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/><path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/>';
    requestAnimationFrame(updateVideoBorderPosition);
}

function updateFullscreenBtnVisibility(show) {
    document.getElementById('fullscreenBtn').classList.toggle('visible', show);
    if (!show && _isVideoFullscreen) toggleVideoFullscreen();
    updateSubtitleBtnVisibility(show);
}

function syncFullscreenButtons(running) {
    document.getElementById('fsBtnStart').disabled = running;
    document.getElementById('fsBtnStop').disabled = !running;
    document.getElementById('fsBtnForceListen').disabled = !running;
    document.getElementById('fsBtnHD').disabled = !running;
}

function syncFullscreenQueueButtons(phase) {
    const fsStart = document.getElementById('fsBtnStart');
    const fsStop = document.getElementById('fsBtnStop');
    const fsFl = document.getElementById('fsBtnForceListen');
    const fsHD = document.getElementById('fsBtnHD');
    const fsPause = document.getElementById('fsBtnPause');
    if (phase === 'queuing' || phase === 'almost') {
        if (fsStart) { fsStart.disabled = true; fsStart.textContent = 'Queued'; }
        if (fsStop) { fsStop.disabled = false; fsStop.classList.add('cancel'); }
        if (fsFl) fsFl.disabled = true;
        if (fsHD) fsHD.disabled = true;
        if (fsPause) fsPause.disabled = true;
    } else if (phase === 'assigned') {
        if (fsStart) { fsStart.disabled = true; fsStart.textContent = 'Preparing...'; }
        if (fsStop) { fsStop.disabled = true; fsStop.classList.remove('cancel'); }
        if (fsFl) fsFl.disabled = true;
        if (fsHD) fsHD.disabled = true;
        if (fsPause) fsPause.disabled = true;
    } else {
        if (fsStart) fsStart.textContent = 'Start';
        if (fsStop) fsStop.classList.remove('cancel');
    }
}

(function() {
    const btn = document.getElementById('fullscreenBtn');
    let lastTime = 0;
    function doToggle(e) { e.preventDefault(); if (Date.now() - lastTime < 400) return; lastTime = Date.now(); toggleVideoFullscreen(); }
    btn.addEventListener('touchend', doToggle, { passive: false });
    btn.addEventListener('click', doToggle);
})();

(function() {
    function bindFsBtn(id, action) {
        const btn = document.getElementById(id);
        let last = 0;
        function handler(e) { e.preventDefault(); if (Date.now() - last < 500) return; last = Date.now(); action(); }
        btn.addEventListener('touchend', handler, { passive: false });
        btn.addEventListener('click', handler);
    }
    bindFsBtn('fsBtnForceListen', () => toggleForceListen());
    bindFsBtn('fsBtnHD', () => toggleHD());
    bindFsBtn('fsBtnStart', () => startSession());
    bindFsBtn('fsBtnPause', () => pauseSession());
    bindFsBtn('fsBtnStop', () => stopSession());
})();

// Subtitle settings
function applySubtitleSettings() {
    const hVal = Math.max(5, Math.min(80, parseInt(document.getElementById('fsSubHeight').value, 10) || 20));
    const botA = Math.max(0, Math.min(100, parseInt(document.getElementById('fsAlphaBottom').value, 10) || 80)) / 100;
    const topA = Math.max(0, Math.min(100, parseInt(document.getElementById('fsAlphaTop').value, 10) || 30)) / 100;
    const root = document.documentElement;
    root.style.setProperty('--fs-chat-height', hVal + 'vh');
    const mid1 = topA + (botA - topA) * 0.3;
    const mid2 = topA + (botA - topA) * 0.7;
    root.style.setProperty('--fs-chat-mask',
        `linear-gradient(to bottom, rgba(0,0,0,${topA}) 0%, rgba(0,0,0,${mid1}) 30%, rgba(0,0,0,${mid2}) 60%, rgba(0,0,0,${botA}) 85%)`);
}
document.getElementById('fsSubHeight').addEventListener('change', applySubtitleSettings);
document.getElementById('fsAlphaBottom').addEventListener('change', applySubtitleSettings);
document.getElementById('fsAlphaTop').addEventListener('change', applySubtitleSettings);
applySubtitleSettings();

let _subtitleOn = true;
function toggleSubtitle() {
    _subtitleOn = !_subtitleOn;
    document.getElementById('fsChatOverlay').classList.toggle('subtitle-on', _subtitleOn);
    document.getElementById('subtitleToggleBtn').classList.toggle('active', _subtitleOn);
}
function updateSubtitleBtnVisibility(show) {
    document.getElementById('subtitleToggleBtn').classList.toggle('visible', show);
    document.getElementById('subtitleToggleBtn').classList.toggle('active', _subtitleOn);
}
(function() {
    const btn = document.getElementById('subtitleToggleBtn');
    let last = 0;
    function handler(e) { e.preventDefault(); if (Date.now() - last < 400) return; last = Date.now(); toggleSubtitle(); }
    btn.addEventListener('touchend', handler, { passive: false });
    btn.addEventListener('click', handler);
})();

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
    document.getElementById('modeBadge').textContent = isFile ? 'File' : 'Live';
    if (!isFile) { startCameraPreview(); }
    else {
        if (cameraPreview) { cameraPreview.stopPreview(); cameraPreview = null; }
        // Ensure video element is hidden even if camera was still initializing
        document.getElementById('videoEl').style.display = 'none';
        document.getElementById('videoEl').srcObject = null;
        document.getElementById('videoPlaceholder').style.display = 'flex';
        document.getElementById('camFlipBtn').classList.remove('visible');
        document.getElementById('mirrorBtn').classList.remove('visible');
        updateFullscreenBtnVisibility(false);
        setStatusLamp('hidden');
    }
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
        // Show duration
        const video = document.createElement('video');
        video.preload = 'metadata';
        const url = URL.createObjectURL(selectedFile);
        video.src = url;
        video.onloadedmetadata = () => {
            const dur = video.duration;
            const label = dur > FILE_MAX_DURATION
                ? `${dur.toFixed(1)}s (will truncate to ${FILE_MAX_DURATION}s)`
                : `${dur.toFixed(1)}s`;
            document.getElementById('fileDuration').textContent = label;
            URL.revokeObjectURL(url);
        };
        video.onerror = () => { URL.revokeObjectURL(url); };
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

        addSystemEntry(`File LUFS: ${isFinite(srcLUFS) ? srcLUFS.toFixed(1) : '—'} → auto ${autoGainDb.toFixed(1)} dB`);
    } catch (err) {
        console.warn('measureFileLUFS failed:', err);
    }
}


// ============================================================================
// Session Control
// ============================================================================
async function startSession() {
    if (session) return;
    if (currentMode === 'file' && !selectedFile) { alert('Please select a video file first.'); return; }

    clearConversation();
    metricsPanel.update({ type: 'state', sessionState: 'Starting...' });
    setStatusLamp('preparing');
    metricsPanel.reset();
    document.getElementById('lampTimer').textContent = '';
    document.getElementById('videoPlaceholder').style.display = 'none';
    document.getElementById('videoOverlay').style.display = 'flex';
    _diagEvents = [];

    // Create media provider
    if (currentMode === 'live') {
        if (cameraPreview && cameraPreview._previewing) { media = cameraPreview; cameraPreview = null; }
        else { media = new LiveMediaProvider(); }
    } else {
        if (cameraPreview) { cameraPreview.stopPreview(); cameraPreview = null; }
        const audioMode = document.querySelector('input[name="fileAudioMode"]:checked')?.value || 'video';
        const _f = (id, def) => { const v = parseFloat(document.getElementById(id).value); return Number.isFinite(v) ? v : def; };
        const _i = (id, def) => { const v = parseInt(document.getElementById(id).value, 10); return Number.isFinite(v) ? v : def; };
        media = new FileMediaProvider(selectedFile, {
            audioMode,
            frameOffset: _f('frameOffset', 0.5),
            padBeforeSec: _i('padBeforeSec', 0),
            padAfterSec: _i('padAfterSec', 2),
        });
    }

    // Recording setup (video + stereo audio WebM)
    const recEnabled = document.getElementById('recCheckbox')?.checked;
    if (recEnabled) {
        const videoEl = document.getElementById('videoEl');
        sessionRecorder = new SessionVideoRecorder(videoEl, SAMPLE_RATE_IN, SAMPLE_RATE_OUT);
        lastRecordingBlob = null;
        const dlBtn = document.getElementById('btnDownloadRec');
        if (dlBtn) { dlBtn.style.display = 'none'; dlBtn.disabled = true; }
    }

    // Create DuplexSession + wire hooks
    session = new DuplexSession('omni', {
        getMaxKvTokens: () => parseInt(document.getElementById('maxKvTokens').value, 10) || 8192,
        getPlaybackDelayMs: () => parseInt(document.getElementById('playbackDelay').value, 10) || 200,
        outputSampleRate: SAMPLE_RATE_OUT,
    });
    session.onMetrics = (data) => metricsPanel.update(data);
    session.onSystemLog = addSystemEntry;
    session.onSpeakStart = (text) => {
        const handle = addSpeakEntry(text);
        if (sessionRecorder) sessionRecorder.setSubtitleText(text);
        return handle;
    };
    session.onSpeakUpdate = (el, text) => {
        el.textContent = text;
        updateFsSpeakText(text);
        if (sessionRecorder) sessionRecorder.setSubtitleText(text);
    };
    session.onSpeakEnd = () => {
        finishFsSpeak();
        if (sessionRecorder) sessionRecorder.finalizeSubtitle();
    };
    session.onQueueUpdate = (data) => {
        const lamp = document.getElementById('statusLamp');
        if (data) {
            setStatusLamp('preparing');
            _queueCountdownLabel = lamp?.querySelector('.label');
            _omniCountdown.update(data.estimated_wait_s, data.position, data.queue_length || '?');
            if (data.position === 1 && _queuePhase !== 'almost') {
                _queuePhase = 'almost';
                setQueueButtonStates('almost');
                syncFullscreenQueueButtons('almost');
                if (!_stopDingDong) _stopDingDong = startDingDongLoop();
            } else if (data.position !== 1 && _queuePhase !== 'almost') {
                _queuePhase = 'queuing';
                setQueueButtonStates('queuing');
                syncFullscreenQueueButtons('queuing');
            }
        } else {
            _omniCountdown.stop();
            if (_stopDingDong) { _stopDingDong(); _stopDingDong = null; }
        }
    };
    session.onQueueDone = () => {
        _queuePhase = 'assigned';
        if (_stopDingDong) { _stopDingDong(); _stopDingDong = null; }
        setQueueButtonStates('assigned');
        syncFullscreenQueueButtons('assigned');
        playAlarmBell();
    };
    session.onPrepared = async () => {
        if (session.audioPlayer && session.audioPlayer._ctx) {
            omniDeviceSelector.applySinkId(session.audioPlayer._ctx);
        }
        await playSessionChime();
    };
    session.onCleanup = () => {
        _omniCountdown.stop();
        if (_stopDingDong) { _stopDingDong(); _stopDingDong = null; }
        _queuePhase = null;
        setQueueButtonStates(null);
        syncFullscreenQueueButtons(null);
        // Finalize video recording (async — stop() returns Promise for MediaRecorder)
        if (sessionRecorder && sessionRecorder.recording) {
            const recorder = sessionRecorder;
            sessionRecorder = null;
            recorder.stop().then((result) => {
                if (result.blob.size > 0) {
                    lastRecordingBlob = result.blob;
                    addSystemEntry(`Recording: ${result.durationSec.toFixed(1)}s WebM (${(result.blob.size / 1024 / 1024).toFixed(1)} MB)`);
                    const btn = document.getElementById('btnDownloadRec');
                    if (btn) { btn.style.display = ''; btn.disabled = false; }
                    const ext = result.blob.type?.includes('mp4') ? 'mp4' : 'webm';
                    if (_saveShareUI) _saveShareUI.setRecordingBlob(result.blob, ext);
                }
            }).catch((err) => {
                console.error('[SessionVideoRecorder] stop failed:', err);
            });
        } else {
            sessionRecorder = null;
        }

        if (media) { media.stop(); media = null; }
        mixerCtrl?.stopMixerMeters();
        session = null;
        // Restart standalone mixer mic if mixer is still open
        const mixerPanel = document.getElementById('mixerPanel');
        if (mixerPanel && mixerPanel.style.display === 'block') {
            mixerCtrl?.startMixerMic();
            mixerCtrl?.startMixerMeters();
        }
        setStatusLamp('stopped');
        document.getElementById('videoOverlay').style.display = 'none';
        if (currentMode === 'live') { startCameraPreview(); }
        else { updateFullscreenBtnVisibility(false); document.getElementById('videoPlaceholder').style.display = 'flex'; document.getElementById('videoEl').style.display = 'none'; }
    };
    session.onRunningChange = (running) => setButtonStates(running);
    session.onPauseStateChange = (state) => {
        setPauseBtnState(state);
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
    session.onForceListenChange = (active) => setForceListenBtnState(active);
    session.onExtraResult = (result, recvTime) => {
        if (!result.is_listen) {
            const elapsed = session._sessionStartTime
                ? ((recvTime - session._sessionStartTime) / 1000).toFixed(1) : '?';
            _diagEvents.push({
                ev: 'chunk', n: session._resultCount, t: parseFloat(elapsed),
                model_ms: result.cost_all_ms || 0,
                drift_ms: session._lastDriftMs,
                ahead_ms: session.audioPlayer.lastAheadMs || 0,
                gaps: session.audioPlayer.gapCount || 0,
                shift_ms: session.audioPlayer.totalShiftMs || 0,
                turn: session.audioPlayer.turnIdx || 0,
            });
        }
    };

    session.audioPlayer.onGap = (gapInfo) => {
        setTimeout(() => {
            _sendDiagnostic({ event: 'gap', ...gapInfo, recent_chunks: _diagEvents.slice(-5) });
        }, 0);
    };

    // Build prepare payload
    const preparePayload = {
        config: { length_penalty: parseFloat(document.getElementById('omniLengthPenalty').value) || 1.0 },
        max_slice_nums: getEffectiveMaxSliceNums(),
    };
    const refBase64 = refAudio.getBase64();
    if (refBase64) preparePayload.ref_audio_base64 = refBase64;
    const ttsRef = omniTtsRef.getBase64();
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
            async () => {
                media.onChunk = (chunk) => {
                    const msg = { type: 'audio_chunk', audio_base64: arrayBufferToBase64(chunk.audio.buffer) };
                    if (chunk.frameBase64) msg.frame_base64_list = [chunk.frameBase64];
                    const effectiveSlice = getEffectiveMaxSliceNums();
                    if (effectiveSlice > 1) msg.max_slice_nums = effectiveSlice;
                    session.sendChunk(msg);
                    updateTimeBadge(session.chunksSent);
                    // Fallback path: no-op when direct audio mode is active in recorder
                    if (sessionRecorder) sessionRecorder.pushLeft(chunk.audio);
                };
                media.onEnd = () => {
                    addSystemEntry('File playback completed (including padding). Auto-stopping session.');
                    stopSession();
                };
                await media.start();

                // Start video recording right after media (direct left audio, zero delay)
                if (sessionRecorder) {
                    const leftAudioOptions = {};
                    const audioStream = media.getAudioOutputStream?.();
                    if (audioStream) {
                        leftAudioOptions.leftAudioStream = audioStream;
                    } else {
                        const buf = media.getFullAudioBuffer?.();
                        if (buf) leftAudioOptions.leftAudioBuffer = buf;
                    }
                    const recSettings = recordingSettings ? recordingSettings.getSettings() : {};
                    await sessionRecorder.start(recSettings, leftAudioOptions);
                }

                mixerCtrl?.stopMixerMic();
                mixerCtrl?.startMixerMeters();
            }
        );

        metricsPanel.update({ type: 'state', sessionState: 'Active' });
        setStatusLamp('live');
        updateFullscreenBtnVisibility(true);

        if (_saveShareUI && session && session.recordingSessionId) _saveShareUI.setSessionId(session.recordingSessionId);
    } catch (err) {
        const isCancelled = err.message?.includes('cancelled');
        if (!isCancelled) {
            console.error('Session start failed:', err);
            addSystemEntry(`Failed: ${err.message}`);
        }
        if (session) { try { session.cleanup(); } catch (_) {} }
        session = null;
        media = null;
        setButtonStates(false);
        metricsPanel.update({ type: 'state', sessionState: isCancelled ? 'Cancelled' : 'Error' });
        setStatusLamp(isCancelled ? 'stopped' : 'hidden');
        updateFullscreenBtnVisibility(false);
        document.getElementById('videoPlaceholder').style.display = 'flex';
        document.getElementById('videoOverlay').style.display = 'none';
        document.getElementById('videoEl').style.display = 'none';
        if (currentMode === 'live') startCameraPreview();
    }
}

function pauseSession() { if (session) session.pauseToggle(); }

function stopSession() {
    if (!session) return;
    if (_queuePhase) { session.cancelQueue(); } else { _sendSessionSummary(); session.stop(); }
    session = null;
    metricsPanel.update({ type: 'state', sessionState: 'Stopped' });
}

function toggleForceListen() { if (session) session.toggleForceListen(); }

// ============================================================================
// Camera Preview
// ============================================================================
async function startCameraPreview() {
    if (session) return;
    if (cameraPreview && cameraPreview._previewing) return;
    try {
        cameraPreview = new LiveMediaProvider();
        await cameraPreview.startPreview();
        updateFullscreenBtnVisibility(true);
    } catch (err) {
        console.warn('Camera preview failed:', err.message);
        cameraPreview = null;
        updateFullscreenBtnVisibility(false);
        document.getElementById('videoPlaceholder').style.display = 'flex';
    }
}

if (currentMode === 'live') startCameraPreview();

// ============================================================================
// Wire up inline onclick/onchange → addEventListener
// ============================================================================
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
});

document.getElementById('videoFileInput').addEventListener('change', function() {
    onFileSelected(this);
});

// File audio mode radios
document.querySelectorAll('input[name="fileAudioMode"]').forEach(radio => {
    radio.addEventListener('change', onFileAudioModeChange);
});

// Frame offset slider
document.getElementById('frameOffset').addEventListener('input', function() {
    document.getElementById('frameOffsetVal').textContent = parseFloat(this.value).toFixed(2);
});

// File playback volume slider — real-time update of video element volume
document.getElementById('filePlaybackVol').addEventListener('input', function() {
    const pct = parseInt(this.value);
    document.getElementById('filePlaybackVolVal').textContent = pct + '%';
    const videoEl = document.getElementById('videoEl');
    if (media instanceof FileMediaProvider && !videoEl.muted) {
        videoEl.volume = pct / 100;
    }
});

// Length Penalty preset buttons + active highlight sync
function syncLpPresetHighlight() {
    const val = parseFloat(document.getElementById('omniLengthPenalty')?.value);
    document.querySelectorAll('.lp-preset-btn').forEach(btn => {
        btn.classList.toggle('active', parseFloat(btn.dataset.lp) === val);
    });
}

document.querySelectorAll('.lp-preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const val = parseFloat(btn.dataset.lp);
        const input = document.getElementById('omniLengthPenalty');
        if (input && !isNaN(val)) {
            input.value = val;
            input.dispatchEvent(new Event('change', { bubbles: true }));
        }
        syncLpPresetHighlight();
    });
});

document.getElementById('omniLengthPenalty')?.addEventListener('input', syncLpPresetHighlight);
document.getElementById('omniLengthPenalty')?.addEventListener('change', syncLpPresetHighlight);

// MaxKV hard cap at 8192
document.getElementById('maxKvTokens')?.addEventListener('change', function () {
    if (parseInt(this.value, 10) > 8192) this.value = 8192;
});

// Control buttons (mic calibration is handled by MixerController)
wireDuplexControls({
    onStart: startSession,
    onStop: stopSession,
    onPause: pauseSession,
    onForceListen: toggleForceListen,
});
document.getElementById('btnHD')?.addEventListener('click', toggleHD);

// TTS ref mode radios
document.querySelectorAll('input[name="omniTtsRefMode"]').forEach(radio => {
    radio.addEventListener('change', () => omniTtsRef.onModeChange());
});

// ============================================================================
// Mixer Controller (shared module)
// ============================================================================
mixerCtrl = new MixerController({
    sampleRate: SAMPLE_RATE_IN,
    addLog: addSystemEntry,
    getMedia: () => media,
    monitorElId: 'videoEl',
    isFileMode: () => media instanceof FileMediaProvider,
    getFileChunks: () => media?._videoAudioChunks,
    getSelectedFile: () => selectedFile,
    getLastRecordingBlob: () => lastRecordingBlob,
    getDownloadExt: (blob) => {
        const type = blob.type || '';
        if (type.includes('mp4')) return 'mp4';
        if (type.startsWith('audio/')) return 'wav';
        return 'webm';
    },
    onInit: () => {
        const recSettingsBtn = document.getElementById('btnRecSettings');
        if (recSettingsBtn) {
            recordingSettings = new RecordingSettings(recSettingsBtn);
        }
    },
});

if (document.readyState !== 'loading') {
    mixerCtrl.init();
} else {
    document.addEventListener('DOMContentLoaded', () => mixerCtrl.init());
}

// Cleanup on page unload (release media, WS, AudioContext)
window.addEventListener('beforeunload', () => {
    if (session?.running) session.stop();
});

/* ---------- end of file ---------- */
