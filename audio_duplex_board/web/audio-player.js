/**
 * lib/audio-player.js — AudioBufferSourceNode pre-scheduled audio player (zero DOM dependency)
 *
 * Reports playback metrics via the onMetrics callback instead of writing to DOM directly.
 * The consumer (UI layer / page) wires onMetrics to update the display.
 *
 * @import { resampleAudio } from './duplex-utils.js'
 */

import { resampleAudio } from './duplex-utils.js';

export class AudioPlayer {
    /**
     * @param {object} [options]
     * @param {number} [options.outputSampleRate] - Expected output sample rate (e.g. 24000)
     * @param {function} [options.getPlaybackDelayMs] - Returns playback delay in ms (default: () => 200)
     */
    constructor(options = {}) {
        this._outputSR_expected = options.outputSampleRate || 24000;
        this._getDelayMs = options.getPlaybackDelayMs || (() => 200);

        this._ctx = null;
        this._outputSR = 0;
        this._turnActive = false;
        this._turnIdx = 0;
        this._enqueueCount = 0;
        this._nextTime = 0;
        this._playing = false;
        this._sources = [];
        this._delayTimer = null;
        this._pendingChunks = [];  // {resampled: Float32Array, raw: Float32Array}[]

        // Monitoring metrics
        this._firstChunkTime = 0;
        this._playbackStartTime = 0;
        this._playbackStartCtxTime = 0;
        this._gapCount = 0;
        this._totalShiftMs = 0;
        this._lastAheadMs = 0;
        this._lastArrivalTime = 0;
        this._aheadInterval = null;

        /**
         * Metrics callback: (data) => void
         * data shape: { ahead, gapCount, totalShift, turn, pdelay? }
         * Called inside requestAnimationFrame for batched UI updates.
         */
        this.onMetrics = null;

        /**
         * Gap callback: (gapInfo) => void
         * gapInfo shape: { gap_idx, gap_ms, total_shift_ms, chunk_idx, turn }
         */
        this.onGap = null;

        /**
         * Raw audio callback for session recording.
         * Fires from _scheduleChunk with decoded PCM and the actual scheduled
         * playback time (performance.now()-based), NOT the arrival time.
         * (samples: Float32Array, sampleRate: number, playbackTimestamp: number) => void
         */
        this.onRawAudio = null;
    }

    // Public read-only accessors
    get turnActive() { return this._turnActive; }
    get playing() { return this._playing; }
    get ctx() { return this._ctx; }
    get nextTime() { return this._nextTime; }
    get gapCount() { return this._gapCount; }
    get totalShiftMs() { return this._totalShiftMs; }
    get lastAheadMs() { return this._lastAheadMs; }
    get turnIdx() { return this._turnIdx; }

    init() {
        if (!this._ctx || this._ctx.state === 'closed') {
            this._ctx = new AudioContext();
            this._outputSR = this._ctx.sampleRate;
            console.log(`[AudioPlayer] init: outputSR=${this._outputSR}`);
        }
        this._stopAllSources();
        this._turnActive = false;
        this._turnIdx = 0;
        this._enqueueCount = 0;
        this._nextTime = 0;
        this._playing = false;
        this._pendingChunks = [];
        if (this._delayTimer) { clearTimeout(this._delayTimer); this._delayTimer = null; }
    }

    /** Start a new SPEAK turn */
    beginTurn() {
        if (this._turnActive) return;
        this._stopAllSources();
        this._turnActive = true;
        this._playing = false;
        this._turnIdx++;
        this._pendingChunks = [];
        this._nextTime = 0;
        this._firstChunkTime = 0;
        this._playbackStartTime = 0;
        this._gapCount = 0;
        this._totalShiftMs = 0;
        this._lastAheadMs = 0;
        this._lastArrivalTime = 0;
        if (this._delayTimer) { clearTimeout(this._delayTimer); this._delayTimer = null; }
        console.log(`[AudioPlayer] === turn #${this._turnIdx} begin ===`);
    }

    /**
     * Enqueue a SPEAK audio chunk for playback.
     * @param {string} base64Data - Base64-encoded Float32 audio at outputSampleRate
     * @param {number} [arrivalTime] - performance.now() timestamp of arrival
     */
    playChunk(base64Data, arrivalTime) {
        if (!base64Data || !this._ctx) return;
        const t0 = performance.now();

        const binary = atob(base64Data);
        const len = binary.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i += 1024) {
            const end = Math.min(i + 1024, len);
            for (let j = i; j < end; j++) bytes[j] = binary.charCodeAt(j);
        }
        const samples = new Float32Array(bytes.buffer);
        if (samples.length === 0) return;

        const resampled = resampleAudio(samples, this._outputSR_expected, this._outputSR);
        const raw = this.onRawAudio ? samples : null;
        this._enqueueCount++;

        if (!this._firstChunkTime) {
            this._firstChunkTime = arrivalTime || t0;
        }
        this._lastArrivalTime = arrivalTime || t0;

        if (this._playing) {
            this._scheduleChunk(resampled, raw);
            this._lastAheadMs = (this._nextTime - this._ctx.currentTime) * 1000;
            this._emitMetrics();
        } else {
            this._pendingChunks.push({ resampled, raw });
            const delayMs = this._getDelayMs();
            if (!this._delayTimer) {
                if (delayMs <= 0) {
                    this._startPlayback();
                } else {
                    this._delayTimer = setTimeout(() => {
                        this._delayTimer = null;
                        if (this._pendingChunks.length > 0) {
                            this._startPlayback();
                        }
                    }, delayMs);
                }
            }
        }
    }

    /** Current SPEAK turn ended */
    endTurn() {
        if (!this._turnActive) return;
        if (!this._playing && this._pendingChunks.length > 0) {
            if (this._delayTimer) { clearTimeout(this._delayTimer); this._delayTimer = null; }
            this._startPlayback();
        }
        this._turnActive = false;
        this._stopAheadMonitor();
        const ahead = this._playing
            ? ((this._nextTime - this._ctx.currentTime) * 1000).toFixed(0)
            : '0';
        console.log(`[AudioPlayer] === turn #${this._turnIdx} end (remaining=${ahead}ms) ===`);
    }

    _startPlayback() {
        if (this._playing) return;
        this._playing = true;
        this._playbackStartTime = performance.now();
        this._playbackStartCtxTime = this._ctx.currentTime;
        if (this._ctx.state === 'suspended') this._ctx.resume();

        this._nextTime = this._ctx.currentTime;
        for (const chunk of this._pendingChunks) {
            this._scheduleChunk(chunk.resampled, chunk.raw);
        }
        this._pendingChunks = [];

        const pdelay = this._firstChunkTime ? (this._playbackStartTime - this._firstChunkTime) : 0;
        this._lastAheadMs = (this._nextTime - this._ctx.currentTime) * 1000;
        this._emitMetrics({ pdelay });
        this._startAheadMonitor();

        console.log(`[AudioPlayer] playback started (buffered=${this._lastAheadMs.toFixed(0)}ms, pdelay=${pdelay.toFixed(0)}ms)`);
    }

    _scheduleChunk(samples, rawSamples) {
        const buffer = this._ctx.createBuffer(1, samples.length, this._outputSR);
        buffer.getChannelData(0).set(samples);
        const source = this._ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(this._ctx.destination);

        const now = this._ctx.currentTime;
        if (this._nextTime < now) {
            const gapMs = (now - this._nextTime) * 1000;
            if (gapMs > 10) {
                this._gapCount++;
                this._totalShiftMs += gapMs;
                this._emitMetrics();
                if (this.onGap) {
                    const info = {
                        gap_idx: this._gapCount,
                        gap_ms: gapMs,
                        total_shift_ms: this._totalShiftMs,
                        chunk_idx: this._enqueueCount,
                        turn: this._turnIdx,
                    };
                    setTimeout(() => this.onGap(info), 0);
                }
            }
            this._nextTime = now;
        }

        if (this.onRawAudio && rawSamples) {
            const playbackMs = this._playbackStartTime +
                (this._nextTime - this._playbackStartCtxTime) * 1000;
            this.onRawAudio(rawSamples, this._outputSR_expected, playbackMs);
        }

        source.start(this._nextTime);
        this._nextTime += buffer.duration;

        this._sources.push(source);
        source.onended = () => {
            const idx = this._sources.indexOf(source);
            if (idx >= 0) this._sources.splice(idx, 1);
        };
    }

    /** Emit metrics via callback inside rAF */
    _emitMetrics(extra) {
        if (!this.onMetrics) return;
        const data = {
            ahead: this._lastAheadMs,
            gapCount: this._gapCount,
            totalShift: this._totalShiftMs,
            turn: this._turnIdx,
            ...extra,
        };
        requestAnimationFrame(() => {
            if (this.onMetrics) this.onMetrics(data);
        });
    }

    _startAheadMonitor() {
        this._stopAheadMonitor();
        this._aheadInterval = setInterval(() => {
            if (!this._playing || !this._ctx) {
                this._stopAheadMonitor();
                return;
            }
            const ahead = (this._nextTime - this._ctx.currentTime) * 1000;
            this._lastAheadMs = Math.max(0, ahead);
            this._emitMetrics();
        }, 200);
    }

    _stopAheadMonitor() {
        if (this._aheadInterval) { clearInterval(this._aheadInterval); this._aheadInterval = null; }
    }

    _stopAllSources() {
        if (this._delayTimer) { clearTimeout(this._delayTimer); this._delayTimer = null; }
        this._stopAheadMonitor();
        for (const src of this._sources) {
            try { src.stop(); } catch (_) {}
            try { src.disconnect(); } catch (_) {}
        }
        this._sources = [];
        this._playing = false;
        this._pendingChunks = [];
    }

    /** Stop all playback immediately (public API) */
    stopAll() {
        this._stopAllSources();
    }

    /** Full session stop */
    stop() {
        console.log(`[AudioPlayer] session stop (${this._enqueueCount} chunks total)`);
        this._stopAllSources();
        this._turnActive = false;
    }
}
