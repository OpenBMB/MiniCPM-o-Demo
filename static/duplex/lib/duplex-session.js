/**
 * lib/duplex-session.js — Duplex session lifecycle manager (zero DOM dependency)
 *
 * Manages: WebSocket lifecycle, message dispatch, result processing,
 * pause/resume state machine, force-listen, AudioPlayer coordination.
 *
 * Reports all metrics/state changes via callbacks — no DOM access.
 */

import { AudioPlayer } from './audio-player.js';

export class DuplexSession {
    /**
     * @param {string} prefix - Session ID prefix ('omni' or 'adx')
     * @param {object} [config]
     * @param {number} [config.maxKvTokens=8192] - KV cache limit for auto-stop
     * @param {function} [config.getPlaybackDelayMs] - Returns playback delay in ms
     * @param {number} [config.outputSampleRate=24000] - Audio output sample rate
     */
    constructor(prefix, config = {}) {
        this.prefix = prefix;
        this.config = {
            getMaxKvTokens: config.getMaxKvTokens || (() => 8192),
            getPlaybackDelayMs: config.getPlaybackDelayMs || (() => 200),
            outputSampleRate: config.outputSampleRate || 24000,
            getWsUrl: config.getWsUrl || ((sessionId) => {
                const proto = location.protocol === 'https:' ? 'wss' : 'ws';
                return `${proto}://${location.host}/ws/duplex/${sessionId}`;
            }),
        };

        this.ws = null;
        this.audioPlayer = new AudioPlayer({
            outputSampleRate: this.config.outputSampleRate,
            getPlaybackDelayMs: this.config.getPlaybackDelayMs,
        });
        this.sessionId = '';
        this.chunksSent = 0;
        this.paused = false;
        this.pauseState = 'active'; // 'active' | 'pausing' | 'paused'
        this.serverPauseConfirmed = false;
        this.forceListenActive = false;
        this.currentSpeakText = '';
        this._speakHandle = null;
        this._started = false;

        // Timing metrics
        this._sessionStartTime = 0;
        this._lastListenTime = 0;
        this._wasListening = true;
        this._lastTTFS = 0;
        this._lastResultTime = 0;
        this._firstServerTs = 0;
        this._firstClientTs = 0;
        this._resultCount = 0;
        this._lastDriftMs = null;

        // Bridge AudioPlayer metrics to session onMetrics
        this.audioPlayer.onMetrics = (data) => {
            this.onMetrics({
                type: 'audio',
                ahead: data.ahead,
                gapCount: data.gapCount,
                totalShift: data.totalShift,
                turn: data.turn,
                pdelay: data.pdelay,
            });
        };
    }

    /** Whether the session is actively running (connected + receiving). */
    get running() { return this._started; }

    // ==== Hooks (set before calling start) ====
    // All have sensible no-op defaults; override as needed.

    /** Add system log entry. */
    onSystemLog(text) {}

    /**
     * Queue status update.
     * @param {object|null} data - { position, estimated_wait_s, ticket_id, queue_length } or null when done
     */
    onQueueUpdate(data) {}

    /** Worker 分配完成（queue_done），在 onQueueUpdate(null) 之前调用。 */
    onQueueDone() {}

    /** AI starts speaking. Return an opaque handle for subsequent text updates. */
    onSpeakStart(text) { return null; }

    /** Update AI speak text. handle is the value returned by onSpeakStart. */
    onSpeakUpdate(handle, text) {}

    /** AI finished a turn of speaking. */
    onSpeakEnd() {}

    /** Handle a listen-mode result (e.g. show user transcript). */
    onListenResult(result) {}

    /** Additional per-result processing (diagnostics, etc.). Called inside rAF. */
    onExtraResult(result, recvTime) {}

    /** Called after WS prepare succeeds, before media starts. Async OK. */
    async onPrepared() {}

    /** Page-specific cleanup (stop media, reset UI elements). */
    onCleanup() {}

    /**
     * Metrics callback — all numeric/state metrics flow through here.
     * @param {object} data - Metric data with a `type` field:
     *   type='audio': { ahead, gapCount, totalShift, turn, pdelay? }
     *   type='result': { latencyMs, costAllMs, driftMs, kvCacheLength, ttfsMs, modelState, chunksSent }
     *   type='state': { sessionState, modelState? }
     */
    onMetrics(data) {}

    /** Session running state changed. */
    onRunningChange(running) {}

    /** Pause state changed ('active'/'pausing'/'paused'). */
    onPauseStateChange(state) {}

    /** Force listen state changed. */
    onForceListenChange(active) {}

    // ==== Core API ====

    /**
     * Connect, prepare, init audio, start media, begin receiving.
     * @param {string} systemPrompt - System prompt text
     * @param {object} preparePayload - Extra fields for the 'prepare' WS message
     * @param {function} [startMediaFn] - Async function to start media capture
     * @throws {Error} on WS connection or prepare failure
     */
    async start(systemPrompt, preparePayload, startMediaFn) {
        this._reset();
        this.sessionId = `${this.prefix}_${Date.now().toString(36)}`;
        this.onMetrics({ type: 'state', sessionState: 'Connecting...', sessionId: this.sessionId });

        const wsUrl = this.config.getWsUrl(this.sessionId);

        try {
            // 1. Connect
            await new Promise((resolve, reject) => {
                this.ws = new WebSocket(wsUrl);
                this.ws.onopen = () => resolve();
                this.ws.onerror = () => reject(new Error('WebSocket connection failed'));
                this.ws.onclose = () => {
                    if (!this._started) reject(new Error('WebSocket closed before ready'));
                };
            });

            // 2. Wait for queue (if any) + Prepare
            await new Promise((resolve, reject) => {
                let queueDone = false;
                let prepareSent = false;
                this._queueReject = reject;

                const sendPrepare = () => {
                    if (prepareSent) return;
                    prepareSent = true;
                    this.ws.send(JSON.stringify({
                        type: 'prepare',
                        system_prompt: systemPrompt,
                        ...preparePayload,
                    }));
                };

                this.ws.onmessage = (e) => {
                    const msg = JSON.parse(e.data);
                    if (msg.type === 'queued') {
                        this.onQueueUpdate({
                            position: msg.position,
                            estimated_wait_s: msg.estimated_wait_s,
                            ticket_id: msg.ticket_id,
                            queue_length: msg.queue_length,
                        });
                        const total = msg.queue_length || '?';
                        this.onSystemLog(`排队 ${msg.position}/${total}，预计 ~${Math.round(msg.estimated_wait_s)}s`);
                    } else if (msg.type === 'queue_update') {
                        this.onQueueUpdate({
                            position: msg.position,
                            estimated_wait_s: msg.estimated_wait_s,
                            queue_length: msg.queue_length,
                        });
                    } else if (msg.type === 'queue_done') {
                        queueDone = true;
                        this._queueReject = null;
                        this.onQueueDone();
                        this.onQueueUpdate(null);
                        this.onSystemLog('Worker assigned, preparing...');
                        sendPrepare();
                    } else if (msg.type === 'prepared') {
                        this._queueReject = null;
                        this.onQueueUpdate(null);
                        if (msg.recording_session_id) {
                            this.recordingSessionId = msg.recording_session_id;
                        }
                        this.onSystemLog(`Prepared (${msg.prompt_length} tokens)`);
                        resolve();
                    } else if (msg.type === 'error') {
                        this._queueReject = null;
                        reject(new Error(msg.error));
                    }
                };

                // If not queued, send prepare immediately
                // Gateway sends queue messages before proxying starts,
                // so we wait a short time for potential queue messages
                setTimeout(() => {
                    if (!queueDone) sendPrepare();
                }, 100);
            });

            // 3. Post-prepare hook
            await this.onPrepared();

            // 4. Init audio player
            this.audioPlayer.init();

            // 5. Start media (page-specific)
            if (startMediaFn) await startMediaFn();

            // 6. Begin receiving
            this._started = true;
            this.onRunningChange(true);
            this.ws.onmessage = (e) => this._handleMessage(JSON.parse(e.data));
            this.ws.onclose = () => {
                this.onSystemLog('Session ended');
                this.cleanup();
            };
        } catch (err) {
            if (this.ws) { try { this.ws.close(); } catch (_) {} this.ws = null; }
            this._started = false;
            throw err;
        }
    }

    /**
     * Send an audio chunk (+ optional extras) to the server.
     * @param {object} msg - Message object (must include type + audio_base64)
     */
    sendChunk(msg) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        if (this.paused) return;
        if (this.forceListenActive) msg.force_listen = true;
        this.ws.send(JSON.stringify(msg));
        this.chunksSent++;
        this.onMetrics({ type: 'result', chunksSent: this.chunksSent });
    }

    /** Toggle pause/resume. */
    pauseToggle() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        if (this.pauseState === 'active') {
            this.paused = true;
            this.pauseState = 'pausing';
            this.serverPauseConfirmed = false;
            this.onPauseStateChange('pausing');
            this.onMetrics({ type: 'state', sessionState: 'Pausing...' });
            this.ws.send(JSON.stringify({ type: 'pause' }));
        } else if (this.pauseState === 'paused') {
            this.paused = false;
            this.pauseState = 'active';
            this.onPauseStateChange('active');
            this.onMetrics({ type: 'state', sessionState: 'Active' });
            this.ws.send(JSON.stringify({ type: 'resume' }));
        }
    }

    /** Toggle force-listen mode. */
    toggleForceListen() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        this.forceListenActive = !this.forceListenActive;
        this.onForceListenChange(this.forceListenActive);
        if (this.forceListenActive) {
            this.onSystemLog('Force Listen ON — model will only listen');
            this.audioPlayer.stopAll();
            if (this.audioPlayer.turnActive) this.audioPlayer.endTurn();
        } else {
            if (this.audioPlayer.turnActive) this.audioPlayer.endTurn();
            this.onSystemLog('Force Listen OFF — model may speak');
        }
    }

    /** Send stop command and cleanup. */
    stop() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'stop' }));
        }
        this.cleanup();
    }

    /** 取消排队（排队等待阶段调用，关闭 WS 触发后端 cancel ticket）。 */
    cancelQueue() {
        const reject = this._queueReject;
        this._queueReject = null;
        this.cleanup();
        if (reject) reject(new Error('Queue cancelled by user'));
    }

    /** Full cleanup: page cleanup + audio + WS + reset state. */
    cleanup() {
        this.onCleanup();
        this.audioPlayer.stop();
        if (this.ws) {
            this.ws.onclose = null;
            try { this.ws.close(); } catch (_) {}
            this.ws = null;
        }
        this._started = false;
        this.paused = false;
        this.pauseState = 'active';
        this.serverPauseConfirmed = false;
        this.forceListenActive = false;
        this.onRunningChange(false);
        this.onForceListenChange(false);
        this.onPauseStateChange('active');
        this.onMetrics({ type: 'state', sessionState: 'Stopped' });
    }

    // ==== Internal ====

    _reset() {
        this._sessionStartTime = performance.now();
        this._lastListenTime = 0;
        this._wasListening = true;
        this._lastTTFS = 0;
        this._lastResultTime = 0;
        this._firstServerTs = 0;
        this._firstClientTs = 0;
        this._resultCount = 0;
        this._lastDriftMs = null;
        this.chunksSent = 0;
        this.currentSpeakText = '';
        this._speakHandle = null;
        this.paused = false;
        this.pauseState = 'active';
        this.serverPauseConfirmed = false;
        this.forceListenActive = false;
        this._queueReject = null;
    }

    _handleMessage(msg) {
        switch (msg.type) {
            case 'result': this._handleResult(msg); break;
            case 'audio_only':
                if (msg.audio_data) {
                    this.audioPlayer.playChunk(msg.audio_data, performance.now());
                }
                break;
            case 'error':
                if (this.paused && msg.error && msg.error.includes('paused')) break;
                this.onSystemLog(`Error: ${msg.error}`);
                break;
            case 'paused':
                this.serverPauseConfirmed = true;
                this._tryCompletePause();
                break;
            case 'resumed':
                this.paused = false;
                this.pauseState = 'active';
                this.onPauseStateChange('active');
                this.onSystemLog('Session resumed');
                break;
            case 'interrupted':
                this.onSystemLog('Interrupted (deprecated)');
                break;
            case 'stopped':
                this.onSystemLog('Session stopped by server');
                this.cleanup();
                break;
            case 'timeout':
                this.onSystemLog(`Timeout: ${msg.reason}`);
                this.cleanup();
                break;
            // Queue messages (defensive — normally handled during start phase)
            case 'queued':
            case 'queue_update':
                this.onQueueUpdate({
                    position: msg.position,
                    estimated_wait_s: msg.estimated_wait_s,
                    queue_length: msg.queue_length,
                });
                break;
            case 'queue_done':
                this.onQueueUpdate(null);
                break;
        }
    }

    _handleResult(result) {
        const recvTime = performance.now();
        this._resultCount++;

        // Listen/Speak transition tracking
        if (result.is_listen) {
            this._lastListenTime = recvTime;
            this._wasListening = true;
        } else {
            if (this._wasListening) {
                this._wasListening = false;
                this._lastTTFS = this._lastListenTime > 0
                    ? recvTime - this._lastListenTime : 0;
            }
        }

        // Audio player management
        if (!result.is_listen) {
            if (result.audio_data) {
                if (!this.audioPlayer.turnActive) this.audioPlayer.beginTurn();
                this.audioPlayer.playChunk(result.audio_data, recvTime);
            }
            if (result.end_of_turn && this.audioPlayer.turnActive) this.audioPlayer.endTurn();
        } else if (result.end_of_turn) {
            if (this.audioPlayer.turnActive) this.audioPlayer.endTurn();
        }

        // Compute drift
        this._lastDriftMs = null;
        if (result.server_send_ts) {
            const serverSendSec = result.server_send_ts;
            const clientRecvSec = Date.now() / 1000;
            if (!this._firstServerTs) {
                this._firstServerTs = serverSendSec;
                this._firstClientTs = clientRecvSec;
            }
            this._lastDriftMs = (clientRecvSec - serverSendSec
                - (this._firstClientTs - this._firstServerTs)) * 1000;
        }

        // KV cache auto-stop check
        const maxKv = this.config.getMaxKvTokens();
        if (result.kv_cache_length !== undefined && result.kv_cache_length > 0 && result.kv_cache_length >= maxKv) {
            this.onSystemLog(`\u26a0 KV cache (${result.kv_cache_length.toLocaleString()}) reached limit. Auto-stopping.`);
            setTimeout(() => this.stop(), 0);
        }

        // Deferred UI + metrics update via rAF
        requestAnimationFrame(() => {
            // Emit result metrics
            this.onMetrics({
                type: 'result',
                latencyMs: result.wall_clock_ms || result.cost_all_ms || 0,
                costAllMs: result.cost_all_ms,
                driftMs: this._lastDriftMs,
                kvCacheLength: result.kv_cache_length,
                maxKvTokens: maxKv,
                ttfsMs: (!result.is_listen && this._lastTTFS) ? this._lastTTFS : null,
                modelState: result.is_listen ? 'listening' : (result.end_of_turn ? 'end_of_turn' : 'speaking'),
                chunksSent: this.chunksSent,
                visionSlices: result.vision_slices,
                visionTokens: result.vision_tokens,
            });

            // Clear TTFS after emitting
            if (!result.is_listen && this._lastTTFS) {
                this._lastTTFS = 0;
            }

            // Text accumulation — 与 is_listen 解耦：
            // C++ 双工可能在同一个 decode 里先推 SPEAK 文本再推 __IS_LISTEN__
            // 以前看到 is_listen=true 就丢弃 currentSpeakText，导致气泡不连续
            if (result.text) {
                this.currentSpeakText += result.text;
                if (!this._speakHandle) {
                    this._speakHandle = this.onSpeakStart(this.currentSpeakText);
                } else {
                    this.onSpeakUpdate(this._speakHandle, this.currentSpeakText);
                }
            }

            if (result.is_listen) {
                // 切到 listen：先 finalize 当前 speak bubble（如果有），再通知 listen
                if (this._speakHandle) {
                    this.onSpeakEnd();
                    this._speakHandle = null;
                    this.currentSpeakText = '';
                }
                this.onListenResult(result);
            } else if (result.end_of_turn) {
                this.onSpeakEnd();
                this._speakHandle = null;
                this.currentSpeakText = '';
                this.onSystemLog('— end of turn —');
            }

            // Page-specific extensions
            this.onExtraResult(result, recvTime);

            this._lastResultTime = recvTime;
        });
    }

    _tryCompletePause() {
        if (this.pauseState !== 'pausing') return;
        if (!this.serverPauseConfirmed) return;
        const ap = this.audioPlayer;
        if (ap.playing && ap.ctx && ap.nextTime > ap.ctx.currentTime + 0.05) {
            setTimeout(() => this._tryCompletePause(), 100);
            return;
        }
        if (ap.turnActive) ap.endTurn();
        this.pauseState = 'paused';
        this.paused = true;
        this.onPauseStateChange('paused');
        this.onMetrics({ type: 'state', sessionState: 'Paused' });
        this.onSystemLog('Session paused');
    }
}
