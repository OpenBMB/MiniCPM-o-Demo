/**
 * lib/realtime-session.js — OpenAI Realtime-style session manager
 *
 * Realtime session client that speaks the API V2 protocol:
 *   session.init / input.append / response.output.delta(kind=...)
 *
 * Keeps the established callback interface so UI code remains thin.
 */

import { AudioPlayer } from './audio-player.js';

export class RealtimeSession {
    constructor(prefix, config = {}) {
        this.prefix = prefix;
        this.config = {
            getMaxKvTokens: config.getMaxKvTokens || (() => 8192),
            getPlaybackDelayMs: config.getPlaybackDelayMs || (() => 200),
            getStopOnSlidingWindow: config.getStopOnSlidingWindow || (() => false),
            outputSampleRate: config.outputSampleRate || 24000,
            getWsUrl: config.getWsUrl || (() => {
                const proto = location.protocol === 'https:' ? 'wss' : 'ws';
                const url = `${proto}://${location.host}/v1/realtime`;
                return window.ClientIdentity ? window.ClientIdentity.appendToUrl(url) : url;
            }),
        };

        this.ws = null;
        this.audioPlayer = new AudioPlayer({
            outputSampleRate: this.config.outputSampleRate,
            getPlaybackDelayMs: this.config.getPlaybackDelayMs,
        });
        this.sessionId = '';
        this.recordingSessionId = '';
        this.chunksSent = 0;
        this.paused = false;
        this.pauseState = 'active';
        this.forceListenActive = false;
        this.currentSpeakText = '';
        this._speakHandle = null;
        this._started = false;

        this._sessionStartTime = 0;
        this._lastListenTime = 0;
        this._wasListening = true;
        this._lastTTFS = 0;
        this._lastResultTime = 0;
        this._firstServerTs = 0;
        this._firstClientTs = 0;
        this._resultCount = 0;
        this._lastDriftMs = null;
        this._lastKvCacheLength = 0;
        this._lastFrameMetrics = {};

        // Protocol event log for the data flow panel
        this._eventLog = [];
        this._maxEventLog = 200;

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

    get running() { return this._started; }
    get eventLog() { return this._eventLog; }

    // ==== Hooks ====
    onSystemLog(text) {}
    onQueueUpdate(data) {}
    onQueueDone() {}
    onSpeakStart(text) { return null; }
    onSpeakUpdate(handle, text) {}
    onSpeakEnd() {}
    onListenResult(result) {}
    onExtraResult(result, recvTime) {}
    async onPrepared() {}
    onCleanup() {}
    onMetrics(data) {}
    onRunningChange(running) {}
    onForceListenChange(active) {}
    onPauseStateChange(state) {}
    /** New: protocol event logged (for data flow panel). */
    onProtocolEvent(entry) {}

    // ==== Protocol event logging ====
    _logProtoEvent(dir, type, summary, full) {
        const entry = {
            ts: Date.now(),
            dir, // 'client' | 'server'
            type,
            summary: summary || '',
            full: full || null,
        };
        this._eventLog.push(entry);
        if (this._eventLog.length > this._maxEventLog) this._eventLog.shift();
        this.onProtocolEvent(entry);
    }

    // ==== Core API ====

    async start(systemPrompt, preparePayload, startMediaFn) {
        this._reset();
        this.sessionId = '';
        this.recordingSessionId = '';
        this.onMetrics({ type: 'state', sessionState: 'Connecting...' });

        const wsUrl = this.config.getWsUrl();

        try {
            await new Promise((resolve, reject) => {
                this.ws = new WebSocket(wsUrl);
                this.ws.onopen = () => resolve();
                this.ws.onerror = () => reject(new Error('WebSocket connection failed'));
                this.ws.onclose = () => {
                    if (!this._started) reject(new Error('WebSocket closed before ready'));
                };
            });

            // Wait for queue + send session.init
            await new Promise((resolve, reject) => {
                let queueDone = false;
                let initSent = false;
                this._queueReject = reject;

                const sendSessionInit = () => {
                    if (initSent) return;
                    initSent = true;

                    const sessionInit = {
                        type: 'session.init',
                        payload: {
                            system_prompt: systemPrompt,
                            ...preparePayload,
                        },
                    };
                    this.ws.send(JSON.stringify(sessionInit));
                    this._logProtoEvent('client', 'session.init',
                        `system_prompt="${systemPrompt.slice(0, 40)}…"`, sessionInit);
                };

                this.ws.onmessage = (e) => {
                    const msg = JSON.parse(e.data);

                    if (msg.type === 'session.queued') {
                        this._logProtoEvent('server', 'session.queued',
                            `pos=${msg.position}`, msg);
                        this.onQueueUpdate({
                            position: msg.position,
                            estimated_wait_s: msg.estimated_wait_s,
                            ticket_id: msg.ticket_id,
                            queue_length: msg.queue_length,
                        });
                    } else if (msg.type === 'session.queue_update') {
                        this._logProtoEvent('server', 'session.queue_update',
                            `pos=${msg.position}`, msg);
                        this.onQueueUpdate({
                            position: msg.position,
                            estimated_wait_s: msg.estimated_wait_s,
                            queue_length: msg.queue_length,
                        });
                    } else if (msg.type === 'session.queue_done') {
                        queueDone = true;
                        this._queueReject = null;
                        this._logProtoEvent('server', 'session.queue_done', '', msg);
                        this.onQueueDone();
                        this.onQueueUpdate(null);
                        this.onSystemLog('Worker assigned, preparing...');
                        sendSessionInit();

                    // Backward compat: old protocol queue messages
                    } else if (msg.type === 'queued') {
                        this._logProtoEvent('server', 'queued (compat)', `pos=${msg.position}`, msg);
                        this.onQueueUpdate({
                            position: msg.position,
                            estimated_wait_s: msg.estimated_wait_s,
                            ticket_id: msg.ticket_id,
                            queue_length: msg.queue_length,
                        });
                    } else if (msg.type === 'queue_done') {
                        queueDone = true;
                        this._queueReject = null;
                        this._logProtoEvent('server', 'queue_done (compat)', '', msg);
                        this.onQueueDone();
                        this.onQueueUpdate(null);
                        this.onSystemLog('Worker assigned, preparing...');
                        sendSessionInit();

                    } else if (msg.type === 'session.created') {
                        this._queueReject = null;
                        this.sessionId = msg.session_id || '';
                        this.recordingSessionId = this.sessionId;
                        this._logProtoEvent('server', 'session.created',
                            `session_id=${this.sessionId}`, msg);
                        this.onQueueUpdate(null);
                        this.onMetrics({ type: 'state', sessionId: this.sessionId });
                        this.onSystemLog(`Session created: ${this.sessionId} (${msg.prompt_length || '?'} tokens)`);
                        resolve();
                    } else if (msg.type === 'error') {
                        this._queueReject = null;
                        this._logProtoEvent('server', 'error',
                            `${msg.error?.code}: ${msg.error?.message}`, msg);
                        const errMsg = msg.error?.message || msg.error || 'Unknown error';
                        reject(new Error(errMsg));
                    }
                };

                setTimeout(() => {
                    if (!queueDone) sendSessionInit();
                }, 100);
            });

            await this.onPrepared();
            this.audioPlayer.init();
            if (startMediaFn) await startMediaFn();

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
     * Send audio chunk using the API V2 protocol.
     * Accepts the OLD format { type: 'audio_chunk', audio_base64, ... }
     * and translates to the new { type: 'input.append', input: { audio, ... } }
     */
    sendChunk(msg) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        if (this.paused) return;

        const newMsg = {
            type: 'input.append',
            input: {
                audio: msg.audio_base64,
            },
        };

        if (this.forceListenActive || msg.force_listen) {
            newMsg.input.force_listen = true;
        }
        if (msg.frame_base64_list) {
            newMsg.input.video_frames = msg.frame_base64_list;
        }
        if (msg.max_slice_nums) {
            newMsg.input.max_slice_nums = msg.max_slice_nums;
        }

        this.ws.send(JSON.stringify(newMsg));
        this.chunksSent++;

        const hasVideo = newMsg.input.video_frames ? ` +${newMsg.input.video_frames.length}fr` : '';
        this._logProtoEvent('client', 'input.append',
            `#${this.chunksSent}${hasVideo}${newMsg.input.force_listen ? ' force' : ''}`, newMsg);

        this.onMetrics({ type: 'result', chunksSent: this.chunksSent });
    }

    toggleForceListen() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        this.forceListenActive = !this.forceListenActive;
        this.onForceListenChange(this.forceListenActive);
        if (this.forceListenActive) {
            this.onSystemLog('Force Listen ON');
            this.audioPlayer.stopAll();
            if (this.audioPlayer.turnActive) this.audioPlayer.endTurn();
        } else {
            if (this.audioPlayer.turnActive) this.audioPlayer.endTurn();
            this.onSystemLog('Force Listen OFF');
        }
    }

    pauseToggle() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        if (this.pauseState === 'active') {
            this.paused = true;
            this.pauseState = 'paused';
            this.onPauseStateChange('paused');
            this.onMetrics({ type: 'state', sessionState: 'Paused' });
            this.onSystemLog('Session paused');
        } else if (this.pauseState === 'paused') {
            this.paused = false;
            this.pauseState = 'active';
            this.onPauseStateChange('active');
            this.onMetrics({ type: 'state', sessionState: 'Active' });
            this.onSystemLog('Session resumed');
        }
    }

    stop() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const msg = { type: 'session.close', reason: 'user_stop' };
            this.ws.send(JSON.stringify(msg));
            this._logProtoEvent('client', 'session.close', 'user_stop');
        }
        this.cleanup();
    }

    cancelQueue() {
        const reject = this._queueReject;
        this._queueReject = null;
        this.cleanup();
        if (reject) reject(new Error('Queue cancelled by user'));
    }

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
        this._lastKvCacheLength = 0;
        this._lastFrameMetrics = {};
        this.chunksSent = 0;
        this.currentSpeakText = '';
        this._speakHandle = null;
        this.paused = false;
        this.pauseState = 'active';
        this.forceListenActive = false;
        this._queueReject = null;
        this._eventLog = [];
    }

    _handleMessage(msg) {
        const type = msg.type || '';

        switch (type) {
            case 'response.metrics':
                this._logProtoEvent('server', 'response.metrics',
                    `kv=${msg.kv_cache_length}`, msg);
                this._handleMetrics(msg);
                break;

            case 'response.listen':
                this._logProtoEvent('server', 'response.listen',
                    'listen', msg);
                this._handleListen(msg);
                break;

            case 'response.output_audio.delta':
                this._logProtoEvent('server', 'response.output_audio.delta',
                    `"${(msg.text||'').slice(0,30)}" eot=${msg.end_of_turn}`, msg);
                this._handleSpeak(msg);
                break;

            case 'response.output.delta':
                this._handleOutputDelta(msg);
                break;

            case 'session.closed':
                this._logProtoEvent('server', 'session.closed',
                    `reason=${msg.reason}`, msg);
                this.onSystemLog(`Session closed: ${msg.reason}`);
                this.cleanup();
                break;

            case 'error':
                this._logProtoEvent('server', 'error',
                    `${msg.error?.code}: ${msg.error?.message}`, msg);
                this.onSystemLog(`Error: ${msg.error?.message || msg.error}`);
                break;

            // Backward compat: old protocol events
            case 'result':
                this._handleResultCompat(msg);
                break;
            case 'stopped':
                this.onSystemLog('Session stopped');
                this.cleanup();
                break;
            case 'timeout':
                this.onSystemLog(`Timeout: ${msg.reason}`);
                this.cleanup();
                break;
            case 'queued':
            case 'queue_update':
            case 'session.queued':
            case 'session.queue_update':
                this.onQueueUpdate({
                    position: msg.position,
                    estimated_wait_s: msg.estimated_wait_s,
                    queue_length: msg.queue_length,
                });
                break;
            case 'queue_done':
            case 'session.queue_done':
                this.onQueueUpdate(null);
                break;
        }
    }

    /** Network drift: latency change vs the first result (ported from duplex-session.js). */
    _updateDrift(msg) {
        this._lastDriftMs = null;
        const serverSendSec = msg && msg.server_send_ts;
        if (!serverSendSec) return;
        const clientRecvSec = Date.now() / 1000;
        if (!this._firstServerTs) {
            this._firstServerTs = serverSendSec;
            this._firstClientTs = clientRecvSec;
        }
        this._lastDriftMs = (clientRecvSec - serverSendSec
            - (this._firstClientTs - this._firstServerTs)) * 1000;
    }

    _applyFrameMetrics(msg) {
        this._updateDrift(msg);
        if (msg && typeof msg.metrics === 'object' && msg.metrics !== null) {
            this._lastFrameMetrics = msg.metrics;
            return;
        }
        if (msg && (msg.kv_cache_length !== undefined || msg.wall_clock_ms !== undefined || msg.generate_ms !== undefined)) {
            this._lastFrameMetrics = {
                ...this._lastFrameMetrics,
                kv_cache_length: msg.kv_cache_length,
                wall_clock_ms: msg.wall_clock_ms,
                generate_ms: msg.generate_ms,
            };
        }
    }

    _handleOutputDelta(msg) {
        const kind = msg.kind || '';
        this._applyFrameMetrics(msg);
        this._logProtoEvent('server', `response.output.delta/${kind}`,
            kind === 'text' ? `"${(msg.text || '').slice(0, 30)}"`
                : kind === 'audio' ? `audio=${msg.audio ? msg.audio.length : 0}`
                : kind || 'unknown',
            msg);

        if (kind === 'listen') {
            this._handleListen(msg);
        } else if (kind === 'text') {
            this._handleSpeak({
                ...msg,
                audio: undefined,
            });
        } else if (kind === 'audio') {
            this._handleSpeak({
                ...msg,
                text: '',
            });
        }
    }

    /** Handle new protocol response.listen */
    _handleListen(msg) {
        this._applyFrameMetrics(msg);
        const recvTime = performance.now();
        this._resultCount++;
        this._lastListenTime = recvTime;
        this._wasListening = true;

        if (this.audioPlayer.turnActive) this.audioPlayer.endTurn();

        const result = {
            is_listen: true,
            kv_cache_length: this._lastFrameMetrics.kv_cache_length,
        };

        this._checkKvCache(result);
        this._emitMetrics(result, recvTime);

        if (this._speakHandle) {
            this.onSpeakEnd();
            this._speakHandle = null;
            this.currentSpeakText = '';
            this.onSystemLog('— end of turn —');
        }
        this.onListenResult(result);
        this.onExtraResult(result, recvTime);
        this._lastResultTime = recvTime;
    }

    /** Handle new protocol response.output_audio.delta */
    _handleSpeak(msg) {
        this._applyFrameMetrics(msg);
        const recvTime = performance.now();
        this._resultCount++;

        if (this._wasListening) {
            this._wasListening = false;
            this._lastTTFS = this._lastListenTime > 0
                ? recvTime - this._lastListenTime : 0;
        }

        if (msg.audio) {
            if (!this.audioPlayer.turnActive) this.audioPlayer.beginTurn();
            this.audioPlayer.playChunk(msg.audio, recvTime);
        }

        const result = {
            is_listen: false,
            text: msg.text || '',
            audio_data: msg.audio,
            end_of_turn: msg.end_of_turn || false,
            kv_cache_length: this._lastFrameMetrics.kv_cache_length,
        };

        this._checkKvCache(result);
        this._emitMetrics(result, recvTime);

        if (result.text) {
            this.currentSpeakText += result.text;
            if (!this._speakHandle) {
                this._speakHandle = this.onSpeakStart(this.currentSpeakText);
            } else {
                this.onSpeakUpdate(this._speakHandle, this.currentSpeakText);
            }
        }

        this.onExtraResult(result, recvTime);
        this._lastResultTime = recvTime;
    }

    /** Handle old protocol 'result' for backward compat (when gateway doesn't translate) */
    _handleResultCompat(result) {
        if (result.is_listen) {
            this._handleListen({
                kv_cache_length: result.kv_cache_length,
            });
        } else {
            this._handleSpeak({
                text: result.text,
                audio: result.audio_data,
                end_of_turn: result.end_of_turn,
                kv_cache_length: result.kv_cache_length,
            });
        }
    }

    _emitMetrics(result, recvTime) {
        const maxKv = this.config.getMaxKvTokens();
        const metrics = this._lastFrameMetrics || {};
        requestAnimationFrame(() => {
            this.onMetrics({
                type: 'result',
                latencyMs: metrics.wall_clock_ms || metrics.generate_ms,
                costAllMs: metrics.generate_ms,
                driftMs: this._lastDriftMs,
                kvCacheLength: result.kv_cache_length,
                maxKvTokens: maxKv,
                ttfsMs: (!result.is_listen && this._lastTTFS) ? this._lastTTFS : null,
                modelState: result.is_listen ? 'listening' : (result.end_of_turn ? 'end_of_turn' : 'speaking'),
                chunksSent: this.chunksSent,
                visionSlices: metrics.vision_slices,
                visionTokens: metrics.vision_tokens,
            });
            if (!result.is_listen && this._lastTTFS) this._lastTTFS = 0;
        });
    }

    _handleMetrics(metrics) {
        this._lastFrameMetrics = metrics || {};
        const maxKv = this.config.getMaxKvTokens();
        const kvCacheLength = this._lastFrameMetrics.kv_cache_length;
        this._checkKvCache({ kv_cache_length: kvCacheLength });
        this.onMetrics({
            type: 'result',
            latencyMs: this._lastFrameMetrics.wall_clock_ms || this._lastFrameMetrics.generate_ms,
            costAllMs: this._lastFrameMetrics.generate_ms,
            driftMs: this._lastDriftMs,
            kvCacheLength,
            maxKvTokens: maxKv,
            chunksSent: this.chunksSent,
            visionSlices: this._lastFrameMetrics.vision_slices,
            visionTokens: this._lastFrameMetrics.vision_tokens,
        });
    }

    _checkKvCache(result) {
        const maxKv = this.config.getMaxKvTokens();
        const curKv = result.kv_cache_length;
        if (curKv !== undefined && curKv > 0) {
            if (curKv >= maxKv) {
                this.onSystemLog(`⚠ KV cache (${curKv.toLocaleString()}) reached limit. Auto-stopping.`);
                setTimeout(() => this.stop(), 0);
            } else if (this._lastKvCacheLength > 0 && curKv < this._lastKvCacheLength) {
                const prev = this._lastKvCacheLength;
                this.onSystemLog(`✂ KV pruned: ${prev.toLocaleString()} → ${curKv.toLocaleString()}`);
                if (this.config.getStopOnSlidingWindow()) {
                    this.onSystemLog('⚠ Stop-on-sliding-window. Auto-stopping.');
                    setTimeout(() => this.stop(), 0);
                }
            }
            this._lastKvCacheLength = curKv;
        }
    }

}
