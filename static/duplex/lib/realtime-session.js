/**
 * lib/realtime-session.js — OpenAI Realtime-style session manager
 *
 * Drop-in replacement for DuplexSession that speaks the new protocol:
 *   session.update / input_audio_buffer.append / response.listen / response.output_audio.delta
 *
 * Same callback interface as DuplexSession so UI code can swap with zero changes.
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
                return `${proto}://${location.host}/v1/realtime`;
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

            // Wait for queue + send session.update
            await new Promise((resolve, reject) => {
                let queueDone = false;
                let updateSent = false;
                this._queueReject = reject;

                const sendSessionUpdate = () => {
                    if (updateSent) return;
                    updateSent = true;

                    const sessionUpdate = {
                        type: 'session.update',
                        session: {
                            instructions: systemPrompt,
                            ...preparePayload,
                        },
                    };
                    this.ws.send(JSON.stringify(sessionUpdate));
                    this._logProtoEvent('client', 'session.update',
                        `instructions="${systemPrompt.slice(0, 40)}…"`, sessionUpdate);
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
                        sendSessionUpdate();

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
                        sendSessionUpdate();

                    } else if (msg.type === 'session.created') {
                        this._queueReject = null;
                        this.sessionId = msg.session_id || '';
                        this._logProtoEvent('server', 'session.created',
                            `session_id=${this.sessionId}`, msg);
                        this.onQueueUpdate(null);
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
                    if (!queueDone) sendSessionUpdate();
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
     * Send audio chunk using the new protocol.
     * Accepts the OLD format { type: 'audio_chunk', audio_base64, ... }
     * and translates to the new { type: 'input_audio_buffer.append', audio, ... }
     */
    sendChunk(msg) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        if (this.paused) return;

        const newMsg = {
            type: 'input_audio_buffer.append',
            audio: msg.audio_base64,
        };

        if (this.forceListenActive || msg.force_listen) {
            newMsg.force_listen = true;
        }
        if (msg.frame_base64_list) {
            newMsg.video_frames = msg.frame_base64_list;
        }
        if (msg.max_slice_nums) {
            newMsg.max_slice_nums = msg.max_slice_nums;
        }

        this.ws.send(JSON.stringify(newMsg));
        this.chunksSent++;

        const hasVideo = newMsg.video_frames ? ` +${newMsg.video_frames.length}fr` : '';
        this._logProtoEvent('client', 'input_audio_buffer.append',
            `#${this.chunksSent}${hasVideo}${newMsg.force_listen ? ' force' : ''}`);

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
        this.forceListenActive = false;
        this.onRunningChange(false);
        this.onForceListenChange(false);
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
        this.chunksSent = 0;
        this.currentSpeakText = '';
        this._speakHandle = null;
        this.paused = false;
        this.forceListenActive = false;
        this._queueReject = null;
        this._eventLog = [];
    }

    _handleMessage(msg) {
        const type = msg.type || '';

        switch (type) {
            case 'response.listen':
                this._logProtoEvent('server', 'response.listen',
                    `kv=${msg.kv_cache_length}`, msg);
                this._handleListen(msg);
                break;

            case 'response.output_audio.delta':
                this._logProtoEvent('server', 'response.output_audio.delta',
                    `"${(msg.text||'').slice(0,30)}" eot=${msg.end_of_turn}`, msg);
                this._handleSpeak(msg);
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

    /** Handle new protocol response.listen */
    _handleListen(msg) {
        const recvTime = performance.now();
        this._resultCount++;
        this._lastListenTime = recvTime;
        this._wasListening = true;

        if (this.audioPlayer.turnActive) this.audioPlayer.endTurn();

        const result = {
            is_listen: true,
            kv_cache_length: msg.kv_cache_length,
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
            kv_cache_length: msg.kv_cache_length,
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
        requestAnimationFrame(() => {
            this.onMetrics({
                type: 'result',
                driftMs: this._lastDriftMs,
                kvCacheLength: result.kv_cache_length,
                maxKvTokens: maxKv,
                ttfsMs: (!result.is_listen && this._lastTTFS) ? this._lastTTFS : null,
                modelState: result.is_listen ? 'listening' : (result.end_of_turn ? 'end_of_turn' : 'speaking'),
                chunksSent: this.chunksSent,
            });
            if (!result.is_listen && this._lastTTFS) this._lastTTFS = 0;
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
