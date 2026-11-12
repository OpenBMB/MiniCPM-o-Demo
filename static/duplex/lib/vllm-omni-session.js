import { RealtimeSession } from './realtime-session.js';
import { arrayBufferToBase64 } from './duplex-utils.js';

function decodeBase64(data) {
    return Uint8Array.from(atob(data), char => char.charCodeAt(0));
}

function pcm16(data) {
    const samples = new Float32Array(decodeBase64(data).buffer);
    const pcm = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
        const value = Math.max(-1, Math.min(1, samples[i]));
        pcm[i] = Math.round(value * (value < 0 ? 32768 : 32767));
    }
    return pcm;
}

function referenceWav(data) {
    const pcm = pcm16(data);
    const buffer = new ArrayBuffer(44 + pcm.byteLength);
    const view = new DataView(buffer);
    const text = (offset, value) => [...value].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
    text(0, 'RIFF'); view.setUint32(4, 36 + pcm.byteLength, true); text(8, 'WAVE');
    text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, 16000, true);
    view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    text(36, 'data'); view.setUint32(40, pcm.byteLength, true);
    new Uint8Array(buffer, 44).set(new Uint8Array(pcm.buffer));
    return `data:audio/wav;base64,${arrayBufferToBase64(buffer)}`;
}

export class VllmOmniSession extends RealtimeSession {
    constructor(prefix, config, backend) {
        super(prefix, config);
        this.backend = backend;
        this.responses = new Map();
    }

    async start(systemPrompt, payload, startMediaFn) {
        this._reset();
        const url = new URL(this.backend.url);
        url.searchParams.set('duplex', '1');
        url.searchParams.set('autostart', '0');
        url.searchParams.set('model', this.backend.model);
        const session = {
            modalities: ['audio', 'text'], instructions: systemPrompt,
            overlap_policy: 'listen_only', playback_commit_policy: 'ack_only',
            extra_body: { auto_response: true, force_listen_count: 0 },
        };
        if (payload.ref_audio_base64) session.ref_audio = referenceWav(payload.ref_audio_base64);
        this.onMetrics({ type: 'state', sessionState: 'Connecting...' });
        try {
            await new Promise((resolve, reject) => {
                const timer = setTimeout(() => reject(new Error('vLLM-Omni handshake timed out')), 30000);
                const finish = (error) => {
                    clearTimeout(timer);
                    this._openingReject = null;
                    if (error) reject(error); else resolve();
                };
                this._openingReject = finish;
                this.ws = new WebSocket(url);
                this.ws.onopen = () => this._send({ type: 'session.update', session });
                this.ws.onerror = () => finish(new Error('Cannot connect to vLLM-Omni'));
                this.ws.onclose = () => finish(new Error('vLLM-Omni closed before the session was ready'));
                this.ws.onmessage = ({ data }) => {
                    const event = JSON.parse(data);
                    if (event.type === 'error') finish(new Error(event.error?.message || event.message));
                    if (event.type === 'session.updated') {
                        this.sessionId = event.session?.id || event.session_id;
                        finish();
                    }
                };
            });
            this.onMetrics({ type: 'state', sessionId: this.sessionId });
            this.audioPlayer.init();
            this.ws.onmessage = ({ data }) => this._handleMessage(JSON.parse(data));
            this.ws.onclose = () => { this.onSystemLog('Session ended'); this.cleanup(); };
            await this.onPrepared();
            if (this.ws?.readyState !== WebSocket.OPEN) throw new Error('Session cancelled');
            this._started = true;
            this.onRunningChange(true);
            if (startMediaFn) await startMediaFn();
            if (!this.running) throw new Error('Session cancelled');
        } catch (error) {
            this.cleanup();
            throw error;
        }
    }

    _send(event) {
        if (this.ws?.readyState !== WebSocket.OPEN) return;
        this.ws.send(JSON.stringify(event));
        this._logProtoEvent('client', event.type, '', event);
    }

    sendChunk(chunk) {
        if (!this.running || this.paused) return;
        const event = { type: 'input_audio_buffer.append', format: 'pcm16', sample_rate_hz: 16000 };
        if (chunk.audio_base64) event.audio = arrayBufferToBase64(pcm16(chunk.audio_base64).buffer);
        if (chunk.frame_base64_list) event.video_frames = chunk.frame_base64_list;
        if (this.forceListenActive || chunk.force_listen) event.force_listen = true;
        this._send(event);
        this.chunksSent++;
        this.onMetrics({ type: 'result', chunksSent: this.chunksSent });
    }

    _response(id) {
        if (!this.responses.has(id)) {
            this.responses.set(id, { id, pending: 0, playedMs: 0, done: false, interrupted: false });
        }
        return this.responses.get(id);
    }

    _ack(response, interrupted = false) {
        this._send({
            type: 'playback.ack', response_id: response.id, item_id: `item_${response.id}`,
            played_ms: Math.round(response.playedMs), committed_ms: Math.round(response.playedMs),
            ...(interrupted ? { truncate: true } : {}),
        });
    }

    _finishResponse(response) {
        if (!response.done || response.pending) return;
        this.responses.delete(response.id);
        if (!this.responses.size) this.audioPlayer.endTurn();
    }

    _handleMessage(event) {
        this._logProtoEvent('server', event.type, '', event);
        const id = event.response_id || event.response?.id;
        switch (event.type) {
            case 'response.created':
                this._response(id);
                break;
            case 'response.audio.delta':
            case 'response.output_audio.delta': {
                const response = this._response(id);
                if (response.interrupted) return;
                const pcm = new Int16Array(decodeBase64(event.delta).buffer);
                const audio = Float32Array.from(pcm, value => value / 32768);
                response.pending++;
                if (!this.audioPlayer.turnActive) this.audioPlayer.beginTurn();
                // Keep later responses behind audio already scheduled in the player.
                this.audioPlayer.playChunk(arrayBufferToBase64(audio.buffer), performance.now(), (playedMs, interrupted) => {
                    response.pending--;
                    response.playedMs += playedMs;
                    if (!interrupted) this._ack(response);
                    this._finishResponse(response);
                });
                break;
            }
            case 'response.audio_transcript.delta':
            case 'response.output_audio_transcript.delta':
                if (!this.responses.get(id)?.interrupted) this._handleSpeak({ text: event.delta });
                break;
            case 'response.done': {
                const response = this._response(id);
                response.done = true;
                if (event.response?.status === 'failed') {
                    this.onSystemLog(`Response failed: ${JSON.stringify(event.response.status_details)}`);
                }
                this._finishResponse(response);
                this.onSpeakEnd();
                this._speakHandle = null;
                this.currentSpeakText = '';
                break;
            }
            case 'response.listen':
                this._lastListenTime = performance.now();
                this._wasListening = true;
                this.onListenResult({ is_listen: true });
                break;
            case 'error':
                this.onSystemLog(`Error: ${event.error?.message || event.message}`);
                this.cleanup();
                break;
            case 'session.closed':
                this.cleanup();
                break;
        }
    }

    toggleForceListen() {
        if (!this.running) return;
        this.forceListenActive = !this.forceListenActive;
        this.onForceListenChange(this.forceListenActive);
        if (this.forceListenActive) {
            this._interruptPlayback();
        }
    }

    _interruptPlayback() {
        const responses = [...this.responses.values()];
        for (const response of responses) response.interrupted = true;
        this.audioPlayer.stopAll();
        for (const response of responses) {
            this._ack(response, true);
            if (!response.done) this._send({ type: 'response.cancel', response_id: response.id });
        }
    }

    cleanup() {
        if (!this.ws && !this.running) return;
        this._openingReject?.(new Error('Session cancelled'));
        this._interruptPlayback();
        this.responses.clear();
        super.cleanup();
    }

    stop() {
        this._interruptPlayback();
        this.responses.clear();
        this._send({ type: 'session.close' });
        this.cleanup();
    }
}
