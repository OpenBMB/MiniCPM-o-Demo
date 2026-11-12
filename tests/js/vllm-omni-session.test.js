import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { RealtimeSession } from '../../static/duplex/lib/realtime-session.js';
import { VllmOmniSession } from '../../static/duplex/lib/vllm-omni-session.js';
import { loadRealtimeBackend } from '../../static/duplex/ui/vllm-omni.js';
import { initHealthCheck } from '../../static/duplex/ui/duplex-ui.js';

const base64 = array => Buffer.from(array.buffer).toString('base64');
let session;

class Socket {
    static OPEN = 1;
    constructor(url) {
        this.url = new URL(url);
        this.readyState = 1;
        this.sent = [];
        queueMicrotask(() => this.onopen());
    }
    send(data) {
        const event = JSON.parse(data);
        this.sent.push(event);
        if (event.type === 'session.update') {
            queueMicrotask(() => this.receive({ type: 'session.updated', session: { id: 'session-1' } }));
        }
    }
    receive(event) { this.onmessage({ data: JSON.stringify(event) }); }
    close() { this.readyState = 3; }
}

class AudioContext {
    constructor() {
        this.currentTime = 0;
        this.sampleRate = 24000;
        this.state = 'running';
        this.sources = [];
    }
    createBuffer(channels, length, rate) {
        return { duration: length / rate, getChannelData: () => new Float32Array(length) };
    }
    createBufferSource() {
        const source = { connect() {}, disconnect() {}, start: vi.fn(), stop: vi.fn() };
        this.sources.push(source);
        return source;
    }
}

async function connect() {
    session = new VllmOmniSession('test', { getPlaybackDelayMs: () => 0 }, {
        url: 'ws://localhost:8097/v1/realtime', model: 'minicpmo',
    });
    await session.start('Talk naturally', {}, async () => {});
    return session;
}

const audio = id => ({
    type: 'response.audio.delta', response_id: id, delta: base64(new Int16Array(24000).fill(1000)),
});
const done = id => ({ type: 'response.done', response: { id, status: 'completed' } });
const acks = socket => socket.sent.filter(event => event.type === 'playback.ack');

beforeEach(() => {
    vi.stubGlobal('WebSocket', Socket);
    vi.stubGlobal('AudioContext', AudioContext);
    vi.stubGlobal('requestAnimationFrame', callback => callback());
});

afterEach(() => {
    session?.cleanup();
    session = null;
    vi.unstubAllGlobals();
});

describe('optional vLLM-Omni session', () => {
    it('keeps the provider session when Omni is not configured', async () => {
        vi.stubGlobal('fetch', async () => ({ ok: true, json: async () => ({}) }));
        const backend = await loadRealtimeBackend();
        expect(backend.Session).toBe(RealtimeSession);
        expect(backend.initHealthCheck).toBe(initHealthCheck);
    });

    it('converts captured float audio to PCM16 and preserves camera frames', async () => {
        await connect();
        const socket = session.ws;
        expect(socket.url.searchParams.get('duplex')).toBe('1');
        expect(socket.sent[0].session.playback_commit_policy).toBe('ack_only');
        session.sendChunk({
            audio_base64: base64(new Float32Array([-2, -0.5, 0, 0.5, 2])),
            frame_base64_list: ['jpeg-frame'], force_listen: true,
        });
        const event = socket.sent.at(-1);
        expect(event.type).toBe('input_audio_buffer.append');
        expect(event.sample_rate_hz).toBe(16000);
        const bytes = Buffer.from(event.audio, 'base64');
        expect(Array.from({ length: 5 }, (_, i) => bytes.readInt16LE(i * 2)))
            .toEqual([-32768, -16384, 0, 16384, 32767]);
        expect(event.video_frames).toEqual(['jpeg-frame']);
        expect(event.force_listen).toBe(true);
    });

    it('wraps reference samples in a mono 16-kHz WAV for Omni', async () => {
        await connect();
        session.cleanup();
        await session.start('Voice', { ref_audio_base64: base64(new Float32Array([0, 0.5])) });
        const wav = Buffer.from(session.ws.sent[0].session.ref_audio.split(',')[1], 'base64');
        expect(wav.toString('ascii', 0, 4)).toBe('RIFF');
        expect(wav.readUInt32LE(24)).toBe(16000);
        expect(wav.readUInt16LE(22)).toBe(1);
        expect(wav.readUInt32LE(40)).toBe(4);
    });

    it('acknowledges played audio and keeps consecutive responses in order', async () => {
        await connect();
        const socket = session.ws;
        socket.receive(audio('first'));
        socket.receive(done('first'));
        socket.receive(audio('second'));
        socket.receive(done('second'));
        expect(acks(socket)).toEqual([]);
        const context = session.audioPlayer.ctx;
        expect(context.sources.map(source => source.start.mock.calls[0][0])).toEqual([0, 1]);
        expect(context.sources[0].stop).not.toHaveBeenCalled();
        context.currentTime = 1;
        context.sources[0].onended();
        context.currentTime = 2;
        context.sources[1].onended();
        expect(acks(socket).map(event => [event.response_id, event.played_ms]))
            .toEqual([['first', 1000], ['second', 1000]]);
    });

    it('reports only the played prefix when force-listen interrupts queued audio', async () => {
        await connect();
        const socket = session.ws;
        socket.receive(audio('first'));
        socket.receive(audio('first'));
        session.audioPlayer.ctx.currentTime = 0.5;
        session.toggleForceListen();
        expect(acks(socket)).toEqual([expect.objectContaining({ played_ms: 500, truncate: true })]);
        expect(socket.sent).toContainEqual({ type: 'response.cancel', response_id: 'first' });
        socket.receive(audio('first'));
        expect(session.audioPlayer.ctx.sources).toHaveLength(2);
        session.sendChunk({ audio_base64: base64(new Float32Array(16000)) });
        expect(socket.sent.at(-1).force_listen).toBe(true);
    });

    it('cleans up a failed connection and allows a replacement session', async () => {
        await connect();
        const socket = session.ws;
        socket.receive({ type: 'error', error: { message: 'Session capacity exhausted' } });
        expect(socket.readyState).toBe(3);
        expect(session.running).toBe(false);
        expect(session.responses.size).toBe(0);
        await session.start('Retry', {});
        expect(session.running).toBe(true);
    });

    it('reports interrupted playback before closing the server session', async () => {
        await connect();
        const socket = session.ws;
        socket.receive(audio('first'));
        session.audioPlayer.ctx.currentTime = 0.25;
        session.stop();
        expect(acks(socket)).toEqual([expect.objectContaining({ played_ms: 250, truncate: true })]);
        expect(socket.sent.at(-1)).toEqual({ type: 'session.close' });
        expect(session.running).toBe(false);
    });

    it('can interrupt generation before the first audio arrives', async () => {
        await connect();
        const socket = session.ws;
        socket.receive({ type: 'response.created', response: { id: 'first' } });
        session.toggleForceListen();
        expect(socket.sent).toContainEqual({ type: 'response.cancel', response_id: 'first' });
        socket.receive(audio('first'));
        expect(session.audioPlayer.ctx.sources).toHaveLength(0);
    });

    it('discards buffered audio when stopped before playback begins', async () => {
        await connect();
        session.audioPlayer._getDelayMs = () => 200;
        const socket = session.ws;
        socket.receive(audio('first'));
        socket.receive(done('first'));
        session.stop();
        expect(session.audioPlayer.ctx.sources).toHaveLength(0);
        expect(acks(socket)).toEqual([expect.objectContaining({ played_ms: 0, truncate: true })]);
    });
});
