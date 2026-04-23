/**
 * pcm-capture-turnbased.js
 *
 * Tiny AudioWorklet used by the mobile turn-based "press to talk"
 * recorder. It forwards raw PCM render quanta back to the main thread
 * while capture is enabled.
 *
 * We intentionally avoid ScriptProcessorNode here because Android
 * Chrome can leave it in a bad cold-start state where
 * `onaudioprocess` never fires even though the AudioContext is
 * already "running".
 *
 * Commands via port.postMessage:
 *   { type: 'capture', value: boolean }
 *
 * Emits via port.postMessage:
 *   { type: 'pcm', samples: Float32Array, frame: number }
 */
class PcmCaptureTurnbasedProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._capturing = false;
        this._frame = 0;
        this.port.onmessage = (event) => {
            const data = event.data;
            if (!data || typeof data !== 'object') return;
            if (data.type === 'capture') {
                this._capturing = !!data.value;
            }
        };
    }

    process(inputs) {
        const input = inputs[0]?.[0];
        if (!this._capturing || !input || input.length === 0) {
            return true;
        }

        const copy = new Float32Array(input.length);
        copy.set(input);
        this._frame += 1;
        this.port.postMessage(
            { type: 'pcm', samples: copy, frame: this._frame },
            [copy.buffer]
        );
        return true;
    }
}

registerProcessor('pcm-capture-turnbased', PcmCaptureTurnbasedProcessor);
