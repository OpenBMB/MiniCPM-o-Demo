const TARGET_SAMPLE_RATE = 16000;

export class LiveMicProvider {
  constructor({ onChunk, onStatus }) {
    this.onChunk = onChunk;
    this.onStatus = onStatus;
    this.audioContext = null;
    this.stream = null;
    this.source = null;
    this.processor = null;
    this.pending = new Float32Array(0);
    this.running = false;
  }

  async start() {
    if (this.running) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    this.source = this.audioContext.createMediaStreamSource(this.stream);
    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
    this.processor.onaudioprocess = (event) => this._handleAudio(event.inputBuffer.getChannelData(0));
    this.source.connect(this.processor);
    this.processor.connect(this.audioContext.destination);
    this.running = true;
    this.onStatus?.(`Mic live @ ${Math.round(this.audioContext.sampleRate)}Hz`);
  }

  stop() {
    this.running = false;
    if (this.processor) {
      this.processor.disconnect();
      this.processor.onaudioprocess = null;
    }
    if (this.source) this.source.disconnect();
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
    }
    if (this.audioContext) this.audioContext.close().catch(() => {});
    this.audioContext = null;
    this.stream = null;
    this.source = null;
    this.processor = null;
    this.pending = new Float32Array(0);
  }

  _handleAudio(input) {
    if (!this.running || !this.audioContext) return;
    this.pending = concatFloat32(this.pending, input);
    const sourceRate = this.audioContext.sampleRate;
    const sourceSamplesPerUnit = Math.round(sourceRate);
    while (this.pending.length >= sourceSamplesPerUnit) {
      const oneSecond = this.pending.slice(0, sourceSamplesPerUnit);
      this.pending = this.pending.slice(sourceSamplesPerUnit);
      const resampled = resampleLinear(oneSecond, sourceRate, TARGET_SAMPLE_RATE);
      this.onChunk?.(resampled);
    }
  }
}

function concatFloat32(left, right) {
  const out = new Float32Array(left.length + right.length);
  out.set(left, 0);
  out.set(right, left.length);
  return out;
}

function resampleLinear(samples, sourceRate, targetRate) {
  if (sourceRate === targetRate) return samples;
  const ratio = sourceRate / targetRate;
  const outLength = Math.max(1, Math.round(samples.length / ratio));
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const pos = i * ratio;
    const lo = Math.floor(pos);
    const hi = Math.min(samples.length - 1, lo + 1);
    const frac = pos - lo;
    out[i] = samples[lo] * (1 - frac) + samples[hi] * frac;
  }
  return out;
}
