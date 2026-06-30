const SAMPLE_RATE = 16000;
const CHUNK_SAMPLES = SAMPLE_RATE;

export async function decodeFileToChunks(file, { padBeforeSec = 0, padAfterSec = 2 } = {}) {
  const arrayBuffer = await file.arrayBuffer();
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
  const mono = mixToMono(decoded);
  const resampled = await resampleMono(mono, decoded.sampleRate, SAMPLE_RATE);
  await audioContext.close();

  const silence = () => new Float32Array(CHUNK_SAMPLES);
  const chunks = [];
  for (let i = 0; i < Math.max(0, Math.floor(padBeforeSec)); i++) chunks.push(silence());
  for (let start = 0; start < resampled.length; start += CHUNK_SAMPLES) {
    const chunk = new Float32Array(CHUNK_SAMPLES);
    chunk.set(resampled.slice(start, Math.min(start + CHUNK_SAMPLES, resampled.length)));
    chunks.push(chunk);
  }
  for (let i = 0; i < Math.max(0, Math.floor(padAfterSec)); i++) chunks.push(silence());
  return chunks;
}

export function float32ToBase64(float32Array) {
  const bytes = new Uint8Array(float32Array.buffer, float32Array.byteOffset, float32Array.byteLength);
  let binary = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

function mixToMono(audioBuffer) {
  const length = audioBuffer.length;
  const channels = audioBuffer.numberOfChannels;
  const out = new Float32Array(length);
  for (let ch = 0; ch < channels; ch++) {
    const data = audioBuffer.getChannelData(ch);
    for (let i = 0; i < length; i++) out[i] += data[i] / channels;
  }
  return out;
}

async function resampleMono(samples, srcRate, dstRate) {
  if (srcRate === dstRate) return samples;
  const duration = samples.length / srcRate;
  const frameCount = Math.max(1, Math.ceil(duration * dstRate));
  const offline = new OfflineAudioContext(1, frameCount, dstRate);
  const buffer = offline.createBuffer(1, samples.length, srcRate);
  buffer.copyToChannel(samples, 0);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start(0);
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0).slice();
}
