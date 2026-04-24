type AudioContextCtor = typeof AudioContext

function getAudioContextCtor(): AudioContextCtor {
  const ctor =
    window.AudioContext ??
    (window as Window & { webkitAudioContext?: AudioContextCtor })
      .webkitAudioContext

  if (!ctor) {
    throw new Error('AudioContext is not supported in this browser.')
  }

  return ctor
}

export class StreamingPcmPlayer {
  private readonly audioCtx: AudioContext
  private readonly sampleRate: number
  private readonly chunks: Float32Array[] = []
  private nextStartTime = 0
  private finished = false
  private disposed = false

  constructor(sampleRate = 24000) {
    this.sampleRate = sampleRate
    const Ctor = getAudioContextCtor()
    this.audioCtx = new Ctor({ sampleRate })
  }

  pushBase64(base64Data: string): void {
    if (this.disposed) {
      return
    }

    const binary = atob(base64Data)
    const bytes = new Uint8Array(binary.length)

    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index)
    }

    if (bytes.byteLength === 0) {
      return
    }

    const float32 = new Float32Array(bytes.buffer.slice(0))

    if (float32.length === 0) {
      return
    }

    this.chunks.push(float32)
    this.scheduleChunk(float32)
  }

  private scheduleChunk(float32: Float32Array): void {
    if (this.audioCtx.state === 'suspended') {
      void this.audioCtx.resume().catch(() => {})
    }

    const buffer = this.audioCtx.createBuffer(1, float32.length, this.sampleRate)
    buffer.getChannelData(0).set(float32)

    const source = this.audioCtx.createBufferSource()
    source.buffer = buffer
    source.connect(this.audioCtx.destination)

    const now = this.audioCtx.currentTime
    const when = Math.max(now + 0.02, this.nextStartTime)

    source.start(when)
    this.nextStartTime = when + buffer.duration
  }

  markFinished(): void {
    this.finished = true
  }

  isFinished(): boolean {
    return this.finished
  }

  getMergedFloat32(): Float32Array | null {
    if (this.chunks.length === 0) {
      return null
    }

    const total = this.chunks.reduce((sum, chunk) => sum + chunk.length, 0)
    const merged = new Float32Array(total)
    let offset = 0

    for (const chunk of this.chunks) {
      merged.set(chunk, offset)
      offset += chunk.length
    }

    return merged
  }

  getSampleRate(): number {
    return this.sampleRate
  }

  async dispose(): Promise<void> {
    this.disposed = true

    try {
      await this.audioCtx.close()
    } catch {
      /* ignore */
    }
  }

  /**
   * 等所有已排队的 chunk 自然播完再 close AudioContext。
   * 与 dispose() 不同，这不会切断仍在排队的回放。
   * 调用前应先 markFinished() 表示不会再有新 chunk push 进来。
   */
  disposeAfterDrain(onDrained?: () => void): void {
    if (this.disposed) {
      onDrained?.()
      return
    }

    const now = this.audioCtx.currentTime
    const drainSeconds = Math.max(0, this.nextStartTime - now)
    const safetyPadMs = 500
    const delayMs = Math.ceil(drainSeconds * 1000) + safetyPadMs

    setTimeout(() => {
      void this.dispose()
      onDrained?.()
    }, delayMs)
  }
}

export function float32ToWavBlobUrl(
  float32: Float32Array,
  sampleRate: number,
): string {
  const buffer = new ArrayBuffer(44 + float32.length * 2)
  const view = new DataView(buffer)

  function writeString(offset: number, value: string) {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index))
    }
  }

  writeString(0, 'RIFF')
  view.setUint32(4, 36 + float32.length * 2, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeString(36, 'data')
  view.setUint32(40, float32.length * 2, true)

  let offset = 44

  for (let index = 0; index < float32.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, float32[index] ?? 0))
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
    offset += 2
  }

  return URL.createObjectURL(new Blob([buffer], { type: 'audio/wav' }))
}
