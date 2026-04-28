import type { RefAudioState, PresetMetadata } from './settings-types'
import { EMPTY_REF_AUDIO } from './settings-types'

export function createId(prefix: string): string {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`
}

export function cloneRefAudio(r: RefAudioState): RefAudioState {
  return { ...r }
}

export function extractRefAudioFromPreset(preset: PresetMetadata): RefAudioState {
  if (preset.ref_audio?.data) {
    return {
      source: 'preset',
      name: preset.ref_audio.name || '预设参考音频',
      duration: preset.ref_audio.duration || 0,
      base64: preset.ref_audio.data,
    }
  }
  if (preset.system_content) {
    for (const item of preset.system_content) {
      if (item.type === 'audio' && item.data) {
        return {
          source: 'preset',
          name: item.name || '预设参考音频',
          duration: item.duration || 0,
          base64: item.data,
        }
      }
    }
  }
  return cloneRefAudio(EMPTY_REF_AUDIO)
}

export function extractPromptFromPreset(preset: PresetMetadata): string {
  if (preset.system_prompt) return preset.system_prompt
  if (preset.system_content) {
    return preset.system_content
      .filter((it) => it.type === 'text' && it.text)
      .map((it) => it.text!)
      .join('\n\n')
      .trim()
  }
  return ''
}

export function getAudioContextCtor(): typeof AudioContext | null {
  return (
    window.AudioContext ??
    (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext ??
    null
  )
}

export function float32ToBase64(samples: Float32Array): string {
  const bytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength)
  const chunkSize = 0x8000
  let binary = ''
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const slice = bytes.subarray(offset, Math.min(offset + chunkSize, bytes.length))
    binary += String.fromCharCode.apply(null, Array.from(slice) as number[])
  }
  return btoa(binary)
}

export function concatFloat32(chunks: Float32Array[]): Float32Array {
  let total = 0
  for (const chunk of chunks) total += chunk.length
  const out = new Float32Array(total)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.length
  }
  return out
}

export function resampleLinear(input: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate || input.length === 0) return input
  const ratio = fromRate / toRate
  const outLength = Math.max(1, Math.floor(input.length / ratio))
  const out = new Float32Array(outLength)
  for (let i = 0; i < outLength; i++) {
    const srcPos = i * ratio
    const idx = Math.floor(srcPos)
    const frac = srcPos - idx
    const a = input[idx] ?? 0
    const b = input[idx + 1] ?? a
    out[i] = a + (b - a) * frac
  }
  return out
}

export async function convertAudioBlobToFloat32Base64(blob: Blob): Promise<string> {
  const Ctor = getAudioContextCtor()
  if (!Ctor) throw new Error('AudioContext not supported')
  const ctx = new Ctor()
  try {
    const buf = await blob.arrayBuffer()
    const decoded = await ctx.decodeAudioData(buf)
    const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000)
    const src = offline.createBufferSource()
    src.buffer = decoded
    src.connect(offline.destination)
    src.start()
    const rendered = await offline.startRendering()
    return float32ToBase64(rendered.getChannelData(0))
  } finally {
    await ctx.close()
  }
}
