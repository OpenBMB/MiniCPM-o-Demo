export type RefAudioState = {
  source: 'none' | 'default' | 'preset' | 'upload'
  name: string
  duration: number
  base64: string | null
}

export type PresetMetadata = {
  id: string
  name: string
  description?: string
  system_prompt?: string
  system_content?: { type: string; text?: string; data?: string; path?: string; name?: string; duration?: number }[]
  ref_audio?: {
    data?: string | null
    path?: string
    name?: string
    duration?: number
  }
}

export type UserPreset = {
  id: string
  name: string
  systemPrompt: string
  refAudio: RefAudioState
  createdAt: number
  updatedAt: number
}

export const EMPTY_REF_AUDIO: RefAudioState = {
  source: 'none',
  name: '未设置',
  duration: 0,
  base64: null,
}

export type OmniBridge = {
  getSystemPrompt: () => string
  setSystemPrompt: (v: string) => void
  getLengthPenalty: () => number
  setLengthPenalty: (v: number) => void
  getPlaybackDelay: () => number
  setPlaybackDelay: (v: number) => void
  getMaxKv: () => number
  setMaxKv: (v: number) => void
  getStopOnPrune: () => boolean
  setStopOnPrune: (v: boolean) => void
  getRefAudioBase64: () => string | null
  setRefAudioBase64: (b64: string | null, name: string, duration: number) => void
}
