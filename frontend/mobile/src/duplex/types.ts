import type { ComponentType } from 'react'

export type DuplexEntry = {
  id: string
  role: 'assistant' | 'user' | 'system'
  text: string
}

export type DuplexStatus =
  | 'idle'
  | 'starting'
  | 'queueing'
  | 'live'
  | 'paused'
  | 'stopped'
  | 'error'

export type DuplexMode = 'audio' | 'video'

export type DuplexScreenName = 'audio-duplex' | 'video-duplex'

export type DuplexPauseState = 'active' | 'pausing' | 'paused'

// Structural subset of App.SettingsState. App passes a wider object that
// satisfies this shape via TypeScript structural typing.
export type DuplexRefAudio = {
  source: 'none' | 'default' | 'preset' | 'upload'
  name: string
  duration: number
  base64: string | null
}

export type DuplexModeSettings = {
  presetId: string | null
  systemPrompt: string
  refAudio: DuplexRefAudio
}

export type DuplexSettings = {
  audio_duplex: DuplexModeSettings
  omni: DuplexModeSettings
  audioDuplexLengthPenalty: number
  videoDuplexLengthPenalty: number
}

// Components / icons that screens accept from App via props (alpha mode:
// keep App.tsx untouched outside duplex code, including its icon
// definitions and SettingsSummary).
export type DuplexIconComponent = ComponentType<{ className?: string }>

export type DuplexIcons = {
  Settings: DuplexIconComponent
  Transcript: DuplexIconComponent
  Mic: DuplexIconComponent
  Pause: DuplexIconComponent
  Play: DuplexIconComponent
  Close: DuplexIconComponent
  Wave: DuplexIconComponent
  FlipCamera: DuplexIconComponent
}

export type SettingsSummaryComponent = ComponentType<{
  modeLabel: string
  presetName: string
  refAudio: DuplexRefAudio
  systemPrompt: string
  lengthPenalty: number
  onOpen: () => void
}>
