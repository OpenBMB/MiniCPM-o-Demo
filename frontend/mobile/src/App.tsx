import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
} from 'react'
import {
  MobileLiveMediaProvider,
  loadDuplexRuntime,
  type DuplexResultLike,
  type DuplexSessionLike,
} from './mobile-duplex'
import { StreamingPcmPlayer, float32ToWavBlobUrl } from './streaming-player'
import './App.css'

type BackendContentItem =
  | {
      type: 'text'
      text: string
    }
  | {
      type: 'audio'
      data: string
      path?: string
      name?: string
      duration?: number
    }
  | {
      type: 'image'
      data: string
    }
  | {
      type: 'video'
      data: string
      duration?: number
    }

type BackendMessage = {
  role: 'assistant' | 'user' | 'system'
  content: string | BackendContentItem[]
}

type ConversationEntry =
  | {
      id: string
      role: 'assistant'
      kind: 'assistant'
      text: string
      error?: boolean
      audioPreviewUrl?: string | null
      recordingSessionId?: string | null
    }
  | {
      id: string
      role: 'user'
      kind: 'text'
      text: string
    }
  | {
      id: string
      role: 'user'
      kind: 'voice'
      audioBase64: string
      durationMs: number
      previewUrl: string
    }

type PendingReply = {
  id: string
  role: 'assistant'
  kind: 'pending'
  text: string
}

type ThreadEntry = ConversationEntry | PendingReply

type ServiceStatusResponse = {
  gateway_healthy: boolean
  total_workers: number
  idle_workers: number
  busy_workers: number
  queue_length: number
  offline_workers: number
}

type ServiceState = {
  phase: 'loading' | 'ready' | 'error'
  summary: string
  detail: string
}

type DuplexEntry = {
  id: string
  role: 'assistant' | 'user' | 'system'
  text: string
}

type DuplexStatus =
  | 'idle'
  | 'starting'
  | 'queueing'
  | 'live'
  | 'paused'
  | 'stopped'
  | 'error'

type Screen = 'turn' | 'audio-duplex' | 'video-duplex'

type DuplexMode = 'audio' | 'video'

type PresetMode = 'turnbased' | 'audio_duplex' | 'omni'

type RefAudioState = {
  source: 'none' | 'default' | 'preset' | 'upload'
  name: string
  duration: number
  base64: string | null
}

type PresetMetadata = {
  id: string
  order?: number
  name: string
  description?: string
  system_prompt?: string
  system_content?: BackendContentItem[]
  ref_audio?: {
    data?: string | null
    path?: string
    name?: string
    duration?: number
  }
}

type ModeSettings = {
  presetId: string | null
  systemPrompt: string
  refAudio: RefAudioState
}

type SettingsState = {
  turnbased: ModeSettings
  audio_duplex: ModeSettings
  omni: ModeSettings
  maxNewTokens: number
  turnLengthPenalty: number
  audioDuplexLengthPenalty: number
  videoDuplexLengthPenalty: number
  turnTtsEnabled: boolean
  turnStreamingEnabled: boolean
}

type IconProps = {
  className?: string
}

const EMPTY_REF_AUDIO: RefAudioState = {
  source: 'none',
  name: '未设置',
  duration: 0,
  base64: null,
}

const DEFAULT_SETTINGS: SettingsState = {
  turnbased: {
    presetId: null,
    systemPrompt:
      '你的任务是作为一个助手认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。',
    refAudio: EMPTY_REF_AUDIO,
  },
  audio_duplex: {
    presetId: null,
    systemPrompt:
      '请作为一个自然、口语化的语音助手与用户实时对话。你处于音频双工模式，可以一边听一边说。',
    refAudio: EMPTY_REF_AUDIO,
  },
  omni: {
    presetId: null,
    systemPrompt: 'Streaming Omni Conversation.',
    refAudio: EMPTY_REF_AUDIO,
  },
  maxNewTokens: 256,
  turnLengthPenalty: 1.1,
  audioDuplexLengthPenalty: 1.05,
  videoDuplexLengthPenalty: 1.1,
  turnTtsEnabled: true,
  turnStreamingEnabled: true,
}

function createId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`
}

function formatDurationMs(durationMs: number): string {
  return `${(durationMs / 1000).toFixed(1)}s`
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }

  return 'Unknown error'
}

function getDuplexModeLabel(mode: DuplexMode): string {
  return mode === 'audio' ? '音频双工' : '视频双工'
}

function getDuplexScreen(mode: DuplexMode): Screen {
  return mode === 'audio' ? 'audio-duplex' : 'video-duplex'
}

function getDuplexBadgeText(status: DuplexStatus, mode: DuplexMode): string {
  switch (status) {
    case 'live':
      return `${getDuplexModeLabel(mode)}进行中`
    case 'queueing':
      return '排队中'
    case 'paused':
      return '已暂停'
    case 'error':
      return '连接异常'
    case 'stopped':
      return '已结束'
    default:
      return '连接中'
  }
}

function getPresetModeForScreen(screen: Screen): PresetMode {
  if (screen === 'turn') {
    return 'turnbased'
  }

  return screen === 'audio-duplex' ? 'audio_duplex' : 'omni'
}

function getPresetModeLabel(mode: PresetMode): string {
  if (mode === 'turnbased') {
    return 'Turn-based'
  }

  return mode === 'audio_duplex' ? '音频双工' : '视频双工'
}

function getLengthPenaltyForMode(
  settings: SettingsState,
  presetMode: PresetMode,
): number {
  if (presetMode === 'turnbased') {
    return settings.turnLengthPenalty
  }

  return presetMode === 'audio_duplex'
    ? settings.audioDuplexLengthPenalty
    : settings.videoDuplexLengthPenalty
}

function summarizePrompt(prompt: string): string {
  const compact = prompt.replace(/\s+/g, ' ').trim()

  if (!compact) {
    return '未设置'
  }

  return compact.length > 48 ? `${compact.slice(0, 48)}...` : compact
}

function cloneRefAudio(refAudio: RefAudioState): RefAudioState {
  return {
    ...refAudio,
  }
}

function buildModeSettings(
  previous: ModeSettings,
  next: Partial<ModeSettings>,
): ModeSettings {
  return {
    ...previous,
    ...next,
    refAudio: next.refAudio ? cloneRefAudio(next.refAudio) : cloneRefAudio(previous.refAudio),
  }
}

function extractPromptFromPreset(preset: PresetMetadata): string {
  if (preset.system_prompt?.trim()) {
    return preset.system_prompt.trim()
  }

  const textParts =
    preset.system_content
      ?.filter(
        (
          item,
        ): item is Extract<BackendContentItem, { type: 'text'; text: string }> =>
          item.type === 'text' && Boolean(item.text?.trim()),
      )
      .map((item) => item.text.trim()) ?? []

  return textParts.join('\n\n').trim()
}

function extractRefAudioFromPreset(preset: PresetMetadata): RefAudioState {
  if (preset.ref_audio?.data) {
    return {
      source: 'preset',
      name: preset.ref_audio.name || '预设参考音频',
      duration: preset.ref_audio.duration || 0,
      base64: preset.ref_audio.data,
    }
  }

  const systemAudio = preset.system_content?.find(
    (
      item,
    ): item is Extract<BackendContentItem, { type: 'audio'; data: string }> =>
      item.type === 'audio' && Boolean(item.data),
  )

  if (systemAudio?.data) {
    return {
      source: 'preset',
      name: systemAudio.name || '预设参考音频',
      duration: systemAudio.duration || 0,
      base64: systemAudio.data,
    }
  }

  return cloneRefAudio(EMPTY_REF_AUDIO)
}

function PhoneIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        d="M6.6 4.8h3.2l1 3.8-1.8 1.8a15.4 15.4 0 0 0 4.7 4.7l1.8-1.8 3.8 1v3.2a1.6 1.6 0 0 1-1.7 1.6A15.9 15.9 0 0 1 4.9 6.5 1.6 1.6 0 0 1 6.6 4.8Z"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  )
}

function KeyboardIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <rect
        x="3.5"
        y="6"
        width="17"
        height="12"
        rx="2.5"
        stroke="currentColor"
        strokeWidth="1.8"
      />
      <path
        d="M7.5 10h9M7.5 13h4.5M14 13h2.5M7.5 16h7"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.8"
      />
    </svg>
  )
}

function MicIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        d="M12 4.5A2.5 2.5 0 0 1 14.5 7v4a2.5 2.5 0 0 1-5 0V7A2.5 2.5 0 0 1 12 4.5Z"
        stroke="currentColor"
        strokeWidth="1.8"
      />
      <path
        d="M7.5 10.5a4.5 4.5 0 0 0 9 0M12 15v4M9 19h6"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  )
}

function WaveIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        d="M4 13h2l1.4-4 2.4 9 2.4-12 2.1 7H20"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  )
}

function TranscriptIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <rect
        x="4"
        y="5"
        width="16"
        height="14"
        rx="3"
        stroke="currentColor"
        strokeWidth="1.8"
      />
      <path
        d="M8 10h8M8 14h5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.8"
      />
    </svg>
  )
}

function FlipCameraIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        d="M4.5 9.5 7 7h10l2.5 2.5v7A2.5 2.5 0 0 1 17 19H7a2.5 2.5 0 0 1-2.5-2.5Z"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
      <path
        d="M15.5 13a3.5 3.5 0 0 1-5.8 2.6M8.5 13a3.5 3.5 0 0 1 5.8-2.6M9.6 17.5l-2-.2.2-2M14.4 8.5l2 .2-.2 2"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  )
}

function PauseIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <rect x="7" y="6" width="3.2" height="12" rx="1.2" fill="currentColor" />
      <rect x="13.8" y="6" width="3.2" height="12" rx="1.2" fill="currentColor" />
    </svg>
  )
}

function PlayIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <path d="M9 7.5v9l7-4.5-7-4.5Z" fill="currentColor" />
    </svg>
  )
}

function VideoCallIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        d="M4.5 8.25A2.25 2.25 0 0 1 6.75 6h7.5a2.25 2.25 0 0 1 2.25 2.25v7.5A2.25 2.25 0 0 1 14.25 18h-7.5A2.25 2.25 0 0 1 4.5 15.75Z"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
      <path
        d="m16.5 10.1 2.9-1.8a.7.7 0 0 1 1.1.6v6.2a.7.7 0 0 1-1.1.6l-2.9-1.8"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
      <circle cx="10.5" cy="12" r="2.6" stroke="currentColor" strokeWidth="1.8" />
    </svg>
  )
}

function CloseIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        d="m8 8 8 8M16 8l-8 8"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.8"
      />
    </svg>
  )
}

function StopIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <rect x="7" y="7" width="10" height="10" rx="2.4" fill="currentColor" />
    </svg>
  )
}

function SendIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        d="M4.5 11.5 19 5l-4.5 14-2.6-5-7.4-2.5Z"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
      <path
        d="M19 5 11.8 14"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.8"
      />
    </svg>
  )
}

function SettingsIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        d="M10.3 4.8h3.4l.5 2.1a5.8 5.8 0 0 1 1.5.9l2-.7 1.7 2.9-1.5 1.4a6 6 0 0 1 0 1.7l1.5 1.4-1.7 2.9-2-.7a5.8 5.8 0 0 1-1.5.9l-.5 2.1h-3.4l-.5-2.1a5.8 5.8 0 0 1-1.5-.9l-2 .7-1.7-2.9 1.5-1.4a6 6 0 0 1 0-1.7L4.6 10l1.7-2.9 2 .7a5.8 5.8 0 0 1 1.5-.9Z"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.6"
      />
      <circle cx="12" cy="12" r="2.5" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  )
}

function buildRequestMessages(
  entries: ConversationEntry[],
  systemMessage?: string | BackendContentItem[] | null,
): BackendMessage[] {
  const messages: BackendMessage[] = []

  if (typeof systemMessage === 'string' && systemMessage.trim()) {
    messages.push({
      role: 'system',
      content: systemMessage.trim(),
    })
  } else if (Array.isArray(systemMessage) && systemMessage.length) {
    messages.push({
      role: 'system',
      content: systemMessage,
    })
  }

  const conversationMessages: BackendMessage[] = entries.map((entry): BackendMessage => {
    if (entry.role === 'assistant') {
      return {
        role: 'assistant',
        content: entry.text,
      }
    }

    if (entry.kind === 'text') {
      return {
        role: 'user',
        content: entry.text,
      }
    }

    return {
      role: 'user',
      content: [
        {
          type: 'audio',
          data: entry.audioBase64,
        },
      ],
    }
  })

  return [...messages, ...conversationMessages]
}

async function convertAudioBlobToFloat32Base64(blob: Blob): Promise<string> {
  const AudioContextCtor =
    window.AudioContext ??
    (
      window as Window & {
        webkitAudioContext?: typeof AudioContext
      }
    ).webkitAudioContext

  if (!AudioContextCtor) {
    throw new Error('This browser does not support AudioContext.')
  }

  const audioContext = new AudioContextCtor()

  try {
    const arrayBuffer = await blob.arrayBuffer()
    const decoded = await audioContext.decodeAudioData(arrayBuffer)
    const offlineContext = new OfflineAudioContext(
      1,
      Math.ceil(decoded.duration * 16000),
      16000,
    )
    const source = offlineContext.createBufferSource()

    source.buffer = decoded
    source.connect(offlineContext.destination)
    source.start()

    const rendered = await offlineContext.startRendering()
    const pcm = rendered.getChannelData(0)
    const bytes = new Uint8Array(pcm.buffer)
    let binary = ''

    for (let index = 0; index < bytes.length; index += 1) {
      binary += String.fromCharCode(bytes[index] ?? 0)
    }

    return btoa(binary)
  } finally {
    await audioContext.close()
  }
}

function base64ToBytes(base64Data: string): Uint8Array {
  const binary = atob(base64Data)
  const raw = new Uint8Array(binary.length)

  for (let index = 0; index < binary.length; index += 1) {
    raw[index] = binary.charCodeAt(index)
  }

  return raw
}

function isWavBytes(bytes: Uint8Array): boolean {
  return (
    bytes.length >= 44 &&
    bytes[0] === 0x52 && // R
    bytes[1] === 0x49 && // I
    bytes[2] === 0x46 && // F
    bytes[3] === 0x46 && // F
    bytes[8] === 0x57 && // W
    bytes[9] === 0x41 && // A
    bytes[10] === 0x56 && // V
    bytes[11] === 0x45 // E
  )
}

function bytesToBlobUrl(bytes: Uint8Array, type: string): string {
  const copy = new ArrayBuffer(bytes.byteLength)
  new Uint8Array(copy).set(bytes)
  return URL.createObjectURL(new Blob([copy], { type }))
}

function audioBase64ToBlobUrl(
  base64Data: string,
  sampleRate = 24000,
): string {
  const bytes = base64ToBytes(base64Data)

  if (isWavBytes(bytes)) {
    return bytesToBlobUrl(bytes, 'audio/wav')
  }

  return float32PcmBytesToWavUrl(bytes, sampleRate)
}

function float32PcmBytesToWavUrl(raw: Uint8Array, sampleRate: number): string {
  const float32 = new Float32Array(raw.buffer)
  const wavBuffer = new ArrayBuffer(44 + float32.length * 2)
  const view = new DataView(wavBuffer)

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
    view.setInt16(
      offset,
      sample < 0 ? sample * 0x8000 : sample * 0x7fff,
      true,
    )
    offset += 2
  }

  return URL.createObjectURL(new Blob([wavBuffer], { type: 'audio/wav' }))
}

function playPcmBase64(base64Data: string, sampleRate = 16000) {
  const url = audioBase64ToBlobUrl(base64Data, sampleRate)
  const audio = new Audio(url)

  audio.onended = () => {
    URL.revokeObjectURL(url)
  }
  audio.onerror = () => {
    URL.revokeObjectURL(url)
  }

  void audio.play().catch(() => {
    URL.revokeObjectURL(url)
  })
}

type AudioPlayPillProps = {
  url: string
  className?: string
  playLabel?: string
  pauseLabel?: string
}

function AudioPlayPill({
  url,
  className,
  playLabel = '播放',
  pauseLabel = '暂停',
}: AudioPlayPillProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)

  useEffect(() => {
    const audio = new Audio(url)
    audioRef.current = audio

    const handlePlay = () => setIsPlaying(true)
    const handlePause = () => setIsPlaying(false)
    const handleEnded = () => setIsPlaying(false)

    audio.addEventListener('play', handlePlay)
    audio.addEventListener('pause', handlePause)
    audio.addEventListener('ended', handleEnded)

    return () => {
      audio.removeEventListener('play', handlePlay)
      audio.removeEventListener('pause', handlePause)
      audio.removeEventListener('ended', handleEnded)
      try {
        audio.pause()
      } catch {
        /* ignore */
      }
      audioRef.current = null
    }
  }, [url])

  const handleClick = () => {
    const audio = audioRef.current
    if (!audio) {
      return
    }

    if (isPlaying) {
      audio.pause()
      return
    }

    try {
      audio.currentTime = 0
    } catch {
      /* ignore: some browsers throw if not seekable yet */
    }
    void audio.play().catch(() => {
      setIsPlaying(false)
    })
  }

  return (
    <button
      className={['voice-pill', isPlaying ? 'is-playing' : '', className]
        .filter(Boolean)
        .join(' ')}
      type="button"
      onClick={handleClick}
    >
      {isPlaying ? (
        <PauseIcon className="app-icon app-icon-sm" />
      ) : (
        <PlayIcon className="app-icon app-icon-sm" />
      )}
      <span>{isPlaying ? pauseLabel : playLabel}</span>
    </button>
  )
}

function MessageBubble({ entry }: { entry: ThreadEntry }) {
  if (entry.kind === 'pending') {
    return (
      <div className="msg assistant pending">
        <span>{entry.text}</span>
        <span className="pending-dots" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
      </div>
    )
  }

  if (entry.role === 'user' && entry.kind === 'voice') {
    return (
      <div className="msg user-voice">
        <div className="voice-row">
          <AudioPlayPill url={entry.previewUrl} />
          <div className="voice-wave" />
          <div className="voice-time">{formatDurationMs(entry.durationMs)}</div>
        </div>
      </div>
    )
  }

  return (
    <div
      className={[
        'msg',
        entry.role === 'assistant' ? 'assistant' : 'user-text',
        entry.role === 'assistant' && entry.error ? 'error' : '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {entry.text}
      {entry.role === 'assistant' && entry.recordingSessionId ? (
        <div className="msg-meta">session: {entry.recordingSessionId}</div>
      ) : null}
      {entry.role === 'assistant' && entry.audioPreviewUrl ? (
        <div className="assistant-actions">
          <AudioPlayPill
            url={entry.audioPreviewUrl}
            className="audio-replay"
            playLabel="收听"
            pauseLabel="暂停"
          />
        </div>
      ) : null}
    </div>
  )
}

function DuplexLogBubble({ entry }: { entry: DuplexEntry }) {
  return <div className={['duplex-log', entry.role].join(' ')}>{entry.text}</div>
}

type SettingsSummaryProps = {
  modeLabel: string
  presetName: string
  refAudio: RefAudioState
  systemPrompt: string
  lengthPenalty: number
  maxNewTokens?: number
  turnTtsEnabled?: boolean
  turnStreamingEnabled?: boolean
  onOpen: () => void
}

function SettingsSummary({
  modeLabel,
  presetName,
  refAudio,
  systemPrompt,
  lengthPenalty,
  maxNewTokens,
  turnTtsEnabled,
  turnStreamingEnabled,
  onOpen,
}: SettingsSummaryProps) {
  return (
    <div className="settings-summary-card">
      <div className="settings-summary-head">
        <div className="settings-summary-title">当前参数</div>
        <button className="settings-link-button" type="button" onClick={onOpen}>
          <SettingsIcon className="app-icon app-icon-sm" />
          <span>设置</span>
        </button>
      </div>
      <div className="settings-chip-row">
        <span className="settings-chip">{modeLabel}</span>
        <span className="settings-chip">Preset: {presetName}</span>
        <span className="settings-chip">
          Ref: {refAudio.base64 ? refAudio.name : '未设置'}
        </span>
        <span className="settings-chip">Len: {lengthPenalty.toFixed(2)}</span>
        {typeof maxNewTokens === 'number' ? (
          <span className="settings-chip">Tokens: {maxNewTokens}</span>
        ) : null}
        {typeof turnTtsEnabled === 'boolean' ? (
          <span className="settings-chip">{turnTtsEnabled ? '语音回复开' : '语音回复关'}</span>
        ) : null}
        {typeof turnStreamingEnabled === 'boolean' ? (
          <span className="settings-chip">
            {turnStreamingEnabled ? '流式输出开' : '流式输出关'}
          </span>
        ) : null}
      </div>
      <div className="settings-summary-prompt">{summarizePrompt(systemPrompt)}</div>
    </div>
  )
}

type SettingsSheetProps = {
  open: boolean
  activeMode: PresetMode
  activeLabel: string
  activeSettings: ModeSettings
  activePresets: PresetMetadata[]
  defaultRefAudio: RefAudioState | null
  lengthPenalty: number
  maxNewTokens: number
  turnTtsEnabled: boolean
  turnStreamingEnabled: boolean
  onClose: () => void
  onSelectPreset: (presetId: string) => void
  onPromptChange: (value: string) => void
  onLengthPenaltyChange: (value: number) => void
  onMaxTokensChange: (value: number) => void
  onTurnTtsEnabledChange: (value: boolean) => void
  onTurnStreamingEnabledChange: (value: boolean) => void
  onUseDefaultRefAudio: () => void
  onClearRefAudio: () => void
  onUploadRefAudio: () => void
  onPlayRefAudio: () => void
}

function SettingsSheet({
  open,
  activeMode,
  activeLabel,
  activeSettings,
  activePresets,
  defaultRefAudio,
  lengthPenalty,
  maxNewTokens,
  turnTtsEnabled,
  turnStreamingEnabled,
  onClose,
  onSelectPreset,
  onPromptChange,
  onLengthPenaltyChange,
  onMaxTokensChange,
  onTurnTtsEnabledChange,
  onTurnStreamingEnabledChange,
  onUseDefaultRefAudio,
  onClearRefAudio,
  onUploadRefAudio,
  onPlayRefAudio,
}: SettingsSheetProps) {
  if (!open) {
    return null
  }

  return (
    <div className="settings-sheet-backdrop" onClick={onClose}>
      <div
        className="settings-sheet"
        onClick={(event) => {
          event.stopPropagation()
        }}
      >
        <div className="settings-sheet-head">
          <div>
            <div className="settings-sheet-title">设置</div>
            <div className="settings-sheet-subtitle">{activeLabel}</div>
          </div>
          <button className="settings-close-button" type="button" onClick={onClose}>
            <CloseIcon className="app-icon app-icon-md" />
          </button>
        </div>

        <div className="settings-section">
          <div className="settings-section-title">Preset</div>
          <div className="preset-chip-row">
            {activePresets.length ? (
              activePresets.map((preset) => (
                <button
                  key={preset.id}
                  className={[
                    'preset-chip',
                    activeSettings.presetId === preset.id ? 'active' : '',
                  ]
                    .filter(Boolean)
                    .join(' ')}
                  type="button"
                  onClick={() => {
                    onSelectPreset(preset.id)
                  }}
                >
                  {preset.name}
                </button>
              ))
            ) : (
              <div className="settings-empty-copy">当前模式暂无可用 preset。</div>
            )}
          </div>
        </div>

        <div className="settings-section">
          <div className="settings-section-title">参考音频</div>
          <div className="ref-audio-card">
            <div className="ref-audio-title">
              {activeSettings.refAudio.base64 ? activeSettings.refAudio.name : '未设置参考音频'}
            </div>
            <div className="ref-audio-meta">
              来源：{activeSettings.refAudio.source}
              {activeSettings.refAudio.duration
                ? ` · ${activeSettings.refAudio.duration.toFixed(1)}s`
                : ''}
            </div>
            <div className="ref-audio-actions">
              <button
                className="secondary-btn compact"
                type="button"
                onClick={onUseDefaultRefAudio}
                disabled={!defaultRefAudio?.base64}
              >
                默认
              </button>
              <button
                className="secondary-btn compact"
                type="button"
                onClick={onUploadRefAudio}
              >
                上传
              </button>
              <button
                className="secondary-btn compact"
                type="button"
                onClick={onPlayRefAudio}
                disabled={!activeSettings.refAudio.base64}
              >
                播放
              </button>
              <button
                className="secondary-btn compact"
                type="button"
                onClick={onClearRefAudio}
              >
                清空
              </button>
            </div>
          </div>
        </div>

        <div className="settings-section">
          <label className="settings-section-title" htmlFor="settings-system-prompt">
            System Prompt
          </label>
          <textarea
            id="settings-system-prompt"
            className="settings-textarea"
            value={activeSettings.systemPrompt}
            onChange={(event) => {
              onPromptChange(event.target.value)
            }}
          />
        </div>

        <div className="settings-section">
          <div className="settings-section-title">参数</div>
          <div className="settings-grid">
            <label className="settings-field">
              <span>Length Penalty</span>
              <input
                className="settings-input"
                type="number"
                min="0.1"
                max="5"
                step="0.05"
                value={lengthPenalty}
                onChange={(event) => {
                  onLengthPenaltyChange(Number(event.target.value))
                }}
              />
            </label>

            {activeMode === 'turnbased' ? (
              <label className="settings-field">
                <span>Max Tokens</span>
                <input
                  className="settings-input"
                  type="number"
                  min="1"
                  max="2048"
                  step="1"
                  value={maxNewTokens}
                  onChange={(event) => {
                    onMaxTokensChange(Number(event.target.value))
                  }}
                />
              </label>
            ) : null}
          </div>

          {activeMode === 'turnbased' ? (
            <>
              <label className="settings-toggle">
                <input
                  type="checkbox"
                  checked={turnTtsEnabled}
                  onChange={(event) => {
                    onTurnTtsEnabledChange(event.target.checked)
                  }}
                />
                <span>Turn-based 语音回复</span>
              </label>
              <label className="settings-toggle">
                <input
                  type="checkbox"
                  checked={turnStreamingEnabled}
                  onChange={(event) => {
                    onTurnStreamingEnabledChange(event.target.checked)
                  }}
                />
                <span>Turn-based 流式输出</span>
              </label>
            </>
          ) : null}
        </div>
      </div>
    </div>
  )
}

function App() {
  const [screen, setScreen] = useState<Screen>('turn')
  const [composeMode, setComposeMode] = useState<'voice' | 'text'>('voice')
  const [draft, setDraft] = useState('')
  const [messages, setMessages] = useState<ConversationEntry[]>([])
  const [pendingReply, setPendingReply] = useState<PendingReply | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [isPreparingRecording, setIsPreparingRecording] = useState(false)
  const [recordError, setRecordError] = useState<string | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [presetsByMode, setPresetsByMode] = useState<Record<PresetMode, PresetMetadata[]>>({
    turnbased: [],
    audio_duplex: [],
    omni: [],
  })
  const [defaultRefAudio, setDefaultRefAudio] = useState<RefAudioState | null>(null)
  const [settings, setSettings] = useState<SettingsState>({
    ...DEFAULT_SETTINGS,
    turnbased: buildModeSettings(DEFAULT_SETTINGS.turnbased, {}),
    audio_duplex: buildModeSettings(DEFAULT_SETTINGS.audio_duplex, {}),
    omni: buildModeSettings(DEFAULT_SETTINGS.omni, {}),
  })
  const [serviceState, setServiceState] = useState<ServiceState>({
    phase: 'loading',
    summary: 'Checking backend',
    detail: 'Polling /status...',
  })
  const [lastSessionId, setLastSessionId] = useState<string | null>(null)
  const [duplexEntries, setDuplexEntries] = useState<DuplexEntry[]>([])
  const [duplexStatus, setDuplexStatus] = useState<DuplexStatus>('idle')
  const [duplexStatusText, setDuplexStatusText] = useState('等待进入全双工')
  const [duplexMode, setDuplexMode] = useState<DuplexMode>('audio')
  const [duplexMicEnabled, setDuplexMicEnabled] = useState(true)
  const [duplexTextPanelOpen, setDuplexTextPanelOpen] = useState(true)
  const [duplexPauseState, setDuplexPauseState] = useState<
    'active' | 'pausing' | 'paused'
  >('active')

  const messagesRef = useRef<ConversationEntry[]>([])
  const abortRef = useRef<AbortController | null>(null)
  const streamingWsRef = useRef<WebSocket | null>(null)
  const streamingPlayerRef = useRef<StreamingPcmPlayer | null>(null)
  const streamingStopRef = useRef<(() => void) | null>(null)
  const threadEndRef = useRef<HTMLDivElement | null>(null)
  const duplexEndRef = useRef<HTMLDivElement | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const recordingStartRef = useRef<number>(0)
  const recordingActionRef = useRef<'send' | 'cancel'>('send')
  const refAudioInputRef = useRef<HTMLInputElement | null>(null)
  const duplexVideoRef = useRef<HTMLVideoElement | null>(null)
  const duplexCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const duplexSessionRef = useRef<DuplexSessionLike | null>(null)
  const duplexMediaRef = useRef<MobileLiveMediaProvider | null>(null)
  const duplexStartInFlightRef = useRef(false)
  const duplexListenEntryIdRef = useRef<string | null>(null)

  const threadEntries: ThreadEntry[] = pendingReply
    ? [...messages, pendingReply]
    : messages
  const duplexBadgeText = getDuplexBadgeText(duplexStatus, duplexMode)
  const audioDuplexScreenOpen = screen === 'audio-duplex'
  const videoDuplexScreenOpen = screen === 'video-duplex'
  const activePresetMode = getPresetModeForScreen(screen)
  const activeModeSettings = settings[activePresetMode]
  const activeModePresets = presetsByMode[activePresetMode]
  const activeLengthPenalty = getLengthPenaltyForMode(settings, activePresetMode)
  const activeModeLabel = getPresetModeLabel(activePresetMode)
  const turnPresetName =
    presetsByMode.turnbased.find((preset) => preset.id === settings.turnbased.presetId)
      ?.name ?? '自定义'
  const audioPresetName =
    presetsByMode.audio_duplex.find(
      (preset) => preset.id === settings.audio_duplex.presetId,
    )?.name ?? '自定义'
  const videoPresetName =
    presetsByMode.omni.find((preset) => preset.id === settings.omni.presetId)?.name ??
    '自定义'

  useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  useEffect(() => {
    threadEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [threadEntries.length])

  useEffect(() => {
    duplexEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [duplexEntries.length])

  useEffect(() => {
    let cancelled = false

    async function refreshStatus() {
      try {
        const response = await fetch('/status')

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }

        const data = (await response.json()) as ServiceStatusResponse

        if (cancelled) {
          return
        }

        setServiceState({
          phase: 'ready',
          summary: data.gateway_healthy ? 'Backend ready' : 'Gateway degraded',
          detail: `${data.idle_workers}/${data.total_workers} idle, queue ${data.queue_length}, offline ${data.offline_workers}`,
        })
      } catch (error) {
        if (cancelled) {
          return
        }

        setServiceState({
          phase: 'error',
          summary: 'Backend unreachable',
          detail: getErrorMessage(error),
        })
      }
    }

    void refreshStatus()

    const interval = window.setInterval(() => {
      void refreshStatus()
    }, 15000)

    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    async function hydratePresetAudio(
      mode: PresetMode,
      preset: PresetMetadata,
    ): Promise<PresetMetadata> {
      const needsRefAudio = Boolean(preset.ref_audio?.path && !preset.ref_audio.data)
      const needsSystemAudio = Boolean(
        preset.system_content?.some(
          (item) => item.type === 'audio' && !item.data && 'path' in item,
        ),
      )

      if (!needsRefAudio && !needsSystemAudio) {
        return preset
      }

      const response = await fetch(`/api/presets/${mode}/${preset.id}/audio`)

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const payload = (await response.json()) as {
        system_content_audio?: Array<{
          data?: string | null
          name?: string
          duration?: number
        }>
        ref_audio?: {
          data?: string | null
          name?: string
          duration?: number
        }
      }

      const nextPreset: PresetMetadata = {
        ...preset,
        system_content: preset.system_content?.map((item) => ({ ...item })),
        ref_audio: preset.ref_audio ? { ...preset.ref_audio } : undefined,
      }

      if (payload.system_content_audio && nextPreset.system_content) {
        let audioIndex = 0

        nextPreset.system_content = nextPreset.system_content.map((item) => {
          if (item.type !== 'audio') {
            return item
          }

          const loaded = payload.system_content_audio?.[audioIndex]

          audioIndex += 1

          return {
            ...item,
            data: loaded?.data || item.data,
            name: loaded?.name || item.name,
            duration: loaded?.duration || item.duration,
          }
        })
      }

      if (payload.ref_audio?.data && nextPreset.ref_audio) {
        nextPreset.ref_audio = {
          ...nextPreset.ref_audio,
          data: payload.ref_audio.data,
          name: payload.ref_audio.name || nextPreset.ref_audio.name,
          duration: payload.ref_audio.duration || nextPreset.ref_audio.duration,
        }
      }

      return nextPreset
    }

    async function loadSettingsData() {
      try {
        const [presetsResponse, defaultRefResponse] = await Promise.all([
          fetch('/api/presets'),
          fetch('/api/default_ref_audio'),
        ])

        const defaultRefPayload = defaultRefResponse.ok
          ? ((await defaultRefResponse.json()) as {
              name?: string
              duration?: number
              base64?: string | null
            })
          : null

        const nextDefaultRefAudio: RefAudioState | null = defaultRefPayload?.base64
          ? {
              source: 'default',
              name: defaultRefPayload.name || '默认参考音频',
              duration: defaultRefPayload.duration || 0,
              base64: defaultRefPayload.base64,
            }
          : null

        const presetPayload = presetsResponse.ok
          ? ((await presetsResponse.json()) as Partial<Record<PresetMode, PresetMetadata[]>>)
          : {}

        const hydratedPresets: Record<PresetMode, PresetMetadata[]> = {
          turnbased: [...(presetPayload.turnbased ?? [])],
          audio_duplex: [...(presetPayload.audio_duplex ?? [])],
          omni: [...(presetPayload.omni ?? [])],
        }

        for (const mode of ['turnbased', 'audio_duplex', 'omni'] as PresetMode[]) {
          const firstPreset = hydratedPresets[mode][0]

          if (!firstPreset) {
            continue
          }

          try {
            hydratedPresets[mode][0] = await hydratePresetAudio(mode, firstPreset)
          } catch (error) {
            console.warn(`Failed to hydrate preset ${mode}/${firstPreset.id}`, error)
          }
        }

        if (cancelled) {
          return
        }

        setPresetsByMode(hydratedPresets)
        setDefaultRefAudio(nextDefaultRefAudio)
        setSettings((previous) => {
          const nextSettings: SettingsState = {
            ...previous,
            turnbased: buildModeSettings(previous.turnbased, {}),
            audio_duplex: buildModeSettings(previous.audio_duplex, {}),
            omni: buildModeSettings(previous.omni, {}),
          }

          for (const mode of ['turnbased', 'audio_duplex', 'omni'] as PresetMode[]) {
            const firstPreset = hydratedPresets[mode][0]

            if (!firstPreset) {
              continue
            }

            const extractedRefAudio = extractRefAudioFromPreset(firstPreset)

            nextSettings[mode] = buildModeSettings(nextSettings[mode], {
              presetId: firstPreset.id,
              systemPrompt:
                extractPromptFromPreset(firstPreset) || nextSettings[mode].systemPrompt,
              refAudio: extractedRefAudio.base64
                ? extractedRefAudio
                : cloneRefAudio(nextSettings[mode].refAudio),
            })
          }

          if (
            nextDefaultRefAudio?.base64 &&
            !nextSettings.turnbased.refAudio.base64 &&
            !nextSettings.turnbased.presetId
          ) {
            nextSettings.turnbased = buildModeSettings(nextSettings.turnbased, {
              refAudio: nextDefaultRefAudio,
            })
          }

          return nextSettings
        })
      } catch (error) {
        console.warn('Failed to load mobile settings data', error)
      }
    }

    void loadSettingsData()

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    duplexMediaRef.current?.setMicEnabled(duplexMicEnabled)
  }, [duplexMicEnabled])

  useEffect(() => {
    return () => {
      abortRef.current?.abort()
      duplexSessionRef.current?.stop()
      duplexMediaRef.current?.stop()
      mediaRecorderRef.current?.stream
        .getTracks()
        .forEach((track) => track.stop())
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop())

      for (const entry of messagesRef.current) {
        if (entry.role === 'user' && entry.kind === 'voice') {
          URL.revokeObjectURL(entry.previewUrl)
        }
        if (entry.role === 'assistant' && entry.audioPreviewUrl) {
          URL.revokeObjectURL(entry.audioPreviewUrl)
        }
      }
    }
  }, [])

  function reportSettingsMessage(text: string) {
    if (screen === 'turn') {
      setRecordError(text)
      return
    }

    appendDuplexEntry('system', text)
  }

  function updateModeSettings(mode: PresetMode, patch: Partial<ModeSettings>) {
    setSettings((previous) => ({
      ...previous,
      [mode]: buildModeSettings(previous[mode], patch),
    }))
  }

  async function ensurePresetLoaded(
    mode: PresetMode,
    preset: PresetMetadata,
  ): Promise<PresetMetadata> {
    const needsRefAudio = Boolean(preset.ref_audio?.path && !preset.ref_audio.data)
    const needsSystemAudio = Boolean(
      preset.system_content?.some(
        (item) => item.type === 'audio' && item.path && !item.data,
      ),
    )

    if (!needsRefAudio && !needsSystemAudio) {
      return preset
    }

    const response = await fetch(`/api/presets/${mode}/${preset.id}/audio`)

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const payload = (await response.json()) as {
      system_content_audio?: Array<{
        data?: string | null
        name?: string
        duration?: number
      }>
      ref_audio?: {
        data?: string | null
        name?: string
        duration?: number
      }
    }

    const nextPreset: PresetMetadata = {
      ...preset,
      system_content: preset.system_content?.map((item) => ({ ...item })),
      ref_audio: preset.ref_audio ? { ...preset.ref_audio } : undefined,
    }

    if (payload.system_content_audio && nextPreset.system_content) {
      let audioIndex = 0

      nextPreset.system_content = nextPreset.system_content.map((item) => {
        if (item.type !== 'audio') {
          return item
        }

        const loaded = payload.system_content_audio?.[audioIndex]

        audioIndex += 1

        return {
          ...item,
          data: loaded?.data || item.data,
          name: loaded?.name || item.name,
          duration: loaded?.duration || item.duration,
        }
      })
    }

    if (payload.ref_audio?.data && nextPreset.ref_audio) {
      nextPreset.ref_audio = {
        ...nextPreset.ref_audio,
        data: payload.ref_audio.data,
        name: payload.ref_audio.name || nextPreset.ref_audio.name,
        duration: payload.ref_audio.duration || nextPreset.ref_audio.duration,
      }
    }

    setPresetsByMode((previous) => ({
      ...previous,
      [mode]: previous[mode].map((item) =>
        item.id === nextPreset.id ? nextPreset : item,
      ),
    }))

    return nextPreset
  }

  async function handleSelectPreset(mode: PresetMode, presetId: string) {
    const preset = presetsByMode[mode].find((item) => item.id === presetId)

    if (!preset) {
      return
    }

    try {
      const loadedPreset = await ensurePresetLoaded(mode, preset)
      const extractedRefAudio = extractRefAudioFromPreset(loadedPreset)

      updateModeSettings(mode, {
        presetId: loadedPreset.id,
        systemPrompt:
          extractPromptFromPreset(loadedPreset) || settings[mode].systemPrompt,
        refAudio: extractedRefAudio.base64
          ? extractedRefAudio
          : cloneRefAudio(EMPTY_REF_AUDIO),
      })
    } catch (error) {
      reportSettingsMessage(`加载 preset 失败：${getErrorMessage(error)}`)
    }
  }

  function handleChangePrompt(mode: PresetMode, value: string) {
    updateModeSettings(mode, {
      presetId: null,
      systemPrompt: value,
    })
  }

  function handleUseDefaultRefAudio(mode: PresetMode) {
    if (!defaultRefAudio?.base64) {
      reportSettingsMessage('当前没有可用的默认参考音频。')
      return
    }

    updateModeSettings(mode, {
      presetId: null,
      refAudio: defaultRefAudio,
    })
  }

  function handleClearRefAudio(mode: PresetMode) {
    updateModeSettings(mode, {
      presetId: null,
      refAudio: EMPTY_REF_AUDIO,
    })
  }

  async function handleRefAudioInputChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]

    if (!file) {
      return
    }

    try {
      const base64 = await convertAudioBlobToFloat32Base64(file)
      const durationAudio = new Audio(URL.createObjectURL(file))

      durationAudio.onloadedmetadata = () => {
        updateModeSettings(activePresetMode, {
          presetId: null,
          refAudio: {
            source: 'upload',
            name: file.name,
            duration: Number.isFinite(durationAudio.duration)
              ? durationAudio.duration
              : 0,
            base64,
          },
        })
        URL.revokeObjectURL(durationAudio.src)
      }
      durationAudio.onerror = () => {
        updateModeSettings(activePresetMode, {
          presetId: null,
          refAudio: {
            source: 'upload',
            name: file.name,
            duration: 0,
            base64,
          },
        })
        URL.revokeObjectURL(durationAudio.src)
      }
    } catch (error) {
      reportSettingsMessage(`处理参考音频失败：${getErrorMessage(error)}`)
    } finally {
      event.target.value = ''
    }
  }

  function buildTurnSystemMessage(): string | BackendContentItem[] | null {
    const items: BackendContentItem[] = []
    const prompt = settings.turnbased.systemPrompt.trim()
    const refAudio = settings.turnbased.refAudio.base64

    if (refAudio) {
      items.push({
        type: 'text',
        text: '模仿音频样本的音色并生成新的内容。',
      })
      items.push({
        type: 'audio',
        data: refAudio,
        name: settings.turnbased.refAudio.name,
        duration: settings.turnbased.refAudio.duration,
      })
    }

    if (prompt) {
      items.push({
        type: 'text',
        text: prompt,
      })
    }

    if (items.length === 0) {
      return null
    }

    if (items.length === 1 && items[0]?.type === 'text') {
      return items[0].text
    }

    return items
  }

  function handleLengthPenaltyChange(mode: PresetMode, value: number) {
    const nextValue = Number.isFinite(value) ? value : 1.1

    setSettings((previous) => {
      if (mode === 'turnbased') {
        return {
          ...previous,
          turnLengthPenalty: nextValue,
        }
      }

      return mode === 'audio_duplex'
        ? {
            ...previous,
            audioDuplexLengthPenalty: nextValue,
          }
        : {
            ...previous,
            videoDuplexLengthPenalty: nextValue,
          }
    })
  }

  function handlePlayActiveRefAudio() {
    if (!activeModeSettings.refAudio.base64) {
      reportSettingsMessage('当前没有可播放的参考音频。')
      return
    }

    playPcmBase64(activeModeSettings.refAudio.base64, 16000)
  }

  function resetRecorderResources() {
    mediaRecorderRef.current = null
    audioChunksRef.current = []
    recordingStartRef.current = 0
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop())
    mediaStreamRef.current = null
  }

  function stopCurrentReply() {
    abortRef.current?.abort()

    const stop = streamingStopRef.current
    if (stop) {
      streamingStopRef.current = null
      stop()
    }
  }

  function appendDuplexEntry(role: DuplexEntry['role'], text: string): string {
    const id = createId(`duplex-${role}`)

    setDuplexEntries((previous) => [
      ...previous,
      {
        id,
        role,
        text,
      },
    ])

    return id
  }

  function updateDuplexEntry(id: string, text: string) {
    setDuplexEntries((previous) =>
      previous.map((entry) =>
        entry.id === id
          ? {
              ...entry,
              text,
            }
          : entry,
      ),
    )
  }

  function stopDuplexSession(options?: { preserveScreen?: boolean }) {
    const modeLabel = getDuplexModeLabel(duplexMode)

    duplexStartInFlightRef.current = false
    duplexListenEntryIdRef.current = null
    duplexMediaRef.current?.stop()
    duplexMediaRef.current = null

    if (duplexSessionRef.current) {
      duplexSessionRef.current.stop()
      duplexSessionRef.current = null
    }

    setDuplexStatus('stopped')
    setDuplexStatusText(`${modeLabel}会话已结束`)
    setDuplexPauseState('active')

    if (!options?.preserveScreen) {
      setScreen('turn')
    }
  }

  function openDuplexScreen(mode: DuplexMode) {
    setDuplexMode(mode)
    setScreen(getDuplexScreen(mode))

    requestAnimationFrame(() => {
      void startDuplexSession(mode)
    })
  }

  async function startDuplexSession(mode: DuplexMode) {
    if (
      duplexStartInFlightRef.current ||
      duplexSessionRef.current ||
      !duplexVideoRef.current ||
      !duplexCanvasRef.current
    ) {
      return
    }

    const modeLabel = getDuplexModeLabel(mode)
    const withVideo = mode === 'video'
    const duplexSettings = withVideo ? settings.omni : settings.audio_duplex

    duplexStartInFlightRef.current = true
    duplexListenEntryIdRef.current = null
    setDuplexMode(mode)
    setDuplexEntries([])
    setDuplexMicEnabled(true)
    setDuplexPauseState('active')
    setDuplexStatus('starting')
    setDuplexStatusText(
      withVideo ? '正在请求摄像头和麦克风...' : '正在请求麦克风...',
    )

    try {
      const runtime = await loadDuplexRuntime()
      const media = new MobileLiveMediaProvider({
        videoEl: duplexVideoRef.current,
        canvasEl: duplexCanvasRef.current,
      })
      const session = new runtime.DuplexSession(withVideo ? 'omni' : 'adx', {
        getMaxKvTokens: () => 8192,
        getPlaybackDelayMs: () => 200,
        outputSampleRate: 24000,
      })

      duplexMediaRef.current = media
      duplexSessionRef.current = session

      media.setMicEnabled(true)
      await media.setCameraEnabled(withVideo)

      session.onSystemLog = (text) => {
        appendDuplexEntry('system', text)
      }
      session.onQueueUpdate = (data) => {
        if (data) {
          setDuplexStatus('queueing')
          setDuplexStatusText(
            `排队 ${data.position ?? '?'} / ${data.queue_length ?? '?'}，预计 ${Math.round(
              data.estimated_wait_s ?? 0,
            )}s`,
          )
        }
      }
      session.onQueueDone = () => {
        setDuplexStatus('starting')
        setDuplexStatusText(`Worker 已分配，正在准备${modeLabel}会话...`)
      }
      session.onPrepared = () => {
        setDuplexStatusText(`会话已准备，开始${modeLabel}推流...`)
      }
      session.onRunningChange = (running) => {
        setDuplexStatus(running ? 'live' : 'stopped')
        setDuplexStatusText(running ? `${modeLabel}进行中` : `${modeLabel}会话已停止`)
      }
      session.onPauseStateChange = (state) => {
        setDuplexPauseState(state)
        setDuplexStatus(state === 'active' ? 'live' : 'paused')
        setDuplexStatusText(
          state === 'active'
            ? `${modeLabel}进行中`
            : state === 'pausing'
              ? `正在暂停${modeLabel}...`
              : `${modeLabel}已暂停`,
        )
      }
      session.onMetrics = (data) => {
        if (data.type === 'result') {
          setDuplexStatusText(
            `${data.modelState ?? 'live'} · ${Math.round(
              data.latencyMs ?? 0,
            )}ms · KV ${data.kvCacheLength ?? '-'}`,
          )
        }
      }
      session.onListenResult = (result: DuplexResultLike) => {
        const listenText = result.text?.trim()

        if (!listenText) {
          return
        }

        if (!duplexListenEntryIdRef.current) {
          duplexListenEntryIdRef.current = appendDuplexEntry('user', listenText)
        } else {
          updateDuplexEntry(duplexListenEntryIdRef.current, listenText)
        }
      }
      session.onSpeakStart = (text) => {
        duplexListenEntryIdRef.current = null
        return appendDuplexEntry('assistant', text)
      }
      session.onSpeakUpdate = (handle, text) => {
        if (typeof handle === 'string') {
          updateDuplexEntry(handle, text)
        }
      }
      session.onSpeakEnd = () => {
        duplexListenEntryIdRef.current = null
      }
      session.onCleanup = () => {
        duplexStartInFlightRef.current = false
        duplexListenEntryIdRef.current = null
        duplexMediaRef.current?.stop()
        duplexMediaRef.current = null
        duplexSessionRef.current = null
        setDuplexPauseState('active')
        setDuplexStatus('stopped')
        setDuplexStatusText(`${modeLabel}会话已结束`)
      }

      const preparePayload: Record<string, unknown> = {
        config: {
          length_penalty: withVideo
            ? settings.videoDuplexLengthPenalty
            : settings.audioDuplexLengthPenalty,
        },
      }

      if (duplexSettings.refAudio.base64) {
        preparePayload.ref_audio_base64 = duplexSettings.refAudio.base64
        preparePayload.tts_ref_audio_base64 = duplexSettings.refAudio.base64
      }

      if (withVideo) {
        preparePayload.max_slice_nums = 1
        preparePayload.deferred_finalize = true
      }

      await session.start(
        duplexSettings.systemPrompt.trim() || 'You are a helpful assistant.',
        preparePayload,
        async () => {
          media.onChunk = (chunk) => {
            const message: Record<string, unknown> = {
              type: 'audio_chunk',
              audio_base64: runtime.arrayBufferToBase64(chunk.audio.buffer),
            }

            if (withVideo && chunk.frameBase64) {
              message.frame_base64_list = [chunk.frameBase64]
            }

            session.sendChunk(message)
          }

          await media.start()
        },
      )

      if (session.recordingSessionId) {
        setLastSessionId(session.recordingSessionId)
      }
    } catch (error) {
      duplexMediaRef.current?.stop()
      duplexMediaRef.current = null
      duplexSessionRef.current = null
      setDuplexStatus('error')
      setDuplexStatusText(`启动失败：${getErrorMessage(error)}`)
      appendDuplexEntry('system', `启动失败：${getErrorMessage(error)}`)
    } finally {
      duplexStartInFlightRef.current = false
    }
  }

  function buildChatRequestBody(
    nextMessages: ConversationEntry[],
    systemMessage: string | BackendContentItem[] | null | undefined,
    streaming: boolean,
  ) {
    return {
      messages: buildRequestMessages(nextMessages, systemMessage),
      streaming,
      generation: {
        max_new_tokens: settings.maxNewTokens,
        length_penalty: settings.turnLengthPenalty,
      },
      ...(settings.turnTtsEnabled
        ? {
            use_tts_template: true,
          }
        : {}),
      tts: {
        enabled: settings.turnTtsEnabled,
        ...(settings.turnTtsEnabled && settings.turnbased.refAudio.base64
          ? {
              mode: 'audio_assistant',
              ref_audio_data: settings.turnbased.refAudio.base64,
            }
          : settings.turnTtsEnabled
            ? {
                mode: 'audio_assistant',
              }
            : {}),
      },
    }
  }

  async function submitConversation(nextMessages: ConversationEntry[]) {
    if (settings.turnStreamingEnabled) {
      await submitConversationStreaming(nextMessages)
    } else {
      await submitConversationNonStreaming(nextMessages)
    }
  }

  async function submitConversationNonStreaming(
    nextMessages: ConversationEntry[],
  ) {
    const systemMessage = buildTurnSystemMessage()

    setPendingReply({
      id: createId('pending'),
      role: 'assistant',
      kind: 'pending',
      text: '正在思考',
    })
    setIsGenerating(true)

    const controller = new AbortController()

    abortRef.current = controller

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(
          buildChatRequestBody(nextMessages, systemMessage, false),
        ),
        signal: controller.signal,
      })

      const rawText = await response.text()
      let payload: {
        text?: string
        error?: string
        success?: boolean
        audio_data?: string | null
        audio_sample_rate?: number
        recording_session_id?: string | null
      }

      try {
        payload = JSON.parse(rawText) as {
          text?: string
          error?: string
          success?: boolean
          audio_data?: string | null
          audio_sample_rate?: number
          recording_session_id?: string | null
        }
      } catch {
        throw new Error(rawText || `HTTP ${response.status}`)
      }

      if (!response.ok || payload.success === false) {
        throw new Error(payload.error || `HTTP ${response.status}`)
      }

      let assistantAudioUrl: string | null = null

      if (payload.audio_data) {
        try {
          assistantAudioUrl = audioBase64ToBlobUrl(
            payload.audio_data,
            payload.audio_sample_rate ?? 24000,
          )
        } catch {
          assistantAudioUrl = null
        }
      }

      const assistantEntry: ConversationEntry = {
        id: createId('assistant'),
        role: 'assistant',
        kind: 'assistant',
        text: payload.text?.trim() || '(空回复)',
        audioPreviewUrl: assistantAudioUrl,
        recordingSessionId: payload.recording_session_id ?? null,
      }

      setMessages([...nextMessages, assistantEntry])
      setLastSessionId(payload.recording_session_id ?? null)
    } catch (error) {
      const errorText =
        controller.signal.aborted
          ? '已停止当前回复。'
          : `请求失败：${getErrorMessage(error)}`

      setMessages([
        ...nextMessages,
        {
          id: createId('assistant'),
          role: 'assistant',
          kind: 'assistant',
          text: errorText,
          error: true,
        },
      ])
    } finally {
      abortRef.current = null
      setPendingReply(null)
      setIsGenerating(false)
    }
  }

  async function submitConversationStreaming(
    nextMessages: ConversationEntry[],
  ) {
    const systemMessage = buildTurnSystemMessage()

    const pendingId = createId('pending')

    setPendingReply({
      id: pendingId,
      role: 'assistant',
      kind: 'pending',
      text: '正在生成…',
    })
    setIsGenerating(true)

    const wsProto =
      window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${wsProto}//${window.location.host}/ws/chat`

    let ws: WebSocket
    try {
      ws = new WebSocket(wsUrl)
    } catch (error) {
      setMessages([
        ...nextMessages,
        {
          id: createId('assistant'),
          role: 'assistant',
          kind: 'assistant',
          text: `连接失败：${getErrorMessage(error)}`,
          error: true,
        },
      ])
      setPendingReply(null)
      setIsGenerating(false)
      return
    }

    streamingWsRef.current = ws

    let player: StreamingPcmPlayer | null = null
    if (settings.turnTtsEnabled) {
      try {
        player = new StreamingPcmPlayer(24000)
        streamingPlayerRef.current = player
      } catch {
        player = null
        streamingPlayerRef.current = null
      }
    }

    let fullText = ''
    let stoppedByUser = false
    let finished = false
    let lastSampleRate = 24000

    const finalize = (
      entry: ConversationEntry | null,
      options: { errorMessage?: string; cutPlayback?: boolean } = {},
    ) => {
      if (finished) {
        return
      }
      finished = true

      const { errorMessage, cutPlayback = false } = options

      let resolvedEntry = entry

      if (player) {
        const merged = player.getMergedFloat32()
        let mergedUrl: string | null = null
        if (merged && merged.length > 0) {
          try {
            mergedUrl = float32ToWavBlobUrl(merged, lastSampleRate)
          } catch {
            mergedUrl = null
          }
        }
        if (mergedUrl && resolvedEntry && resolvedEntry.kind === 'assistant') {
          resolvedEntry = { ...resolvedEntry, audioPreviewUrl: mergedUrl }
        }

        if (cutPlayback) {
          void player.dispose()
        } else {
          player.markFinished()
          player.disposeAfterDrain()
        }
        player = null
      }

      if (resolvedEntry) {
        setMessages([...nextMessages, resolvedEntry])
      } else if (errorMessage) {
        setMessages([
          ...nextMessages,
          {
            id: createId('assistant'),
            role: 'assistant',
            kind: 'assistant',
            text: errorMessage,
            error: true,
          },
        ])
      }

      setPendingReply(null)
      setIsGenerating(false)

      if (streamingWsRef.current === ws) {
        streamingWsRef.current = null
      }

      if (streamingPlayerRef.current === player) {
        streamingPlayerRef.current = null
      }

      if (streamingStopRef.current) {
        streamingStopRef.current = null
      }
    }

    streamingStopRef.current = () => {
      stoppedByUser = true
      try {
        ws.close()
      } catch {
        /* ignore */
      }
    }

    ws.onopen = () => {
      try {
        ws.send(
          JSON.stringify(buildChatRequestBody(nextMessages, systemMessage, true)),
        )
      } catch (error) {
        finalize(null, {
          errorMessage: `发送失败：${getErrorMessage(error)}`,
          cutPlayback: true,
        })
        try {
          ws.close()
        } catch {
          /* ignore */
        }
      }
    }

    ws.onmessage = (event) => {
      let msg: {
        type?: string
        text_delta?: string
        text?: string
        audio_data?: string
        audio_sample_rate?: number
        recording_session_id?: string | null
        error?: string
      }

      try {
        msg = JSON.parse(event.data)
      } catch {
        return
      }

      if (msg.type === 'prefill_done') {
        return
      }

      if (msg.type === 'chunk') {
        if (typeof msg.text_delta === 'string' && msg.text_delta) {
          fullText += msg.text_delta
          setPendingReply({
            id: pendingId,
            role: 'assistant',
            kind: 'pending',
            text: fullText,
          })
        }

        if (msg.audio_data && player) {
          if (typeof msg.audio_sample_rate === 'number') {
            lastSampleRate = msg.audio_sample_rate
          }
          try {
            player.pushBase64(msg.audio_data)
          } catch {
            /* ignore */
          }
        }
        return
      }

      if (msg.type === 'done') {
        const finalText = (fullText || msg.text || '').trim() || '(空回复)'
        const recordingSessionId = msg.recording_session_id ?? null

        if (recordingSessionId) {
          setLastSessionId(recordingSessionId)
        }

        finalize(
          {
            id: createId('assistant'),
            role: 'assistant',
            kind: 'assistant',
            text: finalText,
            audioPreviewUrl: null,
            recordingSessionId,
          },
          { cutPlayback: false },
        )

        try {
          ws.close()
        } catch {
          /* ignore */
        }
        return
      }

      if (msg.type === 'error') {
        finalize(null, {
          errorMessage: `请求失败：${msg.error || 'unknown error'}`,
          cutPlayback: true,
        })
        try {
          ws.close()
        } catch {
          /* ignore */
        }
      }
    }

    ws.onerror = () => {
      finalize(null, {
        errorMessage: 'WebSocket 连接异常',
        cutPlayback: true,
      })
    }

    ws.onclose = () => {
      if (finished) {
        return
      }

      if (stoppedByUser) {
        finalize(
          {
            id: createId('assistant'),
            role: 'assistant',
            kind: 'assistant',
            text: fullText.trim() || '已停止当前回复。',
            audioPreviewUrl: null,
            recordingSessionId: null,
          },
          { cutPlayback: true },
        )
      } else {
        finalize(null, {
          errorMessage: '连接已关闭',
          cutPlayback: true,
        })
      }
    }
  }

  async function sendTextMessage() {
    const text = draft.trim()

    if (!text || isGenerating || isPreparingRecording) {
      return
    }

    setDraft('')
    setRecordError(null)

    const nextMessages: ConversationEntry[] = [
      ...messagesRef.current,
      {
        id: createId('user'),
        role: 'user',
        kind: 'text',
        text,
      },
    ]

    setMessages(nextMessages)
    await submitConversation(nextMessages)
  }

  async function startRecording() {
    if (
      isGenerating ||
      isRecording ||
      isPreparingRecording ||
      composeMode !== 'voice' ||
      screen !== 'turn'
    ) {
      return
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setRecordError('当前浏览器不支持麦克风录音。')
      return
    }

    if (typeof MediaRecorder === 'undefined') {
      setRecordError('当前浏览器不支持 MediaRecorder。')
      return
    }

    setRecordError(null)
    setIsPreparingRecording(true)

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)

      mediaStreamRef.current = stream
      mediaRecorderRef.current = recorder
      audioChunksRef.current = []
      recordingActionRef.current = 'send'

      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      recorder.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, {
          type: recorder.mimeType || 'audio/webm',
        })
        const durationMs = Math.max(0, performance.now() - recordingStartRef.current)
        const shouldSend = recordingActionRef.current === 'send'

        resetRecorderResources()

        try {
          if (!shouldSend) {
            return
          }

          if (durationMs < 300 || blob.size === 0) {
            setRecordError('录音太短了，请再试一次。')
            return
          }

          const audioBase64 = await convertAudioBlobToFloat32Base64(blob)
          const previewUrl = URL.createObjectURL(blob)
          const nextMessages: ConversationEntry[] = [
            ...messagesRef.current,
            {
              id: createId('voice'),
              role: 'user',
              kind: 'voice',
              audioBase64,
              durationMs,
              previewUrl,
            },
          ]

          setMessages(nextMessages)
          await submitConversation(nextMessages)
        } catch (error) {
          setRecordError(`录音处理失败：${getErrorMessage(error)}`)
        } finally {
          setIsPreparingRecording(false)
        }
      }

      recorder.start()
      recordingStartRef.current = performance.now()
      setIsRecording(true)
    } catch (error) {
      setRecordError(`无法开始录音：${getErrorMessage(error)}`)
      resetRecorderResources()
      setIsPreparingRecording(false)
    }
  }

  function stopRecording(action: 'send' | 'cancel') {
    recordingActionRef.current = action

    const recorder = mediaRecorderRef.current

    if (recorder && recorder.state === 'recording') {
      recorder.stop()
    } else {
      resetRecorderResources()
      setIsPreparingRecording(false)
    }

    setIsRecording(false)
  }

  function handleComposerSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void sendTextMessage()
  }

  const voiceMainLabel = isRecording
    ? '松开发送'
    : isPreparingRecording
      ? '处理中...'
      : isGenerating
        ? 'AI 回复中...'
        : '按住说话'

  const voiceSubLabel = isRecording
    ? '继续按住保持录音'
    : isPreparingRecording
      ? '正在把录音转成 16k 音频'
      : '录完后自动发给模型'

  const secondaryLabel = isGenerating
    ? '停止'
    : composeMode === 'voice'
      ? '文字'
      : '语音'

  const talkVisualIcon =
    isRecording || isPreparingRecording ? (
      <WaveIcon className="app-icon app-icon-xl talk-icon" />
    ) : (
      <MicIcon className="app-icon app-icon-xl talk-icon" />
    )

  return (
    <div className="mobile-app">
      <input
        ref={refAudioInputRef}
        className="hidden-file-input"
        type="file"
        accept="audio/*"
        onChange={handleRefAudioInputChange}
      />
      <SettingsSheet
        open={settingsOpen}
        activeMode={activePresetMode}
        activeLabel={activeModeLabel}
        activeSettings={activeModeSettings}
        activePresets={activeModePresets}
        defaultRefAudio={defaultRefAudio}
        lengthPenalty={activeLengthPenalty}
        maxNewTokens={settings.maxNewTokens}
        turnTtsEnabled={settings.turnTtsEnabled}
        turnStreamingEnabled={settings.turnStreamingEnabled}
        onClose={() => {
          setSettingsOpen(false)
        }}
        onSelectPreset={(presetId) => {
          void handleSelectPreset(activePresetMode, presetId)
        }}
        onPromptChange={(value) => {
          handleChangePrompt(activePresetMode, value)
        }}
        onLengthPenaltyChange={(value) => {
          handleLengthPenaltyChange(activePresetMode, value)
        }}
        onMaxTokensChange={(value) => {
          setSettings((previous) => ({
            ...previous,
            maxNewTokens: Number.isFinite(value) ? value : previous.maxNewTokens,
          }))
        }}
        onTurnTtsEnabledChange={(value) => {
          setSettings((previous) => ({
            ...previous,
            turnTtsEnabled: value,
          }))
        }}
        onTurnStreamingEnabledChange={(value) => {
          setSettings((previous) => ({
            ...previous,
            turnStreamingEnabled: value,
          }))
        }}
        onUseDefaultRefAudio={() => {
          handleUseDefaultRefAudio(activePresetMode)
        }}
        onClearRefAudio={() => {
          handleClearRefAudio(activePresetMode)
        }}
        onUploadRefAudio={() => {
          refAudioInputRef.current?.click()
        }}
        onPlayRefAudio={handlePlayActiveRefAudio}
      />
      {screen === 'turn' ? (
        <div className="turn-screen">
          <div className="turn-topbar">
            <div className={`service-chip ${serviceState.phase}`}>
              <span className="service-dot" />
              <span>{serviceState.summary}</span>
            </div>
            <div className="turn-top-actions">
              <button
                className="entry-action audio"
                type="button"
                onClick={() => {
                  openDuplexScreen('audio')
                }}
                disabled={isGenerating || isRecording || isPreparingRecording}
                aria-label="进入音频双工"
              >
                <PhoneIcon className="app-icon app-icon-md" />
                <span className="button-inline-label">音频</span>
              </button>
              <button
                className="entry-action video"
                type="button"
                onClick={() => {
                  openDuplexScreen('video')
                }}
                disabled={isGenerating || isRecording || isPreparingRecording}
                aria-label="进入视频双工"
              >
                <VideoCallIcon className="app-icon app-icon-md" />
                <span className="button-inline-label">视频</span>
              </button>
            </div>
          </div>

          <div className="turn-settings-wrap">
            <SettingsSummary
              modeLabel="Turn-based"
              presetName={turnPresetName}
              refAudio={settings.turnbased.refAudio}
              systemPrompt={settings.turnbased.systemPrompt}
              lengthPenalty={settings.turnLengthPenalty}
              maxNewTokens={settings.maxNewTokens}
              turnTtsEnabled={settings.turnTtsEnabled}
              turnStreamingEnabled={settings.turnStreamingEnabled}
              onOpen={() => {
                setSettingsOpen(true)
              }}
            />
          </div>

          <div className="thread-wrap">
            {!messages.length && !pendingReply ? (
              <div className="empty-hint">
                你好，这里已经接上真实 turn-based 后端。你可以先发一条文字消息，也可以按住说话试一下。
              </div>
            ) : null}

            <div className="thread">
              {threadEntries.map((entry) => (
                <MessageBubble key={entry.id} entry={entry} />
              ))}
              <div ref={threadEndRef} />
            </div>
          </div>

          <div className="composer">
            {recordError ? <div className="helper-error">{recordError}</div> : null}

            {composeMode === 'text' ? (
              <form className="text-composer" onSubmit={handleComposerSubmit}>
                <textarea
                  className="text-input"
                  placeholder="输入文字后发送..."
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                      event.preventDefault()
                      void sendTextMessage()
                    }
                  }}
                  disabled={isGenerating || isPreparingRecording}
                />
                <div className="text-actions">
                  <button
                    className="send-btn"
                    type={isGenerating ? 'button' : 'submit'}
                    onClick={isGenerating ? stopCurrentReply : undefined}
                  >
                    {isGenerating ? (
                      <StopIcon className="app-icon app-icon-md" />
                    ) : (
                      <SendIcon className="app-icon app-icon-md" />
                    )}
                    <span className="button-inline-label">
                      {isGenerating ? '停止' : '发送'}
                    </span>
                  </button>
                  <button
                    className="secondary-btn"
                    type="button"
                    onClick={() => {
                      if (isGenerating) {
                        stopCurrentReply()
                        return
                      }

                      setComposeMode('voice')
                    }}
                  >
                    {isGenerating ? (
                      <StopIcon className="app-icon app-icon-md" />
                    ) : (
                      <MicIcon className="app-icon app-icon-md" />
                    )}
                    <span className="button-inline-label">{secondaryLabel}</span>
                  </button>
                </div>
              </form>
            ) : (
              <div className="talk-row">
                <button
                  className={[
                    'talk-bar',
                    isRecording ? 'recording' : '',
                    isGenerating || isPreparingRecording ? 'disabled' : '',
                  ]
                    .filter(Boolean)
                    .join(' ')}
                  type="button"
                  onPointerDown={() => {
                    void startRecording()
                  }}
                  onPointerUp={() => stopRecording('send')}
                  onPointerLeave={() => {
                    if (isRecording) {
                      stopRecording('send')
                    }
                  }}
                  onPointerCancel={() => stopRecording('cancel')}
                  disabled={isGenerating || isPreparingRecording}
                >
                  <div className="talk-visual" aria-hidden="true">
                    <div className="talk-orb">{talkVisualIcon}</div>
                  </div>
                  <div className="talk-copy">
                    <div className="talk-main">{voiceMainLabel}</div>
                    <div className="talk-sub">{voiceSubLabel}</div>
                  </div>
                </button>
                <button
                  className="text-side-btn"
                  type="button"
                  onClick={() => {
                    if (isGenerating) {
                      stopCurrentReply()
                      return
                    }

                    setComposeMode('text')
                  }}
                >
                  {isGenerating ? (
                    <StopIcon className="app-icon app-icon-lg" />
                  ) : (
                    <KeyboardIcon className="app-icon app-icon-lg" />
                  )}
                  <span className="button-stack-label">{secondaryLabel}</span>
                </button>
              </div>
            )}

            <div className="helper-row">
              <span className="helper-text">{serviceState.detail}</span>
              {lastSessionId ? (
                <span className="helper-text helper-text-strong">{lastSessionId}</span>
              ) : null}
            </div>
          </div>
        </div>
      ) : audioDuplexScreenOpen ? (
        <div className="duplex-screen audio-mode">
          <div className="top-actions">
            <button
              className="top-action-button"
              type="button"
              onClick={() => {
                setSettingsOpen(true)
              }}
            >
              <SettingsIcon className="app-icon app-icon-md" />
              <span className="button-inline-label">设置</span>
            </button>
            <button
              className="top-action-button"
              type="button"
              onClick={() => {
                setDuplexTextPanelOpen((previous) => !previous)
              }}
            >
              <TranscriptIcon className="app-icon app-icon-md" />
              <span className="button-inline-label">字幕</span>
            </button>
          </div>

          <div className="duplex-badge-row">
            <div className={`duplex-badge ${duplexStatus}`}>{duplexBadgeText}</div>
            <div className="duplex-status-copy">{duplexStatusText}</div>
          </div>

          <div className="duplex-settings-wrap">
            <SettingsSummary
              modeLabel="音频双工"
              presetName={audioPresetName}
              refAudio={settings.audio_duplex.refAudio}
              systemPrompt={settings.audio_duplex.systemPrompt}
              lengthPenalty={settings.audioDuplexLengthPenalty}
              onOpen={() => {
                setSettingsOpen(true)
              }}
            />
          </div>

          <div className="duplex-stage audio-stage">
            <video
              ref={duplexVideoRef}
              className="duplex-video hidden"
              autoPlay
              muted
              playsInline
            />
            <canvas ref={duplexCanvasRef} className="duplex-capture-canvas" />

            <div className="audio-stage-core">
              <div className="audio-stage-orb" aria-hidden="true">
                <WaveIcon className="app-icon app-icon-xl" />
              </div>
              <div className="audio-stage-title">纯音频双工</div>
              <div className="audio-stage-copy">
                这一页只上传麦克风音频，不发送摄像头画面，适合耳机或通话场景。
              </div>
            </div>

            {duplexTextPanelOpen ? (
              <div className="duplex-transcript">
                {duplexEntries.length ? (
                  duplexEntries.map((entry) => (
                    <DuplexLogBubble key={entry.id} entry={entry} />
                  ))
                ) : (
                  <div className="duplex-transcript-empty">
                    进入后会在这里显示系统状态、用户转写和模型回复。
                  </div>
                )}
                <div ref={duplexEndRef} />
              </div>
            ) : null}
          </div>

          <div className="control-strip">
            <button
              className={['circle-btn', duplexMicEnabled ? '' : 'muted']
                .filter(Boolean)
                .join(' ')}
              type="button"
              onClick={() => {
                setDuplexMicEnabled((previous) => !previous)
              }}
            >
              <MicIcon className="app-icon app-icon-lg" />
              <span className="circle-btn-label">
                {duplexMicEnabled ? '麦克风' : '已静音'}
              </span>
            </button>
            <button
              className={['circle-btn', duplexPauseState === 'active' ? '' : 'muted']
                .filter(Boolean)
                .join(' ')}
              type="button"
              onClick={() => {
                duplexSessionRef.current?.pauseToggle()
              }}
              disabled={!duplexSessionRef.current}
            >
              {duplexPauseState === 'active' ? (
                <PauseIcon className="app-icon app-icon-lg" />
              ) : (
                <PlayIcon className="app-icon app-icon-lg" />
              )}
              <span className="circle-btn-label">
                {duplexPauseState === 'active' ? '暂停' : '继续'}
              </span>
            </button>
            <button
              className="circle-btn danger"
              type="button"
              onClick={() => {
                stopDuplexSession()
              }}
            >
              <CloseIcon className="app-icon app-icon-lg" />
              <span className="circle-btn-label">退出</span>
            </button>
          </div>
        </div>
      ) : (
        <div className="duplex-screen">
          <div className="top-actions">
            <button
              className="top-action-button"
              type="button"
              onClick={() => {
                setSettingsOpen(true)
              }}
            >
              <SettingsIcon className="app-icon app-icon-md" />
              <span className="button-inline-label">设置</span>
            </button>
            <button
              className="top-action-button"
              type="button"
              onClick={() => {
                setDuplexTextPanelOpen((previous) => !previous)
              }}
            >
              <TranscriptIcon className="app-icon app-icon-md" />
              <span className="button-inline-label">字幕</span>
            </button>
            <button
              className="top-action-button"
              type="button"
              onClick={() => {
                const action = duplexMediaRef.current?.flipCamera()

                if (action) {
                  void action.catch((error: unknown) => {
                    appendDuplexEntry(
                      'system',
                      `翻转摄像头失败：${getErrorMessage(error)}`,
                    )
                  })
                }
              }}
            >
              <FlipCameraIcon className="app-icon app-icon-md" />
              <span className="button-inline-label">翻转</span>
            </button>
          </div>

          <div className="duplex-badge-row">
            <div className={`duplex-badge ${duplexStatus}`}>{duplexBadgeText}</div>
            <div className="duplex-status-copy">{duplexStatusText}</div>
          </div>

          <div className="duplex-settings-wrap">
            <SettingsSummary
              modeLabel="视频双工"
              presetName={videoPresetName}
              refAudio={settings.omni.refAudio}
              systemPrompt={settings.omni.systemPrompt}
              lengthPenalty={settings.videoDuplexLengthPenalty}
              onOpen={() => {
                setSettingsOpen(true)
              }}
            />
          </div>

          <div className="duplex-stage">
            <video
              ref={duplexVideoRef}
              className={['duplex-video', videoDuplexScreenOpen ? 'visible' : 'hidden']
                .filter(Boolean)
                .join(' ')}
              autoPlay
              muted
              playsInline
            />
            <canvas ref={duplexCanvasRef} className="duplex-capture-canvas" />

            {duplexTextPanelOpen ? (
              <div className="duplex-transcript">
                {duplexEntries.length ? (
                  duplexEntries.map((entry) => (
                    <DuplexLogBubble key={entry.id} entry={entry} />
                  ))
                ) : (
                  <div className="duplex-transcript-empty">
                    进入后会在这里显示系统状态、用户转写和模型回复。
                  </div>
                )}
                <div ref={duplexEndRef} />
              </div>
            ) : null}
          </div>

          <div className="control-strip">
            <button
              className={['circle-btn', duplexMicEnabled ? '' : 'muted']
                .filter(Boolean)
                .join(' ')}
              type="button"
              onClick={() => {
                setDuplexMicEnabled((previous) => !previous)
              }}
            >
              <MicIcon className="app-icon app-icon-lg" />
              <span className="circle-btn-label">
                {duplexMicEnabled ? '麦克风' : '已静音'}
              </span>
            </button>
            <button
              className={['circle-btn', duplexPauseState === 'active' ? '' : 'muted']
                .filter(Boolean)
                .join(' ')}
              type="button"
              onClick={() => {
                duplexSessionRef.current?.pauseToggle()
              }}
              disabled={!duplexSessionRef.current}
            >
              {duplexPauseState === 'active' ? (
                <PauseIcon className="app-icon app-icon-lg" />
              ) : (
                <PlayIcon className="app-icon app-icon-lg" />
              )}
              <span className="circle-btn-label">
                {duplexPauseState === 'active' ? '暂停' : '继续'}
              </span>
            </button>
            <button
              className="circle-btn danger"
              type="button"
              onClick={() => {
                stopDuplexSession()
              }}
            >
              <CloseIcon className="app-icon app-icon-lg" />
              <span className="circle-btn-label">退出</span>
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
