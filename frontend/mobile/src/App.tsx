import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react'
import { AudioDuplexScreen } from './duplex/AudioDuplexScreen'
import { VideoDuplexScreen } from './duplex/VideoDuplexScreen'
import { useDuplexSession } from './duplex/useDuplexSession'
import type { DuplexIcons } from './duplex/types'
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

type Attachment =
  | {
      id: string
      kind: 'image'
      previewUrl: string
      base64: string
      name: string
    }
  | {
      id: string
      kind: 'audio'
      previewUrl: string
      base64: string
      name: string
      duration?: number
    }
  | {
      id: string
      kind: 'video'
      previewUrl: string
      base64: string
      name: string
      duration?: number
    }

type ConversationEntry =
  | {
      id: string
      role: 'assistant'
      kind: 'assistant'
      text: string
      error?: boolean
      interrupted?: boolean
      audioPreviewUrl?: string | null
      recordingSessionId?: string | null
    }
  | {
      id: string
      role: 'user'
      kind: 'text'
      text: string
      attachments?: Attachment[]
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

type Screen = 'turn' | 'audio-duplex' | 'video-duplex'

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

const CANCEL_DRAG_PX = 80
const MIN_HOLD_MS = 250
const TAP_HINT_MS = 1200

const ACTIVE_SESSION_STORAGE_KEY = 'mobile.turn.activeSessionId.v1'
const SESSIONS_DB_NAME = 'mobile-turn-db'
const SESSIONS_DB_VERSION = 1
const SESSIONS_STORE = 'sessions'

type ChatSession = {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  messages: ConversationEntry[]
}

function deriveSessionTitle(messages: ConversationEntry[]): string {
  for (const m of messages) {
    if (m.role !== 'user') continue
    if (m.kind === 'text' && m.text.trim()) {
      const t = m.text.trim().replace(/\s+/g, ' ')
      return t.length > 28 ? `${t.slice(0, 28)}…` : t
    }
    if (m.kind === 'voice') return '语音对话'
    if (m.kind === 'text' && m.attachments && m.attachments.length > 0) {
      const a = m.attachments[0]
      if (a.kind === 'image') return '图片对话'
      if (a.kind === 'audio') return '音频对话'
      if (a.kind === 'video') return '视频对话'
    }
  }
  return '新对话'
}

function stripBlobUrls(messages: ConversationEntry[]): ConversationEntry[] {
  return messages.map((m) => {
    if (m.role === 'user' && m.kind === 'voice') {
      return { ...m, previewUrl: '' }
    }
    if (m.role === 'user' && m.kind === 'text' && m.attachments) {
      return {
        ...m,
        attachments: m.attachments.map((a) => ({ ...a, previewUrl: '' })),
      }
    }
    if (m.role === 'assistant') {
      return { ...m, audioPreviewUrl: null }
    }
    return m
  })
}

function base64ToBlob(base64: string, mime: string): Blob | null {
  try {
    const bin = atob(base64)
    const len = bin.length
    const bytes = new Uint8Array(len)
    for (let i = 0; i < len; i += 1) bytes[i] = bin.charCodeAt(i)
    return new Blob([bytes], { type: mime })
  } catch {
    return null
  }
}

function rehydrateMessages(messages: ConversationEntry[]): ConversationEntry[] {
  return messages.map((m) => {
    if (m.role === 'user' && m.kind === 'text' && m.attachments) {
      return {
        ...m,
        attachments: m.attachments.map((a) => {
          if (a.previewUrl) return a
          let mime = 'application/octet-stream'
          if (a.kind === 'image') mime = 'image/jpeg'
          else if (a.kind === 'audio') mime = 'audio/webm'
          else if (a.kind === 'video') mime = 'video/mp4'
          const blob = base64ToBlob(a.base64, mime)
          return blob
            ? { ...a, previewUrl: URL.createObjectURL(blob) }
            : a
        }),
      }
    }
    return m
  })
}

let _dbPromise: Promise<IDBDatabase> | null = null
function openSessionsDb(): Promise<IDBDatabase> {
  if (typeof indexedDB === 'undefined') {
    return Promise.reject(new Error('IndexedDB unavailable'))
  }
  if (_dbPromise) return _dbPromise
  _dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(SESSIONS_DB_NAME, SESSIONS_DB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(SESSIONS_STORE)) {
        db.createObjectStore(SESSIONS_STORE, { keyPath: 'id' })
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
  return _dbPromise
}

async function idbGetAllSessions(): Promise<ChatSession[]> {
  try {
    const db = await openSessionsDb()
    return await new Promise<ChatSession[]>((resolve, reject) => {
      const tx = db.transaction(SESSIONS_STORE, 'readonly')
      const store = tx.objectStore(SESSIONS_STORE)
      const req = store.getAll()
      req.onsuccess = () => {
        const rows = (req.result as ChatSession[]) || []
        resolve(
          rows
            .filter((s) => s && typeof s.id === 'string' && Array.isArray(s.messages))
            .map((s) => ({
              id: s.id,
              title: s.title || '新对话',
              createdAt: Number(s.createdAt) || Date.now(),
              updatedAt: Number(s.updatedAt) || Number(s.createdAt) || Date.now(),
              messages: rehydrateMessages(s.messages),
            })),
        )
      }
      req.onerror = () => reject(req.error)
    })
  } catch (err) {
    console.warn('IDB read failed', err)
    return []
  }
}

async function idbPutSession(session: ChatSession): Promise<void> {
  try {
    const db = await openSessionsDb()
    const record: ChatSession = {
      ...session,
      messages: stripBlobUrls(session.messages),
    }
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(SESSIONS_STORE, 'readwrite')
      tx.objectStore(SESSIONS_STORE).put(record)
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  } catch (err) {
    console.warn('IDB write failed', err)
  }
}

async function idbDeleteSession(id: string): Promise<void> {
  try {
    const db = await openSessionsDb()
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(SESSIONS_STORE, 'readwrite')
      tx.objectStore(SESSIONS_STORE).delete(id)
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  } catch (err) {
    console.warn('IDB delete failed', err)
  }
}

async function idbClearAll(): Promise<void> {
  try {
    const db = await openSessionsDb()
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(SESSIONS_STORE, 'readwrite')
      tx.objectStore(SESSIONS_STORE).clear()
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  } catch (err) {
    console.warn('IDB clear failed', err)
  }
}

function formatRelativeTime(ts: number): string {
  const diff = Date.now() - ts
  if (diff < 60_000) return '刚刚'
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} 分钟前`
  const today = new Date()
  const d = new Date(ts)
  if (
    today.getFullYear() === d.getFullYear() &&
    today.getMonth() === d.getMonth() &&
    today.getDate() === d.getDate()
  ) {
    return `${d.getHours().toString().padStart(2, '0')}:${d
      .getMinutes()
      .toString()
      .padStart(2, '0')}`
  }
  const yesterday = new Date(Date.now() - 86_400_000)
  if (
    yesterday.getFullYear() === d.getFullYear() &&
    yesterday.getMonth() === d.getMonth() &&
    yesterday.getDate() === d.getDate()
  ) {
    return '昨天'
  }
  return `${d.getMonth() + 1}/${d.getDate()}`
}

function autoGrowTextarea(el: HTMLTextAreaElement | null): void {
  if (!el) return
  el.style.height = 'auto'
  const max = 140
  const next = Math.min(el.scrollHeight, max)
  el.style.height = `${next}px`
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

function CameraSnapIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        d="M5 9.5A2.5 2.5 0 0 1 7.5 7h1.6l1.4-2h3l1.4 2h1.6A2.5 2.5 0 0 1 19 9.5v7A2.5 2.5 0 0 1 16.5 19h-9A2.5 2.5 0 0 1 5 16.5Z"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.6"
      />
      <circle cx="12" cy="13" r="3.2" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  )
}

function HamburgerIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        d="M4.5 7h15M4.5 12h15M4.5 17h15"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.8"
      />
    </svg>
  )
}

function CopyIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <rect
        x="8.5"
        y="8.5"
        width="10"
        height="11"
        rx="2.2"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.6"
      />
      <path
        d="M15.5 6h-7A2 2 0 0 0 6.5 8v9"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.6"
      />
    </svg>
  )
}

function RefreshIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        d="M5.5 12a6.5 6.5 0 0 1 11.2-4.5L19 10"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.7"
      />
      <path
        d="M19 5v5h-5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.7"
      />
      <path
        d="M18.5 12a6.5 6.5 0 0 1-11.2 4.5L5 14"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.7"
      />
      <path
        d="M5 19v-5h5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.7"
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

function TrashIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        d="M5 7h14M9.5 7V5.5a1.5 1.5 0 0 1 1.5-1.5h2a1.5 1.5 0 0 1 1.5 1.5V7m-7.5 0 .8 11.2a1.8 1.8 0 0 0 1.8 1.6h6.8a1.8 1.8 0 0 0 1.8-1.6L17.5 7"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.6"
      />
      <path
        d="M10.5 11v5M13.5 11v5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.6"
      />
    </svg>
  )
}

function EditSquareIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        d="M5 6.8A2.2 2.2 0 0 1 7.2 4.6h5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.6"
      />
      <path
        d="M19.4 11v5.8a2.2 2.2 0 0 1-2.2 2.2H7.2A2.2 2.2 0 0 1 5 16.8V11"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.6"
      />
      <path
        d="m13.6 11.5 6-6a1.6 1.6 0 0 1 2.3 2.3l-6 6-2.7.4.4-2.7Z"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.6"
      />
    </svg>
  )
}

function PlusIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle cx="12" cy="12" r="9.2" stroke="currentColor" strokeWidth="1.6" />
      <path
        d="M12 8v8M8 12h8"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.8"
      />
    </svg>
  )
}

function PhotoIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <rect
        x="3.5"
        y="5.5"
        width="17"
        height="13"
        rx="2.2"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.6"
      />
      <circle cx="9" cy="10.5" r="1.6" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="M3.7 16.5 9 12l4 3.5 3-2.5 4.3 4"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.6"
      />
    </svg>
  )
}

function MusicIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        d="M9 17V6.5l9-1.7v10.4"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.6"
      />
      <ellipse cx="7" cy="17.5" rx="2.5" ry="2" stroke="currentColor" strokeWidth="1.6" />
      <ellipse cx="16" cy="15.5" rx="2.5" ry="2" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  )
}

function FileIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        d="M14 3.5H7.5A2 2 0 0 0 5.5 5.5v13a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V8.2L14 3.5Z"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.6"
      />
      <path
        d="M13.5 3.5v4.7h5"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.6"
      />
    </svg>
  )
}

function FilmIcon({ className }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
    >
      <rect
        x="3.5"
        y="5.5"
        width="17"
        height="13"
        rx="1.8"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.6"
      />
      <path
        d="M3.5 9h17M3.5 15h17M8 5.5v13M16 5.5v13"
        stroke="currentColor"
        strokeWidth="1.4"
      />
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
      const atts = entry.attachments ?? []
      if (atts.length === 0) {
        return {
          role: 'user',
          content: entry.text,
        }
      }
      const items: BackendContentItem[] = []
      for (const a of atts) {
        if (a.kind === 'image') {
          items.push({ type: 'image', data: a.base64 })
        } else if (a.kind === 'audio') {
          items.push({ type: 'audio', data: a.base64, name: a.name, duration: a.duration })
        } else {
          items.push({ type: 'video', data: a.base64, duration: a.duration })
        }
      }
      if (entry.text) {
        items.push({ type: 'text', text: entry.text })
      }
      return {
        role: 'user',
        content: items,
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

async function fileToBase64Stripped(file: Blob): Promise<string> {
  return await new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = String(reader.result ?? '')
      const i = result.indexOf(',')
      resolve(i >= 0 ? result.slice(i + 1) : result)
    }
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

async function readFileAsDataUrl(file: Blob): Promise<string> {
  return await new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result ?? ''))
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

async function downscaleImageToAttachment(
  file: File,
  maxEdge = 1280,
  quality = 0.85,
): Promise<Attachment> {
  const dataUrl = await readFileAsDataUrl(file)
  const img: HTMLImageElement = await new Promise((resolve, reject) => {
    const i = new Image()
    i.onload = () => resolve(i)
    i.onerror = () => reject(new Error('image load failed'))
    i.src = dataUrl
  })

  let w = img.naturalWidth
  let h = img.naturalHeight
  const longEdge = Math.max(w, h)
  if (longEdge > maxEdge) {
    const scale = maxEdge / longEdge
    w = Math.round(w * scale)
    h = Math.round(h * scale)
  }

  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    throw new Error('canvas 2d unavailable')
  }
  ctx.drawImage(img, 0, 0, w, h)
  const outDataUrl = canvas.toDataURL('image/jpeg', quality)
  const base64 = outDataUrl.slice(outDataUrl.indexOf(',') + 1)
  return {
    id: createId('att'),
    kind: 'image',
    previewUrl: outDataUrl,
    base64,
    name: file.name || 'photo.jpg',
  }
}

async function mediaFileToAttachment(
  file: File,
  kind: 'audio' | 'video',
): Promise<Attachment> {
  const base64 = await fileToBase64Stripped(file)
  const previewUrl = URL.createObjectURL(file)
  let duration: number | undefined
  try {
    duration = await new Promise<number>((resolve) => {
      const el = document.createElement(kind === 'audio' ? 'audio' : 'video') as
        | HTMLAudioElement
        | HTMLVideoElement
      el.preload = 'metadata'
      const onLoaded = () => {
        const d = Number.isFinite(el.duration) ? el.duration : 0
        resolve(d)
      }
      el.addEventListener('loadedmetadata', onLoaded, { once: true })
      el.addEventListener('error', () => resolve(0), { once: true })
      el.src = previewUrl
    })
  } catch {
    duration = undefined
  }
  return {
    id: createId('att'),
    kind,
    previewUrl,
    base64,
    name: file.name || (kind === 'audio' ? 'audio' : 'video'),
    duration,
  }
}

function getAudioContextCtor(): typeof AudioContext | null {
  return (
    window.AudioContext ??
    (
      window as Window & {
        webkitAudioContext?: typeof AudioContext
      }
    ).webkitAudioContext ??
    null
  )
}

function float32ToBase64(samples: Float32Array): string {
  const bytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength)
  const chunkSize = 0x8000
  let binary = ''

  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const slice = bytes.subarray(offset, Math.min(offset + chunkSize, bytes.length))
    binary += String.fromCharCode.apply(null, Array.from(slice) as number[])
  }

  return btoa(binary)
}

function concatFloat32(chunks: Float32Array[]): Float32Array {
  let total = 0
  for (const chunk of chunks) {
    total += chunk.length
  }
  const out = new Float32Array(total)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.length
  }
  return out
}

function resampleLinear(
  input: Float32Array,
  fromRate: number,
  toRate: number,
): Float32Array {
  if (fromRate === toRate || input.length === 0) {
    return input
  }
  const ratio = fromRate / toRate
  const outLength = Math.max(1, Math.floor(input.length / ratio))
  const out = new Float32Array(outLength)
  for (let i = 0; i < outLength; i += 1) {
    const srcPos = i * ratio
    const idx = Math.floor(srcPos)
    const frac = srcPos - idx
    const a = input[idx] ?? 0
    const b = input[idx + 1] ?? a
    out[i] = a + (b - a) * frac
  }
  return out
}

async function convertAudioBlobToFloat32Base64(blob: Blob): Promise<string> {
  const AudioContextCtor = getAudioContextCtor()

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
    return float32ToBase64(pcm)
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

const CAMERA_QUICK_PROMPTS = [
  '这是什么？',
  '描述图中的场景',
  '提取图中文字',
  '图里的内容讲给我听',
] as const

type CameraReviewOverlayProps = {
  attachment: Attachment
  draft: string
  onDraftChange: (v: string) => void
  onClose: () => void
  onRetake: () => void
  onSend: (text: string) => void
  disabled: boolean
}

function CameraReviewOverlay({
  attachment,
  draft,
  onDraftChange,
  onClose,
  onRetake,
  onSend,
  disabled,
}: CameraReviewOverlayProps) {
  return (
    <div className="camera-review">
      <div className="camera-review-topbar">
        <button
          type="button"
          className="camera-review-icon-btn"
          onClick={onClose}
          aria-label="放弃"
        >
          <CloseIcon className="app-icon app-icon-md" />
        </button>
        <div className="camera-review-topbar-spacer" />
        <button
          type="button"
          className="camera-review-icon-btn"
          onClick={onRetake}
          aria-label="重拍"
        >
          <RefreshIcon className="app-icon app-icon-md" />
        </button>
      </div>

      <div className="camera-review-stage">
        <img src={attachment.previewUrl} alt="拍摄的照片" />
      </div>

      <div className="camera-review-bottom">
        <div className="camera-review-chips">
          {CAMERA_QUICK_PROMPTS.map((p) => (
            <button
              key={p}
              type="button"
              className="camera-review-chip"
              disabled={disabled}
              onClick={() => onSend(p)}
            >
              {p}
              <span className="camera-review-chip-arrow">→</span>
            </button>
          ))}
        </div>

        <form
          className="camera-review-composer"
          onSubmit={(e) => {
            e.preventDefault()
            onSend(draft)
          }}
        >
          <input
            className="camera-review-input"
            type="text"
            value={draft}
            placeholder="问点关于这张图的…"
            onChange={(e) => onDraftChange(e.target.value)}
            disabled={disabled}
            autoFocus
          />
          <button
            type="submit"
            className="camera-review-send"
            disabled={disabled}
            aria-label="发送"
          >
            <SendIcon className="app-icon app-icon-md" />
          </button>
        </form>
      </div>
    </div>
  )
}

function MessageAttachment({ attachment }: { attachment: Attachment }) {
  if (attachment.kind === 'image') {
    return (
      <div className="msg-att msg-att-image">
        <img src={attachment.previewUrl} alt={attachment.name} />
      </div>
    )
  }
  if (attachment.kind === 'audio') {
    return (
      <div className="msg-att msg-att-audio">
        <AudioPlayPill url={attachment.previewUrl} />
      </div>
    )
  }
  return (
    <div className="msg-att msg-att-video">
      <video src={attachment.previewUrl} controls preload="metadata" playsInline />
    </div>
  )
}

type MessageBubbleProps = {
  entry: ThreadEntry
  isLastAssistant?: boolean
  canRegenerate?: boolean
  onRegenerate?: () => void
}

function MessageBubble({
  entry,
  isLastAssistant,
  canRegenerate,
  onRegenerate,
}: MessageBubbleProps) {
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

  const isAssistant = entry.role === 'assistant'
  const audioUrl = isAssistant ? entry.audioPreviewUrl ?? null : null
  const showActions = isAssistant && !entry.error
  const attachments =
    !isAssistant && entry.kind === 'text' ? entry.attachments ?? [] : []

  return (
    <div
      className={[
        'msg',
        isAssistant ? 'assistant' : 'user-text',
        isAssistant && entry.error ? 'error' : '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {attachments.length > 0 ? (
        <div className="msg-attachments">
          {attachments.map((a) => (
            <MessageAttachment key={a.id} attachment={a} />
          ))}
        </div>
      ) : null}
      {entry.text ? <div className="msg-text">{entry.text}</div> : null}
      {isAssistant && entry.interrupted ? (
        <div className="msg-interrupted">已中断</div>
      ) : null}
      {showActions ? (
        <div className="msg-actions">
          <CopyButton text={entry.text} />
          <AssistantPlayButton url={audioUrl} />
          {isLastAssistant ? (
            <button
              className="msg-action msg-action-trailing"
              type="button"
              onClick={onRegenerate}
              disabled={!canRegenerate || !onRegenerate}
              aria-label="重新生成"
            >
              <RefreshIcon className="app-icon app-icon-md" />
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

function AssistantPlayButton({ url }: { url: string | null }) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)

  useEffect(() => {
    if (!url) {
      setIsPlaying(false)
      audioRef.current = null
      return
    }

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
      setIsPlaying(false)
    }
  }, [url])

  const disabled = !url

  function handleClick() {
    const audio = audioRef.current
    if (!audio) return
    if (isPlaying) {
      audio.pause()
      return
    }
    try {
      audio.currentTime = 0
    } catch {
      /* ignore */
    }
    void audio.play().catch(() => setIsPlaying(false))
  }

  return (
    <button
      className={['msg-action', isPlaying ? 'is-playing' : ''].filter(Boolean).join(' ')}
      type="button"
      onClick={handleClick}
      disabled={disabled}
      aria-label={isPlaying ? '暂停播放' : '朗读'}
    >
      {isPlaying ? (
        <PauseIcon className="app-icon app-icon-md" />
      ) : (
        <PlayIcon className="app-icon app-icon-md" />
      )}
    </button>
  )
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!copied) return
    const timer = window.setTimeout(() => setCopied(false), 1200)
    return () => window.clearTimeout(timer)
  }, [copied])

  async function handleClick() {
    if (!text) return
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text)
      } else {
        const ta = document.createElement('textarea')
        ta.value = text
        ta.style.position = 'fixed'
        ta.style.opacity = '0'
        document.body.appendChild(ta)
        ta.select()
        try {
          document.execCommand('copy')
        } finally {
          document.body.removeChild(ta)
        }
      }
      setCopied(true)
    } catch {
      /* ignore */
    }
  }

  return (
    <button
      className={['msg-action', copied ? 'is-copied' : ''].filter(Boolean).join(' ')}
      type="button"
      onClick={() => {
        void handleClick()
      }}
      aria-label={copied ? '已复制' : '复制'}
    >
      <CopyIcon className="app-icon app-icon-md" />
    </button>
  )
}

function RecordingOverlay({ willCancel }: { willCancel: boolean }) {
  return (
    <div
      className={['recording-overlay', willCancel ? 'will-cancel' : ''].filter(Boolean).join(' ')}
      aria-live="polite"
      aria-atomic="true"
    >
      <div className="recording-overlay-bg" aria-hidden="true" />
      <div className="recording-overlay-inner">
        <div className="recording-overlay-text">
          {willCancel ? '松手取消' : '松手发送，上移取消'}
        </div>
        <div className="recording-waveform" aria-hidden="true">
          {Array.from({ length: 28 }).map((_, i) => (
            <span key={i} style={{ animationDelay: `${(i % 14) * 60}ms` }} />
          ))}
        </div>
      </div>
    </div>
  )
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

const duplexIcons: DuplexIcons = {
  Settings: SettingsIcon,
  Transcript: TranscriptIcon,
  Mic: MicIcon,
  Pause: PauseIcon,
  Play: PlayIcon,
  Close: CloseIcon,
  Wave: WaveIcon,
  FlipCamera: FlipCameraIcon,
}

type HistoryDrawerProps = {
  open: boolean
  sessions: ChatSession[]
  activeId: string
  onClose: () => void
  onNewSession: () => void
  onSwitch: (id: string) => void
  onDelete: (id: string) => void
  onClearAll: () => void
  onOpenSettings: () => void
}

function HistoryDrawer({
  open,
  sessions,
  activeId,
  onClose,
  onNewSession,
  onSwitch,
  onDelete,
  onClearAll,
  onOpenSettings,
}: HistoryDrawerProps) {
  const sorted = sessions.slice().sort((a, b) => b.updatedAt - a.updatedAt)
  return (
    <div
      className={`history-drawer-root ${open ? 'is-open' : ''}`}
      aria-hidden={!open}
    >
      <div className="history-drawer-backdrop" onClick={onClose} />
      <aside className="history-drawer" role="dialog" aria-label="历史会话">
        <div className="history-drawer-top">
          <button
            type="button"
            className="history-drawer-new"
            onClick={onNewSession}
          >
            <EditSquareIcon className="app-icon app-icon-md" />
            <span>新建对话</span>
          </button>
        </div>

        <div className="history-drawer-list">
          {sorted.length === 0 ? (
            <div className="history-drawer-empty">还没有历史对话</div>
          ) : (
            sorted.map((s) => (
              <div
                key={s.id}
                className={[
                  'history-drawer-item',
                  s.id === activeId ? 'is-active' : '',
                ]
                  .filter(Boolean)
                  .join(' ')}
              >
                <button
                  type="button"
                  className="history-drawer-item-main"
                  onClick={() => onSwitch(s.id)}
                >
                  <span className="history-drawer-item-title">{s.title}</span>
                  <span className="history-drawer-item-time">
                    {formatRelativeTime(s.updatedAt)}
                  </span>
                </button>
                <button
                  type="button"
                  className="history-drawer-item-delete"
                  onClick={(e) => {
                    e.stopPropagation()
                    if (window.confirm(`删除「${s.title}」？此操作不可撤销。`)) {
                      onDelete(s.id)
                    }
                  }}
                  aria-label="删除"
                >
                  <TrashIcon className="app-icon app-icon-sm" />
                </button>
              </div>
            ))
          )}
        </div>

        <div className="history-drawer-bottom">
          <button
            type="button"
            className="history-drawer-bottom-item"
            onClick={onOpenSettings}
          >
            <SettingsIcon className="app-icon app-icon-md" />
            <span>设置</span>
          </button>
          <button
            type="button"
            className="history-drawer-bottom-item is-danger"
            onClick={() => {
              if (
                window.confirm(
                  '确定清空本机所有对话和媒体？此操作不可撤销，相当于该手机从未使用过本应用。',
                )
              ) {
                onClearAll()
              }
            }}
          >
            <TrashIcon className="app-icon app-icon-md" />
            <span>清空全部数据</span>
          </button>
        </div>
      </aside>
    </div>
  )
}

function App() {
  const [screen, setScreen] = useState<Screen>('turn')
  const [composeMode, setComposeMode] = useState<'voice' | 'text'>('voice')
  const [draft, setDraft] = useState('')

  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string>(() => createId('session'))
  const [historyOpen, setHistoryOpen] = useState(false)
  const [sessionsHydrated, setSessionsHydrated] = useState(false)
  const activeSessionIdRef = useRef(activeSessionId)

  const [messages, setMessages] = useState<ConversationEntry[]>([])
  const [pendingReply, setPendingReply] = useState<PendingReply | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [isPreparingRecording, setIsPreparingRecording] = useState(false)
  const [recordingWillCancel, setRecordingWillCancel] = useState(false)
  const recordingPointerStartYRef = useRef<number | null>(null)
  const recordingPointerIdRef = useRef<number | null>(null)
  const recordingWillCancelRef = useRef(false)
  const holdArmTimerRef = useRef<number | null>(null)
  const tapHintTimerRef = useRef<number | null>(null)
  const wasGeneratingAtDownRef = useRef(false)
  const [showTapHint, setShowTapHint] = useState(false)
  const [recordError, setRecordError] = useState<string | null>(null)
  const [pendingAttachments, setPendingAttachments] = useState<Attachment[]>([])
  const [attachMenuOpen, setAttachMenuOpen] = useState(false)
  const [cameraReview, setCameraReview] = useState<Attachment | null>(null)
  const [reviewDraft, setReviewDraft] = useState('')
  const cameraInputRef = useRef<HTMLInputElement | null>(null)
  const albumInputRef = useRef<HTMLInputElement | null>(null)
  const audioInputRef = useRef<HTMLInputElement | null>(null)
  const videoInputRef = useRef<HTMLInputElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
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
  const [, setLastSessionId] = useState<string | null>(null)

  const messagesRef = useRef<ConversationEntry[]>([])
  const abortRef = useRef<AbortController | null>(null)
  const streamingWsRef = useRef<WebSocket | null>(null)
  const streamingPlayerRef = useRef<StreamingPcmPlayer | null>(null)
  const streamingStopRef = useRef<(() => void) | null>(null)
  const threadEndRef = useRef<HTMLDivElement | null>(null)
  const textInputRef = useRef<HTMLTextAreaElement | null>(null)
  const textInputAutoFocusRef = useRef(false)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const audioCaptureCtxRef = useRef<AudioContext | null>(null)
  const audioCaptureSourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const audioCaptureProcessorRef = useRef<ScriptProcessorNode | null>(null)
  const audioCaptureChunksRef = useRef<Float32Array[]>([])
  const audioCaptureSampleRateRef = useRef<number>(16000)
  const recordingStartRef = useRef<number>(0)
  const recordingActionRef = useRef<'send' | 'cancel'>('send')
  const refAudioInputRef = useRef<HTMLInputElement | null>(null)

  const duplex = useDuplexSession({
    screen,
    setScreen,
    settings,
    setLastSessionId,
  })

  const threadEntries: ThreadEntry[] = pendingReply
    ? [...messages, pendingReply]
    : messages
  const activePresetMode = getPresetModeForScreen(screen)
  const activeModeSettings = settings[activePresetMode]
  const activeModePresets = presetsByMode[activePresetMode]
  const activeLengthPenalty = getLengthPenaltyForMode(settings, activePresetMode)
  const activeModeLabel = getPresetModeLabel(activePresetMode)
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
    activeSessionIdRef.current = activeSessionId
  }, [activeSessionId])

  useEffect(() => {
    let cancelled = false
    void (async () => {
      const all = await idbGetAllSessions()
      if (cancelled) return
      const sorted = all.slice().sort((a, b) => b.updatedAt - a.updatedAt)
      setSessions(sorted)

      let nextActiveId: string | null = null
      let nextMessages: ConversationEntry[] | null = null
      try {
        const saved =
          typeof localStorage !== 'undefined'
            ? localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY)
            : null
        if (saved) {
          const found = sorted.find((s) => s.id === saved)
          if (found) {
            nextActiveId = found.id
            nextMessages = found.messages
          }
        }
        if (!nextActiveId && sorted.length > 0) {
          nextActiveId = sorted[0].id
          nextMessages = sorted[0].messages
        }
      } catch {
        /* ignore */
      }
      if (nextActiveId) setActiveSessionId(nextActiveId)
      if (nextMessages) setMessages(nextMessages)
      setSessionsHydrated(true)
    })()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!sessionsHydrated) return
    if (typeof localStorage === 'undefined') return
    try {
      localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, activeSessionId)
    } catch {
      /* ignore quota */
    }
  }, [activeSessionId, sessionsHydrated])

  useEffect(() => {
    if (!sessionsHydrated) return
    const now = Date.now()
    setSessions((prev) => {
      const idx = prev.findIndex((s) => s.id === activeSessionId)
      if (messages.length === 0) {
        if (idx === -1) return prev
        const updated: ChatSession = {
          ...prev[idx],
          messages,
          updatedAt: now,
          title: '新对话',
        }
        const next = prev.slice()
        next[idx] = updated
        void idbPutSession(updated)
        return next
      }
      const title = deriveSessionTitle(messages)
      if (idx === -1) {
        const created: ChatSession = {
          id: activeSessionId,
          title,
          createdAt: now,
          updatedAt: now,
          messages,
        }
        void idbPutSession(created)
        return [created, ...prev]
      }
      const updated: ChatSession = {
        ...prev[idx],
        title,
        messages,
        updatedAt: now,
      }
      const next = prev.slice()
      next[idx] = updated
      void idbPutSession(updated)
      return next
    })
  }, [messages, activeSessionId, sessionsHydrated])

  useEffect(() => {
    threadEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [threadEntries.length])

  useEffect(() => {
    if (!attachMenuOpen) return
    // Mimic Doubao: opening the + drawer pushes the thread to the bottom
    // so the drawer feels like it's "popping up" rather than overlaying
    // somewhere mid-screen.
    const id = window.requestAnimationFrame(() => {
      threadEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
    })
    return () => window.cancelAnimationFrame(id)
  }, [attachMenuOpen])

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
    return () => {
      abortRef.current?.abort()
      try {
        audioCaptureProcessorRef.current?.disconnect()
      } catch {
        // ignore
      }
      try {
        audioCaptureSourceRef.current?.disconnect()
      } catch {
        // ignore
      }
      const ctx = audioCaptureCtxRef.current
      if (ctx && ctx.state !== 'closed') {
        void ctx.close().catch(() => {})
      }
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

    duplex.appendEntry('system', text)
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
    try {
      audioCaptureProcessorRef.current?.disconnect()
    } catch {
      // ignore
    }
    try {
      audioCaptureSourceRef.current?.disconnect()
    } catch {
      // ignore
    }
    audioCaptureProcessorRef.current = null
    audioCaptureSourceRef.current = null
    const ctx = audioCaptureCtxRef.current
    if (ctx && ctx.state !== 'closed') {
      void ctx.close().catch(() => {})
    }
    audioCaptureCtxRef.current = null
    audioCaptureChunksRef.current = []
    recordingStartRef.current = 0
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop())
    mediaStreamRef.current = null
  }

  function startNewSession() {
    const newId = createId('session')
    activeSessionIdRef.current = newId
    stopCurrentReply()
    setActiveSessionId(newId)
    setMessages([])
    setDraft('')
    setPendingReply(null)
    setIsGenerating(false)
    setRecordError(null)
    setHistoryOpen(false)
  }

  function switchToSession(id: string) {
    if (id === activeSessionId) {
      setHistoryOpen(false)
      return
    }
    const target = sessions.find((s) => s.id === id)
    if (!target) return
    activeSessionIdRef.current = id
    stopCurrentReply()
    setActiveSessionId(id)
    setMessages(rehydrateMessages(target.messages))
    setDraft('')
    setPendingReply(null)
    setIsGenerating(false)
    setRecordError(null)
    setHistoryOpen(false)
  }

  function deleteSession(id: string) {
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id)
      void idbDeleteSession(id)
      return next
    })
    if (id === activeSessionId) {
      const remaining = sessions
        .filter((s) => s.id !== id)
        .slice()
        .sort((a, b) => b.updatedAt - a.updatedAt)
      if (remaining.length > 0) {
        const top = remaining[0]
        activeSessionIdRef.current = top.id
        stopCurrentReply()
        setActiveSessionId(top.id)
        setMessages(rehydrateMessages(top.messages))
      } else {
        const newId = createId('session')
        activeSessionIdRef.current = newId
        stopCurrentReply()
        setActiveSessionId(newId)
        setMessages([])
      }
      setDraft('')
      setPendingReply(null)
      setIsGenerating(false)
      setRecordError(null)
    }
  }

  async function clearAllData() {
    const newId = createId('session')
    activeSessionIdRef.current = newId
    stopCurrentReply()
    await idbClearAll()
    if (typeof localStorage !== 'undefined') {
      try {
        localStorage.removeItem(ACTIVE_SESSION_STORAGE_KEY)
      } catch {
        /* ignore */
      }
    }
    setSessions([])
    setActiveSessionId(newId)
    setMessages([])
    setDraft('')
    setPendingReply(null)
    setIsGenerating(false)
    setRecordError(null)
    setHistoryOpen(false)
  }

  function stopCurrentReply() {
    abortRef.current?.abort()

    const stop = streamingStopRef.current
    if (stop) {
      streamingStopRef.current = null
      stop()
    }
  }

  function persistEntryToSession(
    sessionId: string,
    finalMessages: ConversationEntry[],
  ) {
    if (!sessionId) return
    const now = Date.now()
    setSessions((prev) => {
      const idx = prev.findIndex((s) => s.id === sessionId)
      const title = deriveSessionTitle(finalMessages)
      if (idx === -1) {
        const created: ChatSession = {
          id: sessionId,
          title: title || '新对话',
          createdAt: now,
          updatedAt: now,
          messages: finalMessages,
        }
        void idbPutSession(created)
        return [created, ...prev]
      }
      const updated: ChatSession = {
        ...prev[idx],
          title: title || prev[idx].title,
        messages: finalMessages,
        updatedAt: now,
      }
      const next = prev.slice()
      next[idx] = updated
      void idbPutSession(updated)
      return next
    })
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
    const submissionSessionId = activeSessionIdRef.current
    const isStillActive = () =>
      activeSessionIdRef.current === submissionSessionId

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

      if (isStillActive()) {
        setMessages([...nextMessages, assistantEntry])
        setLastSessionId(payload.recording_session_id ?? null)
      } else {
        // Reply landed after the user already switched away. Save it
        // back into the originating session marked as interrupted so
        // the partial / completed response is not lost on switch-back.
        persistEntryToSession(submissionSessionId, [
          ...nextMessages,
          { ...assistantEntry, interrupted: true },
        ])
      }
    } catch (error) {
      const errorText =
        controller.signal.aborted
          ? '已停止当前回复。'
          : `请求失败：${getErrorMessage(error)}`

      if (isStillActive() && !controller.signal.aborted) {
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
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null
      }
      if (isStillActive()) {
        setPendingReply(null)
        setIsGenerating(false)
      }
    }
  }

  async function submitConversationStreaming(
    nextMessages: ConversationEntry[],
  ) {
    const systemMessage = buildTurnSystemMessage()
    const submissionSessionId = activeSessionIdRef.current
    const isStillActive = () =>
      activeSessionIdRef.current === submissionSessionId

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
      if (isStillActive()) {
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
      }
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

      if (isStillActive()) {
        if (resolvedEntry) {
          setMessages([...nextMessages, resolvedEntry])
        } else if (errorMessage && !stoppedByUser) {
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
      } else if (resolvedEntry && resolvedEntry.kind === 'assistant') {
        // User switched away from this session before the reply finished.
        // Save what we got back into the originating session so they can
        // come back to it instead of losing the partial response.
        const interruptedEntry: ConversationEntry = {
          ...resolvedEntry,
          interrupted: true,
        }
        persistEntryToSession(submissionSessionId, [
          ...nextMessages,
          interruptedEntry,
        ])
      }

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
          if (isStillActive()) {
            setPendingReply({
              id: pendingId,
              role: 'assistant',
              kind: 'pending',
              text: fullText,
            })
          }
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
    const atts = pendingAttachments

    if ((!text && atts.length === 0) || isGenerating || isPreparingRecording) {
      return
    }

    setDraft('')
    setPendingAttachments([])
    setRecordError(null)

    const nextMessages: ConversationEntry[] = [
      ...messagesRef.current,
      {
        id: createId('user'),
        role: 'user',
        kind: 'text',
        text,
        attachments: atts.length > 0 ? atts : undefined,
      },
    ]

    setMessages(nextMessages)
    await submitConversation(nextMessages)
  }

  async function handleAttachFiles(
    files: FileList | null,
    kind: 'image' | 'audio' | 'video',
  ) {
    if (!files || files.length === 0) return
    const list = Array.from(files)
    const built: Attachment[] = []
    for (const f of list) {
      try {
        if (kind === 'image') {
          built.push(await downscaleImageToAttachment(f))
        } else {
          built.push(await mediaFileToAttachment(f, kind))
        }
      } catch (err) {
        console.warn('attach failed', f.name, err)
      }
    }
    if (built.length > 0) {
      setPendingAttachments((prev) => [...prev, ...built])
    }
  }

  async function handleAttachMixedFiles(files: FileList | null) {
    if (!files || files.length === 0) return
    const list = Array.from(files)
    const built: Attachment[] = []
    for (const f of list) {
      const t = f.type || ''
      try {
        if (t.startsWith('image/')) {
          built.push(await downscaleImageToAttachment(f))
        } else if (t.startsWith('audio/')) {
          built.push(await mediaFileToAttachment(f, 'audio'))
        } else if (t.startsWith('video/')) {
          built.push(await mediaFileToAttachment(f, 'video'))
        } else {
          console.warn('unsupported file type', f.name, t)
        }
      } catch (err) {
        console.warn('attach failed', f.name, err)
      }
    }
    if (built.length > 0) {
      setPendingAttachments((prev) => [...prev, ...built])
    }
  }

  function removePendingAttachment(id: string) {
    setPendingAttachments((prev) => {
      const target = prev.find((a) => a.id === id)
      if (target && target.kind !== 'image') {
        try {
          URL.revokeObjectURL(target.previewUrl)
        } catch {
          /* ignore */
        }
      }
      return prev.filter((a) => a.id !== id)
    })
  }

  async function handleCameraCapture(files: FileList | null) {
    if (!files || files.length === 0) return
    const f = files[0]
    if (!f) return
    try {
      const att = await downscaleImageToAttachment(f)
      setReviewDraft('')
      setCameraReview(att)
    } catch (err) {
      console.warn('camera capture failed', err)
    }
  }

  async function sendCameraReview(text: string) {
    const att = cameraReview
    if (!att || isGenerating || isPreparingRecording) return
    setCameraReview(null)
    setReviewDraft('')
    setRecordError(null)
    const trimmed = text.trim()
    const nextMessages: ConversationEntry[] = [
      ...messagesRef.current,
      {
        id: createId('user'),
        role: 'user',
        kind: 'text',
        text: trimmed,
        attachments: [att],
      },
    ]
    setMessages(nextMessages)
    await submitConversation(nextMessages)
  }

  async function regenerateLastReply() {
    if (isGenerating || isPreparingRecording) {
      return
    }

    const current = messagesRef.current
    let lastUserIndex = -1
    for (let i = current.length - 1; i >= 0; i -= 1) {
      if (current[i].role === 'user') {
        lastUserIndex = i
        break
      }
    }

    if (lastUserIndex < 0) {
      return
    }

    const trimmed = current.slice(0, lastUserIndex + 1)
    setMessages(trimmed)
    setRecordError(null)
    await submitConversation(trimmed)
  }

  async function startRecording(options?: { skipGenerationCheck?: boolean }) {
    if (
      (!options?.skipGenerationCheck && isGenerating) ||
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

    const AudioContextCtor = getAudioContextCtor()
    if (!AudioContextCtor) {
      setRecordError('当前浏览器不支持 AudioContext，无法录音。')
      return
    }

    setRecordError(null)
    setIsPreparingRecording(true)

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const audioContext = new AudioContextCtor()
      // Some browsers (iOS Safari) start the context suspended.
      if (audioContext.state === 'suspended') {
        try {
          await audioContext.resume()
        } catch {
          // ignore
        }
      }
      const source = audioContext.createMediaStreamSource(stream)
      const processor = audioContext.createScriptProcessor(4096, 1, 1)

      mediaStreamRef.current = stream
      audioCaptureCtxRef.current = audioContext
      audioCaptureSourceRef.current = source
      audioCaptureProcessorRef.current = processor
      audioCaptureChunksRef.current = []
      audioCaptureSampleRateRef.current = audioContext.sampleRate
      recordingActionRef.current = 'send'

      processor.onaudioprocess = (event: AudioProcessingEvent) => {
        const input = event.inputBuffer.getChannelData(0)
        // Copy because the underlying buffer is reused by the audio thread.
        const copy = new Float32Array(input.length)
        copy.set(input)
        audioCaptureChunksRef.current.push(copy)
      }

      source.connect(processor)
      // ScriptProcessor will not fire onaudioprocess unless connected
      // somewhere; route to destination at zero gain to keep it silent.
      const muteGain = audioContext.createGain()
      muteGain.gain.value = 0
      processor.connect(muteGain)
      muteGain.connect(audioContext.destination)

      recordingStartRef.current = performance.now()
      setIsRecording(true)
    } catch (error) {
      setRecordError(`无法开始录音：${getErrorMessage(error)}`)
      resetRecorderResources()
      setIsPreparingRecording(false)
    }
  }

  async function finalizeRecording() {
    const durationMs = Math.max(0, performance.now() - recordingStartRef.current)
    const shouldSend = recordingActionRef.current === 'send'
    const chunks = audioCaptureChunksRef.current
    const sampleRate = audioCaptureSampleRateRef.current

    resetRecorderResources()

    try {
      if (!shouldSend) {
        return
      }

      if (durationMs < 300 || chunks.length === 0) {
        setRecordError('录音太短了，请再试一次。')
        return
      }

      const merged = concatFloat32(chunks)
      if (merged.length === 0) {
        setRecordError('未采集到音频，请重试。')
        return
      }

      const resampled = resampleLinear(merged, sampleRate, 16000)
      const audioBase64 = float32ToBase64(resampled)
      const previewUrl = float32ToWavBlobUrl(resampled, 16000)
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

  function stopRecording(action: 'send' | 'cancel') {
    recordingActionRef.current = action

    const wasCapturing = audioCaptureCtxRef.current !== null

    if (wasCapturing) {
      void finalizeRecording()
    } else {
      resetRecorderResources()
      setIsPreparingRecording(false)
    }

    setIsRecording(false)
    setRecordingWillCancel(false)
    recordingWillCancelRef.current = false
    recordingPointerStartYRef.current = null
    recordingPointerIdRef.current = null
  }

  function clearHoldArmTimer() {
    if (holdArmTimerRef.current !== null) {
      window.clearTimeout(holdArmTimerRef.current)
      holdArmTimerRef.current = null
    }
  }

  function flashTapHint() {
    if (tapHintTimerRef.current !== null) {
      window.clearTimeout(tapHintTimerRef.current)
    }
    setShowTapHint(true)
    tapHintTimerRef.current = window.setTimeout(() => {
      tapHintTimerRef.current = null
      setShowTapHint(false)
    }, TAP_HINT_MS)
  }

  function handleTalkPointerDown(event: ReactPointerEvent<HTMLButtonElement>) {
    if (isPreparingRecording || isRecording) return
    recordingPointerStartYRef.current = event.clientY
    recordingPointerIdRef.current = event.pointerId
    recordingWillCancelRef.current = false
    wasGeneratingAtDownRef.current = isGenerating
    setRecordingWillCancel(false)
    setShowTapHint(false)
    try {
      event.currentTarget.setPointerCapture(event.pointerId)
    } catch {
      /* ignore */
    }
    clearHoldArmTimer()
    holdArmTimerRef.current = window.setTimeout(() => {
      holdArmTimerRef.current = null
      const wasGenerating = wasGeneratingAtDownRef.current
      if (wasGenerating) {
        stopCurrentReply()
      }
      void startRecording({ skipGenerationCheck: wasGenerating })
    }, MIN_HOLD_MS)
  }

  function handleTalkPointerMove(event: ReactPointerEvent<HTMLButtonElement>) {
    if (recordingPointerIdRef.current !== event.pointerId) return
    const startY = recordingPointerStartYRef.current
    if (startY === null) return
    const deltaY = startY - event.clientY
    const cancel = deltaY > CANCEL_DRAG_PX
    if (cancel !== recordingWillCancelRef.current) {
      recordingWillCancelRef.current = cancel
      setRecordingWillCancel(cancel)
    }
  }

  function handleTalkPointerUp(event: ReactPointerEvent<HTMLButtonElement>) {
    if (recordingPointerIdRef.current !== event.pointerId && recordingPointerIdRef.current !== null) return
    try {
      event.currentTarget.releasePointerCapture(event.pointerId)
    } catch {
      /* ignore */
    }

    const wasArming = holdArmTimerRef.current !== null
    const wasGeneratingAtDown = wasGeneratingAtDownRef.current
    clearHoldArmTimer()

    if (wasArming) {
      recordingPointerStartYRef.current = null
      recordingPointerIdRef.current = null
      if (wasGeneratingAtDown) {
        stopCurrentReply()
      } else {
        flashTapHint()
      }
      return
    }

    if (!isRecording && !isPreparingRecording) {
      recordingPointerStartYRef.current = null
      recordingPointerIdRef.current = null
      return
    }
    stopRecording(recordingWillCancelRef.current ? 'cancel' : 'send')
  }

  function handleTalkPointerCancel(event: ReactPointerEvent<HTMLButtonElement>) {
    if (recordingPointerIdRef.current !== event.pointerId && recordingPointerIdRef.current !== null) return
    clearHoldArmTimer()
    if (isRecording || isPreparingRecording) {
      stopRecording('cancel')
    } else {
      recordingPointerStartYRef.current = null
      recordingPointerIdRef.current = null
    }
  }

  function handleComposerSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void sendTextMessage()
  }

  const voiceMainLabel = isRecording
    ? '松开发送'
    : isPreparingRecording
      ? '处理中...'
      : '按住说话'

  return (
    <div className="mobile-app">
      <input
        ref={refAudioInputRef}
        className="hidden-file-input"
        type="file"
        accept="audio/*"
        onChange={handleRefAudioInputChange}
      />
      <HistoryDrawer
        open={historyOpen}
        sessions={sessions}
        activeId={activeSessionId}
        onClose={() => setHistoryOpen(false)}
        onNewSession={startNewSession}
        onSwitch={switchToSession}
        onDelete={deleteSession}
        onClearAll={() => {
          void clearAllData()
        }}
        onOpenSettings={() => {
          setHistoryOpen(false)
          setSettingsOpen(true)
        }}
      />
      {isRecording ? <RecordingOverlay willCancel={recordingWillCancel} /> : null}
      {cameraReview ? (
        <CameraReviewOverlay
          attachment={cameraReview}
          draft={reviewDraft}
          onDraftChange={setReviewDraft}
          onClose={() => {
            setCameraReview(null)
            setReviewDraft('')
          }}
          onRetake={() => {
            setCameraReview(null)
            setReviewDraft('')
            setTimeout(() => cameraInputRef.current?.click(), 0)
          }}
          onSend={(text) => {
            void sendCameraReview(text)
          }}
          disabled={isGenerating || isPreparingRecording}
        />
      ) : null}
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
          <header className="turn-topbar">
            <button
              className="topbar-icon-btn"
              type="button"
              onClick={() => setHistoryOpen(true)}
              aria-label="打开菜单"
            >
              <HamburgerIcon className="app-icon app-icon-md" />
            </button>

            <div className="topbar-title" aria-live="polite">
              <div className="topbar-title-main">
                {sessions.find((s) => s.id === activeSessionId)?.title ||
                  (messages.length > 0
                    ? deriveSessionTitle(messages)
                    : '新对话')}
              </div>
              <div className={`topbar-title-sub ${serviceState.phase}`}>
                <span className="service-tiny-dot" aria-hidden="true" />
                <span>{serviceState.summary}</span>
              </div>
            </div>

            <div className="topbar-actions">
              <button
                className="topbar-icon-btn"
                type="button"
                onClick={() => duplex.openScreen('audio')}
                disabled={isGenerating || isRecording || isPreparingRecording}
                aria-label="进入音频双工"
              >
                <PhoneIcon className="app-icon app-icon-md" />
              </button>
              <button
                className="topbar-icon-btn"
                type="button"
                onClick={() => duplex.openScreen('video')}
                disabled={isGenerating || isRecording || isPreparingRecording}
                aria-label="进入视频双工"
              >
                <VideoCallIcon className="app-icon app-icon-md" />
              </button>
            </div>
          </header>

          <div className="thread-wrap">
            <div className="thread">
              {threadEntries.map((entry, index) => {
                const isLastAssistant =
                  entry.kind === 'assistant' &&
                  entry.role === 'assistant' &&
                  index === threadEntries.length - 1
                return (
                  <MessageBubble
                    key={entry.id}
                    entry={entry}
                    isLastAssistant={isLastAssistant}
                    canRegenerate={!isGenerating && !isPreparingRecording}
                    onRegenerate={() => {
                      void regenerateLastReply()
                    }}
                  />
                )
              })}
              <div ref={threadEndRef} />
            </div>
          </div>

          <div className="composer">
            {recordError ? <div className="helper-error">{recordError}</div> : null}

            {pendingAttachments.length > 0 ? (
              <div className="attach-strip">
                {pendingAttachments.map((a) => (
                  <div
                    key={a.id}
                    className={`attach-chip attach-chip-${a.kind}`}
                    title={a.name}
                  >
                    {a.kind === 'image' ? (
                      <img src={a.previewUrl} alt={a.name} />
                    ) : a.kind === 'video' ? (
                      <FilmIcon className="app-icon app-icon-md attach-chip-icon" />
                    ) : (
                      <MusicIcon className="app-icon app-icon-md attach-chip-icon" />
                    )}
                    {a.kind !== 'image' ? (
                      <span className="attach-chip-name">{a.name}</span>
                    ) : null}
                    <button
                      type="button"
                      className="attach-chip-remove"
                      onClick={() => removePendingAttachment(a.id)}
                      aria-label="移除附件"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            ) : null}

            <div
              className={[
                'pill-bar',
                composeMode === 'voice' ? 'voice-mode' : 'text-mode',
                isRecording ? 'recording' : '',
                isGenerating ? 'generating' : '',
              ]
                .filter(Boolean)
                .join(' ')}
            >
              <input
                ref={cameraInputRef}
                type="file"
                accept="image/*"
                capture="environment"
                hidden
                onChange={(e) => {
                  void handleCameraCapture(e.target.files)
                  e.target.value = ''
                }}
              />
              <input
                ref={albumInputRef}
                type="file"
                accept="image/*"
                multiple
                hidden
                onChange={(e) => {
                  void handleAttachFiles(e.target.files, 'image')
                  e.target.value = ''
                }}
              />
              <input
                ref={audioInputRef}
                type="file"
                accept="audio/*"
                multiple
                hidden
                onChange={(e) => {
                  void handleAttachFiles(e.target.files, 'audio')
                  e.target.value = ''
                }}
              />
              <input
                ref={videoInputRef}
                type="file"
                accept="video/*"
                multiple
                hidden
                onChange={(e) => {
                  void handleAttachFiles(e.target.files, 'video')
                  e.target.value = ''
                }}
              />
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,audio/*,video/*"
                multiple
                hidden
                onChange={(e) => {
                  void handleAttachMixedFiles(e.target.files)
                  e.target.value = ''
                }}
              />

              <button
                className="pill-side"
                type="button"
                onClick={() => cameraInputRef.current?.click()}
                disabled={isGenerating || isPreparingRecording}
                aria-label="拍照"
              >
                <CameraSnapIcon className="app-icon app-icon-md" />
              </button>

              {composeMode === 'text' ? (
                <form
                  className="pill-main pill-main-text"
                  onSubmit={handleComposerSubmit}
                >
                  <textarea
                    ref={(node) => {
                      textInputRef.current = node
                      if (node && textInputAutoFocusRef.current) {
                        textInputAutoFocusRef.current = false
                        try {
                          node.focus({ preventScroll: false })
                        } catch {
                          node.focus()
                        }
                      }
                    }}
                    className="pill-input"
                    placeholder="发消息…"
                    rows={1}
                    value={draft}
                    onChange={(event) => {
                      setDraft(event.target.value)
                      autoGrowTextarea(event.target)
                    }}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' && !event.shiftKey) {
                        event.preventDefault()
                        void sendTextMessage()
                      }
                    }}
                    disabled={isGenerating || isPreparingRecording}
                  />
                  <button type="submit" className="pill-form-submit" aria-hidden="true" tabIndex={-1} />
                </form>
              ) : (
                <button
                  className="pill-main pill-talk"
                  type="button"
                  onPointerDown={handleTalkPointerDown}
                  onPointerMove={handleTalkPointerMove}
                  onPointerUp={handleTalkPointerUp}
                  onPointerCancel={handleTalkPointerCancel}
                  disabled={isPreparingRecording}
                >
                  <span className="pill-talk-label">
                    {isRecording ? '说话中…' : voiceMainLabel}
                  </span>
                  {showTapHint ? (
                    <span className="pill-talk-hint" role="status">
                      按住才能说话
                    </span>
                  ) : null}
                </button>
              )}

              <button
                className="pill-side"
                type="button"
                onClick={() => {
                  if (isGenerating) {
                    stopCurrentReply()
                    return
                  }
                  if (composeMode === 'voice') {
                    // Mark before setState so the textarea's ref callback,
                    // which fires during the commit triggered by this click,
                    // can synchronously call .focus() and pop the keyboard
                    // on iOS Safari (programmatic focus is only allowed
                    // inside the user gesture stack).
                    textInputAutoFocusRef.current = true
                    setComposeMode('text')
                  } else {
                    setComposeMode('voice')
                  }
                }}
                aria-label={composeMode === 'voice' ? '切换到键盘' : '切换到语音'}
              >
                {composeMode === 'voice' ? (
                  <KeyboardIcon className="app-icon app-icon-md" />
                ) : (
                  <WaveIcon className="app-icon app-icon-md" />
                )}
              </button>

              <button
                className={['pill-side', attachMenuOpen ? 'is-open' : ''].filter(Boolean).join(' ')}
                type="button"
                onClick={() => setAttachMenuOpen((v) => !v)}
                disabled={isGenerating || isPreparingRecording}
                aria-label={attachMenuOpen ? '关闭附件菜单' : '附件'}
                aria-expanded={attachMenuOpen}
              >
                {attachMenuOpen ? (
                  <CloseIcon className="app-icon app-icon-md" />
                ) : (
                  <PlusIcon className="app-icon app-icon-md" />
                )}
              </button>

              {isGenerating ||
              (composeMode === 'text' && draft.trim()) ||
              pendingAttachments.length > 0 ? (
                <button
                  className={['pill-send', isGenerating ? 'is-stop' : 'is-active']
                    .filter(Boolean)
                    .join(' ')}
                  type="button"
                  onClick={() => {
                    if (isGenerating) {
                      stopCurrentReply()
                      return
                    }
                    void sendTextMessage()
                  }}
                  disabled={!isGenerating && isPreparingRecording}
                  aria-label={isGenerating ? '停止' : '发送'}
                >
                  {isGenerating ? (
                    <StopIcon className="app-icon app-icon-md" />
                  ) : (
                    <SendIcon className="app-icon app-icon-md" />
                  )}
                </button>
              ) : null}
            </div>

            {attachMenuOpen ? (
              <div
                className="attach-drawer"
                role="dialog"
                aria-label="选择附件"
              >
                <button
                  type="button"
                  className="attach-drawer-item"
                  onClick={() => {
                    setAttachMenuOpen(false)
                    cameraInputRef.current?.click()
                  }}
                >
                  <span className="attach-drawer-icon attach-drawer-icon-camera">
                    <CameraSnapIcon className="app-icon app-icon-lg" />
                  </span>
                  <span className="attach-drawer-label">相机</span>
                </button>
                <button
                  type="button"
                  className="attach-drawer-item"
                  onClick={() => {
                    setAttachMenuOpen(false)
                    albumInputRef.current?.click()
                  }}
                >
                  <span className="attach-drawer-icon attach-drawer-icon-album">
                    <PhotoIcon className="app-icon app-icon-lg" />
                  </span>
                  <span className="attach-drawer-label">相册</span>
                </button>
                <button
                  type="button"
                  className="attach-drawer-item"
                  onClick={() => {
                    setAttachMenuOpen(false)
                    fileInputRef.current?.click()
                  }}
                >
                  <span className="attach-drawer-icon attach-drawer-icon-file">
                    <FileIcon className="app-icon app-icon-lg" />
                  </span>
                  <span className="attach-drawer-label">文件</span>
                </button>
                <button
                  type="button"
                  className="attach-drawer-item"
                  onClick={() => {
                    setAttachMenuOpen(false)
                    duplex.openScreen('audio')
                  }}
                  disabled={isGenerating || isRecording || isPreparingRecording}
                >
                  <span className="attach-drawer-icon attach-drawer-icon-phone">
                    <PhoneIcon className="app-icon app-icon-lg" />
                  </span>
                  <span className="attach-drawer-label">打电话</span>
                </button>
              </div>
            ) : null}

          </div>
        </div>
      ) : duplex.audioScreenOpen ? (
        <AudioDuplexScreen
          duplex={duplex}
          icons={duplexIcons}
          settingsSummary={{
            Component: SettingsSummary,
            presetName: audioPresetName,
            refAudio: settings.audio_duplex.refAudio,
            systemPrompt: settings.audio_duplex.systemPrompt,
            lengthPenalty: settings.audioDuplexLengthPenalty,
          }}
          onOpenSettings={() => {
            setSettingsOpen(true)
          }}
        />
      ) : (
        <VideoDuplexScreen
          duplex={duplex}
          icons={duplexIcons}
          settingsSummary={{
            Component: SettingsSummary,
            presetName: videoPresetName,
            refAudio: settings.omni.refAudio,
            systemPrompt: settings.omni.systemPrompt,
            lengthPenalty: settings.videoDuplexLengthPenalty,
          }}
          onOpenSettings={() => {
            setSettingsOpen(true)
          }}
        />
      )}
    </div>
  )
}

export default App
