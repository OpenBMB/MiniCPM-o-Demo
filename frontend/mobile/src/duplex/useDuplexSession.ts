import { useEffect, useRef, useState, type RefObject } from 'react'
import {
  MobileLiveMediaProvider,
  loadDuplexRuntime,
  type DuplexResultLike,
  type DuplexSessionLike,
} from '../mobile-duplex'
import {
  getDuplexBadgeText,
  getDuplexModeLabel,
  getDuplexScreenName,
} from './helpers'
import type {
  DuplexEntry,
  DuplexMode,
  DuplexPauseState,
  DuplexScreenName,
  DuplexSettings,
  DuplexStatus,
} from './types'

function createDuplexId(role: DuplexEntry['role']): string {
  return `duplex-${role}-${Math.random().toString(36).slice(2, 10)}`
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }

  return 'Unknown error'
}

export type UseDuplexSessionInput = {
  /** App-level current screen, used to derive screenOpen flags. */
  screen: string
  /** Navigate to the given screen (turn or one of the duplex screens). */
  setScreen: (next: 'turn' | DuplexScreenName) => void
  /** Settings — App.SettingsState satisfies this structurally. */
  settings: DuplexSettings
  /** Persist last duplex recording session id. */
  setLastSessionId: (id: string) => void
}

export type UseDuplexSessionApi = {
  videoRef: RefObject<HTMLVideoElement | null>
  canvasRef: RefObject<HTMLCanvasElement | null>
  endRef: RefObject<HTMLDivElement | null>
  entries: DuplexEntry[]
  status: DuplexStatus
  statusText: string
  mode: DuplexMode
  micEnabled: boolean
  mirrorEnabled: boolean
  textPanelOpen: boolean
  pauseState: DuplexPauseState
  forceListen: boolean
  badgeText: string
  audioScreenOpen: boolean
  videoScreenOpen: boolean
  hasSession: boolean
  openScreen: (mode: DuplexMode) => void
  /** Half-duplex audio screen: re-issue start() for the current mode. */
  startCurrent: () => void
  /** Video duplex (legacy React path): begin the WebSocket session. */
  startSession: () => void
  /** Video duplex (legacy React path): stop session but keep preview alive. */
  stopSession: () => void
  /** Tear down session + media and navigate back to the turn screen. */
  stop: (options?: { preserveScreen?: boolean }) => void
  toggleMic: () => void
  togglePause: () => void
  toggleForceListen: () => void
  toggleTextPanel: () => void
  flipCamera: () => void
  flipMirror: () => void
  appendEntry: (role: DuplexEntry['role'], text: string) => string
  getAnalyser: () => AnalyserNode | null
}

export function useDuplexSession(
  input: UseDuplexSessionInput,
): UseDuplexSessionApi {
  const { screen, setScreen, settings, setLastSessionId } = input

  const [entries, setEntries] = useState<DuplexEntry[]>([])
  const [status, setStatus] = useState<DuplexStatus>('idle')
  const [statusText, setStatusText] = useState('等待进入全双工')
  const [mode, setMode] = useState<DuplexMode>('audio')
  const [micEnabled, setMicEnabled] = useState(true)
  const [mirrorEnabled, setMirrorEnabled] = useState(false)
  const [textPanelOpen, setTextPanelOpen] = useState(true)
  const [pauseState, setPauseState] = useState<DuplexPauseState>('active')
  const [forceListen, setForceListen] = useState(false)
  const [hasSession, setHasSession] = useState(false)

  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)
  const sessionRef = useRef<DuplexSessionLike | null>(null)
  const mediaRef = useRef<MobileLiveMediaProvider | null>(null)
  const startInFlightRef = useRef(false)
  const listenEntryIdRef = useRef<string | null>(null)

  const audioScreenOpen = screen === 'audio-duplex'
  const videoScreenOpen = screen === 'video-duplex'
  const badgeText = getDuplexBadgeText(status, mode)

  function appendEntry(role: DuplexEntry['role'], text: string): string {
    const id = createDuplexId(role)

    setEntries((previous) => [
      ...previous,
      {
        id,
        role,
        text,
      },
    ])

    return id
  }

  function updateEntry(id: string, text: string) {
    setEntries((previous) =>
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

  function stop(options?: { preserveScreen?: boolean }) {
    const modeLabel = getDuplexModeLabel(mode)

    startInFlightRef.current = false
    listenEntryIdRef.current = null
    mediaRef.current?.stop()
    mediaRef.current = null

    if (sessionRef.current) {
      sessionRef.current.stop()
      sessionRef.current = null
    }

    setHasSession(false)
    setStatus('stopped')
    setStatusText(`${modeLabel}会话已结束`)
    setPauseState('active')
    setForceListen(false)

    if (!options?.preserveScreen) {
      setScreen('turn')
    }
  }

  async function attachPreview(nextMode: DuplexMode) {
    if (mediaRef.current || !videoRef.current || !canvasRef.current) {
      return
    }

    try {
      const media = new MobileLiveMediaProvider({
        videoEl: videoRef.current,
        canvasEl: canvasRef.current,
      })

      mediaRef.current = media
      media.setMicEnabled(true)

      if (nextMode === 'video') {
        await media.setCameraEnabled(true)
      }
    } catch (error) {
      mediaRef.current = null
      appendEntry('system', `相机预览失败：${getErrorMessage(error)}`)
    }
  }

  function openScreen(nextMode: DuplexMode) {
    setMode(nextMode)
    setEntries([])
    setMicEnabled(true)
    setPauseState('active')
    setHasSession(false)
    setScreen(getDuplexScreenName(nextMode))

    if (nextMode === 'video') {
      // Desktop omni opens the video panel into a preview state and waits for
      // the user to press Start before opening the WebSocket session.
      setStatus('idle')
      setStatusText('点击 Start 进入全双工')
      requestAnimationFrame(() => {
        void attachPreview('video')
      })
      return
    }

    // Audio mode keeps the existing auto-start behaviour because there is no
    // explicit Start button on the audio duplex screen.
    requestAnimationFrame(() => {
      void start(nextMode)
    })
  }

  function startSession() {
    if (sessionRef.current || startInFlightRef.current) {
      return
    }

    void start(mode)
  }

  function stopSession() {
    if (!sessionRef.current) {
      return
    }

    sessionRef.current.stop()

    // The session's onCleanup fires synchronously and tears down the media
    // provider. Re-attach a camera preview so the user can press Start again.
    if (mode === 'video') {
      requestAnimationFrame(() => {
        void attachPreview('video')
      })
    }
  }

  function flipMirror() {
    setMirrorEnabled((previous) => !previous)
  }

  async function start(nextMode: DuplexMode) {
    if (
      startInFlightRef.current ||
      sessionRef.current ||
      !videoRef.current ||
      !canvasRef.current
    ) {
      return
    }

    const modeLabel = getDuplexModeLabel(nextMode)
    const withVideo = nextMode === 'video'
    const duplexSettings = withVideo ? settings.omni : settings.audio_duplex

    startInFlightRef.current = true
    listenEntryIdRef.current = null
    setMode(nextMode)
    setEntries([])
    setMicEnabled(true)
    setPauseState('active')
    setStatus('starting')
    setStatusText(
      withVideo ? '正在请求摄像头和麦克风...' : '正在请求麦克风...',
    )

    try {
      const runtime = await loadDuplexRuntime()
      const media = new MobileLiveMediaProvider({
        videoEl: videoRef.current,
        canvasEl: canvasRef.current,
      })
      const session = new runtime.DuplexSession(withVideo ? 'omni' : 'adx', {
        getMaxKvTokens: () => 8192,
        getPlaybackDelayMs: () => 200,
        outputSampleRate: 24000,
      })

      mediaRef.current = media
      sessionRef.current = session
      setHasSession(true)

      media.setMicEnabled(true)
      await media.setCameraEnabled(withVideo)

      session.onSystemLog = (text) => {
        appendEntry('system', text)
      }
      session.onQueueUpdate = (data) => {
        if (data) {
          setStatus('queueing')
          setStatusText(
            `排队 ${data.position ?? '?'} / ${data.queue_length ?? '?'}，预计 ${Math.round(
              data.estimated_wait_s ?? 0,
            )}s`,
          )
        }
      }
      session.onQueueDone = () => {
        setStatus('starting')
        setStatusText(`Worker 已分配，正在准备${modeLabel}会话...`)
      }
      session.onPrepared = () => {
        setStatusText(`会话已准备，开始${modeLabel}推流...`)
      }
      session.onRunningChange = (running) => {
        setStatus(running ? 'live' : 'stopped')
        setStatusText(
          running ? `${modeLabel}进行中` : `${modeLabel}会话已停止`,
        )
      }
      session.onPauseStateChange = (state) => {
        setPauseState(state)
        setStatus(state === 'active' ? 'live' : 'paused')
        setStatusText(
          state === 'active'
            ? `${modeLabel}进行中`
            : state === 'pausing'
              ? `正在暂停${modeLabel}...`
              : `${modeLabel}已暂停`,
        )
      }
      session.onForceListenChange = (active) => {
        setForceListen(active)
      }
      session.onMetrics = (data) => {
        if (data.type === 'result') {
          setStatusText(
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

        if (!listenEntryIdRef.current) {
          listenEntryIdRef.current = appendEntry('user', listenText)
        } else {
          updateEntry(listenEntryIdRef.current, listenText)
        }
      }
      session.onSpeakStart = (text) => {
        listenEntryIdRef.current = null
        return appendEntry('assistant', text)
      }
      session.onSpeakUpdate = (handle, text) => {
        if (typeof handle === 'string') {
          updateEntry(handle, text)
        }
      }
      session.onSpeakEnd = () => {
        listenEntryIdRef.current = null
      }
      session.onCleanup = () => {
        startInFlightRef.current = false
        listenEntryIdRef.current = null
        mediaRef.current?.stop()
        mediaRef.current = null
        sessionRef.current = null
        setHasSession(false)
        setPauseState('active')
        setForceListen(false)
        setStatus('stopped')
        setStatusText(`${modeLabel}会话已结束`)
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
      mediaRef.current?.stop()
      mediaRef.current = null
      sessionRef.current = null
      setHasSession(false)
      setStatus('error')
      setStatusText(`启动失败：${getErrorMessage(error)}`)
      appendEntry('system', `启动失败：${getErrorMessage(error)}`)
    } finally {
      startInFlightRef.current = false
    }
  }

  function toggleMic() {
    setMicEnabled((previous) => !previous)
  }

  function togglePause() {
    sessionRef.current?.pauseToggle()
  }

  function toggleForceListen() {
    sessionRef.current?.toggleForceListen()
  }

  function startCurrent() {
    void start(mode)
  }

  function getAnalyser(): AnalyserNode | null {
    return mediaRef.current?.getAnalyser() ?? null
  }

  function toggleTextPanel() {
    setTextPanelOpen((previous) => !previous)
  }

  function flipCamera() {
    const action = mediaRef.current?.flipCamera()

    if (action) {
      void action.catch((error: unknown) => {
        appendEntry('system', `翻转摄像头失败：${getErrorMessage(error)}`)
      })
    }
  }

  useEffect(() => {
    mediaRef.current?.setMicEnabled(micEnabled)
  }, [micEnabled])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [entries.length])

  useEffect(() => {
    return () => {
      sessionRef.current?.stop()
      mediaRef.current?.stop()
    }
  }, [])

  return {
    videoRef,
    canvasRef,
    endRef,
    entries,
    status,
    statusText,
    mode,
    micEnabled,
    mirrorEnabled,
    textPanelOpen,
    pauseState,
    forceListen,
    badgeText,
    audioScreenOpen,
    videoScreenOpen,
    hasSession,
    openScreen,
    startCurrent,
    startSession,
    stopSession,
    stop,
    toggleMic,
    togglePause,
    toggleForceListen,
    toggleTextPanel,
    flipCamera,
    flipMirror,
    appendEntry,
    getAnalyser,
  }
}
