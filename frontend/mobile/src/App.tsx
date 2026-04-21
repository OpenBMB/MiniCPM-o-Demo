import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from 'react'
import {
  MobileLiveMediaProvider,
  loadDuplexRuntime,
  type DuplexResultLike,
  type DuplexSessionLike,
} from './mobile-duplex'
import './App.css'

type BackendMessage =
  | {
      role: 'assistant' | 'user'
      content: string
    }
  | {
      role: 'user'
      content: Array<{
        type: 'audio'
        data: string
      }>
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

function buildRequestMessages(entries: ConversationEntry[]): BackendMessage[] {
  return entries.map((entry) => {
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

function float32Base64ToWavUrl(
  base64Data: string,
  sampleRate = 24000,
): string {
  const binary = atob(base64Data)
  const raw = new Uint8Array(binary.length)

  for (let index = 0; index < binary.length; index += 1) {
    raw[index] = binary.charCodeAt(index)
  }

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
          <button
            className="voice-pill"
            type="button"
            onClick={() => {
              const audio = new Audio(entry.previewUrl)
              void audio.play()
            }}
          >
            PLAY
          </button>
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
          <button
            className="audio-replay"
            type="button"
            onClick={() => {
              const audio = new Audio(entry.audioPreviewUrl ?? undefined)
              void audio.play()
            }}
          >
            PLAY AUDIO
          </button>
        </div>
      ) : null}
    </div>
  )
}

function DuplexLogBubble({ entry }: { entry: DuplexEntry }) {
  return <div className={['duplex-log', entry.role].join(' ')}>{entry.text}</div>
}

function App() {
  const [screen, setScreen] = useState<'turn' | 'duplex'>('turn')
  const [composeMode, setComposeMode] = useState<'voice' | 'text'>('voice')
  const [draft, setDraft] = useState('')
  const [messages, setMessages] = useState<ConversationEntry[]>([])
  const [pendingReply, setPendingReply] = useState<PendingReply | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [isPreparingRecording, setIsPreparingRecording] = useState(false)
  const [recordError, setRecordError] = useState<string | null>(null)
  const [serviceState, setServiceState] = useState<ServiceState>({
    phase: 'loading',
    summary: 'Checking backend',
    detail: 'Polling /status...',
  })
  const [lastSessionId, setLastSessionId] = useState<string | null>(null)
  const [duplexEntries, setDuplexEntries] = useState<DuplexEntry[]>([])
  const [duplexStatus, setDuplexStatus] = useState<DuplexStatus>('idle')
  const [duplexStatusText, setDuplexStatusText] = useState('等待进入全双工')
  const [duplexMicEnabled, setDuplexMicEnabled] = useState(true)
  const [duplexCameraEnabled, setDuplexCameraEnabled] = useState(true)
  const [duplexTextPanelOpen, setDuplexTextPanelOpen] = useState(true)
  const [duplexPauseState, setDuplexPauseState] = useState<
    'active' | 'pausing' | 'paused'
  >('active')

  const messagesRef = useRef<ConversationEntry[]>([])
  const abortRef = useRef<AbortController | null>(null)
  const threadEndRef = useRef<HTMLDivElement | null>(null)
  const duplexEndRef = useRef<HTMLDivElement | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const recordingStartRef = useRef<number>(0)
  const recordingActionRef = useRef<'send' | 'cancel'>('send')
  const duplexVideoRef = useRef<HTMLVideoElement | null>(null)
  const duplexCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const duplexSessionRef = useRef<DuplexSessionLike | null>(null)
  const duplexMediaRef = useRef<MobileLiveMediaProvider | null>(null)
  const duplexStartInFlightRef = useRef(false)
  const duplexListenEntryIdRef = useRef<string | null>(null)

  const threadEntries: ThreadEntry[] = pendingReply
    ? [...messages, pendingReply]
    : messages

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
    duplexMediaRef.current?.setMicEnabled(duplexMicEnabled)
  }, [duplexMicEnabled])

  useEffect(() => {
    const media = duplexMediaRef.current

    if (!media) {
      return
    }

    void media.setCameraEnabled(duplexCameraEnabled).catch((error: unknown) => {
      setDuplexCameraEnabled(false)
      appendDuplexEntry('system', `切换摄像头失败：${getErrorMessage(error)}`)
    })
  }, [duplexCameraEnabled])

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

  function resetRecorderResources() {
    mediaRecorderRef.current = null
    audioChunksRef.current = []
    recordingStartRef.current = 0
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop())
    mediaStreamRef.current = null
  }

  function stopCurrentReply() {
    abortRef.current?.abort()
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
    duplexStartInFlightRef.current = false
    duplexListenEntryIdRef.current = null
    duplexMediaRef.current?.stop()
    duplexMediaRef.current = null

    if (duplexSessionRef.current) {
      duplexSessionRef.current.stop()
      duplexSessionRef.current = null
    }

    setDuplexStatus('stopped')
    setDuplexStatusText('全双工会话已结束')
    setDuplexPauseState('active')

    if (!options?.preserveScreen) {
      setScreen('turn')
    }
  }

  async function startDuplexSession() {
    if (
      duplexStartInFlightRef.current ||
      duplexSessionRef.current ||
      !duplexVideoRef.current ||
      !duplexCanvasRef.current
    ) {
      return
    }

    duplexStartInFlightRef.current = true
    duplexListenEntryIdRef.current = null
    setDuplexEntries([])
    setDuplexMicEnabled(true)
    setDuplexCameraEnabled(true)
    setDuplexPauseState('active')
    setDuplexStatus('starting')
    setDuplexStatusText('正在请求摄像头和麦克风...')

    try {
      const runtime = await loadDuplexRuntime()
      const media = new MobileLiveMediaProvider({
        videoEl: duplexVideoRef.current,
        canvasEl: duplexCanvasRef.current,
      })
      const session = new runtime.DuplexSession('omni', {
        getMaxKvTokens: () => 8192,
        getPlaybackDelayMs: () => 200,
        outputSampleRate: 24000,
      })

      duplexMediaRef.current = media
      duplexSessionRef.current = session

      media.setMicEnabled(true)
      await media.setCameraEnabled(true)

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
        setDuplexStatusText('Worker 已分配，正在准备会话...')
      }
      session.onPrepared = () => {
        setDuplexStatusText('会话已准备，开始双工推流...')
      }
      session.onRunningChange = (running) => {
        setDuplexStatus(running ? 'live' : 'stopped')
        setDuplexStatusText(running ? '双工进行中' : '全双工会话已停止')
      }
      session.onPauseStateChange = (state) => {
        setDuplexPauseState(state)
        setDuplexStatus(state === 'active' ? 'live' : 'paused')
        setDuplexStatusText(
          state === 'active'
            ? '双工进行中'
            : state === 'pausing'
              ? '正在暂停双工...'
              : '双工已暂停',
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
        setDuplexStatusText('全双工会话已结束')
      }

      await session.start(
        'You are a helpful assistant.',
        {
          config: {
            length_penalty: 1.1,
          },
          max_slice_nums: 1,
          deferred_finalize: true,
        },
        async () => {
          media.onChunk = (chunk) => {
            const message: Record<string, unknown> = {
              type: 'audio_chunk',
              audio_base64: runtime.arrayBufferToBase64(chunk.audio.buffer),
            }

            if (chunk.frameBase64) {
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

  async function submitConversation(nextMessages: ConversationEntry[]) {
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
        body: JSON.stringify({
          messages: buildRequestMessages(nextMessages),
          generation: {
            max_new_tokens: 256,
            length_penalty: 1.1,
          },
          tts: {
            enabled: false,
          },
        }),
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
          assistantAudioUrl = float32Base64ToWavUrl(
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
    ? '停'
    : composeMode === 'voice'
      ? '文'
      : '麦'

  return (
    <div className="mobile-app">
      {screen === 'turn' ? (
        <div className="turn-screen">
          <div className="turn-topbar">
            <div className={`service-chip ${serviceState.phase}`}>
              <span className="service-dot" />
              <span>{serviceState.summary}</span>
            </div>
            <button
              className="entry-phone"
              type="button"
              onClick={() => {
                setScreen('duplex')
                requestAnimationFrame(() => {
                  void startDuplexSession()
                })
              }}
              disabled={isGenerating || isRecording || isPreparingRecording}
            >
              TEL
            </button>
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
                    {isGenerating ? '停止' : '发送'}
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
                    {secondaryLabel}
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
                  <div className="talk-main">{voiceMainLabel}</div>
                  <div className="talk-sub">{voiceSubLabel}</div>
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
                  {secondaryLabel}
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
      ) : (
        <div className="duplex-screen">
          <div className="top-actions">
            <button
              className="top-action-button"
              type="button"
              onClick={() => {
                setDuplexTextPanelOpen((previous) => !previous)
              }}
            >
              文
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
              翻
            </button>
          </div>

          <div className="duplex-badge-row">
            <div className={`duplex-badge ${duplexStatus}`}>
              {duplexStatus === 'live'
                ? 'FULL DUPLEX LIVE'
                : duplexStatus === 'queueing'
                  ? 'QUEUEING'
                  : duplexStatus === 'paused'
                    ? 'PAUSED'
                    : duplexStatus === 'error'
                      ? 'ERROR'
                      : 'CONNECTING'}
            </div>
            <div className="duplex-status-copy">{duplexStatusText}</div>
          </div>

          <div className="duplex-stage">
            <video
              ref={duplexVideoRef}
              className={[
                'duplex-video',
                duplexCameraEnabled ? 'visible' : 'hidden',
              ]
                .filter(Boolean)
                .join(' ')}
              autoPlay
              muted
              playsInline
            />
            <canvas ref={duplexCanvasRef} className="duplex-capture-canvas" />

            {!duplexCameraEnabled ? (
              <div className="duplex-stage-overlay camera-off">
                摄像头已关闭，当前继续走纯音频双工。
              </div>
            ) : null}

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
              {duplexMicEnabled ? 'MIC' : 'MIC OFF'}
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
              {duplexPauseState === 'active' ? 'PAUSE' : 'RESUME'}
            </button>
            <button
              className={['circle-btn', 'white', duplexCameraEnabled ? '' : 'muted']
                .filter(Boolean)
                .join(' ')}
              type="button"
              onClick={() => {
                setDuplexCameraEnabled((previous) => !previous)
              }}
            >
              {duplexCameraEnabled ? 'CAM' : 'CAM OFF'}
            </button>
            <button
              className="circle-btn danger"
              type="button"
              onClick={() => {
                stopDuplexSession()
              }}
            >
              EXIT
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
