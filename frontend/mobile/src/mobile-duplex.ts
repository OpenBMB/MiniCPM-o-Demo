type DuplexQueueUpdate = {
  position?: number
  estimated_wait_s?: number
  queue_length?: number
}

type DuplexMetrics =
  | {
      type: 'state'
      sessionState?: string
    }
  | {
      type: 'result'
      latencyMs?: number
      kvCacheLength?: number
      modelState?: string
      chunksSent?: number
    }
  | {
      type: 'audio'
      ahead?: number
    }

export type DuplexResultLike = {
  is_listen: boolean
  text?: string
  audio_data?: string
  end_of_turn?: boolean
  wall_clock_ms?: number
  cost_all_ms?: number
  kv_cache_length?: number
  server_send_ts?: number
  recording_session_id?: string
}

export type DuplexSessionLike = {
  running: boolean
  forceListenActive: boolean
  recordingSessionId?: string
  onSystemLog: (text: string) => void
  onQueueUpdate: (data: DuplexQueueUpdate | null) => void
  onQueueDone: () => void
  onPrepared: () => Promise<void> | void
  onCleanup: () => void
  onMetrics: (data: DuplexMetrics) => void
  onRunningChange: (running: boolean) => void
  onPauseStateChange: (state: 'active' | 'pausing' | 'paused') => void
  onForceListenChange: (active: boolean) => void
  onListenResult: (result: DuplexResultLike) => void
  onSpeakStart: (text: string) => unknown
  onSpeakUpdate: (handle: unknown, text: string) => void
  onSpeakEnd: () => void
  start: (
    systemPrompt: string,
    preparePayload: Record<string, unknown>,
    startMediaFn?: () => Promise<void>,
  ) => Promise<void>
  pauseToggle: () => void
  toggleForceListen: () => void
  stop: () => void
  cleanup: () => void
  sendChunk: (msg: Record<string, unknown>) => void
}

type DuplexSessionConstructor = new (
  prefix: string,
  config?: {
    getMaxKvTokens?: () => number
    getPlaybackDelayMs?: () => number
    outputSampleRate?: number
  },
) => DuplexSessionLike

type DuplexRuntime = {
  DuplexSession: DuplexSessionConstructor
  arrayBufferToBase64: (buffer: ArrayBufferLike) => string
}

type MobileChunk = {
  audio: Float32Array
  frameBase64: string | null
}

let duplexRuntimePromise: Promise<DuplexRuntime> | null = null

export async function loadDuplexRuntime(): Promise<DuplexRuntime> {
  if (!duplexRuntimePromise) {
    const duplexSessionUrl = '/static/duplex/lib/duplex-session.js'
    const duplexUtilsUrl = '/static/duplex/lib/duplex-utils.js'

    duplexRuntimePromise = Promise.all([
      import(/* @vite-ignore */ duplexSessionUrl),
      import(/* @vite-ignore */ duplexUtilsUrl),
    ]).then(([sessionModule, utilsModule]) => ({
      DuplexSession: sessionModule.DuplexSession as DuplexSessionConstructor,
      arrayBufferToBase64: utilsModule.arrayBufferToBase64 as (
        buffer: ArrayBufferLike,
      ) => string,
    }))
  }

  return duplexRuntimePromise
}

type MobileLiveMediaProviderOptions = {
  videoEl: HTMLVideoElement
  canvasEl: HTMLCanvasElement
  sampleRate?: number
}

export class MobileLiveMediaProvider {
  private videoEl: HTMLVideoElement

  private canvasEl: HTMLCanvasElement

  private ctx2d: CanvasRenderingContext2D

  private readonly sampleRate: number

  private audioStream: MediaStream | null = null

  private videoStream: MediaStream | null = null

  private audioContext: AudioContext | null = null

  private audioSource: MediaStreamAudioSourceNode | null = null

  private captureNode: AudioWorkletNode | null = null

  private sinkGain: GainNode | null = null

  private analyserNode: AnalyserNode | null = null

  private usingFrontCamera = true

  private micEnabled = true

  private cameraEnabled = true

  running = false

  onChunk: ((chunk: MobileChunk) => void) | null = null

  constructor(options: MobileLiveMediaProviderOptions) {
    this.videoEl = options.videoEl
    this.canvasEl = options.canvasEl
    this.sampleRate = options.sampleRate ?? 16000

    const ctx2d = this.canvasEl.getContext('2d')

    if (!ctx2d) {
      throw new Error('Unable to initialize duplex frame canvas.')
    }

    this.ctx2d = ctx2d
  }

  /**
   * Re-attach this media provider to a new pair of DOM elements (e.g. when
   * React re-mounted the duplex screen and gave us fresh `<video>` /
   * `<canvas>` nodes). Keeps the underlying MediaStream + AudioContext so we
   * don't lose the camera permission or trigger another getUserMedia prompt.
   */
  rebindElements(elements: {
    videoEl: HTMLVideoElement
    canvasEl: HTMLCanvasElement
  }) {
    const sameVideo = this.videoEl === elements.videoEl
    const sameCanvas = this.canvasEl === elements.canvasEl
    if (sameVideo && sameCanvas) return

    if (!sameVideo) {
      // Detach from the old <video> so it doesn't keep painting a frozen frame.
      try {
        this.videoEl.pause()
      } catch {
        /* ignore — element may already be removed */
      }
      try {
        this.videoEl.srcObject = null
      } catch {
        /* ignore */
      }
      this.videoEl = elements.videoEl
      if (this.videoStream) {
        this.videoEl.srcObject = this.videoStream
        this.videoEl.style.display = 'block'
        this.videoEl.style.transform = this.usingFrontCamera
          ? 'scaleX(-1)'
          : 'none'
        void this.videoEl.play().catch(() => {
          /* iOS / autoplay restrictions: keep stream attached */
        })
      } else {
        this.videoEl.style.display = 'none'
      }
    }

    if (!sameCanvas) {
      this.canvasEl = elements.canvasEl
      const ctx2d = this.canvasEl.getContext('2d')
      if (!ctx2d) {
        throw new Error('Unable to initialize duplex frame canvas.')
      }
      this.ctx2d = ctx2d
    }
  }

  setMicEnabled(enabled: boolean) {
    this.micEnabled = enabled
  }

  getAnalyser(): AnalyserNode | null {
    return this.analyserNode
  }

  async setCameraEnabled(enabled: boolean) {
    this.cameraEnabled = enabled

    if (!enabled) {
      this.stopVideoStream()
      return
    }

    // Open the camera as soon as it's enabled so a preview can be shown
    // before `start()` (which only acquires the microphone) is invoked.
    if (!this.videoStream) {
      await this.openVideoStream(this.usingFrontCamera)
    }
  }

  async flipCamera() {
    this.usingFrontCamera = !this.usingFrontCamera

    if (!this.cameraEnabled) {
      return
    }

    this.stopVideoStream()
    await this.openVideoStream(this.usingFrontCamera)
  }

  async start() {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('浏览器不支持 getUserMedia，请使用 HTTPS 打开。')
    }

    if (this.cameraEnabled && !this.videoStream) {
      await this.openVideoStream(this.usingFrontCamera)
    }

    this.audioStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    })

    const AudioContextCtor =
      window.AudioContext ??
      (
        window as Window & {
          webkitAudioContext?: typeof AudioContext
        }
      ).webkitAudioContext

    if (!AudioContextCtor) {
      throw new Error('浏览器不支持 AudioContext。')
    }

    this.audioContext = new AudioContextCtor({
      sampleRate: this.sampleRate,
    })

    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume()
    }

    await this.audioContext.audioWorklet.addModule(
      '/static/duplex/lib/capture-processor.js',
    )

    this.audioSource =
      this.audioContext.createMediaStreamSource(this.audioStream)
    this.captureNode = new AudioWorkletNode(
      this.audioContext,
      'capture-processor',
      {
        processorOptions: {
          chunkSize: this.sampleRate,
        },
      },
    )
    this.sinkGain = this.audioContext.createGain()
    this.sinkGain.gain.value = 0
    this.analyserNode = this.audioContext.createAnalyser()
    this.analyserNode.fftSize = 1024
    this.analyserNode.smoothingTimeConstant = 0.6

    this.audioSource.connect(this.analyserNode)
    this.audioSource.connect(this.captureNode)
    this.captureNode.connect(this.sinkGain)
    this.sinkGain.connect(this.audioContext.destination)

    this.captureNode.port.onmessage = (event: MessageEvent<{ type: string; audio: Float32Array }>) => {
      if (!this.running || event.data.type !== 'chunk') {
        return
      }

      const sourceAudio = event.data.audio
      const audio = this.micEnabled
        ? sourceAudio
        : new Float32Array(sourceAudio.length)
      const frameBase64 = this.captureFrame()

      this.onChunk?.({
        audio,
        frameBase64,
      })
    }

    this.captureNode.port.postMessage({ command: 'start' })
    this.running = true
  }

  stop() {
    this.running = false

    if (this.captureNode) {
      this.captureNode.port.postMessage({ command: 'stop' })
      this.captureNode.disconnect()
      this.captureNode = null
    }

    if (this.audioSource) {
      this.audioSource.disconnect()
      this.audioSource = null
    }

    if (this.sinkGain) {
      this.sinkGain.disconnect()
      this.sinkGain = null
    }

    if (this.analyserNode) {
      try {
        this.analyserNode.disconnect()
      } catch {
        // analyser may already be disconnected if context closed first
      }
      this.analyserNode = null
    }

    if (this.audioContext) {
      void this.audioContext.close()
      this.audioContext = null
    }

    if (this.audioStream) {
      this.audioStream.getTracks().forEach((track) => track.stop())
      this.audioStream = null
    }

    this.stopVideoStream()
    this.ctx2d.clearRect(0, 0, this.canvasEl.width, this.canvasEl.height)
  }

  private async openVideoStream(frontCamera: boolean) {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('浏览器不支持摄像头访问。')
    }

    const facingMode = frontCamera ? 'user' : 'environment'

    this.videoStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode,
      },
    })

    this.videoEl.srcObject = this.videoStream
    this.videoEl.style.display = 'block'
    this.videoEl.style.transform = frontCamera ? 'scaleX(-1)' : 'none'

    try {
      await this.videoEl.play()
    } catch {
      // iOS / autoplay restrictions: keep the stream attached and wait for user gesture.
    }
  }

  private stopVideoStream() {
    if (this.videoStream) {
      this.videoStream.getTracks().forEach((track) => track.stop())
      this.videoStream = null
    }

    this.videoEl.pause()
    this.videoEl.srcObject = null
    this.videoEl.style.display = 'none'
    this.videoEl.style.transform = 'none'
  }

  private captureFrame(): string | null {
    if (!this.cameraEnabled) {
      return null
    }

    if (!this.videoEl.videoWidth || !this.videoEl.videoHeight) {
      return null
    }

    const width = this.videoEl.videoWidth
    const height = this.videoEl.videoHeight

    this.canvasEl.width = width
    this.canvasEl.height = height
    this.ctx2d.drawImage(this.videoEl, 0, 0, width, height)

    return this.canvasEl.toDataURL('image/jpeg', 0.7).split(',')[1] ?? null
  }
}
