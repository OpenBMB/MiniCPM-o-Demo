import { useEffect, useRef, useState } from 'react'
import { DuplexLogBubble } from './DuplexLogBubble'
import type {
  DuplexIcons,
  DuplexRefAudio,
  SettingsSummaryComponent,
} from './types'
import type { UseDuplexSessionApi } from './useDuplexSession'
import { useI18n } from '../i18n'

export type AudioDuplexScreenProps = {
  duplex: UseDuplexSessionApi
  icons: DuplexIcons
  settingsSummary: {
    Component: SettingsSummaryComponent
    presetName: string
    refAudio: DuplexRefAudio
    systemPrompt: string
    lengthPenalty: number
  }
  onOpenSettings: () => void
  /** True once a backend session id is available to share. */
  shareReady?: boolean
  /** Open the shared App-level ShareDialog (comment + upload + clipboard). */
  onOpenShare?: () => void
}

function EarIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M6 9a6 6 0 1 1 12 0c0 2.7-1.5 4-3 5.4-.8.7-1.5 1.4-1.5 2.6 0 1.7-1.3 3-3 3-1.5 0-2.5-1-2.5-2.5" />
      <path d="M9 9a3 3 0 0 1 6 0" />
    </svg>
  )
}

function PhoneHangupIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden="true"
    >
      <path
        d="M3.6 13.6c-1-1-1-2.6 0-3.6 4.6-4.6 12.2-4.6 16.8 0 1 1 1 2.6 0 3.6l-1.4 1.4c-.6.6-1.6.6-2.2 0l-1.5-1.5c-.4-.4-.5-1-.3-1.5l.4-1c-2-1-4.4-1-6.4 0l.4 1c.2.5.1 1.1-.3 1.5l-1.5 1.5c-.6.6-1.6.6-2.2 0l-1.4-1.4Z"
        transform="rotate(135 12 12)"
      />
    </svg>
  )
}

function PhoneStartIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M6.6 10.8c1.4 2.8 3.7 5 6.5 6.5l2.2-2.2c.3-.3.7-.4 1-.3 1.2.4 2.5.6 3.8.6.6 0 1 .4 1 1V20c0 .6-.4 1-1 1A17 17 0 0 1 3 4c0-.6.4-1 1-1h3.6c.6 0 1 .4 1 1 0 1.3.2 2.6.6 3.8.1.4 0 .8-.3 1l-2.3 2Z" />
    </svg>
  )
}

function BackIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="m15 6-6 6 6 6" />
    </svg>
  )
}

function ShareIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 3v12" />
      <path d="m7 8 5-5 5 5" />
      <path d="M5 14v5a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-5" />
    </svg>
  )
}

export function AudioDuplexScreen({
  duplex,
  icons,
  settingsSummary,
  onOpenSettings,
  shareReady = false,
  onOpenShare,
}: AudioDuplexScreenProps) {
  const { t: i18n } = useI18n()
  const SettingsIcon = icons.Settings
  const TranscriptIcon = icons.Transcript
  const PauseIcon = icons.Pause
  const PlayIcon = icons.Play

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [transcriptOpen, setTranscriptOpen] = useState(false)

  const { hasSession, status, statusText, pauseState, forceListen } = duplex

  // Live waveform animation. Reads time-domain data from the analyser
  // exposed by the duplex session and paints a centered scrolling
  // bar-style waveform. Falls back to an idle pulse when there is no
  // analyser yet (session not started or already torn down).
  useEffect(() => {
    const canvas = canvasRef.current

    if (!canvas) {
      return undefined
    }

    let raf = 0
    let idleTick = 0

    function resize() {
      if (!canvas) return
      const dpr = window.devicePixelRatio || 1
      const rect = canvas.getBoundingClientRect()
      canvas.width = Math.round(rect.width * dpr)
      canvas.height = Math.round(rect.height * dpr)
    }

    function paintBars(samples: Float32Array | null) {
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      const w = canvas.width
      const h = canvas.height
      ctx.clearRect(0, 0, w, h)

      const barCount = 48
      const gap = Math.max(2, w / barCount / 4)
      const barWidth = (w - gap * (barCount + 1)) / barCount
      const centerY = h / 2

      ctx.fillStyle = forceListen
        ? 'rgba(255, 199, 90, 0.95)'
        : pauseState !== 'active'
          ? 'rgba(180, 196, 220, 0.6)'
          : 'rgba(120, 170, 255, 0.95)'

      for (let i = 0; i < barCount; i += 1) {
        let amp = 0

        if (samples && samples.length) {
          // Sample a small window and take peak so quiet speech still
          // registers.
          const start = Math.floor((i / barCount) * samples.length)
          const end = Math.floor(((i + 1) / barCount) * samples.length)
          let peak = 0
          for (let s = start; s < end; s += 1) {
            const v = Math.abs(samples[s] - 128) / 128
            if (v > peak) peak = v
          }
          amp = peak
        } else {
          // Idle pulse: gentle sine breathing so the bar looks alive
          // even before the session starts.
          const phase = (idleTick / 60) + i * 0.18
          amp = 0.06 + 0.04 * Math.sin(phase)
        }

        const barHeight = Math.max(2, amp * h * 0.9)
        const x = gap + i * (barWidth + gap)
        const y = centerY - barHeight / 2

        const radius = Math.min(barWidth / 2, 6)
        // Round-rect bar
        ctx.beginPath()
        ctx.moveTo(x + radius, y)
        ctx.lineTo(x + barWidth - radius, y)
        ctx.quadraticCurveTo(x + barWidth, y, x + barWidth, y + radius)
        ctx.lineTo(x + barWidth, y + barHeight - radius)
        ctx.quadraticCurveTo(
          x + barWidth,
          y + barHeight,
          x + barWidth - radius,
          y + barHeight,
        )
        ctx.lineTo(x + radius, y + barHeight)
        ctx.quadraticCurveTo(x, y + barHeight, x, y + barHeight - radius)
        ctx.lineTo(x, y + radius)
        ctx.quadraticCurveTo(x, y, x + radius, y)
        ctx.closePath()
        ctx.fill()
      }
    }

    function tick() {
      const analyser = duplex.getAnalyser()

      if (analyser) {
        const buffer = new Uint8Array(analyser.fftSize)
        analyser.getByteTimeDomainData(buffer)
        // Convert Uint8Array to Float32Array shape paintBars expects
        // (we keep the 0-255 range and let paintBars normalize).
        const f32 = new Float32Array(buffer.length)
        for (let i = 0; i < buffer.length; i += 1) {
          f32[i] = buffer[i]
        }
        paintBars(f32)
      } else {
        idleTick += 1
        paintBars(null)
      }

      raf = requestAnimationFrame(tick)
    }

    resize()
    window.addEventListener('resize', resize)
    raf = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [duplex, forceListen, pauseState])

  // Auto-scroll transcript to bottom when new entries arrive.
  useEffect(() => {
    duplex.endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [duplex.entries.length, transcriptOpen, duplex.endRef])

  const liveAssistantText = (() => {
    for (let i = duplex.entries.length - 1; i >= 0; i -= 1) {
      const entry = duplex.entries[i]
      if (entry.role === 'assistant') {
        return entry.text
      }
    }
    return ''
  })()

  function handleHangup() {
    if (hasSession) {
      duplex.stop()
      return
    }
    duplex.stop()
  }

  function handleStartStop() {
    if (hasSession) {
      duplex.stop({ preserveScreen: true })
      return
    }
    duplex.startCurrent()
  }

  return (
    <div className="halfduplex-screen">
      <div className="halfduplex-topbar">
        <button
          className="halfduplex-top-btn"
          type="button"
          onClick={handleHangup}
          aria-label="返回"
        >
          <BackIcon className="app-icon app-icon-md" />
        </button>
        <div className="halfduplex-top-title">
          <div className="halfduplex-top-title-main">音频通话</div>
          <div className={`halfduplex-top-title-sub status-${status}`}>
            {statusText}
          </div>
        </div>
        <div className="halfduplex-top-actions">
          <button
            className={[
              'halfduplex-top-btn',
              transcriptOpen ? 'active' : '',
            ]
              .filter(Boolean)
              .join(' ')}
            type="button"
            onClick={() => setTranscriptOpen((p) => !p)}
            aria-label="字幕"
          >
            <TranscriptIcon className="app-icon app-icon-md" />
          </button>
          {onOpenShare ? (
            <button
              className="halfduplex-top-btn"
              type="button"
              onClick={onOpenShare}
              disabled={!shareReady}
              aria-label="分享通话"
              title={shareReady ? '分享通话' : '通话开始后可分享'}
            >
              <ShareIcon className="app-icon app-icon-md" />
            </button>
          ) : null}
          <button
            className="halfduplex-top-btn"
            type="button"
            onClick={onOpenSettings}
            aria-label="设置"
          >
            <SettingsIcon className="app-icon app-icon-md" />
          </button>
        </div>
      </div>

      {/* Hidden video / canvas required by duplex session refs */}
      <video
        ref={duplex.videoRef}
        className="duplex-video hidden"
        autoPlay
        muted
        playsInline
      />
      <canvas ref={duplex.canvasRef} className="duplex-capture-canvas" />

      <div className="halfduplex-stage">
        <div
          className={[
            'halfduplex-state-pill',
            `state-${status}`,
            forceListen ? 'force-listen' : '',
          ]
            .filter(Boolean)
            .join(' ')}
        >
          {forceListen
            ? '强制收听中'
            : pauseState !== 'active'
              ? '已暂停'
              : status === 'live'
                ? hasSession
                  ? '正在通话'
                  : '准备中'
                : status === 'queueing'
                  ? '排队中'
                  : status === 'starting'
                    ? '连接中'
                    : status === 'error'
                      ? '出错'
                      : '未连接'}
        </div>

        <canvas ref={canvasRef} className="halfduplex-wave-canvas" />

        <div className="halfduplex-live-text" aria-live="polite">
          {liveAssistantText
            ? liveAssistantText
            : hasSession
              ? '说话试试，模型在听...'
              : '点击下方电话按钮开始通话'}
        </div>
      </div>

      {transcriptOpen ? (
        <div
          className="halfduplex-transcript-sheet"
          onClick={(event) => {
            if (event.target === event.currentTarget) {
              setTranscriptOpen(false)
            }
          }}
        >
          <div className="halfduplex-transcript-panel">
            <div className="halfduplex-transcript-head">
              <div className="halfduplex-transcript-title">通话字幕</div>
              <button
                className="halfduplex-transcript-close"
                type="button"
                onClick={() => setTranscriptOpen(false)}
                aria-label="关闭字幕"
              >
                ×
              </button>
            </div>
            <div className="halfduplex-transcript-body">
              {duplex.entries.length ? (
                duplex.entries.map((entry) => (
                  <DuplexLogBubble key={entry.id} entry={entry} />
                ))
              ) : (
                <div className="halfduplex-transcript-empty">
                  通话开始后这里会显示完整字幕。
                </div>
              )}
              <div ref={duplex.endRef} />
            </div>
          </div>
        </div>
      ) : null}

      <div className="halfduplex-controls">
        <button
          className={[
            'halfduplex-ctrl-btn',
            forceListen ? 'force-active' : '',
          ]
            .filter(Boolean)
            .join(' ')}
          type="button"
          onClick={() => duplex.toggleForceListen()}
          disabled={!hasSession}
        >
          <EarIcon className="app-icon app-icon-lg" />
          <span className="halfduplex-ctrl-label">
            {forceListen ? '收听中' : '强制收听'}
          </span>
        </button>

        <button
          className={[
            'halfduplex-ctrl-btn',
            pauseState !== 'active' ? 'paused' : '',
          ]
            .filter(Boolean)
            .join(' ')}
          type="button"
          onClick={() => duplex.togglePause()}
          disabled={!hasSession}
        >
          {pauseState !== 'active' ? (
            <PlayIcon className="app-icon app-icon-lg" />
          ) : (
            <PauseIcon className="app-icon app-icon-lg" />
          )}
          <span className="halfduplex-ctrl-label">
            {pauseState !== 'active' ? '继续' : '暂停'}
          </span>
        </button>

        <button
          className={[
            'halfduplex-call-btn',
            hasSession ? 'is-stop' : 'is-start',
          ]
            .filter(Boolean)
            .join(' ')}
          type="button"
          onClick={handleStartStop}
          aria-label={hasSession ? '结束' : '开始'}
        >
          {hasSession ? (
            <PhoneHangupIcon className="app-icon app-icon-xl" />
          ) : (
            <PhoneStartIcon className="app-icon app-icon-xl" />
          )}
        </button>
      </div>

      {/* settingsSummary kept in props for compatibility; not visualized
          on the call screen. */}
      <span className="visually-hidden">
        {settingsSummary.presetName}
      </span>
    </div>
  )
}
