import { useEffect, useState } from 'react'
import './duplex.css'
import type { DuplexEntry, DuplexIcons } from './types'
import type { UseDuplexSessionApi } from './useDuplexSession'

const SUBTITLE_KEEP = 6

export type VideoDuplexScreenProps = {
  duplex: UseDuplexSessionApi
  icons: DuplexIcons
  onOpenSettings: () => void
}

function lampClassFor(status: UseDuplexSessionApi['status']): string {
  if (status === 'live') return 'live'
  if (status === 'starting' || status === 'queueing') return 'preparing'
  if (status === 'paused') return 'paused'
  if (status === 'error') return 'error'
  return 'stopped'
}

function lampLabelFor(
  status: UseDuplexSessionApi['status'],
  pause: UseDuplexSessionApi['pauseState'],
): string {
  if (status === 'queueing') return 'QUEUE'
  if (status === 'starting') return 'PREP'
  if (status === 'live') return 'LIVE'
  if (status === 'paused' || pause === 'paused') return 'PAUSE'
  if (status === 'error') return 'ERR'
  return 'READY'
}

function formatTimer(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function pauseLabelFor(state: UseDuplexSessionApi['pauseState']): string {
  if (state === 'pausing') return 'Pausing...'
  if (state === 'paused') return 'Resume'
  return 'Pause'
}

export function VideoDuplexScreen({
  duplex,
  icons,
  onOpenSettings,
}: VideoDuplexScreenProps) {
  void icons
  void onOpenSettings // settings entry not surfaced in faithful-omni layout

  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    if (duplex.status !== 'live') {
      if (
        duplex.status === 'idle' ||
        duplex.status === 'starting' ||
        duplex.status === 'stopped'
      ) {
        setElapsed(0)
      }
      return
    }

    const start = Date.now() - elapsed * 1000
    const id = window.setInterval(() => {
      setElapsed(Math.floor((Date.now() - start) / 1000))
    }, 1000)
    return () => {
      window.clearInterval(id)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [duplex.status])

  const aiEntries: DuplexEntry[] = duplex.entries
    .filter((entry) => entry.role === 'assistant')
    .slice(-SUBTITLE_KEEP)
  const subtitleOn = duplex.textPanelOpen
  const lampClass = lampClassFor(duplex.status)
  const lampLabel = lampLabelFor(duplex.status, duplex.pauseState)
  const showTimer = duplex.status === 'live' || duplex.status === 'paused'
  const borderActive = duplex.status === 'live'

  const startRunning = duplex.status === 'live' || duplex.status === 'paused'
  const startPreparing =
    duplex.status === 'starting' || duplex.status === 'queueing'
  const startDisabled = duplex.hasSession || startPreparing
  const startLabel = startRunning
    ? '● Live'
    : duplex.status === 'queueing'
      ? 'Queued'
      : duplex.status === 'starting'
        ? 'Preparing...'
        : 'Start'

  const pauseDisabled =
    !duplex.hasSession || duplex.pauseState === 'pausing'
  const stopDisabled = !duplex.hasSession
  const pauseLabel = pauseLabelFor(duplex.pauseState)

  const videoClass = ['vd-video', duplex.mirrorEnabled ? 'mirrored' : '']
    .filter(Boolean)
    .join(' ')

  return (
    <div className="vd-screen">
      <div className="vd-stage">
        <video
          ref={duplex.videoRef}
          className={videoClass}
          autoPlay
          muted
          playsInline
        />
        <canvas ref={duplex.canvasRef} className="vd-capture-canvas" />

        <div
          className={['vd-video-border', borderActive ? 'active' : '']
            .filter(Boolean)
            .join(' ')}
          aria-hidden="true"
        />

        <div className={['vd-status-lamp', lampClass].join(' ')}>
          <span className="vd-dot" aria-hidden="true" />
          <span className="vd-label">{lampLabel}</span>
          {showTimer ? (
            <span className="vd-timer">{formatTimer(elapsed)}</span>
          ) : null}
        </div>

        <button
          className={[
            'vd-corner-btn vd-mirror',
            duplex.mirrorEnabled ? 'active' : '',
          ]
            .filter(Boolean)
            .join(' ')}
          type="button"
          onClick={duplex.flipMirror}
          aria-label="Mirror flip"
          title="Mirror flip"
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <line x1="12" y1="3" x2="12" y2="21" strokeDasharray="2 2" />
            <polygon
              points="5,6 5,18 1,12"
              fill="currentColor"
              stroke="none"
            />
            <polygon
              points="19,6 19,18 23,12"
              fill="currentColor"
              stroke="none"
            />
            <path d="M8 6h-1a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h1" />
            <path d="M16 6h1a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1h-1" />
          </svg>
        </button>

        <button
          className="vd-corner-btn vd-cam-flip"
          type="button"
          onClick={duplex.flipCamera}
          aria-label="Flip camera"
          title="Flip camera"
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M11 19H4a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h5" />
            <path d="M13 5h7a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-5" />
            <circle cx="12" cy="12" r="3" />
            <path d="m18 22-3-3 3-3" />
            <path d="m6 2 3 3-3 3" />
          </svg>
        </button>

        <div
          className={['vd-chat-overlay', subtitleOn ? '' : 'hidden']
            .filter(Boolean)
            .join(' ')}
          aria-live="polite"
        >
          <div className="vd-chat-inner">
            {aiEntries.map((entry) => (
              <div key={entry.id} className="vd-chat-msg">
                <span className="vd-msg-icon" aria-hidden="true">
                  🤖
                </span>
                <span className="vd-msg-text">{entry.text}</span>
              </div>
            ))}
          </div>
        </div>

        <button
          className={[
            'vd-edge-btn vd-subtitle-toggle',
            subtitleOn ? 'active' : '',
          ]
            .filter(Boolean)
            .join(' ')}
          type="button"
          onClick={duplex.toggleTextPanel}
          aria-label={subtitleOn ? 'Hide subtitles' : 'Show subtitles'}
          title="Subtitles on/off"
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={2.5}
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M4 6h16" />
            <path d="M12 6v14" />
          </svg>
        </button>

        <button
          className="vd-edge-btn vd-fullscreen-exit"
          type="button"
          onClick={() => {
            duplex.stop()
          }}
          aria-label="Exit"
          title="Exit"
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M8 3H5a2 2 0 0 0-2 2v3" />
            <path d="M21 8V5a2 2 0 0 0-2-2h-3" />
            <path d="M3 16v3a2 2 0 0 0 2 2h3" />
            <path d="M16 21h3a2 2 0 0 0 2-2v-3" />
          </svg>
        </button>
      </div>

      <div className="vd-controls">
        <button
          className="vd-ctrl-btn"
          type="button"
          disabled
          title="Force Listen (not supported on mobile)"
        >
          Force Listen
        </button>
        <button
          className="vd-ctrl-btn"
          type="button"
          disabled
          title="HD (not supported on mobile)"
        >
          HD
        </button>
        <button
          className={['vd-ctrl-btn vd-start', startRunning ? 'live' : '']
            .filter(Boolean)
            .join(' ')}
          type="button"
          disabled={startDisabled}
          onClick={duplex.startSession}
          title={startLabel}
        >
          {startRunning ? null : (
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="currentColor"
              aria-hidden="true"
            >
              <polygon points="6,3 20,12 6,21" />
            </svg>
          )}
          {startLabel}
        </button>
        <button
          className="vd-ctrl-btn"
          type="button"
          disabled={pauseDisabled}
          onClick={duplex.togglePause}
          title={pauseLabel}
        >
          {pauseLabel}
        </button>
        <button
          className="vd-ctrl-btn vd-stop"
          type="button"
          disabled={stopDisabled}
          onClick={duplex.stopSession}
          title="Stop"
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="currentColor"
            aria-hidden="true"
          >
            <rect x="4" y="4" width="16" height="16" rx="2" />
          </svg>
          Stop
        </button>
      </div>
    </div>
  )
}
