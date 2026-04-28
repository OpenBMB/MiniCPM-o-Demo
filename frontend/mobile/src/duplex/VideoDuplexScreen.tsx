import { useEffect, useState } from 'react'
import './duplex.css'
import type { DuplexEntry, DuplexIcons } from './types'
import type { UseDuplexSessionApi } from './useDuplexSession'
import { useI18n, type Translations } from '../i18n'

const SUBTITLE_KEEP = 6

export type VideoDuplexScreenProps = {
  duplex: UseDuplexSessionApi
  icons: DuplexIcons
  onOpenSettings: () => void
}

/**
 * Phases mirror the desktop omni fullscreen state machine, derived from
 * (status, pauseState, hasSession). See setButtonStates / setPauseBtnState /
 * setQueueButtonStates in static/omni/omni-app.js for the source of truth.
 */
type Phase =
  | 'idle' // no session, never started (initial preview)
  | 'queuing' // _queuePhase = queuing | almost
  | 'preparing' // _queuePhase = assigned, or status=starting before queue
  | 'live' // running, pauseState = active
  | 'pausing' // pauseState = pausing
  | 'paused' // pauseState = paused
  | 'stopped' // session ended via Stop
  | 'error'

function derivePhase(api: UseDuplexSessionApi): Phase {
  if (api.status === 'error') return 'error'
  if (api.status === 'queueing') return 'queuing'
  if (api.status === 'starting') return 'preparing'
  if (api.pauseState === 'pausing') return 'pausing'
  if (api.pauseState === 'paused') return 'paused'
  if (api.status === 'live') return 'live'
  if (api.status === 'stopped') return 'stopped'
  return 'idle'
}

type LampSpec = {
  visible: boolean
  className: string // 'live' | 'preparing' | 'stopped' | 'error'
  label: string
}

function lampForPhase(phase: Phase, t: Translations): LampSpec {
  switch (phase) {
    case 'live':
      return { visible: true, className: 'live', label: t.live }
    case 'queuing':
    case 'preparing':
    case 'pausing':
    case 'paused':
      return { visible: true, className: 'preparing', label: t.preparing }
    case 'stopped':
      return { visible: true, className: 'stopped', label: t.stopped }
    case 'idle':
    case 'error':
    default:
      return { visible: false, className: 'stopped', label: '' }
  }
}

type StartSpec = { label: string; live: boolean; disabled: boolean }

function startForPhase(phase: Phase, t: Translations): StartSpec {
  switch (phase) {
    case 'idle':
    case 'stopped':
    case 'error':
      return { label: t.start, live: false, disabled: false }
    case 'queuing':
      return { label: t.queuing, live: false, disabled: true }
    case 'preparing':
      return { label: t.preparing + '...', live: false, disabled: true }
    case 'live':
    case 'pausing':
    case 'paused':
      return { label: '● ' + t.live, live: true, disabled: true }
  }
}

type PauseSpec = { label: string; disabled: boolean }

function pauseForPhase(phase: Phase, t: Translations): PauseSpec {
  switch (phase) {
    case 'live':
      return { label: t.pause, disabled: false }
    case 'pausing':
      return { label: t.pause + '...', disabled: true }
    case 'paused':
      return { label: t.continue_, disabled: false }
    default:
      return { label: t.pause, disabled: true }
  }
}

type StopSpec = { label: string; disabled: boolean; cancel: boolean }

function stopForPhase(phase: Phase, t: Translations): StopSpec {
  switch (phase) {
    case 'queuing':
      return { label: t.cancel, disabled: false, cancel: true }
    case 'live':
    case 'pausing':
    case 'paused':
      return { label: t.stop, disabled: false, cancel: false }
    default:
      return { label: t.stop, disabled: true, cancel: false }
  }
}

function ancillaryDisabledForPhase(phase: Phase): boolean {
  // Mirror desktop syncFullscreenButtons: Force Listen / HD only enabled when
  // the session is actually running (live / pausing / paused).
  return phase !== 'live' && phase !== 'pausing' && phase !== 'paused'
}

function formatTimer(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

export function VideoDuplexScreen({
  duplex,
  icons,
  onOpenSettings,
}: VideoDuplexScreenProps) {
  void icons
  void onOpenSettings // settings entry not surfaced in faithful-omni layout

  const { t: i18n } = useI18n()
  const phase = derivePhase(duplex)

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

  const lamp = lampForPhase(phase, i18n)
  const start = startForPhase(phase, i18n)
  const pause = pauseForPhase(phase, i18n)
  const stop = stopForPhase(phase, i18n)
  const ancillaryDisabled = ancillaryDisabledForPhase(phase)
  const showTimer = phase === 'live' || phase === 'paused' || phase === 'pausing'

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

        {lamp.visible ? (
          <div className={['vd-status-lamp', lamp.className].join(' ')}>
            <span className="vd-dot" aria-hidden="true" />
            <span className="vd-label">{lamp.label}</span>
            {showTimer ? (
              <span className="vd-timer">{formatTimer(elapsed)}</span>
            ) : null}
          </div>
        ) : null}

        <button
          className={[
            'vd-corner-btn vd-mirror',
            duplex.mirrorEnabled ? 'active' : '',
          ]
            .filter(Boolean)
            .join(' ')}
          type="button"
          onClick={duplex.flipMirror}
          aria-label={i18n.mirrorFlip}
          title={i18n.mirrorFlip}
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
          aria-label={i18n.flipCamera}
          title={i18n.flipCamera}
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
          aria-label={subtitleOn ? i18n.hideSubtitles : i18n.showSubtitles}
          title={i18n.subtitlesOnOff}
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
          aria-label={i18n.exit}
          title={i18n.exit}
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
          disabled={ancillaryDisabled}
          title={i18n.forceListen}
        >
          {i18n.forceListen}
        </button>
        <button
          className="vd-ctrl-btn"
          type="button"
          disabled={ancillaryDisabled}
          title={i18n.hd}
        >
          {i18n.hd}
        </button>
        <button
          className={['vd-ctrl-btn vd-start', start.live ? 'live' : '']
            .filter(Boolean)
            .join(' ')}
          type="button"
          disabled={start.disabled}
          onClick={duplex.startSession}
          title={start.label}
        >
          {start.live ? null : (
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
          {start.label}
        </button>
        <button
          className="vd-ctrl-btn"
          type="button"
          disabled={pause.disabled}
          onClick={duplex.togglePause}
          title={pause.label}
        >
          {pause.label}
        </button>
        <button
          className={['vd-ctrl-btn vd-stop', stop.cancel ? 'cancel' : '']
            .filter(Boolean)
            .join(' ')}
          type="button"
          disabled={stop.disabled}
          onClick={duplex.stopSession}
          title={stop.label}
        >
          {stop.cancel ? null : (
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="currentColor"
              aria-hidden="true"
            >
              <rect x="4" y="4" width="16" height="16" rx="2" />
            </svg>
          )}
          {stop.label}
        </button>
      </div>
    </div>
  )
}
