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

function statusClass(status: UseDuplexSessionApi['status']): string {
  if (status === 'live') return 'live'
  if (status === 'starting' || status === 'queueing') return 'preparing'
  if (status === 'paused') return 'paused'
  if (status === 'error') return 'error'
  return 'stopped'
}

function statusLabel(status: UseDuplexSessionApi['status']): string {
  if (status === 'live') return 'LIVE'
  if (status === 'starting') return 'PREP'
  if (status === 'queueing') return 'QUEUE'
  if (status === 'paused') return 'PAUSE'
  if (status === 'error') return 'ERR'
  if (status === 'stopped') return 'STOP'
  return 'IDLE'
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
  void onOpenSettings // settings sheet entry not surfaced in faithful-omni layout
  const FlipCameraIcon = icons.FlipCamera
  const TranscriptIcon = icons.Transcript
  const CloseIcon = icons.Close
  const MicIcon = icons.Mic

  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    if (duplex.status !== 'live') {
      if (duplex.status === 'idle' || duplex.status === 'starting') {
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
  }, [duplex.status])

  const recentEntries: DuplexEntry[] = duplex.entries.slice(-SUBTITLE_KEEP)
  const subtitleOn = duplex.textPanelOpen
  const lampClass = statusClass(duplex.status)
  const lampLabel = statusLabel(duplex.status)
  const showTimer = duplex.status === 'live' || duplex.status === 'paused'

  const startDisabled = duplex.hasSession || duplex.status !== 'idle'
  const pauseDisabled = !duplex.hasSession
  const stopDisabled = !duplex.hasSession
  const pauseLabel = duplex.pauseState === 'active' ? 'Pause' : 'Resume'

  return (
    <div className="vd-screen">
      <div className="vd-stage">
        <video
          ref={duplex.videoRef}
          className="vd-video"
          autoPlay
          muted
          playsInline
        />
        <canvas ref={duplex.canvasRef} className="vd-capture-canvas" />

        <div className={['vd-status-lamp', lampClass].join(' ')}>
          <span className="vd-dot" aria-hidden="true" />
          <span className="vd-label">{lampLabel}</span>
          {showTimer ? <span className="vd-timer">{formatTimer(elapsed)}</span> : null}
        </div>

        <button
          className="vd-corner-btn vd-cam-flip"
          type="button"
          onClick={duplex.flipCamera}
          aria-label="Flip camera"
          title="Flip camera"
        >
          <FlipCameraIcon className="app-icon app-icon-md" />
        </button>

        <button
          className={['vd-corner-btn vd-mic-toggle', duplex.micEnabled ? 'active' : '']
            .filter(Boolean)
            .join(' ')}
          type="button"
          onClick={duplex.toggleMic}
          aria-label={duplex.micEnabled ? 'Mute mic' : 'Unmute mic'}
          title="Mic"
        >
          <MicIcon className="app-icon app-icon-md" />
        </button>

        <div
          className={['vd-chat-overlay', subtitleOn ? '' : 'hidden']
            .filter(Boolean)
            .join(' ')}
          aria-live="polite"
        >
          <div className="vd-chat-inner">
            {recentEntries.map((entry) => (
              <div key={entry.id} className="vd-chat-msg">
                <span className="vd-msg-icon" aria-hidden="true">
                  {entry.role === 'user' ? '🙂' : entry.role === 'assistant' ? '🤖' : '·'}
                </span>
                <span className="vd-msg-text">{entry.text}</span>
              </div>
            ))}
          </div>
        </div>

        <button
          className={['vd-edge-btn vd-subtitle-toggle', subtitleOn ? 'active' : '']
            .filter(Boolean)
            .join(' ')}
          type="button"
          onClick={duplex.toggleTextPanel}
          aria-label={subtitleOn ? 'Hide subtitles' : 'Show subtitles'}
          title="Subtitles on/off"
        >
          <TranscriptIcon className="app-icon app-icon-md" />
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
          <CloseIcon className="app-icon app-icon-md" />
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
          className="vd-ctrl-btn vd-start"
          type="button"
          disabled={startDisabled}
          title="Start"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="currentColor"
            aria-hidden="true"
          >
            <polygon points="6,3 20,12 6,21" />
          </svg>
          Start
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
          onClick={() => {
            duplex.stop()
          }}
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
