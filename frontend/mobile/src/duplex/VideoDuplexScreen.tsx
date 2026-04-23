import './duplex.css'
import type { DuplexEntry, DuplexIcons } from './types'
import type { UseDuplexSessionApi } from './useDuplexSession'

const SUBTITLE_KEEP = 4

export type VideoDuplexScreenProps = {
  duplex: UseDuplexSessionApi
  icons: DuplexIcons
  onOpenSettings: () => void
}

export function VideoDuplexScreen({
  duplex,
  icons,
  onOpenSettings,
}: VideoDuplexScreenProps) {
  const SettingsIcon = icons.Settings
  const TranscriptIcon = icons.Transcript
  const FlipCameraIcon = icons.FlipCamera
  const MicIcon = icons.Mic
  const PauseIcon = icons.Pause
  const PlayIcon = icons.Play
  const CloseIcon = icons.Close

  const recentEntries: DuplexEntry[] = duplex.entries.slice(-SUBTITLE_KEEP)
  const showSubtitle = duplex.textPanelOpen && recentEntries.length > 0

  return (
    <div className="vd-screen">
      <video
        ref={duplex.videoRef}
        className="vd-video"
        autoPlay
        muted
        playsInline
      />
      <canvas ref={duplex.canvasRef} className="vd-capture-canvas" />

      <div className="vd-topbar">
        <div />
        <div
          className={['vd-topbar-status', duplex.status].filter(Boolean).join(' ')}
        >
          <span className="vd-dot" aria-hidden="true" />
          <span>{duplex.badgeText}</span>
        </div>
        <div className="vd-topbar-actions">
          <button
            className="vd-topbar-btn"
            type="button"
            onClick={onOpenSettings}
            aria-label="设置"
          >
            <SettingsIcon className="app-icon app-icon-md" />
          </button>
          <button
            className={['vd-topbar-btn', duplex.textPanelOpen ? '' : 'is-off']
              .filter(Boolean)
              .join(' ')}
            type="button"
            onClick={duplex.toggleTextPanel}
            aria-label={duplex.textPanelOpen ? '关字幕' : '开字幕'}
          >
            <TranscriptIcon className="app-icon app-icon-md" />
          </button>
        </div>
      </div>

      {showSubtitle ? (
        <div className="vd-subtitle" aria-live="polite">
          {recentEntries.map((entry) => (
            <div
              key={entry.id}
              className={['vd-subtitle-msg', entry.role].join(' ')}
            >
              {entry.text}
            </div>
          ))}
        </div>
      ) : null}

      <div className="vd-controls">
        <button
          className={['vd-circle', duplex.micEnabled ? '' : 'muted']
            .filter(Boolean)
            .join(' ')}
          type="button"
          onClick={duplex.toggleMic}
          aria-label={duplex.micEnabled ? '关闭麦克风' : '打开麦克风'}
        >
          <MicIcon className="app-icon app-icon-lg" />
        </button>
        <button
          className={['vd-circle', duplex.pauseState === 'active' ? '' : 'muted']
            .filter(Boolean)
            .join(' ')}
          type="button"
          onClick={duplex.togglePause}
          disabled={!duplex.hasSession}
          aria-label={duplex.pauseState === 'active' ? '暂停' : '继续'}
        >
          {duplex.pauseState === 'active' ? (
            <PauseIcon className="app-icon app-icon-lg" />
          ) : (
            <PlayIcon className="app-icon app-icon-lg" />
          )}
        </button>
        <button
          className="vd-circle"
          type="button"
          onClick={duplex.flipCamera}
          aria-label="翻转摄像头"
        >
          <FlipCameraIcon className="app-icon app-icon-lg" />
        </button>
        <button
          className="vd-circle danger"
          type="button"
          onClick={() => {
            duplex.stop()
          }}
          aria-label="退出视频通话"
        >
          <CloseIcon className="app-icon app-icon-lg" />
        </button>
      </div>
    </div>
  )
}
