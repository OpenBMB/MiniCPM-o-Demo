import { DuplexLogBubble } from './DuplexLogBubble'
import type {
  DuplexIcons,
  DuplexRefAudio,
  SettingsSummaryComponent,
} from './types'
import type { UseDuplexSessionApi } from './useDuplexSession'

export type VideoDuplexScreenProps = {
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
}

export function VideoDuplexScreen({
  duplex,
  icons,
  settingsSummary,
  onOpenSettings,
}: VideoDuplexScreenProps) {
  const SettingsSummary = settingsSummary.Component
  const SettingsIcon = icons.Settings
  const TranscriptIcon = icons.Transcript
  const FlipCameraIcon = icons.FlipCamera
  const MicIcon = icons.Mic
  const PauseIcon = icons.Pause
  const PlayIcon = icons.Play
  const CloseIcon = icons.Close

  return (
    <div className="duplex-screen">
      <div className="top-actions">
        <button
          className="top-action-button"
          type="button"
          onClick={onOpenSettings}
        >
          <SettingsIcon className="app-icon app-icon-md" />
          <span className="button-inline-label">设置</span>
        </button>
        <button
          className="top-action-button"
          type="button"
          onClick={duplex.toggleTextPanel}
        >
          <TranscriptIcon className="app-icon app-icon-md" />
          <span className="button-inline-label">字幕</span>
        </button>
        <button
          className="top-action-button"
          type="button"
          onClick={duplex.flipCamera}
        >
          <FlipCameraIcon className="app-icon app-icon-md" />
          <span className="button-inline-label">翻转</span>
        </button>
      </div>

      <div className="duplex-badge-row">
        <div className={`duplex-badge ${duplex.status}`}>{duplex.badgeText}</div>
        <div className="duplex-status-copy">{duplex.statusText}</div>
      </div>

      <div className="duplex-settings-wrap">
        <SettingsSummary
          modeLabel="视频双工"
          presetName={settingsSummary.presetName}
          refAudio={settingsSummary.refAudio}
          systemPrompt={settingsSummary.systemPrompt}
          lengthPenalty={settingsSummary.lengthPenalty}
          onOpen={onOpenSettings}
        />
      </div>

      <div className="duplex-stage">
        <video
          ref={duplex.videoRef}
          className={['duplex-video', duplex.videoScreenOpen ? 'visible' : 'hidden']
            .filter(Boolean)
            .join(' ')}
          autoPlay
          muted
          playsInline
        />
        <canvas ref={duplex.canvasRef} className="duplex-capture-canvas" />

        {duplex.textPanelOpen ? (
          <div className="duplex-transcript">
            {duplex.entries.length ? (
              duplex.entries.map((entry) => (
                <DuplexLogBubble key={entry.id} entry={entry} />
              ))
            ) : (
              <div className="duplex-transcript-empty">
                进入后会在这里显示系统状态、用户转写和模型回复。
              </div>
            )}
            <div ref={duplex.endRef} />
          </div>
        ) : null}
      </div>

      <div className="control-strip">
        <button
          className={['circle-btn', duplex.micEnabled ? '' : 'muted']
            .filter(Boolean)
            .join(' ')}
          type="button"
          onClick={duplex.toggleMic}
        >
          <MicIcon className="app-icon app-icon-lg" />
          <span className="circle-btn-label">
            {duplex.micEnabled ? '麦克风' : '已静音'}
          </span>
        </button>
        <button
          className={['circle-btn', duplex.pauseState === 'active' ? '' : 'muted']
            .filter(Boolean)
            .join(' ')}
          type="button"
          onClick={duplex.togglePause}
          disabled={!duplex.hasSession}
        >
          {duplex.pauseState === 'active' ? (
            <PauseIcon className="app-icon app-icon-lg" />
          ) : (
            <PlayIcon className="app-icon app-icon-lg" />
          )}
          <span className="circle-btn-label">
            {duplex.pauseState === 'active' ? '暂停' : '继续'}
          </span>
        </button>
        <button
          className="circle-btn danger"
          type="button"
          onClick={() => {
            duplex.stop()
          }}
        >
          <CloseIcon className="app-icon app-icon-lg" />
          <span className="circle-btn-label">退出</span>
        </button>
      </div>
    </div>
  )
}
