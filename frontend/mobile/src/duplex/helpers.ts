import type { Translations } from '../i18n/types'
import type { DuplexMode, DuplexScreenName, DuplexStatus } from './types'

export function getDuplexModeLabel(mode: DuplexMode, t: Translations): string {
  return mode === 'audio' ? t.audioDuplex : t.videoDuplex
}

export function getDuplexScreenName(mode: DuplexMode): DuplexScreenName {
  return mode === 'audio' ? 'audio-duplex' : 'video-duplex'
}

export function getDuplexBadgeText(
  status: DuplexStatus,
  mode: DuplexMode,
  t: Translations,
): string {
  const label = getDuplexModeLabel(mode, t)
  switch (status) {
    case 'live':
      return t.duplexInProgress(label)
    case 'queueing':
      return t.queuing
    case 'paused':
      return t.paused
    case 'error':
      return t.error
    case 'stopped':
      return t.duplexEnded(label)
    default:
      return t.connecting
  }
}
