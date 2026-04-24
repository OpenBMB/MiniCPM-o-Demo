import type { DuplexMode, DuplexScreenName, DuplexStatus } from './types'

export function getDuplexModeLabel(mode: DuplexMode): string {
  return mode === 'audio' ? '音频双工' : '视频双工'
}

export function getDuplexScreenName(mode: DuplexMode): DuplexScreenName {
  return mode === 'audio' ? 'audio-duplex' : 'video-duplex'
}

export function getDuplexBadgeText(
  status: DuplexStatus,
  mode: DuplexMode,
): string {
  switch (status) {
    case 'live':
      return `${getDuplexModeLabel(mode)}进行中`
    case 'queueing':
      return '排队中'
    case 'paused':
      return '已暂停'
    case 'error':
      return '连接异常'
    case 'stopped':
      return '已结束'
    default:
      return '连接中'
  }
}
