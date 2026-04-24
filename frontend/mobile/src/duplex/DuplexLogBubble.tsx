import type { DuplexEntry } from './types'

export function DuplexLogBubble({ entry }: { entry: DuplexEntry }) {
  return (
    <div className={['duplex-log', entry.role].join(' ')}>{entry.text}</div>
  )
}
