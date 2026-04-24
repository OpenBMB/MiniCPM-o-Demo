/**
 * Mobile <-> desktop 共用的分享协议（与 static/shared/save-share.js 对齐）
 *
 *   POST /api/sessions/{id}/upload-recording   (FormData, file)
 *   POST /api/sessions/{id}/comment            (JSON: {comment: string})
 *   GET  /api/sessions/{id}/comment            → {comment: string}
 *   localStorage["minicpmo45_recent_sessions"]: [{id, appType, savedAt}, ...]  最多 20 条
 *
 * 老 session（无 comment.txt）→ GET 返回 ""，UI 不渲染评语。前向兼容。
 */

const RECENT_SESSIONS_KEY = 'minicpmo45_recent_sessions'
const MAX_RECENT = 20

export type RecentSession = {
  id: string
  appType: string
  savedAt: string
}

export function buildShareUrl(sessionId: string): string {
  return `${window.location.origin}/s/${sessionId}`
}

export async function saveSessionComment(
  sessionId: string,
  comment: string,
): Promise<void> {
  const text = (comment || '').trim()
  const resp = await fetch(`/api/sessions/${sessionId}/comment`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ comment: text }),
  })
  if (!resp.ok) {
    const detail = await resp.text().catch(() => '')
    throw new Error(`Comment save failed: ${resp.status} ${detail}`)
  }
}

export async function fetchSessionComment(
  sessionId: string,
): Promise<string> {
  try {
    const resp = await fetch(`/api/sessions/${sessionId}/comment`)
    if (!resp.ok) return ''
    const j = (await resp.json()) as { comment?: string }
    return j.comment ?? ''
  } catch {
    return ''
  }
}

export async function uploadSessionRecording(
  sessionId: string,
  blob: Blob,
  ext: string,
): Promise<void> {
  const form = new FormData()
  form.append('file', blob, `recording.${ext || 'webm'}`)
  const resp = await fetch(`/api/sessions/${sessionId}/upload-recording`, {
    method: 'POST',
    body: form,
  })
  if (!resp.ok) {
    const detail = await resp.text().catch(() => '')
    throw new Error(`Upload failed: ${resp.status} ${detail}`)
  }
}

export function addToRecentSessions(sessionId: string, appType: string): void {
  if (!sessionId) return
  let list: RecentSession[] = []
  try {
    const raw = localStorage.getItem(RECENT_SESSIONS_KEY)
    const parsed = raw ? JSON.parse(raw) : []
    if (Array.isArray(parsed)) list = parsed
  } catch {
    list = []
  }
  const idx = list.findIndex((s) => s && s.id === sessionId)
  if (idx !== -1) list.splice(idx, 1)
  list.unshift({
    id: sessionId,
    appType: appType || 'unknown',
    savedAt: new Date().toISOString(),
  })
  if (list.length > MAX_RECENT) list.length = MAX_RECENT
  try {
    localStorage.setItem(RECENT_SESSIONS_KEY, JSON.stringify(list))
  } catch {
    /* quota — silently drop */
  }
}

export function getRecentSessions(): RecentSession[] {
  try {
    const raw = localStorage.getItem(RECENT_SESSIONS_KEY)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function clearRecentSessions(): void {
  localStorage.removeItem(RECENT_SESSIONS_KEY)
}

export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    /* fall through */
  }
  try {
    const ta = document.createElement('textarea')
    ta.value = text
    ta.style.position = 'fixed'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(ta)
    return ok
  } catch {
    return false
  }
}
