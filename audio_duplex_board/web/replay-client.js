export async function fetchDefaults() {
  // no-store: 防止浏览器缓存把 /api/defaults 卡在旧值上，
  // 也避免 self-signed cert / service worker 层的诡异缓存导致 badge 一直显示 (loading)。
  const res = await fetch('/api/defaults', { cache: 'no-store' });
  if (!res.ok) throw new Error(`defaults failed: ${res.status}`);
  return res.json();
}

export async function replayCase({ casePath }) {
  const res = await fetch('/api/replay-case', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ case_path: casePath }),
  });
  const payload = await res.json();
  if (!res.ok) {
    throw new Error(payload.detail || `replay failed: ${res.status}`);
  }
  return payload;
}
