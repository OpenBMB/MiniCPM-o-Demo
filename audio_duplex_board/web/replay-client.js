export async function fetchDefaults() {
  const res = await fetch('/api/defaults');
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
