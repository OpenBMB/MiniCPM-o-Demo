import { useEffect, useRef, useState, type ChangeEvent } from 'react'
import type {
  RefAudioState,
  PresetMetadata,
  UserPreset,
  OmniBridge,
} from './settings-types'
import { EMPTY_REF_AUDIO } from './settings-types'
import {
  createId,
  cloneRefAudio,
  extractRefAudioFromPreset,
  extractPromptFromPreset,
  getAudioContextCtor,
  float32ToBase64,
  concatFloat32,
  resampleLinear,
  convertAudioBlobToFloat32Base64,
} from './settings-utils'

const PCM_WORKLET_URL = '/static/duplex/lib/pcm-capture-turnbased.js'
const IDB_NAME = 'omni-settings-db'
const IDB_VERSION = 1
const IDB_STORE = 'user-presets'

function openDb(): Promise<IDBDatabase> {
  if (typeof indexedDB === 'undefined') return Promise.reject(new Error('no IDB'))
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, IDB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE, { keyPath: 'id' })
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

async function idbGetAll(): Promise<UserPreset[]> {
  try {
    const db = await openDb()
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readonly')
      const req = tx.objectStore(IDB_STORE).getAll()
      req.onsuccess = () => resolve(
        ((req.result as UserPreset[]) || [])
          .filter((p) => p && typeof p.id === 'string')
          .sort((a, b) => a.createdAt - b.createdAt),
      )
      req.onerror = () => reject(req.error)
    })
  } catch { return [] }
}

async function idbPut(preset: UserPreset): Promise<void> {
  try {
    const db = await openDb()
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readwrite')
      tx.objectStore(IDB_STORE).put(preset)
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  } catch { /* */ }
}

async function idbDel(id: string): Promise<void> {
  try {
    const db = await openDb()
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readwrite')
      tx.objectStore(IDB_STORE).delete(id)
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
  } catch { /* */ }
}

function audioBase64ToBlobUrl(base64Data: string, sampleRate = 16000): string {
  const binary = atob(base64Data)
  const raw = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) raw[i] = binary.charCodeAt(i)

  if (
    raw.length >= 44 &&
    raw[0] === 0x52 && raw[1] === 0x49 && raw[2] === 0x46 && raw[3] === 0x46 &&
    raw[8] === 0x57 && raw[9] === 0x41 && raw[10] === 0x56 && raw[11] === 0x45
  ) {
    const buf = new ArrayBuffer(raw.byteLength)
    new Uint8Array(buf).set(raw)
    return URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }))
  }

  const float32 = new Float32Array(raw.buffer)
  const wavBuf = new ArrayBuffer(44 + float32.length * 2)
  const dv = new DataView(wavBuf)
  const ws = (o: number, s: string) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)) }
  ws(0, 'RIFF'); dv.setUint32(4, 36 + float32.length * 2, true); ws(8, 'WAVE')
  ws(12, 'fmt '); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true)
  dv.setUint32(24, sampleRate, true); dv.setUint32(28, sampleRate * 2, true)
  dv.setUint16(32, 2, true); dv.setUint16(34, 16, true)
  ws(36, 'data'); dv.setUint32(40, float32.length * 2, true)
  let offset = 44
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i] ?? 0))
    dv.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    offset += 2
  }
  return URL.createObjectURL(new Blob([wavBuf], { type: 'audio/wav' }))
}

function playPcmBase64(b64: string, rate = 16000) {
  const url = audioBase64ToBlobUrl(b64, rate)
  const audio = new Audio(url)
  const cleanup = () => URL.revokeObjectURL(url)
  audio.onended = cleanup
  audio.onerror = cleanup
  void audio.play().catch(cleanup)
}

const ICON_CLOSE = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`

export type OmniSettingsWidgetProps = {
  open: boolean
  bridge: OmniBridge
  onClose: () => void
}

export function OmniSettingsWidget({ open, bridge, onClose }: OmniSettingsWidgetProps) {
  const [systemPrompt, setSystemPrompt] = useState('')
  const [lengthPenalty, setLengthPenalty] = useState(1.1)
  const [playbackDelay, setPlaybackDelay] = useState(0)
  const [maxKv, setMaxKv] = useState(8192)

  const [presets, setPresets] = useState<PresetMetadata[]>([])
  const [userPresets, setUserPresets] = useState<UserPreset[]>([])
  const [activePresetId, setActivePresetId] = useState<string | null>(null)

  const [refAudio, setRefAudio] = useState<RefAudioState>(cloneRefAudio(EMPTY_REF_AUDIO))
  const [defaultRefAudio, setDefaultRefAudio] = useState<RefAudioState | null>(null)
  const [recording, setRecording] = useState(false)
  const [toast, setToast] = useState('')

  const fileInputRef = useRef<HTMLInputElement>(null)

  const recStreamRef = useRef<MediaStream | null>(null)
  const recCtxRef = useRef<AudioContext | null>(null)
  const recWorkletRef = useRef<AudioWorkletNode | null>(null)
  const recProcessorRef = useRef<ScriptProcessorNode | null>(null)
  const recChunksRef = useRef<Float32Array[]>([])
  const recSampleRateRef = useRef(48000)
  const recStartRef = useRef(0)

  function showToast(msg: string) {
    setToast(msg)
    setTimeout(() => setToast(''), 2500)
  }

  useEffect(() => {
    if (!open) return
    setSystemPrompt(bridge.getSystemPrompt())
    setLengthPenalty(bridge.getLengthPenalty())
    setPlaybackDelay(bridge.getPlaybackDelay())
    setMaxKv(bridge.getMaxKv())

    const existingB64 = bridge.getRefAudioBase64()
    if (existingB64) {
      setRefAudio((prev) => (prev.base64 === existingB64 ? prev : { ...prev, base64: existingB64, source: prev.source === 'none' ? 'default' : prev.source }))
    }

    void loadPresets()
    void idbGetAll().then(setUserPresets)
  }, [open]) // eslint-disable-line react-hooks/exhaustive-deps

  async function loadPresets() {
    try {
      const [presetsRes, defaultRefRes] = await Promise.all([
        fetch('/api/presets'),
        fetch('/api/default_ref_audio'),
      ])

      if (presetsRes.ok) {
        const data = (await presetsRes.json()) as Partial<Record<string, PresetMetadata[]>>
        const omniPresets = data.omni ?? []

        if (omniPresets.length > 0) {
          const first = omniPresets[0]!
          const needsHydration = Boolean(first.ref_audio?.path && !first.ref_audio.data)
          if (needsHydration) {
            try {
              const res = await fetch(`/api/presets/omni/${first.id}/audio`)
              if (res.ok) {
                const payload = await res.json() as { ref_audio?: { data?: string; name?: string; duration?: number } }
                if (payload.ref_audio?.data && first.ref_audio) {
                  first.ref_audio = { ...first.ref_audio, ...payload.ref_audio }
                }
              }
            } catch { /* non-fatal */ }
          }
        }

        setPresets(omniPresets)
      }

      if (defaultRefRes.ok) {
        const payload = await defaultRefRes.json() as { name?: string; duration?: number; base64?: string | null }
        if (payload.base64) {
          const dra: RefAudioState = {
            source: 'default',
            name: payload.name || '默认参考音频',
            duration: payload.duration || 0,
            base64: payload.base64,
          }
          setDefaultRefAudio(dra)
          setRefAudio((prev) => {
            if (prev.source === 'none' || !prev.base64) return dra
            return prev
          })
        }
      }
    } catch (err) {
      console.warn('loadPresets failed', err)
    }
  }

  function syncToBridge(patch: {
    prompt?: string
    lp?: number
    delay?: number
    kv?: number
    ref?: RefAudioState
  }) {
    if (patch.prompt !== undefined) bridge.setSystemPrompt(patch.prompt)
    if (patch.lp !== undefined) bridge.setLengthPenalty(patch.lp)
    if (patch.delay !== undefined) bridge.setPlaybackDelay(patch.delay)
    if (patch.kv !== undefined) bridge.setMaxKv(patch.kv)
    if (patch.ref !== undefined) {
      bridge.setRefAudioBase64(patch.ref.base64, patch.ref.name, patch.ref.duration)
    }
  }

  function selectPreset(preset: PresetMetadata) {
    const prompt = extractPromptFromPreset(preset)
    const ra = extractRefAudioFromPreset(preset)
    setSystemPrompt(prompt)
    setRefAudio(ra.base64 ? ra : refAudio.base64 ? refAudio : cloneRefAudio(EMPTY_REF_AUDIO))
    setActivePresetId(preset.id)
    syncToBridge({ prompt, ref: ra.base64 ? ra : undefined })
  }

  function selectUserPreset(preset: UserPreset) {
    setSystemPrompt(preset.systemPrompt)
    setRefAudio(preset.refAudio.base64 ? cloneRefAudio(preset.refAudio) : cloneRefAudio(EMPTY_REF_AUDIO))
    setActivePresetId(`user:${preset.id}`)
    syncToBridge({
      prompt: preset.systemPrompt,
      ref: preset.refAudio.base64 ? preset.refAudio : undefined,
    })
  }

  function handleSaveAsPreset() {
    const name = window.prompt('为这个预设起个名字', '')?.trim()
    if (!name) return
    const now = Date.now()
    const p: UserPreset = {
      id: createId('upreset'),
      name,
      systemPrompt,
      refAudio: cloneRefAudio(refAudio),
      createdAt: now,
      updatedAt: now,
    }
    setUserPresets((prev) => [...prev, p])
    setActivePresetId(`user:${p.id}`)
    void idbPut(p)
    showToast(`已保存预设"${name}"`)
  }

  function handleDeleteUserPreset(presetId: string) {
    setUserPresets((prev) => prev.filter((p) => p.id !== presetId))
    void idbDel(presetId)
    if (activePresetId === `user:${presetId}`) setActivePresetId(null)
  }

  function handleUseDefault() {
    if (!defaultRefAudio?.base64) {
      showToast('没有可用的默认参考音频。')
      return
    }
    setRefAudio(defaultRefAudio)
    setActivePresetId(null)
    syncToBridge({ ref: defaultRefAudio })
  }

  function handleClear() {
    const empty = cloneRefAudio(EMPTY_REF_AUDIO)
    setRefAudio(empty)
    setActivePresetId(null)
    syncToBridge({ ref: empty })
  }

  function handlePlay() {
    if (!refAudio.base64) {
      showToast('没有可播放的参考音频。')
      return
    }
    playPcmBase64(refAudio.base64, 16000)
  }

  function handleUploadClick() {
    fileInputRef.current?.click()
  }

  async function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      const base64 = await convertAudioBlobToFloat32Base64(file)
      const tmp = new Audio(URL.createObjectURL(file))
      tmp.onloadedmetadata = () => {
        const dur = Number.isFinite(tmp.duration) ? tmp.duration : 0
        const ra: RefAudioState = { source: 'upload', name: file.name, duration: dur, base64 }
        setRefAudio(ra)
        setActivePresetId(null)
        syncToBridge({ ref: ra })
        URL.revokeObjectURL(tmp.src)
      }
      tmp.onerror = () => {
        const ra: RefAudioState = { source: 'upload', name: file.name, duration: 0, base64 }
        setRefAudio(ra)
        setActivePresetId(null)
        syncToBridge({ ref: ra })
        URL.revokeObjectURL(tmp.src)
      }
    } catch (err) {
      showToast(`处理音频失败：${err instanceof Error ? err.message : String(err)}`)
    } finally {
      e.target.value = ''
    }
  }

  async function handleToggleRecord() {
    if (recording) {
      setRecording(false)
      const chunks = recChunksRef.current
      const sr = recSampleRateRef.current
      const ms = performance.now() - recStartRef.current

      try { recWorkletRef.current?.port.postMessage({ type: 'capture', value: false }) } catch { /* */ }
      try { recWorkletRef.current?.disconnect() } catch { /* */ }
      try { recProcessorRef.current?.disconnect() } catch { /* */ }
      recWorkletRef.current = null
      recProcessorRef.current = null
      const ctx = recCtxRef.current
      if (ctx && ctx.state !== 'closed') void ctx.close().catch(() => {})
      recCtxRef.current = null
      recStreamRef.current?.getTracks().forEach((t) => t.stop())
      recStreamRef.current = null

      if (chunks.length === 0) { showToast('录制太短，请重试。'); return }
      const merged = concatFloat32(chunks)
      if (merged.length === 0) { showToast('录制失败。'); return }
      const resampled = resampleLinear(merged, sr, 16000)
      const base64 = float32ToBase64(resampled)
      const dur = Math.round((ms / 1000) * 10) / 10
      const ra: RefAudioState = { source: 'upload', name: `录制 ${new Date().toLocaleTimeString()}`, duration: dur, base64 }
      setRefAudio(ra)
      setActivePresetId(null)
      syncToBridge({ ref: ra })
      showToast(`已录制 ${dur.toFixed(1)}s 参考音频`)
      return
    }

    if (!navigator.mediaDevices?.getUserMedia) { showToast('不支持麦克风。'); return }
    const Ctor = getAudioContextCtor()
    if (!Ctor) { showToast('不支持录音。'); return }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const ctx = new Ctor()
      if (ctx.state === 'suspended') try { await ctx.resume() } catch { /* */ }
      const source = ctx.createMediaStreamSource(stream)
      recChunksRef.current = []
      recSampleRateRef.current = ctx.sampleRate

      const useWorklet = typeof AudioWorkletNode !== 'undefined' && typeof ctx.audioWorklet?.addModule === 'function'
      if (useWorklet) {
        try {
          await ctx.audioWorklet.addModule(PCM_WORKLET_URL)
        } catch {
          setupScriptProcessor(stream, ctx, source)
          return
        }
        const node = new AudioWorkletNode(ctx, 'pcm-capture-turnbased')
        node.port.onmessage = (ev: MessageEvent) => {
          const d = ev.data as { type: string; samples: Float32Array } | undefined
          if (d?.type === 'pcm') recChunksRef.current.push(d.samples)
        }
        source.connect(node)
        node.port.postMessage({ type: 'capture', value: true })
        recWorkletRef.current = node
      } else {
        setupScriptProcessor(stream, ctx, source)
        return
      }

      recStreamRef.current = stream
      recCtxRef.current = ctx
      recStartRef.current = performance.now()
      setRecording(true)
    } catch (err) {
      showToast(`无法录制：${err instanceof Error ? err.message : String(err)}`)
    }
  }

  function setupScriptProcessor(stream: MediaStream, ctx: AudioContext, source: MediaStreamAudioSourceNode) {
    const processor = ctx.createScriptProcessor(4096, 1, 1)
    const mute = ctx.createGain()
    mute.gain.value = 0
    processor.onaudioprocess = (ev: AudioProcessingEvent) => {
      const input = ev.inputBuffer.getChannelData(0)
      const copy = new Float32Array(input.length)
      copy.set(input)
      recChunksRef.current.push(copy)
    }
    source.connect(processor)
    processor.connect(mute)
    mute.connect(ctx.destination)
    recProcessorRef.current = processor
    recStreamRef.current = stream
    recCtxRef.current = ctx
    recStartRef.current = performance.now()
    setRecording(true)
  }

  function handlePromptChange(value: string) {
    setSystemPrompt(value)
    setActivePresetId(null)
    syncToBridge({ prompt: value })
  }

  function handleLpChange(value: number) {
    const v = Number.isFinite(value) ? value : 1.1
    setLengthPenalty(v)
    syncToBridge({ lp: v })
  }

  function handleDelayChange(value: number) {
    const v = Number.isFinite(value) ? value : 0
    setPlaybackDelay(v)
    syncToBridge({ delay: v })
  }

  function handleKvChange(value: number) {
    const v = Number.isFinite(value) ? value : 8192
    setMaxKv(v)
    syncToBridge({ kv: v })
  }

  if (!open) return null

  return (
    <div className="settings-sheet-backdrop" onClick={onClose}>
      <div className="settings-sheet" onClick={(e) => e.stopPropagation()}>
        <div className="settings-sheet-head">
          <div>
            <div className="settings-sheet-title">设置</div>
            <div className="settings-sheet-subtitle">视频双工</div>
          </div>
          <button
            className="settings-close-button"
            type="button"
            onClick={onClose}
            dangerouslySetInnerHTML={{ __html: ICON_CLOSE }}
          />
        </div>

        <div className="settings-section">
          <div className="settings-section-title">Preset</div>
          <div className="preset-chip-row">
            {presets.map((p) => (
              <button
                key={p.id}
                className={['preset-chip', activePresetId === p.id ? 'active' : ''].filter(Boolean).join(' ')}
                type="button"
                onClick={() => selectPreset(p)}
              >
                {p.name}
              </button>
            ))}
            {userPresets.map((p) => (
              <button
                key={`u-${p.id}`}
                className={['preset-chip user-preset-chip', activePresetId === `user:${p.id}` ? 'active' : ''].filter(Boolean).join(' ')}
                type="button"
                onClick={() => selectUserPreset(p)}
                onContextMenu={(e) => {
                  e.preventDefault()
                  if (window.confirm(`删除预设"${p.name}"？`)) handleDeleteUserPreset(p.id)
                }}
              >
                {p.name}
              </button>
            ))}
            <button className="save-preset-chip" type="button" onClick={handleSaveAsPreset}>
              + 保存当前
            </button>
          </div>
          {presets.length === 0 && userPresets.length === 0 && (
            <div className="settings-empty-copy" style={{ marginTop: 4 }}>
              暂无预设
            </div>
          )}
        </div>

        <div className="settings-section">
          <div className="settings-section-title">参考音频</div>
          <div className="ref-audio-card">
            <div className="ref-audio-title">{refAudio.base64 ? refAudio.name : '未设置参考音频'}</div>
            <div className="ref-audio-meta">
              来源：{refAudio.source}
              {refAudio.duration ? ` · ${refAudio.duration.toFixed(1)}s` : ''}
            </div>
            <div className="ref-audio-actions">
              <button className="secondary-btn compact" type="button" onClick={handleUseDefault} disabled={!defaultRefAudio?.base64}>默认</button>
              <button className="secondary-btn compact" type="button" onClick={handleUploadClick}>上传</button>
              <button className={`secondary-btn compact${recording ? ' ref-audio-recording-active' : ''}`} type="button" onClick={handleToggleRecord}>
                {recording ? <><span className="rec-dot" />停止录制</> : '录制'}
              </button>
              <button className="secondary-btn compact" type="button" onClick={handlePlay} disabled={!refAudio.base64}>播放</button>
              <button className="secondary-btn compact" type="button" onClick={handleClear}>清空</button>
            </div>
          </div>
          <input ref={fileInputRef} type="file" accept="audio/*" hidden onChange={handleFileChange} />
        </div>

        <div className="settings-section">
          <label className="settings-section-title" htmlFor="omni-settings-prompt">System Prompt</label>
          <textarea
            id="omni-settings-prompt"
            className="settings-textarea"
            value={systemPrompt}
            onChange={(e) => handlePromptChange(e.target.value)}
          />
        </div>

        <div className="settings-section">
          <div className="settings-section-title">参数</div>
          <div className="settings-grid">
            <label className="settings-field">
              <span>Length Penalty</span>
              <input className="settings-input" type="number" min="0.1" max="5" step="0.05" value={lengthPenalty} onChange={(e) => handleLpChange(Number(e.target.value))} />
            </label>
            <label className="settings-field">
              <span>Playback Delay (ms)</span>
              <input className="settings-input" type="number" min="0" max="2000" step="50" value={playbackDelay} onChange={(e) => handleDelayChange(Number(e.target.value))} />
            </label>
            <label className="settings-field">
              <span>Max KV (tok)</span>
              <input className="settings-input" type="number" min="512" max="16384" step="512" value={maxKv} onChange={(e) => handleKvChange(Number(e.target.value))} />
            </label>
          </div>
        </div>

        {toast && (
          <div style={{
            position: 'fixed', bottom: 'calc(env(safe-area-inset-bottom, 0px) + 24px)',
            left: '50%', transform: 'translateX(-50%)',
            background: '#1f2937', color: '#fff', padding: '10px 20px',
            borderRadius: 12, fontSize: 13, fontWeight: 600,
            zIndex: 99999, pointerEvents: 'none',
          }}>
            {toast}
          </div>
        )}
      </div>
    </div>
  )
}

export default OmniSettingsWidget
