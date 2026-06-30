import { BoardState } from './board-state.js';
import { decodeFileToChunks, float32ToBase64 } from './file-audio-provider.js';
import { LiveBoardClient } from './live-board-client.js';
import { LiveMicProvider } from './live-mic-provider.js';
import { fetchDefaults, replayCase } from './replay-client.js';
// 2 x 3 grid = 6 可见 card，FIFO 挤出最早。
const state = new BoardState({ maxCards: 6 });
// non_spoken_block 两层 streaming 状态：block_id -> { kind, streamingPieces[], fullText?, closed }
const nonSpokenBlocks = new Map();
let liveClient = null;
let micProvider = null;
// Demo-aligned gapless audio player. Lazy-init on user gesture.
const audioPlayer = new AudioPlayer({ outputSampleRate: 24000 });
let audioPlayerReady = false;
// Cache of /api/defaults so the settings panel can show + override them.
let cachedDefaults = null;
const el = {
  status: document.getElementById('status'),
  micLevelBar: document.getElementById('micLevelBar'),
  micLevelText: document.getElementById('micLevelText'),
  statusLamp: document.getElementById('statusLamp'),
  modeBadge: document.getElementById('modeBadge'),
  wsState: document.getElementById('wsState'),
  sentChunks: document.getElementById('sentChunks'),
  aiAudioCount: document.getElementById('aiAudioCount'),
  liveHint: document.getElementById('liveHint'),
  casePath: document.getElementById('casePath'),
  loadDefaults: document.getElementById('loadDefaults'),
  runReplay: document.getElementById('runReplay'),
  audioFileInput: document.getElementById('audioFileInput'),
  runFileReplay: document.getElementById('runFileReplay'),
  startMicLive: document.getElementById('startMicLive'),
  stopMicLive: document.getElementById('stopMicLive'),
  userAudioPlayer: document.getElementById('userAudioPlayer'),
  aiAudioList: document.getElementById('aiAudioList'),
  padBeforeSec: document.getElementById('padBeforeSec'),
  padAfterSec: document.getElementById('padAfterSec'),
  generateAudio: document.getElementById('generateAudio'),
  timeline: document.getElementById('timeline'),
  board: document.getElementById('board'),
  aiSpeech: document.getElementById('aiSpeech'),
  nsStream: document.getElementById('nonSpokenStream'),
  kvMode: document.getElementById('kvMode'),
  kvCkpt: document.getElementById('kvCkpt'),
  kvTools: document.getElementById('kvTools'),
  systemPrompt: document.getElementById('systemPrompt'),
  refAudioPath: document.getElementById('refAudioPath'),
  genAudioToggle: document.getElementById('genAudioToggle'),
  resetSystemPrompt: document.getElementById('resetSystemPrompt'),
  resetRefAudio: document.getElementById('resetRefAudio'),
  debugToggle: document.getElementById('debugToggle'),
  debugDrawer: document.getElementById('debugDrawer'),
};

let sentChunkCount = 0;
let aiAudioCount = 0;

el.loadDefaults.addEventListener('click', async () => {
  setStatus('Loading defaults...');
  const defaults = await fetchDefaults();
  if (defaults.case_folder) {
    el.casePath.value = `${defaults.case_folder}/dob_midtrain_v1_20260628_animal_seed_ct01_and_04842.json`;
  }
  setStatus('Idle');
});

// Populate header mode badge + session settings panel from /api/defaults
(async () => {
  if (!el.modeBadge) return;
  try {
    cachedDefaults = await fetchDefaults();
    const defaults = cachedDefaults;
    if (defaults.use_mock_view) {
      el.modeBadge.textContent = 'MOCK';
      el.modeBadge.classList.add('mode-mock');
    } else {
      el.modeBadge.textContent = 'REAL MODEL';
      el.modeBadge.classList.add('mode-real');
    }
    if (el.kvMode) {
      el.kvMode.textContent = defaults.use_mock_view ? 'mock' : 'real';
    }
    if (el.kvCkpt) {
      const ckpt = (defaults.pt_path || '').split('/').slice(-2).join('/');
      el.kvCkpt.textContent = ckpt || '(no overlay)';
    }
    if (el.kvTools) {
      const tools = defaults.default_tools || [];
      el.kvTools.textContent = tools.length
        ? tools.map((t) => t?.function?.name || '?').join(', ')
        : '(none)';
    }
    // Settings panel editable defaults
    if (el.systemPrompt) {
      el.systemPrompt.value = defaults.default_system_prompt || '';
    }
    if (el.refAudioPath) {
      el.refAudioPath.value = defaults.default_ref_audio_path || '';
    }
  } catch (err) {
    el.modeBadge.textContent = `error`;
    el.modeBadge.classList.add('mode-error');
    el.modeBadge.title = err.message;
  }
})();

// Reset buttons restore the training-aligned defaults.
el.resetSystemPrompt?.addEventListener('click', () => {
  if (el.systemPrompt && cachedDefaults?.default_system_prompt) {
    el.systemPrompt.value = cachedDefaults.default_system_prompt;
  }
});
el.resetRefAudio?.addEventListener('click', () => {
  if (el.refAudioPath && cachedDefaults?.default_ref_audio_path) {
    el.refAudioPath.value = cachedDefaults.default_ref_audio_path;
  }
});

// Debug drawer toggle
el.debugToggle?.addEventListener('click', () => {
  el.debugDrawer?.classList.toggle('open');
});

el.runReplay.addEventListener('click', async () => {
  const casePath = el.casePath.value.trim();
  if (!casePath) {
    setStatus('Please enter a case path');
    return;
  }
  setStatus('Running replay...');
  clearViews();
  try {
    const response = await replayCase({ casePath });
    for (const event of response.events || []) {
      applyEvent(event);
    }
    setStatus(response.success ? 'Replay finished' : `Replay failed: ${response.error}`);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
});

el.startMicLive.addEventListener('click', async () => {
  clearViews();
  // AudioPlayer needs a user gesture before AudioContext can resume —
  // Start mic click qualifies.
  try {
    audioPlayer.init();
    audioPlayerReady = true;
  } catch (err) {
    console.warn('[AudioPlayer] init failed:', err);
  }
  try {
    liveClient = await createPreparedClient();
    micProvider = new LiveMicProvider({
      onChunk: (chunk) => {
        liveClient.send('audio_chunk', {
          audio_base64: float32ToBase64(chunk),
          sample_rate: 16000,
        });
        sentChunkCount += 1;
        updateStats();
      },
      onLevel: updateMicLevel,
      onPermissionHint: (text) => {
        if (el.liveHint) {
          el.liveHint.textContent = text;
          el.liveHint.classList.add('warning');
        }
      },
      onState: setMicState,
      onStatus: setStatus,
    });
    await micProvider.start();
    setMicState('live');
    setStatus('Listening · 自然说话，提到具体物体会出现在画板');
  } catch (err) {
    setStatus(`Mic live error: ${err.message}`);
    stopMicLive();
  }
});

el.stopMicLive.addEventListener('click', () => {
  stopMicLive();
  setStatus('Stopped');
});

function setMicState(state) {
  // state ∈ {requesting mic, live, stopped}
  if (!el.statusLamp) return;
  if (state === 'live') {
    el.statusLamp.classList.add('live');
    const label = el.statusLamp.querySelector('.label');
    if (label) label.textContent = 'LIVE';
  } else {
    el.statusLamp.classList.remove('live');
    const label = el.statusLamp.querySelector('.label');
    if (label) label.textContent = state ? state.toUpperCase() : 'IDLE';
  }
}

el.audioFileInput.addEventListener('change', () => {
  const file = el.audioFileInput.files?.[0];
  if (!file) return;
  el.userAudioPlayer.src = URL.createObjectURL(file);
});

el.runFileReplay.addEventListener('click', async () => {
  const file = el.audioFileInput.files?.[0];
  if (!file) {
    setStatus('Please choose an audio file');
    return;
  }
  clearViews();
  setStatus('Decoding file...');
  const chunks = await decodeFileToChunks(file, {
    padBeforeSec: Number(el.padBeforeSec.value || 0),
    padAfterSec: Number(el.padAfterSec.value || 2),
  });
  const client = await createPreparedClient();
  setStatus(`Streaming ${chunks.length} chunks...`);
  el.userAudioPlayer.currentTime = 0;
  el.userAudioPlayer.play().catch(() => {});
  for (let i = 0; i < chunks.length; i++) {
    client.send('audio_chunk', {
      audio_base64: float32ToBase64(chunks[i]),
      sample_rate: 16000,
    });
    setStatus(`Sent chunk ${i + 1}/${chunks.length}`);
    await sleep(1000);
  }
  client.send('finish', { reason: 'file_replay_finished' });
  setStatus('File replay finished');
});

async function createPreparedClient() {
  // Make sure we have defaults (for tools list at minimum).
  if (!cachedDefaults) {
    setStatus('Loading defaults…');
    cachedDefaults = await fetchDefaults();
  }
  // Pick up live edits from the settings panel; fall back to training defaults.
  const systemPrompt =
    (el.systemPrompt?.value || '').trim() ||
    cachedDefaults.default_system_prompt ||
    '';
  const refAudioPath =
    (el.refAudioPath?.value || '').trim() ||
    cachedDefaults.default_ref_audio_path ||
    '';
  if (!systemPrompt) {
    throw new Error('System prompt is empty and server returned no default.');
  }
  if (!refAudioPath) {
    throw new Error('Reference audio path is empty and server returned no default.');
  }
  const generateAudio = Boolean(el.genAudioToggle?.checked);

  const client = new LiveBoardClient({
    onEvent: applyEvent,
    onStatus: setStatus,
  });
  setStatus('Connecting…');
  await client.connect();
  setWsState('connected');
  client.send('prepare', {
    system_prompt: systemPrompt,
    tools: cachedDefaults.default_tools || [],
    ref_audio_path: refAudioPath,
    generate_audio: generateAudio,
  });
  return client;
}

function setWsState(state) {
  if (!el.wsState) return;
  el.wsState.textContent = state;
  el.wsState.classList.remove('connected', 'closed');
  if (state === 'connected') el.wsState.classList.add('connected');
  else if (state === 'closed' || state === 'idle') el.wsState.classList.add('closed');
}

function stopMicLive() {
  if (micProvider) {
    micProvider.stop();
    micProvider = null;
  }
  if (liveClient) {
    try {
      liveClient.send('finish', { reason: 'mic_live_stopped' });
    } catch (_) {}
    liveClient.close();
    liveClient = null;
  }
  setMicState('stopped');
  setWsState('closed');
}

function applyEvent(event) {
  // Always feed the raw event to the debug timeline (collapsed by default).
  appendTimeline(`${event.type} u=${event.unit_index ?? '-'}`);
  switch (event.type) {
    case 'spoken_final':
      // Just the AI text. No token counts, no unit numbers — user wants the
      // conversation, not telemetry.
      if (event.text) appendSpeech(event.text);
      // Prefer demo-style gapless playback: raw Float32 base64 → AudioPlayer.
      // Falls back to <audio src="data:audio/wav;base64,..."> in the debug
      // drawer for inspection.
      if (audioPlayerReady && event.payload?.audio_float32_base64) {
        if (!audioPlayer.turnActive) audioPlayer.beginTurn();
        audioPlayer.playChunk(event.payload.audio_float32_base64, performance.now());
        if (event.payload?.spoken_turn_eos) audioPlayer.endTurn?.();
      }
      if (event.payload?.audio_wav_base64) {
        appendAiAudio(event.unit_index, event.payload.audio_wav_base64);
      }
      return;
    case 'non_spoken_block_started':
      beginNonSpokenBlock(event);
      return;
    case 'non_spoken_delta':
      appendNonSpokenDelta(event);
      return;
    case 'non_spoken_block_closed':
      closeNonSpokenBlock(event);
      return;
    case 'think_final':
    case 'tool_call_final':
      // Already rendered in the two-layer block by non_spoken_block_closed.
      return;
    case 'board_card_created':
    case 'board_card_updated':
      state.upsert(event.card);
      renderBoard();
      return;
    case 'session_finished':
    case 'session_error':
    case 'unit_started':
    case 'unit_finished':
    case 'session_started':
    default:
      // Stay in the debug timeline only.
      return;
  }
}

function beginNonSpokenBlock(event) {
  const blockId = event.block_id;
  if (!blockId) return;
  const kind = event.block_kind || 'unknown';
  const placeholder = el.nsStream.querySelector('.placeholder');
  if (placeholder) placeholder.remove();
  const wrap = document.createElement('article');
  wrap.className = `ns-block kind-${kind}`;
  wrap.dataset.blockId = blockId;
  wrap.dataset.kind = kind;
  wrap.innerHTML = `
    <header class="ns-block-header">
      <span class="kind-tag">${escapeHtml(kind)}</span>
      <span class="status-tag">streaming…</span>
    </header>
    <section class="ns-layer streaming">
      <div class="layer-tag">streaming</div>
      <pre class="layer-body" data-role="streaming"></pre>
    </section>
    <section class="ns-layer full">
      <div class="layer-tag">full</div>
      <pre class="layer-body" data-role="full"></pre>
    </section>
  `;
  el.nsStream.appendChild(wrap);
  // keep newest at bottom and auto-scroll
  el.nsStream.scrollTop = el.nsStream.scrollHeight;
  nonSpokenBlocks.set(blockId, { kind, streamingPieces: [], fullText: null, closed: false, node: wrap });
}

function appendNonSpokenDelta(event) {
  const blockId = event.block_id;
  if (!blockId) return;
  const block = nonSpokenBlocks.get(blockId);
  if (!block) return;
  // Prefer the id-to-token vocab piece (what user explicitly asked for —
  // raw token strings; fall back to step_text BPE chunk only if token_strs
  // is empty (e.g. tokenizer adapter glitch on real model).
  const pieces = (event.token_strs && event.token_strs.length)
    ? event.token_strs
    : (event.step_text ? [event.step_text] : []);
  if (pieces.length) block.streamingPieces.push(...pieces);
  if (block.kind === 'unknown' && event.block_kind && event.block_kind !== 'unknown') {
    block.kind = event.block_kind;
    block.node.classList.remove('kind-unknown');
    block.node.classList.add(`kind-${block.kind}`);
    const kindTag = block.node.querySelector('.kind-tag');
    if (kindTag) kindTag.textContent = block.kind;
  }
  const target = block.node.querySelector('[data-role="streaming"]');
  if (target) {
    target.textContent = block.streamingPieces.join('');
  }
  el.nsStream.scrollTop = el.nsStream.scrollHeight;
}

function closeNonSpokenBlock(event) {
  const blockId = event.block_id;
  if (!blockId) return;
  const block = nonSpokenBlocks.get(blockId);
  if (!block) return;
  block.closed = true;
  block.fullText = event.full_text || null;
  block.node.classList.add('closed');
  const statusTag = block.node.querySelector('.status-tag');
  if (statusTag) statusTag.textContent = 'closed';
  const full = block.node.querySelector('[data-role="full"]');
  if (full) {
    if (block.fullText) {
      full.textContent = block.fullText;
    } else {
      // 原始需求：tool_call 非法 / parser 拿不到 full 时，full 层不显示。
      const section = full.closest('section.ns-layer.full');
      if (section) section.style.display = 'none';
    }
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function renderBoard() {
  if (!state.cards.length) {
    el.board.classList.add('empty-board');
    el.board.innerHTML = '<div class="empty-state">说出具体物体，例如「你看这只猫」「桌上有个苹果」</div>';
    return;
  }
  el.board.classList.remove('empty-board');
  el.board.innerHTML = state.cards.map((card) => {
    const image = card.image?.image_url
      ? `<img src="${escapeHtml(card.image.image_url)}" alt="${escapeHtml(card.query)}" />`
      : `<div class="placeholder">${card.status === 'searching' ? '搜图中…' : (card.error || '—')}</div>`;
    return `<article class="card ${escapeHtml(card.status)}" title="${escapeHtml(card.tool_call_id || '')}">
      ${image}
      <div class="card-body">
        <strong>${escapeHtml(card.query)}</strong>
        <span>${escapeHtml(card.status)}</span>
      </div>
    </article>`;
  }).join('');
}

function appendTimeline(text) {
  const item = document.createElement('div');
  item.className = 'timeline-row';
  item.textContent = text;
  el.timeline.appendChild(item);
}

function appendSpeech(text) {
  const placeholder = el.aiSpeech.querySelector('.placeholder');
  if (placeholder) placeholder.remove();
  // Append text to the last speech row if it's recent (< 1.5s ago), so the
  // AI's per-unit fragments coalesce into readable sentences instead of one
  // 1-3 char chip per unit.
  const last = el.aiSpeech.lastElementChild;
  const now = Date.now();
  if (last && last.classList.contains('speech-row') && (now - Number(last.dataset.ts || 0)) < 1500) {
    last.textContent = (last.textContent || '') + text;
    last.dataset.ts = String(now);
  } else {
    const item = document.createElement('div');
    item.className = 'speech-row';
    item.dataset.ts = String(now);
    item.textContent = text;
    el.aiSpeech.appendChild(item);
  }
  el.aiSpeech.scrollTop = el.aiSpeech.scrollHeight;
}

function clearViews() {
  state.cards = [];
  nonSpokenBlocks.clear();
  el.timeline.innerHTML = '';
  el.board.innerHTML = '';
  renderBoard();
  el.aiSpeech.innerHTML = '<div class="placeholder">AI 还没开口</div>';
  el.nsStream.innerHTML = '<div class="placeholder">还没有 think / tool_call 块</div>';
  el.aiAudioList.innerHTML = '';
  sentChunkCount = 0;
  aiAudioCount = 0;
  updateStats();
  updateMicLevel(0);
}

function setStatus(text) {
  el.status.textContent = text;
}

function appendAiAudio(unitIndex, wavBase64) {
  const wrap = document.createElement('div');
  wrap.className = 'audio-item';
  const label = document.createElement('span');
  label.textContent = `AI unit ${unitIndex}`;
  const audio = document.createElement('audio');
  audio.controls = true;
  audio.src = `data:audio/wav;base64,${wavBase64}`;
  wrap.appendChild(label);
  wrap.appendChild(audio);
  el.aiAudioList.appendChild(wrap);
  aiAudioCount += 1;
  updateStats();
  audio.play().catch(() => {});
}

function updateMicLevel(level) {
  const clamped = Math.max(0, Math.min(1, level / 0.08));
  if (el.micLevelBar) el.micLevelBar.style.width = `${Math.round(clamped * 100)}%`;
  if (el.micLevelText) el.micLevelText.textContent = level.toFixed(2);
}

function updateStats() {
  el.sentChunks.textContent = String(sentChunkCount);
  el.aiAudioCount.textContent = String(aiAudioCount);
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}
