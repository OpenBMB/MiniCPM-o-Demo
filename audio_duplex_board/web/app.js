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
const el = {
  status: document.getElementById('status'),
  micLevelBar: document.getElementById('micLevelBar'),
  micLevelText: document.getElementById('micLevelText'),
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
  eventLog: document.getElementById('eventLog'),
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

// Reflect runtime mode in the hero badge so the page itself answers "mock or real?".
(async () => {
  const badge = document.getElementById('modeBadge');
  if (!badge) return;
  try {
    const defaults = await fetchDefaults();
    if (defaults.use_mock_view) {
      badge.textContent = 'MOCK (no model)';
      badge.style.color = '#fbbf24';
    } else {
      const ckpt = (defaults.pt_path || '').split('/').slice(-2).join('/');
      badge.textContent = `REAL MODEL · ${ckpt || 'no overlay'}`;
      badge.style.color = '#22c55e';
    }
  } catch (err) {
    badge.textContent = `error: ${err.message}`;
    badge.style.color = '#ef4444';
  }
})();

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
        el.liveHint.textContent = text;
        el.liveHint.classList.add('warning');
      },
      onState: (text) => {
        el.wsState.textContent = text;
      },
      onStatus: setStatus,
    });
    await micProvider.start();
    setStatus('Mic live: speak loudly to trigger board cards');
  } catch (err) {
    setStatus(`Mic live error: ${err.message}`);
    stopMicLive();
  }
});

el.stopMicLive.addEventListener('click', () => {
  stopMicLive();
  setStatus('Mic live stopped');
});

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
  const client = new LiveBoardClient({
    onEvent: applyEvent,
    onStatus: setStatus,
  });
  setStatus('Connecting...');
  await client.connect();
  el.wsState.textContent = 'connected';
  client.send('prepare', {
    system_prompt: '你是一个实时语音助手。听到适合展示到画板上的具体物体时，在后台调用 display_object_on_board。',
    tools: [{
      type: 'function',
      function: {
        name: 'display_object_on_board',
        description: 'Display a named object on the board.',
        parameters: {
          type: 'object',
          properties: { name: { type: 'string' } },
          required: ['name'],
        },
      },
    }],
    generate_audio: Boolean(el.generateAudio.checked),
  });
  return client;
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
  el.wsState.textContent = 'closed';
}

function applyEvent(event) {
  switch (event.type) {
    case 'unit_started':
      appendTimeline(`unit ${event.unit_index}: audio=${event.payload?.n_audio ?? 0}, speaking=${event.payload?.is_speaking}`);
      return;
    case 'spoken_final':
      appendTimeline(`unit ${event.unit_index}: spoken "${event.text || ''}" (${(event.token_ids || []).length} tokens)`);
      if (event.text) appendSpeech(event.unit_index, event.text);
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
      // 已通过 non_spoken_block_closed 渲染 full 层；这里只做 log
      appendLog('think', event.think_text || '');
      return;
    case 'tool_call_final':
      appendLog('tool_call', JSON.stringify(event.tool_call, null, 2));
      return;
    case 'board_card_created':
    case 'board_card_updated':
      state.upsert(event.card);
      renderBoard();
      return;
    case 'session_finished':
      appendLog('summary', JSON.stringify(event.payload || {}, null, 2));
      return;
    case 'session_error':
      appendLog('error', event.text || 'unknown error');
      return;
    default:
      // 未识别事件不阻塞，但写进 log 便于排查 protocol 演进
      appendLog('event', `${event.type}: ${JSON.stringify(event)}`);
      return;
  }
}

function beginNonSpokenBlock(event) {
  const blockId = event.block_id;
  if (!blockId) return;
  const kind = event.block_kind || 'unknown';
  const wrap = document.createElement('article');
  wrap.className = `ns-block kind-${kind}`;
  wrap.dataset.blockId = blockId;
  wrap.dataset.kind = kind;
  wrap.innerHTML = `
    <header class="ns-block-header">
      <span class="kind-tag">${escapeHtml(kind)}</span>
      <span class="unit-tag">unit ${event.unit_index}</span>
      <span class="status-tag">streaming</span>
    </header>
    <section class="ns-layer streaming">
      <div class="layer-tag">streaming_content (id-to-token)</div>
      <pre class="layer-body" data-role="streaming"></pre>
    </section>
    <section class="ns-layer full">
      <div class="layer-tag">full_content (BPE merged after close)</div>
      <pre class="layer-body" data-role="full"></pre>
    </section>
  `;
  el.eventLog.appendChild(wrap);
  nonSpokenBlocks.set(blockId, { kind, streamingPieces: [], fullText: null, closed: false, node: wrap });
}

function appendNonSpokenDelta(event) {
  const blockId = event.block_id;
  if (!blockId) return;
  const block = nonSpokenBlocks.get(blockId);
  if (!block) return;
  const pieces = event.token_strs || [];
  block.streamingPieces.push(...pieces);
  // 如果 kind 在 started 时还是 unknown 而 delta 已经能判断，更新一下
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
      // tool_call 非法 / parser 拿不到 full 时，原始需求要求 full 层不显示
      const section = full.closest('section.ns-layer.full');
      if (section) section.classList.add('empty');
    }
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function renderBoard() {
  if (!state.cards.length) {
    el.board.classList.add('empty-board');
    el.board.innerHTML = '<div class="empty-state">No cards yet. Speak loudly to trigger 苹果 / 番茄 / 腌白菜.</div>';
    return;
  }
  el.board.classList.remove('empty-board');
  el.board.innerHTML = state.cards.map((card) => {
    const image = card.image?.image_url
      ? `<img src="${escapeHtml(card.image.image_url)}" alt="${escapeHtml(card.query)}" />`
      : `<div class="placeholder">Searching...</div>`;
    return `<article class="card ${escapeHtml(card.status)}">
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

function appendSpeech(unitIndex, text) {
  const item = document.createElement('div');
  item.className = 'speech-row';
  item.innerHTML = `<span>unit ${unitIndex}</span><strong>${escapeHtml(text)}</strong>`;
  el.aiSpeech.appendChild(item);
}

function appendLog(kind, text) {
  const item = document.createElement('pre');
  item.className = `log-row ${kind}`;
  item.textContent = `[${kind}]\n${text}`;
  el.eventLog.appendChild(item);
}

function clearViews() {
  state.cards = [];
  nonSpokenBlocks.clear();
  el.timeline.innerHTML = '';
  el.board.innerHTML = '';
  renderBoard();
  el.aiSpeech.innerHTML = '';
  el.eventLog.innerHTML = '';
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
  el.micLevelBar.style.width = `${Math.round(clamped * 100)}%`;
  el.micLevelText.textContent = level.toFixed(4);
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
