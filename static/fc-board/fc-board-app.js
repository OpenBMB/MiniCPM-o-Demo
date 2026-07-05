import { BoardState } from './board-state.js';
import { decodeFileToChunks, float32ToBase64 } from './file-audio-provider.js';
import { FcRealtimeClient } from './fc-realtime-client.js';
import { LiveMicProvider } from './live-mic-provider.js';
import { AudioPlayer } from './audio-player.js';

const DEFAULT_SYSTEM_PROMPT = `你是一个可以一边听用户说话、一边思考并调用工具的语音助手。用户让你把故事或描述中出现的动物、植物或具体物体放到画板上时，使用 display_object_on_board 工具。不要等用户完全讲完才思考；在你确认具体对象后，可以调用工具把对象放到画板。`;

const DISPLAY_OBJECT_TOOL = {
  type: 'function',
  function: {
    name: 'display_object_on_board',
    description: 'Display a named concrete object on the visual board.',
    parameters: {
      type: 'object',
      properties: {
        name: { type: 'string', description: 'The object name to display on the board.' },
      },
      required: ['name'],
    },
  },
};

const state = new BoardState({ maxCards: 6 });
const nonSpokenBlocks = new Map();
const toolCallBlocks = new Map();
const streamEvents = new Map();
const audioPlayer = new AudioPlayer({ outputSampleRate: 24000 });

let liveClient = null;
let micProvider = null;
let audioPlayerReady = false;
let sentChunkCount = 0;
let aiAudioCount = 0;
let micPeakSinceStart = 0;
let blockSeq = 0;
let activeNonSpokenBlockId = null;
let fcBoardDefaults = null;
let micLiveState = 'idle';
let streamEventSeq = 0;
const speechQueue = [];
let speechDrainer = null;

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
  audioFileInput: document.getElementById('audioFileInput'),
  runFileReplay: document.getElementById('runFileReplay'),
  startMicLive: document.getElementById('startMicLive'),
  stopMicLive: document.getElementById('stopMicLive'),
  userAudioPlayer: document.getElementById('userAudioPlayer'),
  aiAudioList: document.getElementById('aiAudioList'),
  padBeforeSec: document.getElementById('padBeforeSec'),
  padAfterSec: document.getElementById('padAfterSec'),
  timeline: document.getElementById('timeline'),
  streamFeed: document.getElementById('streamFeed'),
  streamFilter: document.getElementById('streamFilter'),
  openStreamDialog: document.getElementById('openStreamDialog'),
  streamDialog: document.getElementById('streamDialog'),
  streamModalCount: document.getElementById('streamModalCount'),
  streamModalList: document.getElementById('streamModalList'),
  board: document.getElementById('board'),
  aiSpeech: document.getElementById('aiSpeech'),
  nsStream: document.getElementById('nonSpokenStream'),
  kvMode: document.getElementById('kvMode'),
  kvCkpt: document.getElementById('kvCkpt'),
  kvTools: document.getElementById('kvTools'),
  systemPrompt: document.getElementById('systemPrompt'),
  refAudioPath: document.getElementById('refAudioPath'),
  nonSpokenScheduling: document.getElementById('nonSpokenScheduling'),
  nonSpokenBudget: document.getElementById('nonSpokenBudget'),
  debugBudgetUsed: document.getElementById('debugBudgetUsed'),
  debugBudgetMax: document.getElementById('debugBudgetMax'),
  debugBudgetUpdated: document.getElementById('debugBudgetUpdated'),
  resetSystemPrompt: document.getElementById('resetSystemPrompt'),
  resetRefAudio: document.getElementById('resetRefAudio'),
  debugToggle: document.getElementById('debugToggle'),
  debugDrawer: document.getElementById('debugDrawer'),
};

initPage();

function initPage() {
  el.modeBadge.textContent = 'Realtime API';
  el.modeBadge.className = 'mode-tag mode-real';
  el.kvMode.textContent = '/v1/realtime?mode=audio';
  el.kvCkpt.textContent = 'backend selected';
  el.kvTools.textContent = DISPLAY_OBJECT_TOOL.function.name;
  applyDefaults({});
  setWsState('idle');
  setMicLiveState('idle');
  setStatus('Loading defaults…');
  renderBoard();
  loadFcBoardDefaults();
}

el.resetSystemPrompt?.addEventListener('click', () => {
  el.systemPrompt.value = defaultSystemPrompt();
});

el.resetRefAudio?.addEventListener('click', () => {
  el.refAudioPath.value = defaultRefAudioPath();
});

el.debugToggle?.addEventListener('click', () => {
  el.debugDrawer?.classList.toggle('open');
});

el.streamFilter?.addEventListener('change', applyStreamFilter);

el.openStreamDialog?.addEventListener('click', () => {
  openStreamDialog();
});

el.streamDialog?.addEventListener('click', (event) => {
  if (event.target === el.streamDialog) el.streamDialog.close();
});

el.startMicLive.addEventListener('click', async () => {
  if (micLiveState !== 'idle' && micLiveState !== 'stopped' && micLiveState !== 'closed' && micLiveState !== 'error') return;
  setMicLiveState('starting');
  clearViews();
  resetMicPeak();
  try {
    audioPlayer.init();
    audioPlayerReady = true;
  } catch (err) {
    console.warn('[AudioPlayer] init failed:', err);
  }
  try {
    liveClient = await createRealtimeSession();
    micProvider = new LiveMicProvider({
      onChunk: (chunk) => {
        liveClient.appendAudio({ audioBase64: float32ToBase64(chunk), sampleRate: 16000 });
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
    setMicLiveState('live');
    setStatus('Listening via /v1/realtime · mention concrete objects for the board');
  } catch (err) {
    setStatus(`Mic live error: ${err.message}`);
    stopMicLive('error');
  }
});

el.stopMicLive.addEventListener('click', () => {
  if (micLiveState === 'idle' || micLiveState === 'stopped' || micLiveState === 'closed') return;
  stopMicLive();
  setStatus('Stopped');
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
  const client = await createRealtimeSession();
  setStatus(`Streaming ${chunks.length} chunks through /v1/realtime...`);
  el.userAudioPlayer.currentTime = 0;
  el.userAudioPlayer.play().catch(() => {});
  for (let i = 0; i < chunks.length; i++) {
    client.appendAudio({ audioBase64: float32ToBase64(chunks[i]), sampleRate: 16000 });
    sentChunkCount += 1;
    updateStats();
    setStatus(`Sent chunk ${i + 1}/${chunks.length}`);
    await sleep(1000);
  }
  client.close('file_replay_finished');
  setWsState('closed');
  setStatus('File replay finished');
});

async function createRealtimeSession() {
  const client = new FcRealtimeClient({
    onEvent: applyApiEvent,
    onSend: (message) => appendStreamEvent('tx', message),
    onStatus: setStatus,
  });
  setStatus('Connecting to /v1/realtime...');
  await client.connect();
  setWsState('connected');
  client.initSession(buildSessionInitPayload());
  return client;
}

function buildSessionInitPayload() {
  const refAudioPath = (el.refAudioPath?.value || '').trim();
  const nonSpokenScheduling = ['quality', 'latency'].includes(el.nonSpokenScheduling?.value)
    ? el.nonSpokenScheduling.value
    : 'quality';
  const nonSpokenBudget = clampInt(el.nonSpokenBudget?.value, 1, 512, 12);
  const payload = {
    mode: 'full_duplex',
    fc_duplex: true,
    system_prompt: (el.systemPrompt?.value || '').trim() || defaultSystemPrompt(),
    tools: defaultTools(),
    generate_audio: true,
    config: {
      runtime: 'fc_duplex',
      auto_execute_tools: false,
      non_spoken_scheduling: nonSpokenScheduling,
      sample_rate: 16000,
      max_spoken_tokens: 24,
      non_spoken_budget_per_unit: nonSpokenBudget,
      decode_mode: 'greedy',
    },
  };
  if (refAudioPath) payload.ref_audio_path = refAudioPath;
  return payload;
}

async function loadFcBoardDefaults() {
  try {
    const response = await fetch('/api/fc_board/defaults', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    fcBoardDefaults = await response.json();
    applyDefaults(fcBoardDefaults);
    setStatus('Click Start to begin.');
  } catch (err) {
    console.warn('[fc-board] failed to load defaults:', err);
    fcBoardDefaults = null;
    applyDefaults({});
    setStatus('Click Start to begin. Defaults endpoint unavailable; using fallback prompt.');
  }
}

function applyDefaults(defaults) {
  if (el.systemPrompt) el.systemPrompt.value = defaults.default_system_prompt || DEFAULT_SYSTEM_PROMPT;
  if (el.refAudioPath) el.refAudioPath.value = defaults.default_ref_audio_path || '';
  if (el.kvTools) el.kvTools.textContent = defaultTools().map(tool => tool?.function?.name || tool?.name || 'tool').join(', ');
}

function defaultSystemPrompt() {
  return fcBoardDefaults?.default_system_prompt || DEFAULT_SYSTEM_PROMPT;
}

function defaultRefAudioPath() {
  return fcBoardDefaults?.default_ref_audio_path || '';
}

function defaultTools() {
  const tools = fcBoardDefaults?.default_tools;
  return Array.isArray(tools) && tools.length ? tools : [DISPLAY_OBJECT_TOOL];
}

function applyApiEvent(event) {
  appendStreamEvent('rx', event);
  appendTimeline(`${event.type} ${event.kind || ''} ${event.tool_call_id || ''}`.trim());
  switch (event.type) {
    case 'session.created':
      setStatus('Session created');
      return;
    case 'session.closed':
      setWsState('closed');
      releaseMicLive('closed');
      setStatus(`Session closed: ${event.reason || 'closed'}`);
      return;
    case 'response.output.delta':
      handleOutputDelta(event);
      return;
    case 'response.think.begin':
      beginNonSpokenBlock({ block_id: blockIdFor(event, 'think'), block_kind: 'think' });
      return;
    case 'response.think.delta':
      appendNonSpokenDelta({ block_id: blockIdFor(event, 'think'), step_text: event.delta || '' });
      return;
    case 'response.think.end':
      closeNonSpokenBlock({ block_id: blockIdFor(event, 'think'), block_kind: 'think' });
      return;
    case 'response.tool_call.args.begin':
      toolCallBlocks.set(event.tool_call_id, blockIdFor(event, 'tool_call'));
      beginNonSpokenBlock({ block_id: toolCallBlocks.get(event.tool_call_id), block_kind: 'tool_call' });
      return;
    case 'response.tool_call.args.delta':
      appendNonSpokenDelta({ block_id: blockIdFor(event, 'tool_call'), step_text: event.delta || '' });
      return;
    case 'response.tool_call.args.end':
      closeNonSpokenBlock({ block_id: blockIdFor(event, 'tool_call'), block_kind: 'tool_call' });
      return;
    case 'response.tool_call.args.raw':
      handleToolCallRaw(event);
      return;
    case 'response.tool_result':
      handleToolResult(event);
      return;
    case 'response.output.sp_tokens':
      handleSpToken(event);
      return;
    case 'response.debug':
      handleDebugEvent(event);
      return;
    default:
      return;
  }
}

function handleOutputDelta(event) {
  if (event.kind === 'listen') {
    handleSpokenOutput({ isListen: true, isSpeaking: false });
    closeActiveNonSpokenBlock();
    return;
  }
  if (event.kind === 'text' && event.text) {
    enqueueSpeech(event.text, 0);
    return;
  }
  if (event.kind === 'audio' && event.audio) {
    handleSpokenOutput({ isListen: false, isSpeaking: true, audioBase64: event.audio, sampleRate: event.sample_rate || 24000 });
    return;
  }
  if (event.kind === 'non_spoken') {
    if (!activeNonSpokenBlockId) {
      activeNonSpokenBlockId = `non_spoken:${event.input_id || 'input'}:${++blockSeq}`;
      beginNonSpokenBlock({ block_id: activeNonSpokenBlockId, block_kind: 'unknown' });
    }
    const text = event.text || (event.token_strs || []).join('');
    appendNonSpokenDelta({ block_id: activeNonSpokenBlockId, step_text: text });
  }
}

function handleSpToken(event) {
  const token = String(event.token || '');
  if (['no_action', 'non_spoken_eos', 'non_spoken_budget_reached', 'non_spoken_hold', 'non_spoken_abort'].includes(token)) {
    closeActiveNonSpokenBlock();
  }
}

function handleDebugEvent(event) {
  const debug = event.debug || {};
  if ('used' in debug && el.debugBudgetUsed) {
    el.debugBudgetUsed.textContent = formatMaybeNumber(debug.used);
  }
  if ('estimated_max_budget_1s' in debug && el.debugBudgetMax) {
    el.debugBudgetMax.textContent = formatMaybeNumber(debug.estimated_max_budget_1s);
  }
  if (el.debugBudgetUpdated) {
    el.debugBudgetUpdated.textContent = new Date().toLocaleTimeString();
  }
}

function handleToolCallRaw(event) {
  const blockId = blockIdFor(event, 'tool_call');
  const raw = event.raw || {};
  closeNonSpokenBlock({
    block_id: blockId,
    block_kind: 'tool_call',
    full_text: formatToolCall(raw),
  });
  if (raw.name !== 'display_object_on_board' || raw.error) return;
  const args = parseArguments(raw.arguments);
  const query = String(args.name || '').trim();
  if (!query) return;
  state.upsert({
    card_id: cardIdFor(event.tool_call_id),
    tool_call_id: event.tool_call_id,
    query,
    status: 'searching',
  });
  renderBoard();
  executeDisplayObjectOnBoard({ query, toolCallId: event.tool_call_id });
}

function handleToolResult(event) {
  const result = event.result || {};
  const query = result.query || result.image?.query || event.name || event.tool_call_id;
  state.upsert({
    card_id: cardIdFor(event.tool_call_id),
    tool_call_id: event.tool_call_id,
    query,
    status: result.error ? 'error' : 'ready',
    image: result.image,
    error: result.error,
  });
  renderBoard();
}

async function executeDisplayObjectOnBoard({ query, toolCallId }) {
  let result;
  try {
    const response = await fetch('/api/fc_board/tools/display_object_on_board', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ name: query, tool_call_id: toolCallId }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    result = await response.json();
  } catch (err) {
    result = {
      query,
      image: {
        query,
        asset_id: `client-tool:${toolCallId || query}`,
        image_url: placeholderImageDataUrl(query),
        source_url: null,
        title: query,
        elapsed_ms: 0,
        error: String(err?.message || err),
      },
      error: String(err?.message || err),
      tool_response_content: JSON.stringify({
        status: 'displayed',
        name: query,
        reason: '已在画板显示该对象。',
      }),
    };
  }
  state.upsert({
    card_id: cardIdFor(toolCallId),
    tool_call_id: toolCallId,
    query: result.query || query,
    status: result.error ? 'error' : 'ready',
    image: result.image,
    error: result.error,
  });
  renderBoard();
  sendDisplayObjectToolResult({
    toolCallId,
    query: result.query || query,
    content: result.tool_response_content,
  });
}

function sendDisplayObjectToolResult({ toolCallId, query, content }) {
  if (!liveClient || !toolCallId) return;
  const text = content || JSON.stringify({
    status: 'displayed',
    name: query,
    reason: '已在画板显示该对象。',
  });
  liveClient.sendToolResult({
    toolCallId,
    contents: [{ kind: 'text', text }],
  });
}

function blockIdFor(event, kind) {
  if (kind === 'tool_call') {
    const key = event.tool_call_id || 'pending';
    if (!toolCallBlocks.has(key)) toolCallBlocks.set(key, `tool_call:${key}`);
    return toolCallBlocks.get(key);
  }
  return `${kind}:${event.input_id || 'input'}:${event.response_id || 'resp'}`;
}

function cardIdFor(toolCallId) {
  return `card:${toolCallId || 'pending'}`;
}

function closeActiveNonSpokenBlock() {
  if (!activeNonSpokenBlockId) return;
  closeNonSpokenBlock({ block_id: activeNonSpokenBlockId, block_kind: 'unknown' });
  activeNonSpokenBlockId = null;
}

function beginNonSpokenBlock(event) {
  const blockId = event.block_id;
  if (!blockId || nonSpokenBlocks.has(blockId)) return;
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
  nonSpokenBlocks.set(blockId, { kind, streamingPieces: [], fullText: null, closed: false, node: wrap });
  scrollNsToBottom();
}

function appendNonSpokenDelta(event) {
  const blockId = event.block_id;
  if (!blockId) return;
  if (!nonSpokenBlocks.has(blockId)) beginNonSpokenBlock({ block_id: blockId, block_kind: 'unknown' });
  const block = nonSpokenBlocks.get(blockId);
  const piece = event.step_text || event.delta || '';
  if (!piece) return;
  block.streamingPieces.push(piece);
  const target = block.node.querySelector('[data-role="streaming"]');
  if (target) target.textContent = block.streamingPieces.join('');
  scrollNsToBottom();
}

function closeNonSpokenBlock(event) {
  const blockId = event.block_id;
  if (!blockId) return;
  const block = nonSpokenBlocks.get(blockId);
  if (!block) return;
  block.closed = true;
  block.fullText = event.full_text || block.fullText || null;
  const finalKind = (event.block_kind && event.block_kind !== 'unknown') ? event.block_kind : block.kind;
  if (finalKind && finalKind !== 'unknown' && block.kind !== finalKind) {
    block.kind = finalKind;
    block.node.classList.remove('kind-unknown');
    block.node.classList.add(`kind-${finalKind}`);
    const kindTag = block.node.querySelector('.kind-tag');
    if (kindTag) kindTag.textContent = finalKind;
  }
  block.node.classList.add('closed');
  const statusTag = block.node.querySelector('.status-tag');
  if (statusTag) statusTag.textContent = 'closed';
  const streamingTarget = block.node.querySelector('[data-role="streaming"]');
  if (streamingTarget && finalKind && finalKind !== 'unknown') {
    const inner = block.streamingPieces.join('');
    streamingTarget.textContent = `<${finalKind}>\n${inner}\n</${finalKind}>`;
  }
  const full = block.node.querySelector('[data-role="full"]');
  if (full) {
    if (block.fullText) {
      full.textContent = block.fullText;
    } else {
      const section = full.closest('section.ns-layer.full');
      if (section) section.style.display = 'none';
    }
  }
  scrollNsToBottom();
}

function handleSpokenOutput({ isListen, isSpeaking, audioBase64, sampleRate }) {
  if (!audioPlayerReady) return;
  if (isListen) {
    if (audioPlayer.turnActive) audioPlayer.endTurn();
    return;
  }
  if (isSpeaking && audioBase64) {
    if (!audioPlayer.turnActive) audioPlayer.beginTurn();
    audioPlayer.playChunk(audioBase64, performance.now());
    aiAudioCount += 1;
    updateStats();
    appendAiAudio(aiAudioCount, audioBase64, sampleRate || 24000);
  }
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

function enqueueSpeech(text, audioMs) {
  if (!text) return;
  const chars = [...text];
  const pendingBudget = Math.max(0, audioMs || chars.length * 80);
  const backlog = speechQueue.length;
  const stepMs = Math.max(15, Math.round(pendingBudget / Math.max(1, chars.length + backlog * 0.5)));
  for (const ch of chars) speechQueue.push({ ch, stepMs });
  pumpSpeechQueue();
}

function pumpSpeechQueue() {
  if (speechDrainer) return;
  const step = () => {
    const item = speechQueue.shift();
    if (!item) {
      speechDrainer = null;
      return;
    }
    appendSpeechChar(item.ch);
    speechDrainer = setTimeout(step, item.stepMs);
  };
  speechDrainer = setTimeout(step, 0);
}

function appendSpeechChar(ch) {
  const placeholder = el.aiSpeech.querySelector('.placeholder');
  if (placeholder) placeholder.remove();
  const last = el.aiSpeech.lastElementChild;
  const now = Date.now();
  if (last && last.classList.contains('speech-row') && (now - Number(last.dataset.ts || 0)) < 1500) {
    last.textContent = (last.textContent || '') + ch;
    last.dataset.ts = String(now);
  } else {
    const item = document.createElement('div');
    item.className = 'speech-row';
    item.dataset.ts = String(now);
    item.textContent = ch;
    el.aiSpeech.appendChild(item);
  }
  const speechScroller = el.aiSpeech.parentElement;
  if (speechScroller) requestAnimationFrame(() => { speechScroller.scrollTop = speechScroller.scrollHeight; });
}

function appendAiAudio(index, audioBase64, sampleRate) {
  if (!el.debugDrawer?.classList.contains('open')) return;
  const wavBase64 = float32Base64ToWavBase64(audioBase64, sampleRate || 24000);
  const wrap = document.createElement('div');
  wrap.className = 'audio-item';
  const label = document.createElement('span');
  label.textContent = `AI chunk ${index}`;
  const audio = document.createElement('audio');
  audio.controls = true;
  audio.src = `data:audio/wav;base64,${wavBase64}`;
  wrap.appendChild(label);
  wrap.appendChild(audio);
  el.aiAudioList.appendChild(wrap);
}

function stopMicLive(finalState = 'stopped') {
  setMicLiveState('stopping');
  if (micProvider) {
    micProvider.stop();
    micProvider = null;
  }
  if (liveClient) {
    try { liveClient.close('mic_live_stopped'); } catch (_) {}
    liveClient = null;
  }
  setMicLiveState(finalState);
  setWsState('closed');
}

function releaseMicLive(finalState = 'closed') {
  if (micProvider) {
    micProvider.stop();
    micProvider = null;
  }
  liveClient = null;
  setMicLiveState(finalState);
}

function clearViews() {
  state.cards = [];
  nonSpokenBlocks.clear();
  toolCallBlocks.clear();
  activeNonSpokenBlockId = null;
  el.timeline.innerHTML = '';
  el.streamFeed.innerHTML = '<div class="placeholder">还没有数据流</div>';
  el.board.innerHTML = '';
  renderBoard();
  speechQueue.length = 0;
  if (speechDrainer) { clearTimeout(speechDrainer); speechDrainer = null; }
  el.aiSpeech.innerHTML = '<div class="placeholder">AI 还没开口</div>';
  el.nsStream.innerHTML = '<div class="placeholder">还没有 think / tool_call 块</div>';
  el.aiAudioList.innerHTML = '';
  sentChunkCount = 0;
  aiAudioCount = 0;
  updateStats();
  updateMicLevel(0);
}

function setMicState(state) {
  if (!el.statusLamp) return;
  const label = el.statusLamp.querySelector('.label');
  if (state === 'live') {
    el.statusLamp.classList.add('live');
    if (label) label.textContent = 'LIVE';
  } else {
    el.statusLamp.classList.remove('live');
    if (label) label.textContent = state ? state.toUpperCase() : 'IDLE';
  }
}

function setMicLiveState(state) {
  micLiveState = state;
  setMicState(state);
  updateControlState();
}

function updateControlState() {
  const active = micLiveState === 'starting' || micLiveState === 'live' || micLiveState === 'stopping';
  if (el.startMicLive) el.startMicLive.disabled = active;
  if (el.stopMicLive) el.stopMicLive.disabled = !active || micLiveState === 'stopping';
}

function setWsState(state) {
  if (!el.wsState) return;
  el.wsState.textContent = state;
  el.wsState.classList.remove('connected', 'closed');
  if (state === 'connected') el.wsState.classList.add('connected');
  else if (state === 'closed' || state === 'idle') el.wsState.classList.add('closed');
}

function updateMicLevel(level) {
  const clamped = Math.max(0, Math.min(1, level / 0.08));
  if (el.micLevelBar) el.micLevelBar.style.width = `${Math.round(clamped * 100)}%`;
  if (level > micPeakSinceStart) micPeakSinceStart = level;
  if (el.micLevelText) el.micLevelText.textContent = `${level.toFixed(2)}  (peak ${micPeakSinceStart.toFixed(2)})`;
}

function resetMicPeak() {
  micPeakSinceStart = 0;
}

function updateStats() {
  el.sentChunks.textContent = String(sentChunkCount);
  el.aiAudioCount.textContent = String(aiAudioCount);
}

function setStatus(text) {
  el.status.textContent = text;
}

function appendTimeline(text) {
  const item = document.createElement('div');
  item.className = 'timeline-row';
  item.textContent = text;
  el.timeline.appendChild(item);
  el.timeline.scrollTop = el.timeline.scrollHeight;
}

function appendStreamEvent(direction, event) {
  if (!el.streamFeed) return;
  const placeholder = el.streamFeed.querySelector('.placeholder');
  if (placeholder) placeholder.remove();
  const eventId = `stream_evt_${++streamEventSeq}`;
  streamEvents.set(eventId, { direction, event });
  const row = document.createElement('details');
  row.className = `stream-row ${direction}`;
  row.dataset.eventId = eventId;
  const type = event.type || '(unknown)';
  row.dataset.category = streamEventCategory(direction, event);
  row.innerHTML = renderStreamRowContent({ direction, event, type, timeText: new Date().toLocaleTimeString() });
  el.streamFeed.appendChild(row);
  applyStreamFilter();
  el.streamFeed.scrollTop = el.streamFeed.scrollHeight;
}

function openStreamDialog() {
  if (!el.streamDialog || !el.streamModalList) return;
  const records = [...streamEvents.values()];
  if (el.streamModalCount) {
    el.streamModalCount.textContent = `${records.length} event${records.length === 1 ? '' : 's'}`;
  }
  el.streamModalList.innerHTML = records.map((record, index) => {
    const type = record.event.type || '(unknown)';
    return `<details class="stream-row ${record.direction} modal-stream-row">
      ${renderStreamRowContent({
        direction: record.direction,
        event: record.event,
        type,
        timeText: `#${index + 1}`,
      })}
    </details>`;
  }).join('') || '<div class="placeholder">No stream events yet</div>';
  el.streamDialog.showModal();
}

function renderStreamRowContent({ direction, event, type, timeText }) {
  const summary = summarizeStreamEvent(event);
  const json = prettyJson(event);
  const media = renderStreamEventMedia(event);
  return `
    <summary class="stream-head">
      <span class="stream-dir">${direction.toUpperCase()}</span>
      <span class="stream-type">${escapeHtml(type)}</span>
      <span class="stream-time">${escapeHtml(timeText)}</span>
    </summary>
    <div class="stream-body">${escapeHtml(summary)}</div>
    ${media}
    <pre class="stream-json">${escapeHtml(json)}</pre>
  `;
}

function renderStreamEventMedia(event) {
  let audioBase64 = '';
  let sampleRate = 24000;
  if (event.type === 'response.output.delta' && event.kind === 'audio' && event.audio) {
    audioBase64 = event.audio;
    sampleRate = Number(event.sample_rate || 24000);
  } else if (event.type === 'input.append' && event.input?.audio_base64) {
    audioBase64 = event.input.audio_base64;
    sampleRate = Number(event.input.sample_rate || 16000);
  }
  if (!audioBase64) return '';
  try {
    const wavBase64 = float32Base64ToWavBase64(audioBase64, sampleRate);
    return `<div class="stream-media"><audio controls src="data:audio/wav;base64,${wavBase64}"></audio></div>`;
  } catch (err) {
    return `<div class="stream-media error">audio decode failed: ${escapeHtml(err?.message || err)}</div>`;
  }
}

function applyStreamFilter() {
  if (!el.streamFeed) return;
  const selected = el.streamFilter?.value || 'all';
  for (const row of el.streamFeed.querySelectorAll('.stream-row')) {
    const category = row.dataset.category || '';
    const direction = row.classList.contains('tx') ? 'tx' : 'rx';
    const visible = selected === 'all' || selected === category || selected === direction;
    row.classList.toggle('hidden', !visible);
  }
}

function streamEventCategory(direction, event) {
  const type = event.type || '';
  if (type === 'error' || type.endsWith('.error')) return 'error';
  if (type.startsWith('session.')) return 'session';
  if (type.startsWith('input.')) return 'input';
  if (type === 'response.output.sp_tokens') return 'sp';
  if (type === 'response.debug') return 'debug';
  if (type === 'response.output.delta') return 'output';
  if (type.startsWith('response.think')) return 'think';
  if (type.startsWith('response.tool_call')) return 'tool_call';
  if (type === 'response.tool_result') return 'tool_result';
  return direction;
}

function summarizeStreamEvent(event) {
  const type = event.type || '';
  if (type === 'input.append') {
    const audio = event.input?.audio_base64 || '';
    return `audio_base64=${audio.length} chars · sample_rate=${event.input?.sample_rate || '-'}`;
  }
  if (type === 'session.init') {
    const payload = event.payload || {};
    return [
      `mode=${payload.mode || '-'}`,
      `fc_duplex=${Boolean(payload.fc_duplex)}`,
      `tools=${(payload.tools || []).map((tool) => tool?.function?.name || '?').join(',') || '-'}`,
      `generate_audio=${Boolean(payload.generate_audio)}`,
    ].join(' · ');
  }
  if (type === 'response.output.delta') {
    if (event.kind === 'audio') return `kind=audio · audio=${String(event.audio || '').length} chars · sample_rate=${event.sample_rate || '-'}`;
    if (event.kind === 'text') return `kind=text · ${event.text || ''}`;
    if (event.kind === 'non_spoken') return `kind=non_spoken · ${event.text || (event.token_strs || []).join('')}`;
    return `kind=${event.kind || '-'}`;
  }
  if (type.startsWith('response.tool_call')) {
    return `tool_call_id=${event.tool_call_id || '-'} · ${event.delta || formatRawForSummary(event.raw) || ''}`;
  }
  if (type === 'response.tool_result') {
    const result = event.result || {};
    return `tool_call_id=${event.tool_call_id || '-'} · query=${result.query || '-'} · error=${result.error || '-'}`;
  }
  if (type.startsWith('response.think')) {
    return event.delta || '';
  }
  if (type === 'response.output.sp_tokens') {
    return `token=${event.token || '-'}`;
  }
  if (type === 'response.debug') {
    const debug = event.debug || {};
    return `used=${formatMaybeNumber(debug.used)} · max/1s=${formatMaybeNumber(debug.estimated_max_budget_1s)}`;
  }
  if (type === 'session.created') {
    return `session_id=${event.session_id || '-'} · mode=${event.mode || '-'}`;
  }
  if (type === 'session.queued' || type === 'session.queue_update') {
    return `position=${event.position ?? '-'} · eta=${event.estimated_wait_s ?? '-'}s`;
  }
  if (type === 'session.close' || type === 'session.closed') {
    return `reason=${event.reason || '-'}`;
  }
  return compactJson(event);
}

function formatRawForSummary(raw) {
  if (!raw) return '';
  if (raw.error) return `error=${raw.error}`;
  return `${raw.name || ''} ${raw.arguments || ''}`.trim();
}

function compactJson(value) {
  try { return JSON.stringify(value); } catch (_) { return String(value); }
}

function clampInt(value, min, max, fallback) {
  const parsed = Number.parseInt(String(value ?? ''), 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function prettyJson(value) {
  try { return JSON.stringify(value, null, 2); } catch (_) { return String(value); }
}

function scrollNsToBottom() {
  const scroller = el.nsStream.parentElement;
  if (scroller) requestAnimationFrame(() => { scroller.scrollTop = scroller.scrollHeight; });
}

function parseArguments(value) {
  if (!value) return {};
  if (typeof value === 'object') return value;
  try { return JSON.parse(value); } catch (_) { return {}; }
}

function formatToolCall(raw) {
  if (!raw || raw.error) return raw?.error || '';
  return `<function name="${raw.name}">${raw.arguments || ''}</function>`;
}

function placeholderImageDataUrl(label) {
  const safeLabel = escapeHtml(label || 'object');
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="640" height="420" viewBox="0 0 640 420">
      <rect width="640" height="420" fill="#0f172a"/>
      <rect x="42" y="42" width="556" height="336" rx="18" fill="#1e293b" stroke="#38bdf8" stroke-width="3"/>
      <text x="320" y="200" fill="#e2e8f0" font-family="sans-serif" font-size="36" font-weight="700" text-anchor="middle">${safeLabel}</text>
      <text x="320" y="250" fill="#94a3b8" font-family="sans-serif" font-size="20" text-anchor="middle">display_object_on_board</text>
    </svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function float32Base64ToWavBase64(audioBase64, sampleRate) {
  const binary = atob(audioBase64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const samples = new Float32Array(bytes.buffer);
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  const wav = new ArrayBuffer(44 + pcm.byteLength);
  const view = new DataView(wav);
  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + pcm.byteLength, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, 'data');
  view.setUint32(40, pcm.byteLength, true);
  new Uint8Array(wav, 44).set(new Uint8Array(pcm.buffer));
  let out = '';
  const wavBytes = new Uint8Array(wav);
  for (let i = 0; i < wavBytes.length; i += 0x8000) {
    out += String.fromCharCode(...wavBytes.subarray(i, i + 0x8000));
  }
  return btoa(out);
}

function writeAscii(view, offset, value) {
  for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function formatMaybeNumber(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'number' && Number.isFinite(value)) return String(Math.round(value));
  return String(value);
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}
