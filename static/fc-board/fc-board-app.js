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
let generationStepCount = 0;
let latestCheckpoint = null;
let resumeInProgress = false;
// Mic level display uses a LUFS-like unit (dBFS) instead of raw RMS.
//
// 换算依据（`scripts/diagnostics/lufs_survey.py` 采样训练数据，见 o45-fc 分支）：
//   training set: RMS mean = 0.069  ↔  LUFS mean = -23.0
//   公式 20·log10(RMS) 在这个 RMS 上算得 -23.2 —— 跟真 LUFS 只差 0.2 dB
// 结论：对语音信号，`20·log10(rms)` 就够当 LUFS 近似值显示了（K-weighting
// 对语音频谱几乎无影响，真 LUFS 需要 400ms 窗 + 门控，UI 场景不必如此严格）。
//
// bar 映射范围：-50 dB → -10 dB（覆盖训练分布 -23 附近 ±15 dB）
// target zone：-28 dB → -18 dB（训练集 5th → 95th percentile RMS 换算）
const MIC_METER_MIN_DB = -50;    // 静音底
const MIC_METER_MAX_DB = -10;    // 满格
const MIC_TARGET_LO_DB = -28;    // 训练集 p5 附近
const MIC_TARGET_HI_DB = -18;    // 训练集 p95 附近

function rmsToDb(rms) {
  if (!(rms > 0)) return MIC_METER_MIN_DB;
  const db = 20 * Math.log10(rms);
  return Math.max(MIC_METER_MIN_DB, db);   // 静音底
}

let micPeakDbSinceStart = MIC_METER_MIN_DB;
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
  generationSteps: document.getElementById('generationSteps'),
  resumeCheckpoint: document.getElementById('resumeCheckpoint'),
  resumeSummary: document.getElementById('resumeSummary'),
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
  el.kvCkpt.textContent = 'No checkpoint';
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
    micProvider = createMicProvider(liveClient);
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

el.resumeCheckpoint?.addEventListener('click', async () => {
  if (
    resumeInProgress
    || !liveClient
    || latestCheckpoint?.resume?.status !== 'available'
  ) return;

  resumeInProgress = true;
  updateControlState();
  const throughUnitIndex = Number(latestCheckpoint.unit_index);
  try {
    const resumePayload = liveClient.buildResumePayload(throughUnitIndex);
    if (micProvider) {
      micProvider.stop();
      micProvider = null;
    }
    const disconnectedClient = liveClient;
    liveClient = null;
    disconnectedClient.disconnect('demo_resume_test');
    setMicLiveState('starting');
    setWsState('reconnecting');
    setStatus(`Reconnecting from Unit ${throughUnitIndex} checkpoint…`);
    await sleep(600);

    liveClient = await resumeRealtimeSession(resumePayload);
    micProvider = createMicProvider(liveClient);
    await micProvider.start();
    setMicLiveState('live');
    setWsState('connected');
    setStatus(`Resume succeeded · continuing after Unit ${throughUnitIndex}`);
  } catch (err) {
    console.error('[FC resume]', err);
    setMicLiveState('error');
    setWsState('closed');
    setStatus(`Resume failed: ${err.message}`);
  } finally {
    resumeInProgress = false;
    updateControlState();
  }
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
  const client = await createConnectedClient();
  client.initSession(buildSessionInitPayload());
  return client;
}

async function createConnectedClient(onControlEvent = null) {
  const client = new FcRealtimeClient({
    onEvent: (event) => {
      applyApiEvent(event);
      onControlEvent?.(event);
    },
    onSend: (message) => appendStreamEvent('tx', message),
    onStatus: setStatus,
  });
  setStatus('Connecting to /v1/realtime...');
  await client.connect();
  setWsState('connected');
  return client;
}

async function resumeRealtimeSession(payload) {
  let resolveResume;
  let rejectResume;
  const resumed = new Promise((resolve, reject) => {
    resolveResume = resolve;
    rejectResume = reject;
  });
  const client = await createConnectedClient((event) => {
    if (event.type === 'session.resumed') resolveResume(event);
    if (event.type === 'session.resume.failed') {
      rejectResume(new Error(`${event.code || 'resume_failed'}: ${event.message || ''}`));
    }
  });
  client.resumeSession(payload);
  const timeout = new Promise((_, reject) => {
    setTimeout(() => reject(new Error('session.resume timed out')), 120000);
  });
  await Promise.race([resumed, timeout]);
  return client;
}

function createMicProvider(client) {
  return new LiveMicProvider({
    onChunk: (chunk) => {
      client.appendAudio({ audioBase64: float32ToBase64(chunk), sampleRate: 16000 });
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
    case 'session.resumed':
      setStatus(`Session resumed after Unit ${event.through_unit_index}`);
      return;
    case 'session.resume.failed':
      setStatus(`Resume failed: ${event.code || 'unknown'}`);
      return;
    case 'response.warning':
      setStatus(`Warning: ${event.message || event.code || 'unknown'}`);
      return;
    case 'session.closed':
      setWsState('closed');
      releaseMicLive('closed');
      setStatus(`Session closed: ${event.reason || 'closed'}`);
      return;
    case 'response.unit.started':
      return;
    case 'response.unit.committed':
      handleUnitCheckpoint(event);
      return;
    case 'response.spoken.delta':
      handleSpokenDelta(event);
      return;
    case 'response.spoken.end':
      handleSpokenEnd(event);
      return;
    case 'response.think.begin':
      activeNonSpokenBlockId = `think:${++blockSeq}`;
      beginNonSpokenBlock({ block_id: activeNonSpokenBlockId, block_kind: 'think' });
      return;
    case 'response.think.delta':
      appendSemanticSteps({
        block_id: activeNonSpokenBlockId,
        unit_index: event.unit_index,
        steps: event.steps,
      });
      return;
    case 'response.think.end':
      closeNonSpokenBlock({
        block_id: activeNonSpokenBlockId,
        block_kind: 'think',
        full_text: event.full_text,
      });
      activeNonSpokenBlockId = null;
      return;
    case 'response.tool_call.begin':
      toolCallBlocks.set(event.tool_call_id, `tool_call:${event.tool_call_id}`);
      activeNonSpokenBlockId = toolCallBlocks.get(event.tool_call_id);
      beginNonSpokenBlock({ block_id: activeNonSpokenBlockId, block_kind: 'tool_call' });
      return;
    case 'response.tool_call.delta':
      appendSemanticSteps({
        block_id: toolCallBlocks.get(event.tool_call_id),
        unit_index: event.unit_index,
        steps: event.steps,
      });
      return;
    case 'response.tool_call.done':
      handleToolCallDone(event);
      return;
    case 'response.debug':
      handleDebugEvent(event);
      return;
    default:
      return;
  }
}

function textFromSteps(steps) {
  return (Array.isArray(steps) ? steps : [])
    .filter((step) => step?.kind === 'text')
    .map((step) => step.text || '')
    .join('');
}

function handleSpokenDelta(event) {
  const steps = Array.isArray(event.steps) ? event.steps : [];
  generationStepCount += steps.length;
  const text = textFromSteps(steps);
  if (text) enqueueSpeech(text, 0);
  if (event.audio) {
    handleSpokenOutput({
      isListen: false,
      isSpeaking: true,
      audioBase64: event.audio,
      sampleRate: event.sample_rate || 24000,
    });
  }
  updateStats();
}

function handleSpokenEnd(event) {
  if (event.reason === 'listen' || event.reason === 'turn_eos') {
    if (audioPlayerReady && audioPlayer.turnActive) audioPlayer.endTurn();
  }
}

function handleUnitCheckpoint(event) {
  latestCheckpoint = event;
  const unitIndex = Number(event.unit_index);
  const resume = event.resume || {};
  const available = resume.status === 'available';
  const detail = available
    ? `Unit ${unitIndex} · available`
    : `Unit ${unitIndex} · unavailable (${resume.reason || 'unknown'})`;
  if (el.kvCkpt) el.kvCkpt.textContent = detail;
  if (el.resumeSummary) {
    el.resumeSummary.textContent = available
      ? `Unit ${unitIndex} can reconnect without server state`
      : detail;
    el.resumeSummary.classList.toggle('available', available);
    el.resumeSummary.classList.toggle('failed', !available);
  }
  updateControlState();
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

function handleToolCallDone(event) {
  const blockId = toolCallBlocks.get(event.tool_call_id);
  const call = event.call || {};
  closeNonSpokenBlock({
    block_id: blockId,
    block_kind: 'tool_call',
    full_text: event.full_text || event.error || '',
  });
  activeNonSpokenBlockId = null;
  if (call.name !== 'display_object_on_board' || event.error) return;
  const args = parseArguments(call.arguments);
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
  let payload = content || {
    status: 'displayed',
    name: query,
    reason: '已在画板显示该对象。',
  };
  if (typeof payload === 'string') {
    try { payload = JSON.parse(payload); } catch (_) { payload = { text: payload }; }
  }
  liveClient.sendToolResult({
    toolCallId,
    content: payload,
  });
}

function cardIdFor(toolCallId) {
  return `card:${toolCallId || 'pending'}`;
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
  nonSpokenBlocks.set(blockId, {
    kind,
    streamingPieces: [],
    segments: new Map(),
    fullText: null,
    closed: false,
    node: wrap,
  });
  scrollNsToBottom();
}

function appendSemanticSteps({ block_id: blockId, unit_index: unitIndex, steps }) {
  if (!blockId) return;
  if (!nonSpokenBlocks.has(blockId)) {
    beginNonSpokenBlock({ block_id: blockId, block_kind: 'unknown' });
  }
  const block = nonSpokenBlocks.get(blockId);
  const normalizedSteps = Array.isArray(steps) ? steps : [];
  generationStepCount += normalizedSteps.length;
  const text = textFromSteps(normalizedSteps);
  const key = String(unitIndex);
  let segment = block.segments.get(key);
  if (!segment) {
    const streamingTarget = block.node.querySelector('[data-role="streaming"]');
    const node = document.createElement('span');
    const tone = Math.abs(Number(unitIndex) || 0) % 4;
    node.className = `unit-segment tone-${tone}`;
    node.dataset.unitIndex = key;
    node.title = `Unit ${key}`;
    node.innerHTML = `
      <span class="unit-segment-label">U${escapeHtml(key)}</span>
      <span class="unit-segment-text"></span>
    `;
    streamingTarget?.appendChild(node);
    segment = { pieces: [], node };
    block.segments.set(key, segment);
  }
  if (text) {
    segment.pieces.push(text);
    block.streamingPieces.push(text);
    const textNode = segment.node.querySelector('.unit-segment-text');
    if (textNode) textNode.textContent = segment.pieces.join('');
  }
  updateStats();
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
  if (
    block.fullText !== null
    && block.streamingPieces.join('') !== block.fullText
  ) {
    block.node.classList.add('stream-mismatch');
    if (statusTag) statusTag.textContent = 'stream/full mismatch';
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
    el.board.innerHTML = `<div class="empty-state">
      <strong>先说一句指令</strong>（约 6-8 秒），比如：<br />
      <em>「我等下讲讲故事里出现的动物，提到的你就帮我放到画板上。」</em><br />
      <em>「等下我描述房间里的东西，提到的你就帮我展示出来。」</em><br />
      等 AI 应一声之后，再自由描述具体物体。<br />
      <span style="color:#999;font-size:11px">句式仿训练数据：\`等下我/我等下…提到 XX 你就 放到画板上\`</span>
    </div>`;
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
  generationStepCount = 0;
  latestCheckpoint = null;
  if (el.kvCkpt) el.kvCkpt.textContent = 'No checkpoint';
  if (el.resumeSummary) {
    el.resumeSummary.textContent = 'No resumable checkpoint';
    el.resumeSummary.classList.remove('available', 'failed');
  }
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
  if (el.resumeCheckpoint) {
    el.resumeCheckpoint.disabled = (
      resumeInProgress
      || micLiveState !== 'live'
      || !liveClient
      || latestCheckpoint?.resume?.status !== 'available'
    );
  }
}

function setWsState(state) {
  if (!el.wsState) return;
  el.wsState.textContent = state;
  el.wsState.classList.remove('connected', 'closed');
  if (state === 'connected') el.wsState.classList.add('connected');
  else if (state === 'closed' || state === 'idle') el.wsState.classList.add('closed');
}

function updateMicLevel(level) {
  // `level` 是 live-mic-provider 传来的原始 RMS ∈ [0, 1]
  const db = rmsToDb(level);
  const clamped = Math.max(0, Math.min(1, (db - MIC_METER_MIN_DB) / (MIC_METER_MAX_DB - MIC_METER_MIN_DB)));
  if (el.micLevelBar) el.micLevelBar.style.width = `${Math.round(clamped * 100)}%`;
  // 用户反馈过"模型不响应"往往是麦克风采到了静音（Chrome noiseSuppression
  // 太狠 / 权限没给 / 硬件哑了）。session peak 让用户一眼就能判断音频真的
  // 有没有进来：peak < -50 dB 基本可以断定是静音。
  if (db > micPeakDbSinceStart) micPeakDbSinceStart = db;
  if (el.micLevelText) {
    const inTarget = (db >= MIC_TARGET_LO_DB && db <= MIC_TARGET_HI_DB) ? '✓' : ' ';
    el.micLevelText.textContent = `${db.toFixed(1)} dB ${inTarget}  (peak ${micPeakDbSinceStart.toFixed(1)}, target ${MIC_TARGET_LO_DB}~${MIC_TARGET_HI_DB} ≈ -23 LUFS)`;
  }
}

function resetMicPeak() {
  micPeakDbSinceStart = MIC_METER_MIN_DB;
}

function updateStats() {
  el.sentChunks.textContent = String(sentChunkCount);
  el.aiAudioCount.textContent = String(aiAudioCount);
  if (el.generationSteps) {
    el.generationSteps.textContent = String(generationStepCount);
  }
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
  if (event.type === 'response.spoken.delta' && event.audio) {
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
  if (type === 'response.warning') return 'warning';
  if (type.startsWith('session.')) return 'session';
  if (type.startsWith('input.')) return 'input';
  if (type.startsWith('response.unit.')) return 'unit';
  if (type.startsWith('response.spoken.')) return 'spoken';
  if (type === 'response.debug') return 'debug';
  if (type.startsWith('response.think')) return 'think';
  if (type.startsWith('response.tool_call')) return 'tool_call';
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
  if (type.endsWith('.delta')) {
    const steps = Array.isArray(event.steps) ? event.steps : [];
    return `unit=${event.unit_index ?? '-'} · text=${textFromSteps(steps)} · steps=${steps.length} · audio=${String(event.audio || '').length}`;
  }
  if (type === 'response.unit.started') {
    return `unit=${event.unit_index ?? '-'} · input_id=${event.input_id || '-'} · tool_events=${(event.tool_events || []).length}`;
  }
  if (type === 'response.unit.committed') {
    return `unit=${event.unit_index ?? '-'} · end=${event.non_spoken_end || '-'} · resume=${event.resume?.status || '-'}${event.resume?.reason ? ` (${event.resume.reason})` : ''}`;
  }
  if (type === 'response.warning') {
    return `unit=${event.unit_index ?? '-'} · code=${event.code || '-'} · ${event.message || ''}`;
  }
  if (type.startsWith('response.tool_call')) {
    return `tool_call_id=${event.tool_call_id || '-'} · unit=${event.unit_index ?? '-'} · ${event.full_text || event.error || ''}`;
  }
  if (type.startsWith('response.think')) {
    return `unit=${event.unit_index ?? '-'} · ${event.full_text || textFromSteps(event.steps)}`;
  }
  if (type === 'response.debug') {
    const debug = event.debug || {};
    return `used=${formatMaybeNumber(debug.used)} · max/1s=${formatMaybeNumber(debug.estimated_max_budget_1s)}`;
  }
  if (type === 'session.created') {
    return `session_id=${event.session_id || '-'} · mode=${event.mode || '-'}`;
  }
  if (type === 'session.resumed') {
    return `session_id=${event.session_id || '-'} · through_unit=${event.through_unit_index ?? '-'} · next_unit=${event.next_unit_index ?? '-'}`;
  }
  if (type === 'session.resume.failed') {
    return `code=${event.code || '-'} · unit=${event.unit_index ?? '-'} · ${event.message || ''}`;
  }
  if (type === 'session.queued' || type === 'session.queue_update') {
    return `position=${event.position ?? '-'} · eta=${event.estimated_wait_s ?? '-'}s`;
  }
  if (type === 'session.close' || type === 'session.closed') {
    return `reason=${event.reason || '-'}`;
  }
  return compactJson(event);
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
