import { BoardState } from './board-state.js';
import { decodeFileToChunks, float32ToBase64 } from './file-audio-provider.js';
import { LiveBoardClient } from './live-board-client.js';
import { LiveMicProvider } from './live-mic-provider.js';
import { fetchDefaults, replayCase } from './replay-client.js';
// Demo's gapless pre-scheduled audio player (verbatim copy from
// static/duplex/lib/audio-player.js). Consumes raw Float32 base64 chunks
// from `spoken_final.payload.audio_float32_base64`.
import { AudioPlayer } from './audio-player.js';

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

// Populate header mode badge + session settings panel from /api/defaults.
// 之前有用户反馈 badge 一直停在 "(loading)"，通常有两种原因：
//   (1) /api/defaults 拉取失败但没有可见反馈；
//   (2) 浏览器缓存了旧的 app.js，根本没跑 IIFE。
// 因此这里加入：超时（5s）、可见错误 + 点击重试、每次调用都写状态到 badge。
async function loadDefaultsIntoUi({ isRetry = false } = {}) {
  if (!el.modeBadge) return;
  el.modeBadge.textContent = isRetry ? 'retrying…' : 'loading…';
  el.modeBadge.className = 'mode-tag mode-loading';
  el.modeBadge.title = '';
  try {
    const fetchP = fetchDefaults();
    const timeoutP = new Promise((_, rej) =>
      setTimeout(() => rej(new Error('timeout after 5s')), 5000),
    );
    cachedDefaults = await Promise.race([fetchP, timeoutP]);
    const defaults = cachedDefaults;
    if (defaults.use_mock_view) {
      el.modeBadge.textContent = 'MOCK';
      el.modeBadge.className = 'mode-tag mode-mock';
    } else {
      el.modeBadge.textContent = 'REAL MODEL';
      el.modeBadge.className = 'mode-tag mode-real';
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
    if (el.systemPrompt) {
      el.systemPrompt.value = defaults.default_system_prompt || '';
    }
    if (el.refAudioPath) {
      el.refAudioPath.value = defaults.default_ref_audio_path || '';
    }
  } catch (err) {
    el.modeBadge.textContent = `error · click to retry`;
    el.modeBadge.className = 'mode-tag mode-error';
    el.modeBadge.title = String(err && err.message ? err.message : err);
    console.error('[modeBadge] fetchDefaults failed:', err);
  }
}
// 点 badge 触发重试，方便前端卡住时用户自己恢复。
el.modeBadge?.addEventListener('click', () => loadDefaultsIntoUi({ isRetry: true }));
loadDefaultsIntoUi();

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
  resetMicPeak();
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
      handleSpokenFinal(event);
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

// AI Speech 用一个字符级别的 "打字机" 队列做 streaming 呈现：spoken_final
// 每个 unit 一次性给一段文字（SDK 一个 spoken slot 是原子的，没有 sub-unit
// 逐 token 事件），我们把它拆成字符压进 speechQueue，然后 speechDrainer
// 每 speechStepMs 弹一个字符出来 append 到当前 speech row。这样视觉上和
// think / tool_call 面板的 token 流一致，也和 AI 语音播放节奏对得上（
// 每 unit ~1s 音频 ≈ 4-8 个汉字 → 每字 ~140ms 显示，够慢能看清、又足够
// 快不会积压）。
//
// 不像 think / tool_call 需要"streaming 层 + full 层"两层（因为 BPE
// merged 后有可能和逐 token 拼接不完全一致），AI spoken 的 spoken_text
// 就是最终 BPE decode，字符流直接就是最终结果，只有 streaming 层。
const speechQueue = [];
let speechDrainer = null;
const SPEECH_STEP_MS = 60;   // 每字符间隔；audio 一般 6-10 字/s

function pumpSpeechQueue() {
  if (speechDrainer) return;
  const step = () => {
    const item = speechQueue.shift();
    if (!item) {
      speechDrainer = null;
      return;
    }
    appendSpeechChar(item);
    // 队列越长踩越紧，避免比音频落后太多
    const dyn = speechQueue.length > 30 ? 20
              : speechQueue.length > 15 ? 35
              : SPEECH_STEP_MS;
    speechDrainer = setTimeout(step, dyn);
  };
  speechDrainer = setTimeout(step, 0);
}

function appendSpeechChar(ch) {
  const placeholder = el.aiSpeech.querySelector('.placeholder');
  if (placeholder) placeholder.remove();
  // 同一 turn 内所有字符都续在一行；跨 turn（>=1.5s 无字符入队）换行。
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
  el.aiSpeech.scrollTop = el.aiSpeech.scrollHeight;
}

function enqueueSpeech(text) {
  if (!text) return;
  for (const ch of text) speechQueue.push(ch);
  pumpSpeechQueue();
}

function clearViews() {
  state.cards = [];
  nonSpokenBlocks.clear();
  el.timeline.innerHTML = '';
  el.board.innerHTML = '';
  renderBoard();
  // 清空 AI Speech 队列，防止上一次会话的残余字符还在滴。
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

function setStatus(text) {
  el.status.textContent = text;
}

// spoken_final 事件到达时的核心处理。**照抄** demo `static/duplex/lib/
// realtime-session.js` 的 turn-based 状态机（_handleListen + _handleSpeak）：
//
//   - is_listen=true：模型这一 unit 不说话。**必须** endTurn（如 turnActive），
//     让 AudioPlayer 优雅收尾 pending chunks，_startAheadMonitor 停掉。这样
//     下一次真正 speak 到来时 beginTurn 打断的只是"已经结束的 turn"，不会
//     截断正在播的 AI 语音。之前只在 spoken_turn_eos 触发 endTurn，模型不
//     吐 eos 就永远不 end，导致后续 beginTurn 拿 _stopAllSources() 硬切上
//     一段的尾巴——就是用户看到的"结尾有截断"。
//
//   - is_speaking=true + audio：turnActive 就当 continuous append（同 turn
//     内按 _nextTime 无缝排队；这是 AudioPlayer 里 _scheduleChunk 的 gapless
//     语义），否则 beginTurn 开新 turn 再 playChunk。同一 turn 内绝不 begin
//     多次，避免 _stopAllSources() 制造重叠 / 截断。
//
// 之前把 begin/play/end 交给"event.payload.spoken_turn_eos"来控制 turn 边界
// 是错的：spoken_turn_eos 是模型 EOS，不一定每次都吐；而且 turn 边界的
// **权威信号是 is_listen** 而不是 EOS。
function handleSpokenFinal(event) {
  const p = event.payload || {};
  const isListen = !!p.is_listen;
  const isSpeaking = !!p.is_speaking;

  if (event.text) enqueueSpeech(event.text);

  if (audioPlayerReady) {
    if (isListen) {
      if (audioPlayer.turnActive) {
        console.log(`[Turn] unit=${event.unit_index} listen → endTurn (turnIdx=${audioPlayer.turnIdx})`);
        audioPlayer.endTurn();
      }
    } else if (isSpeaking && p.audio_float32_base64) {
      const beganTurn = !audioPlayer.turnActive;
      if (beganTurn) {
        audioPlayer.beginTurn();
        console.log(`[Turn] unit=${event.unit_index} speak → beginTurn (turnIdx=${audioPlayer.turnIdx})`);
      }
      const b64Len = p.audio_float32_base64.length;
      const durMs = ((b64Len * 3 / 4) / 4) / (p.audio_sample_rate || 24000) * 1000;
      console.log(`[Turn] unit=${event.unit_index} playChunk dur=${durMs.toFixed(0)}ms turnIdx=${audioPlayer.turnIdx} ahead=${audioPlayer.lastAheadMs.toFixed(0)}ms`);
      audioPlayer.playChunk(p.audio_float32_base64, performance.now());
    }
    // 保险：显式 EOS 也 endTurn（个别 ckpt 会在最后一 speak unit 直接吐
    // eos 同时不切回 listen，也应该收尾）。
    if (p.spoken_turn_eos && audioPlayer.turnActive) {
      console.log(`[Turn] unit=${event.unit_index} spoken_turn_eos → endTurn`);
      audioPlayer.endTurn();
    }
  }

  // 不再把 audio_wav_base64 加到调试面板：AudioPlayer 已经在播了，浏览器
  // 里再挂一串 <audio> 只会占空间、混淆听感，如果用户误点或浏览器 autoplay
  // 甚至可能造成"重复播放"的观感。真需要看单 unit 波形可以打开 debug
  // drawer 里的 event timeline 找。
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

let micPeakSinceStart = 0;
function updateMicLevel(level) {
  const clamped = Math.max(0, Math.min(1, level / 0.08));
  if (el.micLevelBar) el.micLevelBar.style.width = `${Math.round(clamped * 100)}%`;
  // 用户反馈过"模型不响应"往往是麦克风采到了静音（Chrome noiseSuppression
  // 太狠 / 权限没给 / 硬件哑了）。多加一个 session peak 让用户一眼就能判断
  // 自己的音频有没有真的进来：peak < 0.01 基本可以断定是静音。
  if (level > micPeakSinceStart) micPeakSinceStart = level;
  if (el.micLevelText) {
    el.micLevelText.textContent = `${level.toFixed(2)}  (peak ${micPeakSinceStart.toFixed(2)})`;
  }
}
function resetMicPeak() {
  micPeakSinceStart = 0;
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
