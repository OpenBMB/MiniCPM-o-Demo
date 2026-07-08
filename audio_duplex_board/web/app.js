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
    setStatus('Listening · 先说指令（等下我讲 XX，提到的你放画板上），等 AI 回一句后再自由说');
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
  nonSpokenBlocks.set(blockId, { kind, streamingPieces: [], fullText: null, closed: false, node: wrap });
  scrollNsToBottom();
}

// 真正滚动的容器是 `.scroll-body`（`#nonSpokenStream` 的父元素）——
// index.html 里可滚动区域套在外层，nonSpokenStream 只是内容器（overflow: hidden）。
// 之前 el.nsStream.scrollTop = ... 是 no-op（自身没有 overflow）。这里统一
// 用 parentElement 才是真正的 scroller；rAF 里执行确保 layout 已算好新高度，
// 否则新 block/delta 还没提交到 DOM 时 scrollHeight 还是旧值。
function scrollNsToBottom() {
  const scroller = el.nsStream && el.nsStream.parentElement;
  if (!scroller) return;
  requestAnimationFrame(() => {
    scroller.scrollTop = scroller.scrollHeight;
  });
}

function appendNonSpokenDelta(event) {
  const blockId = event.block_id;
  if (!blockId) return;
  const block = nonSpokenBlocks.get(blockId);
  if (!block) return;
  // 原始需求："streaming 里边就去呈现它的原始的那个内容就行了，你不需要
  // 做 XML 解析。包括 <tool_call> 和 </tool_call> 你也是都是带上的。"
  // step_text 是 SDK 每一步正常 BPE decode 出的增量文本，天然包含所有 XML
  // 标签和特殊 marker 的可读形式，是唯一的 streaming 数据源（协议里已经
  // 删掉了 token_strs 字段——它是 `convert_ids_to_tokens` 反查结果，对
  // byte-level BPE 不是合法可读文本，普通内容 token 100% 乱码）。
  const raw = event.step_text || '';
  if (raw) block.streamingPieces.push(raw);
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
  // 用户强调："必须保证 Non-spoken 通道一直处于最下方，这样总能看见最新
  // block 的 streaming 结果"。每次 delta 都强制 scroll 到底。
  scrollNsToBottom();
}

function closeNonSpokenBlock(event) {
  const blockId = event.block_id;
  if (!blockId) return;
  const block = nonSpokenBlocks.get(blockId);
  if (!block) return;
  block.closed = true;
  block.fullText = event.full_text || null;
  const finalKind = (event.block_kind && event.block_kind !== 'unknown') ? event.block_kind : block.kind;
  // upgrade kind visual if we didn't know it during streaming
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
  // 原始需求："streaming 里边就去呈现它的原始的那个内容就行了，你不需要
  // 做 XML 解析。包括 <tool_call> 和 </tool_call> 你也是都是带上的。"
  // SDK 的 per-step BPE decode 不会带 <think> / <tool_call> tag（那是
  // special token id，decode 出来就是空），只能在 client 端 wrap 出来。
  // close 的时候用 finalKind 把 streaming buffer 包一层。
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
      // 原始需求：tool_call 非法 / parser 拿不到 full 时，full 层不显示。
      const section = full.closest('section.ns-layer.full');
      if (section) section.style.display = 'none';
    }
  }
  // 收尾也 scroll 到底，防止 close 事件比最后一 delta 晚到时被卡在上面
  scrollNsToBottom();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function renderBoard() {
  if (!state.cards.length) {
    el.board.classList.add('empty-board');
    el.board.innerHTML = `<div class="empty-state">
      <strong>先说一句指令</strong>（约 6-8 秒），比如：<br />
      <em>「我等下讲讲农场里的动物，提到的你就帮我放到画板上。」</em><br />
      <em>「等下我讲桌上的东西，提到的你就帮我展示出来。」</em><br />
      等 AI 应一声（"好，你说到就放到画板上"）之后，再自由描述具体物体。<br />
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

function appendTimeline(text) {
  const item = document.createElement('div');
  item.className = 'timeline-row';
  item.textContent = text;
  el.timeline.appendChild(item);
}

// AI Speech 字符级 "打字机" streaming：SDK 每 unit 一次性给一段文字，我们
// 按 audio 播放时长把字符匀速铺出来。这样视觉字符与耳朵听到的音节大致
// 对齐 —— 用户明确要求"AI speech 也是同样"，也要有 streaming 呈现。
//
// 关键 trick：每次 enqueueSpeech(text, audioMs) 时，根据剩余队列 + 这一
// 批的 audioMs 动态计算 step delay。如果一批文字对应 1000ms 音频、有 5
// 个字，那么每字 ~200ms 显示 —— 用户看到字符和 AI 说话节奏对得上，不是
// "burst 300ms、静等 700ms"那种明显的 burst 感。
//
// 不像 think / tool_call 需要"streaming 层 + full 层"两层，AI spoken 的
// spoken_text 就是最终 BPE decode，字符流本身就是最终结果 —— 只有一层
// streaming（原始需求："AI spoken 本身没有 full，直接就是 streaming 就好
// 了"）。
const speechQueue = [];            // [{ch, stepMs}]
let speechDrainer = null;

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
  // 首字立即出，之后按每字自己 stepMs 走
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
  // 同 scrollNsToBottom 的问题：真正的 scroller 是 aiSpeech 的父元素
  // `.scroll-body`。给 aiSpeech 本体设 scrollTop 是 no-op。
  const speechScroller = el.aiSpeech.parentElement;
  if (speechScroller) {
    requestAnimationFrame(() => {
      speechScroller.scrollTop = speechScroller.scrollHeight;
    });
  }
}

// text: 这一 unit 的 spoken 文本
// audioMs: 这一 unit TTS 出来的音频时长（毫秒），用来 pace 字符出现速度
function enqueueSpeech(text, audioMs) {
  if (!text) return;
  const chars = [...text];
  // 有音频时按音频时长匀速铺；没音频（罕见）用一个稳的兜底 80ms。
  // 队列已经有堆积则加速消化，避免视觉比音频滞后太多。
  const pendingBudget = Math.max(0, audioMs || chars.length * 80);
  const backlog = speechQueue.length;
  const stepMs = Math.max(15, Math.round(pendingBudget / Math.max(1, chars.length + backlog * 0.5)));
  for (const ch of chars) speechQueue.push({ ch, stepMs });
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

  if (event.text) {
    // 从 audio_float32_base64 反推这一 unit 的音频毫秒数，作为字符打字机
    // 的 pacing 依据，让字符出现节奏和音频播放节奏对齐。
    const b64 = p.audio_float32_base64 || '';
    const sr = p.audio_sample_rate || 24000;
    const audioMs = b64 ? ((b64.length * 3 / 4) / 4) / sr * 1000 : 0;
    enqueueSpeech(event.text, audioMs);
  }

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

// Mic level display uses a LUFS-like unit (dBFS) instead of raw RMS.
//
// 换算依据（`scripts/diagnostics/lufs_survey.py` 采样训练数据）：
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
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}
