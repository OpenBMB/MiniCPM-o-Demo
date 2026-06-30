import { BoardState } from './board-state.js';
import { decodeFileToChunks, float32ToBase64 } from './file-audio-provider.js';
import { LiveBoardClient } from './live-board-client.js';
import { fetchDefaults, replayCase } from './replay-client.js';

const state = new BoardState({ maxCards: 10 });
const el = {
  status: document.getElementById('status'),
  casePath: document.getElementById('casePath'),
  loadDefaults: document.getElementById('loadDefaults'),
  runReplay: document.getElementById('runReplay'),
  audioFileInput: document.getElementById('audioFileInput'),
  runFileReplay: document.getElementById('runFileReplay'),
  userAudioPlayer: document.getElementById('userAudioPlayer'),
  aiAudioList: document.getElementById('aiAudioList'),
  padBeforeSec: document.getElementById('padBeforeSec'),
  padAfterSec: document.getElementById('padAfterSec'),
  generateAudio: document.getElementById('generateAudio'),
  timeline: document.getElementById('timeline'),
  board: document.getElementById('board'),
  eventLog: document.getElementById('eventLog'),
};

el.loadDefaults.addEventListener('click', async () => {
  setStatus('Loading defaults...');
  const defaults = await fetchDefaults();
  if (defaults.case_folder) {
    el.casePath.value = `${defaults.case_folder}/dob_midtrain_v1_20260628_animal_seed_ct01_and_04842.json`;
  }
  setStatus('Idle');
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
  const client = new LiveBoardClient({
    onEvent: applyEvent,
    onStatus: setStatus,
  });
  setStatus('Connecting...');
  await client.connect();
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

function applyEvent(event) {
  if (event.type === 'unit_started') {
    appendTimeline(`unit ${event.unit_index}: audio=${event.payload?.n_audio ?? 0}, speaking=${event.payload?.is_speaking}`);
  } else if (event.type === 'spoken_final') {
    appendTimeline(`unit ${event.unit_index}: spoken "${event.text || ''}" (${(event.payload?.token_ids || []).length} tokens)`);
    if (event.payload?.audio_wav_base64) {
      appendAiAudio(event.unit_index, event.payload.audio_wav_base64);
    }
  } else if (event.type === 'think_final') {
    appendLog('think', event.think_text || '');
  } else if (event.type === 'tool_call_final') {
    appendLog('tool_call', JSON.stringify(event.tool_call, null, 2));
  } else if (event.type === 'board_card_created' || event.type === 'board_card_updated') {
    state.upsert(event.card);
    renderBoard();
  } else if (event.type === 'session_finished') {
    appendLog('summary', JSON.stringify(event.payload || {}, null, 2));
  } else if (event.type === 'session_error') {
    appendLog('error', event.text || 'unknown error');
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function renderBoard() {
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

function appendLog(kind, text) {
  const item = document.createElement('pre');
  item.className = `log-row ${kind}`;
  item.textContent = `[${kind}]\n${text}`;
  el.eventLog.appendChild(item);
}

function clearViews() {
  state.cards = [];
  el.timeline.innerHTML = '';
  el.board.innerHTML = '';
  el.eventLog.innerHTML = '';
  el.aiAudioList.innerHTML = '';
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
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}
