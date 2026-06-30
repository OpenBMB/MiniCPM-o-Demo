import { BoardState } from './board-state.js';
import { fetchDefaults, replayCase } from './replay-client.js';

const state = new BoardState({ maxCards: 10 });
const el = {
  status: document.getElementById('status'),
  casePath: document.getElementById('casePath'),
  loadDefaults: document.getElementById('loadDefaults'),
  runReplay: document.getElementById('runReplay'),
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

function applyEvent(event) {
  if (event.type === 'unit_started') {
    appendTimeline(`unit ${event.unit_index}: audio=${event.payload?.n_audio ?? 0}, speaking=${event.payload?.is_speaking}`);
  } else if (event.type === 'spoken_final') {
    appendTimeline(`unit ${event.unit_index}: spoken tokens=${(event.payload?.token_ids || []).length}`);
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
}

function setStatus(text) {
  el.status.textContent = text;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}
