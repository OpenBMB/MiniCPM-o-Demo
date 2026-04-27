/**
 * ui/duplex-ui.js — UI binding layer for duplex pages
 *
 * Bridges pure-logic metrics data from DuplexSession/AudioPlayer to DOM elements.
 * All DOM access is concentrated here.
 */

// ============================================================================
// MetricsPanel — Translates metrics data objects into DOM updates
// ============================================================================

export class MetricsPanel {
    constructor() {
        // Cache DOM elements on first access
        this._els = {};
    }

    _el(id) {
        if (!this._els[id]) this._els[id] = document.getElementById(id);
        return this._els[id];
    }

    /**
     * Handle a metrics update from DuplexSession or AudioPlayer.
     * @param {object} data - Metrics data with `type` field
     */
    update(data) {
        switch (data.type) {
            case 'audio': this._updateAudioMetrics(data); break;
            case 'result': this._updateResultMetrics(data); break;
            case 'state': this._updateState(data); break;
        }
    }

    _updateAudioMetrics(data) {
        // Ahead
        if (data.ahead !== undefined) {
            const el = this._el('aheadDisplay');
            if (!el) return;
            const aheadStr = `${Math.round(data.ahead)}ms`;
            const turnStr = data.turn ? ` T${data.turn}` : '';
            if (data.gapCount > 0) {
                el.textContent = `${aheadStr} \u26a0${data.gapCount}${turnStr}`;
                el.style.color = '#ff6b6b';
            } else {
                el.textContent = `${aheadStr}${turnStr}`;
                el.style.color = data.ahead > 200 ? '#4ecdc4' : data.ahead > 50 ? '#ffd93d' : '#ff6b6b';
            }
        }
        // Shift
        if (data.totalShift !== undefined) {
            const el = this._el('shiftDisplay');
            if (!el) return;
            const total = Math.round(data.totalShift);
            el.textContent = `${total}ms`;
            el.style.color = total === 0 ? '#4ecdc4' : total < 200 ? '#ffd93d' : '#ff6b6b';
        }
        // PDelay
        if (data.pdelay !== undefined) {
            const el = this._el('pdelayDisplay');
            if (el) el.textContent = `${Math.round(data.pdelay)}ms`;
        }
    }

    _updateResultMetrics(data) {
        // Chunks sent
        if (data.chunksSent !== undefined) {
            const el = this._el('chunkCount');
            if (el) el.textContent = data.chunksSent;
        }

        // Latency (removed cost fields from protocol)

        // TTFS
        if (data.ttfsMs) {
            const el = this._el('ttfsDisplay');
            if (el) {
                el.textContent = `${Math.round(data.ttfsMs)}ms`;
                el.style.color = data.ttfsMs < 1000 ? '#4ecdc4' : data.ttfsMs < 2000 ? '#ffd93d' : '#ff6b6b';
            }
        }

        // Drift
        if (data.driftMs !== undefined && data.driftMs !== null) {
            const el = this._el('driftDisplay');
            if (el) {
                const d = Math.round(data.driftMs);
                el.textContent = `${d > 0 ? '+' : ''}${d}ms`;
                const absd = Math.abs(d);
                el.style.color = absd < 50 ? '#4ecdc4' : absd < 200 ? '#ffd93d' : '#ff6b6b';
            }
        }

        // KV cache
        if (data.kvCacheLength !== undefined && data.kvCacheLength > 0) {
            const el = this._el('kvCacheDisplay');
            if (el) {
                el.textContent = data.kvCacheLength.toLocaleString();
                const maxKv = data.maxKvTokens || 8192;
                const ratio = data.kvCacheLength / maxKv;
                el.style.color = ratio < 0.5 ? '#4ecdc4' : ratio < 0.8 ? '#ffd93d' : '#ff6b6b';
            }
        }

        // Vision: n_vision_images (1 = source only, 3 = 1 source + 2 HD slices, etc.)
        if (data.visionSlices !== undefined) {
            const el = this._el('visionDisplay');
            if (el) {
                const nImg = data.visionSlices;
                const t = data.visionTokens || (nImg * 64);
                const label = nImg <= 1 ? 'Std' : `HD(${nImg})`;
                el.textContent = `${label} ${t} tok`;
                el.style.color = nImg <= 1 ? '#4ecdc4' : '#ffd93d';
            }
        }

        // Model state
        if (data.modelState) {
            const el = this._el('modelState');
            if (el) {
                if (data.modelState === 'listening' || data.modelState === 'end_of_turn') {
                    el.textContent = 'Listening';
                    el.className = 'status-value listening';
                } else if (data.modelState === 'speaking') {
                    el.textContent = 'Speaking';
                    el.className = 'status-value speaking';
                } else {
                    el.textContent = '\u2014';
                    el.className = 'status-value';
                }
            }
        }
    }

    _updateState(data) {
        if (data.sessionId !== undefined) {
            const el = this._el('sessionIdDisplay');
            if (el) el.textContent = data.sessionId ? data.sessionId.slice(0, 12) : '\u2014';
        }
        if (data.sessionState) {
            const el = this._el('sessionState');
            if (el) el.textContent = data.sessionState;
        }
        if (data.modelState !== undefined) {
            this._updateResultMetrics({ modelState: data.modelState });
        }
    }

    /** Reset all metric displays to initial state */
    reset() {
        const defaults = {
            sessionIdDisplay: '\u2014',
            latencyDisplay: '\u2014',
            ttfsDisplay: '\u2014',
            pdelayDisplay: '\u2014',
            aheadDisplay: '\u2014',
            shiftDisplay: '\u2014',
            driftDisplay: '\u2014',
            kvCacheDisplay: '\u2014',
            visionDisplay: '\u2014',
        };
        for (const [id, text] of Object.entries(defaults)) {
            const el = this._el(id);
            if (el) {
                el.textContent = text;
                el.style.color = '';
            }
        }
    }
}

// ============================================================================
// Status Panel HTML Generator
// ============================================================================

/**
 * Returns the inner HTML for the status panel with 11 metric rows.
 * @returns {string}
 */
export function getStatusPanelHTML() {
    return `
        <div class="status-grid">
            <div class="status-row" data-tip="Session state: Idle / Running / Stopped"><span>State</span><span class="status-value" id="sessionState">Idle</span></div>
            <div class="status-row" data-tip="Session ID assigned by server"><span>SID</span><span class="status-value" id="sessionIdDisplay" style="font-size:9px;letter-spacing:-0.5px;">\u2014</span></div>
            <div class="status-row" data-tip="Model state: Listening / Speaking"><span>Model</span><span class="status-value" id="modelState">\u2014</span></div>
            <div class="status-row" data-tip="Total audio chunks received from server"><span>Chunks</span><span class="status-value" id="chunkCount">0</span></div>
            <div class="status-row" data-tip="KV cache total tokens used"><span>KV</span><span class="status-value" id="kvCacheDisplay">\u2014</span></div>
            <div class="status-row" data-tip="Vision quality: slices × 64 tokens per frame"><span>Vision</span><span class="status-value" id="visionDisplay">\u2014</span></div>
            <div class="status-divider"></div>
            <div class="status-row" data-tip="Model inference time per chunk (avg)"><span>Infer</span><span class="status-value" id="latencyDisplay">\u2014</span></div>
            <div class="status-row" data-tip="Time To First Speak: delay from last LISTEN to first SPEAK"><span>TTFS</span><span class="status-value" id="ttfsDisplay">\u2014</span></div>
            <div class="status-row" data-tip="Playback Delay: time from first SPEAK result to audio playback start"><span>PDelay</span><span class="status-value" id="pdelayDisplay">\u2014</span></div>
            <div class="status-row" data-tip="Playback continuity: schedule margin ahead of real-time + gap count"><span>Ahead</span><span class="status-value" id="aheadDisplay">\u2014</span></div>
            <div class="status-row" data-tip="Accumulated time shift: PDelay + playback gaps"><span>Shift</span><span class="status-value" id="shiftDisplay">\u2014</span></div>
            <div class="status-row" data-tip="Network drift: latency change compared to first result"><span>Drift</span><span class="status-value" id="driftDisplay">\u2014</span></div>
        </div>`;
}

// ============================================================================
// Mixer Panel HTML Generator (shared between Omni and Audio Duplex)
// ============================================================================

/**
 * Returns the full innerHTML for the draggable LUFS Mixer panel.
 * Usage: document.getElementById('mixerPanel').innerHTML = getMixerPanelHTML();
 * @returns {string}
 */
export function getMixerPanelHTML() {
    return `
    <div class="mixer-header" id="mixerDragHandle">
        <span class="mixer-header-title" data-tip="LUFS-based audio mixer: normalizes loudness before sending to the model">Mixer</span>
        <span class="mixer-status" data-tip="Audio pipeline status indicators">
            <span class="mixer-dot" id="mixerDotMic"></span> Mic
            <span class="mixer-dot" id="mixerDotFile"></span> File
            <span class="mixer-dot" id="mixerDotWorklet"></span> Worklet
        </span>
        <button class="mixer-close" id="mixerClose">&times;</button>
    </div>

    <div class="mixer-section">
        <div class="mixer-section-title" data-tip="Audio mix sent to the model for inference">TO AI</div>
        <div class="mixer-source">
            <div class="mixer-source-header">
                <span class="mixer-source-name">File</span>
                <span class="mixer-info" data-tip="Target loudness for file audio (LUFS). Auto-gain adjusts to match this level">target <input id="mxFileTarget" type="number" class="mx-num" value="-33"> LUFS</span>
                <span class="mixer-info" data-tip="Measured loudness of the source file (before normalization)">src <span id="mxFileMeasured">\u2014</span> LUFS</span>
            </div>
            <div class="mixer-source-row">
                <span class="mixer-info" data-tip="Auto-calculated gain to reach target LUFS (= target \u2212 measured)">auto <span id="mxFileAuto">\u2014</span> dB</span>
                <span class="mixer-label" data-tip="Manual trim on top of auto-gain (dB)">trim</span>
                <input type="range" id="mxFileTrim" min="-12" max="12" step="1" value="0" data-tip="Manual file audio trim (dB), added on top of auto-gain">
                <span class="mixer-val" id="mxFileTrimVal">0 dB</span>
            </div>
            <div class="meter-row" data-tip="Real-time file audio level meter (after gain)">
                <span class="meter-label">File</span>
                <div class="meter-bar-bg"><div class="meter-bar" id="mxMeterFile"></div></div>
                <span class="meter-val" id="mxMeterFileVal">\u2014</span>
            </div>
        </div>
        <div class="mixer-source">
            <div class="mixer-source-header">
                <span class="mixer-source-name">Mic</span>
                <span class="mixer-info" data-tip="Target loudness for microphone audio (LUFS). Auto-gain adjusts to match this level">target <input id="mxMicTarget" type="number" class="mx-num" value="-23"> LUFS</span>
                <span class="mixer-info" data-tip="Calibrated mic loudness (use Measure to update). * = estimated, not measured">cal <span id="mxMicMeasured">-23*</span> LUFS</span>
                <button class="fo-cal-btn" id="micCalBtn" data-tip="Record 3s of mic audio to measure its actual LUFS loudness">&#x1F3A4; Measure</button>
            </div>
            <div class="mixer-source-row">
                <span class="mixer-info" data-tip="Auto-calculated gain to reach target LUFS (= target \u2212 calibrated)">auto <span id="mxMicAuto">0</span> dB</span>
                <span class="mixer-label" data-tip="Manual trim on top of auto-gain (dB)">trim</span>
                <input type="range" id="mxMicTrim" min="-12" max="12" step="1" value="0" data-tip="Manual mic trim (dB), added on top of auto-gain">
                <span class="mixer-val" id="mxMicTrimVal">0 dB</span>
            </div>
            <div class="meter-row" data-tip="Real-time mic audio level meter (after gain)">
                <span class="meter-label">Mic</span>
                <div class="meter-bar-bg"><div class="meter-bar" id="mxMeterMic"></div></div>
                <span class="meter-val" id="mxMeterMicVal">\u2014</span>
            </div>
        </div>
        <div class="meter-row mixer-mix-meter" data-tip="Combined mix level (file + mic) as sent to the model">
            <span class="meter-label">Mix</span>
            <div class="meter-bar-bg"><div class="meter-bar" id="mxMeterMix"></div></div>
            <span class="meter-val" id="mxMeterMixVal">\u2014</span>
        </div>
    </div>

    <div class="mixer-section">
        <div class="mixer-section-title" data-tip="Audio output to your speakers/headphones for monitoring">TO SPEAKER</div>
        <div class="mixer-row">
            <span class="mixer-label" data-tip="Speaker output volume for file audio monitoring">Volume</span>
            <input type="range" id="mxMonitor" min="0" max="100" step="1" value="50" data-tip="Speaker output volume (0\u2013100%)">
            <span class="mixer-val" id="mxMonitorVal">50%</span>
        </div>
        <div class="mixer-hint">File audio only (no mic, no echo)</div>
    </div>

    <div class="mixer-section">
        <div class="mixer-section-title" data-tip="Record a short clip to preview the final mix before starting a session">PREVIEW</div>
        <div class="mixer-preview">
            <button id="mxPreviewRec" class="mixer-prev-btn" data-tip="Start recording a preview clip of the current mix">\u25cf Rec</button>
            <button id="mxPreviewStop" class="mixer-prev-btn" disabled data-tip="Stop the preview recording">\u25a0 Stop</button>
            <button id="mxPreviewPlay" class="mixer-prev-btn" disabled data-tip="Play back the recorded preview clip">\u25b6 Play</button>
            <span id="mxPreviewDur" class="mixer-info" data-tip="Duration of the recorded preview clip">\u2014</span>
        </div>
    </div>

    <span class="fo-cal-result" id="micCalResult"></span>`;
}

// ============================================================================
// Button State Management (defaults, pages can override)
// ============================================================================

/** Enable/disable the 4 standard duplex buttons based on running state. */
export function setDuplexButtonStates(running) {
    const start = document.getElementById('btnStart');
    const stop = document.getElementById('btnStop');
    const fl = document.getElementById('btnForceListen');
    const pause = document.getElementById('btnPause');
    if (start) { start.disabled = running; start.textContent = running ? '● Live' : 'Start'; start.classList.remove('cancel'); start.classList.toggle('live', running); }
    if (stop) { stop.disabled = !running; stop.textContent = 'Stop'; stop.classList.remove('cancel'); }
    if (fl) fl.disabled = !running;
    if (pause) pause.disabled = !running;
}

/**
 * 排队阶段的按钮状态管理。
 * @param {'queuing'|'almost'|'assigned'|null} phase
 *   - queuing/almost: Start="Queued" disabled, Stop="Cancel" enabled
 *   - assigned: Start="Preparing..." disabled, Stop disabled
 *   - null: 恢复默认文字（调用方之后用 setDuplexButtonStates 设最终状态）
 */
export function setQueueButtonStates(phase) {
    const start = document.getElementById('btnStart');
    const stop = document.getElementById('btnStop');
    const fl = document.getElementById('btnForceListen');
    const pause = document.getElementById('btnPause');

    if (phase === 'queuing' || phase === 'almost') {
        if (start) { start.disabled = true; start.textContent = 'Queued'; }
        if (stop) { stop.disabled = false; stop.textContent = 'Cancel'; stop.classList.add('cancel'); }
        if (fl) fl.disabled = true;
        if (pause) pause.disabled = true;
    } else if (phase === 'assigned') {
        if (start) { start.disabled = true; start.textContent = 'Preparing...'; }
        if (stop) { stop.disabled = true; stop.textContent = 'Stop'; stop.classList.remove('cancel'); }
        if (fl) fl.disabled = true;
        if (pause) pause.disabled = true;
    } else {
        if (start) { start.textContent = 'Start'; start.classList.remove('cancel'); }
        if (stop) { stop.textContent = 'Stop'; stop.classList.remove('cancel'); }
    }
}

/** Default pause button text/state handler. */
export function setDefaultPauseBtnState(state) {
    const btn = document.getElementById('btnPause');
    if (!btn) return;
    switch (state) {
        case 'active': btn.textContent = 'Pause'; btn.disabled = false; break;
        case 'pausing': btn.textContent = 'Pausing...'; btn.disabled = true; break;
        case 'paused': btn.textContent = 'Resume'; btn.disabled = false; break;
    }
}

/** Default force-listen button text/state handler. */
export function setDefaultForceListenBtnState(active) {
    const btn = document.getElementById('btnForceListen');
    if (!btn) return;
    btn.textContent = active ? 'Release' : 'Force Listen';
    btn.classList.toggle('force-listen-active', active);
}

// ============================================================================
// Health / Status Check
// ============================================================================

/**
 * Start periodic health check, updating a status badge element.
 * @param {string} badgeId - DOM element ID for the status badge
 */
export function initHealthCheck(badgeId) {
    const statusEl = document.getElementById(badgeId);
    if (!statusEl) return () => {};

    async function check() {
        try {
            const resp = await fetch('/status');
            const data = await resp.json();
            statusEl.textContent = `Workers: ${data.idle_workers}/${data.total_workers}`;
            statusEl.className = 'status-badge' + (data.idle_workers > 0 ? ' online' : '');
        } catch {
            statusEl.textContent = 'Offline';
            statusEl.className = 'status-badge';
        }
    }

    check();
    const intervalId = setInterval(check, 10000);
    return () => clearInterval(intervalId);
}

// ============================================================================
// Control Button Wiring
// ============================================================================

/** Bind the 4 standard duplex control buttons to handler functions. */
export function wireDuplexControls({ onStart, onStop, onPause, onForceListen }) {
    document.getElementById('btnStart')?.addEventListener('click', onStart);
    document.getElementById('btnStop')?.addEventListener('click', onStop);
    document.getElementById('btnPause')?.addEventListener('click', onPause);
    document.getElementById('btnForceListen')?.addEventListener('click', onForceListen);
}

// ============================================================================
// Load Frontend Defaults from server
// ============================================================================

export async function loadFrontendDefaults() {
    try {
        const resp = await fetch('/api/frontend_defaults');
        if (!resp.ok) return;
        const defaults = await resp.json();
        if (defaults.playback_delay_ms != null) {
            const el = document.getElementById('playbackDelay');
            if (el) el.value = defaults.playback_delay_ms;
        }
    } catch (e) {
        console.warn('[frontend_defaults] fetch failed, using HTML defaults:', e.message);
    }
}

// ============================================================================
// SettingsPersistence — Save/restore user-configurable parameters via localStorage
// ============================================================================

/**
 * 声明式 localStorage 持久化。
 *
 * fieldDefs 元素格式:
 *   { id: string, type: 'number'|'range'|'textarea'|'checkbox' }
 *   { type: 'radio', name: string }          — radio group (by name attribute)
 *   { type: 'mode', selector: string }       — toggle buttons (.mode-btn active)
 *
 * 用法:
 *   const sp = new SettingsPersistence('omni_settings', [ ... ]);
 *   sp.restore();           // 页面加载后调用
 *   sp.clear();             // 恢复默认 + 刷新
 */
export class SettingsPersistence {
    /**
     * @param {string} storageKey - localStorage key name
     * @param {Array<object>} fieldDefs - 参数声明列表
     */
    constructor(storageKey, fieldDefs) {
        this._key = storageKey;
        this._defs = fieldDefs;
        this._saveTimer = 0;
    }

    /** 收集当前 DOM 值，写入 localStorage */
    save() {
        const data = {};
        for (const def of this._defs) {
            try {
                if (def.type === 'radio') {
                    const checked = document.querySelector(`input[name="${def.name}"]:checked`);
                    if (checked) data[`radio:${def.name}`] = checked.value;
                } else if (def.type === 'mode') {
                    const active = document.querySelector(`${def.selector}.active`);
                    if (active) data[`mode:${def.selector}`] = active.dataset.mode;
                } else if (def.type === 'checkbox') {
                    const el = document.getElementById(def.id);
                    if (el) data[def.id] = el.checked;
                } else {
                    const el = document.getElementById(def.id);
                    if (el) data[def.id] = el.value;
                }
            } catch (_) { /* skip inaccessible elements */ }
        }
        try {
            localStorage.setItem(this._key, JSON.stringify(data));
        } catch (e) {
            console.warn(`[SettingsPersistence] save failed:`, e.message);
        }
    }

    /** 从 localStorage 读取并恢复到 DOM，触发 change/input 事件 */
    restore() {
        let data;
        try {
            const raw = localStorage.getItem(this._key);
            if (!raw) { this._bindAutoSave(); return; }
            data = JSON.parse(raw);
        } catch {
            this._bindAutoSave();
            return;
        }

        for (const def of this._defs) {
            try {
                if (def.type === 'radio') {
                    const val = data[`radio:${def.name}`];
                    if (val == null) continue;
                    const radio = document.querySelector(`input[name="${def.name}"][value="${val}"]`);
                    if (radio) {
                        radio.checked = true;
                        radio.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                } else if (def.type === 'mode') {
                    const val = data[`mode:${def.selector}`];
                    if (val == null) continue;
                    const btn = document.querySelector(`${def.selector}[data-mode="${val}"]`);
                    if (btn && !btn.classList.contains('active')) {
                        btn.click();
                    }
                } else if (def.type === 'checkbox') {
                    const val = data[def.id];
                    if (val == null) continue;
                    const el = document.getElementById(def.id);
                    if (el) {
                        el.checked = !!val;
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                } else {
                    const val = data[def.id];
                    if (val == null) continue;
                    const el = document.getElementById(def.id);
                    if (el) {
                        el.value = val;
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
            } catch (_) { /* skip */ }
        }

        console.log(`[SettingsPersistence] restored from "${this._key}"`);
        this._bindAutoSave();
    }

    /** 移除 localStorage 并刷新页面 */
    clear() {
        try { localStorage.removeItem(this._key); } catch (_) {}
        location.reload();
    }

    /** 绑定自动保存：监听所有目标元素的 change/input 事件 */
    _bindAutoSave() {
        const debouncedSave = () => {
            clearTimeout(this._saveTimer);
            this._saveTimer = setTimeout(() => this.save(), 300);
        };

        for (const def of this._defs) {
            try {
                if (def.type === 'radio') {
                    const radios = document.querySelectorAll(`input[name="${def.name}"]`);
                    radios.forEach(r => r.addEventListener('change', debouncedSave));
                } else if (def.type === 'mode') {
                    const btns = document.querySelectorAll(def.selector);
                    const observer = new MutationObserver(debouncedSave);
                    btns.forEach(b => observer.observe(b, { attributes: true, attributeFilter: ['class'] }));
                } else {
                    const el = document.getElementById(def.id);
                    if (el) {
                        el.addEventListener('change', debouncedSave);
                        if (def.type === 'range' || def.type === 'textarea') {
                            el.addEventListener('input', debouncedSave);
                        }
                    }
                }
            } catch (_) { /* skip */ }
        }
    }
}

// ============================================================================
// Instant Tooltip — follows cursor, appears at bottom-right of mouse
// ============================================================================
export function initDataTipTooltips() {
    let popup = null;
    function show(e) {
        const tip = e.currentTarget.getAttribute('data-tip');
        if (!tip) return;
        if (!popup) {
            popup = document.createElement('div');
            popup.className = 'data-tip-popup';
            document.body.appendChild(popup);
        }
        popup.textContent = tip;
        popup.style.left = (e.clientX + 12) + 'px';
        popup.style.top = (e.clientY + 12) + 'px';
        popup.style.display = 'block';
    }
    function move(e) {
        if (popup && popup.style.display === 'block') {
            popup.style.left = (e.clientX + 12) + 'px';
            popup.style.top = (e.clientY + 12) + 'px';
        }
    }
    function hide() {
        if (popup) popup.style.display = 'none';
    }
    document.addEventListener('mouseover', (e) => {
        const target = e.target.closest('[data-tip]');
        if (target) { show({ currentTarget: target, clientX: e.clientX, clientY: e.clientY }); }
    });
    document.addEventListener('mousemove', (e) => {
        const target = e.target.closest('[data-tip]');
        if (target) { move(e); } else { hide(); }
    });
    document.addEventListener('mouseout', (e) => {
        const target = e.target.closest('[data-tip]');
        if (target && !target.contains(e.relatedTarget)) { hide(); }
    });
}
