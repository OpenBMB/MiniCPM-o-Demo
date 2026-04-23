/* ============================================================
 * mobile-bridge.js
 *
 * Adapts the desktop omni page DOM (loaded by omni-app.js) for mobile:
 *   1. Force `body.video-fullscreen` so the page boots in fullscreen.
 *   2. Inject a top-left ← back button (returns to /mobile/).
 *   3. Re-skin the bottom-right #fullscreenBtn as a ⚙️ settings opener
 *      that shows a feature-rich bottom-sheet matching the merge-branch
 *      SettingsSheet design (Preset / Ref Audio / Prompt / Params).
 *   4. Camera extras: torch toggle + pinch-to-zoom on supported devices.
 *   5. Bridge `sessionStorage['mobileOmni:settings']` -> #systemPrompt.
 * ========================================================= */

(function bootstrapMobileOmni() {
    const BACK_URL = '/mobile/';

    const ICON_BACK = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"/></svg>';
    const ICON_GEAR = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h0a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h0a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v0a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>';
    const ICON_CLOSE = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
    const ICON_TORCH = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 2h6l-1 6h-4z"/><path d="M8 8h8l-2 8h-4z"/><path d="M11 16v6"/></svg>';

    function ensureFullscreenClass() {
        document.body.classList.add('video-fullscreen');
        document.body.classList.add('mobile-omni');
    }

    // ========================================================================
    // Back button (top-left)
    // ========================================================================
    function injectBackButton() {
        const container = document.getElementById('videoContainer');
        if (!container || container.querySelector('.mobile-back-btn')) return;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'mobile-back-btn';
        btn.setAttribute('aria-label', 'Back to mobile turn page');
        btn.innerHTML = ICON_BACK;
        const goBack = (e) => {
            if (e) { e.preventDefault(); e.stopPropagation(); }
            try { window.location.assign(BACK_URL); } catch (_) { window.location.href = BACK_URL; }
        };
        btn.addEventListener('click', goBack);
        btn.addEventListener('touchend', goBack, { passive: false });
        container.appendChild(btn);
    }

    // ========================================================================
    // Settings sheet — aligned to merge-branch SettingsSheet design
    // ========================================================================
    let presetData = [];
    let selectedPresetId = null;
    let refAudioState = { name: '未设置参考音频', source: 'none', duration: 0, hasAudio: false };

    function buildSettingsSheet() {
        if (document.getElementById('mobileSettingsSheet')) return;

        const backdrop = document.createElement('div');
        backdrop.className = 'mobile-settings-backdrop';
        backdrop.id = 'mobileSettingsBackdrop';

        const sheet = document.createElement('div');
        sheet.className = 'mobile-settings-sheet';
        sheet.id = 'mobileSettingsSheet';
        sheet.innerHTML = `
            <div class="ms-head">
                <div>
                    <div class="ms-title">设置</div>
                    <div class="ms-subtitle">视频双工</div>
                </div>
                <button class="ms-close-btn" type="button" id="msCloseBtn">${ICON_CLOSE}</button>
            </div>
            <div class="ms-section">
                <div class="ms-section-title">Preset</div>
                <div class="ms-preset-row" id="msPresetRow">
                    <div class="ms-preset-loading">Loading presets...</div>
                </div>
            </div>
            <div class="ms-section">
                <div class="ms-section-title">参考音频</div>
                <div class="ms-ref-card" id="msRefCard">
                    <div class="ms-ref-title" id="msRefTitle">未设置参考音频</div>
                    <div class="ms-ref-meta" id="msRefMeta">来源：none</div>
                    <div class="ms-ref-actions">
                        <button class="ms-ref-btn" type="button" id="msRefDefault">默认</button>
                        <button class="ms-ref-btn" type="button" id="msRefUpload">上传</button>
                        <button class="ms-ref-btn" type="button" id="msRefPlay" disabled>播放</button>
                        <button class="ms-ref-btn" type="button" id="msRefClear">清空</button>
                    </div>
                </div>
            </div>
            <div class="ms-section">
                <label class="ms-section-title" for="msSystemPrompt">System Prompt</label>
                <textarea class="ms-textarea" id="msSystemPrompt" rows="4"></textarea>
            </div>
            <div class="ms-section">
                <div class="ms-section-title">参数</div>
                <div class="ms-grid">
                    <label class="ms-field">
                        <span>Length Penalty</span>
                        <input class="ms-input" type="number" id="msLengthPenalty" min="0.1" max="5" step="0.05">
                    </label>
                    <label class="ms-field">
                        <span>Playback Delay (ms)</span>
                        <input class="ms-input" type="number" id="msPlaybackDelay" min="0" max="2000" step="50">
                    </label>
                    <label class="ms-field">
                        <span>Max KV (tok)</span>
                        <input class="ms-input" type="number" id="msMaxKv" min="512" max="16384" step="512">
                    </label>
                </div>
            </div>
            <button class="ms-done-btn" type="button" id="msDoneBtn">完成</button>
        `;

        document.body.appendChild(backdrop);
        document.body.appendChild(sheet);

        // stop sheet clicks from closing
        sheet.addEventListener('click', (e) => e.stopPropagation());

        const close = () => { backdrop.classList.remove('open'); sheet.classList.remove('open'); };
        backdrop.addEventListener('click', close);
        document.getElementById('msCloseBtn').addEventListener('click', close);
        document.getElementById('msDoneBtn').addEventListener('click', close);

        bindMirror('msSystemPrompt', 'systemPrompt');
        bindMirror('msLengthPenalty', 'omniLengthPenalty');
        bindMirror('msPlaybackDelay', 'playbackDelay');
        bindMirror('msMaxKv', 'maxKvTokens');

        loadPresets();
        wireRefAudioActions();
        refreshRefAudioDisplay();
    }

    function bindMirror(mobileId, desktopId) {
        const m = document.getElementById(mobileId);
        const d = document.getElementById(desktopId);
        if (!m || !d) return;
        m.value = d.value;
        m.addEventListener('input', () => {
            d.value = m.value;
            d.dispatchEvent(new Event('input', { bubbles: true }));
            d.dispatchEvent(new Event('change', { bubbles: true }));
        });
        const sync = () => { if (m.value !== d.value) m.value = d.value; };
        d.addEventListener('input', sync);
        d.addEventListener('change', sync);
    }

    // ---- Presets ----
    async function loadPresets() {
        const row = document.getElementById('msPresetRow');
        if (!row) return;
        try {
            const resp = await fetch('/api/presets');
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            presetData = data.omni || [];
        } catch (e) {
            console.warn('[mobile-omni] preset load failed:', e);
            presetData = [];
        }
        renderPresets();
    }

    function renderPresets() {
        const row = document.getElementById('msPresetRow');
        if (!row) return;
        row.innerHTML = '';
        if (presetData.length === 0) {
            row.innerHTML = '<div class="ms-preset-loading">暂无预设</div>';
            return;
        }

        // detect which desktop preset is currently active
        const desktopBtns = document.querySelectorAll('#presetSelectorOmni .preset-btn.active');
        const desktopActiveId = desktopBtns.length ? desktopBtns[0].dataset.presetId : null;
        if (desktopActiveId) selectedPresetId = desktopActiveId;
        if (!selectedPresetId && presetData.length) selectedPresetId = presetData[0].id;

        for (const p of presetData) {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'ms-preset-chip' + (p.id === selectedPresetId ? ' active' : '');
            chip.textContent = p.name;
            chip.title = p.description || p.name;
            chip.dataset.presetId = p.id;
            chip.addEventListener('click', () => selectPreset(p.id));
            row.appendChild(chip);
        }
    }

    async function selectPreset(presetId) {
        selectedPresetId = presetId;
        // click the corresponding desktop preset button so omni-app.js handles
        // the full logic (apply prompt, lazy-load audio, etc.)
        const desktopBtn = document.querySelector(`#presetSelectorOmni .preset-btn[data-preset-id="${presetId}"]`);
        if (desktopBtn) desktopBtn.click();

        // update mobile chip highlight
        const row = document.getElementById('msPresetRow');
        if (row) {
            for (const c of row.querySelectorAll('.ms-preset-chip')) {
                c.classList.toggle('active', c.dataset.presetId === presetId);
            }
        }

        // refresh system prompt from desktop after a tick (preset may async-update it)
        setTimeout(() => {
            const m = document.getElementById('msSystemPrompt');
            const d = document.getElementById('systemPrompt');
            if (m && d) m.value = d.value;
            refreshRefAudioDisplay();
        }, 500);
    }

    // ---- Ref Audio ----
    function getDesktopRefAudioPlayer() {
        const container = document.getElementById('omniRefAudioPlayer');
        if (!container) return null;
        const nameEl = container.querySelector('.rap-name');
        const durationEl = container.querySelector('.rap-duration');
        return { container, nameEl, durationEl };
    }

    function refreshRefAudioDisplay() {
        const rap = getDesktopRefAudioPlayer();
        const titleEl = document.getElementById('msRefTitle');
        const metaEl = document.getElementById('msRefMeta');
        const playBtn = document.getElementById('msRefPlay');

        if (rap && rap.nameEl) {
            const name = rap.nameEl.textContent?.trim();
            if (name && name !== '未设置' && name !== 'No audio') {
                refAudioState.name = name;
                refAudioState.hasAudio = true;
                const dur = rap.durationEl?.textContent?.trim() || '';
                refAudioState.duration = parseFloat(dur) || 0;
                refAudioState.source = 'preset';
            } else {
                refAudioState = { name: '未设置参考音频', source: 'none', duration: 0, hasAudio: false };
            }
        }
        if (titleEl) titleEl.textContent = refAudioState.hasAudio ? refAudioState.name : '未设置参考音频';
        if (metaEl) {
            metaEl.textContent = refAudioState.hasAudio
                ? `来源：${refAudioState.source}${refAudioState.duration ? ` · ${refAudioState.duration.toFixed(1)}s` : ''}`
                : '来源：none';
        }
        if (playBtn) playBtn.disabled = !refAudioState.hasAudio;
    }

    function wireRefAudioActions() {
        // Default — click the desktop Remove button first (resets to default), then let it reload
        document.getElementById('msRefDefault')?.addEventListener('click', () => {
            const removeBtn = document.querySelector('#omniRefAudioPlayer .rap-remove-btn');
            if (removeBtn) removeBtn.click();
            setTimeout(refreshRefAudioDisplay, 1500);
        });

        // Upload — trigger the desktop upload input
        document.getElementById('msRefUpload')?.addEventListener('click', () => {
            const uploadInput = document.querySelector('#omniRefAudioPlayer input[type="file"]');
            if (uploadInput) {
                uploadInput.click();
                uploadInput.addEventListener('change', () => {
                    setTimeout(refreshRefAudioDisplay, 1000);
                }, { once: true });
            }
        });

        // Play — trigger the desktop play button
        document.getElementById('msRefPlay')?.addEventListener('click', () => {
            const playBtn = document.querySelector('#omniRefAudioPlayer .rap-play-btn');
            if (playBtn) playBtn.click();
        });

        // Clear — click the desktop remove, then prevent default-reload by immediately setting empty
        document.getElementById('msRefClear')?.addEventListener('click', () => {
            const removeBtn = document.querySelector('#omniRefAudioPlayer .rap-remove-btn');
            if (removeBtn) removeBtn.click();
            setTimeout(refreshRefAudioDisplay, 500);
        });
    }

    // ---- Open sheet ----
    function openSettingsSheet() {
        const backdrop = document.getElementById('mobileSettingsBackdrop');
        const sheet = document.getElementById('mobileSettingsSheet');
        if (!backdrop || !sheet) return;

        // sync inputs from desktop
        const pairs = [
            ['msSystemPrompt', 'systemPrompt'],
            ['msLengthPenalty', 'omniLengthPenalty'],
            ['msPlaybackDelay', 'playbackDelay'],
            ['msMaxKv', 'maxKvTokens'],
        ];
        for (const [mid, did] of pairs) {
            const m = document.getElementById(mid);
            const d = document.getElementById(did);
            if (m && d) m.value = d.value;
        }

        renderPresets();
        refreshRefAudioDisplay();

        backdrop.classList.add('open');
        sheet.classList.add('open');
    }

    // ========================================================================
    // ⚙️ on the bottom-right
    // ========================================================================
    function rebindFullscreenButtonAsGear() {
        const btn = document.getElementById('fullscreenBtn');
        if (!btn) return;
        const clone = btn.cloneNode(false);
        clone.id = 'fullscreenBtn';
        clone.className = btn.className;
        clone.classList.add('visible');
        clone.setAttribute('title', 'Settings');
        clone.setAttribute('aria-label', 'Open settings');
        clone.innerHTML = ICON_GEAR;
        btn.parentNode.replaceChild(clone, btn);
        const open = (e) => { if (e) { e.preventDefault(); e.stopPropagation(); } openSettingsSheet(); };
        clone.addEventListener('click', open);
        clone.addEventListener('touchend', open, { passive: false });
    }

    function keepGearVisible() {
        const btn = document.getElementById('fullscreenBtn');
        if (!btn) return;
        const obs = new MutationObserver(() => {
            if (!btn.classList.contains('visible')) btn.classList.add('visible');
        });
        obs.observe(btn, { attributes: true, attributeFilter: ['class'] });
    }

    // ========================================================================
    // Session storage bridge from /mobile/
    // ========================================================================
    function applyMobileSettings() {
        let raw;
        try { raw = sessionStorage.getItem('mobileOmni:settings'); } catch (_) { return; }
        if (!raw) return;
        let s;
        try { s = JSON.parse(raw); } catch (_) { return; }
        if (!s || typeof s !== 'object') return;
        if (typeof s.systemPrompt === 'string') {
            const ta = document.getElementById('systemPrompt');
            if (ta) {
                ta.value = s.systemPrompt;
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
    }

    // ========================================================================
    // Camera extras: torch + pinch zoom
    // ========================================================================
    let torchBtn = null;
    let lastVideoTrack = null;
    let pinchState = null;

    function getActiveVideoTrack() {
        const v = document.getElementById('videoEl');
        if (!v || !v.srcObject) return null;
        const stream = v.srcObject;
        if (typeof stream.getVideoTracks !== 'function') return null;
        const tracks = stream.getVideoTracks();
        return tracks && tracks.length ? tracks[0] : null;
    }

    function injectTorchButton() {
        const container = document.getElementById('videoContainer');
        if (!container) return;
        if (container.querySelector('.mobile-torch-btn')) {
            torchBtn = container.querySelector('.mobile-torch-btn');
            return;
        }
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'mobile-torch-btn';
        btn.setAttribute('aria-label', 'Toggle flashlight');
        btn.innerHTML = ICON_TORCH;
        btn.addEventListener('click', toggleTorch);
        btn.addEventListener('touchend', (e) => { e.preventDefault(); toggleTorch(); }, { passive: false });
        container.appendChild(btn);
        torchBtn = btn;
    }

    async function toggleTorch() {
        const track = getActiveVideoTrack();
        if (!track || typeof track.applyConstraints !== 'function') return;
        const caps = typeof track.getCapabilities === 'function' ? track.getCapabilities() : {};
        if (!caps || !caps.torch) return;
        const settings = typeof track.getSettings === 'function' ? track.getSettings() : {};
        const next = !settings.torch;
        try {
            await track.applyConstraints({ advanced: [{ torch: next }] });
            torchBtn?.classList.toggle('active', next);
        } catch (err) {
            console.warn('[mobile-omni] torch toggle failed:', err);
        }
    }

    function refreshCameraCapabilities() {
        const track = getActiveVideoTrack();
        if (track === lastVideoTrack) { updateTorchVisibility(track); return; }
        lastVideoTrack = track;
        torchBtn?.classList.remove('active');
        updateTorchVisibility(track);
    }

    function updateTorchVisibility(track) {
        if (!torchBtn) return;
        let supported = false;
        if (track && typeof track.getCapabilities === 'function') {
            try { supported = !!(track.getCapabilities().torch); } catch (_) {}
        }
        torchBtn.classList.toggle('visible', supported);
    }

    function watchVideoElement() {
        const v = document.getElementById('videoEl');
        if (!v) return;
        setInterval(refreshCameraCapabilities, 1000);
    }

    // pinch zoom
    function distance(t1, t2) { return Math.hypot(t1.clientX - t2.clientX, t1.clientY - t2.clientY); }

    async function handlePinchStart(e) {
        if (e.touches.length !== 2) return;
        const track = getActiveVideoTrack();
        if (!track || typeof track.getCapabilities !== 'function') return;
        let caps = {};
        try { caps = track.getCapabilities() || {}; } catch (_) {}
        if (!caps.zoom) return;
        const settings = typeof track.getSettings === 'function' ? track.getSettings() : {};
        pinchState = {
            startDist: distance(e.touches[0], e.touches[1]),
            startZoom: settings.zoom || caps.zoom.min || 1,
            min: caps.zoom.min || 1,
            max: caps.zoom.max || 1,
            track,
        };
        e.preventDefault();
    }

    function handlePinchMove(e) {
        if (!pinchState || e.touches.length !== 2) return;
        e.preventDefault();
        const d = distance(e.touches[0], e.touches[1]);
        const ratio = d / pinchState.startDist;
        let next = pinchState.startZoom * ratio;
        if (next < pinchState.min) next = pinchState.min;
        if (next > pinchState.max) next = pinchState.max;
        pinchState.track.applyConstraints({ advanced: [{ zoom: next }] }).catch(() => {});
    }

    function handlePinchEnd(e) {
        if (e.touches.length < 2) pinchState = null;
    }

    function bindPinchZoom() {
        const v = document.getElementById('videoContainer');
        if (!v) return;
        v.addEventListener('touchstart', handlePinchStart, { passive: false });
        v.addEventListener('touchmove', handlePinchMove, { passive: false });
        v.addEventListener('touchend', handlePinchEnd, { passive: false });
        v.addEventListener('touchcancel', handlePinchEnd, { passive: false });
    }

    // ========================================================================
    // Init
    // ========================================================================
    function init() {
        ensureFullscreenClass();
        setTimeout(() => {
            injectBackButton();
            buildSettingsSheet();
            rebindFullscreenButtonAsGear();
            keepGearVisible();
            applyMobileSettings();
            injectTorchButton();
            watchVideoElement();
            bindPinchZoom();
        }, 0);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init, { once: true });
    } else {
        init();
    }
})();
