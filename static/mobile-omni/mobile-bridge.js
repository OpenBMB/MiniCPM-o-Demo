/* ============================================================
 * mobile-bridge.js
 *
 * Adapts the desktop omni page DOM (loaded by omni-app.js) for mobile:
 *   1. Force `body.video-fullscreen` so the page boots in fullscreen.
 *   2. Inject a top-left ← back button (returns to /mobile/).
 *   3. Re-skin the bottom-right #fullscreenBtn as a ⚙️ settings opener
 *      that pops up a bottom-sheet with prompt / length-penalty / delay /
 *      maxKV controls bound to the underlying omni inputs.
 *   4. Camera extras: torch toggle + pinch-to-zoom on supported devices.
 *   5. Bridge `sessionStorage['mobileOmni:settings']` -> #systemPrompt.
 * ========================================================= */

(function bootstrapMobileOmni() {
    const BACK_URL = '/mobile/';

    // ----- icons (inline SVG) -----
    const ICON_BACK = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"/></svg>';
    const ICON_GEAR = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h0a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h0a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v0a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>';
    const ICON_TORCH = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 2h6l-1 6h-4z"/><path d="M8 8h8l-2 8h-4z"/><path d="M11 16v6"/></svg>';

    function ensureFullscreenClass() {
        document.body.classList.add('video-fullscreen');
        document.body.classList.add('mobile-omni');
    }

    // -------- back button (top-left) --------
    function injectBackButton() {
        const container = document.getElementById('videoContainer');
        if (!container || container.querySelector('.mobile-back-btn')) return;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'mobile-back-btn';
        btn.setAttribute('aria-label', 'Back to mobile turn page');
        btn.innerHTML = ICON_BACK;
        const goBack = (e) => {
            if (e) {
                e.preventDefault();
                e.stopPropagation();
            }
            try {
                window.location.assign(BACK_URL);
            } catch (_) {
                window.location.href = BACK_URL;
            }
        };
        btn.addEventListener('click', goBack);
        btn.addEventListener('touchend', goBack, { passive: false });
        container.appendChild(btn);
    }

    // -------- settings sheet --------
    function buildSettingsSheet() {
        if (document.getElementById('mobileSettingsSheet')) return;

        const backdrop = document.createElement('div');
        backdrop.className = 'mobile-settings-backdrop';
        backdrop.id = 'mobileSettingsBackdrop';

        const sheet = document.createElement('div');
        sheet.className = 'mobile-settings-sheet';
        sheet.id = 'mobileSettingsSheet';
        sheet.innerHTML = `
            <div class="ms-handle"></div>
            <h3>设置</h3>
            <div class="ms-row">
                <label for="mSystemPrompt">System Prompt</label>
                <textarea id="mSystemPrompt" rows="3" placeholder="System prompt..."></textarea>
            </div>
            <div class="ms-row ms-inline">
                <label for="mLengthPenalty">Length Penalty</label>
                <input type="number" id="mLengthPenalty" min="0.1" max="5" step="0.05">
            </div>
            <div class="ms-row ms-inline">
                <label for="mPlaybackDelay">Playback Delay (ms)</label>
                <input type="number" id="mPlaybackDelay" min="0" max="2000" step="50">
            </div>
            <div class="ms-row ms-inline">
                <label for="mMaxKv">Max KV (tok)</label>
                <input type="number" id="mMaxKv" min="512" max="8192" step="512">
            </div>
            <button type="button" class="ms-close" id="mSettingsClose">完成</button>
        `;

        document.body.appendChild(backdrop);
        document.body.appendChild(sheet);

        // wire two-way bridges to the underlying desktop inputs
        bindMirror('mSystemPrompt', 'systemPrompt');
        bindMirror('mLengthPenalty', 'omniLengthPenalty');
        bindMirror('mPlaybackDelay', 'playbackDelay');
        bindMirror('mMaxKv', 'maxKvTokens');

        const close = () => {
            backdrop.classList.remove('open');
            sheet.classList.remove('open');
        };
        backdrop.addEventListener('click', close);
        document.getElementById('mSettingsClose').addEventListener('click', close);
    }

    function bindMirror(mobileId, desktopId) {
        const m = document.getElementById(mobileId);
        const d = document.getElementById(desktopId);
        if (!m || !d) return;
        // initial sync: desktop -> mobile
        m.value = d.value;
        // mobile -> desktop on every input
        m.addEventListener('input', () => {
            d.value = m.value;
            d.dispatchEvent(new Event('input', { bubbles: true }));
            d.dispatchEvent(new Event('change', { bubbles: true }));
        });
        // also reflect desktop changes (e.g. when a preset auto-fills)
        d.addEventListener('input', () => {
            if (m.value !== d.value) m.value = d.value;
        });
        d.addEventListener('change', () => {
            if (m.value !== d.value) m.value = d.value;
        });
    }

    function openSettingsSheet() {
        const backdrop = document.getElementById('mobileSettingsBackdrop');
        const sheet = document.getElementById('mobileSettingsSheet');
        if (!backdrop || !sheet) return;
        // refresh from desktop in case anything changed
        ['mSystemPrompt:systemPrompt', 'mLengthPenalty:omniLengthPenalty',
         'mPlaybackDelay:playbackDelay', 'mMaxKv:maxKvTokens'].forEach((pair) => {
            const [mid, did] = pair.split(':');
            const m = document.getElementById(mid);
            const d = document.getElementById(did);
            if (m && d) m.value = d.value;
        });
        backdrop.classList.add('open');
        sheet.classList.add('open');
    }

    // -------- ⚙️ on the bottom-right --------
    function rebindFullscreenButtonAsGear() {
        const btn = document.getElementById('fullscreenBtn');
        if (!btn) return;
        // clone removes all listeners registered by omni-app.js's IIFE
        const clone = btn.cloneNode(false);
        clone.id = 'fullscreenBtn'; // keep id so omni-app.js's visibility toggles still find it
        clone.className = btn.className;
        clone.classList.add('visible');
        clone.setAttribute('title', 'Settings');
        clone.setAttribute('aria-label', 'Open settings');
        clone.innerHTML = ICON_GEAR;
        btn.parentNode.replaceChild(clone, btn);
        const open = (e) => {
            if (e) {
                e.preventDefault();
                e.stopPropagation();
            }
            openSettingsSheet();
        };
        clone.addEventListener('click', open);
        clone.addEventListener('touchend', open, { passive: false });
    }

    // -------- session storage bridge from /mobile/ -> systemPrompt --------
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
        if (track === lastVideoTrack) {
            // same track, just re-evaluate capability visibility
            updateTorchVisibility(track);
            return;
        }
        lastVideoTrack = track;
        // turn off torch state when camera changes
        torchBtn?.classList.remove('active');
        updateTorchVisibility(track);
    }

    function updateTorchVisibility(track) {
        if (!torchBtn) return;
        let supported = false;
        if (track && typeof track.getCapabilities === 'function') {
            try {
                const caps = track.getCapabilities();
                supported = !!(caps && caps.torch);
            } catch (_) {
                supported = false;
            }
        }
        torchBtn.classList.toggle('visible', supported);
    }

    function watchVideoElement() {
        const v = document.getElementById('videoEl');
        if (!v) return;
        // poll srcObject changes every 1s (cheap; no MediaStream change events)
        setInterval(refreshCameraCapabilities, 1000);
    }

    // -------- pinch zoom on the video container --------
    function distance(t1, t2) {
        const dx = t1.clientX - t2.clientX;
        const dy = t1.clientY - t2.clientY;
        return Math.hypot(dx, dy);
    }

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
        // prevent the browser from page-zooming while we're handling the pinch
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
        if (e.touches.length < 2) {
            pinchState = null;
        }
    }

    function bindPinchZoom() {
        const v = document.getElementById('videoContainer');
        if (!v) return;
        v.addEventListener('touchstart', handlePinchStart, { passive: false });
        v.addEventListener('touchmove', handlePinchMove, { passive: false });
        v.addEventListener('touchend', handlePinchEnd, { passive: false });
        v.addEventListener('touchcancel', handlePinchEnd, { passive: false });
    }

    // -------- block desktop omni from killing our gear button visibility --------
    function keepGearVisible() {
        const btn = document.getElementById('fullscreenBtn');
        if (!btn) return;
        // omni-app.js calls .classList.toggle('visible', false) in some states;
        // re-add it so the gear stays accessible.
        const obs = new MutationObserver(() => {
            if (!btn.classList.contains('visible')) {
                btn.classList.add('visible');
            }
        });
        obs.observe(btn, { attributes: true, attributeFilter: ['class'] });
    }

    function init() {
        ensureFullscreenClass();
        // omni-app.js binds its IIFE to #fullscreenBtn after DOMContentLoaded.
        // Run our DOM mutations on the next tick to land after that binding.
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
