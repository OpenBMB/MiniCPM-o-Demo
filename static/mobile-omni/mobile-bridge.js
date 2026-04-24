/* ============================================================
 * mobile-bridge.js
 *
 * Adapts the desktop omni page DOM (loaded by omni-app.js) for mobile:
 *   1. Force `body.video-fullscreen` so the page boots in fullscreen.
 *   2. Inject a top-left back button (returns to /mobile/).
 *   3. Re-skin bottom-right #fullscreenBtn as a gear icon that opens
 *      the React-based OmniSettingsWidget (shared with mobile app).
 *   4. Camera extras: torch toggle + pinch-to-zoom.
 *   5. Bridge `sessionStorage['mobileOmni:settings']` -> desktop DOM.
 * ========================================================= */

(function bootstrapMobileOmni() {
    const BACK_URL = '/mobile/';

    const ICON_BACK = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"/></svg>';
    const ICON_GEAR = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h0a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h0a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v0a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>';
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
            // Eagerly release camera + mic before navigating away so the OS
            // camera handle is freed by the time the user returns. Without
            // this we observed a black-screen camera on the *second* entry
            // into /mobile-omni/ — beforeunload alone is too late on some
            // mobile browsers (Android WebView / iOS Safari).
            try {
                if (typeof window.__omniCleanupMedia === 'function') {
                    window.__omniCleanupMedia();
                }
            } catch (_) {}
            try { window.location.assign(BACK_URL); } catch (_) { window.location.href = BACK_URL; }
        };
        btn.addEventListener('click', goBack);
        btn.addEventListener('touchend', goBack, { passive: false });
        container.appendChild(btn);
    }

    // ========================================================================
    // React Settings Widget (shared with mobile app)
    // ========================================================================
    let widgetApi = null;

    function getDesktopInput(id) {
        return document.getElementById(id);
    }

    function setDesktopInput(id, value) {
        const el = document.getElementById(id);
        if (!el) return;
        el.value = value;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
    }

    function getDesktopRefAudioPlayer() {
        return document.getElementById('omniRefAudioPlayer');
    }

    function buildBridge() {
        return {
            getSystemPrompt() {
                return getDesktopInput('systemPrompt')?.value || '';
            },
            setSystemPrompt(v) {
                setDesktopInput('systemPrompt', v);
            },
            getLengthPenalty() {
                return parseFloat(getDesktopInput('omniLengthPenalty')?.value) || 1.1;
            },
            setLengthPenalty(v) {
                setDesktopInput('omniLengthPenalty', String(v));
            },
            getPlaybackDelay() {
                return parseInt(getDesktopInput('playbackDelay')?.value, 10) || 0;
            },
            setPlaybackDelay(v) {
                setDesktopInput('playbackDelay', String(v));
            },
            getMaxKv() {
                return parseInt(getDesktopInput('maxKvTokens')?.value, 10) || 8192;
            },
            setMaxKv(v) {
                setDesktopInput('maxKvTokens', String(v));
            },
            getRefAudioBase64() {
                return null;
            },
            setRefAudioBase64(b64, name, duration) {
                const rap = getDesktopRefAudioPlayer();
                if (!rap) return;
                if (b64 === null) {
                    const removeBtn = rap.querySelector('.rap-remove-btn');
                    if (removeBtn) removeBtn.click();
                    return;
                }
                const ev = new CustomEvent('mobileSetRefAudio', {
                    detail: { base64: b64, name: name, duration: duration },
                });
                rap.dispatchEvent(ev);
            },
        };
    }

    function mountSettingsWidget() {
        if (typeof window.mountOmniSettings !== 'function') {
            console.warn('[mobile-omni] settings-widget.js not loaded');
            return;
        }
        const container = document.createElement('div');
        container.id = 'omniSettingsWidgetRoot';
        document.body.appendChild(container);
        const bridge = buildBridge();
        widgetApi = window.mountOmniSettings(container, bridge);
    }

    function openSettingsSheet() {
        if (widgetApi) {
            widgetApi.open();
        }
    }

    // ========================================================================
    // Gear button on bottom-right
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
            setDesktopInput('systemPrompt', s.systemPrompt);
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
        setInterval(refreshCameraCapabilities, 1000);
    }

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
            mountSettingsWidget();
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
