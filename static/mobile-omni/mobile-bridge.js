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
    // Outward-pointing arrow over a tray — matches iOS-style "share" idiom
    // and is recognizable at 22px.
    const ICON_SHARE = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v12"/><path d="m7 8 5-5 5 5"/><path d="M5 14v5a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-5"/></svg>';

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
    // DEBUG probe: a magenta square right next to the gear, sharing the
    // gear's exact parent (#videoContainer) and coordinate system. If the
    // user can see it, then the spot itself is reachable and the previous
    // share-button invisibility was specifically about #save-share-container
    // living at <body> level. If the user CAN'T see this either, then the
    // spot itself is being covered by something on their device and we need
    // to chase z-index / stacking context.
    // ========================================================================
    function injectProbeButton() {
        const container = document.getElementById('videoContainer');
        if (!container) return;
        if (container.querySelector('.mb-probe-btn')) return;
        const probe = document.createElement('button');
        probe.type = 'button';
        probe.className = 'mb-probe-btn';
        probe.setAttribute('aria-label', 'probe');
        probe.title = 'probe';
        // Inline styles so we don't depend on external CSS load order
        // for this particular diagnostic.
        probe.style.cssText = [
            'position:absolute',
            'bottom:calc(env(safe-area-inset-bottom, 0px) + 70px)',
            'right:56px',
            'width:36px',
            'height:36px',
            'border:0',
            'border-radius:10px',
            'background:#ff00ff',
            'color:#fff',
            'font-size:14px',
            'font-weight:700',
            'z-index:9999',
            'display:flex',
            'align-items:center',
            'justify-content:center',
            'cursor:pointer',
            'box-shadow:0 0 0 2px #fff, 0 0 8px rgba(255,0,255,0.7)',
        ].join(';');
        probe.textContent = 'P';
        probe.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            alert('probe clicked — this spot IS reachable');
        });
        container.appendChild(probe);
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
    // Share — surfaced inside the settings sheet (proxies clicks to the
    // hidden #save-share-container .ss-btn from save-share.js).
    //
    // Why: the original button placed on the camera stage proved hard to
    // make reliably visible across mobile devices (CSS override / dark
    // blending issues). The settings sheet is the natural home — users
    // already open it via the gear, and the React widget renders a
    // predictable list of `.settings-section` cards we can append to.
    //
    // What this does: when `.settings-sheet` appears in the DOM, inject
    // a "分享通话" section right after its head. The injected button
    // forwards `click` to `#save-share-container .ss-btn`, which means
    // every downstream behavior (comment modal, upload, copy link,
    // toast) is unchanged from the desktop / turn-based experience.
    //
    // The injected button mirrors the source button's `disabled` state
    // so users get the same "no session yet → grayed out" feedback.
    // ========================================================================
    let shareDomObserver = null;

    function findSourceShareButton() {
        return document.querySelector('#save-share-container .ss-btn');
    }

    function buildShareSection(sourceBtn) {
        const section = document.createElement('div');
        section.className = 'settings-section mb-share-section';
        section.innerHTML = `
            <div class="settings-section-title">分享</div>
            <div class="mb-share-row">
                <button class="secondary-btn compact mb-share-btn" type="button">
                    <span class="mb-share-icon" aria-hidden="true"></span>
                    分享通话
                </button>
                <span class="mb-share-hint"></span>
            </div>
        `;
        const iconHost = section.querySelector('.mb-share-icon');
        if (iconHost) iconHost.innerHTML = ICON_SHARE;
        const btn = section.querySelector('.mb-share-btn');
        const hint = section.querySelector('.mb-share-hint');

        function syncState() {
            const disabled = !sourceBtn || !!sourceBtn.disabled;
            btn.disabled = disabled;
            hint.textContent = disabled
                ? '通话开始后可分享'
                : '上传录制并复制链接，可附评语';
        }
        syncState();
        if (sourceBtn) {
            new MutationObserver(syncState).observe(sourceBtn, {
                attributes: true,
                attributeFilter: ['disabled'],
            });
        }

        btn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (!sourceBtn || sourceBtn.disabled) return;
            // Defer so the click handler returns synchronously before
            // SaveShareUI opens its own modal — keeps event ordering sane
            // on touch devices that synthesize click → focus → blur.
            setTimeout(() => sourceBtn.click(), 0);
        });

        return section;
    }

    function injectShareIntoSheet() {
        const sheet = document.querySelector('.settings-sheet');
        if (!sheet) return false;
        if (sheet.querySelector('.mb-share-section')) return true; // already done
        const sourceBtn = findSourceShareButton();
        // Even if no source button yet, inject a disabled placeholder so
        // the section is visible — it will hot-update once SaveShareUI
        // wires up via the MutationObserver above.
        const section = buildShareSection(sourceBtn);
        const head = sheet.querySelector('.settings-sheet-head');
        if (head?.nextSibling) {
            sheet.insertBefore(section, head.nextSibling);
        } else {
            sheet.appendChild(section);
        }
        return true;
    }

    function watchForSettingsSheet() {
        // The React widget mounts/unmounts `.settings-sheet` each open/close
        // cycle, so we re-inject every time it appears.
        if (shareDomObserver) return;
        shareDomObserver = new MutationObserver(() => {
            injectShareIntoSheet();
        });
        shareDomObserver.observe(document.body, { childList: true, subtree: true });
        // First-shot in case sheet is already mounted.
        injectShareIntoSheet();
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
            watchForSettingsSheet();
            injectProbeButton();
        }, 0);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init, { once: true });
    } else {
        init();
    }
})();
