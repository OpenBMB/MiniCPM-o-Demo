/* ============================================================
 * mobile-bridge.js
 *
 * Adapts the desktop omni page DOM (loaded by omni-app.js) for mobile:
 *   1. Force `body.video-fullscreen` so the page boots in fullscreen.
 *   2. Replace #fullscreenBtn behavior so tapping it returns to /mobile/
 *      instead of toggling fullscreen off (which would expose the desktop
 *      layout we hide with CSS).
 *   3. Optionally apply systemPrompt / refAudio carried over via
 *      sessionStorage (set by the mobile React app before navigating here).
 * ========================================================= */

(function bootstrapMobileOmni() {
    const BACK_URL = '/mobile/';

    function ensureFullscreenClass() {
        document.body.classList.add('video-fullscreen');
        document.body.classList.add('mobile-omni');
    }

    function rebindFullscreenButton() {
        const btn = document.getElementById('fullscreenBtn');
        if (!btn) return;
        // Clone removes all listeners registered by omni-app.js's IIFE.
        const clone = btn.cloneNode(true);
        btn.parentNode.replaceChild(clone, btn);
        clone.classList.add('visible'); // ensure CSS overrides + base rules agree
        clone.setAttribute('title', 'Back to mobile');
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
        clone.addEventListener('click', goBack);
        clone.addEventListener('touchend', goBack, { passive: false });
    }

    function applyMobileSettings() {
        let settingsRaw;
        try {
            settingsRaw = sessionStorage.getItem('mobileOmni:settings');
        } catch (_) {
            return;
        }
        if (!settingsRaw) return;
        let settings;
        try {
            settings = JSON.parse(settingsRaw);
        } catch (_) {
            return;
        }
        if (!settings || typeof settings !== 'object') return;

        if (typeof settings.systemPrompt === 'string') {
            const ta = document.getElementById('systemPrompt');
            if (ta) {
                ta.value = settings.systemPrompt;
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
        // ref audio bridging is intentionally skipped in v1; the omni page
        // will use its preset default. Hook a future enhancement here.
    }

    function patchFullscreenVisibilityToggler() {
        // omni-app.js calls updateFullscreenBtnVisibility(false) in many
        // states which would hide our exit button. Override globally so
        // it stays visible on mobile.
        const origDescriptor = Object.getOwnPropertyDescriptor(window, 'updateFullscreenBtnVisibility');
        if (origDescriptor) {
            return; // module export — leave alone, CSS forces display:flex
        }
    }

    function init() {
        ensureFullscreenClass();
        // omni-app.js binds its IIFE to #fullscreenBtn after DOMContentLoaded.
        // Run our rebind on the next tick to land after that binding.
        setTimeout(() => {
            rebindFullscreenButton();
            applyMobileSettings();
            patchFullscreenVisibilityToggler();
        }, 0);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init, { once: true });
    } else {
        init();
    }
})();
