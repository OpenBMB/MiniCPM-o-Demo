/**
 * Dynamic navigation bar — fetches enabled apps from /api/apps and renders nav links.
 * If the current page's app is disabled, redirects to home.
 *
 * Usage (ES Module):
 *   import { initNav } from '/static/shared/app-nav.js';
 *   initNav('omni');          // on omni page
 *   initNav(null);            // on home page
 *
 * Usage (classic script):
 *   <script src="/static/shared/app-nav.js"></script>
 *   <script>AppNav.init('turnbased');</script>
 */

import { t } from '/static/shared/i18n-module.js';

const _NAV_SELECTOR = '.nav-links';

// Apps to hide from the global nav (still reachable by direct route).
const _NAV_HIDDEN_APP_IDS = new Set(['half_duplex_audio']);

// Map API app_id → i18n key for display name override.
const _APP_NAME_I18N = {
    turnbased: 'turnbasedChat',
    omni: 'omniFullDuplex',
    audio_duplex: 'audioDuplexTitle',
};

async function _fetchApps() {
    try {
        const resp = await fetch('/api/apps');
        if (!resp.ok) return null;
        const data = await resp.json();
        return data.apps;
    } catch {
        return null;
    }
}

function _renderNav(apps, currentAppId) {
    const navEl = document.querySelector(_NAV_SELECTOR);
    if (!navEl) return;

    const homeActive = !currentAppId ? ' class="active"' : '';

    const extras = `<a href="/mobile">${t.mobile}</a>`;

    const links = apps
        .filter(a => !_NAV_HIDDEN_APP_IDS.has(a.app_id))
        .map(a => {
            const active = a.app_id === currentAppId ? ' class="active"' : '';
            const i18nKey = _APP_NAME_I18N[a.app_id];
            const label = (i18nKey && t[i18nKey]) || a.name;
            return `<a href="${a.route}"${active}>${label}</a>`;
        });

    navEl.innerHTML = `<a href="/"${homeActive}>${t.home}</a>` + extras + links.join('');

}

/**
 * Initialize the navigation bar.
 * @param {string|null} currentAppId - The app_id of the current page, or null for home.
 * @returns {Promise<Array|null>} The list of enabled apps, or null on failure.
 */
async function initNav(currentAppId) {
    const apps = await _fetchApps();
    if (!apps) return null;

    if (currentAppId && !apps.find(a => a.app_id === currentAppId)) {
        window.location.href = '/';
        return null;
    }

    _renderNav(apps, currentAppId);
    return apps;
}

// Support both ES Module and classic script usage
if (typeof window !== 'undefined') {
    window.AppNav = { init: initNav };
}

export { initNav };
