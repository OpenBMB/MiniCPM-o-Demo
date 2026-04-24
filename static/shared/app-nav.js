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

const _NAV_SELECTOR = '.nav-links';

// Apps to hide from the global nav (still reachable by direct route).
const _NAV_HIDDEN_APP_IDS = new Set(['half_duplex_audio']);

// Extra static links injected after Home and before the dynamic apps list.
const _NAV_EXTRA_LINKS = [
    { route: '/mobile', name: 'Mobile' },
];

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

    const extras = _NAV_EXTRA_LINKS.map(e => {
        return `<a href="${e.route}">${e.name}</a>`;
    });

    const links = apps
        .filter(a => !_NAV_HIDDEN_APP_IDS.has(a.app_id))
        .map(a => {
            const active = a.app_id === currentAppId ? ' class="active"' : '';
            return `<a href="${a.route}"${active}>${a.name}</a>`;
        });

    navEl.innerHTML = `<a href="/"${homeActive}>Home</a>` + extras.join('') + links.join('');
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
