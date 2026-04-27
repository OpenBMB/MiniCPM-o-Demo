/**
 * PresetSelector — System Prompt preset picker with lazy audio loading
 *
 * Loads preset metadata (text only) from /api/presets on init.
 * Audio data is loaded on-demand from /api/presets/{mode}/{id}/audio
 * when a preset is selected, with a progress overlay blocking interaction.
 *
 * onSelect callback receives two args: (preset, { audioLoaded: bool })
 *   - audioLoaded=false: text applied, audio still loading
 *   - audioLoaded=true:  audio data merged into preset, ready to use
 */
class PresetSelector {
    constructor({ container, page, detailsEl, onSelect, storageKey }) {
        this._container = container;
        this._page = page;
        this._detailsEl = detailsEl;
        this._onSelect = onSelect;
        this._storageKey = storageKey || `${page}_preset`;
        this._presets = [];
        this._selectedId = null;
        this._btnRow = null;
        this._advBtn = null;
        this._overlay = null;
        this._loading = false;
    }

    async init() {
        try {
            const resp = await fetch('/api/presets');
            if (!resp.ok) return;
            const data = await resp.json();
            this._presets = data[this._page] || [];
        } catch (e) {
            console.warn('Failed to load presets:', e);
            return;
        }

        if (this._presets.length === 0) {
            this._container.style.display = 'none';
            return;
        }

        this._render();

        const saved = localStorage.getItem(this._storageKey);
        if (saved && this._presets.some(p => p.id === saved)) {
            this.select(saved, false);
        } else {
            this.select(this._presets[0].id, false);
        }
    }

    _t(key, fallback) {
        const i = window.I18n && window.I18n.t;
        return (i && i[key]) || fallback;
    }

    _render() {
        const wrap = document.createElement('div');
        wrap.className = 'preset-selector-wrap';

        const header = document.createElement('div');
        header.className = 'preset-header';
        header.innerHTML = `
            <span class="preset-title">${this._t('presetTitle', 'Preset System Prompt')}</span>
            <span class="preset-subtitle">${this._t('presetSubtitle', 'Controls response language, voice style, rhythm and timbre. Customizable via Advanced. You can customize the reference audio and system prompt in advanced settings. More presets are coming soon.')}</span>
        `;
        wrap.appendChild(header);

        const row = document.createElement('div');
        row.className = 'preset-row';

        const btnRow = document.createElement('div');
        btnRow.className = 'preset-btn-row';
        for (const preset of this._presets) {
            const btn = document.createElement('button');
            btn.className = 'preset-btn';
            btn.dataset.presetId = preset.id;
            btn.textContent = preset.name;
            btn.title = preset.description || preset.name;
            btn.addEventListener('click', () => this.select(preset.id, true));
            btnRow.appendChild(btn);
        }
        row.appendChild(btnRow);

        const advBtn = document.createElement('button');
        advBtn.className = 'preset-adv-btn';
        advBtn.textContent = this._t('presetAdvanced', 'Advanced') + ' \u25BE';
        advBtn.title = this._t('presetAdvancedTooltip', 'Show/hide system prompt details for customization');
        advBtn.addEventListener('click', () => this._toggleAdvanced());
        this._advBtn = advBtn;
        row.appendChild(advBtn);

        wrap.appendChild(row);

        // Loading overlay (hidden by default)
        const overlay = document.createElement('div');
        overlay.className = 'preset-loading-overlay';
        overlay.innerHTML = `
            <div class="preset-loading-inner">
                <div class="preset-loading-bar"><div class="preset-loading-fill"></div></div>
                <span class="preset-loading-text">${this._t('presetLoadingMedia', 'Loading preset media…')}</span>
            </div>
        `;
        overlay.style.display = 'none';
        this._overlay = overlay;
        wrap.appendChild(overlay);

        this._container.appendChild(wrap);
        this._btnRow = btnRow;
    }

    async select(presetId, isUserAction) {
        if (this._loading) return;
        if (this._selectedId === presetId && !isUserAction) return;

        const preset = this._presets.find(p => p.id === presetId);
        if (!preset) return;

        this._selectedId = presetId;
        localStorage.setItem(this._storageKey, presetId);

        for (const btn of this._btnRow.querySelectorAll('.preset-btn')) {
            btn.classList.toggle('active', btn.dataset.presetId === presetId);
        }

        if (this._detailsEl) this._detailsEl.removeAttribute('open');
        if (this._advBtn) this._advBtn.textContent = this._t('presetAdvanced', 'Advanced') + ' \u25BE';

        // Phase 1: apply text immediately
        if (this._onSelect) {
            this._onSelect(preset, { audioLoaded: !!preset._audioLoaded });
        }

        // Phase 2: lazy load audio if needed
        if (!preset._audioLoaded && this._hasAudioFields(preset)) {
            await this._loadAudio(preset);
        }
    }

    _hasAudioFields(preset) {
        if (preset.system_content) {
            if (preset.system_content.some(it => it.type === 'audio' && it.path && !it.data)) return true;
        }
        if (preset.ref_audio && preset.ref_audio.path && !preset.ref_audio.data) return true;
        return false;
    }

    async _loadAudio(preset) {
        this._loading = true;
        this._showOverlay();

        try {
            const resp = await fetch(`/api/presets/${this._page}/${preset.id}/audio`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();

            // Merge audio into system_content
            if (data.system_content_audio && preset.system_content) {
                let audioIdx = 0;
                for (const item of preset.system_content) {
                    if (item.type === 'audio' && item.path && !item.data) {
                        const loaded = data.system_content_audio[audioIdx];
                        if (loaded && loaded.data) {
                            item.data = loaded.data;
                            item.name = loaded.name || item.name;
                            item.duration = loaded.duration || item.duration;
                        }
                        audioIdx++;
                    }
                }
            }

            // Merge ref_audio
            if (data.ref_audio && data.ref_audio.data && preset.ref_audio) {
                preset.ref_audio.data = data.ref_audio.data;
                preset.ref_audio.name = data.ref_audio.name || preset.ref_audio.name;
                preset.ref_audio.duration = data.ref_audio.duration || preset.ref_audio.duration;
            }

            preset._audioLoaded = true;

            // Phase 2 callback: audio now available
            if (this._onSelect && this._selectedId === preset.id) {
                this._onSelect(preset, { audioLoaded: true });
            }
        } catch (e) {
            console.error('[PresetSelector] audio load failed:', e);
        } finally {
            this._loading = false;
            this._hideOverlay();
        }
    }

    _showOverlay() {
        if (!this._overlay) return;
        this._overlay.style.display = '';
        const fill = this._overlay.querySelector('.preset-loading-fill');
        if (fill) {
            fill.style.width = '0%';
            // Animate to 90% over ~2s, the remaining 10% completes on hide
            requestAnimationFrame(() => { fill.style.width = '90%'; });
        }
        // Disable all buttons
        for (const btn of this._btnRow.querySelectorAll('.preset-btn')) btn.disabled = true;
        if (this._advBtn) this._advBtn.disabled = true;
    }

    _hideOverlay() {
        if (!this._overlay) return;
        const fill = this._overlay.querySelector('.preset-loading-fill');
        if (fill) fill.style.width = '100%';
        setTimeout(() => {
            if (this._overlay) this._overlay.style.display = 'none';
            // Re-enable buttons
            for (const btn of this._btnRow.querySelectorAll('.preset-btn')) btn.disabled = false;
            if (this._advBtn) this._advBtn.disabled = false;
        }, 200);
    }

    getSelectedId() {
        return this._selectedId;
    }

    _toggleAdvanced() {
        if (!this._detailsEl) return;
        if (this._detailsEl.hasAttribute('open')) {
            this._detailsEl.removeAttribute('open');
            this._advBtn.textContent = this._t('presetAdvanced', 'Advanced') + ' \u25BE';
        } else {
            this._detailsEl.setAttribute('open', '');
            this._advBtn.textContent = this._t('presetAdvanced', 'Advanced') + ' \u25B4';
        }
    }
}

(function injectPresetCSS() {
    if (document.getElementById('preset-selector-css')) return;
    const style = document.createElement('style');
    style.id = 'preset-selector-css';
    style.textContent = `
.preset-selector-wrap {
    display: flex;
    flex-direction: column;
    gap: 6px;
    position: relative;
}
.preset-header {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.preset-title {
    font-size: 13px;
    font-weight: 600;
    color: #444;
}
.preset-subtitle {
    font-size: 11px;
    color: #aaa;
    line-height: 1.4;
}
.preset-row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.preset-btn-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
}
.preset-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 14px;
    border: 1.5px solid #e0ddd8;
    border-radius: 20px;
    background: #fff;
    font-size: 13px;
    font-weight: 500;
    color: #555;
    cursor: pointer;
    transition: all 0.15s ease;
    font-family: inherit;
    line-height: 1.3;
}
.preset-btn:hover {
    border-color: #bbb;
    background: #fafaf8;
}
.preset-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
.preset-btn.active {
    border-color: #2d2d2d;
    background: #2d2d2d;
    color: #fff;
    outline: 1.5px solid #2d2d2d;
    outline-offset: 2px;
}
.preset-adv-btn {
    padding: 4px 10px;
    border: none;
    background: transparent;
    font-size: 11px;
    color: #999;
    cursor: pointer;
    font-family: inherit;
    transition: color 0.15s;
    white-space: nowrap;
}
.preset-adv-btn:hover {
    color: #666;
}
.preset-adv-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

/* Loading overlay */
.preset-loading-overlay {
    position: absolute;
    inset: 0;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(2px);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10;
}
.preset-loading-inner {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
}
.preset-loading-bar {
    width: 180px;
    height: 4px;
    background: #e5e5e0;
    border-radius: 2px;
    overflow: hidden;
}
.preset-loading-fill {
    height: 100%;
    background: #2d2d2d;
    border-radius: 2px;
    width: 0%;
    transition: width 2s ease-out;
}
.preset-loading-text {
    font-size: 12px;
    color: #888;
}
`;
    document.head.appendChild(style);
})();

window.PresetSelector = PresetSelector;
