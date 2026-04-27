/**
 * RefAudioPlayer — 自定义 Ref Audio 播放器组件
 *
 * 功能：
 * - 播放/暂停 PCM float32 16kHz base64 音频
 * - 显示文件名、时长
 * - 进度条（可点击跳转）
 * - 上传按钮（重采样到 16kHz mono）
 * - 重置按钮（恢复默认 ref audio）
 * - 支持 light/dark 主题
 *
 * 用法：
 *   const player = new RefAudioPlayer(container, {
 *       theme: 'light',          // 'light' | 'dark'
 *       onUpload(base64, name, duration) { ... },
 *       onRemove() { ... },
 *   });
 *   player.setAudio(base64, name, duration);
 */

/* ── CSS 注入（仅一次） ── */
(function injectCSS() {
    if (document.getElementById('ref-audio-player-css')) return;
    const style = document.createElement('style');
    style.id = 'ref-audio-player-css';
    style.textContent = `
.rap-wrap {
    border-radius: 8px;
    padding: 10px 12px;
    transition: all 0.15s;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* ── light theme ── */
.rap-wrap.rap-light {
    border: 1px dashed #d5d5d0;
    background: #fafaf8;
    color: #2d2d2d;
}
.rap-wrap.rap-light.rap-has-audio {
    border: 1px solid #d4a574;
    background: #faf5f0;
}
.rap-wrap.rap-light .rap-btn {
    background: #f0ece6;
    border: 1px solid #d5d5d0;
    color: #555;
}
.rap-wrap.rap-light .rap-btn:hover { background: #e8e4de; }
.rap-wrap.rap-light .rap-progress-bg { background: #e8e4de; }
.rap-wrap.rap-light .rap-progress-fill { background: #d4a574; }
.rap-wrap.rap-light .rap-dur { color: #999; }
.rap-wrap.rap-light .rap-name { color: #2d2d2d; }
.rap-wrap.rap-light .rap-empty { color: #999; }

/* ── dark theme ── */
.rap-wrap.rap-dark {
    border: 1px dashed #444;
    background: #1e1e1e;
    color: #ddd;
}
.rap-wrap.rap-dark.rap-has-audio {
    border: 1px solid #d4a574;
    background: #2a2520;
}
.rap-wrap.rap-dark .rap-btn {
    background: #2a2a2a;
    border: 1px solid #444;
    color: #aaa;
}
.rap-wrap.rap-dark .rap-btn:hover { background: #3a3a3a; }
.rap-wrap.rap-dark .rap-progress-bg { background: #333; }
.rap-wrap.rap-dark .rap-progress-fill { background: #d4a574; }
.rap-wrap.rap-dark .rap-dur { color: #888; }
.rap-wrap.rap-dark .rap-name { color: #e0e0e0; }
.rap-wrap.rap-dark .rap-empty { color: #666; }

/* ── 布局 ── */
.rap-row {
    display: flex;
    align-items: center;
    gap: 8px;
}
.rap-play-btn {
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    flex-shrink: 0;
    font-size: 12px;
    transition: all 0.12s;
    border: none;
    user-select: none;
}
.rap-info {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.rap-meta {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    line-height: 1;
}
.rap-name {
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 180px;
}
.rap-dur {
    font-size: 11px;
    white-space: nowrap;
    flex-shrink: 0;
}
.rap-progress-bg {
    width: 100%;
    height: 4px;
    border-radius: 2px;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}
.rap-progress-fill {
    height: 100%;
    border-radius: 2px;
    width: 0%;
    transition: width 0.05s linear;
}
.rap-actions {
    display: flex;
    align-items: center;
    gap: 4px;
    flex-shrink: 0;
}
.rap-btn {
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 4px;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.12s;
    line-height: 1.2;
}
.rap-btn-remove {
    background: transparent !important;
    border-color: #cf222e !important;
    color: #cf222e !important;
    padding: 3px 5px;
}
.rap-btn-remove:hover {
    background: rgba(207, 34, 46, 0.1) !important;
}

/* ── empty state ── */
.rap-empty {
    text-align: center;
    font-size: 12px;
    cursor: pointer;
    padding: 4px 0;
}
.rap-empty-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}

/* hidden file input */
.rap-file-input { display: none; }
`;
    document.head.appendChild(style);
})();


function _rapT(key, fallback) {
    return window.I18n?.t?.[key] ?? fallback;
}

class RefAudioPlayer {
    /**
     * @param {HTMLElement} container
     * @param {Object} options
     * @param {'light'|'dark'} [options.theme='light']
     * @param {Function} [options.onUpload]  (base64, name, duration) => void
     * @param {Function} [options.onRemove]  () => void
     */
    constructor(container, options = {}) {
        this.container = container;
        this.theme = options.theme || 'light';
        this.onUpload = options.onUpload || (() => {});
        this.onRemove = options.onRemove || (() => {});

        // audio state
        this._base64 = null;
        this._name = '';
        this._duration = 0;

        // playback state
        this._playing = false;
        this._audioCtx = null;
        this._sourceNode = null;
        this._audioBuffer = null;
        this._startTime = 0;       // audioCtx.currentTime when playback started
        this._startOffset = 0;     // offset in seconds (for resume after seek)
        this._rafId = null;

        this._render();
    }

    /** 设置音频数据 */
    setAudio(base64, name, duration) {
        this._stop();
        this._base64 = base64;
        this._name = name || 'audio';
        this._duration = duration || 0;
        this._audioBuffer = null; // invalidate decoded buffer
        this._updateUI();
    }

    /** 清除 */
    clear() {
        this._stop();
        this._base64 = null;
        this._name = '';
        this._duration = 0;
        this._audioBuffer = null;
        this._updateUI();
    }

    // ── 内部：渲染 ──

    _render() {
        const wrap = document.createElement('div');
        wrap.className = `rap-wrap rap-${this.theme}`;

        // file input (hidden)
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'audio/*';
        fileInput.className = 'rap-file-input';
        fileInput.addEventListener('change', (e) => this._handleUpload(e));
        wrap.appendChild(fileInput);
        this._fileInput = fileInput;

        // player content (filled by _updateUI)
        const content = document.createElement('div');
        content.className = 'rap-content';
        wrap.appendChild(content);
        this._contentEl = content;

        this.container.innerHTML = '';
        this.container.appendChild(wrap);
        this._wrapEl = wrap;

        this._updateUI();
    }

    _updateUI() {
        if (this._base64) {
            this._renderPlayer();
        } else {
            this._renderEmpty();
        }
    }

    _renderEmpty() {
        this._wrapEl.classList.remove('rap-has-audio');
        this._contentEl.innerHTML = '';

        const row = document.createElement('div');
        row.className = 'rap-empty';
        row.innerHTML = `
            <div class="rap-empty-row">
                <span>${_rapT('noRefAudio', 'No reference audio')}</span>
                <span class="rap-btn" style="display:inline-block">${_rapT('upload', 'Upload')}</span>
            </div>
        `;
        row.addEventListener('click', () => this._fileInput.click());
        this._contentEl.appendChild(row);
    }

    _renderPlayer() {
        this._wrapEl.classList.add('rap-has-audio');
        this._contentEl.innerHTML = '';

        const row = document.createElement('div');
        row.className = 'rap-row';

        // play button
        const playBtn = document.createElement('button');
        playBtn.className = `rap-play-btn rap-btn`;
        playBtn.textContent = '▶';
        playBtn.title = 'Play / Pause';
        playBtn.addEventListener('click', () => this._togglePlay());
        row.appendChild(playBtn);
        this._playBtn = playBtn;

        // info section
        const info = document.createElement('div');
        info.className = 'rap-info';

        // meta: name + duration
        const meta = document.createElement('div');
        meta.className = 'rap-meta';
        const nameEl = document.createElement('span');
        nameEl.className = 'rap-name';
        nameEl.textContent = this._name;
        nameEl.title = this._name;
        meta.appendChild(nameEl);

        const durEl = document.createElement('span');
        durEl.className = 'rap-dur';
        durEl.textContent = this._formatTime(this._duration);
        meta.appendChild(durEl);
        this._durEl = durEl;
        info.appendChild(meta);

        // progress bar
        const progressBg = document.createElement('div');
        progressBg.className = 'rap-progress-bg';
        progressBg.addEventListener('click', (e) => this._handleSeek(e));
        const progressFill = document.createElement('div');
        progressFill.className = 'rap-progress-fill';
        progressBg.appendChild(progressFill);
        info.appendChild(progressBg);
        this._progressFill = progressFill;
        this._progressBg = progressBg;

        row.appendChild(info);

        // action buttons
        const actions = document.createElement('div');
        actions.className = 'rap-actions';

        const uploadBtn = document.createElement('span');
        uploadBtn.className = 'rap-btn';
        uploadBtn.textContent = _rapT('upload', 'Upload');
        uploadBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this._fileInput.click();
        });
        actions.appendChild(uploadBtn);

        const removeBtn = document.createElement('span');
        removeBtn.className = 'rap-btn rap-btn-remove';
        removeBtn.textContent = '✕';
        removeBtn.title = _rapT('resetToDefault', 'Reset to default');
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this._stop();
            this.onRemove();
        });
        actions.appendChild(removeBtn);

        row.appendChild(actions);
        this._contentEl.appendChild(row);
    }

    // ── 内部：播放 ──

    async _togglePlay() {
        if (this._playing) {
            this._stop();
            return;
        }
        if (!this._base64) return;

        try {
            if (!this._audioCtx) {
                this._audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (this._audioCtx.state === 'suspended') {
                await this._audioCtx.resume();
            }

            // decode if needed
            if (!this._audioBuffer) {
                this._audioBuffer = await this._decodeBase64PCM(this._base64);
            }

            // create source
            this._sourceNode = this._audioCtx.createBufferSource();
            this._sourceNode.buffer = this._audioBuffer;
            this._sourceNode.connect(this._audioCtx.destination);
            this._sourceNode.onended = () => {
                if (this._playing) {
                    this._playing = false;
                    this._startOffset = 0;
                    this._updatePlayBtn();
                    this._setProgress(0);
                    this._stopRAF();
                }
            };

            this._startTime = this._audioCtx.currentTime;
            this._sourceNode.start(0, this._startOffset);
            this._playing = true;
            this._updatePlayBtn();
            this._startRAF();
        } catch (e) {
            console.error('RefAudioPlayer: playback failed', e);
        }
    }

    _stop() {
        if (this._sourceNode) {
            try { this._sourceNode.stop(); } catch (_) {}
            try { this._sourceNode.disconnect(); } catch (_) {}
            this._sourceNode = null;
        }
        if (this._playing) {
            // save current offset for resume
            this._startOffset = 0;
        }
        this._playing = false;
        this._updatePlayBtn();
        this._setProgress(0);
        this._stopRAF();
    }

    _handleSeek(e) {
        if (!this._duration) return;
        const rect = this._progressBg.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const seekTo = ratio * this._duration;

        if (this._playing) {
            // restart from new position
            if (this._sourceNode) {
                try { this._sourceNode.stop(); } catch (_) {}
                try { this._sourceNode.disconnect(); } catch (_) {}
            }
            this._startOffset = seekTo;
            this._playing = false;
            this._togglePlay(); // restart
        } else {
            this._startOffset = seekTo;
            this._setProgress(ratio * 100);
        }
    }

    /** Decode PCM float32 16kHz base64 → AudioBuffer (at device sample rate) */
    async _decodeBase64PCM(base64) {
        // decode base64 → Float32Array (16kHz mono)
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const pcm16k = new Float32Array(bytes.buffer);

        // create AudioBuffer at 16kHz
        const sr = 16000;
        const buf = this._audioCtx.createBuffer(1, pcm16k.length, sr);
        buf.getChannelData(0).set(pcm16k);
        return buf;
    }

    // ── 内部：UI 辅助 ──

    _updatePlayBtn() {
        if (!this._playBtn) return;
        this._playBtn.textContent = this._playing ? '⏸' : '▶';
    }

    _setProgress(pct) {
        if (this._progressFill) {
            this._progressFill.style.width = Math.min(100, Math.max(0, pct)) + '%';
        }
    }

    _startRAF() {
        this._stopRAF();
        const tick = () => {
            if (!this._playing || !this._audioCtx) return;
            const elapsed = (this._audioCtx.currentTime - this._startTime) + this._startOffset;
            const pct = this._duration > 0 ? (elapsed / this._duration) * 100 : 0;
            this._setProgress(pct);

            // update duration display with current time
            if (this._durEl) {
                this._durEl.textContent = `${this._formatTime(elapsed)} / ${this._formatTime(this._duration)}`;
            }
            this._rafId = requestAnimationFrame(tick);
        };
        this._rafId = requestAnimationFrame(tick);
    }

    _stopRAF() {
        if (this._rafId) {
            cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }
        // restore duration display
        if (this._durEl && !this._playing) {
            this._durEl.textContent = this._formatTime(this._duration);
        }
    }

    _formatTime(sec) {
        if (!sec || sec <= 0) return '0.0s';
        if (sec < 60) return sec.toFixed(1) + 's';
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    }

    // ── 内部：上传处理 ──

    async _handleUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        event.target.value = '';

        try {
            const arrayBuffer = await file.arrayBuffer();
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const decoded = await audioCtx.decodeAudioData(arrayBuffer.slice(0));

            if (decoded.duration > 20) {
                const msg = typeof window.I18n?.t?.audioTooLong === 'function'
                    ? window.I18n.t.audioTooLong(decoded.duration.toFixed(1), 20)
                    : `Audio too long: ${decoded.duration.toFixed(1)}s (max 20s)`;
                alert(msg);
                audioCtx.close();
                return;
            }

            // 重采样到 16kHz mono → PCM float32 base64
            const offlineCtx = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
            const source = offlineCtx.createBufferSource();
            source.buffer = decoded;
            source.connect(offlineCtx.destination);
            source.start();
            const resampled = await offlineCtx.startRendering();
            audioCtx.close();

            const pcm = resampled.getChannelData(0);
            const bytes = new Uint8Array(pcm.buffer);
            let binary = '';
            for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
            const base64 = btoa(binary);

            // stop current playback
            this._stop();

            // update state
            this._base64 = base64;
            this._name = file.name;
            this._duration = decoded.duration;
            this._audioBuffer = null; // will re-decode on play
            this._updateUI();

            // notify parent
            this.onUpload(base64, file.name, decoded.duration);
        } catch (e) {
            const msg = typeof window.I18n?.t?.processAudioFailed === 'function'
                ? window.I18n.t.processAudioFailed(e.message)
                : 'Failed to process audio: ' + e.message;
            alert(msg);
        }
    }
}

// Expose to ES Modules (class declarations don't auto-attach to window)
window.RefAudioPlayer = RefAudioPlayer;
