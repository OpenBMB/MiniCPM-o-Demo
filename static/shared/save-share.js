/**
 * Session Save & Share 共享组件
 *
 * 聊天页面：Upload & Share 按钮 → 有前端录制 blob 时先上传再复制链接，无 blob 直接复制链接 + 5s toast。
 * 首页：读取 localStorage 展示 Recent Sessions 列表。
 *
 * 使用方式（聊天页面）：
 *   const ui = new SaveShareUI({
 *     containerId: 'save-share-container',
 *     appType: 'omni_duplex',
 *     collectComment: true, // 可选：分享前弹评语对话框
 *   });
 *   ui.setSessionId('omni_xxx');
 *   // 前端录制完成后：
 *   ui.setRecordingBlob(blob, 'webm');
 *
 * 协议（移动端、桌面端均共用）：
 *   POST /api/sessions/{id}/upload-recording  (FormData: file)
 *   POST /api/sessions/{id}/comment           (JSON:    {comment: string})
 *   GET  /api/sessions/{id}/comment           → {comment: string}
 *   localStorage["minicpmo45_recent_sessions"]: [{id, appType, savedAt}, ...]  (最多 20 条)
 */

const RECENT_SESSIONS_KEY = 'minicpmo45_recent_sessions';
const MAX_RECENT = 20;

/* ─────────── 静态 helpers（与移动端共用同一协议） ─────────── */
async function uploadRecording(sessionId, blob, ext) {
    const form = new FormData();
    form.append('file', blob, `recording.${ext || 'webm'}`);
    const resp = await fetch(`/api/sessions/${sessionId}/upload-recording`, {
        method: 'POST', body: form,
    });
    if (!resp.ok) {
        const detail = await resp.text().catch(() => '');
        throw new Error(`Upload failed: ${resp.status} ${detail}`);
    }
    return resp.json().catch(() => ({}));
}

async function saveComment(sessionId, comment) {
    const text = (comment || '').trim();
    const resp = await fetch(`/api/sessions/${sessionId}/comment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comment: text }),
    });
    if (!resp.ok) {
        const detail = await resp.text().catch(() => '');
        throw new Error(`Comment save failed: ${resp.status} ${detail}`);
    }
    return resp.json().catch(() => ({}));
}

async function fetchComment(sessionId) {
    const resp = await fetch(`/api/sessions/${sessionId}/comment`);
    if (!resp.ok) return '';
    const j = await resp.json().catch(() => ({ comment: '' }));
    return j.comment || '';
}

function buildShareUrl(sessionId) {
    return `${window.location.origin}/s/${sessionId}`;
}

function addToRecent(sessionId, appType) {
    if (!sessionId) return;
    let list = [];
    try { list = JSON.parse(localStorage.getItem(RECENT_SESSIONS_KEY) || '[]'); } catch { list = []; }
    if (!Array.isArray(list)) list = [];
    const idx = list.findIndex(s => s && s.id === sessionId);
    if (idx !== -1) list.splice(idx, 1);
    list.unshift({ id: sessionId, appType: appType || 'unknown', savedAt: new Date().toISOString() });
    if (list.length > MAX_RECENT) list.length = MAX_RECENT;
    try { localStorage.setItem(RECENT_SESSIONS_KEY, JSON.stringify(list)); } catch { /* quota */ }
}

function getRecentSessions() {
    try { return JSON.parse(localStorage.getItem(RECENT_SESSIONS_KEY) || '[]'); }
    catch { return []; }
}

function clearRecentSessions() {
    localStorage.removeItem(RECENT_SESSIONS_KEY);
}

class SaveShareUI {
    /**
     * @param {Object} opts
     * @param {string} opts.containerId - 挂载容器的 DOM id
     * @param {string} [opts.appType] - 应用类型标识
     * @param {boolean} [opts.collectComment=false] - 分享前弹评语对话框
     */
    constructor(opts) {
        this.appType = opts.appType || 'unknown';
        this.collectComment = !!opts.collectComment;
        this._sessionId = null;
        this._recordingBlob = null;
        this._recordingExt = null;
        this._uploading = false;
        this._container = document.getElementById(opts.containerId);
        if (!this._container) return;
        this._render();
    }

    setSessionId(sessionId) {
        this._sessionId = sessionId;
        this._updateBtn();
    }

    /**
     * 设置前端录制的 Blob，Upload & Share 时会先上传此文件
     * @param {Blob} blob - 录制的音频/视频 blob
     * @param {string} ext - 文件扩展名 ('webm', 'wav', 'mp4')
     */
    setRecordingBlob(blob, ext) {
        this._recordingBlob = blob;
        this._recordingExt = ext || 'webm';
        this._updateBtn();
    }

    _updateBtn() {
        const btn = this._container?.querySelector('.ss-btn');
        if (btn) {
            btn.disabled = !this._sessionId || this._uploading;
            btn.textContent = this._uploading ? 'Uploading...' : 'Upload & Share';
        }
    }

    _render() {
        this._container.innerHTML = `
            <button class="ss-btn" disabled>Upload & Share</button>
            <div class="ss-toast" style="display:none;"></div>
        `;
        this._container.querySelector('.ss-btn').addEventListener('click', () => this._onClick());
    }

    async _onClick() {
        if (!this._sessionId || this._uploading) return;
        if (!this.collectComment) {
            return this._doShare(null);
        }
        const comment = await this._promptComment();
        if (comment === null) return; // 用户取消
        return this._doShare(comment);
    }

    /** 弹一个评语对话框，resolve(string) 或 resolve(null) 取消。 */
    _promptComment() {
        return new Promise((resolve) => {
            const overlay = document.createElement('div');
            overlay.className = 'ss-modal-overlay';
            overlay.innerHTML = `
                <div class="ss-modal" role="dialog" aria-modal="true" aria-labelledby="ss-modal-title">
                    <div class="ss-modal-title" id="ss-modal-title">分享对话</div>
                    <div class="ss-modal-hint">可以加一句评语（可选），帮助回看时记住这是哪段对话。</div>
                    <textarea class="ss-modal-input" maxlength="2000" placeholder="评语（可选）"></textarea>
                    <div class="ss-modal-actions">
                        <button type="button" class="ss-modal-btn ss-modal-cancel">取消</button>
                        <button type="button" class="ss-modal-btn ss-modal-ok">分享</button>
                    </div>
                </div>
            `;
            const close = (val) => {
                overlay.removeEventListener('keydown', keyHandler);
                overlay.remove();
                resolve(val);
            };
            const keyHandler = (e) => {
                if (e.key === 'Escape') { e.preventDefault(); close(null); }
                else if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                    e.preventDefault();
                    close(overlay.querySelector('.ss-modal-input').value);
                }
            };
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) close(null);
            });
            overlay.querySelector('.ss-modal-cancel').addEventListener('click', () => close(null));
            overlay.querySelector('.ss-modal-ok').addEventListener('click', () => {
                close(overlay.querySelector('.ss-modal-input').value);
            });
            overlay.addEventListener('keydown', keyHandler);
            document.body.appendChild(overlay);
            // Focus & select for snappy typing
            const ta = overlay.querySelector('.ss-modal-input');
            setTimeout(() => ta.focus(), 0);
        });
    }

    async _doShare(comment) {
        const sid = this._sessionId;
        const url = buildShareUrl(sid);

        // 1. Save comment first (best-effort; failure shouldn't block link).
        if (comment !== null && comment !== undefined && String(comment).trim()) {
            try { await saveComment(sid, comment); }
            catch (e) {
                console.warn('[SaveShare] saveComment failed:', e);
                this._showToast(`评语保存失败：${e.message}\n继续生成分享链接...`, true);
            }
        }

        // 2. Upload front-end recording blob if any.
        if (this._recordingBlob && this._recordingBlob.size > 0) {
            this._uploading = true;
            this._updateBtn();
            this._showToast('上传前端录制中...');
            try {
                await uploadRecording(sid, this._recordingBlob, this._recordingExt);
            } catch (e) {
                console.error('[SaveShare] upload error:', e);
                this._showToast(`上传失败: ${e.message}\n链接仍可用（无前端录制）: ${url}`, true);
                this._uploading = false;
                this._updateBtn();
                addToRecent(sid, this.appType);
                return;
            }
            this._uploading = false;
            this._updateBtn();
        }

        // 3. Add to recent + copy link.
        addToRecent(sid, this.appType);
        navigator.clipboard.writeText(url).then(() => {
            this._showToast(`已复制到剪贴板\n${url}`);
        }).catch(() => {
            this._showToast(`分享链接: ${url}`, true);
        });
    }

    _showToast(text, isManual) {
        const toast = this._container?.querySelector('.ss-toast');
        if (!toast) return;
        toast.textContent = text;
        toast.style.display = 'block';
        toast.classList.toggle('manual', !!isManual);
        clearTimeout(this._toastTimer);
        this._toastTimer = setTimeout(() => { toast.style.display = 'none'; }, 5000);
    }

    static getRecentSessions() { return getRecentSessions(); }
    static clearRecentSessions() { clearRecentSessions(); }
    static buildShareUrl(sid) { return buildShareUrl(sid); }
    static addToRecent(sid, appType) { addToRecent(sid, appType); }
    static uploadRecording(sid, blob, ext) { return uploadRecording(sid, blob, ext); }
    static saveComment(sid, comment) { return saveComment(sid, comment); }
    static fetchComment(sid) { return fetchComment(sid); }
}

/* Inject minimal styles */
(function() {
    if (document.getElementById('ss-styles')) return;
    const s = document.createElement('style');
    s.id = 'ss-styles';
    s.textContent = `
        .ss-btn {
            padding: 7px 16px; border-radius: 8px; font-size: 13px; font-weight: 500;
            background: #2d2d2d; color: #fff; border: none; cursor: pointer;
            transition: opacity 0.15s;
        }
        .ss-btn:disabled { opacity: 0.35; cursor: not-allowed; }
        .ss-btn:not(:disabled):hover { opacity: 0.8; }
        .ss-toast {
            position: fixed; bottom: 56px; right: 16px;
            background: rgba(30,30,30,0.92); color: #fff;
            padding: 10px 16px; border-radius: 10px; font-size: 13px;
            line-height: 1.5; white-space: pre-line; max-width: 360px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.18);
            animation: ss-fade-in 0.2s ease;
            z-index: 10000;
        }
        .ss-toast.manual { user-select: all; }
        @keyframes ss-fade-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

        .ss-modal-overlay {
            position: fixed; inset: 0; background: rgba(20,20,20,0.45);
            display: flex; align-items: center; justify-content: center;
            z-index: 10001; padding: 20px;
            animation: ss-fade-in 0.15s ease;
        }
        .ss-modal {
            background: #fff; border-radius: 14px; padding: 20px 22px;
            width: 100%; max-width: 420px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.25);
            display: flex; flex-direction: column; gap: 12px;
        }
        .ss-modal-title { font-size: 16px; font-weight: 600; color: #1c1c1c; }
        .ss-modal-hint { font-size: 12.5px; color: #666; line-height: 1.45; }
        .ss-modal-input {
            min-height: 90px; max-height: 220px; resize: vertical;
            padding: 10px 12px; border: 1px solid #d4d4cf; border-radius: 8px;
            font-size: 13.5px; line-height: 1.5; font-family: inherit;
            outline: none; transition: border-color 0.15s;
            color: #1c1c1c; background: #fff;
        }
        .ss-modal-input:focus { border-color: #2d2d2d; }
        .ss-modal-actions { display: flex; justify-content: flex-end; gap: 8px; }
        .ss-modal-btn {
            padding: 7px 16px; border-radius: 8px; font-size: 13px;
            font-weight: 500; cursor: pointer; border: 1px solid transparent;
            transition: opacity 0.15s, background 0.15s;
        }
        .ss-modal-cancel { background: #f3f3ee; color: #444; border-color: #e5e5e0; }
        .ss-modal-cancel:hover { background: #e8e8e3; }
        .ss-modal-ok { background: #2d2d2d; color: #fff; }
        .ss-modal-ok:hover { opacity: 0.85; }
    `;
    document.head.appendChild(s);
})();
