/**
 * SystemContentEditor — System Content List 编辑器
 *
 * 将 System Prompt 建模为一个 content list，每个 item 可以是 text 或 audio。
 * 用户可以任意添加、删除、重排项目。
 *
 * 默认结构（模型最佳实践）：
 *   [text] 模仿音频样本的音色并生成新的内容。
 *   [audio] (default ref audio)
 *   [text] 你的任务是用这种声音模式来当一个助手。...
 *
 * 依赖: ref-audio-player.js (RefAudioPlayer)
 *
 * 用法：
 *   const editor = new SystemContentEditor(container, {
 *       theme: 'light',
 *       onChange(items) { ... },
 *   });
 *   editor.setItems([...]);
 *   const items = editor.getItems();
 */

/* ── CSS 注入 ── */
(function injectCSS() {
    if (document.getElementById('sce-css')) return;
    const style = document.createElement('style');
    style.id = 'sce-css';
    style.textContent = `
/* ── 容器 ── */
.sce-wrap {
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* ── 单个 item ── */
.sce-item {
    border-radius: 6px;
    padding: 6px 8px;
    display: flex;
    gap: 6px;
    align-items: flex-start;
    transition: all 0.12s;
}
.sce-item.sce-light {
    background: #f8f7f4;
    border: 1px solid #e8e6e0;
}
.sce-item.sce-light:hover {
    border-color: #ccc;
}
.sce-item.sce-dark {
    background: #222;
    border: 1px solid #3a3a3a;
}
.sce-item.sce-dark:hover {
    border-color: #555;
}

/* ── item 序号 + 类型 ── */
.sce-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 3px;
    white-space: nowrap;
    flex-shrink: 0;
    margin-top: 2px;
    user-select: none;
}
.sce-badge-text.sce-light { background: #e8e4de; color: #666; }
.sce-badge-text.sce-dark  { background: #333; color: #aaa; }
.sce-badge-audio.sce-light { background: #f0e6d8; color: #a07040; }
.sce-badge-audio.sce-dark  { background: #3a2e20; color: #d4a574; }

/* ── 内容区 ── */
.sce-content {
    flex: 1;
    min-width: 0;
}
.sce-textarea {
    width: 100%;
    border-radius: 4px;
    padding: 6px 8px;
    font-size: 12px;
    line-height: 1.5;
    resize: vertical;
    min-height: 28px;
    font-family: inherit;
    transition: border-color 0.15s;
}
.sce-textarea.sce-light {
    border: 1px solid #ddd;
    background: #fff;
    color: #2d2d2d;
}
.sce-textarea.sce-light:focus {
    border-color: #999;
    outline: none;
}
.sce-textarea.sce-dark {
    border: 1px solid #444;
    background: #1a1a1a;
    color: #ddd;
}
.sce-textarea.sce-dark:focus {
    border-color: #666;
    outline: none;
}

/* ── 操作按钮区 ── */
.sce-actions {
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex-shrink: 0;
}
.sce-act-btn {
    width: 20px;
    height: 18px;
    border: none;
    border-radius: 3px;
    cursor: pointer;
    font-size: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.1s;
    padding: 0;
    line-height: 1;
}
.sce-act-btn.sce-light {
    background: #eee;
    color: #888;
}
.sce-act-btn.sce-light:hover { background: #ddd; color: #444; }
.sce-act-btn.sce-dark {
    background: #333;
    color: #888;
}
.sce-act-btn.sce-dark:hover { background: #444; color: #ccc; }
.sce-act-btn:disabled { opacity: 0.3; cursor: default; }
.sce-act-btn.sce-remove {
    color: #cf222e;
}
.sce-act-btn.sce-remove:hover {
    background: rgba(207,34,46,0.1);
}

/* ── 底部添加按钮 ── */
.sce-add-row {
    display: flex;
    gap: 6px;
    margin-top: 2px;
}
.sce-add-btn {
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.12s;
    border: 1px dashed;
}
.sce-add-btn.sce-light {
    border-color: #ddd;
    background: #fafaf8;
    color: #888;
}
.sce-add-btn.sce-light:hover { border-color: #999; background: #f0ece6; color: #555; }
.sce-add-btn.sce-dark {
    border-color: #444;
    background: #1e1e1e;
    color: #888;
}
.sce-add-btn.sce-dark:hover { border-color: #666; background: #2a2a2a; color: #ccc; }

/* ── audio item 内的 RefAudioPlayer 适配 ── */
.sce-audio-container .rap-wrap {
    margin: 0;
    border: none !important;
    padding: 4px 0;
    background: transparent !important;
}
`;
    document.head.appendChild(style);
})();


class SystemContentEditor {
    /**
     * @param {HTMLElement} container
     * @param {Object} options
     * @param {'light'|'dark'} [options.theme='light']
     * @param {Function} [options.onChange] (items) => void
     */
    constructor(container, options = {}) {
        this.container = container;
        this.theme = options.theme || 'light';
        this.onChange = options.onChange || (() => {});

        /**
         * items: Array<{type: 'text', text: string} | {type: 'audio', data: string|null, name: string, duration: number}>
         */
        this._items = [];
        this._players = new Map(); // index -> RefAudioPlayer
        this._rendering = false; // 防止 onChange → setItems → _render 的循环

        this._wrap = document.createElement('div');
        this._wrap.className = 'sce-wrap';
        this.container.innerHTML = '';
        this.container.appendChild(this._wrap);
    }

    /** 获取当前 content list */
    getItems() {
        return JSON.parse(JSON.stringify(this._items));
    }

    /** 设置 content list 并重新渲染（若当前正在渲染或 IME 输入中则跳过） */
    setItems(items) {
        if (this._rendering) return; // 防止循环: onChange → setItems → _render → onChange
        this._destroyPlayers();
        this._items = items.map(it => ({ ...it }));
        this._render();
    }

    /** 更新指定 audio item 的数据（不触发全量重渲染） */
    setAudioData(index, data, name, duration) {
        const item = this._items[index];
        if (!item || item.type !== 'audio') return;
        item.data = data;
        item.name = name || '';
        item.duration = duration || 0;
        // 更新对应的 player
        const player = this._players.get(index);
        if (player) {
            if (data) {
                player.setAudio(data, name, duration);
            } else {
                player.clear();
            }
        }
        this.onChange(this.getItems());
    }

    /** 获取第一个 audio item 的 base64 数据（用于 TTS ref audio） */
    getFirstAudioBase64() {
        for (const item of this._items) {
            if (item.type === 'audio' && item.data) {
                return item.data;
            }
        }
        return null;
    }

    // ── 渲染 ──

    _render() {
        this._rendering = true;
        this._destroyPlayers();
        this._wrap.innerHTML = '';

        this._items.forEach((item, idx) => {
            const el = this._renderItem(item, idx);
            this._wrap.appendChild(el);
        });

        // add buttons row
        const addRow = document.createElement('div');
        addRow.className = 'sce-add-row';

        const addTextBtn = document.createElement('span');
        addTextBtn.className = `sce-add-btn sce-${this.theme}`;
        addTextBtn.textContent = '+ Text';
        addTextBtn.addEventListener('click', () => this._addItem('text'));
        addRow.appendChild(addTextBtn);

        const addAudioBtn = document.createElement('span');
        addAudioBtn.className = `sce-add-btn sce-${this.theme}`;
        addAudioBtn.textContent = '+ Audio';
        addAudioBtn.addEventListener('click', () => this._addItem('audio'));
        addRow.appendChild(addAudioBtn);

        this._wrap.appendChild(addRow);
        this._rendering = false;
    }

    _renderItem(item, idx) {
        const row = document.createElement('div');
        row.className = `sce-item sce-${this.theme}`;

        // badge
        const badge = document.createElement('span');
        if (item.type === 'text') {
            badge.className = `sce-badge sce-badge-text sce-${this.theme}`;
            badge.textContent = `T`;
            badge.title = 'Text';
        } else {
            badge.className = `sce-badge sce-badge-audio sce-${this.theme}`;
            badge.textContent = `A`;
            badge.title = 'Audio';
        }
        row.appendChild(badge);

        // content
        const content = document.createElement('div');
        content.className = 'sce-content';

        if (item.type === 'text') {
            const textarea = document.createElement('textarea');
            textarea.className = `sce-textarea sce-${this.theme}`;
            textarea.value = item.text || '';
            textarea.placeholder = 'Enter text...';
            textarea.rows = 1;
            // auto resize（使用 scrollHeight 差值避免 height:auto 导致的闪烁）
            const autoResize = () => {
                textarea.style.height = '0';
                textarea.style.height = Math.max(28, textarea.scrollHeight) + 'px';
            };
            // IME 输入法支持：composing 期间不触发 onChange（防止 DOM 重建中断输入法）
            let composing = false;
            textarea.addEventListener('compositionstart', () => { composing = true; });
            textarea.addEventListener('compositionend', () => {
                composing = false;
                item.text = textarea.value;
                autoResize();
                this.onChange(this.getItems());
            });
            textarea.addEventListener('input', () => {
                item.text = textarea.value;
                autoResize();
                if (!composing) {
                    this.onChange(this.getItems());
                }
            });
            content.appendChild(textarea);
            // initial auto resize after DOM insertion
            requestAnimationFrame(autoResize);
        } else {
            // audio item - use RefAudioPlayer
            const audioContainer = document.createElement('div');
            audioContainer.className = 'sce-audio-container';
            content.appendChild(audioContainer);

            const player = new RefAudioPlayer(audioContainer, {
                theme: this.theme,
                onUpload: (base64, name, duration) => {
                    item.data = base64;
                    item.name = name;
                    item.duration = duration;
                    this.onChange(this.getItems());
                },
                onRemove: () => {
                    item.data = null;
                    item.name = '';
                    item.duration = 0;
                    this.onChange(this.getItems());
                },
            });
            if (item.data) {
                player.setAudio(item.data, item.name, item.duration);
            }
            this._players.set(idx, player);
        }

        row.appendChild(content);

        // action buttons
        const actions = document.createElement('div');
        actions.className = 'sce-actions';

        // move up
        const upBtn = document.createElement('button');
        upBtn.className = `sce-act-btn sce-${this.theme}`;
        upBtn.textContent = '↑';
        upBtn.title = 'Move up';
        upBtn.disabled = (idx === 0);
        upBtn.addEventListener('click', () => this._moveItem(idx, -1));
        actions.appendChild(upBtn);

        // move down
        const downBtn = document.createElement('button');
        downBtn.className = `sce-act-btn sce-${this.theme}`;
        downBtn.textContent = '↓';
        downBtn.title = 'Move down';
        downBtn.disabled = (idx === this._items.length - 1);
        downBtn.addEventListener('click', () => this._moveItem(idx, 1));
        actions.appendChild(downBtn);

        // delete
        const delBtn = document.createElement('button');
        delBtn.className = `sce-act-btn sce-${this.theme} sce-remove`;
        delBtn.textContent = '✕';
        delBtn.title = 'Remove';
        delBtn.addEventListener('click', () => this._removeItem(idx));
        actions.appendChild(delBtn);

        row.appendChild(actions);
        return row;
    }

    // ── 操作 ──

    _addItem(type) {
        if (type === 'text') {
            this._items.push({ type: 'text', text: '' });
        } else {
            this._items.push({ type: 'audio', data: null, name: '', duration: 0 });
        }
        this._render();
        this.onChange(this.getItems());
    }

    _removeItem(idx) {
        this._items.splice(idx, 1);
        this._render();
        this.onChange(this.getItems());
    }

    _moveItem(idx, dir) {
        const newIdx = idx + dir;
        if (newIdx < 0 || newIdx >= this._items.length) return;
        const tmp = this._items[idx];
        this._items[idx] = this._items[newIdx];
        this._items[newIdx] = tmp;
        this._render();
        this.onChange(this.getItems());
    }

    _destroyPlayers() {
        // RefAudioPlayer doesn't have a destroy method, but clearing the map is enough
        this._players.clear();
    }
}
