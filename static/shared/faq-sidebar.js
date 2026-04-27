/**
 * faq-sidebar.js — VS Code-style embedded FAQ sidebar
 *
 * Usage:
 *   import { initFaqSidebar } from '/static/shared/faq-sidebar.js';
 *   initFaqSidebar('turnbased');   // loads /static/faq/{lang}/turnbased.md
 */

const STORAGE_KEY_LANG = 'minicpmo_lang';

/* ── lightweight markdown → HTML ── */
function md2html(src) {
    const lines = src.split('\n');
    const out = [];
    let inUl = false, inOl = false, inCode = false, codeBuf = [];

    const flush = (tag) => {
        if (tag === 'ul' && inUl) { out.push('</ul>'); inUl = false; }
        if (tag === 'ol' && inOl) { out.push('</ol>'); inOl = false; }
    };
    const flushAll = () => { flush('ul'); flush('ol'); };
    const inline = (t) => t
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    for (let i = 0; i < lines.length; i++) {
        const raw = lines[i];
        if (inCode) {
            if (raw.trimEnd() === '```') {
                out.push('<pre><code>' + codeBuf.join('\n').replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</code></pre>');
                codeBuf = []; inCode = false;
            } else { codeBuf.push(raw); }
            continue;
        }
        if (raw.trimStart().startsWith('```')) { flushAll(); inCode = true; continue; }

        const trimmed = raw.trim();
        if (!trimmed) { flushAll(); continue; }
        if (trimmed.startsWith('---') && trimmed.replace(/-/g, '').trim() === '') { flushAll(); out.push('<hr>'); continue; }

        const hm = trimmed.match(/^(#{1,3})\s+(.*)/);
        if (hm) { flushAll(); const lvl = hm[1].length; out.push(`<h${lvl}>${inline(hm[2])}</h${lvl}>`); continue; }

        if (trimmed.match(/^[-*]\s+/)) {
            flush('ol');
            if (!inUl) { out.push('<ul>'); inUl = true; }
            out.push('<li>' + inline(trimmed.replace(/^[-*]\s+/, '')) + '</li>');
            continue;
        }
        const olm = trimmed.match(/^\d+\.\s+(.*)/);
        if (olm) {
            flush('ul');
            if (!inOl) { out.push('<ol>'); inOl = true; }
            out.push('<li>' + inline(olm[1]) + '</li>');
            continue;
        }
        if (trimmed.startsWith('>')) {
            flushAll();
            let bq = [trimmed.replace(/^>\s?/, '')];
            while (i + 1 < lines.length && lines[i + 1].trim().startsWith('>')) {
                i++; bq.push(lines[i].trim().replace(/^>\s?/, ''));
            }
            out.push('<blockquote><p>' + inline(bq.join(' ')) + '</p></blockquote>');
            continue;
        }
        flushAll();
        out.push('<p>' + inline(trimmed) + '</p>');
    }
    flushAll();
    return out.join('\n');
}

/* ── make h2 sections into collapsible accordion ── */
function enableAccordion(container) {
    const headings = container.querySelectorAll('h2');
    headings.forEach((h2) => {
        const answer = document.createElement('div');
        answer.className = 'faq-answer';
        let sib = h2.nextSibling;
        while (sib && !(sib.nodeType === 1 && /^(H[12]|HR)$/.test(sib.tagName))) {
            const next = sib.nextSibling;
            answer.appendChild(sib);
            sib = next;
        }
        h2.after(answer);
        h2.addEventListener('click', () => {
            const open = h2.classList.toggle('faq-open');
            answer.classList.toggle('faq-show', open);
        });
    });
    if (headings.length > 0) {
        headings[0].classList.add('faq-open');
        headings[0].nextElementSibling.classList.add('faq-show');
    }
}

/* ── build sidebar DOM ── */
function buildSidebar(lang) {
    const sidebar = document.createElement('div');
    sidebar.className = 'faq-sidebar';
    sidebar.id = 'faqSidebar';

    sidebar.innerHTML = `
        <div class="faq-collapse-strip" id="faqCollapseBtn" title="Toggle FAQ">
            <div class="faq-collapse-pill">
                <svg class="faq-collapse-chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="9 18 15 12 9 6"/>
                </svg>
                <span class="faq-collapse-label">FAQ</span>
            </div>
        </div>
        <div class="faq-sidebar-inner">
            <div class="faq-header">
                <span class="faq-header-title">FAQ</span>
                <div class="faq-lang-toggle" id="faqLangToggle">
                    <button class="faq-lang-btn ${lang === 'zh' ? 'active' : ''}" data-lang="zh">中文</button>
                    <button class="faq-lang-btn ${lang === 'en' ? 'active' : ''}" data-lang="en">EN</button>
                </div>
            </div>
            <div class="faq-body">
                <div class="faq-content" id="faqContent">
                    <p style="color:#999;">Loading...</p>
                </div>
            </div>
        </div>`;
    return sidebar;
}

/* ── wrap existing page content ── */
function wrapPageContent(sidebar) {
    const wrapper = document.createElement('div');
    wrapper.className = 'faq-layout-wrapper';

    const content = document.createElement('div');
    content.className = 'faq-main-content';

    const children = Array.from(document.body.childNodes);
    children.forEach(ch => content.appendChild(ch));

    wrapper.appendChild(content);
    wrapper.appendChild(sidebar);
    document.body.appendChild(wrapper);

    document.body.style.overflow = 'hidden';
}

/* ── load and render FAQ content ── */
async function loadFaq(mode, lang, container) {
    const mdPath = `/static/faq/${lang}/${mode}.md`;
    try {
        const resp = await fetch(mdPath);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const md = await resp.text();
        container.innerHTML = md2html(md);
        enableAccordion(container);
    } catch (e) {
        container.innerHTML = `<p style="color:#cf222e;">FAQ load failed: ${e.message}</p>`;
    }
}

/* ── public API ── */
export async function initFaqSidebar(mode) {
    const cssHref = '/static/shared/faq-sidebar.css';
    if (!document.querySelector(`link[href="${cssHref}"]`)) {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = cssHref;
        document.head.appendChild(link);
    }

    let lang = window.I18n?.getLang?.() || localStorage.getItem(STORAGE_KEY_LANG) || 'zh';

    const sidebar = buildSidebar(lang);
    wrapPageContent(sidebar);

    const collapseBtn = document.getElementById('faqCollapseBtn');
    const langToggle  = document.getElementById('faqLangToggle');
    const content     = document.getElementById('faqContent');

    collapseBtn.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
    });

    langToggle.addEventListener('click', async (e) => {
        const btn = e.target.closest('.faq-lang-btn');
        if (!btn || btn.classList.contains('active')) return;
        langToggle.querySelectorAll('.faq-lang-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        lang = btn.dataset.lang;
        localStorage.setItem(STORAGE_KEY_LANG, lang);
        await loadFaq(mode, lang, content);
        if (window.I18n?.setLang) window.I18n.setLang(lang, true);
    });

    await loadFaq(mode, lang, content);
}
