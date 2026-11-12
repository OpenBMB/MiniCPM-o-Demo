import { RealtimeSession } from '../lib/realtime-session.js';
import { initHealthCheck as providerHealthCheck } from './duplex-ui.js';

function initOmniHealthCheck(badgeId) {
    const badge = document.getElementById(badgeId);
    if (!badge) return () => {};
    async function check() {
        try {
            const response = await fetch('/api/vllm_omni/health');
            if (!response.ok) throw new Error('Service unavailable');
            badge.textContent = 'vLLM-Omni';
            badge.className = 'status-badge online';
        } catch {
            badge.textContent = 'Offline';
            badge.className = 'status-badge';
        }
    }
    check();
    const interval = setInterval(check, 10000);
    return () => clearInterval(interval);
}

function configureControls() {
    for (const id of ['maxKvTokens', 'duplexLengthPenalty', 'omniLengthPenalty', 'ttsEnabled', 'visionHD']) {
        const input = document.getElementById(id);
        if (input) {
            input.disabled = true;
            input.title = 'Controlled by the vLLM-Omni deployment';
        }
    }
    document.getElementById('ttsEnabled').checked = true;
    const style = document.createElement('style');
    style.textContent = '#save-share-container, #btnHD, #fsBtnHD,'
        + ' .cg-row:has(input[name$="TtsRefMode"]), a[href="/mobile"] { display: none !important; }';
    document.head.appendChild(style);
}

export async function loadRealtimeBackend() {
    const response = await fetch('/api/frontend_defaults');
    if (!response.ok) throw new Error('Cannot load session configuration');
    const { vllm_omni: backend } = await response.json();
    if (!backend) {
        return { Session: RealtimeSession, initHealthCheck: providerHealthCheck, configureControls() {} };
    }
    const { VllmOmniSession } = await import('../lib/vllm-omni-session.js');
    return {
        Session: class extends VllmOmniSession {
            constructor(prefix, config) { super(prefix, config, backend); }
        },
        initHealthCheck: initOmniHealthCheck,
        configureControls,
    };
}
