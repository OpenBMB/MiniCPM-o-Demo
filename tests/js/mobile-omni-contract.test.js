import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const mobileHtml = readFileSync('static/mobile-omni/index.html', 'utf8');
const omniApp = readFileSync('static/omni/omni-app.js', 'utf8');

describe('mobile omni DOM contract', () => {
    it('provides the TTS control required by the shared omni app', () => {
        expect(mobileHtml).toMatch(/<input\b[^>]*\bid="ttsEnabled"[^>]*\bchecked\b[^>]*>/);
    });

    it('defaults TTS on when a host page omits the control', () => {
        expect(omniApp).toContain("document.getElementById('ttsEnabled')?.checked ?? true");
    });
});
