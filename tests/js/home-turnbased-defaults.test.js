import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const homeHtml = readFileSync('static/index.html', 'utf8');
const i18n = readFileSync('static/shared/i18n.js', 'utf8');
const turnbasedHtml = readFileSync('static/turnbased.html', 'utf8');

describe('home resources and desktop turn-based defaults', () => {
    it('links to the hosted example from the home page', () => {
        expect(homeHtml).toContain('href="https://openbmb.github.io/minicpm-o-4_5/"');
        expect(homeHtml).toContain('id="resource-example"');
        expect(homeHtml).toContain('t.exampleCase');
        expect(i18n).toContain("exampleCase: '范例case'");
        expect(i18n).toContain("exampleCase: 'Example Case'");
    });

    it('enables voice responses by default in both desktop turn-based views', () => {
        expect(turnbasedHtml).toMatch(/id="enableTtsInit" checked/);
        expect(turnbasedHtml).toMatch(/id="enableTts" checked/);
    });
});
