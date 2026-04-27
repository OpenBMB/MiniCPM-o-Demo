/**
 * ES module wrapper for i18n.js.
 * Ensures window.I18n is populated, then re-exports its members.
 *
 * Usage:
 *   import { t, setLang, createLangToggle } from '/static/shared/i18n-module.js';
 */

// i18n.js is expected to be loaded as a classic <script src> before this
// module executes, setting window.I18n synchronously.  If somehow it
// hasn't loaded yet (e.g. only this module is imported), we can't
// dynamically await a classic script, so we just read what's there.

const I = window.I18n || {};

export const getLang = I.getLang || (() => 'zh');
export const getT = I.getT || (() => I.t || {});
export const setLang = I.setLang || (() => {});
export const createLangToggle = I.createLangToggle || (() => document.createElement('div'));
export const t = I.t || {};
