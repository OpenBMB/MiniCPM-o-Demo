/**
 * lib/duplex-utils.js — Pure utility functions (zero DOM dependency)
 */

/**
 * Linear-interpolation resample audio samples between sample rates.
 * @param {Float32Array} samples - Input audio samples
 * @param {number} fromRate - Source sample rate
 * @param {number} toRate - Target sample rate
 * @returns {Float32Array} Resampled audio
 */
export function resampleAudio(samples, fromRate, toRate) {
    if (fromRate === toRate) return samples;
    const ratio = fromRate / toRate;
    const newLen = Math.round(samples.length / ratio);
    const result = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
        const srcIdx = i * ratio;
        const floor = Math.floor(srcIdx);
        const frac = srcIdx - floor;
        const next = Math.min(floor + 1, samples.length - 1);
        result[i] = samples[floor] * (1 - frac) + samples[next] * frac;
    }
    return result;
}

/** Alias for backward compat */
export const downsample = resampleAudio;

/**
 * Convert an ArrayBuffer to a base64 string.
 * @param {ArrayBuffer} buffer
 * @returns {string}
 */
export function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    const chunks = [];
    for (let i = 0; i < bytes.length; i += 8192) {
        chunks.push(String.fromCharCode.apply(null, bytes.subarray(i, i + 8192)));
    }
    return btoa(chunks.join(''));
}

/**
 * Escape HTML special characters.
 * @param {string} s
 * @returns {string}
 */
export function escapeHtml(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
