import { createContext, useContext } from 'react'
import type { Lang, Translations } from './types'
import { zh } from './zh'
import { en } from './en'

export type { Lang, Translations }
export { zh, en }

const bundles: Record<Lang, Translations> = { zh, en }

const STORAGE_KEY = 'minicpmo_lang'

/** Detect language from (in priority order):
 *  1. URL search param `?lang=zh` / `?lang=en`
 *  2. localStorage
 *  3. navigator.language
 *  4. fallback 'zh'
 */
export function detectLang(): Lang {
  try {
    const params = new URLSearchParams(window.location.search)
    const fromUrl = params.get('lang')
    if (fromUrl === 'en' || fromUrl === 'zh') return fromUrl
  } catch { /* SSR or no URL */ }

  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === 'en' || stored === 'zh') return stored
  } catch { /* private browsing */ }

  try {
    const nav = navigator.language.toLowerCase()
    if (nav.startsWith('en')) return 'en'
  } catch { /* no navigator */ }

  return 'zh'
}

export function persistLang(lang: Lang): void {
  try {
    localStorage.setItem(STORAGE_KEY, lang)
  } catch { /* ignore */ }
}

export function t(lang: Lang): Translations {
  return bundles[lang] ?? bundles.zh
}

export const I18nContext = createContext<{
  lang: Lang
  setLang: (lang: Lang) => void
  t: Translations
}>({
  lang: 'zh',
  setLang: () => {},
  t: zh,
})

export function useI18n() {
  return useContext(I18nContext)
}
