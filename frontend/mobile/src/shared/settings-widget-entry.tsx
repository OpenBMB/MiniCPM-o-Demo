import { createRoot, type Root } from 'react-dom/client'
import { useState, useCallback, useRef, useEffect } from 'react'
import { OmniSettingsWidget } from './OmniSettingsWidget'
import type { OmniBridge } from './settings-types'
import { I18nContext, detectLang, persistLang, t as getT } from '../i18n'
import type { Lang } from '../i18n'

function WidgetHost({ bridge, openSignal }: { bridge: OmniBridge; openSignal: { current: number } }) {
  const [open, setOpen] = useState(false)
  const [lang, setLangState] = useState<Lang>(detectLang)
  const i18n = getT(lang)
  const setLang = (l: Lang) => { setLangState(l); persistLang(l); window.location.reload() }
  const lastSignal = useRef(0)

  useEffect(() => {
    const id = setInterval(() => {
      if (openSignal.current !== lastSignal.current) {
        lastSignal.current = openSignal.current
        setOpen(true)
      }
    }, 50)
    return () => clearInterval(id)
  }, [openSignal])

  const handleClose = useCallback(() => setOpen(false), [])

  return (
    <I18nContext.Provider value={{ lang, setLang, t: i18n }}>
      <OmniSettingsWidget open={open} bridge={bridge} onClose={handleClose} />
    </I18nContext.Provider>
  )
}

export function mountOmniSettings(
  container: HTMLElement,
  bridge: OmniBridge,
): { open: () => void; destroy: () => void } {
  const openSignal = { current: 0 }
  const root: Root = createRoot(container)
  root.render(<WidgetHost bridge={bridge} openSignal={openSignal} />)

  return {
    open() {
      openSignal.current = Date.now()
    },
    destroy() {
      root.unmount()
    },
  }
}

(window as unknown as Record<string, unknown>).mountOmniSettings = mountOmniSettings
