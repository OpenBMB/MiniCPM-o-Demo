import { createRoot, type Root } from 'react-dom/client'
import { useState, useCallback, useRef, useEffect } from 'react'
import { OmniSettingsWidget } from './OmniSettingsWidget'
import type { OmniBridge } from './settings-types'

function WidgetHost({ bridge, openSignal }: { bridge: OmniBridge; openSignal: { current: number } }) {
  const [open, setOpen] = useState(false)
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

  return <OmniSettingsWidget open={open} bridge={bridge} onClose={handleClose} />
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
