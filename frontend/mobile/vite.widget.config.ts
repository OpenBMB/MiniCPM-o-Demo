import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'

export default defineConfig({
  plugins: [react()],
  define: { 'process.env.NODE_ENV': '"production"' },
  build: {
    outDir: resolve(__dirname, '../../static/mobile-omni'),
    emptyDir: false,
    lib: {
      entry: resolve(__dirname, 'src/shared/settings-widget-entry.tsx'),
      formats: ['iife'],
      name: 'OmniSettingsWidget',
      fileName: () => 'settings-widget.js',
    },
    cssFileName: 'settings-widget',
    rollupOptions: {
      output: { inlineDynamicImports: true },
    },
  },
})
