import { resolve } from 'node:path'
import { defineConfig } from 'electron-vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import type { Plugin } from 'vite'

// The dev server needs inline scripts for React Fast Refresh, so the strict CSP is only
// injected into production builds.
const PROD_CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  "connect-src 'self'"
].join('; ')

function productionCsp(): Plugin {
  return {
    name: 'rulz-production-csp',
    apply: 'build',
    transformIndexHtml(html) {
      return html.replace(
        '<head>',
        `<head>\n    <meta http-equiv="Content-Security-Policy" content="${PROD_CSP}" />`
      )
    }
  }
}

const shared = { '@shared': resolve(__dirname, 'src/shared') }

export default defineConfig({
  main: {
    resolve: { alias: shared },
    build: {
      rollupOptions: {
        // The PDF worker thread is a main-process file of its own (out/main/pdfWorker.js), so it also works from inside app.asar.
        input: { index: resolve(__dirname, 'src/main/index.ts'), pdfWorker: resolve(__dirname, 'src/main/pdf/worker.ts') }
      }
    }
  },
  preload: {
    resolve: { alias: shared }
  },
  renderer: {
    resolve: {
      alias: {
        ...shared,
        '@renderer': resolve(__dirname, 'src/renderer/src')
      }
    },
    plugins: [react(), tailwindcss(), productionCsp()]
  }
})
