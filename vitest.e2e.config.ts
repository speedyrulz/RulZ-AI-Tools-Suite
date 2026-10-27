import { resolve } from 'node:path'
import { defineConfig } from 'vitest/config'

// End-to-end tests drive the built app (run `npm run build` first; `npm run test:e2e` does both).
export default defineConfig({
  resolve: { alias: { '@shared': resolve(__dirname, 'src/shared') } },
  test: { include: ['tests/e2e/**/*.e2e.ts'], environment: 'node', testTimeout: 120_000, hookTimeout: 60_000, fileParallelism: false }
})
