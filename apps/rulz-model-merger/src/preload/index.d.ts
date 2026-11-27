import type { RulzApi } from '../shared/ipc'

declare global {
  interface Window {
    rulz: RulzApi
  }
}

export {}
