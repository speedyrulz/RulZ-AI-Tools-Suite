// check_page's hidden window with a stand-in for Electron: Stop must end the check at once.
import { beforeEach, describe, expect, it, vi } from 'vitest'
// vi.mock below is hoisted above this import, so the module gets the stand-in.
import { checkPageInWindow } from '../src/main/project/pageWindow'

const fake = vi.hoisted(() => ({
  /** What loadURL and executeJavaScript return for the next window. */
  load: (): Promise<void> => Promise.resolve(),
  script: (): Promise<unknown> => Promise.resolve({ title: 'My app', text: 'Hello', controls: [], visuals: 0 }),
  windows: [] as { destroyed: boolean }[]
}))

vi.mock('electron', async () => {
  const { EventEmitter } = await import('node:events')
  class WebContents extends EventEmitter {
    setAudioMuted(): void {}
    setWindowOpenHandler(): void {}
    executeJavaScript(): Promise<unknown> {
      return fake.script()
    }
  }
  class BrowserWindow {
    webContents = new WebContents()
    destroyed = false
    constructor() {
      fake.windows.push(this)
    }
    loadURL(): Promise<void> {
      return fake.load()
    }
    isDestroyed(): boolean {
      return this.destroyed
    }
    destroy(): void {
      this.destroyed = true
    }
  }
  const ses = Object.assign(new EventEmitter(), {
    clearStorageData: async () => undefined,
    clearCache: async () => undefined,
    setPermissionRequestHandler: () => undefined,
    webRequest: { onCompleted: () => undefined, onErrorOccurred: () => undefined }
  })
  return { BrowserWindow, session: { fromPartition: () => ses } }
})

const target = { url: 'http://localhost:5173/', allowedPrefix: 'http://localhost:5173' }
const never = <T>(): Promise<T> => new Promise<T>(() => undefined)

async function timed<T>(p: Promise<T>): Promise<{ value: T; ms: number }> {
  const t = performance.now()
  const value = await p
  return { value, ms: performance.now() - t }
}

describe('check_page window', () => {
  beforeEach(() => {
    fake.windows.length = 0
    fake.load = () => Promise.resolve()
    fake.script = () => Promise.resolve({ title: 'My app', text: 'Hello', controls: [], visuals: 0 })
  })

  it('reads the page when nothing stops it', async () => {
    const report = await checkPageInWindow(target, 0, new AbortController().signal)
    expect(report).toMatchObject({ title: 'My app', text: 'Hello' })
    expect(fake.windows[0].destroyed).toBe(true)
  })

  it('stops at once while the page is still loading', async () => {
    fake.load = never
    const stop = new AbortController()
    setTimeout(() => stop.abort(), 50)
    const { value, ms } = await timed(checkPageInWindow(target, 2000, stop.signal))
    expect(ms).toBeLessThan(1000)
    expect(value.crashed).toBeUndefined()
    expect(fake.windows[0].destroyed).toBe(true)
  })

  it('stops at once while waiting for the page to answer', async () => {
    fake.script = never
    const stop = new AbortController()
    setTimeout(() => stop.abort(), 50)
    const { value, ms } = await timed(checkPageInWindow(target, 0, stop.signal))
    expect(ms).toBeLessThan(1000)
    expect(value.crashed).toBeUndefined()
    expect(fake.windows[0].destroyed).toBe(true)
  })

  it('stops during the wait before reading the page, without reading it', async () => {
    let asked = false
    fake.script = () => ((asked = true), Promise.resolve({ title: 'x', text: '', controls: [], visuals: 0 }))
    const stop = new AbortController()
    setTimeout(() => stop.abort(), 50)
    const { ms } = await timed(checkPageInWindow(target, 5000, stop.signal))
    expect(ms).toBeLessThan(1000)
    expect(asked).toBe(false)
    expect(fake.windows[0].destroyed).toBe(true)
  })

  it('returns at once when already stopped', async () => {
    fake.load = never
    const stop = new AbortController()
    stop.abort()
    const { ms } = await timed(checkPageInWindow(target, 2000, stop.signal))
    expect(ms).toBeLessThan(500)
    expect(fake.windows[0].destroyed).toBe(true)
  })
})
