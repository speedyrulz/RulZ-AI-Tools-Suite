// Opens a page of the owner's app in a hidden, sandboxed window and collects what check_page reports.
import { BrowserWindow, session as electronSession } from 'electron'
import { PAGE_SNAPSHOT_SCRIPT, type PageMessage, type PageReport, type PageTarget } from '../agent/pageCheck'

const PARTITION = 'check-page'
const LOAD_TIMEOUT_MS = 20_000
const SNAPSHOT_TIMEOUT_MS = 5_000

const timeout = <T>(p: Promise<T>, ms: number, what: string): Promise<T> =>
  Promise.race([p, new Promise<T>((_, reject) => setTimeout(() => reject(new Error(what)), ms))])

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const t = setTimeout(resolve, ms)
    signal?.addEventListener('abort', () => (clearTimeout(t), resolve()), { once: true })
  })
}

export async function checkPageInWindow(target: PageTarget, waitMs: number, signal?: AbortSignal): Promise<PageReport> {
  const ses = electronSession.fromPartition(PARTITION)
  await ses.clearStorageData().catch(() => undefined)
  await ses.clearCache().catch(() => undefined)
  ses.setPermissionRequestHandler((_wc, _perm, callback) => callback(false))
  ses.on('will-download', (e) => e.preventDefault())

  const report: PageReport = { url: target.url, title: '', messages: [], failedRequests: [], text: '', controls: [], visuals: 0 }
  const seen = new Set<string>()
  const addFailure = (url: string, problem: string): void => {
    if (url.startsWith('devtools:') || url.startsWith('chrome-extension:') || seen.has(url)) return
    seen.add(url)
    report.failedRequests.push({ url, problem })
  }
  ses.webRequest.onCompleted((d) => {
    if (d.statusCode >= 400 && d.resourceType !== 'mainFrame') addFailure(d.url, `status ${d.statusCode}`)
  })
  ses.webRequest.onErrorOccurred((d) => {
    if (d.error !== 'net::ERR_ABORTED' && d.resourceType !== 'mainFrame') addFailure(d.url, d.error)
  })

  const win = new BrowserWindow({
    show: false,
    width: 1280,
    height: 800,
    webPreferences: { partition: PARTITION, sandbox: true, contextIsolation: true, nodeIntegration: false, backgroundThrottling: false }
  })
  const wc = win.webContents
  wc.setAudioMuted(true)
  wc.setWindowOpenHandler(() => ({ action: 'deny' }))
  wc.on('will-navigate', (e, url) => {
    if (!url.startsWith(target.allowedPrefix)) e.preventDefault()
  })
  wc.on('console-message', (e) => {
    const level = e.level === 'error' ? 'error' : e.level === 'warning' ? 'warning' : null
    if (!level || report.messages.length >= 200) return
    const m: PageMessage = { level, message: e.message, source: e.sourceId || undefined, line: e.lineNumber || undefined }
    report.messages.push(m)
  })
  wc.on('did-navigate', (_e, url, code) => {
    report.url = url
    if (code > 0) report.status = code
  })
  wc.on('render-process-gone', (_e, d) => {
    report.crashed = `the page crashed (${d.reason}).`
  })
  wc.on('unresponsive', () => {
    report.crashed = 'the page stopped responding (a script may be stuck in an endless loop).'
  })

  wc.on('did-fail-load', (_e, code, description, url, isMainFrame) => {
    // -3 (ABORTED) is the page moving on to another address by itself.
    if (isMainFrame && code !== -3) report.loadError = `${description || `error ${code}`} (${url})`
  })

  try {
    try {
      await timeout(win.loadURL(target.url), LOAD_TIMEOUT_MS, 'slow')
    } catch (e) {
      // did-fail-load has the details of a failed page; a timeout is reported here.
      if ((e as Error).message === 'slow' && !report.loadError) report.crashed = `the page took more than ${LOAD_TIMEOUT_MS / 1000} seconds to load.`
    }
    if (!report.loadError) {
      await sleep(waitMs, signal)
      try {
        const snap = (await timeout(wc.executeJavaScript(PAGE_SNAPSHOT_SCRIPT, true), SNAPSHOT_TIMEOUT_MS, 'no answer')) as Pick<
          PageReport,
          'title' | 'text' | 'controls' | 'visuals'
        >
        Object.assign(report, snap)
      } catch {
        report.crashed ??= 'the page stopped responding (a script may be stuck in an endless loop).'
      }
    }
  } finally {
    ses.webRequest.onCompleted(null)
    ses.webRequest.onErrorOccurred(null)
    ses.removeAllListeners('will-download')
    if (!win.isDestroyed()) win.destroy()
  }
  return report
}
