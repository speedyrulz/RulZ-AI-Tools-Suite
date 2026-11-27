import { join } from 'node:path'
import { app, BrowserWindow, nativeTheme, shell } from 'electron'
import type { LoraMergeJob } from '@shared/loraMerge'
import type { MergeJob } from '@shared/merge'
import type { ConnectionStatus, PreflightReport, ServerState, Theme } from '@shared/types'
import { ComfyClient, probeServer } from './comfy/client'
import { normalizeServerUrl } from './comfy/parse'
import { broadcast, handle } from './ipc'
import { LoraMergeService } from './loras/service'
import loraWorkerPath from './loras/worker?modulePath'
import { listGpus } from './server/gpus'
import { MergeRunner } from './merge/runner'
import { OwnServer } from './server/ownServer'
import { runPreflight } from './server/preflight'
import { cleanupOldTempScripts, createWslRunner } from './server/wsl'
import { SettingsStore } from './settings'

const isDev = !app.isPackaged && !!process.env.ELECTRON_RENDERER_URL

const isPrimaryInstance = app.requestSingleInstanceLock()
if (!isPrimaryInstance) app.quit()

let mainWindow: BrowserWindow | null = null
let quitting = false

const wsl = createWslRunner()
let settings!: SettingsStore
let ownServer!: OwnServer
let mergeRunner!: MergeRunner
let loraMerge!: LoraMergeService
/** Servers the app did not start whose queue the user agreed to share, this session only. */
const acceptedSharedQueues = new Set<string>()

function windowBackground(theme: Theme): string {
  return theme === 'dark' ? '#050505' : '#ffffff'
}

function applyTheme(theme: Theme): void {
  // The native title bar follows the app's theme, not the system's.
  nativeTheme.themeSource = theme
  mainWindow?.setBackgroundColor(windowBackground(theme))
}

/** The URL jobs go to right now, or null when the own server is chosen but not running. */
function currentUrl(): { url: string | null; own: boolean } {
  const c = settings.get().connection
  if (c.mode === 'own') return { url: ownServer.state().status === 'running' ? ownServer.state().url : null, own: true }
  return { url: normalizeServerUrl(c.url), own: false }
}

async function connectionStatus(): Promise<ConnectionStatus> {
  let target: { url: string | null; own: boolean }
  try {
    target = currentUrl()
  } catch (e) {
    return emptyStatus(null, false, (e as Error).message)
  }
  if (!target.url) {
    return emptyStatus(null, true, 'The own server is not running. Start it on the Server page.')
  }
  try {
    const stats = await new ComfyClient(target.url, { timeoutMs: 5000 }).systemStats()
    return {
      url: target.url,
      reachable: true,
      own: target.own,
      version: stats.version,
      devices: stats.devices,
      sharedQueueAccepted: target.own || acceptedSharedQueues.has(target.url),
      error: null
    }
  } catch (e) {
    return emptyStatus(target.url, target.own, (e as Error).message)
  }
}

function emptyStatus(url: string | null, own: boolean, error: string): ConnectionStatus {
  return { url, reachable: false, own, version: null, devices: [], sharedQueueAccepted: false, error }
}

let statusGeneration = 0

/** Each refresh gets a number; only the newest finished refresh is stored and broadcast. */
async function refreshStatus(): Promise<ConnectionStatus> {
  const generation = ++statusGeneration
  const status = await connectionStatus()
  if (generation !== statusGeneration) return status
  broadcast('comfy:status', status)
  return status
}

async function preflight(): Promise<PreflightReport> {
  const gpus = await listGpus()
  const st = ownServer.state()
  return runPreflight({
    wsl,
    settings: settings.get().ownServer,
    listGpus: async () => gpus,
    probe: (url) => probeServer(url, gpus, { timeoutMs: 1000 }),
    ownRunning: st.status !== 'stopped' && st.status !== 'error',
    ownPort: st.port
  })
}

async function startOwnServer(confirmed: string[]): Promise<ServerState> {
  const report = await preflight()
  const blocks = report.issues.filter((i) => i.level === 'block')
  if (blocks.length) throw new Error(blocks.map((i) => i.message).join(' '))
  const unconfirmed = report.issues.filter((i) => i.level === 'confirm' && !confirmed.includes(i.code))
  if (unconfirmed.length) throw new Error(`Confirm first: ${unconfirmed.map((i) => i.message).join(' ')}`)
  if (report.port === null) throw new Error('No free port was found for the server.')
  return ownServer.start(report.port)
}

function registerIpc(): void {
  handle('app:version', () => app.getVersion())
  handle('settings:get', () => settings.get())
  handle('settings:set', async (patch) => {
    const before = settings.get()
    const next = await settings.set(patch)
    if (next.theme !== before.theme) applyTheme(next.theme)
    if (JSON.stringify(next.connection) !== JSON.stringify(before.connection)) void refreshStatus()
    return next
  })
  handle('gpu:list', () => listGpus())
  handle('server:preflight', () => preflight())
  handle('server:start', (confirmed) => startOwnServer(confirmed))
  handle('server:stop', () => ownServer.stop())
  handle('server:state', () => ownServer.state())
  handle('server:log', () => ownServer.logTail())
  handle('comfy:status', () => refreshStatus())
  handle('comfy:acceptSharedQueue', async (url) => {
    acceptedSharedQueues.add(normalizeServerUrl(url))
    return refreshStatus()
  })
  handle('comfy:capabilities', async () => {
    const { url } = currentUrl()
    if (!url) throw new Error('No ComfyUI server is connected.')
    return new ComfyClient(url).capabilities()
  })
  handle('merge:inspect', (req) => mergeRunner.inspect(req))
  handle('merge:start', (recipe) => mergeRunner.start(recipe))
  handle('merge:cancel', (id) => mergeRunner.cancel(id))
  handle('merge:jobs', () => mergeRunner.jobs())
  handle('merge:moveToModels', (id) => mergeRunner.moveToModels(id))
  handle('loraMerge:inspect', (req) => loraMerge.inspect(req))
  handle('loraMerge:start', (recipe) => loraMerge.start(recipe))
  handle('loraMerge:cancel', (id) => loraMerge.cancel(id))
  handle('loraMerge:jobs', () => loraMerge.jobs())
  handle('loraMerge:list', () => loraMerge.list())
  handle('shell:openExternal', async (url) => {
    if (/^https?:\/\//i.test(url)) await shell.openExternal(url)
  })
}

function createWindow(): void {
  const theme = settings.get().theme
  const win = new BrowserWindow({
    width: 1360,
    height: 880,
    minWidth: 960,
    minHeight: 620,
    show: false,
    title: 'RulZ Model Merger',
    backgroundColor: windowBackground(theme),
    autoHideMenuBar: true,
    // Packaged builds take the icon from the .exe; dev runs need it explicitly.
    ...(app.isPackaged ? {} : { icon: join(__dirname, '../../build/icon.png') }),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  })
  mainWindow = win
  win.setMenuBarVisibility(false)
  win.once('ready-to-show', () => win.show())
  win.on('closed', () => {
    if (mainWindow === win) mainWindow = null
  })
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//i.test(url)) void shell.openExternal(url)
    return { action: 'deny' }
  })
  win.webContents.on('will-navigate', (e, url) => {
    if (!isDev || !url.startsWith(process.env.ELECTRON_RENDERER_URL!)) e.preventDefault()
  })
  win.webContents.on('before-input-event', (e, input) => {
    if (input.type !== 'keyDown') return
    if (input.key === 'F12' || (input.control && input.shift && input.key.toLowerCase() === 'i')) {
      win.webContents.toggleDevTools()
      e.preventDefault()
    } else if (isDev && input.control && input.key.toLowerCase() === 'r') {
      win.webContents.reload()
      e.preventDefault()
    }
  })

  // The page applies this before its first paint; the saved settings take over once they load.
  const query = { theme }
  if (isDev) void win.loadURL(`${process.env.ELECTRON_RENDERER_URL}?theme=${theme}`)
  else void win.loadFile(join(__dirname, '../renderer/index.html'), { query })
}

async function main(): Promise<void> {
  await app.whenReady()
  cleanupOldTempScripts()
  settings = new SettingsStore(join(app.getPath('userData'), 'settings.json'))
  await settings.load()
  applyTheme(settings.get().theme)

  ownServer = new OwnServer({
    wsl,
    settings: () => settings.get().ownServer,
    probe: async (url) => {
      try {
        const stats = await new ComfyClient(url, { timeoutMs: 1500 }).systemStats()
        return { argv: stats.argv }
      } catch {
        return null
      }
    }
  })
  ownServer.on('state', (st: ServerState) => {
    broadcast('server:state', st)
    if (st.status === 'running' || st.status === 'stopped' || st.status === 'error') void refreshStatus()
  })
  ownServer.on('log', (text: string) => broadcast('server:log', { text }))

  mergeRunner = new MergeRunner({
    client: (url) => new ComfyClient(url),
    wsl,
    getConnection: () => connectionStatus(),
    getOwnRecord: () => ownServer.record(),
    ownSettings: () => settings.get().ownServer,
    onServerLog: (listener) => {
      ownServer.on('log', listener)
      return () => {
        ownServer.off('log', listener)
      }
    },
    appVersion: app.getVersion()
  })
  mergeRunner.on('job', (job: MergeJob) => broadcast('merge:job', job))

  loraMerge = new LoraMergeService({
    wsl,
    settings: () => settings.get().ownServer,
    appVersion: app.getVersion(),
    workerPath: loraWorkerPath
  })
  loraMerge.on('job', (job: LoraMergeJob) => broadcast('loraMerge:job', job))

  registerIpc()
  createWindow()

  app.on('second-instance', () => {
    if (!mainWindow) return
    if (mainWindow.isMinimized()) mainWindow.restore()
    mainWindow.show()
    mainWindow.focus()
  })
  app.on('window-all-closed', () => app.quit())
  // Stop the own ComfyUI before exiting, so no server is left running in WSL.
  app.on('before-quit', (e) => {
    if (quitting) return
    e.preventDefault()
    quitting = true
    void ownServer
      .dispose()
      .catch(() => undefined)
      .finally(() => app.quit())
  })
}

if (isPrimaryInstance) void main()
