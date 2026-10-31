import { join, resolve, sep } from 'node:path'
import { pathToFileURL } from 'node:url'
import { app, BrowserWindow, dialog, nativeTheme, net, protocol, shell } from 'electron'
import { findApp } from '@shared/catalog'
import type { ThemeName } from '@shared/types'
import { AppManager } from './apps'
import { setFetch } from './github'
import { broadcast, handle } from './ipc'
import { checkRequirements, installRequirement } from './requirements'
import { checkSelfUpdate, installSelfUpdate } from './selfUpdate'
import { Store } from './store'

const isDev = !app.isPackaged && !!process.env.ELECTRON_RENDERER_URL

// A second copy hands over to the first (see 'second-instance') and does nothing else.
const primary = app.requestSingleInstanceLock()
if (!primary) app.quit()

// rulz://catalog/<app>/... serves the screenshots, icons and README copies shipped with the launcher.
protocol.registerSchemesAsPrivileged([
  { scheme: 'rulz', privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true } }
])

let mainWindow: BrowserWindow | null = null

const WINDOW_COLORS = {
  dark: { background: '#050505', symbols: '#e5e5e5' },
  light: { background: '#ffffff', symbols: '#111114' }
}

/** Title bar buttons, window background and the READMEs' prefers-color-scheme follow the theme. */
function applyTheme(win: BrowserWindow | null, theme: ThemeName): void {
  nativeTheme.themeSource = theme
  if (!win || win.isDestroyed()) return
  const c = WINDOW_COLORS[theme]
  win.setBackgroundColor(c.background)
  win.setTitleBarOverlay({ color: c.background, symbolColor: c.symbols, height: 48 })
}

function createWindow(theme: ThemeName): BrowserWindow {
  const colors = WINDOW_COLORS[theme]
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 640,
    show: false,
    title: 'RulZ AI Tools Suite',
    backgroundColor: colors.background,
    titleBarStyle: 'hidden',
    titleBarOverlay: { color: colors.background, symbolColor: colors.symbols, height: 48 },
    // Packaged builds take the icon from the .exe; dev runs need it explicitly.
    ...(app.isPackaged ? {} : { icon: join(__dirname, '../../build/icon.png') }),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  })
  win.once('ready-to-show', () => win.show())

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

  // The theme goes in the URL so the first paint already has the right colors.
  if (isDev) void win.loadURL(`${process.env.ELECTRON_RENDERER_URL!}?theme=${theme}`)
  else void win.loadFile(join(__dirname, '../renderer/index.html'), { query: { theme } })
  return win
}

async function main(): Promise<void> {
  await app.whenReady()
  app.setAppUserModelId('com.rulz.aitoolssuite')
  setFetch((url, init) => net.fetch(url, init))

  const userData = app.getPath('userData')
  const catalogDir = app.isPackaged ? join(process.resourcesPath, 'catalog') : join(app.getAppPath(), 'catalog')

  protocol.handle('rulz', (req) => {
    const url = new URL(req.url)
    const file = resolve(catalogDir, decodeURIComponent(url.pathname).replace(/^\/+/, ''))
    if (url.host !== 'catalog' || !file.startsWith(catalogDir + sep)) return new Response('Not found', { status: 404 })
    return net.fetch(pathToFileURL(file).toString())
  })

  const store = new Store(userData)
  await store.init()
  const apps = new AppManager({
    store,
    catalogDir,
    downloadsDir: join(app.getPath('temp'), 'RulZ AI Tools Suite'),
    iconsDir: join(userData, 'icons'),
    emitApps: (list) => broadcast('apps:changed', list),
    emitJob: (job) => broadcast('job:update', job),
    emitLog: (lines) => broadcast('job:log', lines)
  })
  // Quick (cached answers and the bundled files); the registry search for outside installs runs once the window is up.
  await apps.init()

  // ---- IPC ----
  handle('app:info', () => ({ version: app.getVersion(), userData, platform: process.platform }))
  handle('settings:get', () => store.getSettings())
  handle('settings:update', async (patch) => {
    const next = await store.updateSettings(patch)
    if (patch.theme) applyTheme(mainWindow, patch.theme)
    return next
  })
  handle('dialog:pickFolder', async ({ title, defaultPath }) => {
    const r = await dialog.showOpenDialog(mainWindow!, { title, defaultPath, properties: ['openDirectory', 'createDirectory'] })
    return r.canceled ? null : (r.filePaths[0] ?? null)
  })
  handle('shell:openExternal', async (url) => {
    if (!/^https?:\/\//i.test(url)) throw new Error('Only web links can be opened.')
    await shell.openExternal(url)
  })

  handle('apps:list', () => apps.views())
  handle('apps:refresh', async () => {
    await apps.refreshLatest(true)
    return apps.views()
  })
  handle('apps:readme', (id) => apps.readme(id, store.getSettings().liveReadme))
  handle('apps:defaultDir', (id) => apps.defaultDir(id))
  handle('apps:install', (id, opts) => apps.install(id, opts))
  handle('apps:update', (id) => apps.update(id))
  handle('apps:repair', (id) => apps.repair(id))
  handle('apps:cancel', (id) => apps.cancel(id))
  handle('apps:launch', async (id) => {
    await apps.launch(id)
    // The app runs on its own, so the launcher can go. Not while an install is still running: quitting would cancel it.
    if (store.getSettings().closeOnLaunch && !apps.hasActiveJobs()) setTimeout(() => app.quit(), 800)
  })
  handle('apps:uninstall', (id) => apps.uninstall(id))
  handle('apps:locate', async (id) => {
    const r = await dialog.showOpenDialog(mainWindow!, {
      title: `Locate ${findApp(id)?.name ?? 'the app'}: pick its program file`,
      filters: [{ name: 'Programs', extensions: ['exe', 'bat', 'cmd'] }],
      properties: ['openFile']
    })
    if (r.canceled || !r.filePaths[0]) return false
    await apps.locate(id, r.filePaths[0])
    return true
  })
  handle('apps:setShortcuts', (id, opts) => apps.setShortcuts(id, opts))
  handle('apps:openFolder', (id) => apps.openFolder(id))
  handle('apps:log', (id) => apps.getLog(id))

  handle('self:check', (force) => checkSelfUpdate(store, !!force).catch(() => null))
  handle('self:install', async (u) => {
    try {
      await installSelfUpdate(u, (f) => broadcast('self:progress', f))
    } catch (e) {
      broadcast('self:progress', -1)
      throw e
    }
  })
  handle('requirements:check', (reloadPath) => checkRequirements(reloadPath))
  handle('requirements:install', async (id) => {
    await installRequirement(id, (line) => broadcast('job:log', [{ appId: `requirement:${id}`, line }]))
    return checkRequirements()
  })

  // ---- Lifecycle ----
  const theme = store.getSettings().theme ?? 'dark'
  applyTheme(null, theme)
  mainWindow = createWindow(theme)
  void apps.detectInstalls().then(() => {
    if (store.getSettings().checkUpdatesOnStart) void apps.refreshLatest(false)
  })

  // Quitting mid-install would leave pip, git or a setup program running on their own, with a half-made install
  // behind; ask first, and cancel the jobs properly when the answer is to quit.
  let quitting = false
  const confirmQuit = (): boolean => {
    if (quitting || !apps.hasActiveJobs()) return true
    const names = apps.activeJobNames()
    const options: Electron.MessageBoxSyncOptions = {
      type: 'warning',
      title: 'RulZ AI Tools Suite',
      message: `${names.join(', ')} ${names.length > 1 ? 'are' : 'is'} still being installed or updated.`,
      detail: 'Quitting now cancels it. You can install or repair it again later.',
      buttons: ['Keep installing', 'Cancel install and quit'],
      defaultId: 0,
      cancelId: 0,
      noLink: true
    }
    const win = mainWindow && !mainWindow.isDestroyed() ? mainWindow : null
    const choice = win ? dialog.showMessageBoxSync(win, options) : dialog.showMessageBoxSync(options)
    if (choice !== 1) return false
    quitting = true
    void apps.cancelAll().finally(() => app.quit())
    return false
  }
  mainWindow.on('close', (e) => {
    if (!confirmQuit()) e.preventDefault()
  })
  app.on('before-quit', (e) => {
    if (!confirmQuit()) e.preventDefault()
  })

  app.on('second-instance', () => {
    if (!mainWindow) return
    if (mainWindow.isMinimized()) mainWindow.restore()
    mainWindow.focus()
  })
  app.on('window-all-closed', () => app.quit())
}

if (primary) void main()
