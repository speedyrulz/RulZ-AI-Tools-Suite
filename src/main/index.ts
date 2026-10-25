import { join, resolve, sep } from 'node:path'
import { pathToFileURL } from 'node:url'
import { app, BrowserWindow, dialog, nativeTheme, net, protocol, shell } from 'electron'
import { findApp } from '@shared/catalog'
import { AppManager } from './apps'
import { setFetch } from './github'
import { broadcast, handle } from './ipc'
import { checkRequirements, installRequirement } from './requirements'
import { Store } from './store'

const isDev = !app.isPackaged && !!process.env.ELECTRON_RENDERER_URL

if (!app.requestSingleInstanceLock()) app.quit()

// rulz://catalog/<app>/... serves the screenshots, icons and README copies shipped with the launcher.
protocol.registerSchemesAsPrivileged([
  { scheme: 'rulz', privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true } }
])

let mainWindow: BrowserWindow | null = null

function createWindow(): BrowserWindow {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 640,
    show: false,
    title: 'RulZ AI Tools Suite',
    backgroundColor: '#050505',
    titleBarStyle: 'hidden',
    titleBarOverlay: { color: '#050505', symbolColor: '#e5e5e5', height: 48 },
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

  if (isDev) void win.loadURL(process.env.ELECTRON_RENDERER_URL!)
  else void win.loadFile(join(__dirname, '../renderer/index.html'))
  return win
}

async function main(): Promise<void> {
  await app.whenReady()
  app.setAppUserModelId('com.rulz.aitoolssuite')
  // READMEs pick their dark-theme images through prefers-color-scheme.
  nativeTheme.themeSource = 'dark'
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
    emitLog: (line) => broadcast('job:log', line)
  })
  await apps.init()

  // ---- IPC ----
  handle('app:info', () => ({ version: app.getVersion(), userData, platform: process.platform }))
  handle('settings:get', () => store.getSettings())
  handle('settings:update', (patch) => store.updateSettings(patch))
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
    await apps.refreshLatest()
    return apps.views()
  })
  handle('apps:readme', (id) => apps.readme(id, store.getSettings().liveReadme))
  handle('apps:defaultDir', (id) => apps.defaultDir(id))
  handle('apps:install', (id, opts) => apps.install(id, opts))
  handle('apps:update', (id) => apps.update(id))
  handle('apps:repair', (id) => apps.repair(id))
  handle('apps:cancel', (id) => apps.cancel(id))
  handle('apps:launch', (id) => apps.launch(id))
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

  handle('requirements:check', () => checkRequirements())
  handle('requirements:install', async (id) => {
    await installRequirement(id, (line) => broadcast('job:log', { appId: `requirement:${id}`, line }))
    return checkRequirements()
  })

  // ---- Lifecycle ----
  mainWindow = createWindow()
  if (store.getSettings().checkUpdatesOnStart) void apps.refreshLatest()

  app.on('second-instance', () => {
    if (!mainWindow) return
    if (mainWindow.isMinimized()) mainWindow.restore()
    mainWindow.focus()
  })
  app.on('window-all-closed', () => app.quit())
}

void main()
