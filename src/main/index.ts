import { promises as fs } from 'node:fs'
import { basename, join, resolve } from 'node:path'
import { app, BrowserWindow, dialog, nativeTheme, shell } from 'electron'
import { CANDIDATE_URLS } from '@shared/defaults'
import type { ConnectionInfo, ModelActivity, ProjectInfo, ProjectState, Theme } from '@shared/types'
import { FIX_RUN_SHOWN, fixRunTask } from './agent/prompt'
import { ProjectSession } from './agent/session'
import { broadcast, handle } from './ipc'
import { checkPageInWindow } from './project/pageWindow'
import { AppRunner } from './project/runApp'
import { CopySnapshots, GitSnapshots, gitInstalled, type Snapshotter } from './project/snapshots'
import { ProjectStore } from './project/store'
import { findServer, normalizeUrl, RulzClient } from './rulz/client'
import { SettingsStore } from './settings'

const isDev = !app.isPackaged && !!process.env.ELECTRON_RENDERER_URL

// Tests run with their own data folder.
if (process.env.RULZ_AGENT_USER_DATA) app.setPath('userData', resolve(process.env.RULZ_AGENT_USER_DATA))

if (!app.requestSingleInstanceLock()) app.quit()

let mainWindow: BrowserWindow | null = null

function isDarkTheme(theme: Theme): boolean {
  return theme === 'dark' || (theme === 'system' && nativeTheme.shouldUseDarkColors)
}
const windowBackground = (theme: Theme): string => (isDarkTheme(theme) ? '#000000' : '#ffffff')

function createWindow(theme: Theme): BrowserWindow {
  const win = new BrowserWindow({
    width: 1360,
    height: 880,
    minWidth: 960,
    minHeight: 620,
    show: false,
    title: 'RulZ Coding Agent',
    backgroundColor: windowBackground(theme),
    autoHideMenuBar: true,
    ...(app.isPackaged ? {} : { icon: join(__dirname, '../../build/icon.png') }),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  })
  win.setMenuBarVisibility(false)
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

  const look = isDarkTheme(theme) ? 'dark' : 'light'
  if (isDev) void win.loadURL(`${process.env.ELECTRON_RENDERER_URL!}?theme=${look}`)
  else void win.loadFile(join(__dirname, '../renderer/index.html'), { query: { theme: look } })
  return win
}

/** Folder names Windows doesn't allow. */
function safeFolderName(name: string): string {
  const clean = name
    .trim()
    .replace(/[<>:"/\\|?*\x00-\x1f]/g, '')
    .replace(/[. ]+$/, '')
  if (!clean) throw new Error('Give the project a name, like "Video Upscaler".')
  if (/^(con|prn|aux|nul|com\d|lpt\d)$/i.test(clean)) throw new Error(`"${clean}" can't be used as a folder name on Windows. Try another name.`)
  return clean
}

async function main(): Promise<void> {
  await app.whenReady()
  app.setAppUserModelId('com.rulz.codingagent')
  const userData = app.getPath('userData')

  const settings = new SettingsStore(userData)
  await settings.init()
  const store = new ProjectStore(userData)
  const hasGit = await gitInstalled()

  // ---- Connection to RulZ LLM Suite ----
  let connection: ConnectionInfo = { status: 'checking', url: '', models: [] }
  const client = new RulzClient(() => ({ url: connection.url, apiKey: settings.get().apiKey }))
  const candidates = (): string[] => {
    const s = settings.get()
    const extra = process.env.RULZ_AGENT_CANDIDATES?.split(',').map((u) => u.trim()) ?? CANDIDATE_URLS
    return [s.serverUrl, s.lastServerUrl, ...extra].filter(Boolean).map(normalizeUrl)
  }
  const setConnection = (next: ConnectionInfo): void => {
    if (next.url !== connection.url) client.reset()
    const changed = JSON.stringify(next) !== JSON.stringify(connection)
    connection = next
    if (changed) broadcast('server:changed', connection)
    if (next.status === 'connected' && next.url !== settings.get().lastServerUrl) void settings.update({ lastServerUrl: next.url })
  }
  let connecting: Promise<ConnectionInfo> | null = null
  const connect = (): Promise<ConnectionInfo> => {
    connecting ??= findServer(candidates(), settings.get().apiKey)
      .then((info) => {
        setConnection(info)
        return info
      })
      .finally(() => {
        connecting = null
      })
    return connecting
  }
  const refreshModels = async (): Promise<void> => {
    if (connection.status !== 'connected') {
      await connect()
      return
    }
    try {
      setConnection({ ...connection, models: await client.listModels() })
    } catch {
      await connect()
    }
  }
  let activity: ModelActivity = 'idle'
  const setActivity = (a: ModelActivity): void => {
    if (a === activity) return
    activity = a
    broadcast('model:activity', a)
  }
  setInterval(() => void refreshModels(), 5000)

  // ---- Projects ----
  const trash = (abs: string): Promise<void> => shell.trashItem(abs)
  let session: ProjectSession | null = null
  let runner: AppRunner | null = null
  let current: ProjectInfo | null = null

  const snapshotter = async (kind: 'git' | 'copy' | 'auto'): Promise<Snapshotter> => {
    if (!session) throw new Error('No project is open.')
    const useGit = kind === 'git' || (kind === 'auto' && hasGit)
    return useGit
      ? new GitSnapshots(store.snapshotDir(session.root), session.root)
      : new CopySnapshots(store.backupDir(session.root), session.root, trash)
  }

  const requireSession = (path: string): ProjectSession => {
    if (!session || resolve(session.root).toLowerCase() !== resolve(path).toLowerCase()) throw new Error('That project is not open.')
    return session
  }
  const requireRunner = (path: string): AppRunner => {
    requireSession(path)
    return runner!
  }

  const closeProject = async (): Promise<void> => {
    if (session?.isRunning) throw new Error('A task is still running. Press Stop first.')
    await runner?.stop()
    await session?.close()
    session = null
    runner = null
    current = null
  }

  const openProject = async (path: string): Promise<ProjectState> => {
    const abs = resolve(path)
    const st = await fs.stat(abs).catch(() => null)
    if (!st?.isDirectory()) throw new Error(`The folder "${abs}" doesn't exist any more.`)
    if (session && resolve(session.root).toLowerCase() === abs.toLowerCase()) {
      return session.state(current!.name, current!.lastOpened)
    }
    await closeProject()
    const data = await store.load(abs)
    session = new ProjectSession(data, {
      client,
      settings: () => settings.get(),
      store,
      emit: (e) => broadcast('agent:event', e),
      activity: setActivity,
      models: () => connection.models,
      refreshModels: () => void refreshModels(),
      snapshotter,
      trash,
      checkPage: checkPageInWindow
    })
    runner = new AppRunner(abs, (e) => broadcast('run:event', e), async (file) => {
      const err = await shell.openPath(file)
      if (err) throw new Error(err)
    })
    current = { path: abs, name: basename(abs), lastOpened: Date.now() }
    await store.touch(current)
    return session.state(current.name, current.lastOpened)
  }

  // ---- IPC ----
  handle('app:info', () => ({ version: app.getVersion(), userData, platform: process.platform, gitInstalled: hasGit }))
  handle('settings:get', () => settings.get())
  handle('settings:update', async (patch) => {
    const before = settings.get()
    const next = await settings.update(patch)
    if (next.theme !== before.theme) mainWindow?.setBackgroundColor(windowBackground(next.theme))
    if (next.serverUrl !== before.serverUrl || next.apiKey !== before.apiKey) void connect()
    if (next.model !== before.model) client.reset()
    return next
  })
  handle('dialog:pickFolder', async ({ title, defaultPath }) => {
    const r = await dialog.showOpenDialog(mainWindow!, { title, defaultPath, properties: ['openDirectory', 'createDirectory'] })
    return r.canceled ? null : (r.filePaths[0] ?? null)
  })
  handle('shell:showItem', (path) => shell.showItemInFolder(path))
  handle('shell:openUrl', async (url) => {
    if (!/^https?:\/\//i.test(url)) throw new Error('Only web addresses can be opened.')
    await shell.openExternal(url)
  })

  handle('server:connect', () => connect())
  handle('server:status', () => connection)

  handle('projects:recent', () => store.recent())
  handle('projects:create', async (name, parentDir) => {
    const folder = safeFolderName(name)
    const parent = resolve(parentDir ?? settings.get().projectsDir)
    const path = join(parent, folder)
    const existing = await fs.readdir(path).catch(() => null)
    if (existing && existing.length) {
      throw new Error(`A folder named "${folder}" already exists in ${parent} and has files in it. Choose another name, or use "Open existing folder".`)
    }
    await fs.mkdir(path, { recursive: true })
    return openProject(path)
  })
  handle('projects:open', (path) => openProject(path))
  handle('projects:forget', (path) => store.forget(path))
  handle('projects:close', () => closeProject())

  handle('task:start', (path, text) => requireSession(path).start(text))
  handle('task:stop', (path) => requireSession(path).stop())
  handle('task:answer', (path, id, answer) => requireSession(path).answer(id, answer))
  handle('task:undo', async (path) => {
    const s = requireSession(path)
    await runner?.stop()
    await s.undo()
    return s.state(current!.name, current!.lastOpened)
  })
  handle('files:diff', (path, taskId, rel) => requireSession(path).diff(taskId, rel))

  handle('run:state', (path) => requireRunner(path).get())
  handle('run:start', (path) => requireRunner(path).start(requireSession(path).data.runCommand))
  handle('run:stop', (path) => requireRunner(path).stop())
  handle('run:setCommand', async (path, command) => {
    const s = requireSession(path)
    s.data.runCommand = command?.trim() || null
    await s.saveNow()
    if (!s.data.runCommand) requireRunner(path).forgetCommand()
    return requireRunner(path).get()
  })
  handle('run:askAgent', (path) => requireSession(path).start('How do I run this app?', { howToRun: true }))
  handle('run:fix', async (path) => {
    const s = requireSession(path)
    if (s.isRunning) throw new Error('A task is already running. Wait for it to finish or press Stop.')
    const r = requireRunner(path)
    const before = r.get()
    // A running app holds its files open.
    if (before.running) await r.stop()
    s.start(fixRunTask(before), { shown: FIX_RUN_SHOWN })
  })

  // ---- Lifecycle ----
  mainWindow = createWindow(settings.get().theme)
  void connect()

  app.on('second-instance', () => {
    if (!mainWindow) return
    if (mainWindow.isMinimized()) mainWindow.restore()
    mainWindow.focus()
  })
  app.on('window-all-closed', () => app.quit())
  app.on('before-quit', () => {
    void runner?.stop()
    session?.stop()
    void session?.background.stopAll()
    if (session) void store.save(session.data)
  })
}

void main()
