import { promises as fs } from 'node:fs'
import { join } from 'node:path'
import { homedir } from 'node:os'
import { randomUUID } from 'node:crypto'
import { app, BrowserWindow, dialog, Menu, nativeTheme, safeStorage, shell, Tray } from 'electron'
import { batchAdapters } from '@shared/batch'
import { resolveDraft } from '@shared/draft'
import type { RemoteIpcInvoke } from '@shared/remote'
import { META_TOOL_DEFS, toolCatalogNote } from '@shared/tools'
import type { LoadConfig, LoadedInstance, McpServerConfig, ModelEntry, Theme } from '@shared/types'
import { ChatService } from './chat/service'
import { ConversationStore } from './chat/conversations'
import { EngineService } from './engine/engine'
import { HardwareMonitor } from './engine/hardware'
import { McpManager } from './mcp/manager'
import { InstanceManager, resolveAdapters } from './engine/instances'
import { broadcast, handle } from './ipc'
import { estimateMemory, type FreeMemory } from './models/estimate'
import { AdapterConverter } from './models/convert'
import { createModelCatalog } from './models/catalog'
import { ModelLibrary } from './models/library'
import { ApiServer } from './server/apiServer'
import { parseLoadConfig } from './settings/schema'
import { SettingsStore } from './settings/store'
import { FILE_TOOLS_SERVER } from './files/fileTools'
import { PdfWorkerReader } from './pdf/reader'
import { readPdfAttachment } from './pdf/tool'
import { API_RE_PREFIX, CLONE_FILE_TOOLS, extractAnswer, PRACTICE_HIDDEN_TOOLS } from '@shared/re'
import { BatchStore } from './re/batches'
import { PracticePool, type PoolHost } from './re/pool'
import { poolHandlers } from './re/poolIpc'
import { createRemoteService } from './remote/service'
import { BenchmarkRunner } from './bench/runner'
import { BenchStore } from './bench/store'
import { WritingBenchStore } from './wbench/store'
import { gradePractice, listPractice, preparePractice } from './re/practice'
import { buildOverview, MAX_OVERVIEW_SESSIONS } from './re/overview'
import { findPython, type PythonCommand } from './re/python'
import { exportDataset, TraceStore, type ExportSession } from './re/traces'
import { registerNovelIpc } from './novel/ipc'
import { NovelStore, runAs, WINDOW_OWNER } from './novel/store'
import { createPhoneService, type PhoneService } from './phone/host'
import { settingsForRenderer, settingsPatchFromRenderer } from './phone/settings'
import { ChatEventTap, chatAdapter, chatBusyReason, modelsAdapter, storeAdapter } from './writing/adapters'
import { WritingEvents, writingHandlers } from './writing/ipc'
import { WritingRuns } from './writing/runs'
import { PHONE_WRITING_CHANNELS } from '@shared/phone'
import { PHONE_ADVENTURE_CHANNELS } from '@shared/adventure'
import { adventureDesktopHandlers, adventureHandlers } from './adventure/ipc'
import { AdventureRuns } from './adventure/runs'
import { AdventureStore } from './adventure/store'
import { ModelGate } from './modelGate'
import { writeTextAtomic } from './util/jsonFile'

const MiB = 1024 * 1024
const isDev = !app.isPackaged && !!process.env.ELECTRON_RENDERER_URL

// A second copy only hands over to the first (second-instance shows its window). It must not run
// main(): its startup cleanup would kill the first copy's models.
const isPrimaryInstance = app.requestSingleInstanceLock()
if (!isPrimaryInstance) app.quit()

let mainWindow: BrowserWindow | null = null
let tray: Tray | null = null
let phone!: PhoneService
/** Set once the app is really exiting, so close-to-tray lets the window close. */
let quitting = false

function showMainWindow(): void {
  if (!mainWindow) return
  mainWindow.show()
  if (mainWindow.isMinimized()) mainWindow.restore()
  mainWindow.focus()
}

/** The tray icon only exists while the window is hidden to it. */
function ensureTray(): void {
  if (tray) return
  // build/ isn't packaged; electron-builder copies the icon into resources.
  const icon = app.isPackaged ? join(process.resourcesPath, 'icon.ico') : join(__dirname, '../../build/icon.ico')
  tray = new Tray(icon)
  tray.setToolTip('RulZ LLM Suite')
  tray.setContextMenu(
    Menu.buildFromTemplate([
      { label: 'Show RulZ LLM Suite', click: showMainWindow },
      { type: 'separator' },
      { label: 'Quit', click: () => app.quit() }
    ])
  )
  tray.on('click', showMainWindow)
}

function destroyTray(): void {
  tray?.destroy()
  tray = null
}

/** Window background for a theme, so no wrong-coloured frame shows while the page loads or resizes. */
function isDarkTheme(theme: Theme): boolean {
  return theme === 'dark' || (theme === 'system' && nativeTheme.shouldUseDarkColors)
}
const windowBackground = (theme: Theme): string => (isDarkTheme(theme) ? '#050505' : '#ffffff')

function createWindow(theme: Theme): BrowserWindow {
  const win = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1024,
    minHeight: 640,
    show: false,
    title: 'RulZ LLM Suite',
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

  // The page applies this before its first paint; the saved settings take over once they load.
  const look = isDarkTheme(theme) ? 'dark' : 'light'
  if (isDev) void win.loadURL(`${process.env.ELECTRON_RENDERER_URL!}?theme=${look}`)
  else void win.loadFile(join(__dirname, '../renderer/index.html'), { query: { theme: look } })
  return win
}

async function main(): Promise<void> {
  await app.whenReady()
  app.setAppUserModelId('com.rulz.llmsuite')
  const userData = app.getPath('userData')

  const settings = new SettingsStore(userData)
  await settings.init()

  const engine = new EngineService(() => settings.get().enginePath)
  const hardware = new HardwareMonitor(
    (s) => broadcast('hardware:update', s),
    () =>
      (!!mainWindow && mainWindow.isVisible() && !mainWindow.isMinimized() && mainWindow.isFocused()) ||
      instances.list().some((i) => i.status === 'loading')
  )
  const library = new ModelLibrary(
    join(userData, 'gguf-cache.json'),
    () => settings.get().modelsDir,
    () => broadcast('models:changed', catalog.list()),
    (adapters) => broadcast('adapters:changed', adapters)
  )

  const freeMemory = (cfg: LoadConfig): FreeMemory => {
    const snap = hardware.snapshot()
    // Models that will be unloaded first give their memory back.
    // Models on rentals use no memory here and are not unloaded by a load here.
    const unloading = settings.get().unloadPreviousOnLoad ? instances.list().filter((i) => !i.host) : []
    const freed = (index: number): number =>
      unloading.reduce((sum, i) => sum + (i.estimate?.perGpu?.find((g) => g.index === index)?.bytes ?? 0), 0)
    const all = [...snap.gpus]
      .sort((a, b) => a.index - b.index)
      .map((g) => {
        const totalBytes = g.memTotalMiB * MiB
        return { index: g.index, name: g.name, totalBytes, freeBytes: Math.min(totalBytes, (g.memTotalMiB - g.memUsedMiB) * MiB + freed(g.index)) }
      })
    const gpus = cfg.splitMode === 'single' ? snap.gpus.filter((g) => g.index === cfg.mainGpu) : snap.gpus
    let vram = gpus.length ? gpus.reduce((sum, g) => sum + (g.memTotalMiB - g.memUsedMiB) * MiB, 0) : null
    if (vram !== null && unloading.length) {
      vram += unloading.reduce((sum, i) => sum + (i.estimate?.gpuBytes ?? 0), 0)
      vram = Math.min(vram, gpus.reduce((sum, g) => sum + g.memTotalMiB * MiB, 0))
    }
    return { vramBytes: vram, ramBytes: snap.ramTotalBytes - snap.ramUsedBytes, gpus: all }
  }
  const draftFor = (model: ModelEntry, cfg: LoadConfig) => resolveDraft(model, cfg, library.list())
  // Adapter files the load would pass to llama-server (same filter as the load itself), for the estimate.
  const adapterBytes = (cfg: LoadConfig): number =>
    resolveAdapters(cfg.adapters ?? [], (id) => library.getAdapter(id)).loaded.reduce((sum, a) => sum + (library.getAdapter(a.id)?.sizeBytes ?? 0), 0)
  const estimate = (model: ModelEntry, cfg: LoadConfig) =>
    estimateMemory({ ...model, draft: draftFor(model, cfg) }, cfg, freeMemory(cfg), adapterBytes(cfg))

  const getDefaults = (id: string): LoadConfig => settings.getModelDefaults(id).config
  // Models on rented vast.ai instances (docs/vast-design.md).
  const remote = createRemoteService({
    userData,
    homedir: homedir(),
    env: process.env,
    safeStorage,
    getLocalModel: (id) => library.get(id),
    getAdapter: (id) => library.getAdapter(id),
    getDefaults,
    draftFor,
    adapterBytes,
    localEngineBuild: async () => {
      const build = /\bbuild:?\s+(\d+)/i.exec((await engine.getInfo()).version ?? '')?.[1]
      return build ? Number(build) : null
    },
    loadOn: (host, modelId, config) => instances.load(modelId, config, host),
    isLoaded: (modelId) => instances.list().some((i) => i.modelId === modelId),
    forgetDefaults: (modelId) => settings.setModelDefaults(modelId, null),
    // A batch run's tasks there go back to pending once its instance is gone.
    onStopping: async (rentalId) => {
      const there = instances.list().filter((i) => i.host?.rentalId === rentalId)
      await Promise.allSettled(there.map((i) => instances.unload(i.instanceId)))
    },
    onChange: (state) => broadcast('remote:changed', state),
    onModels: () => broadcast('models:changed', catalog.list())
  })
  const catalog = createModelCatalog(library, remote.models)

  const instances = new InstanceManager({
    engine,
    launcher: remote.launcher,
    onUse: (instance, by, what) => (instance.host ? remote.rentals.use(instance.host.rentalId, what, by) : () => undefined),
    getModel: (id) => catalog.get(id),
    draftFor,
    getDefaults,
    unloadPreviousOnLoad: () => settings.get().unloadPreviousOnLoad,
    estimate,
    getAdapter: (id) => library.getAdapter(id),
    pidFile: join(userData, 'running-pids.json'),
    onChange: (list) => broadcast('instances:changed', list),
    onError: (e) => broadcast('instances:error', e),
    onGone: (id) => chat.forget(id)
  })
  await instances.cleanupOrphans()

  const converter = new AdapterConverter({
    resolveExe: () => engine.resolveExe(),
    getModelsDir: () => settings.get().modelsDir,
    findPython,
    onLog: (line) => broadcast('adapters:convertLog', line),
    onConverted: () => void library.scan()
  })

  const mcpConfigPath = join(userData, 'mcp.json')
  // The RE calculator (a sandboxed Python MCP server from the RE training kit) runs when Python 3 is installed.
  const calcScript = app.isPackaged ? join(process.resourcesPath, 'calc_mcp.py') : join(__dirname, '../../resources/calc_mcp.py')
  let python: PythonCommand | null = null
  // PDFs are parsed in worker threads (out/main/pdfWorker.js), never here. pdf.js's character maps (for Chinese, Japanese
  // and Korean text) ship next to the app as a resource folder.
  const pdfReader = new PdfWorkerReader({
    workerPath: join(__dirname, 'pdfWorker.js'),
    cmapDir: () => (app.isPackaged ? join(process.resourcesPath, 'pdfjs-cmaps') : join(__dirname, '../../node_modules/pdfjs-dist/cmaps'))
  })
  const mcp = new McpManager(
    mcpConfigPath,
    (s) => broadcast('mcp:changed', s),
    (c) => broadcast('mcp:config', c),
    (): Record<string, McpServerConfig> =>
      python ? { calc: { command: python.command, args: [...python.args, '-I', calcScript] } } : {},
    () => broadcast('mcp:reaTaken', undefined),
    pdfReader
  )
  void findPython()
    .then((p) => {
      python = p
    })
    .finally(() => void mcp.init())
  const traces = new TraceStore(join(userData, 're-data'))
  const practiceRoot = join(userData, 're-practice')
  const batches = new BatchStore(join(userData, 're-batches.json'))
  const apiRe = (id: string): string => API_RE_PREFIX + id
  const benchStore = new BenchStore(join(userData, 'benchmarks.json'))
  const writingBenchStore = new WritingBenchStore(userData)
  const bench = new BenchmarkRunner({
    instances,
    getModel: (id) => catalog.get(id),
    estimate,
    draftFor,
    sampleGpus: async () => (await hardware.refresh()).gpus,
    engineVersion: async () => (await engine.getInfo()).version,
    onChange: (run) => broadcast('bench:changed', run),
    store: benchStore
  })

  const apiServer = new ApiServer({
    instances,
    getModels: () => catalog.list(),
    getAdapters: () => library.listAdapters(),
    getSettings: () => settings.get().apiServer,
    getLoadDefaults: (id) => settings.getModelDefaults(id),
    setLoadDefaults: (id, config) => settings.setModelDefaults(id, config),
    estimate,
    gpus: () => hardware.snapshot().gpus,
    onStatus: (s) => broadcast('server:changed', s),
    onRequest: (e) => broadcast('server:request', e),
    re: {
      enabled: () => settings.get().re.recordApi,
      record: (id, round) => {
        const dir = apiRe(id)
        // A session the app never described still gets an entry, so it shows on the page.
        void traces
          .ensureApiSession(dir, { client: 'Another app', title: id })
          .then(() => traces.record(dir, round, { shrinkIsTrim: true }))
          .then(() => broadcast('re:changed', undefined))
          .catch((err) => console.error('Could not record an RE round:', err))
      },
      session: (id) => traces.apiSession(apiRe(id)),
      update: async (id, patch) => {
        const s = await traces.updateApiSession(apiRe(id), patch)
        broadcast('re:changed', undefined)
        return s
      },
      overview: async () => {
        const chats = []
        const reSummaries = (await conversations.list()).filter((s) => s.reOutcome).slice(0, MAX_OVERVIEW_SESSIONS)
        for (const s of reSummaries) {
          const c = await conversations.get(s.id).catch(() => null)
          if (c?.re) chats.push({ id: c.id, title: c.title, updatedAt: c.updatedAt, re: c.re })
        }
        const re = settings.get().re
        return buildOverview({ chats, traces: await traces.list(), practice: await listPractice(re.kitDir), recording: re.recordApi })
      },
      preparePractice: async (taskId) => {
        const { target, prompt, task } = await preparePractice(settings.get().re.kitDir, practiceRoot, taskId)
        return { target, prompt, family: task.family, tier: task.tier }
      },
      gradePractice: async (taskId, target, text) => {
        const answer = extractAnswer(text)
        const { passed, expected } = await gradePractice(settings.get().re.kitDir, practiceRoot, taskId, target, answer)
        return { answer, passed, expected }
      },
      // The kit field saves `{ re: { ...re, kitDir } }` through settings:update. That handler does not
      // broadcast; the window applies its own reply. re:changed is the event the page already follows
      // when another app changes RE data.
      setKit: async (kitDir) => {
        const saved = await settings.update({ re: { ...settings.get().re, kitDir } })
        broadcast('re:changed', undefined)
        return listPractice(saved.re.kitDir)
      }
    }
  })

  // One model job at a time for the jobs main runs itself: the phone's Writing runs and adventure jobs.
  const modelGate = new ModelGate()
  // Phone writing runs wait for their own requests' events through this tap.
  const chatTap = new ChatEventTap()
  // Requests of the practice pool: the window has no part in them (it would release their tool rounds).
  const poolRequests = new Set<string>()
  const chat = new ChatService(instances, (e) => {
    if (!poolRequests.has(e.requestId)) broadcast('chat:event', e)
    chatTap.push(e)
  }, (req) => {
    const disabled = new Set(settings.get().disabledTools)
    // The write/read file tools exist only in clone sessions.
    if (!req.files) for (const n of CLONE_FILE_TOOLS) disabled.add(n)
    // Practice sessions keep the tools and prompts their recorded data was made with.
    if (req.practice) for (const n of PRACTICE_HIDDEN_TOOLS) disabled.add(n)
    if (req.mode === 'all') return { defs: mcp.toolDefinitions(disabled) }
    // On demand: the three loader tools, the loaded tools' full descriptions, and a name catalog.
    const available = mcp.availableTools(disabled)
    // The built-in file tools are small and often needed first, so they stay loaded.
    const pinned = available.filter((t) => t.server === FILE_TOOLS_SERVER).map((t) => t.qualifiedName)
    const active = [...new Set([...pinned, ...req.active])].filter((n) => available.some((t) => t.qualifiedName === n))
    return {
      defs: [...META_TOOL_DEFS, ...mcp.toolDefinitions(disabled, new Set(active))],
      note: available.length ? toolCatalogNote(available, active) : undefined
    }
  }, (conversationId, round) => void traces.record(conversationId, round))
  const conversations = new ConversationStore(join(userData, 'conversations'))

  // ---- IPC ----
  handle('app:info', () => ({ version: app.getVersion(), userData, platform: process.platform }))
  // The pairing code never goes to the window (only phone:pairing carries it); phone access is switched by phone:*.
  handle('settings:get', () => settingsForRenderer(settings.get()))
  handle('settings:update', async (patch) => {
    const before = settings.get()
    const saved = await settings.update(settingsPatchFromRenderer(patch))
    const next = settingsForRenderer(saved)
    if (saved.phonePort !== before.phonePort) void phone.access.start()
    if (next.modelsDir !== before.modelsDir) void library.scan()
    if (next.enginePath !== before.enginePath) void engine.getInfo(true)
    if (next.theme !== before.theme) mainWindow?.setBackgroundColor(windowBackground(next.theme))
    if (!next.minimizeToTray && !next.closeToTray && tray) showMainWindow()
    // The key, CORS and model settings are read per request; the address needs a new listener, or
    // switching from the network to this computer only would leave it reachable from the network.
    const api = (s: typeof next): string => `${s.apiServer.bind}:${s.apiServer.port}`
    if (api(next) !== api(before) && apiServer.getStatus().running) {
      // A failure (port in use) shows in the server status, not as a failed settings save.
      await apiServer
        .stop()
        .then(() => apiServer.start())
        .catch(() => undefined)
    }
    return next
  })
  handle('dialog:pickFolder', async ({ title, defaultPath }) => {
    const r = await dialog.showOpenDialog(mainWindow!, { title, defaultPath, properties: ['openDirectory'] })
    return r.canceled ? null : (r.filePaths[0] ?? null)
  })
  handle('dialog:pickFile', async ({ title, filters }) => {
    const r = await dialog.showOpenDialog(mainWindow!, { title, filters, properties: ['openFile'] })
    return r.canceled ? null : (r.filePaths[0] ?? null)
  })
  handle('shell:showItem', (path) => shell.showItemInFolder(path))
  handle('shell:openPath', async (path) => {
    await shell.openPath(path)
  })

  handle('engine:info', (refresh) => engine.getInfo(refresh))

  handle('models:list', () => catalog.list())
  handle('models:rescan', async () => {
    await library.scan()
    return catalog.list()
  })
  handle('models:trash', async (modelId) => {
    const model = catalog.get(modelId)
    if (!model) throw new Error('Model not found')
    if (model.remote) throw new Error('This model is defined on the vast.ai page; delete it there.')
    if (instances.list().some((i) => i.modelId === modelId)) throw new Error('Eject the model before deleting it.')
    for (const f of model.files) await shell.trashItem(f)
    await settings.setModelDefaults(modelId, null)
    await library.scan()
  })
  handle('adapters:list', () => library.listAdapters())
  handle('adapters:trash', async (adapterId) => {
    const adapter = library.getAdapter(adapterId)
    if (!adapter) throw new Error('Adapter not found')
    // A loaded adapter is held open by llama-server; its model has to be ejected first.
    if (instances.list().some((i) => !i.host && i.adapters.some((a) => a.id === adapterId))) throw new Error('Eject the model using this adapter before deleting it.')
    await shell.trashItem(adapter.path)
    await library.scan()
  })
  handle('adapters:converterStatus', () => converter.status())
  handle('adapters:convert', (req) => converter.convert(req))
  handle('adapters:cancelConvert', () => converter.cancel())
  handle('models:getDefaults', (modelId) => settings.getModelDefaults(modelId))
  handle('models:setDefaults', (modelId, config) => settings.setModelDefaults(modelId, config))
  handle('models:estimate', (modelId, config) => {
    const model = catalog.get(modelId)
    if (!model) throw new Error('Model not found')
    return estimate(model, config)
  })

  handle('instances:list', () => instances.list())
  handle('instances:load', async (modelId, config) => {
    // Checked like saved settings, so a bad value from the page can't reach the command line.
    const inst = await instances.load(modelId, config ? parseLoadConfig(config) : null)
    // Loading it yourself makes it yours, even if the API server started it: no idle-unload.
    apiServer.forget(inst.instanceId)
    return inst
  })
  // A chat that needs its model: an instance already loading (from the picker, with other settings)
  // is waited for rather than replaced by one with the saved defaults.
  handle('instances:ensure', async (modelId) => {
    const { instance } = await instances.ensure(modelId)
    apiServer.forget(instance.instanceId)
    return instance
  })
  handle('instances:unload', (id) => {
    apiServer.forget(id)
    return instances.unload(id)
  })
  handle('instances:unloadAll', () => {
    for (const i of instances.list()) apiServer.forget(i.instanceId)
    return instances.unloadAll()
  })
  handle('instances:log', (id) => instances.getLog(id))

  handle('hardware:snapshot', () => hardware.refresh())

  handle('chat:start', (req) => {
    void chat.start(req, 'window')
  })
  handle('chat:abort', (id) => chat.abort(id))
  handle('chat:releaseTurn', (id) => chat.releaseTurn(id))
  handle('chat:countTokens', (modelId, messages, tools) => chat.countTokens(modelId, messages, tools))
  handle('chat:countText', (modelId, text) => chat.countText(modelId, text))

  handle('conversations:list', () => conversations.list())
  handle('conversations:get', (id) => conversations.get(id))
  handle('conversations:save', (c) => conversations.save(c))
  handle('conversations:delete', async (id) => {
    await conversations.delete(id)
    // Its recorded RE data goes too (to the Recycle Bin, like the chat), after any round still being
    // written; a round that finishes later (a stopped reply's last event) is not recorded.
    await traces.delete(id, (dir) => shell.trashItem(dir), { forever: true }).catch(() => undefined)
  })
  handle('folders:list', () => conversations.listFolders())
  handle('folders:save', (f) => conversations.saveFolders(f))

  handle('mcp:getConfig', () => mcp.config())
  handle('mcp:saveConfig', (text) => mcp.saveConfig(text))
  handle('mcp:status', () => mcp.status())
  handle('mcp:restart', (name) => mcp.restart(name))
  // Running tool calls by the renderer's key, so Stop can cancel the one in progress.
  const toolCalls = new Map<string, AbortController>()
  handle('mcp:callTool', async (name, args, callKey, files) => {
    const ctrl = new AbortController()
    if (callKey) toolCalls.set(callKey, ctrl)
    try {
      return await mcp.callTool({ name, arguments: args }, ctrl.signal, files)
    } finally {
      if (callKey) toolCalls.delete(callKey)
    }
  })
  handle('mcp:cancelTool', (callKey) => {
    toolCalls.get(callKey)?.abort()
  })
  const pdfReads = new Map<string, AbortController>()
  handle('attachments:readPdf', async (name, bytes, readKey) => {
    const ctrl = new AbortController()
    pdfReads.set(readKey, ctrl)
    try {
      return await readPdfAttachment(name, bytes, pdfReader, ctrl.signal)
    } finally {
      pdfReads.delete(readKey)
    }
  })
  handle('attachments:cancelPdf', (readKey) => {
    pdfReads.get(readKey)?.abort()
  })

  handle('server:status', () => apiServer.getStatus())
  handle('server:start', () => apiServer.start())
  handle('server:stop', () => apiServer.stop())
  handle('server:log', () => apiServer.getLog())
  handle('server:clearLog', () => apiServer.clearLog())

  handle('re:traces', () => traces.list())
  handle('re:deleteTrace', (id) => traces.delete(id))
  handle('re:setApiOutcome', async (id, outcome) => {
    if (!id.startsWith(API_RE_PREFIX) || !(await traces.apiSession(id))) throw new Error('Not an API session.')
    await traces.updateApiSession(id, { outcome })
  })
  handle('re:dataFolder', async () => {
    const dir = join(userData, 're-data')
    await fs.mkdir(dir, { recursive: true })
    return dir
  })
  handle('re:calcStatus', () => ({ available: !!python, python: python ? `${python.command} ${python.version}` : null }))
  handle('re:export', async (opts) => {
    const sessions: ExportSession[] = []
    for (const s of await conversations.list()) {
      if (!s.reOutcome) continue
      const c = await conversations.get(s.id).catch(() => null)
      if (c?.re) sessions.push({ id: c.id, re: c.re })
    }
    // Sessions other apps ran through the API server.
    for (const t of await traces.list()) {
      if (!t.api) continue
      const a = t.api
      sessions.push({
        id: t.conversationId,
        source: a.client,
        kind: a.kind,
        // The practice task too, so these samples carry it and count toward the per-task limit.
        re: { target: a.target, goal: a.goal, collect: true, outcome: a.outcome, notes: a.notes, ...(a.practice ? { practice: a.practice } : {}) }
      })
    }
    return exportDataset(traces, sessions, opts)
  })
  handle('re:practiceList', (kitDir) => listPractice(kitDir))
  const practicePrepare = async (kitDir: string, taskId: string): Promise<{ target: string; prompt: string; family: string; tier: string }> => {
    const { target, prompt, task } = await preparePractice(kitDir, practiceRoot, taskId)
    return { target, prompt, family: task.family, tier: task.tier }
  }
  const practiceGrade = async (kitDir: string, taskId: string, target: string, finalText: string): Promise<{ answer: string | null; passed: boolean; expected: string }> => {
    const answer = extractAnswer(finalText)
    const { passed, expected } = await gradePractice(kitDir, practiceRoot, taskId, target, answer)
    return { answer, passed, expected }
  }
  handle('re:practicePrepare', practicePrepare)
  handle('re:practiceGrade', practiceGrade)

  // ---- Practice batch runs: several sessions at once, here or on rentals (docs/vast-design.md, section 8) ----
  const poolHost = (i: LoadedInstance, dollarsPerHour: number): PoolHost => ({
    instanceId: i.instanceId,
    hostLabel: i.host?.label ?? 'This PC',
    dollarsPerHour,
    ...(i.host ? { rentalId: i.host.rentalId } : {}),
    modelName: i.modelName,
    contextLength: i.contextLength,
    adapters: batchAdapters(i.adapters)
  })
  const pool = new PracticePool({
    now: () => Date.now(),
    sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
    uid: () => randomUUID(),
    store: batches,
    conversations,
    events: {
      batchChanged: (run, active, stopping) => broadcast('re:batchChanged', { run, active, stopping }),
      poolChats: (ids) => broadcast('re:poolChats', ids),
      chatLive: (c) => broadcast('re:chatLive', c),
      conversationSaved: (s) => broadcast('conversations:saved', s)
    },
    busy: (hosts) => {
      if (bench.isRunning()) return 'A load benchmark is running. Wait for it or stop it first.'
      if (!hosts.some((h) => h.kind === 'local')) return null
      const held = modelGate.busyReason()
      if (held) return held
      // A model here that is in use: a reply, a tool round or an API request.
      return instances.list().some((i) => !i.host && i.status === 'ready' && instances.idleSince(i.instanceId) === null) ? 'A reply is still being written on this PC. Wait for it or stop it first.' : null
    },
    prepareHosts: (target, modelId, load) =>
      Promise.all(
        target.hosts.map(async (h): Promise<PoolHost> => {
          const model = catalog.get(modelId)
          if (!model) throw new Error(`${modelId} is no longer in your models.`)
          if (h.kind === 'local') {
            if (model.remote) throw new Error(`${model.name} runs on a vast.ai rental, not on this PC.`)
            const inst = load
              ? await instances.load(modelId, { ...getDefaults(modelId), ...load })
              : (instances.readyFor(modelId) ?? (await instances.ensure(modelId)).instance)
            return poolHost(inst, 0)
          }
          if (!model.remote) throw new Error(`${model.name} runs on this PC. Choose a vast.ai model to run on a rental.`)
          const host = remote.host(h.rentalId)
          const release = remote.rentals.use(h.rentalId, 'loading a model', 'window')
          try {
            const there = instances.list().find((i) => i.modelId === modelId && i.host?.rentalId === h.rentalId && i.status === 'ready')
            const inst = load ? await instances.load(modelId, { ...getDefaults(modelId), ...load }, host) : (there ?? (await instances.load(modelId, null, host)))
            return poolHost(inst, remote.rentals.info(h.rentalId)?.dollarsPerHour ?? 0)
          } finally {
            release()
          }
        })
      ),
    useHost: (instanceId) => instances.acquire(instanceId, 'window', 'batch run'),
    servers: () => mcp.status(),
    openScope: (opts) => mcp.openScope(opts),
    settings: () => settings.get(),
    chat: {
      start: async (req, onEvent) => {
        poolRequests.add(req.requestId)
        const off = chatTap.subscribe(req.requestId, onEvent)
        try {
          await chat.start(req, 'window')
        } finally {
          off()
          poolRequests.delete(req.requestId)
        }
      },
      abort: (id) => chat.abort(id),
      releaseTurn: (id) => chat.releaseTurn(id)
    },
    instance: (id) => instances.get(id),
    instanceState: (id) => {
      const i = instances.get(id)
      if (!i || i.status !== 'ready') return 'gone'
      return i.reconnecting ? 'reconnecting' : 'ready'
    },
    prepare: practicePrepare,
    grade: practiceGrade,
    onEnded: (_run, rentalIds) => {
      if (rentalIds.length) remote.rentals.stopAfterRun(rentalIds)
    },
    warn: (message) => console.warn(message)
  })
  const poolTable = poolHandlers(pool, (id) => batches.delete(id))
  for (const channel of Object.keys(poolTable) as (keyof typeof poolTable)[]) handle(channel, poolTable[channel] as never)
  handle('re:batches', () => batches.list())

  // ---- vast.ai (window only: never registered for the phone or the API server) ----
  for (const channel of Object.keys(remote.handlers) as (keyof RemoteIpcInvoke)[]) handle(channel, remote.handlers[channel] as never)

  handle('bench:list', () => benchStore.list())
  handle('wbench:list', () => writingBenchStore.list())
  handle('wbench:save', (run) => writingBenchStore.save(run))
  handle('wbench:delete', (id) => writingBenchStore.delete(id))
  handle('wbench:prompts', () => writingBenchStore.prompts())
  handle('wbench:savePrompts', (prompts) => writingBenchStore.savePrompts(prompts))
  handle('bench:start', (req) => {
    if (pool.busy) throw new Error('A batch run is in progress. Wait for it or stop it first.')
    return bench.start(req)
  })
  handle('bench:stop', () => bench.stop())
  handle('bench:delete', (id) => benchStore.delete(id))

  // ---- Writing mode ----
  const novel = new NovelStore({
    rootSetting: () => settings.get().writingRoot,
    documents: () => app.getPath('documents'),
    trashItem: (path) => shell.trashItem(path),
    // The window already has the result of its own requests; everything else (a phone's save, an outside edit)
    // goes to the window and to the phones.
    emit: (change, origin) => {
      if (origin !== WINDOW_OWNER) broadcast('novel:changed', change)
      phone.emit('novel:changed', change)
    },
    appVersion: () => app.getVersion()
  })
  // The close-time handshake (data safety rule 4): the renderer saves or journals its dirty buffers and answers.
  let flushTimeoutMs = 3000
  let flushing: Promise<void> | null = null
  let flushDone: (() => void) | null = null
  let novelFlushed = false
  const flushNovel = (): Promise<void> => {
    if (!novel.hasOpenProject()) return Promise.resolve()
    flushing ??= new Promise<void>((resolve) => {
      const finish = (): void => {
        clearTimeout(timer)
        flushDone = null
        // A cancelled shutdown must not leave a settled promise that skips the next flush.
        flushing = null
        resolve()
      }
      const timer = setTimeout(finish, flushTimeoutMs)
      flushDone = finish
      broadcast('novel:flush', undefined)
    })
    return flushing
  }
  registerNovelIpc(novel, () => flushDone?.())

  // ---- Phone access (RulZ Writer for Android) ----
  // `phone` is read by the novel store's emit above, which only runs after startup.
  phone = createPhoneService({
    settings,
    novel,
    version: app.getVersion(),
    emitStatus: (status) => broadcast('phone:changed', status)
  })
  handle('phone:status', () => phone.access.status())
  handle('phone:setEnabled', (enabled) => phone.access.setEnabled(enabled === true))
  handle('phone:pairing', () => phone.access.pairing())
  handle('phone:newToken', () => phone.access.newToken())
  handle('phone:disconnect', (clientId) => phone.access.disconnect(clientId))
  // Writing runs: AI actions started from the phone, run here (docs/phone-access-design.md, "Writing runs").
  const writingEvents = new WritingEvents()
  const chatReason = chatBusyReason(chat, () => bench.isRunning(), modelGate)
  // A batch run on rentals alone leaves this PC's model free.
  const busyReason = (): string | null =>
    pool.busy && pool.activeInstances.some((id) => !instances.get(id)?.host) ? 'A practice batch run is using the model on the computer.' : chatReason()
  const writingRuns = new WritingRuns({
    store: storeAdapter(novel),
    chat: chatAdapter(chat, chatTap, 'phone'),
    models: modelsAdapter(catalog, instances),
    busyReason,
    gate: modelGate,
    // A phone may use a project's jobs while it holds the project open (a reconnected phone has a new id).
    holds: (owner, folder) => novel.ownersOf(folder).includes(owner),
    emit: writingEvents.emit
  })
  writingEvents.subscribe((job) => phone.emit('writing:job', job))
  const writing = writingHandlers(writingRuns)
  for (const channel of PHONE_WRITING_CHANNELS) {
    // runAs: a state patch from Apply is announced as the phone's change, so the window hears about it.
    phone.handlers.register(channel, ((owner: string, ...args: unknown[]) => runAs(owner, () => writing[channel](owner, ...args))) as never)
  }

  // ---- Adventure mode (docs/adventure-mode-design.md): saved files, jobs, and the same channels for window and phone ----
  const adventureStore = new AdventureStore({
    dir: join(userData, 'adventures'),
    trash: (path) => shell.trashItem(path),
    emit: (change) => {
      broadcast('adventure:changed', change)
      phone.emit('adventure:changed', change)
    }
  })
  const adventureRuns = new AdventureRuns({
    store: adventureStore,
    chat: chatAdapter(chat, chatTap),
    models: modelsAdapter(catalog, instances),
    gate: modelGate,
    busyReason,
    emit: (job) => {
      broadcast('adventure:job', job)
      phone.emit('adventure:job', job)
    }
  })
  const adventureDeps = { store: adventureStore, runs: adventureRuns, novel }
  const adventureTable = adventureHandlers(adventureDeps)
  const adventureDesktopTable = adventureDesktopHandlers({
    ...adventureDeps,
    chooseSavePath: async (defaultName) => {
      const r = await dialog.showSaveDialog(mainWindow!, {
        title: 'Export the story',
        defaultPath: join(app.getPath('documents'), defaultName),
        filters: [{ name: 'Markdown', extensions: ['md'] }]
      })
      return r.canceled || !r.filePath ? null : r.filePath
    },
    writeText: (path, text) => writeTextAtomic(path, text)
  })
  for (const channel of Object.keys({ ...adventureTable, ...adventureDesktopTable }) as (keyof typeof adventureTable | keyof typeof adventureDesktopTable)[]) {
    const fn = channel in adventureTable ? adventureTable[channel as keyof typeof adventureTable] : adventureDesktopTable[channel as keyof typeof adventureDesktopTable]
    handle(channel, ((...args: unknown[]) => fn(WINDOW_OWNER, ...args)) as never)
  }
  // The phone may call PHONE_ADVENTURE_CHANNELS only (no deletes, no save dialog); adventure:create's worldFolder is
  // checked in the phone dispatcher (phone/registry.ts).
  for (const channel of PHONE_ADVENTURE_CHANNELS) {
    phone.handlers.register(channel, ((owner: string, ...args: unknown[]) => adventureTable[channel](owner, ...args)) as never)
  }
  void phone.access.start()

  handle('presets:list', () => settings.listPresets())
  handle('presets:save', (p) => settings.savePreset(p))
  handle('presets:delete', (id) => settings.deletePreset(id))

  // ---- Lifecycle ----
  /** Answered once per quit: rentals keep costing money after the app is gone. False = do not quit. */
  let rentalsSettled = false
  let asking: Promise<boolean> | null = null
  const settleRentals = (): Promise<boolean> => {
    if (rentalsSettled) return Promise.resolve(true)
    asking ??= (async () => {
      const { running } = remote.rentals.quitSummary()
      if (!running.length) return true
      const lines = running.map((r) => `${r.label}: ${r.dollarsPerHour > 0 ? `$${r.dollarsPerHour.toFixed(2)} per hour` : 'price not reported'}. ${r.stopsItself ?? 'It will not stop by itself.'}`)
      const opts = {
        type: 'warning' as const,
        title: 'RulZ LLM Suite',
        message: running.length === 1 ? 'A vast.ai rental is still running.' : `${running.length} vast.ai rentals are still running.`,
        detail: `${lines.join('\n')}\n\nStopping ends a batch run or setup on it. A stopped rental still bills its disk.`,
        buttons: ['Stop and quit', 'Leave running and quit', 'Cancel'],
        defaultId: 2,
        cancelId: 2,
        noLink: true
      }
      const { response } = mainWindow && !mainWindow.isDestroyed() ? await dialog.showMessageBox(mainWindow, opts) : await dialog.showMessageBox(opts)
      if (response === 2) return false
      if (response === 1) return true
      const errors = await remote.rentals.stopAllForQuit()
      if (!errors.length) return true
      const again = await dialog.showMessageBox({ type: 'error', title: 'RulZ LLM Suite', message: 'Not every rental was stopped.', detail: errors.join('\n'), buttons: ['Quit anyway', 'Cancel'], defaultId: 1, cancelId: 1, noLink: true })
      return again.response === 0
    })()
      .then((ok) => {
        rentalsSettled = ok
        // Cancelled: the app goes on, and closing to the tray has to work again.
        if (!ok) quitting = false
        return ok
      })
      .finally(() => {
        asking = null
      })
    return asking
  }
  const rentalsRunning = (): boolean => !rentalsSettled && !sessionEnding && remote.rentals.quitSummary().running.length > 0
  let sessionEnding = false
  let poolDisposed = false
  mainWindow = createWindow(settings.get().theme)
  mainWindow.on('minimize', () => {
    if (!settings.get().minimizeToTray) return
    ensureTray()
    mainWindow?.hide()
  })
  mainWindow.on('close', (e) => {
    if (!quitting && settings.get().closeToTray) {
      e.preventDefault()
      ensureTray()
      mainWindow?.hide()
      return
    }
    if (rentalsRunning()) {
      e.preventDefault()
      void settleRentals().then((ok) => {
        if (ok) mainWindow?.close()
      })
      return
    }
    // Really closing: let the Writing page save or journal its open text first, then close for real.
    if (novelFlushed || !novel.hasOpenProject()) return
    e.preventDefault()
    void flushNovel().finally(() => {
      novelFlushed = true
      mainWindow?.close()
    })
  })
  // Windows logoff/shutdown must not be blocked by close-to-tray. The system gives little time, so the
  // Writing page is asked to flush at once and waits at most a second. The real backstop is the journal:
  // the Writing page writes every dirty buffer through novel:journal about 300 ms after each edit, so at
  // most that much typing can be missing after a forced shutdown, and it is offered back on the next open.
  mainWindow.on('session-end', () => {
    quitting = true
    sessionEnding = true
    flushTimeoutMs = 1000
    void flushNovel()
  })
  mainWindow.on('show', destroyTray)
  mainWindow.on('focus', () => hardware.poke())
  hardware.start()
  void remote.load().catch((err) => console.error('Could not read the vast.ai state:', remote.scrub(err)))
  void library.scan()
  void engine.getInfo()
  if (settings.get().apiServer.autoStart) apiServer.start().catch(() => undefined)

  app.on('second-instance', showMainWindow)
  app.on('window-all-closed', () => app.quit())
  app.on('before-quit', (e) => {
    quitting = true
    if (rentalsRunning()) {
      e.preventDefault()
      void settleRentals().then((ok) => {
        if (ok) app.quit()
      })
      return
    }
    // The batch run notes which tasks it did not finish.
    if (pool.busy && !poolDisposed) {
      e.preventDefault()
      // Bounded: a chat or run that cannot be saved must not keep the app from closing.
      const saved = Promise.race([pool.dispose().then(() => pool.whenIdle()), new Promise((r) => setTimeout(r, 8000))])
      void saved.catch(() => undefined).finally(() => {
        poolDisposed = true
        app.quit()
      })
      return
    }
    // app.quit() fires this before the window's close event. Hold the quit until the Writing page has
    // saved or journalled its text, then quit again; the second pass tears everything down.
    if (!novelFlushed && novel.hasOpenProject()) {
      e.preventDefault()
      void flushNovel().finally(() => {
        novelFlushed = true
        app.quit()
      })
      return
    }
    destroyTray()
    novel.dispose()
    void phone.access.dispose()
    hardware.stop()
    library.dispose()
    // The benchmark first: it sets its stop flag and starts unloading before the servers are killed.
    void bench.stop()
    instances.killAllSync()
    remote.rentals.dispose()
    void mcp.dispose()
    void apiServer.stop()
  })
}

if (isPrimaryInstance) void main()
