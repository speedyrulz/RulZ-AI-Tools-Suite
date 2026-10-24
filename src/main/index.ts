import { join } from 'node:path'
import { app, BrowserWindow, dialog, shell } from 'electron'
import { META_TOOL_DEFS, toolCatalogNote } from '@shared/tools'
import type { LoadConfig, ModelEntry } from '@shared/types'
import { ChatService } from './chat/service'
import { ConversationStore } from './chat/conversations'
import { EngineService } from './engine/engine'
import { HardwareMonitor } from './engine/hardware'
import { McpManager } from './mcp/manager'
import { InstanceManager } from './engine/instances'
import { broadcast, handle } from './ipc'
import { estimateMemory } from './models/estimate'
import { ModelLibrary } from './models/library'
import { ApiServer } from './server/apiServer'
import { SettingsStore } from './settings/store'

const MiB = 1024 * 1024
const isDev = !app.isPackaged && !!process.env.ELECTRON_RENDERER_URL

if (!app.requestSingleInstanceLock()) app.quit()

let mainWindow: BrowserWindow | null = null

function createWindow(): BrowserWindow {
  const win = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1024,
    minHeight: 640,
    show: false,
    title: 'RulZ LLM Suite',
    backgroundColor: '#000000',
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

  if (isDev) void win.loadURL(process.env.ELECTRON_RENDERER_URL!)
  else void win.loadFile(join(__dirname, '../renderer/index.html'))
  return win
}

async function main(): Promise<void> {
  await app.whenReady()
  app.setAppUserModelId('com.rulz.llmsuite')
  const userData = app.getPath('userData')

  const settings = new SettingsStore(userData)
  await settings.init()

  const engine = new EngineService(() => settings.get().enginePath)
  const hardware = new HardwareMonitor((s) => broadcast('hardware:update', s))
  const library = new ModelLibrary(
    join(userData, 'gguf-cache.json'),
    () => settings.get().modelsDir,
    (models) => broadcast('models:changed', models)
  )

  const freeMemory = (cfg: LoadConfig): { vramBytes: number | null; ramBytes: number } => {
    const snap = hardware.snapshot()
    const gpus = cfg.splitMode === 'single' ? snap.gpus.filter((g) => g.index === cfg.mainGpu) : snap.gpus
    let vram = gpus.length ? gpus.reduce((sum, g) => sum + (g.memTotalMiB - g.memUsedMiB) * MiB, 0) : null
    // Models that will be unloaded first give their memory back.
    if (vram !== null && settings.get().unloadPreviousOnLoad) {
      vram += instances.list().reduce((sum, i) => sum + (i.estimate?.gpuBytes ?? 0), 0)
      vram = Math.min(vram, gpus.reduce((sum, g) => sum + g.memTotalMiB * MiB, 0))
    }
    return { vramBytes: vram, ramBytes: snap.ramTotalBytes - snap.ramUsedBytes }
  }
  const estimate = (model: ModelEntry, cfg: LoadConfig) => estimateMemory(model, cfg, freeMemory(cfg))

  const instances = new InstanceManager({
    engine,
    getModel: (id) => library.get(id),
    getDefaults: (id) => settings.getModelDefaults(id).config,
    unloadPreviousOnLoad: () => settings.get().unloadPreviousOnLoad,
    estimate,
    pidFile: join(userData, 'running-pids.json'),
    onChange: (list) => broadcast('instances:changed', list),
    onError: (e) => broadcast('instances:error', e)
  })
  await instances.cleanupOrphans()

  const mcpConfigPath = join(userData, 'mcp.json')
  const mcp = new McpManager(mcpConfigPath, (s) => broadcast('mcp:changed', s))
  void mcp.init()

  const apiServer = new ApiServer({
    instances,
    getModels: () => library.list(),
    getSettings: () => settings.get().apiServer,
    onStatus: (s) => broadcast('server:changed', s),
    onRequest: (e) => broadcast('server:request', e)
  })

  const chat = new ChatService(instances, (e) => broadcast('chat:event', e), (req) => {
    const disabled = new Set(settings.get().disabledTools)
    if (req.mode === 'all') return { defs: mcp.toolDefinitions(disabled) }
    // On demand: the three loader tools, the loaded tools' full descriptions, and a name catalog.
    const available = mcp.availableTools(disabled)
    const active = req.active.filter((n) => available.some((t) => t.qualifiedName === n))
    return {
      defs: [...META_TOOL_DEFS, ...mcp.toolDefinitions(disabled, new Set(active))],
      note: available.length ? toolCatalogNote(available, active) : undefined
    }
  })
  const conversations = new ConversationStore(join(userData, 'conversations'))

  // ---- IPC ----
  handle('app:info', () => ({ version: app.getVersion(), userData, platform: process.platform }))
  handle('settings:get', () => settings.get())
  handle('settings:update', async (patch) => {
    const before = settings.get()
    const next = await settings.update(patch)
    if (next.modelsDir !== before.modelsDir) void library.scan()
    if (next.enginePath !== before.enginePath) void engine.getInfo(true)
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

  handle('models:list', () => library.list())
  handle('models:rescan', () => library.scan())
  handle('models:trash', async (modelId) => {
    const model = library.get(modelId)
    if (!model) throw new Error('Model not found')
    if (instances.list().some((i) => i.modelId === modelId)) throw new Error('Eject the model before deleting it.')
    for (const f of model.files) await shell.trashItem(f)
    await settings.setModelDefaults(modelId, null)
    await library.scan()
  })
  handle('models:getDefaults', (modelId) => settings.getModelDefaults(modelId))
  handle('models:setDefaults', (modelId, config) => settings.setModelDefaults(modelId, config))
  handle('models:estimate', (modelId, config) => {
    const model = library.get(modelId)
    if (!model) throw new Error('Model not found')
    return estimate(model, config)
  })

  handle('instances:list', () => instances.list())
  handle('instances:load', (modelId, config) => instances.load(modelId, config))
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
    void chat.start(req)
  })
  handle('chat:abort', (id) => chat.abort(id))
  handle('chat:countTokens', (modelId, messages, tools) => chat.countTokens(modelId, messages, tools))

  handle('conversations:list', () => conversations.list())
  handle('conversations:get', (id) => conversations.get(id))
  handle('conversations:save', (c) => conversations.save(c))
  handle('conversations:delete', (id) => conversations.delete(id))
  handle('folders:list', () => conversations.listFolders())
  handle('folders:save', (f) => conversations.saveFolders(f))

  handle('mcp:getConfig', () => ({ text: mcp.getConfigText(), path: mcpConfigPath }))
  handle('mcp:saveConfig', (text) => mcp.saveConfig(text))
  handle('mcp:status', () => mcp.status())
  handle('mcp:restart', (name) => mcp.restart(name))
  handle('mcp:callTool', (name, args) => mcp.callTool({ name, arguments: args }))

  handle('server:status', () => apiServer.getStatus())
  handle('server:start', () => apiServer.start())
  handle('server:stop', () => apiServer.stop())
  handle('server:log', () => apiServer.getLog())
  handle('server:clearLog', () => apiServer.clearLog())

  handle('presets:list', () => settings.listPresets())
  handle('presets:save', (p) => settings.savePreset(p))
  handle('presets:delete', (id) => settings.deletePreset(id))

  // ---- Lifecycle ----
  mainWindow = createWindow()
  hardware.start()
  void library.scan()
  void engine.getInfo()
  if (settings.get().apiServer.autoStart) apiServer.start().catch(() => undefined)

  app.on('second-instance', () => {
    if (!mainWindow) return
    if (mainWindow.isMinimized()) mainWindow.restore()
    mainWindow.focus()
  })
  app.on('window-all-closed', () => app.quit())
  app.on('before-quit', () => {
    hardware.stop()
    library.dispose()
    instances.killAllSync()
    void mcp.dispose()
    void apiServer.stop()
  })
}

void main()
