import { randomUUID } from 'node:crypto'
import { stat } from 'node:fs/promises'
import http, { type IncomingMessage, type ServerResponse } from 'node:http'
import { isAbsolute } from 'node:path'
import type { AdapterEntry, ApiReSession, ReOverview, RePracticeInfo, ApiRequestLog, ApiServerSettings, ApiServerStatus, GpuStat, LoadConfig, LoadedInstance, MemoryEstimate, ModelEntry } from '@shared/types'
import { isCloudModelId, isLocalInstance } from '@shared/cloud'
import { fileName } from '@shared/draft'
import { modeLoad, modePromptExtra, reloadFor, type AppMode, type ModelModeDefaults } from '@shared/modes'
import { API_RE_ID, RE_OUTCOMES, RE_SESSION_HEADER } from '@shared/re'
import { samplingRequestFields, type RecordedRound } from '../chat/service'
import type { InstanceManager } from '../engine/instances'
import {
  adapterChoiceList,
  adaptersFromRemote,
  applyRemoteLoadPatch,
  authorized,
  draftChoices,
  draftFileFromRemote,
  errorBody,
  hoistSystemMessages,
  overrideSampling,
  parseRemoteLoadPatch,
  pickModel,
  publicModelId,
  remoteLoadView,
  resolveModel,
  type RemoteLoadPatch,
  SAMPLING_PATHS,
  UsageReader
} from './helpers'
import { ChatReplyCapture } from './reCapture'

const MAX_BODY_BYTES = 64 * 1024 * 1024
/** How much of a too-large body is read and thrown away so the client can read the 413. */
const DRAIN_LIMIT_BYTES = 1024 * 1024 * 1024
const MAX_LOG = 300
const IDLE_CHECK_MS = 15_000

/** Paths proxied to a model's llama-server as-is. Everything else under /v1 is rejected. */
const PROXY_POST = new Set([
  '/v1/chat/completions',
  '/v1/completions',
  '/v1/embeddings',
  '/v1/responses',
  '/v1/messages',
  '/v1/messages/count_tokens',
  '/v1/chat/completions/input_tokens',
  '/v1/responses/input_tokens',
  '/completion',
  '/completions',
  '/embedding',
  '/embeddings',
  '/tokenize',
  '/detokenize',
  '/apply-template',
  '/infill',
  '/reranking',
  '/rerank'
])
const PROXY_GET = new Set(['/props', '/slots', '/metrics', '/lora-adapters'])
/** GET or POST /v1/models/{id}/load-config reads or changes a model's saved load settings. */
const LOAD_CONFIG_SUFFIX = '/load-config'
/** POST /v1/models/{id}/unload stops every running instance of a model. */
const UNLOAD_SUFFIX = '/unload'
/** GET or POST /v1/re/sessions/{id} reads or describes and grades an RE session another app runs. */
const RE_SESSIONS_PREFIX = '/v1/re/sessions/'
/** GET /v1/re/overview: every RE session, the practice tasks and the prompt templates. */
const RE_OVERVIEW = '/v1/re/overview'
/** POST /v1/re/practice/{task}/prepare copies a task's binary into a work folder; .../grade checks an answer. */
const RE_PRACTICE_PREFIX = '/v1/re/practice/'
/** POST /v1/re/kit saves the shared RE training kit folder. */
const RE_KIT = '/v1/re/kit'
const PRACTICE_ID = /^[\w-]{1,80}$/
/** The modes other apps may ask about and load with (header X-RulZ-Mode, the mode routes). */
const API_MODES = ['coding', 're'] as const
type ApiMode = (typeof API_MODES)[number]
const isApiMode = (m: unknown): m is ApiMode => typeof m === 'string' && (API_MODES as readonly string[]).includes(m)
/** A request with X-RulZ-Mode leaves a lease on its model until this long after its end (see leasedUntil). */
const LEASE_MS = 2 * 60_000
/** Paths whose request loads a model that is not loaded with the settings of the X-RulZ-Mode header. */
const MODE_LOAD_PATHS = new Set(['/v1/chat/completions', '/props'])

/** The new routes under /v1/models/{id}/: modes, reload and modes/{mode}/load-config. */
type ModeRoute =
  | { kind: 'modes'; id: string }
  | { kind: 'reload'; id: string }
  | { kind: 'loadConfig'; id: string; mode: string }

function parseModeRoute(method: string, path: string): ModeRoute | null {
  const prefix = '/v1/models/'
  if (!path.startsWith(prefix)) return null
  const rest = path.slice(prefix.length)
  if (method === 'GET' && rest.length > '/modes'.length && rest.endsWith('/modes')) return { kind: 'modes', id: rest.slice(0, -'/modes'.length) }
  if (method === 'POST' && rest.length > '/reload'.length && rest.endsWith('/reload')) return { kind: 'reload', id: rest.slice(0, -'/reload'.length) }
  if (method === 'GET' || method === 'POST') {
    const m = /^(.+)\/modes\/([^/]+)\/load-config$/.exec(rest)
    if (m) return { kind: 'loadConfig', id: m[1], mode: m[2] }
  }
  return null
}

/** What asking for a mode reload came to. 'matches' also covers a model that is not loaded: nothing happens then. */
export type ModeReloadResult = { status: 'not-loaded' } | { status: 'matches' } | { status: 'started' } | { status: 'blocked'; why: string }

/** A response header value that Node accepts: plain ASCII as it is, anything else percent-encoded. */
const headerSafe = (v: string): string => (/^[\x20-\x7e]*$/.test(v) ? v : encodeURIComponent(v))

export interface ApiServerDeps {
  instances: InstanceManager
  getModels: () => ModelEntry[]
  /** LoRA adapter files in the models folder. */
  getAdapters?: () => AdapterEntry[]
  getSettings: () => ApiServerSettings
  /** The model's saved load defaults (what a load without explicit settings uses). */
  getLoadDefaults: (modelId: string) => { config: LoadConfig; custom: boolean }
  setLoadDefaults: (modelId: string, config: LoadConfig) => Promise<void>
  estimate: (model: ModelEntry, config: LoadConfig) => MemoryEstimate | null
  gpus: () => GpuStat[]
  onStatus: (s: ApiServerStatus) => void
  onRequest: (entry: ApiRequestLog) => void
  /** Clock for leases; tests. */
  now?: () => number
  /** Whether other apps may use an OpenRouter model now (ticked for other apps, and a key is saved); asked on every request. Without it none may. */
  cloudAllowed?: (modelId: string) => boolean
  /**
   * Per-mode settings and what a reload needs (docs/app-modes-design.md). Without it the mode routes answer 404 and the
   * X-RulZ-Mode header only leaves a lease.
   */
  modes?: {
    /** One model's settings per mode. */
    get: (modelId: string) => ModelModeDefaults
    /** Saves or removes (null) one mode's load settings only; its sampling and prompt stay. */
    setLoad: (modelId: string, mode: AppMode, load: LoadConfig | null | ((current: LoadConfig) => LoadConfig)) => Promise<void>
    /** Tells the window the mode settings changed. */
    changed?: () => void
    /** Why the instance cannot be reloaded right now, or null. `holding`: the gate's holder that asks, for which the gate is no reason. */
    reloadBlock?: (instanceId: string, holding?: string) => string | null
    /** Takes the model gate; the release function, or the sentence saying who holds it. */
    takeGate?: (holder: string) => (() => void) | string
  }
  /** RE sessions other apps run through the server (ids as the app sent them). */
  re?: {
    enabled: () => boolean
    record: (id: string, round: RecordedRound) => void
    session: (id: string) => Promise<ApiReSession | null>
    update: (id: string, patch: Partial<ApiReSession>) => Promise<ApiReSession>
    /** `model`: count practice for that model only (a library id, or the text the client sent). */
    overview: (model?: string) => Promise<ReOverview>
    preparePractice: (taskId: string) => Promise<{ target: string; prompt: string; family: string; tier: string }>
    gradePractice: (taskId: string, target: string, text: string) => Promise<{ answer: string | null; passed: boolean; expected: string }>
    /** Saves the kit folder and returns that kit's practice list (error set when the kit has no tasks). */
    setKit: (kitDir: string) => Promise<RePracticeInfo>
  }
}

/**
 * OpenAI- and Anthropic-compatible HTTP server in front of the loaded models. Requests name a
 * model; if it isn't loaded the server loads it with its saved defaults, and models it loaded
 * are unloaded again after sitting idle.
 */
export class ApiServer {
  private server: http.Server | null = null
  private status: ApiServerStatus = { running: false, host: '127.0.0.1', port: 1234, requestCount: 0, jitInstances: [] }
  private log: ApiRequestLog[] = []
  /** Instances the server loaded itself (and so may unload when idle) → when it last used them. */
  private jit = new Map<string, number>()
  private idleTimer: NodeJS.Timeout | null = null
  private loading = new Map<string, Promise<LoadedInstance>>()
  /** Model id → until when a request with X-RulZ-Mode keeps its model leased. */
  private leases = new Map<string, number>()

  constructor(private readonly deps: ApiServerDeps) {}

  getStatus(): ApiServerStatus {
    return { ...this.status, jitInstances: [...this.jit].map(([instanceId, lastUsedAt]) => ({ instanceId, lastUsedAt })) }
  }

  getLog(): ApiRequestLog[] {
    return this.log.slice()
  }

  clearLog(): void {
    this.log = []
  }

  async start(): Promise<ApiServerStatus> {
    if (this.server) return this.getStatus()
    const s = this.deps.getSettings()
    const host = s.bind === 'lan' ? '0.0.0.0' : '127.0.0.1'
    const server = http.createServer((req, res) => {
      // Last resort: a request that fails outside the handler's own error handling still gets an answer.
      this.handle(req, res).catch((err: unknown) => {
        console.error('API request failed:', err)
        if (!res.headersSent) res.writeHead(500, { 'Content-Type': 'application/json' }).end(errorBody(String(err), 'server_error'))
        else res.destroy()
      })
    })
    server.keepAliveTimeout = 65_000
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject)
      server.listen(s.port, host, () => {
        server.off('error', reject)
        resolve()
      })
    }).catch((err: NodeJS.ErrnoException) => {
      const why =
        err.code === 'EADDRINUSE'
          ? `Port ${s.port} is already in use (LM Studio uses 1234 by default). Choose another port.`
          : err.message
      this.status = { ...this.status, running: false, host, port: s.port, error: why }
      this.emit()
      throw new Error(why)
    })
    this.server = server
    server.on('error', (err) => {
      this.status = { ...this.status, running: false, error: err.message }
      this.emit()
    })
    this.status = { running: true, host, port: s.port, startedAt: Date.now(), requestCount: 0, jitInstances: [] }
    this.idleTimer = setInterval(() => void this.unloadIdle(), IDLE_CHECK_MS)
    this.emit()
    return this.getStatus()
  }

  async stop(): Promise<ApiServerStatus> {
    if (this.idleTimer) clearInterval(this.idleTimer)
    this.idleTimer = null
    const server = this.server
    this.server = null
    if (server) {
      server.closeAllConnections?.()
      await new Promise<void>((resolve) => server.close(() => resolve()))
    }
    this.status = { ...this.status, running: false, error: undefined, startedAt: undefined }
    this.emit()
    return this.getStatus()
  }

  /** Called when the user ejects a model from the UI so it is no longer tracked as server-loaded. */
  forget(instanceId: string): void {
    if (this.jit.delete(instanceId)) this.emit()
  }

  // ---------- request handling ----------

  private async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const settings = this.deps.getSettings()
    let url: URL
    try {
      url = new URL(req.url ?? '/', 'http://localhost')
    } catch {
      // A target like "//[" isn't a URL at all.
      res.writeHead(400, { 'Content-Type': 'application/json' }).end(errorBody('Invalid request URL.', 'invalid_request_error'))
      return
    }
    const path = url.pathname.replace(/\/+$/, '') || '/'
    const method = (req.method ?? 'GET').toUpperCase()

    if (settings.cors) {
      res.setHeader('Access-Control-Allow-Origin', '*')
      res.setHeader('Access-Control-Allow-Headers', `Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta, ${RE_SESSION_HEADER}`)
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    }
    if (method === 'OPTIONS') {
      res.writeHead(204).end()
      return
    }

    const entry: ApiRequestLog = { id: randomUUID(), at: Date.now(), method, path, phase: 'running' }
    this.status.requestCount++
    // The first outcome sticks: an upstream error that follows a client disconnect stays a 499.
    const finish = (status: number, error?: string): void => {
      if (entry.phase === 'done') return
      entry.status = status
      entry.durationMs = Date.now() - entry.at
      entry.error = error
      entry.phase = 'done'
      this.record(entry)
    }
    const fail = (status: number, message: string, type?: string, code?: string): void => {
      res.writeHead(status, { 'Content-Type': 'application/json' }).end(errorBody(message, type, code))
      finish(status, message)
    }
    /** The body as a JSON object; on anything else the request has been answered and this gives null. */
    const jsonBody = async (): Promise<Record<string, unknown> | null> => {
      // A body over the limit throws BodyTooLarge, answered with 413 below.
      const text = (await readBody(req)).toString('utf8')
      let parsed: unknown
      try {
        parsed = JSON.parse(text || '{}')
      } catch {
        fail(400, 'The request body is not valid JSON.')
        return null
      }
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        fail(400, 'The request body must be a JSON object.')
        return null
      }
      return parsed as Record<string, unknown>
    }

    if (!authorized(settings.apiKey, req.headers as Record<string, string>)) {
      return fail(401, 'Invalid API key.', 'authentication_error')
    }
    // Reverse-engineering calls copy, run and grade programs and change the kit folder: only other apps may make
    // them, never a web page (browsers send Origin with every cross-site POST, form posts included).
    if (method === 'POST' && path.startsWith('/v1/re/') && req.headers.origin) {
      return fail(403, 'Reverse-engineering calls cannot come from a web page.', 'permission_error')
    }

    // The mode routes carry system prompts and change load settings: only other apps may call them, never a web page
    // (CORS may be on), and a POST must be JSON, which a plain form cannot send.
    const modeRoute = this.deps.modes ? parseModeRoute(method, path) : null
    // A page that points its own name at this computer (DNS rebinding) sends no Origin with a GET; its Host gives it away.
    if (modeRoute && (req.headers.origin !== undefined || !ownHost(req.headers.host))) return fail(403, 'Mode calls cannot come from a web page.', 'permission_error')
    if (modeRoute && method === 'POST' && !/^application\/json\b/i.test(String(req.headers['content-type'] ?? ''))) {
      return fail(415, 'Send the request as application/json.', 'invalid_request_error')
    }

    // Watched from the start, so a client that leaves while its model loads is noticed too.
    let clientGone = false
    let onClientGone: (() => void) | null = null
    res.on('close', () => {
      if (res.writableFinished) return
      clientGone = true
      finish(499, 'Client disconnected')
      onClientGone?.()
    })

    try {
      if (method === 'GET' && (path === '/' || path === '/health')) {
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify({ status: 'ok', app: 'RulZ LLM Suite' }))
        return finish(200)
      }
      if (method === 'GET' && (path === '/v1/models' || path === '/models' || path === '/api/v0/models')) {
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(this.listModels()))
        return finish(200)
      }
      if (modeRoute) {
        const modes = this.deps.modes!
        const rawMode = modeRoute.kind === 'loadConfig' ? decodeURIComponent(modeRoute.mode) : null
        if (rawMode !== null && !isApiMode(rawMode)) return fail(404, 'Unknown mode: only coding and re are available.', 'not_found_error')
        const model = this.resolveApi(decodeURIComponent(modeRoute.id))
        if (!model || (model.remote && !this.deps.instances.readyFor(model.id))) return fail(404, 'Model not found.', 'not_found_error')
        if (modeRoute.kind === 'modes') {
          res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(this.modesBody(model)))
          return finish(200)
        }
        if (modeRoute.kind === 'reload') {
          const json = await jsonBody()
          if (!json) return
          if (!isApiMode(json.mode)) return fail(400, 'mode must be "coding" or "re".', 'invalid_request_error')
          if (model.remote) return fail(409, remoteRefusal(model, 'reload it'), 'invalid_request_error')
          if (model.cloud) return fail(409, cloudRefusal(model, 'reload it'), 'invalid_request_error')
          const r = this.requestModeReload(model, json.mode)
          if (r.status === 'blocked') return fail(409, r.why, 'invalid_request_error')
          res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(this.modesBody(model, r.status === 'started')))
          return finish(200)
        }
        // GET or POST /v1/models/{id}/modes/{mode}/load-config
        const mode = rawMode as ApiMode
        // Changing the settings could reload the model, and the API never loads a rental's model.
        if (model.remote && method === 'POST') return fail(409, remoteRefusal(model, 'change its load settings'), 'invalid_request_error')
        if (model.cloud && method === 'POST') return fail(409, cloudRefusal(model, 'change its load settings'), 'invalid_request_error')
        let preview: LoadConfig | undefined
        let blocked: string | undefined
        if (method === 'POST') {
          const json = await jsonBody()
          if (!json) return
          const base = this.deps.getLoadDefaults(model.id).config
          let write: LoadConfig | null | ((current: LoadConfig) => LoadConfig) = null
          if (json.config === null) {
            // {"config": null} removes the mode's own load settings.
            if (json.dry_run === true) preview = base
          } else {
            const rawConfig = json.adapters !== undefined ? { ...((json.config as object | undefined) ?? {}), adapters: json.adapters } : (json.config ?? {})
            if (rawConfig && typeof rawConfig === 'object' && (rawConfig as { sampling?: unknown }).sampling !== undefined) {
              return fail(400, 'Invalid load setting sampling: the API server sampling belongs to all modes, set it on /load-config', 'invalid_request_error')
            }
            const checked = this.checkLoadPatch(rawConfig, model)
            if (!checked.ok) return fail(400, checked.error, 'invalid_request_error')
            // Merged when its turn to be saved comes, so a change saved just before is not undone.
            write = (current) => applyRemoteLoadPatch(current, checked.patch)
            if (json.dry_run === true) preview = write(modeLoad(base, modes.get(model.id), mode))
          }
          if (json.dry_run !== true) {
            await modes.setLoad(model.id, mode, write)
            modes.changed?.()
            // A reload is only started when asked for; one that is refused leaves the saved settings in place.
            if (json.reload === true) {
              const r = this.requestModeReload(model, mode)
              if (r.status === 'blocked') blocked = r.why
            }
          }
        }
        const entry = this.loadConfigEntry(model, preview, mode) as Record<string, unknown>
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(blocked ? { ...entry, reload_blocked: blocked } : entry))
        return finish(200)
      }
      if (path.startsWith('/v1/models/') && path.endsWith(LOAD_CONFIG_SUFFIX) && (method === 'GET' || method === 'POST')) {
        const model = this.resolveApi(decodeURIComponent(path.slice('/v1/models/'.length, -LOAD_CONFIG_SUFFIX.length)))
        if (!model || (model.remote && !this.deps.instances.readyFor(model.id))) return fail(404, 'Model not found.', 'not_found_error')
        // Changing the settings would reload the model, and the API never loads a rental's model.
        if (model.remote && method === 'POST') return fail(409, remoteRefusal(model, 'change its load settings'), 'invalid_request_error')
        if (model.cloud && method === 'POST') return fail(409, cloudRefusal(model, 'change its load settings'), 'invalid_request_error')
        let preview: LoadConfig | undefined
        if (method === 'POST') {
          const json = await jsonBody()
          if (!json) return
          // `adapters` may sit beside `config` as well as in it.
          const raw = json.adapters !== undefined ? { ...((json.config as object | undefined) ?? {}), adapters: json.adapters } : (json.config ?? {})
          const checked = this.checkLoadPatch(raw, model)
          if (!checked.ok) return fail(400, checked.error, 'invalid_request_error')
          const merged = applyRemoteLoadPatch(this.deps.getLoadDefaults(model.id).config, checked.patch)
          // dry_run only estimates memory for the changed settings, so a form can show it while editing.
          if (json.dry_run === true) preview = merged
          else {
            await this.deps.setLoadDefaults(model.id, merged)
            if (json.reload !== false) this.reload(model.id)
          }
        }
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(this.loadConfigEntry(model, preview)))
        return finish(200)
      }
      if (method === 'POST' && path.startsWith('/v1/models/') && path.endsWith(UNLOAD_SUFFIX)) {
        const model = this.resolveApi(decodeURIComponent(path.slice('/v1/models/'.length, -UNLOAD_SUFFIX.length)))
        if (!model) return fail(404, 'Model not found.', 'not_found_error')
        if (model.remote) return fail(409, remoteRefusal(model, 'unload it'), 'invalid_request_error')
        if (model.cloud) return fail(409, cloudRefusal(model, 'unload it'), 'invalid_request_error')
        const unloaded = await this.unloadModel(model.id)
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify({ ...(this.modelEntry(model) as object), unloaded }))
        return finish(200)
      }
      if (path === RE_OVERVIEW && method === 'GET') {
        if (!this.deps.re) return fail(404, `Unknown endpoint: ${method} ${path}`, 'not_found_error')
        // ?model={id}: the practice counts are that model's only (see buildOverview).
        const model = url.searchParams.get('model')?.trim().slice(0, 1024)
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(await this.deps.re.overview(model || undefined)))
        return finish(200)
      }
      if (path === RE_KIT && method === 'POST') {
        const re = this.deps.re
        if (!re) return fail(404, `Unknown endpoint: ${method} ${path}`, 'not_found_error')
        // Web pages are refused above; a JSON content type also can't be sent from a plain form.
        if (!/^application\/json\b/i.test(String(req.headers['content-type'] ?? ''))) {
          return fail(415, 'Send the kit folder as application/json.', 'invalid_request_error')
        }
        const json = await jsonBody()
        if (!json) return
        const kitDir = await parseKitDir(json)
        if (typeof kitDir !== 'string') return fail(400, kitDir.error, 'invalid_request_error')
        const out = await re.setKit(kitDir)
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(out))
        return finish(200)
      }
      if (path.startsWith(RE_PRACTICE_PREFIX) && method === 'POST') {
        const re = this.deps.re
        if (!re) return fail(404, `Unknown endpoint: ${method} ${path}`, 'not_found_error')
        const [rawId, action, ...rest] = path.slice(RE_PRACTICE_PREFIX.length).split('/')
        const taskId = decodeURIComponent(rawId ?? '')
        if (rest.length || (action !== 'prepare' && action !== 'grade')) return fail(404, `Unknown endpoint: ${method} ${path}`, 'not_found_error')
        if (!PRACTICE_ID.test(taskId)) return fail(400, 'Practice task ids are 1-80 letters, digits, _ or -.', 'invalid_request_error')
        const json = await jsonBody()
        if (!json) return
        let out: unknown
        try {
          if (action === 'prepare') out = await re.preparePractice(taskId)
          else {
            if (typeof json.target !== 'string' || typeof json.text !== 'string') return fail(400, 'target and text must be strings.', 'invalid_request_error')
            out = await re.gradePractice(taskId, json.target, json.text.slice(-20000))
          }
        } catch (e) {
          return fail(400, e instanceof Error ? e.message : String(e), 'invalid_request_error')
        }
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(out))
        return finish(200)
      }
      if (path.startsWith(RE_SESSIONS_PREFIX) && (method === 'GET' || method === 'POST')) {
        const re = this.deps.re
        if (!re) return fail(404, `Unknown endpoint: ${method} ${path}`, 'not_found_error')
        const id = decodeURIComponent(path.slice(RE_SESSIONS_PREFIX.length))
        if (!API_RE_ID.test(id)) return fail(400, 'Session ids are 1-56 letters, digits, _ or -.', 'invalid_request_error')
        let session: ApiReSession | null
        if (method === 'POST') {
          const json = await jsonBody()
          if (!json) return
          const patch = parseReSessionPatch(json)
          if (typeof patch === 'string') return fail(400, patch, 'invalid_request_error')
          session = re.enabled() || patch.outcome ? await re.update(id, patch) : await re.session(id)
        } else {
          session = await re.session(id)
          if (!session) return fail(404, 'No such RE session.', 'not_found_error')
        }
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify({ id, object: 're.session', recording: re.enabled(), ...session }))
        return finish(200)
      }
      if (method === 'GET' && path.startsWith('/v1/models/')) {
        const model = this.resolveApi(decodeURIComponent(path.slice('/v1/models/'.length)))
        if (!model || (model.remote && !this.deps.instances.readyFor(model.id))) return fail(404, 'Model not found.', 'not_found_error')
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(this.modelEntry(model)))
        return finish(200)
      }

      let body: Buffer | null = null
      // The body parsed once; written back to `body` only if something in it was changed.
      let json: Record<string, unknown> | null = null
      let rewritten = false
      let requestedModel: string | undefined
      let stream = false
      if (method === 'POST' && PROXY_POST.has(path)) {
        body = await readBody(req)
        let parsed: unknown
        try {
          parsed = JSON.parse(body.toString('utf8'))
        } catch {
          return fail(400, 'The request body is not valid JSON.')
        }
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return fail(400, 'The request body must be a JSON object.')
        json = parsed as Record<string, unknown>
        if (typeof json.model === 'string') requestedModel = json.model
        stream = json.stream === true
        if (path.startsWith('/v1/messages')) {
          const hoisted = hoistSystemMessages(json)
          if (hoisted) {
            json = hoisted
            rewritten = true
          }
        }
      } else if (method === 'GET' && PROXY_GET.has(path)) {
        requestedModel = url.searchParams.get('model') ?? undefined
      } else {
        return fail(404, `Unknown endpoint: ${method} ${path}`, 'not_found_error')
      }

      // One entry per model: a rental's model may be ready on several rentals.
      // (An OpenRouter instance is never in this list: the fallback and the single loaded model must not pick a paid model.)
      const ready = [...new Set(this.deps.instances.list().filter((i) => i.status === 'ready' && !i.cloud).map((i) => i.modelId))]
      // A rental's model counts as a model only while it is loaded there, so it can't take over a fuzzy name match.
      const picked = pickModel(this.visibleModels(), requestedModel, ready, settings)
      // Whoever names an OpenRouter model that is not available gets no other model in its place, whatever the fallback.
      if (requestedModel && isCloudModelId(requestedModel.trim()) && (!picked || picked.fallback)) {
        return fail(403, `${requestedModel.trim().slice(0, 200)} is not available to other apps. In RulZ LLM Suite, add it on the OpenRouter page, tick "Allow other apps" and save a key.`, 'permission_error', 'cloud_not_allowed')
      }
      if (!picked) {
        const asked = requestedModel ? this.resolveApi(requestedModel) : undefined
        if (asked?.remote) return fail(409, remoteRefusal(asked, 'load it'), 'invalid_request_error')
        const hint = requestedModel
          ? settings.unknownModelFallback
            ? `Model "${requestedModel}" was not found, and there is no single loaded model or default model to use instead.`
            : `Model "${requestedModel}" was not found.`
          : 'No "model" given and no model is loaded.'
        return fail(404, `${hint} GET /v1/models lists the available ids.`, 'not_found_error')
      }
      const model = picked.model
      entry.model = publicModelId(model, this.deps.getModels())
      if (picked.fallback) entry.requestedModel = requestedModel
      entry.stream = stream

      // X-RulZ-Mode: a model that is not loaded is loaded with that mode's settings; a loaded one is used as it is.
      const modeHeader = headerValue(req, 'x-rulz-mode')
      const requestMode = isApiMode(modeHeader) ? modeHeader : undefined
      // Sent more than once, the values arrive as one list: a single deny in it counts.
      const denyRental = (headerValue(req, 'x-rulz-rental') ?? '').toLowerCase().split(',').some((v) => v.trim() === 'deny')

      if (model.cloud) {
        // OpenRouter is paid and the text leaves this computer: only a request that no web page could have made, and
        // (when the server listens on the network) only with the API key set. Asked again on every request.
        if (!this.deps.cloudAllowed?.(model.id)) return fail(404, 'Model not found.', 'not_found_error')
        if (req.headers.origin !== undefined || !ownHost(req.headers.host)) return fail(403, 'OpenRouter models cannot be used from a web page.', 'permission_error')
        if (settings.bind === 'lan' && !settings.apiKey) return fail(403, 'OpenRouter models need an API key when the server listens on the network. Set one in the API server settings.', 'permission_error')
        if (method === 'POST' && !/^application\/json\b/i.test(String(req.headers['content-type'] ?? ''))) return fail(415, 'Send the request as application/json.', 'invalid_request_error')
        if (!CLOUD_PATHS.has(`${method} ${path}`)) return fail(400, `${model.name} is an OpenRouter model: only chat completions, /props, /tokenize and /apply-template work with it.`, 'invalid_request_error')
        if (denyRental) return fail(409, `${model.name} is an OpenRouter model, and this request says it must not use a rental or another computer.`, 'invalid_request_error')
      }

      let inst = this.deps.instances.readyFor(model.id)
      if (!inst && model.remote) return fail(409, remoteRefusal(model, 'load it'), 'invalid_request_error')
      if (!inst) {
        entry.phase = 'loading'
        this.record(entry)
        try {
          inst = await this.ensureLoaded(model.id, requestMode && MODE_LOAD_PATHS.has(path) && !model.cloud ? requestMode : undefined)
        } catch (err) {
          if (clientGone) return
          return fail(503, model.cloud ? (err as Error).message : `Could not load ${model.name}: ${(err as Error).message}`, 'server_error')
        }
        // Nobody is waiting for the answer any more; don't make the model generate it.
        if (clientGone) return
        entry.phase = 'running'
      }
      if (clientGone) return
      // Checked on the instance that answers, right before the request goes to it: the one a load just made, or one that
      // replaced the one found above, can be a rental's too (the fallback for unknown names may pick a rental's model).
      const answering = inst.instanceId
      inst = this.deps.instances.list().find((i) => i.instanceId === answering) ?? inst
      if (denyRental && inst.host) {
        return fail(409, `${model.name} runs on the rental ${inst.host.label}, and this request says it must not use a rental.`, 'invalid_request_error')
      }

      if (json && inst.config.sampling.override && SAMPLING_PATHS.has(path)) {
        json = overrideSampling(json, inst.config.sampling)
        rewritten = true
      }
      if (json && rewritten) body = Buffer.from(JSON.stringify(json), 'utf8')

      // An RE session another app runs: record the request and its reply, like an in-app round.
      let capture: ((reply: RecordedRound['response']) => void) | undefined
      const reId = req.headers[RE_SESSION_HEADER]
      const re = this.deps.re
      if (re && json && path === '/v1/chat/completions' && typeof reId === 'string' && API_RE_ID.test(reId) && re.enabled()) {
        const requestJson = json
        const at = Date.now()
        const model = entry.model
        capture = (response) =>
          re.record(reId, {
            requestId: entry.id,
            at,
            model,
            messages: Array.isArray(requestJson.messages) ? (requestJson.messages as RecordedRound['messages']) : [],
            tools: Array.isArray(requestJson.tools) ? requestJson.tools : [],
            trimmed: false,
            response
          })
      }

      // Held for the whole request, however long the stream runs, so idle-unload can't stop the model under it.
      const release = this.deps.instances.acquire(inst.instanceId, 'api', 'api request')
      if (this.jit.has(inst.instanceId)) this.jit.set(inst.instanceId, Date.now())
      // Another app's task is between two requests most of the time: the lease keeps its model from being reloaded then.
      if (requestMode && !inst.cloud) this.lease(inst.modelId)
      this.emit()
      try {
        if (inst.cloud) await this.proxyCloud(res, inst, method, path, body, entry, finish, (abort) => (onClientGone = abort), capture, entry.model ?? inst.modelName)
        else await this.proxy(req, res, inst, path + url.search, body, entry, finish, (abort) => (onClientGone = abort), capture, entry.model ?? inst.modelName)
      } finally {
        release()
        if (requestMode && !inst.cloud) this.lease(inst.modelId)
        this.touch(inst.instanceId)
      }
    } catch (err) {
      if (res.headersSent) finish(res.statusCode, (err as Error).message)
      else if (err instanceof BodyTooLarge) {
        // No "Connection: close": closing while the client still sends resets the connection and
        // the client never reads the 413. readBody discards the rest of the body instead.
        fail(413, err.message, 'invalid_request_error')
      } else if (err instanceof URIError) fail(400, 'The request path has a malformed %-escape.', 'invalid_request_error')
      else fail(500, (err as Error).message, 'server_error')
    }
  }

  /**
   * Answers a request for an OpenRouter model: the gateway is called in process (no port), and its Response is passed on
   * the way proxy() passes on a llama-server's, with the same usage log and RE capture. A client that leaves aborts the request.
   */
  private async proxyCloud(
    res: ServerResponse,
    inst: LoadedInstance,
    method: string,
    path: string,
    body: Buffer | null,
    entry: ApiRequestLog,
    finish: (status: number, error?: string) => void,
    onClientGone: (abort: () => void) => void,
    capture?: (reply: RecordedRound['response']) => void,
    publicId?: string
  ): Promise<void> {
    const ctrl = new AbortController()
    onClientGone(() => ctrl.abort())
    let upstream: Response
    try {
      upstream = await this.deps.instances.fetch(inst, path, { method, body: body ? body.toString('utf8') : undefined, signal: ctrl.signal }, 'api')
    } catch (err) {
      // Aborted: the client left, and its 499 is already logged.
      if (ctrl.signal.aborted) return
      const message = err instanceof Error ? err.message : String(err)
      if (!res.headersSent) res.writeHead(502, { 'Content-Type': 'application/json', 'x-rulz-cloud': '1' }).end(errorBody(message, 'server_error'))
      finish(502, message)
      return
    }
    const out: Record<string, string> = { 'x-rulz-cloud': '1' }
    for (const h of ['content-type', 'retry-after']) {
      const v = upstream.headers.get(h)
      if (v) out[h] = v
    }
    if (publicId) out['x-rulz-model'] = headerSafe(publicId)
    res.writeHead(upstream.status, out)
    const sse = /text\/event-stream/i.test(out['content-type'] ?? '')
    const usage = new UsageReader(sse)
    const reply = capture && upstream.status === 200 ? new ChatReplyCapture(sse) : null
    try {
      const reader = upstream.body?.getReader()
      for (;;) {
        const part = reader ? await reader.read() : { done: true as const, value: undefined }
        if (part.done) break
        const chunk = Buffer.from(part.value)
        // A client reading slower than the model writes would otherwise have the whole reply buffered here.
        if (!res.write(chunk) && !res.destroyed) {
          await new Promise<void>((resolve) => {
            const go = (): void => {
              res.off('drain', go)
              res.off('close', go)
              resolve()
            }
            res.on('drain', go)
            res.on('close', go)
          })
        }
        usage.push(chunk)
        reply?.push(chunk)
      }
    } catch (err) {
      if (ctrl.signal.aborted) return
      res.destroy()
      finish(502, err instanceof Error ? err.message : String(err))
      return
    }
    res.end()
    const r = reply?.result()
    if (r && capture) {
      try {
        capture(r)
      } catch (err) {
        console.error('Could not record an RE round:', err)
      }
    }
    Object.assign(entry, usage.end())
    finish(upstream.status, upstream.status >= 400 ? errorMessageFrom(usage.text()) : undefined)
  }

  private proxy(
    req: IncomingMessage,
    res: ServerResponse,
    inst: LoadedInstance,
    pathWithQuery: string,
    body: Buffer | null,
    entry: ApiRequestLog,
    finish: (status: number, error?: string) => void,
    /** Registers what to do if the client disconnects mid-request. */
    onClientGone: (abort: () => void) => void,
    /** Receives the reply of a successful chat completion (RE recording). */
    capture?: (reply: RecordedRound['response']) => void,
    /** The public id of the model that answers, sent back as X-RulZ-Model. */
    publicId?: string
  ): Promise<void> {
    return new Promise((resolve) => {
      const headers: Record<string, string> = { host: `127.0.0.1:${inst.port}` }
      for (const h of ['content-type', 'accept', 'anthropic-version', 'anthropic-beta']) {
        const v = req.headers[h]
        if (typeof v === 'string') headers[h] = v
      }
      if (body) headers['content-length'] = String(body.length)
      const upstream = http.request(
        { host: '127.0.0.1', port: inst.port, method: req.method, path: pathWithQuery, headers },
        (ures) => {
          const out: Record<string, string | string[]> = {}
          for (const [k, v] of Object.entries(ures.headers)) {
            if (v !== undefined && !['connection', 'keep-alive', 'transfer-encoding', 'access-control-allow-origin'].includes(k)) out[k] = v
          }
          // Which model answered, and whether that is a rented computer (the caller may not want its text to go there).
          if (publicId) out['x-rulz-model'] = headerSafe(publicId)
          if (inst.host) out['x-rulz-rental'] = '1'
          res.writeHead(ures.statusCode ?? 502, out)
          // What the model server sent decides how usage is read: an error to a streamed request is plain JSON.
          const sse = /text\/event-stream/i.test(String(ures.headers['content-type'] ?? ''))
          const usage = new UsageReader(sse)
          const reply = capture && ures.statusCode === 200 ? new ChatReplyCapture(sse) : null
          ures.on('data', (chunk: Buffer) => {
            // A client reading slower than the model writes (a big embeddings reply, say) would
            // otherwise have the whole reply buffered here.
            if (!res.write(chunk)) {
              ures.pause()
              res.once('drain', () => ures.resume())
            }
            usage.push(chunk)
            reply?.push(chunk)
          })
          ures.on('end', () => {
            res.end()
            const r = reply?.result()
            if (r && capture) {
              try {
                capture(r)
              } catch (err) {
                console.error('Could not record an RE round:', err)
              }
            }
            Object.assign(entry, usage.end())
            finish(ures.statusCode ?? 502, ures.statusCode && ures.statusCode >= 400 ? errorMessageFrom(usage.text()) : undefined)
            resolve()
          })
          ures.on('error', (err) => {
            res.destroy()
            finish(502, err.message)
            resolve()
          })
        }
      )
      upstream.on('error', (err) => {
        // A rental's tunnel can drop while the model stays loaded there; say so, so the client knows to try again.
        const message = inst.host
          ? `The connection to the rental ${inst.host.label} was lost (${err.message}). The model is still loaded there; try again in a moment.`
          : `Model server error: ${err.message}`
        if (!res.headersSent) res.writeHead(502, { 'Content-Type': 'application/json' }).end(errorBody(message, 'server_error'))
        else res.destroy()
        finish(502, inst.host ? message : err.message)
        resolve()
      })
      // A client that disconnects mid-stream stops generation on the model server too.
      onClientGone(() => {
        upstream.destroy()
        resolve()
      })
      if (body) upstream.end(body)
      else upstream.end()
    })
  }

  // ---------- models ----------

  /**
   * The model's instance for a request. One that is already loaded or loading (by the UI, say) is
   * used as it is, whatever its settings, and stays the user's: only a load started here is
   * unloaded again when idle.
   */
  private ensureLoaded(modelId: string, mode?: AppMode): Promise<LoadedInstance> {
    let p = this.loading.get(modelId)
    if (!p) {
      p = this.deps.instances
        .ensure(modelId, mode)
        .then(({ instance, started }) => {
          // An OpenRouter model has nothing to unload when idle, so it is never tracked as server-loaded.
          if (started && !instance.cloud) {
            this.jit.set(instance.instanceId, Date.now())
            this.emit()
          }
          return instance
        })
        .finally(() => this.loading.delete(modelId))
      this.loading.set(modelId, p)
    }
    return p
  }

  /** Restarts a loaded model so new load defaults take effect; does nothing if it isn't loaded. */
  private reload(modelId: string): void {
    const prev = this.deps.instances.list().find((i) => i.modelId === modelId && i.status !== 'stopping')
    // A rental's model is never reloaded from here.
    if (!prev || prev.host || prev.cloud) return
    // Keep auto-unloading a model only if the server loaded it in the first place.
    const wasJit = this.jit.delete(prev.instanceId)
    const p = this.deps.instances
      .load(modelId, null)
      .then((inst) => {
        if (wasJit) this.jit.set(inst.instanceId, Date.now())
        this.emit()
        return inst
      })
      .finally(() => this.loading.delete(modelId))
    p.catch(() => undefined)
    this.loading.set(modelId, p)
  }

  /** Leaves (or extends) the lease of a model: until LEASE_MS after now. */
  private lease(modelId: string): void {
    this.leases.set(modelId, (this.deps.now?.() ?? Date.now()) + LEASE_MS)
  }

  /** Until when a model is leased by a request that named a mode (see LEASE_MS); null when it is not. */
  leasedUntil(modelId: string): number | null {
    const until = this.leases.get(modelId)
    if (until === undefined) return null
    if (until <= (this.deps.now?.() ?? Date.now())) {
      this.leases.delete(modelId)
      return null
    }
    return until
  }

  /**
   * Loads a loaded model again with the settings a mode has for it, when it runs with others (the reload bar's rule:
   * reloadFor). Refused while it loads, while something uses it or the app is busy (reloadBlock), and for a rental's
   * model. The model gate is held until the load has ended, and a model this server loaded stays one it may unload when
   * idle. `also` is a caller's own reason to refuse (the phone looks at leases), asked after the app's.
   */
  requestModeReload(model: ModelEntry, mode: AppMode, also?: () => string | null): ModeReloadResult {
    const modes = this.deps.modes
    const all = this.deps.instances.list().filter((i) => i.modelId === model.id && i.status !== 'stopping')
    if (!modes || !all.length) return { status: 'not-loaded' }
    if (model.remote) return { status: 'blocked', why: remoteRefusal(model, 'reload it') }
    if (model.cloud) return { status: 'blocked', why: cloudRefusal(model, 'reload it') }
    const inst = all.find((i) => i.status === 'ready') ?? all[0]
    if (inst.status === 'loading' || this.loading.has(model.id)) return { status: 'blocked', why: `${model.name} is still loading.` }
    const wanted = reloadFor(inst.config, this.deps.getLoadDefaults(model.id).config, modes.get(model.id), mode)
    if (!wanted) return { status: 'matches' }
    const why = modes.reloadBlock?.(inst.instanceId) ?? also?.() ?? null
    if (why) return { status: 'blocked', why }
    const holder = `reload:${model.id}`
    const gate = modes.takeGate?.(holder)
    if (typeof gate === 'string') return { status: 'blocked', why: gate }
    // The gate stays taken until the load has ended or failed.
    const wasJit = this.jit.delete(inst.instanceId)
    let p: Promise<LoadedInstance>
    try {
      // Asked again when the load's turn comes: a load queued before it can leave time for a request to start.
      p = this.deps.instances.load(model.id, wanted, undefined, () => modes.reloadBlock?.(inst.instanceId, holder) ?? also?.() ?? null)
    } catch (err) {
      gate?.()
      return { status: 'blocked', why: err instanceof Error ? err.message : String(err) }
    }
    p = p
      .then((i) => {
        if (wasJit) this.jit.set(i.instanceId, this.deps.now?.() ?? Date.now())
        this.emit()
        return i
      })
      .finally(() => {
        gate?.()
        this.loading.delete(model.id)
      })
    p.catch(() => undefined)
    this.loading.set(model.id, p)
    return { status: 'started' }
  }

  private touch(instanceId: string): void {
    if (this.jit.has(instanceId)) {
      this.jit.set(instanceId, Date.now())
      this.emit()
    }
  }

  /** Unloads every instance of a model, loading or ready; resolves how many there were. */
  private async unloadModel(modelId: string): Promise<number> {
    // Never a rental's instance: only the window unloads those.
    const targets = this.deps.instances.list().filter((i) => i.modelId === modelId && i.status !== 'stopping' && isLocalInstance(i))
    for (const i of targets) this.jit.delete(i.instanceId)
    await Promise.all(targets.map((i) => this.deps.instances.unload(i.instanceId)))
    if (targets.length) this.emit()
    return targets.length
  }

  private async unloadIdle(): Promise<void> {
    const minutes = this.deps.getSettings().idleUnloadMinutes
    if (!minutes) return
    const cutoff = Date.now() - minutes * 60_000
    const all = this.deps.instances.list()
    const live = new Set(all.map((i) => i.instanceId))
    const remote = new Set(all.filter((i) => !isLocalInstance(i)).map((i) => i.instanceId))
    let changed = false
    for (const instanceId of [...this.jit.keys()]) {
      if (!live.has(instanceId)) {
        changed = this.jit.delete(instanceId) || changed
        continue
      }
      // Never unloaded here, whatever the idle time: a rental's model is the window's to unload.
      if (remote.has(instanceId)) {
        this.jit.delete(instanceId)
        changed = true
        continue
      }
      // Idle time counts from the last use by anyone (API requests and in-app chats alike); an
      // instance in use right now is never idle.
      const since = this.deps.instances.idleSince(instanceId)
      if (since === null || since >= cutoff) continue
      this.jit.delete(instanceId)
      changed = true
      await this.deps.instances.unload(instanceId)
    }
    if (changed) this.emit()
  }

  /** The models a client can use: the library's, a rental's only while it is loaded on one, and OpenRouter's only when ticked for other apps. */
  private visibleModels(): ModelEntry[] {
    return this.deps.getModels().filter((m) => (m.cloud ? !!this.deps.cloudAllowed?.(m.id) : !m.remote || !!this.deps.instances.readyFor(m.id)))
  }

  /** A model by the name a client sent; an OpenRouter one only by its exact id, and only while other apps may use it. */
  private resolveApi(requested: string): ModelEntry | undefined {
    const m = resolveModel(this.deps.getModels(), requested)
    return m && (!m.cloud || this.deps.cloudAllowed?.(m.id)) ? m : undefined
  }

  private listModels(): unknown {
    return { object: 'list', data: this.visibleModels().map((m) => this.modelEntry(m)) }
  }

  private modelEntry(m: ModelEntry): unknown {
    if (m.cloud) {
      return {
        id: m.id,
        object: 'model',
        created: m.cloud.info.created || Math.floor(m.modifiedAt / 1000),
        owned_by: 'openrouter',
        path: m.id,
        max_context_length: m.cloud.info.contextLength,
        // Always usable: it connects with the first request.
        state: 'loaded',
        loaded_context_length: m.cloud.contextLimit,
        cloud: { provider: 'openrouter', free: m.cloud.info.free }
      }
    }
    const inst = this.deps.instances.readyFor(m.id)
    return {
      id: publicModelId(m, this.deps.getModels()),
      object: 'model',
      created: Math.floor(m.modifiedAt / 1000),
      owned_by: 'rulz',
      // Extra fields, in the spirit of LM Studio's /api/v0/models.
      path: m.id,
      arch: m.info?.architecture,
      quantization: m.info?.quantization,
      max_context_length: m.info?.contextLength,
      state: inst ? 'loaded' : 'not-loaded',
      loaded_context_length: inst?.contextLength ?? undefined,
      // Only while it is loaded on a rental: where its text goes.
      ...(inst?.host ? { rental: { label: inst.host.label, reconnecting: !!inst.reconnecting } } : {})
    }
  }

  /** A client's load-config patch, checked: unlike saved settings, bad values are refused. Errors are the 400 message. */
  private checkLoadPatch(raw: unknown, model: ModelEntry): { ok: true; patch: RemoteLoadPatch } | { ok: false; error: string } {
    const parsed = parseRemoteLoadPatch(raw)
    if (!parsed.ok) return { ok: false, error: `Invalid load setting ${parsed.error}` }
    const patch = { ...parsed.patch }
    if (patch.draftFile !== undefined) {
      const draftFile = draftFileFromRemote(patch.draftFile, model, this.deps.getModels())
      if (draftFile === undefined) return { ok: false, error: 'Invalid load setting draftFile: not one of draft_choices' }
      patch.draftFile = draftFile
    }
    if (patch.adapters !== undefined) {
      const checked = adaptersFromRemote(patch.adapters, model, this.deps.getAdapters?.() ?? [])
      if (!checked.ok) return { ok: false, error: checked.error }
      patch.adapters = checked.adapters
    }
    return { ok: true, patch }
  }

  /** The body of GET /v1/models/{id}/modes. `reloading`: a reload was just started, whose new instance does not exist yet. */
  private modesBody(m: ModelEntry, reloading = false): unknown {
    const all = this.deps.modes?.get(m.id) ?? {}
    const base = this.deps.getLoadDefaults(m.id).config
    const list = this.deps.instances.list().filter((i) => i.modelId === m.id && i.status !== 'stopping')
    const ready = this.deps.instances.readyFor(m.id)
    const inst = reloading ? undefined : (ready ?? list.find((i) => i.status === 'loading'))
    const sampling = (inst?.config ?? base).sampling
    const entryOf = (mode: ApiMode): unknown => {
      const own = all[mode]
      const fields = own?.sampling ? samplingRequestFields(own.sampling) : null
      return {
        own_load: !!own?.load,
        sampling: fields?.sampling ?? null,
        thinking: fields?.thinking ?? null,
        system_prompt: modePromptExtra(all, mode),
        matches: m.cloud ? true : inst ? reloadFor(inst.config, base, all, mode) === null : null
      }
    }
    return {
      object: 'model.modes',
      id: publicModelId(m, this.deps.getModels()),
      state: reloading ? 'loading' : ready ? 'ready' : inst ? 'loading' : 'not-loaded',
      instance: inst?.instanceId ?? null,
      api_override: sampling.override ? overrideSampling({}, sampling) : null,
      modes: Object.fromEntries(API_MODES.map((mode) => [mode, entryOf(mode)]))
    }
  }

  private loadConfigEntry(m: ModelEntry, preview?: LoadConfig, mode?: ApiMode): unknown {
    const saved = this.deps.getLoadDefaults(m.id)
    const modes = mode ? this.deps.modes?.get(m.id) : undefined
    const config = preview ?? (mode ? modeLoad(saved.config, modes, mode) : saved.config)
    const inst = this.deps.instances.list().find((i) => i.modelId === m.id && i.status !== 'stopping')
    return {
      id: publicModelId(m, this.deps.getModels()),
      object: 'model.load_config',
      ...(mode ? { mode, mode_own: !!modes?.[mode]?.load } : {}),
      config: remoteLoadView(config, m),
      custom: saved.custom,
      dry_run: preview ? true : undefined,
      has_draft: !!m.draft?.kind,
      // What "automatic" (draftFile null) uses, and what else can be picked.
      default_draft: m.draft?.kind ? { name: m.draft.builtIn ? 'Built-in MTP layers' : fileName(m.draft.path), kind: m.draft.kind, built_in: !!m.draft.builtIn } : null,
      draft_choices: m.cloud ? [] : draftChoices(m, this.deps.getModels()),
      adapters: config.adapters ?? [],
      adapter_choices: m.cloud ? [] : adapterChoiceList(m, this.deps.getAdapters?.() ?? []),
      gpus: m.remote || m.cloud ? [] : this.deps.gpus().map(({ index, name, memUsedMiB, memTotalMiB }) => ({ index, name, memUsedMiB, memTotalMiB })),
      // The estimate is about this PC's memory, which says nothing about a rental.
      estimate: m.remote || m.cloud ? null : this.deps.estimate(m, config),
      max_context_length: m.cloud ? m.cloud.info.contextLength : m.info?.contextLength,
      // Layers whose experts cpuMoe can keep on the CPU; null for dense models.
      moe_layers: m.info?.expertCount ? (m.info.expertBytesByLayer?.length ?? m.info.blockCount ?? null) : null,
      state: inst?.status ?? 'not-loaded',
      loaded_config: inst ? remoteLoadView(inst.config, m) : undefined,
      loaded_context_length: inst?.contextLength ?? undefined
    }
  }

  private record(entry: ApiRequestLog): void {
    const i = this.log.findIndex((e) => e.id === entry.id)
    if (i >= 0) this.log[i] = { ...entry }
    else {
      this.log.push({ ...entry })
      if (this.log.length > MAX_LOG) this.log.splice(0, this.log.length - MAX_LOG)
    }
    this.deps.onRequest({ ...entry })
  }

  private emit(): void {
    this.deps.onStatus(this.getStatus())
  }
}

/** A request header's value when it is a single string, trimmed. */
function headerValue(req: IncomingMessage, name: string): string | undefined {
  const v = req.headers[name]
  return typeof v === 'string' ? v.trim() : undefined
}

/** A Host that means this computer or a local network name, not a name on the internet a web page could point here. */
export function ownHost(host: string | undefined): boolean {
  const name = (host ?? '').trim().toLowerCase().replace(/:\d+$/, '').replace(/\.$/, '')
  if (!name || name.startsWith('[') || !name.includes('.')) return true
  return /^\d{1,3}(\.\d{1,3}){3}$/.test(name) || /\.(local|localhost|lan|home|internal|ts\.net)$/.test(name)
}

/** What an OpenRouter model answers to: a llama-server's chat, props, tokenize and template routes, as the gateway serves them. */
const CLOUD_PATHS = new Set(['POST /v1/chat/completions', 'POST /v1/chat/completions/input_tokens', 'GET /props', 'POST /tokenize', 'POST /apply-template'])

/** Why the API server won't reload, unload or change the settings of an OpenRouter model. */
function cloudRefusal(model: ModelEntry, what: string): string {
  return `${model.name} is an OpenRouter model, so nothing is loaded here and the API server cannot ${what}. Manage it on the OpenRouter page of the app.`
}

/** Why the API server won't load, unload or change a model that runs on a rented vast.ai instance. */
function remoteRefusal(model: ModelEntry, what: string): string {
  return `${model.name} runs on a rented vast.ai instance. The API server only uses it while it is loaded there and cannot ${what}; load it on the vast.ai page of the app.`
}

class BodyTooLarge extends Error {
  constructor() {
    super(`The request body is larger than ${MAX_BODY_BYTES / (1024 * 1024)} MiB.`)
  }
}

function readBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    let size = 0
    const onData = (c: Buffer): void => {
      size += c.length
      if (size > MAX_BODY_BYTES) {
        // Stop keeping it, but read the rest so the client finishes sending and then reads the 413.
        // Past DRAIN_LIMIT_BYTES more it is cut off: nobody is told anything about an endless body.
        req.off('data', onData)
        chunks.length = 0
        let drained = 0
        req.on('data', (d: Buffer) => {
          drained += d.length
          if (drained > DRAIN_LIMIT_BYTES) req.destroy()
        })
        req.resume()
        reject(new BodyTooLarge())
        return
      }
      chunks.push(c)
    }
    req.on('data', onData)
    req.on('end', () => resolve(Buffer.concat(chunks)))
    req.on('error', reject)
  })
}

/**
 * kitDir from POST /v1/re/kit. Trimmed, the way the kit field saves it. A problem is `{ error }`.
 * An empty or unreadable kit is still a valid folder; the practice list reports that itself.
 */
async function parseKitDir(json: Record<string, unknown>): Promise<string | { error: string }> {
  const kitDir = json.kitDir
  if (typeof kitDir !== 'string' || kitDir.trim().length === 0 || kitDir.length > 1000) {
    return { error: 'kitDir must be a non-empty string of at most 1000 characters.' }
  }
  const dir = kitDir.trim()
  if (!isAbsolute(dir)) return { error: 'kitDir must be an absolute path to an existing folder.' }
  // A network share would let someone else supply the programs practice tasks run.
  if (/^[\\/]{2}/.test(dir)) return { error: 'kitDir must be a folder on this computer, not a network path.' }
  try {
    if (!(await stat(dir)).isDirectory()) return { error: 'kitDir must be an absolute path to an existing folder.' }
  } catch {
    return { error: 'kitDir must be an absolute path to an existing folder.' }
  }
  return dir
}

/** Checks the body of POST /v1/re/sessions/{id}; returns the problem as text. */
export function parseReSessionPatch(json: Record<string, unknown>): Partial<ApiReSession> | string {
  const patch: Partial<ApiReSession> = {}
  for (const k of ['client', 'title', 'target', 'goal', 'notes'] as const) {
    const v = json[k]
    if (v === undefined) continue
    if (typeof v !== 'string' || v.length > 4000) return `${k} must be a string of at most 4000 characters.`
    patch[k] = v
  }
  if (json.outcome !== undefined) {
    if (!(RE_OUTCOMES as readonly unknown[]).includes(json.outcome)) return `outcome must be one of ${RE_OUTCOMES.join(', ')}.`
    patch.outcome = json.outcome as ApiReSession['outcome']
  }
  if (json.kind !== undefined) {
    if (!['analyse', 'practice', 'clone-pieces', 'clone-full'].includes(json.kind as string)) return 'kind must be analyse, practice, clone-pieces or clone-full.'
    patch.kind = json.kind as ApiReSession['kind']
  }
  if (json.practice !== undefined) {
    const p = json.practice as Record<string, unknown> | null
    const str = (v: unknown, max: number): boolean => typeof v === 'string' && v.length <= max
    if (!p || typeof p !== 'object' || !str(p.taskId, 80) || !str(p.family, 80) || !str(p.tier, 40)) return 'practice needs taskId, family and tier strings.'
    if ((p.answer !== undefined && !str(p.answer, 4000)) || (p.expected !== undefined && !str(p.expected, 4000))) return 'practice answer and expected must be strings.'
    patch.practice = {
      taskId: p.taskId as string,
      family: p.family as string,
      tier: p.tier as string,
      ...(p.answer !== undefined ? { answer: p.answer as string } : {}),
      ...(p.expected !== undefined ? { expected: p.expected as string } : {})
    }
  }
  return patch
}

function errorMessageFrom(text: string): string | undefined {
  try {
    const j = JSON.parse(text) as { error?: { message?: string } | string }
    return typeof j.error === 'string' ? j.error : j.error?.message
  } catch {
    return text.trim().slice(0, 300) || undefined
  }
}
