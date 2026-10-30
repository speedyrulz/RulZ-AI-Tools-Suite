import { randomUUID } from 'node:crypto'
import http, { type IncomingMessage, type ServerResponse } from 'node:http'
import type { ApiRequestLog, ApiServerSettings, ApiServerStatus, GpuStat, LoadConfig, LoadedInstance, MemoryEstimate, ModelEntry } from '@shared/types'
import { fileName } from '@shared/draft'
import type { InstanceManager } from '../engine/instances'
import {
  applyRemoteLoadPatch,
  authorized,
  draftChoices,
  draftFileFromRemote,
  errorBody,
  hoistSystemMessages,
  overrideSampling,
  parseRemoteLoadPatch,
  parseUsage,
  pickModel,
  publicModelId,
  remoteLoadView,
  resolveModel,
  SAMPLING_PATHS
} from './helpers'

const MAX_BODY_BYTES = 64 * 1024 * 1024
const MAX_LOG = 300
const USAGE_TAIL_BYTES = 16 * 1024
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

export interface ApiServerDeps {
  instances: InstanceManager
  getModels: () => ModelEntry[]
  getSettings: () => ApiServerSettings
  /** The model's saved load defaults (what a load without explicit settings uses). */
  getLoadDefaults: (modelId: string) => { config: LoadConfig; custom: boolean }
  setLoadDefaults: (modelId: string, config: LoadConfig) => Promise<void>
  estimate: (model: ModelEntry, config: LoadConfig) => MemoryEstimate | null
  gpus: () => GpuStat[]
  onStatus: (s: ApiServerStatus) => void
  onRequest: (entry: ApiRequestLog) => void
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
  private jit = new Map<string, number>() // instanceId → last used
  private idleTimer: NodeJS.Timeout | null = null
  private loading = new Map<string, Promise<LoadedInstance>>()

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
    const server = http.createServer((req, res) => void this.handle(req, res))
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
    const url = new URL(req.url ?? '/', 'http://localhost')
    const path = url.pathname.replace(/\/+$/, '') || '/'
    const method = (req.method ?? 'GET').toUpperCase()

    if (settings.cors) {
      res.setHeader('Access-Control-Allow-Origin', '*')
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta')
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    }
    if (method === 'OPTIONS') {
      res.writeHead(204).end()
      return
    }

    const entry: ApiRequestLog = { id: randomUUID(), at: Date.now(), method, path, phase: 'running' }
    this.status.requestCount++
    const finish = (status: number, error?: string): void => {
      entry.status = status
      entry.durationMs = Date.now() - entry.at
      entry.error = error
      entry.phase = 'done'
      this.record(entry)
    }
    const fail = (status: number, message: string, type?: string): void => {
      res.writeHead(status, { 'Content-Type': 'application/json' }).end(errorBody(message, type))
      finish(status, message)
    }

    if (!authorized(settings.apiKey, req.headers as Record<string, string>)) {
      return fail(401, 'Invalid API key.', 'authentication_error')
    }

    try {
      if (method === 'GET' && (path === '/' || path === '/health')) {
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify({ status: 'ok', app: 'RulZ LLM Suite' }))
        return finish(200)
      }
      if (method === 'GET' && (path === '/v1/models' || path === '/models' || path === '/api/v0/models')) {
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(this.listModels()))
        return finish(200)
      }
      if (path.startsWith('/v1/models/') && path.endsWith(LOAD_CONFIG_SUFFIX) && (method === 'GET' || method === 'POST')) {
        const model = resolveModel(this.deps.getModels(), decodeURIComponent(path.slice('/v1/models/'.length, -LOAD_CONFIG_SUFFIX.length)))
        if (!model) return fail(404, 'Model not found.', 'not_found_error')
        let preview: LoadConfig | undefined
        if (method === 'POST') {
          let json: { config?: unknown; reload?: unknown; dry_run?: unknown }
          try {
            json = JSON.parse((await readBody(req)).toString('utf8') || '{}') as typeof json
          } catch {
            return fail(400, 'The request body is not valid JSON.')
          }
          const parsed = parseRemoteLoadPatch(json.config ?? {})
          if (!parsed.ok) return fail(400, `Invalid load setting ${parsed.error}`, 'invalid_request_error')
          const patch = { ...parsed.patch }
          if (patch.draftFile !== undefined) {
            const draftFile = draftFileFromRemote(patch.draftFile, model, this.deps.getModels())
            if (draftFile === undefined) return fail(400, 'Invalid load setting draftFile: not one of draft_choices', 'invalid_request_error')
            patch.draftFile = draftFile
          }
          const merged = applyRemoteLoadPatch(this.deps.getLoadDefaults(model.id).config, patch)
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
        const model = resolveModel(this.deps.getModels(), decodeURIComponent(path.slice('/v1/models/'.length, -UNLOAD_SUFFIX.length)))
        if (!model) return fail(404, 'Model not found.', 'not_found_error')
        const unloaded = await this.unloadModel(model.id)
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify({ ...(this.modelEntry(model) as object), unloaded }))
        return finish(200)
      }
      if (method === 'GET' && path.startsWith('/v1/models/')) {
        const model = resolveModel(this.deps.getModels(), decodeURIComponent(path.slice('/v1/models/'.length)))
        if (!model) return fail(404, 'Model not found.', 'not_found_error')
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(this.modelEntry(model)))
        return finish(200)
      }

      let body: Buffer | null = null
      let requestedModel: string | undefined
      let stream = false
      if (method === 'POST' && PROXY_POST.has(path)) {
        body = await readBody(req)
        try {
          const json = JSON.parse(body.toString('utf8')) as { model?: unknown; stream?: unknown; system?: unknown; messages?: unknown }
          if (typeof json.model === 'string') requestedModel = json.model
          stream = json.stream === true
          if (path.startsWith('/v1/messages')) {
            const hoisted = hoistSystemMessages(json)
            if (hoisted) body = Buffer.from(JSON.stringify(hoisted), 'utf8')
          }
        } catch {
          return fail(400, 'The request body is not valid JSON.')
        }
      } else if (method === 'GET' && PROXY_GET.has(path)) {
        requestedModel = url.searchParams.get('model') ?? undefined
      } else {
        return fail(404, `Unknown endpoint: ${method} ${path}`, 'not_found_error')
      }

      const ready = this.deps.instances.list().filter((i) => i.status === 'ready').map((i) => i.modelId)
      const picked = pickModel(this.deps.getModels(), requestedModel, ready, settings)
      if (!picked) {
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

      let inst = this.deps.instances.readyFor(model.id)
      if (!inst) {
        entry.phase = 'loading'
        this.record(entry)
        try {
          inst = await this.ensureLoaded(model.id)
        } catch (err) {
          return fail(503, `Could not load ${model.name}: ${(err as Error).message}`, 'server_error')
        }
        entry.phase = 'running'
      }
      this.jit.has(inst.instanceId) && this.jit.set(inst.instanceId, Date.now())
      this.emit()

      if (body && inst.config.sampling.override && SAMPLING_PATHS.has(path)) {
        const json = JSON.parse(body.toString('utf8')) as unknown
        if (json && typeof json === 'object' && !Array.isArray(json)) {
          body = Buffer.from(JSON.stringify(overrideSampling(json as Record<string, unknown>, inst.config.sampling)))
        }
      }
      await this.proxy(req, res, inst, path + url.search, body, stream, entry, finish)
    } catch (err) {
      if (!res.headersSent) fail(500, (err as Error).message, 'server_error')
      else finish(res.statusCode, (err as Error).message)
    }
  }

  private proxy(
    req: IncomingMessage,
    res: ServerResponse,
    inst: LoadedInstance,
    pathWithQuery: string,
    body: Buffer | null,
    stream: boolean,
    entry: ApiRequestLog,
    finish: (status: number, error?: string) => void
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
          res.writeHead(ures.statusCode ?? 502, out)
          // Keep the tail of the response to read token usage from it.
          let tail = ''
          ures.on('data', (chunk: Buffer) => {
            res.write(chunk)
            tail = (tail + chunk.toString('utf8')).slice(-USAGE_TAIL_BYTES)
            if (stream && typeof (res as { flush?: () => void }).flush === 'function') (res as { flush?: () => void }).flush?.()
          })
          ures.on('end', () => {
            res.end()
            Object.assign(entry, parseUsage(tail, stream))
            this.touch(inst.instanceId)
            finish(ures.statusCode ?? 502, ures.statusCode && ures.statusCode >= 400 ? errorMessageFrom(tail) : undefined)
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
        if (!res.headersSent) res.writeHead(502, { 'Content-Type': 'application/json' }).end(errorBody(`Model server error: ${err.message}`, 'server_error'))
        else res.destroy()
        finish(502, err.message)
        resolve()
      })
      // A client that disconnects mid-stream stops generation on the model server too.
      res.on('close', () => {
        if (!res.writableFinished) {
          upstream.destroy()
          if (entry.phase !== 'done') {
            finish(499, 'Client disconnected')
            resolve()
          }
        }
      })
      if (body) upstream.end(body)
      else upstream.end()
    })
  }

  // ---------- models ----------

  private ensureLoaded(modelId: string): Promise<LoadedInstance> {
    let p = this.loading.get(modelId)
    if (!p) {
      p = this.deps.instances
        .load(modelId, null)
        .then((inst) => {
          this.jit.set(inst.instanceId, Date.now())
          this.emit()
          return inst
        })
        .finally(() => this.loading.delete(modelId))
      this.loading.set(modelId, p)
    }
    return p
  }

  /** Restarts a loaded model so new load defaults take effect; does nothing if it isn't loaded. */
  private reload(modelId: string): void {
    const prev = this.deps.instances.list().find((i) => i.modelId === modelId && i.status !== 'stopping')
    if (!prev) return
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

  private touch(instanceId: string): void {
    if (this.jit.has(instanceId)) {
      this.jit.set(instanceId, Date.now())
      this.emit()
    }
  }

  /** Unloads every instance of a model, loading or ready; resolves how many there were. */
  private async unloadModel(modelId: string): Promise<number> {
    const targets = this.deps.instances.list().filter((i) => i.modelId === modelId && i.status !== 'stopping')
    for (const i of targets) this.jit.delete(i.instanceId)
    await Promise.all(targets.map((i) => this.deps.instances.unload(i.instanceId)))
    if (targets.length) this.emit()
    return targets.length
  }

  private async unloadIdle(): Promise<void> {
    const minutes = this.deps.getSettings().idleUnloadMinutes
    if (!minutes) return
    const cutoff = Date.now() - minutes * 60_000
    const live = new Set(this.deps.instances.list().map((i) => i.instanceId))
    for (const [instanceId, lastUsed] of [...this.jit]) {
      if (!live.has(instanceId)) {
        this.jit.delete(instanceId)
        continue
      }
      if (lastUsed < cutoff) {
        this.jit.delete(instanceId)
        await this.deps.instances.unload(instanceId)
      }
    }
    this.emit()
  }

  private listModels(): unknown {
    const models = this.deps.getModels()
    return { object: 'list', data: models.map((m) => this.modelEntry(m)) }
  }

  private modelEntry(m: ModelEntry): unknown {
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
      loaded_context_length: inst?.contextLength ?? undefined
    }
  }

  private loadConfigEntry(m: ModelEntry, preview?: LoadConfig): unknown {
    const saved = this.deps.getLoadDefaults(m.id)
    const config = preview ?? saved.config
    const inst = this.deps.instances.list().find((i) => i.modelId === m.id && i.status !== 'stopping')
    return {
      id: publicModelId(m, this.deps.getModels()),
      object: 'model.load_config',
      config: remoteLoadView(config, m),
      custom: saved.custom,
      dry_run: preview ? true : undefined,
      has_draft: !!m.draft?.kind,
      // What "automatic" (draftFile null) uses, and what else can be picked.
      default_draft: m.draft?.kind ? { name: m.draft.builtIn ? 'Built-in MTP layers' : fileName(m.draft.path), kind: m.draft.kind, built_in: !!m.draft.builtIn } : null,
      draft_choices: draftChoices(m, this.deps.getModels()),
      gpus: this.deps.gpus().map(({ index, name, memUsedMiB, memTotalMiB }) => ({ index, name, memUsedMiB, memTotalMiB })),
      estimate: this.deps.estimate(m, config),
      max_context_length: m.info?.contextLength,
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

function readBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    let size = 0
    req.on('data', (c: Buffer) => {
      size += c.length
      if (size > MAX_BODY_BYTES) {
        reject(new Error('Request body too large'))
        req.destroy()
        return
      }
      chunks.push(c)
    })
    req.on('end', () => resolve(Buffer.concat(chunks)))
    req.on('error', reject)
  })
}

function errorMessageFrom(text: string): string | undefined {
  try {
    const j = JSON.parse(text) as { error?: { message?: string } | string }
    return typeof j.error === 'string' ? j.error : j.error?.message
  } catch {
    return text.trim().slice(0, 300) || undefined
  }
}
