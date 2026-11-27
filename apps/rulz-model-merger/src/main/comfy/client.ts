// HTTP and WebSocket client for ComfyUI (checked against ComfyUI 0.46 server.py and execution.py).
import { WebSocket } from 'ws'
import type { GpuInfo, RunningServer, ServerCapabilities } from '@shared/types'
import { matchGpuIndexes, normalizeServerUrl, parseCapabilities, parseSystemStats, type SystemStatsInfo } from './parse'

export class ComfyError extends Error {
  status?: number
  /** The `node_errors` object ComfyUI returned with an HTTP 400 from POST /prompt. */
  nodeErrors?: unknown
  constructor(message: string, extra?: { status?: number; nodeErrors?: unknown }) {
    super(message)
    this.name = 'ComfyError'
    this.status = extra?.status
    this.nodeErrors = extra?.nodeErrors
  }
}

export interface HistoryImage {
  filename: string
  subfolder: string
  type: string
}

export interface HistoryEntry {
  promptId: string
  completed: boolean
  statusStr: string | null
  images: { nodeId: string; image: HistoryImage }[]
  error: string | null
}

export interface ComfyEventHandlers {
  onStatus?(queueRemaining: number): void
  onProgress?(p: { promptId: string | null; node: string | null; value: number; max: number }): void
  onExecuting?(p: { promptId: string | null; node: string | null }): void
  onExecuted?(p: { promptId: string; node: string; images: HistoryImage[] }): void
  onSuccess?(promptId: string): void
  onError?(p: { promptId: string; message: string; nodeId: string | null }): void
  onClose?(): void
}

export interface ComfyClientOptions {
  fetch?: typeof fetch
  timeoutMs?: number
  webSocket?: typeof WebSocket
}

const DEFAULT_TIMEOUT_MS = 10_000
const OBJECT_INFO_TIMEOUT_MS = 60_000

function rec(v: unknown): Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {}
}
function str(v: unknown): string | null {
  return typeof v === 'string' ? v : null
}

function parseImages(v: unknown): HistoryImage[] {
  if (!Array.isArray(v)) return []
  const out: HistoryImage[] = []
  for (const item of v) {
    const o = rec(item)
    const filename = str(o.filename)
    if (!filename) continue
    out.push({ filename, subfolder: str(o.subfolder) ?? '', type: str(o.type) ?? 'output' })
  }
  return out
}

/** A readable text for a 400 from POST /prompt: the error plus the first problems per node. */
function describePromptError(body: unknown): string {
  const b = rec(body)
  const err = b.error
  let head = ''
  if (typeof err === 'string') head = err
  else {
    const e = rec(err)
    const message = str(e.message) ?? 'The prompt was rejected'
    const details = str(e.details)
    head = details && details !== message ? `${message}: ${details}` : message
  }
  const lines: string[] = []
  for (const [nodeId, value] of Object.entries(rec(b.node_errors))) {
    const ne = rec(value)
    const cls = str(ne.class_type)
    const errs = Array.isArray(ne.errors) ? ne.errors : []
    for (const x of errs) {
      const xe = rec(x)
      const m = str(xe.message) ?? 'error'
      const d = str(xe.details)
      lines.push(`node ${nodeId}${cls ? ` (${cls})` : ''}: ${m}${d ? ` ${d}` : ''}`)
    }
  }
  return lines.length > 0 ? `${head}. ${lines.slice(0, 5).join('; ')}` : head
}

export class ComfyClient {
  readonly baseUrl: string
  private readonly fetchFn: typeof fetch
  private readonly timeoutMs: number
  private readonly WS: typeof WebSocket

  constructor(baseUrl: string, opts: ComfyClientOptions = {}) {
    this.baseUrl = normalizeServerUrl(baseUrl)
    this.fetchFn = opts.fetch ?? ((input, init) => fetch(input, init))
    this.timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS
    this.WS = opts.webSocket ?? WebSocket
  }

  /**
   * One request with a timeout that stays armed until `read` has finished with the body. Network failures and timeouts
   * (during the headers or the body) become ComfyError naming the URL.
   */
  private async call<T>(path: string, init: RequestInit, read: (res: Response) => Promise<T>, timeoutMs = this.timeoutMs): Promise<T> {
    const url = `${this.baseUrl}${path}`
    const ctrl = new AbortController()
    let timedOut = false
    const timer = setTimeout(() => {
      timedOut = true
      ctrl.abort()
    }, timeoutMs)
    try {
      const res = await this.fetchFn(url, { ...init, signal: ctrl.signal })
      return await read(res)
    } catch (e) {
      if (timedOut) throw new ComfyError(`No answer from ${url} after ${Math.round(timeoutMs / 1000)} s.`)
      if (e instanceof ComfyError) throw e
      const reason = e instanceof Error ? (e.cause instanceof Error ? e.cause.message : e.message) : String(e)
      throw new ComfyError(`Could not reach ${url}: ${reason}`)
    } finally {
      clearTimeout(timer)
    }
  }

  private async httpError(res: Response, path: string): Promise<ComfyError> {
    const text = (await res.text()).slice(0, 300)
    return new ComfyError(`${this.baseUrl}${path} answered HTTP ${res.status}${text ? `: ${text}` : ''}`, { status: res.status })
  }

  private async parseJsonBody(res: Response, path: string): Promise<unknown> {
    const text = await res.text()
    try {
      return JSON.parse(text)
    } catch {
      throw new ComfyError(`${this.baseUrl}${path} did not return JSON. Is this a ComfyUI server?`, { status: res.status })
    }
  }

  private getJson(path: string, timeoutMs?: number): Promise<unknown> {
    return this.call(
      path,
      {},
      async (res) => {
        if (!res.ok) throw await this.httpError(res, path)
        return this.parseJsonBody(res, path)
      },
      timeoutMs
    )
  }

  private postJson(path: string, body: unknown): Promise<void> {
    return this.call(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }, async (res) => {
      if (!res.ok) throw await this.httpError(res, path)
      await res.text()
    })
  }

  /** GET /system_stats */
  async systemStats(): Promise<SystemStatsInfo> {
    return parseSystemStats(await this.getJson('/system_stats'))
  }

  /** GET /object_info (large; 60 s timeout). */
  async objectInfo(): Promise<Record<string, unknown>> {
    return rec(await this.getJson('/object_info', Math.max(this.timeoutMs, OBJECT_INFO_TIMEOUT_MS)))
  }

  async capabilities(): Promise<ServerCapabilities> {
    return parseCapabilities(await this.objectInfo())
  }

  /** POST /prompt with `{prompt, client_id}` (plus `extra_data` when given, e.g. extra_pnginfo). HTTP 400 becomes a ComfyError carrying `nodeErrors`. */
  async queuePrompt(
    prompt: Record<string, unknown>,
    clientId: string,
    extraData?: Record<string, unknown>
  ): Promise<{ promptId: string; number: number }> {
    const path = '/prompt'
    const body = extraData ? { prompt, client_id: clientId, extra_data: extraData } : { prompt, client_id: clientId }
    return this.call(
      path,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
      async (res) => {
        if (res.status === 400) {
          const text = await res.text()
          let body: unknown = null
          try {
            body = JSON.parse(text)
          } catch {
            /* not JSON */
          }
          const message = body === null ? `${this.baseUrl}${path} answered HTTP 400: ${text.slice(0, 300)}` : describePromptError(body)
          throw new ComfyError(message, { status: 400, nodeErrors: rec(body).node_errors })
        }
        if (!res.ok) throw await this.httpError(res, path)
        const data = rec(await this.parseJsonBody(res, path))
        const promptId = str(data.prompt_id)
        if (!promptId) throw new ComfyError(`${this.baseUrl}${path} did not return a prompt_id.`, { status: res.status })
        const number = typeof data.number === 'number' ? data.number : 0
        return { promptId, number }
      }
    )
  }

  /** GET /history/{id}. Null while ComfyUI does not know the id (not finished yet, or dropped from history). */
  async history(promptId: string): Promise<HistoryEntry | null> {
    const data = rec(await this.getJson(`/history/${encodeURIComponent(promptId)}`))
    const entry = data[promptId]
    if (entry === undefined || entry === null) return null
    const e = rec(entry)
    const status = rec(e.status)
    const statusStr = str(status.status_str)
    const completed = status.completed === true
    const images: HistoryEntry['images'] = []
    for (const [nodeId, out] of Object.entries(rec(e.outputs))) {
      for (const image of parseImages(rec(out).images)) images.push({ nodeId, image })
    }
    let error: string | null = null
    const messages = Array.isArray(status.messages) ? status.messages : []
    for (const m of messages) {
      if (!Array.isArray(m)) continue
      const kind = m[0]
      const d = rec(m[1])
      if (kind === 'execution_error') {
        const text = str(d.exception_message)?.trim() || 'The server reported an error.'
        const type = str(d.node_type)
        error = type ? `${type}: ${text}` : text
      } else if (kind === 'execution_interrupted' && error === null) {
        error = 'Interrupted'
      }
    }
    if (error === null && statusStr === 'error') error = 'The server reported an error.'
    return { promptId, completed, statusStr, images, error }
  }

  /** GET /queue -> prompt ids (item[1] of each queue tuple). */
  async queue(): Promise<{ running: string[]; pending: string[] }> {
    const data = rec(await this.getJson('/queue'))
    const ids = (v: unknown): string[] =>
      Array.isArray(v) ? v.map((item) => (Array.isArray(item) ? item[1] : null)).filter((x): x is string => typeof x === 'string') : []
    return { running: ids(data.queue_running), pending: ids(data.queue_pending) }
  }

  /**
   * POST /queue `{delete: [ids]}`. Removes pending jobs from the server's queue.
   * ONLY call this on a server this app started: on someone else's server it would touch jobs we do not own. The
   * client does not enforce that; the caller must.
   */
  async deletePending(promptIds: string[]): Promise<void> {
    if (promptIds.length === 0) return
    await this.postJson('/queue', { delete: promptIds })
  }

  /**
   * POST /interrupt. With a prompt id the server interrupts only if that prompt is running; without one it interrupts
   * whatever runs.
   * ONLY call this on a server this app started. The client does not enforce that; the caller must.
   */
  async interrupt(promptId?: string): Promise<void> {
    await this.postJson('/interrupt', promptId ? { prompt_id: promptId } : {})
  }

  /**
   * POST /free `{unload_models, free_memory}`. The server applies it before its next job.
   * ONLY call this on a server this app started: it unloads models other users may be relying on. The client does not
   * enforce that; the caller must.
   */
  async free(opts: { unloadModels: boolean; freeMemory: boolean }): Promise<void> {
    await this.postJson('/free', { unload_models: opts.unloadModels, free_memory: opts.freeMemory })
  }

  /** GET /view?filename=&subfolder=&type= */
  async viewImage(img: HistoryImage): Promise<Buffer> {
    const q = new URLSearchParams({ filename: img.filename, subfolder: img.subfolder, type: img.type })
    const path = `/view?${q.toString()}`
    return this.call(path, {}, async (res) => {
      if (!res.ok) throw await this.httpError(res, '/view')
      return Buffer.from(await res.arrayBuffer())
    })
  }

  /**
   * Open ws://host/ws?clientId=... and route its JSON messages to the handlers. Binary frames (previews) and malformed
   * messages are ignored; a handler that throws does not break the stream.
   */
  openEvents(clientId: string, handlers: ComfyEventHandlers): { close(): void } {
    const wsUrl = `${this.baseUrl.replace(/^http/, 'ws')}/ws?clientId=${encodeURIComponent(clientId)}`
    const ws = new this.WS(wsUrl)
    let closed = false
    const safe = (fn: () => void): void => {
      try {
        fn()
      } catch {
        /* a handler must not break the stream */
      }
    }
    const route = (type: string, d: Record<string, unknown>): void => {
      switch (type) {
        case 'status': {
          const remaining = rec(rec(rec(d.status).exec_info)).queue_remaining
          if (typeof remaining === 'number') safe(() => handlers.onStatus?.(remaining))
          break
        }
        case 'progress': {
          const value = d.value
          const max = d.max
          if (typeof value === 'number' && typeof max === 'number') {
            safe(() => handlers.onProgress?.({ promptId: str(d.prompt_id), node: nodeStr(d.node), value, max }))
          }
          break
        }
        case 'executing':
          safe(() => handlers.onExecuting?.({ promptId: str(d.prompt_id), node: nodeStr(d.node) }))
          break
        case 'executed': {
          const promptId = str(d.prompt_id)
          const node = nodeStr(d.node)
          if (promptId && node) {
            const images = parseImages(rec(d.output).images)
            safe(() => handlers.onExecuted?.({ promptId, node, images }))
          }
          break
        }
        case 'execution_success': {
          const promptId = str(d.prompt_id)
          if (promptId) safe(() => handlers.onSuccess?.(promptId))
          break
        }
        case 'execution_error': {
          const promptId = str(d.prompt_id)
          if (promptId) {
            const message = str(d.exception_message)?.trim() || 'The server reported an error.'
            safe(() => handlers.onError?.({ promptId, message, nodeId: nodeStr(d.node_id) }))
          }
          break
        }
        case 'execution_interrupted': {
          const promptId = str(d.prompt_id)
          if (promptId) safe(() => handlers.onError?.({ promptId, message: 'Interrupted', nodeId: nodeStr(d.node_id) }))
          break
        }
        default:
          break
      }
    }
    ws.on('message', (data: unknown, isBinary: boolean) => {
      if (closed || isBinary) return
      try {
        const text = Buffer.isBuffer(data) ? data.toString('utf8') : String(data)
        const msg = rec(JSON.parse(text))
        const type = str(msg.type)
        if (type) route(type, rec(msg.data))
      } catch {
        /* malformed frame */
      }
    })
    ws.on('error', () => {
      /* a close event follows; nothing to add */
    })
    ws.on('close', () => {
      if (closed) return
      closed = true
      safe(() => handlers.onClose?.())
    })
    return {
      close(): void {
        if (closed) return
        closed = true
        try {
          // close() on a socket that is still connecting throws or errors; terminate() is safe in every state.
          if (ws.readyState === 0) ws.terminate()
          else ws.close()
        } catch {
          /* already closed */
        }
        safe(() => handlers.onClose?.())
      }
    }
  }
}

function nodeStr(v: unknown): string | null {
  if (typeof v === 'string') return v
  if (typeof v === 'number') return String(v)
  return null
}

/**
 * Ask a URL whether it is a ComfyUI server (GET /system_stats, short timeout). Null when it is unreachable or not
 * ComfyUI. `own` is false: this finds servers, it does not start them.
 */
export async function probeServer(
  url: string,
  gpus: GpuInfo[],
  opts: { fetch?: typeof fetch; timeoutMs?: number } = {}
): Promise<RunningServer | null> {
  let base: string
  try {
    base = normalizeServerUrl(url)
  } catch {
    return null
  }
  const client = new ComfyClient(base, { fetch: opts.fetch, timeoutMs: opts.timeoutMs ?? 1500 })
  let raw: unknown
  try {
    const f = opts.fetch ?? ((input: Parameters<typeof fetch>[0], init?: RequestInit) => fetch(input, init))
    const ctrl = new AbortController()
    const timer = setTimeout(() => ctrl.abort(), opts.timeoutMs ?? 1500)
    try {
      const res = await f(`${client.baseUrl}/system_stats`, { signal: ctrl.signal })
      if (!res.ok) return null
      raw = await res.json()
    } finally {
      clearTimeout(timer)
    }
  } catch {
    return null
  }
  const r = rec(raw)
  if (!r.system || typeof r.system !== 'object' || !Array.isArray(r.devices)) return null
  const info = parseSystemStats(raw)
  const u = new URL(client.baseUrl)
  const port = u.port ? Number(u.port) : u.protocol === 'https:' ? 443 : 80
  return {
    url: client.baseUrl,
    port,
    version: info.version,
    devices: info.devices,
    cudaDevice: info.cudaDevice,
    gpuIndexes: matchGpuIndexes(info.devices, gpus),
    own: false
  }
}
