// A stand-in for RulZ LLM Suite's API server: scripted model turns, streamed like llama.cpp.
import { createServer, type IncomingMessage } from 'node:http'
import type { AddressInfo } from 'node:net'

export interface MockTurn {
  /** Thinking sent separately, as reasoning_content, before the content. */
  reasoning?: string
  content?: string
  toolCalls?: { name: string; args: unknown }[]
  finish?: string
  /** Answer with this HTTP status and error instead. */
  status?: number
  error?: string
  /** Wait before answering (ms). */
  delay?: number
  /** Wait between the reasoning and the content (ms), as a model thinking for a while. */
  pause?: number
}

export interface MockRequest {
  path: string
  /** The RE session header RulZ LLM Suite records requests by, if sent. */
  reSession?: string
  /** The X-RulZ-Mode header, if sent. */
  mode?: string
  /** The X-RulZ-Rental header, if sent. */
  rental?: string
  body: {
    model?: string
    stream?: boolean
    messages?: { role: string; content: string; tool_calls?: { id: string; function: { name: string; arguments: string } }[]; tool_call_id?: string }[]
    tools?: unknown[]
    [k: string]: unknown
  }
}

export interface MockOptions {
  /** Pretend to be LM Studio (no RulZ fields, no /health app name). */
  lmStudio?: boolean
  apiKey?: string
  models?: {
    id: string
    state?: 'loaded' | 'not-loaded'
    loaded_context_length?: number
    /** A model on a rental, as LLM Suite 0.35 reports it. */
    rental?: { label: string | null; reconnecting: boolean }
    /** Overrides the path (LLM Suite 0.29 to 0.34 marked a rental by a `vast/` path). */
    path?: string
  }[]
  template?: string
  nCtx?: number
  turns?: MockTurn[] | ((req: MockRequest, index: number) => MockTurn)
  /** Report exact token counts like llama.cpp (timings in the last chunk). */
  timings?: (req: MockRequest) => {
    cache_n: number
    prompt_n: number
    predicted_n: number
    predicted_ms?: number
    draft_n?: number
    draft_n_accepted?: number
  }
  /** Answer GET/POST /v1/models/{id}/load-config like newer RulZ LLM Suite versions. */
  loadConfig?: boolean
  /**
   * LLM Suite 0.35: GET /v1/models/{id}/modes, POST .../reload and .../modes/{mode}/load-config. A body
   * (in LLM Suite's snake_case) is what the modes route answers; without this option every one of them is a 404, like
   * an older LLM Suite. As a function it can answer a status instead.
   */
  modes?: Record<string, unknown> | ((model: string) => { status: number; body: unknown })
  /** Answer POST .../reload with this 409 sentence (a reload that is refused). */
  reloadRefusal?: string
  /** Answer a mode's load-config POST with `reload_blocked: <this>` (settings saved, reload refused). */
  reloadBlocked?: string
  /** GET /v1/re/overview?model=: answer per model (`practice.model` set, `failed` counted). Without it the query is ignored, like an older LLM Suite. */
  practicePerModel?: boolean
}

export const QWEN_CODER_TEMPLATE =
  "{{- '<parameter=' + args_name + '>\n' }}{%- set args_value = args_value if args_value is string else args_value | string %}{{- args_value }}"

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = []
  for await (const c of req) chunks.push(c as Buffer)
  return Buffer.concat(chunks).toString('utf8')
}

export async function startMockRulz(opts: MockOptions = {}) {
  const requests: MockRequest[] = []
  const models = opts.models ?? [{ id: 'Qwen3-Coder-30B-A3B-Instruct-Q5_K_M', state: 'loaded', loaded_context_length: opts.nCtx ?? 65536 }]
  let turnIndex = 0
  const loadConfig = {
    contextLength: 65536,
    gpuLayers: 'auto',
    flashAttention: 'auto',
    cacheTypeK: 'f16',
    cacheTypeV: 'f16',
    parallel: 1,
    splitMode: 'auto',
    mainGpu: 0,
    tensorSplit: '',
    useDraft: true,
    draftTokens: 2,
    draftMinTokens: null,
    draftPMin: null,
    draftSampling: 'auto',
    draftGpu: 'auto',
    draftCacheTypeK: 'f16',
    draftCacheTypeV: 'f16',
    cpuMoe: 'off' as 'off' | 'all' | number
  }
  const GiB = 1024 ** 3
  const gpus = [
    { index: 0, name: 'RTX 4090', memUsedMiB: 1024, memTotalMiB: 24576 },
    { index: 1, name: 'RTX 3090', memUsedMiB: 512, memTotalMiB: 24576 }
  ]
  const loadConfigPosts: unknown[] = []
  /** POST .../modes/{mode}/load-config bodies. */
  const modeLoadConfigPosts: { mode: string; model: string; body: Record<string, unknown> }[] = []
  /** Every request that names a model and is not the model list: its path, method and the mode and rental headers. */
  const headerLog: { path: string; method: string; model: string; mode?: string; rental?: string }[] = []
  /** POST .../reload bodies. */
  const reloads: { model: string; body: Record<string, unknown> }[] = []
  /** The ?model= of each GET /v1/re/overview ('' when none). */
  const overviewQueries: string[] = []
  /** POST /v1/re/kit bodies. */
  const kits: { kitDir?: string }[] = []
  /** POST /v1/re/sessions/{id} bodies, as newer RulZ LLM Suite versions take them. */
  const reSessions: { id: string; body: Record<string, unknown> }[] = []
  const nextTurn = (req: MockRequest): MockTurn => {
    const t = opts.turns
    const i = turnIndex++
    if (typeof t === 'function') return t(req, i)
    return t?.[i] ?? { content: 'All done.' }
  }

  const server = createServer(async (req, res) => {
    const url = new URL(req.url ?? '/', 'http://x')
    const json = (status: number, body: unknown): void => {
      res.writeHead(status, { 'Content-Type': 'application/json' }).end(JSON.stringify(body))
    }
    if (opts.apiKey && req.headers.authorization !== `Bearer ${opts.apiKey}` && url.pathname !== '/health') {
      return json(401, { error: { message: 'Invalid API key.', type: 'authentication_error' } })
    }
    if (url.pathname === '/health') return json(200, opts.lmStudio ? { status: 'ok' } : { status: 'ok', app: 'RulZ LLM Suite' })
    if (url.pathname === '/v1/models') {
      return json(200, {
        object: 'list',
        data: models.map((m) =>
          opts.lmStudio
            ? { id: m.id, object: 'model', owned_by: 'organization_owner' }
            : {
                id: m.id,
                object: 'model',
                owned_by: 'rulz',
                path: m.path ?? `C:\\models\\${m.id}.gguf`,
                arch: 'qwen3moe',
                state: m.state ?? 'not-loaded',
                loaded_context_length: m.loaded_context_length,
                max_context_length: 262144,
                ...(m.rental ? { rental: m.rental } : {})
              }
        )
      })
    }
    // Like LLM Suite 0.35: a request with X-RulZ-Rental: deny gets a 409 instead of an answer from a rental's model.
    const named = url.pathname === '/props' ? url.searchParams.get('model') : null
    const refuseRental = (model: string | null | undefined): boolean => {
      const m = models.find((x) => x.id === model)
      if (!m || !(m.rental || /^vast\//i.test(m.path ?? '')) || req.headers['x-rulz-rental'] !== 'deny') return false
      json(409, { error: { message: `${m.id} runs on a rented computer, and this request does not allow rentals. Turn that on in RulZ Coding Agent, or choose a model on this computer.` } })
      return true
    }
    const modeLog = (path: string, model: string): void => {
      const mode = req.headers['x-rulz-mode']
      const rental = req.headers['x-rulz-rental']
      headerLog.push({ path, method: req.method ?? 'GET', model, ...(typeof mode === 'string' ? { mode } : {}), ...(typeof rental === 'string' ? { rental } : {}) })
    }
    const modelRoute = /^\/v1\/models\/([^/]+)\/(modes|reload)$/.exec(url.pathname)
    if (modelRoute && opts.modes !== undefined && !opts.lmStudio) {
      const model = decodeURIComponent(modelRoute[1])
      modeLog(url.pathname, model)
      if (modelRoute[2] === 'modes' && req.method === 'GET') {
        const r = typeof opts.modes === 'function' ? opts.modes(model) : { status: 200, body: opts.modes }
        return json(r.status, r.body)
      }
      if (modelRoute[2] === 'reload' && req.method === 'POST') {
        reloads.push({ model, body: JSON.parse(await readBody(req)) as Record<string, unknown> })
        if (opts.reloadRefusal) return json(409, { error: { message: opts.reloadRefusal } })
        const r = typeof opts.modes === 'function' ? opts.modes(model) : { status: 200, body: opts.modes }
        return json(r.status, r.body)
      }
    }
    const modeConfig = /^\/v1\/models\/([^/]+)\/modes\/(coding|re)\/load-config$/.exec(url.pathname)
    if (modeConfig && opts.modes !== undefined && opts.loadConfig && !opts.lmStudio) {
      const model = decodeURIComponent(modeConfig[1])
      const mode = modeConfig[2]
      let blocked: string | undefined
      if (req.method === 'POST') {
        const body = JSON.parse(await readBody(req)) as Record<string, unknown>
        if (!body.dry_run) {
          modeLoadConfigPosts.push({ mode, model, body })
          blocked = opts.reloadBlocked
        }
      }
      return json(200, {
        config: loadConfig,
        custom: true,
        max_context_length: 262144,
        state: 'ready',
        loaded_context_length: loadConfig.contextLength,
        gpus,
        mode,
        mode_own: modeLoadConfigPosts.some((p) => p.mode === mode && p.body.config !== null),
        ...(blocked ? { reload_blocked: blocked } : {})
      })
    }
    // (An LLM Suite older than 0.35 has no per-mode path: it falls through to the 404 below.)
    if (opts.loadConfig && url.pathname.startsWith('/v1/models/') && url.pathname.endsWith('/load-config') && !url.pathname.includes('/modes/')) {
      let config = loadConfig
      if (req.method === 'POST') {
        const body = JSON.parse(await readBody(req)) as { config: object; dry_run?: boolean }
        if (body.dry_run) config = { ...loadConfig, ...body.config }
        else {
          loadConfigPosts.push(body)
          Object.assign(loadConfig, body.config)
        }
      }
      const gpuBytes = 18 * GiB + (config.contextLength ?? 0) * 64 * 1024
      const shares = config.splitMode === 'custom' && config.tensorSplit ? config.tensorSplit.split(',').map(Number) : [50, 50]
      const total = shares.reduce((a, b) => a + b, 0) || 1
      const perGpu = gpus.map((g, i) => ({ index: g.index, name: g.name, share: shares[i] / total, bytes: (gpuBytes * shares[i]) / total, freeBytes: (g.memTotalMiB - g.memUsedMiB) * 1024 * 1024, totalBytes: g.memTotalMiB * 1024 * 1024 }))
      return json(200, {
        config,
        custom: loadConfigPosts.length > 0,
        max_context_length: 262144,
        state: 'ready',
        loaded_context_length: loadConfig.contextLength,
        has_draft: true,
        moe_layers: 48,
        gpus,
        estimate: { gpuBytes, totalBytes: gpuBytes, verdict: 'fits', perGpu }
      })
    }
    if (url.pathname === '/v1/re/overview' && !opts.lmStudio) {
      const asked = url.searchParams.get('model') ?? ''
      overviewQueries.push(asked)
      const perModel = opts.practicePerModel && asked
      // Per model, `xor-1` counts as solved for the model that is asked about (and failed once for any other).
      const stats = (id: string): { attempts: number; solved: number; failed?: number } =>
        perModel ? (id === 'xor-1' ? { attempts: 2, solved: 1, failed: 1 } : { attempts: 0, solved: 0, failed: 0 }) : { attempts: 0, solved: 0 }
      return json(200, {
        object: 're.overview',
        recording: true,
        sessions: [],
        practice: { kitDir: 'k', ...(perModel ? { model: asked } : {}), tasks: [
          { id: 'xor-1', family: 'xor', tier: 't1', binary: 'xor.exe', prompt: 'Key of {exe}?', ...stats('xor-1') },
          { id: 'xor-2', family: 'xor', tier: 't1', binary: 'xor.exe', prompt: 'Key of {exe}?', ...stats('xor-2') }
        ] },
        goals: [{ label: 'Overview', text: 'Analyze {target}.' }],
        clonePieces: [],
        cloneLanguages: ['']
      })
    }
    if (url.pathname === '/v1/re/kit' && req.method === 'POST' && !opts.lmStudio) {
      const body = JSON.parse(await readBody(req)) as { kitDir?: string }
      kits.push(body)
      return json(200, { kitDir: body.kitDir, tasks: [] })
    }
    const practice = /^\/v1\/re\/practice\/(xor-\d+)\/(prepare|grade)$/.exec(url.pathname)
    if (practice && practice[2] === 'prepare' && req.method === 'POST' && !opts.lmStudio) {
      const id = practice[1]
      return json(200, { target: `C:\\work\\${id}\\1\\xor.exe`, prompt: `What is the key in C:\\work\\${id}\\1\\xor.exe?`, family: 'xor', tier: 't1' })
    }
    if (practice && practice[2] === 'grade' && req.method === 'POST' && !opts.lmStudio) {
      const body = JSON.parse(await readBody(req)) as { text: string }
      const m = /ANSWER:\s*(\S+)/.exec(body.text)
      return json(200, { answer: m?.[1] ?? null, passed: m?.[1] === '42', expected: '42' })
    }
    if (url.pathname.startsWith('/v1/re/sessions/') && req.method === 'POST' && !opts.lmStudio) {
      const id = decodeURIComponent(url.pathname.slice('/v1/re/sessions/'.length))
      const body = JSON.parse(await readBody(req)) as Record<string, unknown>
      reSessions.push({ id, body })
      return json(200, { id, object: 're.session', recording: true, outcome: 'unrated', ...body })
    }
    if (url.pathname === '/props') {
      modeLog('/props', named ?? '')
      if (refuseRental(named)) return
      return json(200, { chat_template: opts.template ?? QWEN_CODER_TEMPLATE, default_generation_settings: { n_ctx: opts.nCtx ?? 65536 } })
    }
    if (url.pathname === '/v1/chat/completions' && req.method === 'POST') {
      const body = JSON.parse(await readBody(req)) as MockRequest['body']
      const reSession = req.headers['x-rulz-re-session']
      const mode = req.headers['x-rulz-mode']
      const rental = req.headers['x-rulz-rental']
      const record: MockRequest = {
        path: url.pathname,
        body,
        ...(typeof reSession === 'string' ? { reSession } : {}),
        ...(typeof mode === 'string' ? { mode } : {}),
        ...(typeof rental === 'string' ? { rental } : {})
      }
      requests.push(record)
      modeLog(url.pathname, body.model ?? '')
      if (refuseRental(body.model)) return
      // Like llama.cpp: earlier tool calls' arguments must parse as JSON, or the request fails.
      for (const m of body.messages ?? []) {
        for (const c of m.tool_calls ?? []) {
          try {
            JSON.parse(c.function.arguments)
          } catch {
            return json(500, { error: { code: 500, message: 'Failed to parse tool call arguments as JSON: [json.exception.parse_error.101] parse error', type: 'server_error' } })
          }
        }
      }
      if (!body.stream) return json(200, { choices: [{ message: { role: 'assistant', content: 'Summary: the owner asked for things and they were built.' } }] })
      const turn = nextTurn(record)
      if (turn.delay) await new Promise((r) => setTimeout(r, turn.delay))
      if (turn.status) return json(turn.status, { error: { message: turn.error ?? 'error' } })
      res.writeHead(200, { 'Content-Type': 'text/event-stream' })
      const send = (delta: unknown, finish: string | null = null): void => {
        const timings = finish && opts.timings ? { timings: opts.timings(record) } : {}
        res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: finish }], ...timings })}\n\n`)
      }
      const reasoning = turn.reasoning ?? ''
      for (let i = 0; i < reasoning.length; i += 7) send({ reasoning_content: reasoning.slice(i, i + 7) })
      if (turn.pause) await new Promise((r) => setTimeout(r, turn.pause))
      if (res.destroyed) return
      const content = turn.content ?? ''
      for (let i = 0; i < content.length; i += 7) send({ content: content.slice(i, i + 7) })
      turn.toolCalls?.forEach((c, index) => {
        const args = typeof c.args === 'string' ? c.args : JSON.stringify(c.args)
        const half = Math.floor(args.length / 2)
        send({ tool_calls: [{ index, id: `call_${turnIndex}_${index}`, type: 'function', function: { name: c.name, arguments: '' } }] })
        send({ tool_calls: [{ index, function: { arguments: args.slice(0, half) } }] })
        send({ tool_calls: [{ index, function: { arguments: args.slice(half) } }] })
      })
      send({}, turn.finish ?? (turn.toolCalls?.length ? 'tool_calls' : 'stop'))
      res.end('data: [DONE]\n\n')
      return
    }
    json(404, { error: { message: `Unknown endpoint ${url.pathname}` } })
  })
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r))
  const port = (server.address() as AddressInfo).port
  return {
    url: `http://127.0.0.1:${port}`,
    requests,
    loadConfigPosts,
    modeLoadConfigPosts,
    headerLog,
    reloads,
    overviewQueries,
    reSessions,
    kits,
    close: () => new Promise<void>((r) => server.close(() => r()))
  }
}
