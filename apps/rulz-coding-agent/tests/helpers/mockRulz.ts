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
  models?: { id: string; state?: 'loaded' | 'not-loaded'; loaded_context_length?: number }[]
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
    draftCacheTypeV: 'f16'
  }
  const GiB = 1024 ** 3
  const gpus = [
    { index: 0, name: 'RTX 4090', memUsedMiB: 1024, memTotalMiB: 24576 },
    { index: 1, name: 'RTX 3090', memUsedMiB: 512, memTotalMiB: 24576 }
  ]
  const loadConfigPosts: unknown[] = []
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
            : { id: m.id, object: 'model', owned_by: 'rulz', path: `C:\\models\\${m.id}.gguf`, arch: 'qwen3moe', state: m.state ?? 'not-loaded', loaded_context_length: m.loaded_context_length, max_context_length: 262144 }
        )
      })
    }
    if (opts.loadConfig && url.pathname.startsWith('/v1/models/') && url.pathname.endsWith('/load-config')) {
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
        gpus,
        estimate: { gpuBytes, totalBytes: gpuBytes, verdict: 'fits', perGpu }
      })
    }
    if (url.pathname === '/props') {
      return json(200, { chat_template: opts.template ?? QWEN_CODER_TEMPLATE, default_generation_settings: { n_ctx: opts.nCtx ?? 65536 } })
    }
    if (url.pathname === '/v1/chat/completions' && req.method === 'POST') {
      const body = JSON.parse(await readBody(req)) as MockRequest['body']
      const record: MockRequest = { path: url.pathname, body }
      requests.push(record)
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
    close: () => new Promise<void>((r) => server.close(() => r()))
  }
}
