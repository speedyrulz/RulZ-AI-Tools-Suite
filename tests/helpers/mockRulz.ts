// A stand-in for RulZ LLM Suite's API server: scripted model turns, streamed like llama.cpp.
import { createServer, type IncomingMessage } from 'node:http'
import type { AddressInfo } from 'node:net'

export interface MockTurn {
  content?: string
  toolCalls?: { name: string; args: unknown }[]
  finish?: string
  /** Answer with this HTTP status and error instead. */
  status?: number
  error?: string
  /** Wait before answering (ms). */
  delay?: number
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
  const models = opts.models ?? [{ id: 'Qwen3-Coder-30B-A3B-Instruct-Q5_K_M', state: 'loaded', loaded_context_length: 65536 }]
  let turnIndex = 0
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
        res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`)
      }
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
    close: () => new Promise<void>((r) => server.close(() => r()))
  }
}
