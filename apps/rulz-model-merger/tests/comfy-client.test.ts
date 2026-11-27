import http from 'node:http'
import type { AddressInfo } from 'node:net'
import { afterEach, describe, expect, it } from 'vitest'
import { WebSocketServer, type WebSocket as ServerSocket } from 'ws'
import type { GpuInfo } from '@shared/types'
import { ComfyClient, ComfyError, probeServer, type HistoryImage } from '../src/main/comfy/client'

interface Seen {
  method: string
  url: string
  body: unknown
}
type Handler = (req: http.IncomingMessage, res: http.ServerResponse, body: unknown) => void

const closers: Array<() => Promise<void>> = []
afterEach(async () => {
  while (closers.length) await closers.pop()!()
})

async function startServer(handler: Handler): Promise<{ url: string; seen: Seen[]; server: http.Server }> {
  const seen: Seen[] = []
  const server = http.createServer((req, res) => {
    const chunks: Buffer[] = []
    req.on('data', (c: Buffer) => chunks.push(c))
    req.on('end', () => {
      const text = Buffer.concat(chunks).toString('utf8')
      let body: unknown = text
      try {
        body = text ? JSON.parse(text) : undefined
      } catch {
        /* keep text */
      }
      seen.push({ method: req.method ?? '', url: req.url ?? '', body })
      handler(req, res, body)
    })
  })
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r))
  closers.push(
    () =>
      new Promise<void>((r) => {
        server.closeAllConnections()
        server.close(() => r())
      })
  )
  return { url: `http://127.0.0.1:${(server.address() as AddressInfo).port}`, seen, server }
}

function json(res: http.ServerResponse, data: unknown, status = 200): void {
  res.writeHead(status, { 'Content-Type': 'application/json' })
  res.end(JSON.stringify(data))
}

const gpus: GpuInfo[] = [
  { index: 0, name: 'NVIDIA GeForce RTX 5060 Ti', busId: '', memoryUsedMiB: 0, memoryTotalMiB: 16000, utilizationPct: 0, temperatureC: null }
]

describe('ComfyClient HTTP', () => {
  it('normalizes the base url and reads system stats', async () => {
    const s = await startServer((_req, res) =>
      json(res, { system: { comfyui_version: '0.46.0', argv: ['main.py', '--cuda-device', '0'] }, devices: [{ name: 'cuda:0 X' }] })
    )
    const c = new ComfyClient(`${s.url}/`)
    expect(c.baseUrl).toBe(s.url)
    const info = await c.systemStats()
    expect(info.version).toBe('0.46.0')
    expect(info.cudaDevice).toBe('0')
    expect(s.seen[0]).toMatchObject({ method: 'GET', url: '/system_stats' })
  })

  it('reads capabilities from /object_info', async () => {
    const s = await startServer((_req, res) =>
      json(res, { UNETLoader: { input: { required: { unet_name: ['COMBO', { options: ['m.safetensors'] }] } } }, ModelMergeSimple: {} })
    )
    const caps = await new ComfyClient(s.url).capabilities()
    expect(caps.diffusionModels).toEqual(['m.safetensors'])
    expect(caps.nodes.ModelMergeSimple).toBe(true)
    expect(s.seen[0].url).toBe('/object_info')
  })

  it('posts the prompt with client_id and returns the id', async () => {
    const s = await startServer((_req, res) => json(res, { prompt_id: 'abc', number: 7, node_errors: {} }))
    const r = await new ComfyClient(s.url).queuePrompt({ '1': { class_type: 'KSampler', inputs: {} } }, 'client-1')
    expect(r).toEqual({ promptId: 'abc', number: 7 })
    expect(s.seen[0]).toMatchObject({ method: 'POST', url: '/prompt', body: { prompt: { '1': { class_type: 'KSampler', inputs: {} } }, client_id: 'client-1' } })
  })

  it('turns an HTTP 400 into a ComfyError with node errors and a readable message', async () => {
    const nodeErrors = { '5': { class_type: 'LoraLoader', errors: [{ message: 'Value not in list', details: 'lora_name: nope.safetensors not in []' }] } }
    const s = await startServer((_req, res) =>
      json(res, { error: { type: 'prompt_outputs_failed_validation', message: 'Prompt outputs failed validation', details: '' }, node_errors: nodeErrors }, 400)
    )
    const err = await new ComfyClient(s.url).queuePrompt({}, 'c').catch((e) => e)
    expect(err).toBeInstanceOf(ComfyError)
    expect(err.status).toBe(400)
    expect(err.nodeErrors).toEqual(nodeErrors)
    expect(err.message).toContain('Prompt outputs failed validation')
    expect(err.message).toContain('node 5 (LoraLoader): Value not in list')
  })

  it('handles a 400 without JSON and other HTTP errors', async () => {
    let status = 400
    const s = await startServer((_req, res) => {
      res.writeHead(status)
      res.end('oops')
    })
    const c = new ComfyClient(s.url)
    const e400 = await c.queuePrompt({}, 'c').catch((e) => e)
    expect(e400).toBeInstanceOf(ComfyError)
    expect(e400.status).toBe(400)
    status = 500
    const e500 = await c.queuePrompt({}, 'c').catch((e) => e)
    expect(e500.status).toBe(500)
    expect(e500.message).toContain(s.url)
    expect(e500.message).toContain('500')
  })

  it('turns a connection failure into a ComfyError naming the URL', async () => {
    const s = await startServer((_req, res) => json(res, {}))
    const url = s.url
    await closers.pop()!()
    const err = await new ComfyClient(url).systemStats().catch((e) => e)
    expect(err).toBeInstanceOf(ComfyError)
    expect(err.message).toContain(url)
  })

  it('times out requests that never answer', async () => {
    const s = await startServer(() => {
      /* never answer */
    })
    const err = await new ComfyClient(s.url, { timeoutMs: 100 }).systemStats().catch((e) => e)
    expect(err).toBeInstanceOf(ComfyError)
    expect(err.message).toMatch(/No answer from/)
    expect(err.message).toContain(s.url)
  })

  it('rejects a non-JSON answer', async () => {
    const s = await startServer((_req, res) => {
      res.writeHead(200)
      res.end('<html>')
    })
    const err = await new ComfyClient(s.url).systemStats().catch((e) => e)
    expect(err).toBeInstanceOf(ComfyError)
    expect(err.message).toMatch(/JSON/)
  })

  it('times out when the headers arrive but the body stalls', async () => {
    const s = await startServer((_req, res) => {
      res.writeHead(200, { 'Content-Type': 'application/json', 'Content-Length': '1000' })
      res.write('{"system":')
    })
    const started = Date.now()
    const err = await new ComfyClient(s.url, { timeoutMs: 150 }).systemStats().catch((e) => e)
    expect(err).toBeInstanceOf(ComfyError)
    expect(err.message).toMatch(/No answer from/)
    expect(Date.now() - started).toBeLessThan(3000)
    const img = await new ComfyClient(s.url, { timeoutMs: 150 }).viewImage({ filename: 'a.png', subfolder: '', type: 'output' }).catch((e) => e)
    expect(img).toBeInstanceOf(ComfyError)
    expect(img.message).toMatch(/No answer from/)
    const post = await new ComfyClient(s.url, { timeoutMs: 150 }).queuePrompt({}, 'c').catch((e) => e)
    expect(post).toBeInstanceOf(ComfyError)
    expect(post.message).toMatch(/No answer from/)
  })

  it('tells a dropped connection mid-body from a body that is not JSON', async () => {
    const s = await startServer((req, res) => {
      if (req.url === '/object_info') {
        res.writeHead(200, { 'Content-Type': 'application/json', 'Content-Length': '1000' })
        res.write('{"a":')
        setTimeout(() => res.socket?.destroy(), 20)
      } else {
        res.writeHead(200)
        res.end('<html>not json</html>')
      }
    })
    const c = new ComfyClient(s.url)
    const dropped = await c.objectInfo().catch((e) => e)
    expect(dropped).toBeInstanceOf(ComfyError)
    expect(dropped.message).toMatch(/Could not reach/)
    expect(dropped.message).not.toMatch(/did not return JSON/)
    const notJson = await c.systemStats().catch((e) => e)
    expect(notJson).toBeInstanceOf(ComfyError)
    expect(notJson.message).toMatch(/did not return JSON/)
    const img = await c.viewImage({ filename: 'a.png', subfolder: '', type: 'output' })
    expect(img.toString()).toContain('not json')
  })

  it('uses an injected fetch', async () => {
    const calls: string[] = []
    const fake = (async (input: string | URL | Request) => {
      calls.push(String(input))
      return new Response(JSON.stringify({ queue_running: [[0, 'r1', {}, {}, []]], queue_pending: [[1, 'p1', {}, {}, []], [2, 'p2']] }), { status: 200 })
    }) as typeof fetch
    const q = await new ComfyClient('localhost:8188', { fetch: fake }).queue()
    expect(calls).toEqual(['http://localhost:8188/queue'])
    expect(q).toEqual({ running: ['r1'], pending: ['p1', 'p2'] })
  })

  it('parses a completed history entry with images', async () => {
    const s = await startServer((_req, res) =>
      json(res, {
        p1: {
          outputs: { '9': { images: [{ filename: 'a_00001_.png', subfolder: 'sub', type: 'output' }, { subfolder: '' }, { filename: 'b.png' }] }, '10': { text: ['x'] } },
          status: { status_str: 'success', completed: true, messages: [['execution_start', {}], ['execution_success', { prompt_id: 'p1' }]] }
        }
      })
    )
    const h = await new ComfyClient(s.url).history('p1')
    expect(s.seen[0].url).toBe('/history/p1')
    expect(h).toEqual({
      promptId: 'p1',
      completed: true,
      statusStr: 'success',
      images: [
        { nodeId: '9', image: { filename: 'a_00001_.png', subfolder: 'sub', type: 'output' } },
        { nodeId: '9', image: { filename: 'b.png', subfolder: '', type: 'output' } }
      ],
      error: null
    })
  })

  it('parses a failed history entry and an interrupted one', async () => {
    const s = await startServer((req, res) => {
      if (req.url === '/history/bad') {
        json(res, {
          bad: { outputs: {}, status: { status_str: 'error', completed: false, messages: [['execution_error', { node_type: 'KSampler', exception_message: 'out of memory\n' }]] } }
        })
      } else if (req.url === '/history/int') {
        json(res, { int: { outputs: {}, status: { status_str: 'error', completed: false, messages: [['execution_interrupted', {}]] } } })
      } else if (req.url === '/history/odd') {
        json(res, { odd: { outputs: {}, status: { status_str: 'error', completed: false } } })
      } else json(res, {})
    })
    const c = new ComfyClient(s.url)
    const bad = await c.history('bad')
    expect(bad).toMatchObject({ completed: false, statusStr: 'error', error: 'KSampler: out of memory' })
    expect((await c.history('int'))?.error).toBe('Interrupted')
    expect((await c.history('odd'))?.error).toMatch(/error/)
    expect(await c.history('missing')).toBeNull()
  })

  it('reads the queue as prompt ids', async () => {
    const s = await startServer((_req, res) => json(res, { queue_running: [[3, 'run-1', {}, {}, ['9']]], queue_pending: [[4, 'pend-1', {}, {}, []]] }))
    expect(await new ComfyClient(s.url).queue()).toEqual({ running: ['run-1'], pending: ['pend-1'] })
    expect(s.seen[0]).toMatchObject({ method: 'GET', url: '/queue' })
  })

  it('sends the delete, interrupt and free bodies ComfyUI expects', async () => {
    const s = await startServer((_req, res) => {
      res.writeHead(200)
      res.end()
    })
    const c = new ComfyClient(s.url)
    await c.deletePending(['a', 'b'])
    await c.deletePending([])
    await c.interrupt()
    await c.interrupt('xyz')
    await c.free({ unloadModels: true, freeMemory: false })
    expect(s.seen.map((x) => [x.method, x.url, x.body])).toEqual([
      ['POST', '/queue', { delete: ['a', 'b'] }],
      ['POST', '/interrupt', {}],
      ['POST', '/interrupt', { prompt_id: 'xyz' }],
      ['POST', '/free', { unload_models: true, free_memory: false }]
    ])
  })

  it('fetches an image with an encoded query', async () => {
    const png = Buffer.from([137, 80, 78, 71, 1, 2, 3])
    const s = await startServer((_req, res) => {
      res.writeHead(200, { 'Content-Type': 'image/png' })
      res.end(png)
    })
    const img: HistoryImage = { filename: 'a b&c.png', subfolder: 'x/y', type: 'output' }
    const buf = await new ComfyClient(s.url).viewImage(img)
    expect(buf.equals(png)).toBe(true)
    const u = new URL(s.seen[0].url, 'http://x')
    expect(u.pathname).toBe('/view')
    expect(u.searchParams.get('filename')).toBe('a b&c.png')
    expect(u.searchParams.get('subfolder')).toBe('x/y')
    expect(u.searchParams.get('type')).toBe('output')
  })
})

describe('probeServer', () => {
  it('describes a ComfyUI server', async () => {
    const s = await startServer((_req, res) =>
      json(res, { system: { comfyui_version: '0.46.0', argv: ['--cuda-device', '1'] }, devices: [{ name: 'cuda:0 NVIDIA GeForce RTX 5060 Ti : cudaMallocAsync' }] })
    )
    const r = await probeServer(s.url, gpus)
    expect(r).toEqual({
      url: s.url,
      port: Number(new URL(s.url).port),
      version: '0.46.0',
      devices: ['cuda:0 NVIDIA GeForce RTX 5060 Ti : cudaMallocAsync'],
      cudaDevice: '1',
      gpuIndexes: [0],
      own: false
    })
  })

  it('returns null for something that is not ComfyUI, unreachable, slow, or a bad url', async () => {
    const other = await startServer((_req, res) => json(res, { hello: 'world' }))
    expect(await probeServer(other.url, gpus)).toBeNull()
    const err = await startServer((_req, res) => json(res, {}, 404))
    expect(await probeServer(err.url, gpus)).toBeNull()
    const slow = await startServer(() => {
      /* never answer */
    })
    expect(await probeServer(slow.url, gpus, { timeoutMs: 100 })).toBeNull()
    const dead = await startServer((_req, res) => json(res, {}))
    const deadUrl = dead.url
    await closers.pop()!()
    expect(await probeServer(deadUrl, gpus)).toBeNull()
    expect(await probeServer('ftp://nope', gpus)).toBeNull()
  })
})

describe('ComfyClient events', () => {
  async function startWs(): Promise<{ url: string; wss: WebSocketServer; sockets: Promise<{ socket: ServerSocket; url: string }> }> {
    const server = http.createServer()
    const wss = new WebSocketServer({ server })
    await new Promise<void>((r) => server.listen(0, '127.0.0.1', r))
    const sockets = new Promise<{ socket: ServerSocket; url: string }>((resolve) => wss.on('connection', (socket, req) => resolve({ socket, url: req.url ?? '' })))
    closers.push(
      () =>
        new Promise<void>((r) => {
          for (const c of wss.clients) c.terminate()
          wss.close(() => server.close(() => r()))
        })
    )
    return { url: `http://127.0.0.1:${(server.address() as AddressInfo).port}`, wss, sockets }
  }

  it('routes messages, ignores binary and malformed frames, and reports close', async () => {
    const w = await startWs()
    const log: unknown[] = []
    const closed = new Promise<void>((r) => {
      const events = new ComfyClient(w.url).openEvents('my client', {
        onStatus: (n) => log.push(['status', n]),
        onProgress: (p) => log.push(['progress', p]),
        onExecuting: (p) => log.push(['executing', p]),
        onExecuted: (p) => log.push(['executed', p]),
        onSuccess: (id) => {
          log.push(['success', id])
          throw new Error('handler failure must not break the stream')
        },
        onError: (p) => log.push(['error', p]),
        onClose: () => {
          log.push(['close'])
          r()
        }
      })
      void w.sockets.then(({ socket, url }) => {
        log.push(['url', url])
        const send = (type: string, data: unknown): void => socket.send(JSON.stringify({ type, data }))
        send('status', { status: { exec_info: { queue_remaining: 2 } }, sid: 'x' })
        socket.send(Buffer.from([0, 0, 0, 1, 0, 0, 0, 1, 255, 216]), { binary: true })
        socket.send('{not json')
        socket.send('42')
        socket.send(JSON.stringify({ type: 'unknown_type', data: {} }))
        send('progress', { value: 3, max: 20, prompt_id: 'p', node: '4' })
        send('progress', { value: 'x', max: 20 })
        send('executing', { node: '4', prompt_id: 'p' })
        send('executing', { node: null, prompt_id: 'p' })
        send('executed', { node: '9', prompt_id: 'p', output: { images: [{ filename: 'f.png', subfolder: '', type: 'temp' }] } })
        send('execution_success', { prompt_id: 'p' })
        send('execution_error', { prompt_id: 'p', node_id: '4', exception_message: 'boom' })
        send('execution_interrupted', { prompt_id: 'p', node_id: '4' })
        send('status', { status: { exec_info: { queue_remaining: 0 } } })
        setTimeout(() => {
          events.close()
        }, 150)
      })
    })
    await closed
    expect(log).toEqual([
      ['url', '/ws?clientId=my%20client'],
      ['status', 2],
      ['progress', { promptId: 'p', node: '4', value: 3, max: 20 }],
      ['executing', { promptId: 'p', node: '4' }],
      ['executing', { promptId: 'p', node: null }],
      ['executed', { promptId: 'p', node: '9', images: [{ filename: 'f.png', subfolder: '', type: 'temp' }] }],
      ['success', 'p'],
      ['error', { promptId: 'p', message: 'boom', nodeId: '4' }],
      ['error', { promptId: 'p', message: 'Interrupted', nodeId: '4' }],
      ['status', 0],
      ['close']
    ])
  })

  it('ignores messages after close()', async () => {
    const w = await startWs()
    const log: unknown[] = []
    const events = new ComfyClient(w.url).openEvents('c', { onStatus: (n) => log.push(['status', n]), onClose: () => log.push(['close']) })
    const { socket } = await w.sockets
    socket.send(JSON.stringify({ type: 'status', data: { status: { exec_info: { queue_remaining: 1 } } } }))
    await new Promise((r) => setTimeout(r, 100))
    events.close()
    socket.send(JSON.stringify({ type: 'status', data: { status: { exec_info: { queue_remaining: 2 } } } }))
    await new Promise((r) => setTimeout(r, 150))
    expect(log).toEqual([['status', 1], ['close']])
  })

  it('close() while still connecting terminates quietly and fires onClose once', async () => {
    const w = await startWs()
    let closes = 0
    const events = new ComfyClient(w.url).openEvents('c', { onClose: () => closes++ })
    events.close()
    events.close()
    await new Promise((r) => setTimeout(r, 150))
    expect(closes).toBe(1)
  })

  it('calls onClose once when the server drops the connection', async () => {
    const w = await startWs()
    let closes = 0
    const done = new Promise<void>((r) => {
      new ComfyClient(w.url).openEvents('c', {
        onClose: () => {
          closes++
          setTimeout(r, 50)
        }
      })
      void w.sockets.then(({ socket }) => socket.close())
    })
    await done
    expect(closes).toBe(1)
  })

  it('reports close when the server is not there', async () => {
    const s = await startServer((_req, res) => json(res, {}))
    const url = s.url
    await closers.pop()!()
    await new Promise<void>((r) => {
      new ComfyClient(url).openEvents('c', { onClose: r })
    })
  })
})
