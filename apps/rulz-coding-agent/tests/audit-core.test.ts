// Regression tests for the core audit findings (B10-B17, B20, B21).
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { createServer, type Server, type ServerResponse } from 'node:http'
import type { AddressInfo } from 'node:net'
import { tmpdir } from 'node:os'
import { delimiter, join } from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { editFile, globMatcher, readFile, searchText, writeFile } from '../src/main/agent/fileTools'
import { PathError, resolveInProject } from '../src/main/agent/pathGuard'
import { reTask } from '../src/main/agent/reTasks'
import { findProgram } from '../src/main/agent/syntaxCheck'
import { RulzClient, timeouts } from '../src/main/rulz/client'
import { ToolCallAccumulator } from '../src/main/rulz/sse'

let base: string
let root: string
beforeAll(() => {
  base = mkdtempSync(join(tmpdir(), 'rulz-agent-audit-core-'))
  root = join(base, 'project')
  mkdirSync(join(root, 'src', 'a'), { recursive: true })
  mkdirSync(join(root, 'src', 'b'), { recursive: true })
})
afterAll(() => rmSync(base, { recursive: true, force: true }))

describe('B10 reserved Windows names and streams when writing', () => {
  it.each(['nul', 'NUL', 'aux.txt', 'src/con', 'com1', 'lpt9.log', 'a.txt:hidden', 'src/x:y/z.txt', 'name.'])('refuses to write "%s"', async (p) => {
    await expect(resolveInProject(root, p, { write: true })).rejects.toBeInstanceOf(PathError)
  })
  it('still accepts ordinary names, and reading is unchanged', async () => {
    expect((await resolveInProject(root, 'src/console.txt', { write: true })).rel).toBe('src/console.txt')
    expect((await resolveInProject(root, 'nul')).rel).toBe('nul')
  })
  it('write_file reports it in plain words and creates nothing', async () => {
    const out = await writeFile(root, { path: 'nul', content: 'x' })
    expect(out.isError).toBe(true)
    expect(out.text).toMatch(/device/)
  })
})

describe('B11 / B16 stream and summary timeouts', () => {
  const servers: Server[] = []
  const open: ServerResponse[] = []
  afterAll(async () => {
    open.forEach((r) => r.destroy())
    await Promise.all(servers.map((s) => new Promise((r) => (s.closeAllConnections(), s.close(r)))))
  })
  async function serve(handler: (res: ServerResponse) => void): Promise<string> {
    const server = createServer((_req, res) => {
      open.push(res)
      handler(res)
    })
    await new Promise<void>((r) => server.listen(0, '127.0.0.1', r))
    servers.push(server)
    return `http://127.0.0.1:${(server.address() as AddressInfo).port}`
  }
  const body = { model: 'm', messages: [{ role: 'user' as const, content: 'hi' }] }

  it('streamTurn gives up when the server goes quiet, and that is not a Stop', async () => {
    const url = await serve((res) => {
      res.writeHead(200, { 'Content-Type': 'text/event-stream' })
      res.write('data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n')
    })
    const saved = timeouts.streamIdleMs
    timeouts.streamIdleMs = 150
    try {
      const ctl = new AbortController()
      const client = new RulzClient(() => ({ url, apiKey: '' }))
      await expect(client.streamTurn(body, ctl.signal)).rejects.toThrow(/stopped answering/)
      expect(ctl.signal.aborted).toBe(false)
    } finally {
      timeouts.streamIdleMs = saved
    }
  })

  it('streamTurn gives up when the server never starts answering, and that is not a Stop', async () => {
    const url = await serve(() => undefined)
    const saved = timeouts.startMs
    timeouts.startMs = 150
    try {
      const ctl = new AbortController()
      const client = new RulzClient(() => ({ url, apiKey: '' }))
      await expect(client.streamTurn(body, ctl.signal)).rejects.toThrow(/did not start answering/)
      expect(ctl.signal.aborted).toBe(false)
    } finally {
      timeouts.startMs = saved
    }
  })

  it('streamTurn still stops at once when the caller aborts', async () => {
    const url = await serve((res) => {
      res.writeHead(200, { 'Content-Type': 'text/event-stream' })
      res.write('data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n')
    })
    const ctl = new AbortController()
    const client = new RulzClient(() => ({ url, apiKey: '' }))
    const turn = client.streamTurn(body, ctl.signal)
    setTimeout(() => ctl.abort(), 100)
    await expect(turn).rejects.toThrow()
    expect(ctl.signal.aborted).toBe(true)
  })

  it('complete gives up after its total limit', async () => {
    const url = await serve(() => undefined)
    const saved = timeouts.completeMs
    timeouts.completeMs = 150
    try {
      const client = new RulzClient(() => ({ url, apiKey: '' }))
      await expect(client.complete('m', body.messages, 10)).rejects.toThrow(/stopped answering/)
      const ctl = new AbortController()
      const p = client.complete('m', body.messages, 10, ctl.signal)
      ctl.abort()
      await expect(p).rejects.not.toThrow(/stopped answering/)
    } finally {
      timeouts.completeMs = saved
    }
  })

  it('streamTurn closes the stream when an error chunk throws', async () => {
    let closed = false
    const url = await serve((res) => {
      res.on('close', () => (closed = true))
      res.writeHead(200, { 'Content-Type': 'text/event-stream' })
      res.write('data: {"error":{"message":"boom","code":500}}\n\n')
    })
    const client = new RulzClient(() => ({ url, apiKey: '' }))
    await expect(client.streamTurn(body, new AbortController().signal)).rejects.toThrow(/boom/)
    for (let i = 0; i < 40 && !closed; i++) await new Promise((r) => setTimeout(r, 25))
    expect(closed).toBe(true)
  })
})

describe('B21 repeated tool-call names', () => {
  it('does not double a name the server repeats in every delta', () => {
    const acc = new ToolCallAccumulator()
    acc.push([{ index: 0, id: 'c1', name: 'read_file', arguments: '{"pa' }])
    acc.push([{ index: 0, name: 'read_file', arguments: 'th":"a"}' }])
    expect(acc.result()[0].function.name).toBe('read_file')
    expect(acc.result()[0].function.arguments).toBe('{"path":"a"}')
  })
  it('still joins a name that arrives in pieces', () => {
    const acc = new ToolCallAccumulator()
    acc.push([{ index: 0, name: 'read_' }])
    acc.push([{ index: 0, name: 'file' }])
    expect(acc.result()[0].function.name).toBe('read_file')
  })
})

describe('B12 search_text says what it did not search', () => {
  it('counts big, binary and generated-folder files', async () => {
    const dir = join(root, 'srch')
    mkdirSync(join(dir, 'node_modules'), { recursive: true })
    writeFileSync(join(dir, 'big.txt'), 'x'.repeat(1024 * 1024 + 10))
    writeFileSync(join(dir, 'bin.dat'), Buffer.from([0, 1, 2, 0]))
    writeFileSync(join(dir, 'ok.txt'), 'hello\n')
    const out = await searchText(root, { query: 'zzz', path: 'srch' })
    expect(out.text).toMatch(/was not found/)
    expect(out.text).toMatch(/Not searched: 1 over/)
    expect(out.text).toMatch(/1 not text/)
    expect(out.text).toMatch(/1 generated folder/)
    const found = await searchText(root, { query: 'hello', path: 'srch' })
    expect(found.text).toMatch(/ok\.txt:1/)
    // A found answer tells only what could not be searched by mistake (the big file), not what is skipped on purpose.
    expect(found.text).toMatch(/Not searched: 1 over/)
    expect(found.text).not.toMatch(/not text|generated folder/)
  })
})

describe('B13 read_file caps characters', () => {
  it('cuts a huge single line and says how to go on', async () => {
    writeFileSync(join(root, 'min.js'), 'a'.repeat(2_000_000))
    const out = await readFile(root, { path: 'min.js' })
    expect(out.isError).toBe(false)
    expect(out.text.length).toBeLessThan(110_000)
    expect(out.text).toMatch(/was cut after 100,000/)
  })
  it('cuts many long lines at a line end and points at start_line', async () => {
    writeFileSync(join(root, 'wide.txt'), Array.from({ length: 100 }, () => 'b'.repeat(5000)).join('\n') + '\n')
    const out = await readFile(root, { path: 'wide.txt' })
    expect(out.text.length).toBeLessThan(110_000)
    expect(out.text).toMatch(/Cut at about 100,000 characters\.\)\n\(More below: read again with start_line 21\.\)/)
    const next = await readFile(root, { path: 'wide.txt', start_line: 21 })
    expect(next.text).toMatch(/lines 21-40 of 100/)
  })
})

describe('B14 glob with several brace groups', () => {
  it('expands every group', () => {
    const m = globMatcher('src/{a,b}/*.{js,ts}')
    expect(m('src/a/x.js')).toBe(true)
    expect(m('src/b/x.ts')).toBe(true)
    expect(m('src/c/x.js')).toBe(false)
    expect(m('src/a/x.py')).toBe(false)
  })
})

describe('B15 write_file keeps CRLF', () => {
  it('writes CRLF over a CRLF file when the new text has LF only', async () => {
    const f = join(root, 'crlf.txt')
    writeFileSync(f, 'one\r\ntwo\r\n')
    await writeFile(root, { path: 'crlf.txt', content: 'a\nb\nc\n' })
    expect(readFileSync(f, 'utf8')).toBe('a\r\nb\r\nc\r\n')
    writeFileSync(join(root, 'lf.txt'), 'one\ntwo\n')
    await writeFile(root, { path: 'lf.txt', content: 'a\nb\n' })
    expect(readFileSync(join(root, 'lf.txt'), 'utf8')).toBe('a\nb\n')
    expect((await editFile(root, { path: 'crlf.txt', old_text: 'b', new_text: 'B' })).isError).toBe(false)
  })
})

describe('B17 findProgram does not remember a miss', () => {
  it('finds a program installed after the first look', async () => {
    const bin = join(base, 'bin')
    mkdirSync(bin, { recursive: true })
    const env = { PATH: bin } as NodeJS.ProcessEnv
    const exe = process.platform === 'win32' ? 'fakeprog.exe' : 'fakeprog'
    expect(await findProgram('fakeprog', root, env)).toBeNull()
    writeFileSync(join(bin, exe), '')
    expect(existsSync(join(bin, exe))).toBe(true)
    expect(await findProgram('fakeprog', root, env)).toBe(join(bin, exe))
    void delimiter
  })
})

describe('B20 reTask target with replacement patterns', () => {
  it("keeps $& and $' in the target path literally", () => {
    const target = "C:\\tmp\\a$&b$'c$$d"
    const t = reTask({ kind: 'analyse', target, goal: 'Find the key in {target}' }, root)
    // In the goal itself, where {target} was (not only on the "Target:" line added when it is missing).
    expect(t.text).toContain(`Find the key in ${target}`)
    expect(t.text).not.toContain('Target: ')
  })
})
