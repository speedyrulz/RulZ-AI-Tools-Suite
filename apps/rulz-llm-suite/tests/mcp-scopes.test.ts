// MCP scopes: a private REA process per practice lane (src/main/mcp/scopes.ts and McpManager.openScope).
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import type { ToolResult } from '@shared/types'
import { McpManager } from '../src/main/mcp/manager'
import { McpScope, SCOPE_CALL_TIMEOUT_MS, SCOPES_OPEN_MESSAGE, type ScopedProcess } from '../src/main/mcp/scopes'

// ---------- the scope on its own, with fake processes ----------

function fakeProc(id: number, log: string[]): ScopedProcess & { alive_: boolean; dieOnCall?: boolean; timeouts: number[] } {
  const p = {
    alive_: true,
    timeouts: [] as number[],
    dieOnCall: false,
    async callTool(name: string, args: Record<string, unknown>, opts: { timeout: number }) {
      p.timeouts.push(opts.timeout)
      log.push(`p${id}:${name}`)
      if (p.dieOnCall) {
        p.alive_ = false
        throw new Error('Connection closed')
      }
      return { content: [{ type: 'text', text: `p${id} ${name} ${JSON.stringify(args)}` }] }
    },
    pid: () => 1000 + id,
    alive: () => p.alive_,
    stderrTail: () => ['it crashed'],
    close: async () => {
      p.alive_ = false
      log.push(`close p${id}`)
    }
  }
  return p
}

function scopeWith(o: { onClose?: () => void } = {}) {
  const log: string[] = []
  const procs: ReturnType<typeof fakeProc>[] = []
  const sharedCalls: string[] = []
  const scope = new McpScope({
    scoped: new Set(['rea']),
    locate: (q) => (q.startsWith('rea__') ? { server: 'rea', tool: q.slice(5) } : q.startsWith('calc__') ? { server: 'calc', tool: q.slice(6) } : undefined),
    shared: async (call): Promise<ToolResult> => {
      sharedCalls.push(call.name)
      return { text: `shared ${call.name}`, isError: false }
    },
    start: async () => {
      const p = fakeProc(procs.length + 1, log)
      procs.push(p)
      log.push(`start p${p.pid()! - 1000}`)
      return p
    },
    onClose: o.onClose ?? (() => undefined)
  })
  return { scope, log, procs, sharedCalls }
}

describe('McpScope', () => {
  it('starts its process at the first call that needs it, and sends everything else to the shared connection', async () => {
    const { scope, log, sharedCalls } = scopeWith()
    expect(log).toEqual([])
    expect(await scope.callTool({ name: 'calc__calc', arguments: '{"x":1}' })).toEqual({ text: 'shared calc__calc', isError: false })
    expect(await scope.callTool({ name: 'unknown_tool', arguments: '' })).toEqual({ text: 'shared unknown_tool', isError: false })
    expect(sharedCalls).toEqual(['calc__calc', 'unknown_tool'])
    expect(log).toEqual([])
    // The tool's own name is sent to the process, not the qualified one.
    expect(await scope.callTool({ name: 'rea__open_binary', arguments: '{"path":"a"}' })).toEqual({ text: 'p1 open_binary {"path":"a"}', isError: false })
    await scope.callTool({ name: 'rea__xrefs', arguments: '' })
    expect(log).toEqual(['start p1', 'p1:open_binary', 'p1:xrefs'])
  })

  it('gives calls 300 seconds', async () => {
    const { scope, procs } = scopeWith()
    await scope.callTool({ name: 'rea__xrefs', arguments: '' })
    expect(procs[0].timeouts).toEqual([SCOPE_CALL_TIMEOUT_MS])
    expect(SCOPE_CALL_TIMEOUT_MS).toBe(300_000)
  })

  it('starts a fresh process for each task, so nothing open is carried over', async () => {
    const { scope, log } = scopeWith()
    await scope.callTool({ name: 'rea__open_binary', arguments: '' })
    scope.beginTask()
    await scope.callTool({ name: 'rea__open_binary', arguments: '' })
    await Promise.resolve()
    expect(log).toEqual(['start p1', 'p1:open_binary', 'close p1', 'start p2', 'p2:open_binary'])
  })

  it('reports a process that died under a call and starts a new one on the next', async () => {
    const { scope, procs } = scopeWith()
    await scope.callTool({ name: 'rea__open_binary', arguments: '' })
    procs[0].dieOnCall = true
    const r = await scope.callTool({ name: 'rea__xrefs', arguments: '' })
    expect(r.isError).toBe(true)
    expect(r.text).toMatch(/the rea process stopped \(Connection closed\): it crashed\. It starts again on the next call, with nothing open in it/)
    const next = await scope.callTool({ name: 'rea__xrefs', arguments: '' })
    expect(next.text).toBe('p2 xrefs {}')
    expect(procs).toHaveLength(2)
  })

  it('starts one process when two calls arrive together', async () => {
    const { scope, procs } = scopeWith()
    await Promise.all([scope.callTool({ name: 'rea__a', arguments: '' }), scope.callTool({ name: 'rea__b', arguments: '' })])
    expect(procs).toHaveLength(1)
  })

  it('refuses bad arguments and stopped calls without starting anything', async () => {
    const { scope, log } = scopeWith()
    expect((await scope.callTool({ name: 'rea__a', arguments: '{nope' })).text).toMatch(/not valid JSON/)
    const ctrl = new AbortController()
    ctrl.abort()
    expect(await scope.callTool({ name: 'rea__a', arguments: '' }, ctrl.signal)).toEqual({ text: 'The tool call was stopped.', isError: true })
    expect(log).toEqual([])
  })

  it('says why a process could not start', async () => {
    const scope = new McpScope({
      scoped: new Set(['rea']),
      locate: () => ({ server: 'rea', tool: 'x' }),
      shared: async () => ({ text: '', isError: false }),
      start: async () => Promise.reject(new Error('spawn rea ENOENT')),
      onClose: () => undefined
    })
    const r = await scope.callTool({ name: 'rea__x', arguments: '' })
    expect(r).toEqual({ text: 'Tool call failed: could not start rea for this task: spawn rea ENOENT', isError: true })
  })

  it('closes its processes, tells its owner once, and fails calls afterwards', async () => {
    let closed = 0
    const { scope, log } = scopeWith({ onClose: () => void closed++ })
    await scope.callTool({ name: 'rea__a', arguments: '' })
    await scope.close()
    await scope.close()
    expect(closed).toBe(1)
    expect(log).toContain('close p1')
    expect((await scope.callTool({ name: 'rea__a', arguments: '' })).text).toBe('The practice run has ended.')
  })
})

// ---------- McpManager with real stdio servers ----------

const root = mkdtempSync(join(tmpdir(), 'rulz-scopes-'))
afterAll(() => rmSync(root, { recursive: true, force: true }))

const sdk = resolve('node_modules/@modelcontextprotocol/sdk/dist/cjs').replaceAll('\\', '/')
const serverScript = (tool: string): string => `
const { McpServer } = require('${sdk}/server/mcp.js')
const { StdioServerTransport } = require('${sdk}/server/stdio.js')
const s = new McpServer({ name: 't', version: '1' })
s.tool('${tool}', 'Tells which process answered', async () => ({ content: [{ type: 'text', text: String(process.pid) }] }))
s.connect(new StdioServerTransport())
`

function setup(servers: Record<string, unknown>, onReaTaken?: () => void): McpManager {
  const file = join(root, `mcp-${Math.random().toString(36).slice(2)}.json`)
  writeFileSync(file, JSON.stringify({ mcpServers: servers }))
  return new McpManager(file, () => undefined, undefined, undefined, onReaTaken)
}

describe('McpManager.openScope', () => {
  const rea = join(root, 'rea.cjs')
  const other = join(root, 'other.cjs')
  writeFileSync(rea, serverScript('whoami'))
  writeFileSync(other, serverScript('hello'))
  const stdio = (script: string) => ({ command: process.execPath, args: [script] })

  it('gives each scope its own REA process, with the same tools, and shares everything else', async () => {
    const mgr = setup({ rea: stdio(rea), other: stdio(other) })
    await mgr.init()
    try {
      expect(mgr.status().find((s) => s.name === 'rea')?.state).toBe('connected')
      const shared = await mgr.callTool({ name: 'rea__whoami', arguments: '{}' })
      const a = mgr.openScope()
      const b = mgr.openScope()
      const pa = await a.callTool({ name: 'rea__whoami', arguments: '{}' })
      const pb = await b.callTool({ name: 'rea__whoami', arguments: '{}' })
      expect(new Set([shared.text, pa.text, pb.text]).size).toBe(3)
      // The same scope keeps its process between calls; a new task starts a new one.
      expect((await a.callTool({ name: 'rea__whoami', arguments: '{}' })).text).toBe(pa.text)
      a.beginTask()
      const pa2 = await a.callTool({ name: 'rea__whoami', arguments: '{}' })
      expect(pa2.text).not.toBe(pa.text)
      // Another server is the shared connection's.
      const o1 = await a.callTool({ name: 'other__hello', arguments: '{}' })
      const o2 = await b.callTool({ name: 'other__hello', arguments: '{}' })
      expect(o1.text).toBe(o2.text)
      expect(o1.text).toBe((await mgr.callTool({ name: 'other__hello', arguments: '{}' })).text)
      await a.close()
      await b.close()
    } finally {
      await mgr.dispose()
    }
  }, 30_000)

  it('refuses to save or restart the config while a scope is open', async () => {
    const mgr = setup({ rea: stdio(rea) })
    await mgr.init()
    try {
      const scope = mgr.openScope()
      await expect(mgr.saveConfig('{"mcpServers":{}}')).rejects.toThrow(SCOPES_OPEN_MESSAGE)
      await expect(mgr.restart(null)).rejects.toThrow('A practice run is using the MCP servers; stop it first.')
      await scope.close()
      await expect(mgr.restart('rea')).resolves.toBeUndefined()
    } finally {
      await mgr.dispose()
    }
  }, 30_000)

  it('says why it cannot open a scope: REA missing, not connected, or an HTTP server', async () => {
    const none = setup({})
    await none.init()
    expect(() => none.openScope()).toThrow(/REA is not connected/)
    await none.dispose()
    const broken = setup({ rea: { command: join(root, 'no-such-program.exe') } })
    await broken.init()
    expect(() => broken.openScope()).toThrow(/REA is not connected/)
    await broken.dispose()
    const http = setup({ rea: { url: 'http://127.0.0.1:9/mcp' } })
    await http.init()
    expect(() => http.openScope()).toThrow(/"rea" to be a local program.*HTTP server/)
    await http.dispose()
  }, 30_000)

  it('lets a lane that has REA to itself fall back to the shared connection when REA is an HTTP server or is missing', async () => {
    let taken = 0
    const http = setup({ rea: { url: 'http://127.0.0.1:9/mcp' }, other: stdio(other) }, () => taken++)
    await http.init()
    try {
      expect(() => http.openScope()).toThrow(/HTTP server/)
      const alone = http.openScope({ alone: true })
      expect((await alone.callTool({ name: 'other__hello', arguments: '{}' })).isError).toBe(false)
      // Shared, not private: the shared connection answers for REA's tools (here: it has none, so it says so).
      expect((await alone.callTool({ name: 'rea__whoami', arguments: '{}' })).text).toMatch(/not available/)
      // Opening a program there is reported before and after, so a chat does not take it for its own.
      expect(taken).toBe(0)
      await alone.callTool({ name: 'rea__open_binary', arguments: '{}' })
      expect(taken).toBe(2)
      await alone.close()
    } finally {
      await http.dispose()
    }
    const none = setup({ other: stdio(other) })
    await none.init()
    try {
      expect(() => none.openScope()).toThrow(/REA is not connected/)
      const alone = none.openScope({ alone: true })
      expect((await alone.callTool({ name: 'other__hello', arguments: '{}' })).isError).toBe(false)
      await alone.close()
    } finally {
      await none.dispose()
    }
  }, 30_000)

  it('kills the scope processes when the manager is disposed', async () => {
    const mgr = setup({ rea: stdio(rea) })
    await mgr.init()
    const scope = mgr.openScope()
    const pid = Number((await scope.callTool({ name: 'rea__whoami', arguments: '{}' })).text)
    const alive = (): boolean => {
      try {
        process.kill(pid, 0)
        return true
      } catch {
        return false
      }
    }
    expect(alive()).toBe(true)
    await mgr.dispose()
    // The tree is ended without waiting (taskkill on Windows); give the OS a moment.
    for (let i = 0; i < 40 && alive(); i++) await new Promise((r) => setTimeout(r, 100))
    // taskkill /T is what ends the tree; elsewhere the polite close does, which the dispose does not wait for.
    if (process.platform === 'win32') expect(alive()).toBe(false)
    expect(() => mgr.openScope()).toThrow(/closing/)
  }, 30_000)
})
