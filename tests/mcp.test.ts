import { describe, expect, it } from 'vitest'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { McpConfigState, WireMessage } from '@shared/types'
import { pairToolMessages } from '../src/main/chat/context'
import { interpretChunk, ToolCallAccumulator } from '../src/main/chat/sse'
import { contentToText, EMPTY_CONFIG, parseMcpConfig, qualifyToolName, readMcpConfigFile } from '../src/main/mcp/config'
import { McpManager } from '../src/main/mcp/manager'

describe('parseMcpConfig', () => {
  it('accepts stdio and http servers', () => {
    const servers = parseMcpConfig(
      JSON.stringify({ mcpServers: { fs: { command: 'npx', args: ['-y', 'x'] }, web: { url: 'https://h/mcp' } } })
    )
    expect(Object.keys(servers)).toEqual(['fs', 'web'])
  })
  it('explains invalid configs', () => {
    expect(() => parseMcpConfig('{')).toThrow(/not valid JSON/)
    expect(() => parseMcpConfig('{}')).toThrow(/mcpServers/)
    expect(() => parseMcpConfig('{"mcpServers":{"a":{}}}')).toThrow(/"command".*"url"/)
    expect(() => parseMcpConfig('{"mcpServers":{"a":{"command":"x","args":"y"}}}')).toThrow(/args/)
  })
})

describe('tool names and results', () => {
  it('qualifies names safely', () => {
    expect(qualifyToolName('my server', 'read.file')).toBe('my_server__read_file')
    expect(qualifyToolName('s', 'x'.repeat(100))).toHaveLength(64)
  })
  it('flattens MCP content', () => {
    expect(
      contentToText([
        { type: 'text', text: 'hello' },
        { type: 'image', mimeType: 'image/png', data: '...' },
        { type: 'resource', resource: { uri: 'file://a', text: 'body' } }
      ])
    ).toBe('hello\n[image: image/png]\nbody')
  })
})

describe('streamed tool calls', () => {
  it('merges fragments by index', () => {
    const acc = new ToolCallAccumulator()
    acc.push(interpretChunk({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'c1', function: { name: 'fs__read', arguments: '{"pa' } }] } }] }).toolCalls!)
    acc.push(interpretChunk({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: 'th":"a"}' } }] } }] }).toolCalls!)
    acc.push([{ index: 1, name: 'fs__list', arguments: '{}' }])
    expect(acc.result()).toEqual([
      { id: 'c1', type: 'function', function: { name: 'fs__read', arguments: '{"path":"a"}' } },
      { id: 'call_1', type: 'function', function: { name: 'fs__list', arguments: '{}' } }
    ])
  })
})

describe('pairToolMessages', () => {
  const call = (id: string) => ({ id, type: 'function' as const, function: { name: 't', arguments: '{}' } })
  it('drops orphaned results and calls without results', () => {
    const msgs: WireMessage[] = [
      { role: 'user', content: 'q' },
      { role: 'tool', tool_call_id: 'gone', content: 'orphan' },
      { role: 'assistant', content: '', tool_calls: [call('a'), call('b')] },
      { role: 'tool', tool_call_id: 'a', content: 'ra' }
    ]
    const out = pairToolMessages(msgs)
    expect(out.map((m) => m.role)).toEqual(['user', 'assistant', 'tool'])
    expect(out[1].tool_calls?.map((c) => c.id)).toEqual(['a'])
  })
})

describe('readMcpConfigFile', () => {
  const err = (code: string): NodeJS.ErrnoException => Object.assign(new Error(`${code}: nope`), { code })

  it('treats a missing file as an empty config', async () => {
    const r = await readMcpConfigFile('x', async () => Promise.reject(err('ENOENT')), 3, 0)
    expect(r).toEqual({ text: EMPTY_CONFIG, error: null })
  })
  it('retries a briefly locked file', async () => {
    let calls = 0
    const r = await readMcpConfigFile('x', async () => (++calls < 3 ? Promise.reject(err('EBUSY')) : '{"mcpServers":{}}'), 4, 0)
    expect(r.error).toBeNull()
    expect(calls).toBe(3)
  })
  it('reports a file it cannot read', async () => {
    const r = await readMcpConfigFile('x', async () => Promise.reject(err('EPERM')), 2, 0)
    expect(r.text).toBeNull()
    expect(r.error).toMatch(/Couldn't read mcp.json: EPERM/)
  })
})

describe('McpManager config loading', () => {
  const tmp = (): string => mkdtempSync(join(tmpdir(), 'rulz-mcp-'))

  it('reports a broken file instead of hiding it, and Reconnect all re-reads the file', async () => {
    const dir = tmp()
    const file = join(dir, 'mcp.json')
    writeFileSync(file, '{ "mcpServers": ')
    const configs: McpConfigState[] = []
    const mgr = new McpManager(file, () => undefined, (c) => configs.push(c))
    await mgr.init()
    expect(mgr.config().error).toMatch(/not valid JSON/)
    expect(mgr.config().readFailed).toBe(false)
    expect(configs.at(-1)?.error).toMatch(/not valid JSON/)

    // A disabled server needs no process, so the test stays offline.
    writeFileSync(file, JSON.stringify({ mcpServers: { a: { command: 'x', disabled: true } } }))
    await mgr.restart(null)
    expect(mgr.config().error).toBeNull()
    expect(mgr.status().map((s) => [s.name, s.state])).toEqual([['a', 'disabled']])

    // A bad edit later keeps the servers that are already set up.
    writeFileSync(file, 'nope')
    await mgr.restart(null)
    expect(mgr.config().error).toMatch(/not valid JSON/)
    expect(mgr.status()).toHaveLength(1)
    await mgr.dispose()
    rmSync(dir, { recursive: true, force: true })
  })

  it('starts empty, without an error, when there is no file', async () => {
    const dir = tmp()
    const mgr = new McpManager(join(dir, 'mcp.json'), () => undefined)
    await mgr.init()
    expect(mgr.config()).toMatchObject({ text: EMPTY_CONFIG, error: null, readFailed: false })
    rmSync(dir, { recursive: true, force: true })
  })
})
