import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { fakeToolConfig } from './helpers/fakeTools'
import { systemPrompt } from '../src/main/agent/prompt'
import { approvalKey, needsApproval } from '../src/main/agent/safety'
import { describeStep } from '../src/main/agent/tools'
import { parseToolConfig, qualifyToolName, readToolConfig, searchTools } from '../src/main/mcp/config'
import { ToolServers } from '../src/main/mcp/toolServers'

const base = mkdtempSync(join(tmpdir(), 'rulz-agent-tools-'))
afterAll(() => rmSync(base, { recursive: true, force: true }))

describe("RulZ LLM Suite's mcp.json", () => {
  it('reads servers in the usual format and explains mistakes', () => {
    expect(parseToolConfig('{"mcpServers":{"rea":{"command":"npx","args":["-y","rea-agents@6.7.0","mcp"]}}}')).toEqual({
      rea: { command: 'npx', args: ['-y', 'rea-agents@6.7.0', 'mcp'] }
    })
    expect(() => parseToolConfig('{nope')).toThrow(/not valid JSON/)
    expect(() => parseToolConfig('{"servers":{}}')).toThrow(/"mcpServers"/)
    expect(() => parseToolConfig('{"mcpServers":{"x":{}}}')).toThrow(/needs either "command"/)
    expect(() => parseToolConfig('{"mcpServers":{"x":{"command":"a","args":[1]}}}')).toThrow(/list of strings/)
  })

  it('treats a missing file as no servers and reports a broken one', async () => {
    expect(await readToolConfig(join(base, 'missing.json'), (p) => import('node:fs/promises').then((f) => f.readFile(p, 'utf8')))).toEqual({ servers: {}, error: null })
    const bad = join(base, 'bad.json')
    writeFileSync(bad, '{oops')
    const r = await readToolConfig(bad, (p) => import('node:fs/promises').then((f) => f.readFile(p, 'utf8')))
    expect(r.servers).toEqual({})
    expect(r.error).toMatch(/not valid JSON/)
  })

  it('names tools by server and finds them by keywords', () => {
    expect(qualifyToolName('rea', 'open_binary')).toBe('rea__open_binary')
    expect(qualifyToolName('my server', 'do.it')).toBe('my_server__do_it')
    const tools = [
      { name: 'open_binary', description: 'Open a local executable for analysis.' },
      { name: 'procedure_pseudo_code', description: 'Decompile one function to pseudo code.' },
      { name: 'list_strings', description: 'List strings in the binary.' }
    ]
    expect(searchTools(tools, 'decompile function').map((t) => t.name)).toEqual(['procedure_pseudo_code'])
    expect(searchTools(tools, 'binary')[0].name).toBe('open_binary')
    expect(searchTools(tools, '?')).toEqual([])
  })
})

describe('tool servers', () => {
  it('start, list their tools, give advice and run tools', async () => {
    const servers = new ToolServers(fakeToolConfig(base, { off: { command: 'nothing', disabled: true } }))
    try {
      expect(await servers.configured()).toEqual(['fake'])
      expect(servers.status().servers).toEqual([])
      await servers.start()
      expect(servers.status().servers).toMatchObject([{ name: 'fake', state: 'connected', tools: 3 }])
      expect(servers.tools().map((t) => [t.qualifiedName, t.destructive])).toEqual([
        ['fake__count_letters', false],
        ['fake__wipe_notes', true],
        ['fake__huge_tool', false]
      ])
      expect(servers.find('count_letters')?.qualifiedName).toBe('fake__count_letters')
      expect(servers.instructions()).toEqual([{ server: 'fake', text: 'Fake server advice: count before you wipe.' }])
      expect(await servers.call(servers.find('fake__count_letters')!, { text: 'Hello, you!' })).toEqual({ text: 'Letters: 8', isError: false })
    } finally {
      await servers.dispose()
    }
  }, 30_000)

  it('reports a server that cannot start', async () => {
    const path = join(base, 'broken.json')
    writeFileSync(path, JSON.stringify({ mcpServers: { broken: { command: 'this-program-does-not-exist-anywhere' } } }))
    const servers = new ToolServers(path)
    try {
      await servers.start()
      expect(servers.status().servers).toMatchObject([{ name: 'broken', state: 'error' }])
      expect(servers.status().servers[0].error).toBeTruthy()
      expect(servers.tools()).toEqual([])
    } finally {
      await servers.dispose()
    }
  }, 30_000)
})

describe('approvals for tool-server tools', () => {
  const plain = { type: 'tool' as const, server: 'rea', tool: 'open_binary', destructive: false }
  const risky = { type: 'tool' as const, server: 'rea', tool: 'capture_process_scenario', destructive: true }

  it('ask once per server per task for ordinary tools, and every time for destructive ones', () => {
    expect(approvalKey(plain)).toBe('tool:rea')
    expect(approvalKey({ ...plain, tool: 'list_strings' })).toBe('tool:rea')
    expect(approvalKey(risky)).toBe('tool:rea:capture_process_scenario')
    const allowed = new Set(['tool:rea'])
    expect(needsApproval('balanced', plain, new Set())).toBe(true)
    expect(needsApproval('balanced', plain, allowed)).toBe(false)
    expect(needsApproval('balanced', risky, allowed)).toBe(true)
    expect(needsApproval('handsoff', plain, new Set())).toBe(false)
    expect(needsApproval('handsoff', risky, new Set())).toBe(true)
  })

  it('get plain step titles', () => {
    expect(describeStep('rea__procedure_pseudo_code', {})).toBe('Using REA: procedure pseudo code')
    expect(describeStep('github__create_issue', {})).toBe('Using github: create issue')
    expect(describeStep('find_tools', { query: 'decompile' })).toBe('Looking for tools: "decompile"')
  })
})

describe('system prompt with tool servers', () => {
  const base = { projectPath: 'C:\\coding\\My App', projectName: 'My App', date: 'Mon Oct 27 2026' }
  it('explains finding and loading tools only when there are servers', () => {
    expect(systemPrompt(base)).not.toContain('find_tools')
    const p = systemPrompt({ ...base, extraTools: { servers: ['REA'], advice: [{ server: 'REA', text: 'Use the tool that directly answers the question.' }] } })
    expect(p).toContain('## Extra tools from RulZ LLM Suite (REA)')
    expect(p).toContain("don't look for a program of that name on disk")
    expect(p).toContain('outside the project folder')
    expect(p).toContain('Advice from REA: Use the tool that directly answers the question.')
    expect(p.indexOf('## Honesty and safety')).toBeLessThan(p.indexOf('## Extra tools'))
  })
})
