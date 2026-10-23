import { describe, expect, it } from 'vitest'
import {
  COMPACT_MIN_CHARS,
  compactToolResult,
  recentToolCallIds,
  resolveTool,
  searchTools,
  toolCatalogNote,
  toolSummary
} from '@shared/tools'
import type { McpToolInfo } from '@shared/types'
import { withSystemNote } from '../src/main/chat/service'

const tool = (server: string, name: string, description: string): McpToolInfo => ({
  qualifiedName: `${server}__${name}`,
  server,
  name,
  description,
  inputSchema: { type: 'object' }
})

const tools = [
  tool('rea', 'open_binary', 'Open a local executable for analysis. Validates the file first.'),
  tool('rea', 'procedure_pseudo_code', 'Decompile one analyzed procedure by symbol name.'),
  tool('rea', 'batch_decompile', 'Decompile each explicit procedure concurrently.'),
  tool('files', 'read_text_file', 'Read a text file.'),
  tool('other', 'read_text_file', 'Another server with the same tool name.')
]

describe('searchTools', () => {
  it('ranks name matches above description matches', () => {
    const hits = searchTools(tools, 'decompile')
    expect(hits[0].name).toBe('batch_decompile')
    expect(hits.map((h) => h.name)).toContain('procedure_pseudo_code')
    expect(searchTools(tools, 'zzz')).toEqual([])
  })
  it('summarizes to the first sentence', () => {
    expect(toolSummary('Open a file. Then more.\nSecond line')).toBe('Open a file.')
  })
})

describe('resolveTool', () => {
  it('accepts qualified, server-prefixed and unique bare names', () => {
    expect(resolveTool(tools, 'rea__open_binary')?.name).toBe('open_binary')
    expect(resolveTool(tools, 'rea.open_binary')?.name).toBe('open_binary')
    expect(resolveTool(tools, 'open_binary')?.qualifiedName).toBe('rea__open_binary')
    expect(resolveTool(tools, 'read_text_file')).toBeUndefined() // ambiguous
    expect(resolveTool(tools, 'files/read_text_file')?.server).toBe('files')
  })
})

describe('toolCatalogNote', () => {
  it('lists loaded tools and names by server', () => {
    const note = toolCatalogNote(tools, ['rea__open_binary'])
    expect(note).toContain('Loaded tools: rea__open_binary.')
    expect(note).toContain('- rea: open_binary, procedure_pseudo_code, batch_decompile')
  })
  it('caps a huge catalog and points to search', () => {
    const many = Array.from({ length: 800 }, (_, i) => tool(`s${i % 4}`, `some_long_tool_name_${i}`, 'x'))
    const note = toolCatalogNote(many, [])
    expect(note.length).toBeLessThan(7000)
    expect(note).toMatch(/more tools not listed; use rulz_search_tools/)
  })
})

describe('compactToolResult', () => {
  it('leaves short results alone', () => {
    expect(compactToolResult('ok')).toBe('ok')
    expect(compactToolResult('x'.repeat(COMPACT_MIN_CHARS))).toHaveLength(COMPACT_MIN_CHARS)
  })
  it('keeps the beginning and end of long results and says what was cut', () => {
    const lines = Array.from({ length: 400 }, (_, i) => `string ${i}: value`)
    const text = lines.join('\n')
    const out = compactToolResult(text)
    expect(out.length).toBeLessThan(1000)
    expect(out.startsWith('string 0: value')).toBe(true)
    expect(out.endsWith('string 399: value')).toBe(true)
    expect(out).toMatch(/characters of this older tool result omitted/)
    // Cuts land on line boundaries.
    expect(out.split('\n').every((l) => /^string \d+: value$|omitted/.test(l))).toBe(true)
  })
})

describe('recentToolCallIds', () => {
  const call = (id: string, name = 'rea__x') => ({ id, name })
  it('keeps only the most recent calls of the current turn', () => {
    const msgs = [
      { role: 'user' },
      { role: 'assistant', toolCalls: [call('old1'), call('old2')] },
      { role: 'assistant' },
      { role: 'user' },
      { role: 'assistant', toolCalls: [call('a'), call('meta', 'rulz_load_tools')] },
      { role: 'assistant', toolCalls: [call('b'), call('c')] }
    ]
    expect([...recentToolCallIds(msgs, 2)]).toEqual(['b', 'c'])
    expect([...recentToolCallIds(msgs, 10)]).toEqual(['a', 'b', 'c'])
    expect(recentToolCallIds(msgs, 0).size).toBe(0)
  })
  it('compacts everything once the user has replied', () => {
    expect(recentToolCallIds([{ role: 'assistant', toolCalls: [call('a')] }, { role: 'user' }], 3).size).toBe(0)
  })
})

describe('withSystemNote', () => {
  it('appends to an existing system prompt or adds one', () => {
    expect(withSystemNote([{ role: 'system', content: 'Be brief.' }, { role: 'user', content: 'hi' }], 'NOTE')[0].content).toBe('Be brief.\n\nNOTE')
    expect(withSystemNote([{ role: 'user', content: 'hi' }], 'NOTE')[0]).toEqual({ role: 'system', content: 'NOTE' })
    const msgs = [{ role: 'user' as const, content: 'hi' }]
    expect(withSystemNote(msgs, undefined)).toBe(msgs)
  })
})
