import { describe, expect, it } from 'vitest'
import { resolveTool, searchTools, toolCatalogNote, toolSummary } from '@shared/tools'
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

describe('withSystemNote', () => {
  it('appends to an existing system prompt or adds one', () => {
    expect(withSystemNote([{ role: 'system', content: 'Be brief.' }, { role: 'user', content: 'hi' }], 'NOTE')[0].content).toBe('Be brief.\n\nNOTE')
    expect(withSystemNote([{ role: 'user', content: 'hi' }], 'NOTE')[0]).toEqual({ role: 'system', content: 'NOTE' })
    const msgs = [{ role: 'user' as const, content: 'hi' }]
    expect(withSystemNote(msgs, undefined)).toBe(msgs)
  })
})
