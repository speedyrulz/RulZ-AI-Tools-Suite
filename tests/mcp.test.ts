import { describe, expect, it } from 'vitest'
import type { WireMessage } from '@shared/types'
import { pairToolMessages } from '../src/main/chat/context'
import { interpretChunk, ToolCallAccumulator } from '../src/main/chat/sse'
import { contentToText, parseMcpConfig, qualifyToolName } from '../src/main/mcp/config'

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
