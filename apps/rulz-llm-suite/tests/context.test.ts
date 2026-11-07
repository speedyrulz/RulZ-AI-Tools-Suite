import { describe, expect, it } from 'vitest'
import type { WireMessage } from '@shared/types'
import { chatTurns, ContextFullError, dropOrder, fitToContext, textOf, upperBoundTokens } from '../src/main/chat/context'
import { limiter } from '../src/main/chat/service'

const call = (id: string) => ({ id, type: 'function' as const, function: { name: 't', arguments: '{}' } })
/** A message whose text is `len` characters long, labelled so the tests can tell them apart. */
const m = (role: WireMessage['role'], label: string, len: number, extra: Partial<WireMessage> = {}): WireMessage => ({
  role,
  content: label.padEnd(len, '.'),
  ...extra
})
const labels = (ms: WireMessage[]): string[] => ms.map((x) => (x.content as string).replace(/\.+$/, ''))

/** One token per character, 4 template tokens per message, plus the tool definitions. */
function fakeApi(toolTokens = 0) {
  const api = {
    calls: 0,
    countText: async (text: string) => text.length,
    countConversation: async (ms: WireMessage[]) => {
      api.calls++
      return ms.reduce((a, x) => a + textOf(x).length + 4, 0) + toolTokens
    },
    countTools: async () => toolTokens
  }
  return api
}

/** Two earlier turns, then a question the model is answering with a tool call. */
const toolChat: WireMessage[] = [
  m('system', 'sys', 50),
  m('user', 'u1', 100),
  m('assistant', 'a1', 100),
  m('user', 'u2', 100),
  m('assistant', 'a2', 100),
  m('user', 'u3', 100),
  m('assistant', 'a3', 100),
  m('user', 'question', 100),
  m('assistant', 'call', 50, { tool_calls: [call('c1')] }),
  m('tool', 'result', 300, { tool_call_id: 'c1' })
]

describe('chat turns', () => {
  it('starts a turn at each user message and keeps a greeting on its own', () => {
    const ms = [m('system', 's', 1), m('assistant', 'hi', 1), m('user', 'q', 1), m('assistant', 'a', 1), m('tool', 't', 1)]
    expect(chatTurns(ms)).toEqual([[1], [2, 3, 4]])
  })

  it('drops middle turns first, then the first turn, then older steps of the current one', () => {
    const ms = [
      m('system', 's', 1),
      m('user', 'u1', 1),
      m('assistant', 'a1', 1),
      m('user', 'u2', 1),
      m('assistant', 'a2', 1),
      m('user', 'q', 1),
      m('assistant', 'c1', 1, { tool_calls: [call('c1')] }),
      m('tool', 'r1', 1, { tool_call_id: 'c1' }),
      m('assistant', 'c2', 1, { tool_calls: [call('c2')] }),
      m('tool', 'r2', 1, { tool_call_id: 'c2' })
    ]
    expect(dropOrder(ms)).toEqual([[3, 4], [1, 2], [6, 7]])
  })
})

describe('context fitting', () => {
  it('keeps the current question and its tool call, dropping whole earlier turns', async () => {
    const api = fakeApi()
    const fitted = await fitToContext(toolChat, 1200, 200, 'truncateMiddle', api)
    expect(labels(fitted)).toEqual(['sys', 'u1', 'a1', 'u3', 'a3', 'question', 'call', 'result'])
    // User and assistant still alternate (Gemma 3's template raises an error otherwise).
    expect(fitted.map((x) => x.role)).toEqual(['system', 'user', 'assistant', 'user', 'assistant', 'user', 'assistant', 'tool'])
    // One measure up front, one to confirm the trimmed chat fits.
    expect(api.calls).toBe(2)
  })

  it('counts the tool definitions against the budget', async () => {
    expect(await fitToContext(toolChat, 1400, 200, 'truncateMiddle', fakeApi())).toBe(toolChat)
    const fitted = await fitToContext(toolChat, 1400, 200, 'truncateMiddle', fakeApi(300))
    expect(labels(fitted)).toEqual(['sys', 'u1', 'a1', 'question', 'call', 'result'])
  })

  it('drops the first turn, then older tool steps of the current turn, as a last resort', async () => {
    const ms = [
      m('system', 'sys', 50),
      m('user', 'u1', 100),
      m('assistant', 'a1', 100),
      m('user', 'u2', 100),
      m('assistant', 'a2', 100),
      m('user', 'question', 50),
      m('assistant', 'call1', 50, { tool_calls: [call('c1')] }),
      m('tool', 'r1', 400, { tool_call_id: 'c1' }),
      m('assistant', 'call2', 50, { tool_calls: [call('c2')] }),
      m('tool', 'r2', 400, { tool_call_id: 'c2' })
    ]
    const fitted = await fitToContext(ms, 900, 200, 'truncateMiddle', fakeApi())
    expect(labels(fitted)).toEqual(['sys', 'question', 'call2', 'r2'])
    expect(fitted[2].tool_calls?.map((c) => c.id)).toEqual(['c2'])
  })

  it('sends the chat as it is when the question and latest step alone are too big', async () => {
    const ms = [
      m('system', 'sys', 50),
      m('user', 'question', 100),
      m('assistant', 'call', 50, { tool_calls: [call('c1')] }),
      m('tool', 'huge', 2000, { tool_call_id: 'c1' })
    ]
    const api = fakeApi()
    expect(await fitToContext(ms, 900, 200, 'truncateMiddle', api)).toBe(ms)
    expect(api.calls).toBe(1)
  })

  it('throws for the stop policy', async () => {
    await expect(fitToContext(toolChat, 1200, 200, 'stop', fakeApi())).rejects.toBeInstanceOf(ContextFullError)
    expect(await fitToContext(toolChat, 10000, 200, 'stop', fakeApi())).toBe(toolChat)
  })

  it('skips counting when the chat is clearly small enough', async () => {
    const api = fakeApi()
    const small = [m('system', 'sys', 3), m('user', 'hi', 2)]
    expect(await fitToContext(small, 8192, 2048, 'stop', api)).toBe(small)
    expect(api.calls).toBe(0)
    // Multi-byte text is counted by UTF-8 bytes, so it is never underestimated.
    expect(upperBoundTokens([{ role: 'user', content: '漢字' }])).toBe(64 + 16 + 6)
  })
})

describe('limiter', () => {
  it('runs at most n tasks at once and runs them all', async () => {
    const limit = limiter(2)
    let running = 0
    let peak = 0
    const task = async (i: number) => {
      running++
      peak = Math.max(peak, running)
      await new Promise((r) => setTimeout(r, 5))
      running--
      return i
    }
    expect(await Promise.all([1, 2, 3, 4, 5].map((i) => limit(() => task(i))))).toEqual([1, 2, 3, 4, 5])
    expect(peak).toBe(2)
  })
})
