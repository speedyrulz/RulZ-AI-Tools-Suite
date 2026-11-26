/// <reference lib="dom" />
// Renderer audit 2026-11-26: benchmark/chat refusals, chat open (last click wins), number clamping, capped log.
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { BenchmarkRequest, BenchmarkRun } from '../src/shared/benchmark'
import { appendCapped, clampTo, textToStops } from '../src/renderer/src/lib/auditHelpers'

const h = vi.hoisted(() => ({ toast: Object.assign(vi.fn(), { error: vi.fn(), message: vi.fn(), success: vi.fn(), warning: vi.fn() }), invoke: vi.fn() }))
vi.mock('sonner', () => ({ toast: h.toast }))
vi.mock('../src/renderer/src/lib/api', () => ({
  invoke: (...a: unknown[]) => h.invoke(...a),
  on: () => () => undefined,
  errorMessage: (e: unknown) => (e instanceof Error ? e.message : String(e))
}))

// The app store reads window and localStorage when it loads.
;(globalThis as unknown as { window: unknown }).window = { rulz: { invoke: async () => undefined, on: () => () => undefined }, addEventListener: () => undefined, matchMedia: () => ({ matches: true, addEventListener: () => undefined }) }
;(globalThis as unknown as { localStorage: unknown }).localStorage = { getItem: () => null, setItem: () => undefined, removeItem: () => undefined }

const { chatBenchRefusal, useChat } = await import('../src/renderer/src/stores/chat')
const { useBench } = await import('../src/renderer/src/stores/bench')
const { setWbenchRunning } = await import('../src/renderer/src/lib/wbenchFlag')

const runningRun = { id: 'r1', status: 'running' } as unknown as BenchmarkRun

beforeEach(() => {
  h.toast.mockClear()
  h.invoke.mockReset()
  setWbenchRunning(false)
  useBench.setState({ runs: [], starting: false })
})

describe('chatBenchRefusal (A1, A2)', () => {
  it('allows a chat when nothing runs', () => {
    expect(chatBenchRefusal('m.gguf')).toBeNull()
  })
  it('refuses a local model while a hardware benchmark runs, but not an OpenRouter model', () => {
    useBench.setState({ runs: [runningRun] })
    expect(chatBenchRefusal('local.gguf')).toBe('A hardware benchmark is running.')
    expect(chatBenchRefusal(undefined)).toBe('A hardware benchmark is running.')
    expect(chatBenchRefusal('openrouter/openai/gpt-4o')).toBeNull()
  })
  it('refuses while a benchmark is starting', () => {
    useBench.setState({ starting: true })
    expect(chatBenchRefusal('local.gguf')).not.toBeNull()
  })
  it('refuses everything while a writing benchmark runs', () => {
    setWbenchRunning(true)
    expect(chatBenchRefusal('openrouter/openai/gpt-4o')).toMatch(/writing benchmark/)
  })
  it('send refuses with a toast and does not start a chat', async () => {
    useBench.setState({ runs: [runningRun] })
    await useChat.getState().send('hello', [])
    expect(h.toast).toHaveBeenCalledWith('A hardware benchmark is running.')
    expect(h.invoke).not.toHaveBeenCalled()
  })
})

describe('bench.start (A2)', () => {
  it('refuses while a chat reply is running', async () => {
    useChat.setState({ generation: { requestId: 'q', conversationId: 'c', messageId: 'm', promptProgress: null } })
    const run = await useBench.getState().start({} as BenchmarkRequest)
    useChat.setState({ generation: null })
    expect(run).toBeNull()
    expect(h.toast).toHaveBeenCalledWith('A chat reply is running.')
    expect(h.invoke).not.toHaveBeenCalled()
  })
  it('starts when nothing is busy', async () => {
    h.invoke.mockResolvedValue({ id: 'new', status: 'running' })
    const run = await useBench.getState().start({} as BenchmarkRequest)
    expect(run?.id).toBe('new')
  })
})

describe('chat open: last click wins (A6)', () => {
  it('drops a slow read of an older chat', async () => {
    const conv = (id: string) => ({ id, title: id, messages: [], updatedAt: 1, modelId: 'm' })
    let releaseA: (v: unknown) => void = () => undefined
    h.invoke.mockImplementation((ch: string, id: string) => {
      if (ch !== 'conversations:get') return Promise.resolve(undefined)
      return id === 'A' ? new Promise((r) => (releaseA = r)) : Promise.resolve(conv('B'))
    })
    useChat.setState({ current: null, genConversation: null })
    const a = useChat.getState().open('A')
    await useChat.getState().open('B')
    releaseA(conv('A'))
    await a
    expect(useChat.getState().current?.id).toBe('B')
  })
  it('drops a slow read when a new chat is started meanwhile', async () => {
    let releaseA: (v: unknown) => void = () => undefined
    h.invoke.mockImplementation((ch: string) => (ch === 'conversations:get' ? new Promise((r) => (releaseA = r)) : Promise.resolve(undefined)))
    useChat.setState({ current: null, genConversation: null })
    const a = useChat.getState().open('A')
    await useChat.getState().newChat()
    const afterNew = useChat.getState().current?.id
    releaseA({ id: 'A', title: 'A', messages: [], updatedAt: 1, modelId: 'm' })
    await a
    expect(useChat.getState().current?.id).not.toBe('A')
    expect(useChat.getState().current?.id).toBe(afterNew)
  })
})

describe('bench.start refuses during an adventure passage (A2)', () => {
  it('refuses while an adventure job runs', async () => {
    const { useAdventure } = await import('../src/renderer/src/stores/adventure')
    const before = useAdventure.getState().job
    useAdventure.setState({ job: { status: 'running' } as never })
    const run = await useBench.getState().start({} as BenchmarkRequest)
    useAdventure.setState({ job: before })
    expect(run).toBeNull()
    expect(h.toast).toHaveBeenCalledWith('An adventure passage is being written.')
  })
})

describe('helpers', () => {
  it('clampTo pulls a value into min and max (A7)', () => {
    expect(clampTo(1000, 1, 500)).toBe(500)
    expect(clampTo(-5, 0, 10)).toBe(0)
    expect(clampTo(7, 0, 10)).toBe(7)
    expect(clampTo(1000)).toBe(1000)
  })
  it('appendCapped keeps the last lines (A10)', () => {
    let l: number[] = []
    for (let i = 0; i < 2500; i++) l = appendCapped(l, i, 2000)
    expect(l).toHaveLength(2000)
    expect(l[0]).toBe(500)
    expect(l.at(-1)).toBe(2499)
  })
  it('textToStops keeps one string per non-empty line (A5)', () => {
    expect(textToStops('a\n\nb')).toEqual(['a', 'b'])
  })
})
