// src/shared/novelRun.ts: counting and generation go through injected backends (no renderer, no main).
import { describe, expect, it } from 'vitest'
import { clearCountCache, makeIpcCounter, startGeneration, templateOverheadWith, type GenBackend } from '../src/shared/novelRun'
import { DEFAULT_SAMPLING } from '../src/shared/defaults'
import type { ChatRequest, StreamEvent } from '../src/shared/types'

function backend(script: (req: ChatRequest, send: (e: StreamEvent) => void) => Promise<void> | void) {
  const handlers = new Map<string, (e: StreamEvent) => void>()
  const aborted: string[] = []
  let subscriptions = 0
  const b: GenBackend = {
    start: async (req) => script(req, (e) => handlers.get(req.requestId)?.(e)),
    abort: (id) => void aborted.push(id),
    subscribe: (id, h) => {
      subscriptions++
      handlers.set(id, h)
      return () => void handlers.delete(id)
    },
    newId: () => 'r1'
  }
  return { b, handlers, aborted, subs: () => subscriptions }
}

const input = { modelId: 'm', messages: [{ role: 'user' as const, content: 'hi' }], sampling: DEFAULT_SAMPLING, adapters: null, action: 'continue' as const, seed: 7, maxTokens: 50 }

describe('startGeneration with a backend', () => {
  it('collects deltas, reports progress and ends on done; the request is prefitted with the sampling', async () => {
    const seen: ChatRequest[] = []
    const t = backend((req, send) => {
      seen.push(req)
      send({ type: 'progress', requestId: req.requestId, processed: 1, total: 4 })
      send({ type: 'delta', requestId: req.requestId, content: 'Hel' })
      send({ type: 'delta', requestId: req.requestId, content: 'lo' })
      send({ type: 'done', requestId: req.requestId, stats: { stopReason: 'length' } })
    })
    const progress: (number | null)[] = []
    const texts: string[] = []
    const gen = startGeneration({ ...input, onProgress: (p) => progress.push(p), onDelta: (x) => texts.push(x) }, t.b)
    expect(gen.requestId).toBe('r1')
    expect(await gen.done).toEqual({ text: 'Hello', error: null, stopped: false, stopReason: 'length' })
    expect(progress).toEqual([0.25])
    expect(texts).toEqual(['Hel', 'Hello'])
    expect(seen[0]).toMatchObject({ requestId: 'r1', modelId: 'm', overflow: 'stop', prefitted: true, disableThinking: true })
    expect(seen[0].sampling).toMatchObject({ seed: 7, maxTokens: 50, thinking: 'off', jsonSchema: '' })
    // The subscription ended with the request.
    expect(t.handlers.size).toBe(0)
  })

  it('turns an error event and a failed start into an error result', async () => {
    const a = backend((req, send) => send({ type: 'error', requestId: req.requestId, error: 'bad' }))
    expect(await startGeneration(input, a.b).done).toMatchObject({ text: '', error: 'bad', stopped: false })
    const b = backend(() => {
      throw new Error("Error invoking remote method 'chat:start': Error: no model")
    })
    expect(await startGeneration(input, b.b).done).toMatchObject({ error: 'no model' })
  })

  it('aborts through the backend and ends as stopped when the stream sends nothing more', async () => {
    const t = backend(() => new Promise(() => undefined))
    const gen = startGeneration(input, t.b)
    gen.abort()
    expect(t.aborted).toEqual(['r1'])
    // A stopped event from the server ends it right away.
    t.handlers.get('r1')?.({ type: 'done', requestId: 'r1', stats: { stopReason: 'stopped' } })
    expect(await gen.done).toMatchObject({ stopped: true, error: null })
  })
})

describe('injected counting', () => {
  it('caches counts per model and text and falls back to an estimate', async () => {
    clearCountCache()
    let calls = 0
    const counter = makeIpcCounter(async (_m, text) => {
      calls++
      return text === 'fail' ? null : text.length
    })
    let estimates = 0
    const count = counter('model-x', () => estimates++)
    expect(await count('abcd')).toBe(4)
    expect(await count('abcd')).toBe(4)
    expect(calls).toBe(1)
    expect(await count('fail')).toBe(Math.ceil(4 / 3))
    expect(estimates).toBe(1)
    expect(await count('')).toBe(0)
    clearCountCache()
  })

  it('measures the template overhead once per instance and assumes 48 when it cannot count', async () => {
    const count = async (t: string): Promise<number> => t.length
    let calls = 0
    const countTokens = async (): Promise<number | null> => (++calls, 100)
    const first = await templateOverheadWith(countTokens, 'inst-overhead-1', 'm', count)
    expect(first).toBe(100 - 'You are a writer.'.length - 'Write one sentence.'.length)
    expect(await templateOverheadWith(countTokens, 'inst-overhead-1', 'm', count)).toBe(first)
    expect(calls).toBe(1)
    expect(await templateOverheadWith(async () => null, 'inst-overhead-2', 'm', count)).toBe(48)
  })
})
