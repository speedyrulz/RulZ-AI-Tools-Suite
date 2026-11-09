import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { AgentEvent, DraftStats, ProjectState } from '../src/shared/types'
import { thinkingOpen } from '../src/shared/think'
import { draftSummary } from '../src/lib/drafts'

// The workspace store listens for the computer's events; here the test sends them.
const listeners = new Map<string, (payload: unknown) => void>()
vi.mock('../src/lib/api', () => ({
  on: (channel: string, listener: (payload: unknown) => void) => {
    listeners.set(channel, listener)
    return () => listeners.delete(channel)
  },
  invoke: vi.fn(async () => null),
  errorMessage: (e: unknown) => String(e)
}))
vi.mock('sonner', () => ({ toast: { error: vi.fn(), success: vi.fn() } }))

const { useWorkspace } = await import('../src/stores/workspace')
const p = 'C:\p'
const send = (e: AgentEvent): void => listeners.get('agent:event')!(e)

const draft: DraftStats = { drafted: 400, accepted: 300, generated: 1000, genMs: 20_000, savedMs: 4500, measured: true }

describe('thinking and drafts from the computer', () => {
  beforeEach(() => {
    const state: ProjectState = {
      project: { path: p, name: 'P', lastOpened: 0 },
      timeline: [],
      tasks: [{ id: 't1', title: 'T', startedAt: 0, outcome: 'running', steps: [], files: [] }],
      running: true,
      notes: [],
      prompt: null,
      runCommand: null,
      memory: '',
      context: null
    } as ProjectState
    listeners.get('project:changed')!(state)
  })

  it('collects streamed thoughts and keeps a reply that was only thinking', () => {
    send({ type: 'item', projectPath: p, item: { id: 'a', kind: 'assistant', text: '', taskId: 't1' } })
    send({ type: 'reasoning', projectPath: p, itemId: 'a', text: 'Let me ' })
    send({ type: 'reasoning', projectPath: p, itemId: 'a', text: 'look.' })
    send({ type: 'item', projectPath: p, item: { id: 'a', kind: 'assistant', text: '', taskId: 't1', reasoning: 'Let me look.' } })
    expect(useWorkspace.getState().timeline).toEqual([{ id: 'a', kind: 'assistant', text: '', taskId: 't1', reasoning: 'Let me look.' }])
    // A reply with neither words nor thoughts is dropped.
    send({ type: 'item', projectPath: p, item: { id: 'a', kind: 'assistant', text: '', taskId: 't1' } })
    expect(useWorkspace.getState().timeline).toEqual([])
  })

  it('saves the draft stats on the task', () => {
    send({ type: 'draft', projectPath: p, taskId: 't1', draft })
    expect(useWorkspace.getState().tasks[0].draft).toEqual(draft)
  })
})

describe('thinking box', () => {
  it('is open while thinking and folds away after, unless kept open', () => {
    expect(thinkingOpen(null, true, true, false)).toBe(true)
    expect(thinkingOpen(null, false, true, false)).toBe(false)
    expect(thinkingOpen(null, false, true, true)).toBe(true)
    // Earlier tasks start folded, and the owner's tap wins.
    expect(thinkingOpen(null, false, false, true)).toBe(false)
    expect(thinkingOpen(true, false, false, false)).toBe(true)
  })
})

describe('drafts readout', () => {
  it('shows the acceptance rate and time saved', () => {
    expect(draftSummary(draft)).toMatchObject({ rate: 75, saved: '4.50 s saved' })
    expect(draftSummary(draft)!.detail).toBe('400 tokens drafted, 300 accepted (75%), over 1,000 tokens written in 20.0 s.')
  })

  it('marks an upper bound and a slowdown', () => {
    expect(draftSummary({ ...draft, measured: false })!.saved).toBe('≤ 4.50 s saved')
    expect(draftSummary({ ...draft, savedMs: -300 })!.saved).toBe('300 ms slower')
  })

  it('is hidden when the task did not draft', () => {
    expect(draftSummary(undefined)).toBeNull()
    expect(draftSummary({ ...draft, drafted: 0, accepted: 0 })).toBeNull()
  })
})
