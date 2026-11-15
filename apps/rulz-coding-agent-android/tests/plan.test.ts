import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { AgentEvent, PlanItem, ProjectState } from '../src/shared/types'

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

describe('the agent plan from the computer', () => {
  beforeEach(() => {
    const state: ProjectState = {
      project: { path: p, name: 'P', lastOpened: 0 },
      timeline: [],
      tasks: [
        { id: 't1', title: 'One', startedAt: 0, outcome: 'running', steps: [], files: [] },
        { id: 't2', title: 'Two', startedAt: 0, outcome: 'running', steps: [], files: [] }
      ],
      running: true,
      notes: [],
      prompt: null,
      runCommand: null,
      memory: '',
      context: null
    } as ProjectState
    listeners.get('project:changed')!(state)
  })

  it('sets and replaces the plan of one task, leaving the others alone', () => {
    const first: PlanItem[] = [
      { text: 'Read the code', status: 'doing' },
      { text: 'Fix it', status: 'todo' }
    ]
    send({ type: 'plan', projectPath: p, taskId: 't1', plan: first })
    let tasks = useWorkspace.getState().tasks
    expect(tasks.find((t) => t.id === 't1')!.plan).toEqual(first)
    expect(tasks.find((t) => t.id === 't2')!.plan).toBeUndefined()
    const second: PlanItem[] = [
      { text: 'Read the code', status: 'done' },
      { text: 'Fix it', status: 'doing' }
    ]
    send({ type: 'plan', projectPath: p, taskId: 't1', plan: second })
    tasks = useWorkspace.getState().tasks
    expect(tasks.find((t) => t.id === 't1')!.plan).toEqual(second)
    expect(tasks.find((t) => t.id === 't2')!.plan).toBeUndefined()
  })
})
