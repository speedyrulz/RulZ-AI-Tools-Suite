import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { WritingJob } from '../src/shared/phone'

vi.mock('../src/lib/native', () => ({
  loadPref: async (_key: string, fallback: unknown) => fallback,
  savePref: async () => undefined
}))
vi.mock('sonner', () => ({ toast: { error: vi.fn(), success: vi.fn() } }))

type Call = { channel: string; args: unknown[]; resolve: (v: unknown) => void; reject: (e: Error) => void }
const calls: Call[] = []
vi.mock('../src/lib/remote', () => ({
  link: {
    invoke: (channel: string, ...args: unknown[]) => new Promise((resolve, reject) => calls.push({ channel, args, resolve, reject })),
    on: () => () => undefined
  }
}))

const { useInterview, chatKey } = await import('../src/stores/interview')
const { useJobs } = await import('../src/stores/jobs')

const flush = (): Promise<void> => new Promise((r) => setTimeout(r, 0))
const KEY = chatKey('F', 'mara')

function job(id: string, extra: Partial<WritingJob> = {}): WritingJob {
  return {
    id,
    folder: 'F',
    request: { folder: 'F', action: 'interview', sceneId: null, baseHash: '', cursor: 0, selection: null, note: '', interview: { characterId: 'mara', turns: [] } },
    label: 'Interview',
    status: 'writing',
    alt: 1,
    of: 1,
    progress: null,
    partial: '',
    result: null,
    error: null,
    warnings: [],
    startedAt: 1,
    endedAt: null,
    ...extra
  }
}

beforeEach(async () => {
  calls.length = 0
  useInterview.setState({ chats: {}, waiting: {} })
  useJobs.setState({ jobs: {}, memos: {}, receivedAt: {}, removedAt: {} })
  await useInterview.getState().init()
})

describe('the interview wait marker', () => {
  it('is set before the request goes out and cleared when the job could not start', async () => {
    const asked = useInterview.getState().ask('F', null, 'mara', 'Where were you?')
    await flush()
    expect(useInterview.getState().waiting[KEY]).toBe('pending')
    calls.find((c) => c.channel === 'writing:run')!.reject(new Error('Another job is running.'))
    expect(await asked).toBe(false)
    expect(useInterview.getState().waiting[KEY]).toBeUndefined()
  })

  it('points at the job once it started', async () => {
    const asked = useInterview.getState().ask('F', null, 'mara', 'Where were you?')
    await flush()
    calls.find((c) => c.channel === 'writing:run')!.resolve(job('j1'))
    expect(await asked).toBe(true)
    expect(useInterview.getState().waiting[KEY]).toBe('j1')
  })

  it('is dropped when its job is gone (cleared jobs, a listing without it)', async () => {
    useInterview.setState({ waiting: { [KEY]: 'gone' } })
    useJobs.setState({ jobs: { other: job('other') } })
    expect(useInterview.getState().waiting[KEY]).toBeUndefined()
  })

  it('is dropped when the conversation is cleared', () => {
    useInterview.setState({ chats: { [KEY]: [{ role: 'user', text: 'Hi' }] }, waiting: { [KEY]: 'pending' } })
    useInterview.getState().clear('F', 'mara')
    expect(useInterview.getState().waiting[KEY]).toBeUndefined()
    expect(useInterview.getState().chats[KEY]).toBeUndefined()
  })
})
