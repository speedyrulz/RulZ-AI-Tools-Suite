import { describe, expect, it } from 'vitest'
import type { WritingJob } from '../src/shared/phone'
import { checkAccept, mergeSticky, pickAlt, STALE_MESSAGE, WRONG_SCENE_MESSAGE } from '../src/lib/accept'
import { alertForJob, statusLine } from '../src/lib/alerts'
import { isRunning, jobLine, jobsOf, newlyFinished, reduceJobs, runningJob, type JobMap } from '../src/lib/jobs'

function job(id: string, extra: Partial<WritingJob> = {}): WritingJob {
  return {
    id,
    folder: 'F',
    request: { folder: 'F', action: 'continue', sceneId: 's1', baseHash: 'h1', cursor: 5, selection: null, note: '' },
    label: 'Continue',
    status: 'queued',
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

const prose = (extra: Partial<WritingJob> = {}): WritingJob =>
  job('p', { status: 'done', result: { kind: 'prose', alts: [{ text: 'Whole scene A', generated: 'A', seed: 1 }, { text: 'Whole scene B', generated: 'B', seed: 2 }], newSticky: ['mara'] }, ...extra })

describe('job store reducer', () => {
  it('adds and updates jobs from events', () => {
    let s: JobMap = {}
    s = reduceJobs(s, { type: 'update', job: job('a') })
    s = reduceJobs(s, { type: 'update', job: job('a', { status: 'writing', partial: 'The' }) })
    expect(s.a).toMatchObject({ status: 'writing', partial: 'The' })
  })

  it('keeps a finished job finished when a late streaming event arrives', () => {
    let s = reduceJobs({}, { type: 'update', job: job('a', { status: 'done' }) })
    s = reduceJobs(s, { type: 'update', job: job('a', { status: 'writing', partial: 'late' }) })
    expect(s.a.status).toBe('done')
  })

  it('replaces one folder’s jobs with the computer’s listing and leaves other folders alone', () => {
    let s: JobMap = { a: job('a'), b: job('b', { folder: 'G' }), c: job('c') }
    s = reduceJobs(s, { type: 'load', folder: 'F', jobs: [job('c', { status: 'done' }), job('d')] })
    expect(Object.keys(s).sort()).toEqual(['b', 'c', 'd'])
    expect(s.c.status).toBe('done')
  })

  it('does not let an older listing turn a finished job back into a running one', () => {
    let s: JobMap = { a: job('a', { status: 'done' }) }
    s = reduceJobs(s, { type: 'load', folder: 'F', jobs: [job('a', { status: 'writing' })] })
    expect(s.a.status).toBe('done')
  })

  it('removes and clears', () => {
    let s: JobMap = { a: job('a'), b: job('b') }
    expect(reduceJobs(s, { type: 'remove', id: 'zzz' })).toBe(s)
    s = reduceJobs(s, { type: 'remove', id: 'a' })
    expect(Object.keys(s)).toEqual(['b'])
    expect(reduceJobs(s, { type: 'clear' })).toEqual({})
  })

  it('lists a project’s jobs newest first and finds the running one', () => {
    const s: JobMap = { a: job('a', { startedAt: 1, status: 'done' }), b: job('b', { startedAt: 3, status: 'writing' }), c: job('c', { startedAt: 2, folder: 'G' }) }
    expect(jobsOf(s, 'F').map((j) => j.id)).toEqual(['b', 'a'])
    expect(runningJob(s, 'F')?.id).toBe('b')
    expect(isRunning(s.a)).toBe(false)
  })

  it('finds jobs that finished since the last look', () => {
    const before: JobMap = { a: job('a', { status: 'writing' }), b: job('b', { status: 'done' }) }
    const after: JobMap = { a: job('a', { status: 'done' }), b: job('b', { status: 'done' }), c: job('c', { status: 'error' }), d: job('d', { status: 'writing' }) }
    expect(newlyFinished(before, after).map((j) => j.id).sort()).toEqual(['a', 'c'])
  })

  it('says what a job is doing in plain words', () => {
    expect(jobLine(job('a', { status: 'waiting', progress: 0.4 }))).toBe('Continue · reading the story 40%')
    expect(jobLine(job('a', { status: 'writing', alt: 2, of: 3 }))).toBe('Continue · writing 2 of 3')
    expect(jobLine(job('a', { status: 'writing' }))).toBe('Continue · writing')
    expect(jobLine(job('a', { status: 'error' }))).toBe('Continue · didn’t finish')
  })
})

describe('accepting a prose result', () => {
  const input = (over: Partial<Parameters<typeof checkAccept>[0]> = {}): Parameters<typeof checkAccept>[0] => ({
    bufferFolder: 'F',
    bufferPath: 'manuscript/s1.md',
    expectedFolder: 'F',
    expectedPath: 'manuscript/s1.md',
    bufferText: 'Scene text.',
    bufferDirty: true,
    bufferBaseHash: 'h1',
    job: prose(),
    memo: { text: 'Scene text.', baseHash: 'h1' },
    ...over
  })

  it('is allowed when the buffer is still the text the run was started with', () => {
    expect(checkAccept(input())).toEqual({ ok: true })
  })

  it('is allowed for unsaved text that was sent, even though the disk has older text', () => {
    expect(checkAccept(input({ bufferDirty: true, bufferBaseHash: 'h0' }))).toEqual({ ok: true })
  })

  it('is refused when the buffer changed since the run', () => {
    expect(checkAccept(input({ bufferText: 'Scene text. And I typed more.' }))).toEqual({ ok: false, reason: STALE_MESSAGE })
  })

  it('without the remembered text (after a restart) trusts only a clean buffer on the job’s base hash', () => {
    expect(checkAccept(input({ memo: null, bufferDirty: false, bufferBaseHash: 'h1' }))).toEqual({ ok: true })
    expect(checkAccept(input({ memo: null, bufferDirty: true, bufferBaseHash: 'h1' })).ok).toBe(false)
    expect(checkAccept(input({ memo: null, bufferDirty: false, bufferBaseHash: 'h2' })).ok).toBe(false)
  })

  it('is refused when the editor holds another scene or project than the one the result was written for', () => {
    expect(checkAccept(input({ bufferPath: 'manuscript/s2.md' }))).toEqual({ ok: false, reason: WRONG_SCENE_MESSAGE })
    expect(checkAccept(input({ bufferFolder: 'G' })).ok).toBe(false)
    expect(checkAccept(input({ expectedPath: null })).ok).toBe(false)
  })

  it('refuses a result that has no prose', () => {
    expect(checkAccept(input({ job: job('x', { status: 'done', result: { kind: 'interview', characterId: 'c', reply: 'hi' } }) })).ok).toBe(false)
  })

  it('picks the chosen alternative', () => {
    expect(pickAlt(prose(), 1)?.text).toBe('Whole scene B')
    expect(pickAlt(prose(), 5)).toBeNull()
  })

  it('keeps the sticky list in order without repeats', () => {
    expect(mergeSticky(['a', 'b'], ['b', 'c'])).toEqual(['a', 'b', 'c'])
  })
})

describe('notifications', () => {
  it('says a prose result is ready, with its start', () => {
    const a = alertForJob(prose(), 'The Ferry', 0)
    expect(a).toMatchObject({ title: 'Continue is ready · The Ferry', body: 'A' })
  })
  it('reports a failure and stays silent for a stop', () => {
    expect(alertForJob(job('a', { status: 'error', error: 'No model loaded.' }), 'P', 3)?.body).toBe('No model loaded.')
    expect(alertForJob(job('a', { status: 'stopped' }), 'P', 3)).toBeNull()
  })
  it('rotates ids', () => {
    expect(alertForJob(prose(), 'P', 0)!.id).not.toBe(alertForJob(prose(), 'P', 1)!.id)
  })
  it('words the ongoing notification', () => {
    expect(statusLine({ computer: 'pc', link: 'connecting' }).title).toBe('Reconnecting to pc…')
    expect(statusLine({ computer: 'pc', link: 'connected', project: 'Ferry', job: 'Continue · writing' })).toEqual({ title: 'Writing · Ferry', text: 'Continue · writing' })
    expect(statusLine({ computer: 'pc', link: 'connected' }).title).toBe('Connected to pc')
  })
})

describe('listing versus newer events (jobs load)', () => {
  it('keeps a job that arrived by event after the listing was asked for, even if the listing lacks it or is older', () => {
    const state: JobMap = { a: job('a', { status: 'done' }), n: job('n', { status: 'writing', partial: 'newer' }) }
    const next = reduceJobs(state, {
      type: 'load',
      folder: 'F',
      jobs: [job('a', { status: 'writing' })],
      startedAt: 100,
      receivedAt: { a: 50, n: 150 }
    })
    expect(next.n.partial).toBe('newer')
    expect(next.a.status).toBe('done')
  })

  it('prefers the event’s version of a job in the listing when the event is newer', () => {
    const state: JobMap = { a: job('a', { status: 'writing', partial: 'event' }) }
    const next = reduceJobs(state, { type: 'load', folder: 'F', jobs: [job('a', { status: 'preparing' })], startedAt: 100, receivedAt: { a: 120 } })
    expect(next.a.partial).toBe('event')
  })

  it('still drops jobs the computer no longer holds when nothing newer arrived, and ignores jobs dismissed here since', () => {
    const state: JobMap = { gone: job('gone', { status: 'done' }) }
    const next = reduceJobs(state, { type: 'load', folder: 'F', jobs: [job('x'), job('y')], startedAt: 100, receivedAt: { gone: 10 }, removedAt: { y: 130 } })
    expect(Object.keys(next)).toEqual(['x'])
  })
})
