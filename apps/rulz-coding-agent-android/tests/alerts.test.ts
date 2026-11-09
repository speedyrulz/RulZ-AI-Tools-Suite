import { describe, expect, it } from 'vitest'
import type { AgentEvent, TaskRecord, TimelineItem } from '../src/shared/types'
import { alertFor, DEFAULT_ALERT_PREFS, plainText, statusLine, WAITING_ID, type AlertContext } from '../src/lib/alerts'

const task = (outcome: TaskRecord['outcome']): TaskRecord => ({ id: 't1', title: 'Make a hello page', startedAt: 0, outcome, steps: [], files: [] })
const timeline: TimelineItem[] = [
  { id: 'u', kind: 'user', text: 'Make a hello page', taskId: 't1' },
  { id: 'a1', kind: 'assistant', text: "I'll make it.", taskId: 't1' },
  { id: 'a2', kind: 'assistant', text: 'Your page is **ready**. Open `index.html` to see it.', taskId: 't1' }
]
const ctx = (over: Partial<AlertContext> = {}): AlertContext => ({ prefs: DEFAULT_ALERT_PREFS, projectName: 'Phone Test', timeline, finishedCount: 0, ...over })
const p = 'C:\\p'

describe('notifications', () => {
  it('asks for permission requests, with the risk', () => {
    const e: AgentEvent = { type: 'prompt', projectPath: p, prompt: { id: '1', kind: 'approval', action: 'Run "npm install"', why: 'x', detail: 'npm install', risk: 'This uses the internet.' } }
    expect(alertFor(e, ctx())).toEqual([
      { kind: 'show', id: WAITING_ID, channel: 'waiting', title: 'Permission needed · Phone Test', body: 'Run "npm install" This uses the internet.' }
    ])
  })

  it('also tells about questions and pauses', () => {
    expect(alertFor({ type: 'prompt', projectPath: p, prompt: { id: '2', kind: 'question', question: 'Which **colour**?' } }, ctx())).toMatchObject([
      { title: 'Question · Phone Test', body: 'Which colour?' }
    ])
    expect(alertFor({ type: 'prompt', projectPath: p, prompt: { id: '3', kind: 'stuck', message: 'I keep failing.', reason: 'errors' } }, ctx())).toMatchObject([
      { title: 'Paused · Phone Test' }
    ])
  })

  it('removes the waiting notification once it is answered, even with waiting notifications off', () => {
    expect(alertFor({ type: 'prompt', projectPath: p, prompt: null }, ctx({ prefs: { ...DEFAULT_ALERT_PREFS, waiting: false } }))).toEqual([{ kind: 'cancel', id: WAITING_ID }])
  })

  it('says a task is done with the agent’s last words', () => {
    expect(alertFor({ type: 'task-end', projectPath: p, task: task('done') }, ctx({ finishedCount: 21 }))).toEqual([
      { kind: 'cancel', id: WAITING_ID },
      { kind: 'show', id: 101, channel: 'finished', title: 'Done: Make a hello page', body: 'Your page is ready. Open index.html to see it.' }
    ])
  })

  it('says when a task didn’t finish, and stays quiet when it was stopped', () => {
    expect(alertFor({ type: 'task-end', projectPath: p, task: task('error') }, ctx({ timeline: [] }))[1]).toMatchObject({
      title: 'Didn’t finish: Make a hello page',
      body: 'Something went wrong in Phone Test.'
    })
    expect(alertFor({ type: 'task-end', projectPath: p, task: task('stopped') }, ctx())).toEqual([{ kind: 'cancel', id: WAITING_ID }])
  })

  it('follows the settings', () => {
    const off = { ...DEFAULT_ALERT_PREFS, finished: false, waiting: false }
    expect(alertFor({ type: 'task-end', projectPath: p, task: task('done') }, ctx({ prefs: off }))).toEqual([{ kind: 'cancel', id: WAITING_ID }])
    expect(alertFor({ type: 'prompt', projectPath: p, prompt: { id: '1', kind: 'question', question: 'q' } }, ctx({ prefs: off }))).toEqual([])
    expect(alertFor({ type: 'delta', projectPath: p, itemId: 'a', text: 'x' }, ctx())).toEqual([])
  })

  it('shows what the agent is doing in the ongoing notification', () => {
    const base = { computer: 'jeff-ai', link: 'connected' as const, project: 'Phone Test', running: false, prompt: null }
    expect(statusLine(base)).toEqual({ title: 'Connected to jeff-ai', text: 'Nothing running. You’ll get a notification when a task finishes or needs you.' })
    expect(statusLine({ ...base, running: true, task: 'Make a hello page', doing: 'Creating your page' })).toEqual({
      title: 'Working on “Make a hello page”',
      text: 'Creating your page'
    })
    expect(
      statusLine({ ...base, running: true, prompt: { id: '1', kind: 'approval', action: 'Run `npm install`', why: '', detail: '' } })
    ).toEqual({ title: 'Waiting for your OK · Phone Test', text: 'Run npm install' })
    expect(statusLine({ ...base, link: 'connecting' }).title).toBe('Reconnecting to jeff-ai…')
    expect(statusLine({ ...base, link: 'stopped' }).title).toBe('Not connected to jeff-ai')
  })

  it('turns markdown into one line', () => {
    expect(plainText('# Done\n\nSee [the page](http://x) and:\n```js\ncode()\n```\n- **one**')).toBe('Done See the page and: - one')
  })
})
