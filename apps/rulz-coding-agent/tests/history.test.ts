import { describe, expect, it } from 'vitest'
import { DELETED_REPLY, dropExchange, dropReply, exchangeRange, forModel, whyNotDelete, whyNotRedo } from '@shared/history'
import type { TaskRecord, TimelineItem, WireMessage } from '@shared/types'

const task = (id: string, extra: Partial<TaskRecord> = {}): TaskRecord => ({ id, title: id, startedAt: 0, outcome: 'done', steps: [], files: [], ...extra })

const timeline: TimelineItem[] = [
  { id: 'u1', kind: 'user', text: 'one', taskId: 't1' },
  { id: 'a1', kind: 'assistant', text: 'reply one', taskId: 't1' },
  { id: 'n1', kind: 'user', text: 'a note', taskId: 't1', note: 'queued' },
  { id: 'x1', kind: 'notice', text: 'Stopped.', tone: 'info', taskId: 't1' },
  { id: 'u2', kind: 'user', text: 'two', taskId: 't2' },
  { id: 'a2', kind: 'assistant', text: 'reply two', taskId: 't2' }
]

const messages = (): WireMessage[] => [
  { role: 'user', content: 'one', task: 't1' },
  { role: 'assistant', content: 'calling', item: 'a1', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'list_files', arguments: '{}' } }] },
  { role: 'tool', tool_call_id: 'c1', content: 'files' },
  { role: 'user', content: '[Note: a note]' },
  { role: 'assistant', content: 'done one', item: 'a1b' },
  { role: 'user', content: 'two', task: 't2' },
  { role: 'assistant', content: 'reply two', item: 'a2' }
]

describe('which messages can be deleted', () => {
  it('lets replies go, and requests unless they changed files that are still there', () => {
    const tasks = [task('t1'), task('t2', { files: [{ path: 'a.txt', change: 'created' }] })]
    expect(whyNotDelete(timeline, tasks, 'a2', false)).toBeNull()
    expect(whyNotDelete(timeline, tasks, 'u1', false)).toBeNull()
    expect(whyNotDelete(timeline, tasks, 'u2', false)).toMatch(/changed files/)
    expect(whyNotDelete(timeline, [task('t1'), { ...tasks[1], undone: true }], 'u2', false)).toBeNull()
    expect(whyNotDelete(timeline, tasks, 'n1', false)).toMatch(/Notes belong/)
    expect(whyNotDelete(timeline, tasks, 'x1', false)).not.toBeNull()
    expect(whyNotDelete(timeline, tasks, 'a2', true)).toMatch(/Wait/)
  })

  it('runs only the latest request again, and only when its changes can be put back', () => {
    expect(whyNotRedo(timeline, [task('t1'), task('t2')], 'u2', false)).toBeNull()
    expect(whyNotRedo(timeline, [task('t1'), task('t2')], 'u1', false)).toMatch(/latest/)
    expect(whyNotRedo(timeline, [task('t1'), task('t2')], 'a2', false)).toMatch(/request/)
    expect(whyNotRedo(timeline, [task('t1'), task('t2')], 'u2', true)).toMatch(/Wait/)
    const changed = task('t2', { files: [{ path: 'a.txt', change: 'created' }] })
    expect(whyNotRedo(timeline, [changed], 'u2', false)).toMatch(/no restore point/)
    expect(whyNotRedo(timeline, [{ ...changed, snapshot: { kind: 'git', commit: 'abc' } }], 'u2', false)).toBeNull()
  })
})

describe('deleting', () => {
  it('finds a task’s messages up to the next task', () => {
    expect(exchangeRange(messages(), 't1')).toEqual({ from: 0, to: 5 })
    expect(exchangeRange(messages(), 't2')).toEqual({ from: 5, to: 7 })
    expect(exchangeRange(messages(), 'gone')).toBeNull()
  })

  it('drops a whole exchange, keeping the rest paired', () => {
    const data = { messages: messages(), timeline }
    expect(dropExchange(data, 't1')).toBe(true)
    expect(data.messages.map((m) => m.content)).toEqual(['two', 'reply two'])
    expect(data.timeline.map((i) => i.id)).toEqual(['u2', 'a2'])
    // Folded into the summary already: only the timeline changes.
    const later = { messages: messages().slice(5), timeline }
    expect(dropExchange(later, 't1')).toBe(false)
    expect(later.messages).toHaveLength(2)
  })

  it('blanks a reply but keeps its message and tool calls', () => {
    const data = { messages: messages(), timeline }
    dropReply(data, 'a1')
    dropReply(data, 'a1b')
    expect(data.timeline.some((i) => i.id === 'a1')).toBe(false)
    expect(data.messages).toHaveLength(7)
    expect(data.messages[1]).toMatchObject({ content: '', tool_calls: [{ id: 'c1' }] })
    expect(data.messages[4].content).toBe(DELETED_REPLY)
  })

  it('never sends the timeline links to the model', () => {
    const sent = forModel(messages())
    expect(sent.some((m) => 'task' in m || 'item' in m)).toBe(false)
    expect(sent[1].tool_calls).toHaveLength(1)
  })
})
