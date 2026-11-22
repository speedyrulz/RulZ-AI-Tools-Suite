import { describe, expect, it } from 'vitest'
import type { ReTraceInfo } from '@shared/types'
import { practiceRecords, solvedTasks } from '../src/renderer/src/stores/reTraces'

const apiTrace = (id: string, taskId: string, outcome: 'solved' | 'failed' | 'unrated'): Pick<ReTraceInfo, 'conversationId' | 'api'> => ({
  conversationId: id,
  api: {
    client: 'RulZ Coding Agent',
    title: `Practice: ${taskId}`,
    target: 'x.exe',
    goal: 'g',
    outcome,
    createdAt: 0,
    kind: 'practice',
    practice: { taskId, family: 'keygen', tier: 't1' }
  }
})

describe('practice records', () => {
  it('counts practice sessions other apps ran through the API server', () => {
    const records = practiceRecords(
      [{ id: 'c1', rePracticeTask: 'keygen-01', reOutcome: 'failed' }, { id: 'c2' }],
      [apiTrace('a1', 'keygen-02', 'solved'), apiTrace('a2', 'hash-01', 'unrated'), { conversationId: 'c1' }]
    )
    expect(records).toEqual([
      { taskId: 'keygen-01', outcome: 'failed' },
      { taskId: 'keygen-02', outcome: 'solved' },
      { taskId: 'hash-01', outcome: 'unrated' }
    ])
    expect([...solvedTasks(records)]).toEqual(['keygen-02'])
  })

  it('does not count a session twice when it is also a chat here', () => {
    const records = practiceRecords([{ id: 'a1', rePracticeTask: 'keygen-02', reOutcome: 'solved' }], [apiTrace('a1', 'keygen-02', 'solved')])
    expect(records).toHaveLength(1)
  })

  it('skips recorded sessions that are not practice tasks', () => {
    const t = apiTrace('a1', 'x', 'solved')
    delete t.api!.practice
    expect(practiceRecords([], [t])).toEqual([])
  })
})
