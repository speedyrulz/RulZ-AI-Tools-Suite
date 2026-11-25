import { describe, expect, it } from 'vitest'
import type { BatchItem } from '@shared/batch'
import type { ReTraceInfo } from '@shared/types'
import { keyOfModel, modelKey, practiceOf, practiceRecords, solvedTasks, taskRecords, type PracticeSources } from '../src/renderer/src/stores/reTraces'

const apiTrace = (id: string, taskId: string, outcome: 'solved' | 'failed' | 'unrated', models: string[] = ['Orn 1.5']): Pick<ReTraceInfo, 'conversationId' | 'api' | 'models'> => ({
  conversationId: id,
  models,
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

const from = (s: Partial<PracticeSources>): PracticeSources => ({ summaries: [], traces: [], runs: [], models: [], ...s })

describe('practice records', () => {
  it('counts practice sessions other apps ran through the API server', () => {
    const records = practiceRecords(
      from({
        summaries: [{ id: 'c1', rePracticeTask: 'keygen-01', reOutcome: 'failed' }, { id: 'c2' }],
        traces: [apiTrace('a1', 'keygen-02', 'solved'), apiTrace('a2', 'hash-01', 'unrated', []), { conversationId: 'c1', models: ['x'] }]
      })
    )
    expect(records).toEqual([
      { taskId: 'keygen-01', outcome: 'failed', models: [] },
      { taskId: 'keygen-02', outcome: 'solved', models: ['orn-1.5'] },
      { taskId: 'hash-01', outcome: 'unrated', models: [] }
    ])
    expect([...solvedTasks(records)]).toEqual(['keygen-02'])
  })

  it('does not count a session twice when it is also a chat here', () => {
    const records = practiceRecords(from({ summaries: [{ id: 'a1', rePracticeTask: 'keygen-02', reOutcome: 'solved' }], traces: [apiTrace('a1', 'keygen-02', 'solved')] }))
    expect(records).toHaveLength(1)
  })

  it('skips recorded sessions that are not practice tasks', () => {
    const t = apiTrace('a1', 'x', 'solved')
    delete t.api!.practice
    expect(practiceRecords(from({ traces: [t] }))).toEqual([])
  })
})

describe('practice results per model', () => {
  const models = [
    { id: 'RE/Orn 1.5/Orn-1.5-ICE.gguf', name: 'Orn-1.5-ICE' },
    { id: 'vast/orn-1.5-ice', name: 'Orn-1.5-ICE' },
    { id: 'Qwen/Qwen.gguf', name: 'Qwen' }
  ]
  const item = (taskId: string, status: BatchItem['status'], conversationId?: string): BatchItem => ({ taskId, family: 'f', tier: 't', attempt: 1, status, conversationId })

  it('gives a model on this PC, its copy on a rental and its moved file the same key', () => {
    const key = modelKey('Orn-1.5-ICE')
    for (const id of [...models.slice(0, 2).map((m) => m.id), 'Old folder/Orn-1.5-ICE.gguf', 'vast/orn-1.5-ice']) {
      expect(keyOfModel(id, models)).toBe(key)
      expect(keyOfModel(id, [])).toBe(key)
    }
    expect(modelKey('a\\Big-00001-of-00003.gguf')).toBe('big')
    expect(keyOfModel('Qwen/Qwen.gguf', models)).not.toBe(key)
    expect(modelKey('Orn-1.5-ICE-Q4')).not.toBe(key)
  })

  it('counts only the sessions of the chosen model, here, on a rental and in other apps', () => {
    const s = from({
      models,
      summaries: [
        { id: 'c1', rePracticeTask: 'one', reOutcome: 'solved', modelId: models[0].id },
        { id: 'c2', rePracticeTask: 'two', reOutcome: 'solved', modelId: models[1].id },
        { id: 'c3', rePracticeTask: 'three', reOutcome: 'solved', modelId: models[2].id },
        { id: 'c4', rePracticeTask: 'four', reOutcome: 'solved', modelId: null }
      ],
      traces: [apiTrace('a1', 'five', 'solved', ['Orn-1.5-ICE'])]
    })
    for (const id of [models[0].id, models[1].id]) expect([...solvedTasks(practiceOf(id, s))]).toEqual(['one', 'two', 'five'])
    expect([...solvedTasks(practiceOf(models[2].id, s))]).toEqual(['three'])
    expect(solvedTasks(practiceOf(null, s)).size).toBe(5)
  })

  it('keeps the results of batch runs whose chats were deleted, and does not count a kept chat twice', () => {
    const run = {
      modelId: models[1].id,
      modelName: 'Orn-1.5-ICE',
      items: [
        item('kept', 'solved', 'c1'),
        item('gone-solved', 'solved', 'x1'),
        item('gone-wrong', 'wrong', 'x2'),
        item('gone-rounds', 'rounds', 'x3'),
        item('broke', 'error', 'x4'),
        item('stopped', 'stopped', 'x5'),
        item('later', 'pending')
      ]
    }
    // A follow-up with another model in the run's chat does not move its result to that model.
    const s = from({ models, runs: [run], summaries: [{ id: 'c1', rePracticeTask: 'kept', reOutcome: 'solved', modelId: models[2].id }] })
    const t = taskRecords(practiceOf(models[0].id, s))
    expect(Object.fromEntries(t)).toEqual({
      'gone-solved': { attempts: 1, status: 'solved' },
      'gone-wrong': { attempts: 1, status: 'failed' },
      'gone-rounds': { attempts: 1, status: 'failed' },
      kept: { attempts: 1, status: 'solved' }
    })
    expect(practiceOf(models[2].id, s)).toEqual([])
  })

  it('a solved task stays solved after a failed try, and only ungraded tries leave it tried', () => {
    const r = (outcome: string, taskId = 't') => ({ taskId, outcome, models: [] })
    expect(taskRecords([r('solved'), r('failed')]).get('t')).toEqual({ attempts: 2, status: 'solved' })
    expect(taskRecords([r('failed'), r('unrated')]).get('t')?.status).toBe('failed')
    expect(taskRecords([r('unrated'), r('partial')]).get('t')?.status).toBe('tried')
  })
})
