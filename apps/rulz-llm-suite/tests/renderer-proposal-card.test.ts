import { describe, expect, it } from 'vitest'
import { dropPlanItem, isPlanList, planEmptied } from '../src/renderer/src/lib/proposalPlan'
import type { Proposal } from '../src/renderer/src/stores/novel'

const base = { id: 'p1', status: 'ready', raw: '', error: null, summaryOf: '', recId: 'r1', targetId: null, title: 'Plan', again: { count: 3, note: '' } } as const
const chapters = (n: number): Proposal =>
  ({ ...base, kind: 'chapters', chapters: { chapters: Array.from({ length: n }, (_, i) => ({ title: `Chapter ${i + 1}`, act: '', summary: `About ${i + 1}` })) } }) as Proposal
const scenes = (n: number): Proposal =>
  ({ ...base, kind: 'scenes', targetId: 'c1', scenes: { scenes: Array.from({ length: n }, (_, i) => ({ title: `Scene ${i + 1}`, beats: 'b', pov: '', present: [] })) } }) as Proposal

describe('planned chapters and scenes: leaving one out', () => {
  it('drops only the chapter asked for and keeps the rest in order', () => {
    const p = chapters(3)
    const next = dropPlanItem(p, 1)
    expect(next.chapters!.chapters.map((c) => c.title)).toEqual(['Chapter 1', 'Chapter 3'])
    // The proposal itself stays, so the rest can still be accepted.
    expect(next.id).toBe(p.id)
    expect(next.status).toBe('ready')
    expect(p.chapters!.chapters).toHaveLength(3)
    expect(planEmptied(next)).toBe(false)
  })

  it('drops only the scene asked for', () => {
    const next = dropPlanItem(scenes(2), 0)
    expect(next.scenes!.scenes.map((s) => s.title)).toEqual(['Scene 2'])
  })

  it('has nothing to accept once every item is left out', () => {
    expect(planEmptied(dropPlanItem(chapters(1), 0))).toBe(true)
    expect(planEmptied(dropPlanItem(scenes(1), 0))).toBe(true)
  })

  it('leaves a summary alone and does not treat it as a list', () => {
    const summary = { ...base, kind: 'summaryScene', targetId: 's1', summary: { summary: 'Text.' } } as Proposal
    expect(isPlanList(summary)).toBe(false)
    expect(dropPlanItem(summary, 0)).toBe(summary)
    expect(planEmptied(summary)).toBe(false)
    expect(isPlanList(chapters(2))).toBe(true)
  })
})
