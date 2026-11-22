import type { Proposal } from '../stores/novel'

/** A planned-chapters or planned-scenes proposal: its items can be left out one by one. */
export const isPlanList = (p: Proposal): boolean => p.chapters !== undefined || p.scenes !== undefined

/** The proposal with one planned chapter or scene left out; the others stay as they are. */
export function dropPlanItem(p: Proposal, index: number): Proposal {
  if (p.chapters) return { ...p, chapters: { chapters: p.chapters.chapters.filter((_, j) => j !== index) } }
  if (p.scenes) return { ...p, scenes: { scenes: p.scenes.scenes.filter((_, j) => j !== index) } }
  return p
}

/** A plan whose items were all left out has nothing to accept. */
export const planEmptied = (p: Proposal): boolean =>
  (p.chapters !== undefined && p.chapters.chapters.length === 0) || (p.scenes !== undefined && p.scenes.scenes.length === 0)
