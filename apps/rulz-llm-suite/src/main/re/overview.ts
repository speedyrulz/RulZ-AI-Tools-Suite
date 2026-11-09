// What other apps (RulZ Coding Agent and its phone app) see of RE mode: every session with its
// grade, the practice tasks with how often each was solved, and the prompt templates.
import { CLONE_LANGUAGES, CLONE_PIECES, RE_GOALS } from '@shared/re'
import type { RePracticeInfo, ReKind, ReOverview, ReSession, ReSessionSummary, ReTraceInfo } from '@shared/types'

export const SUITE_SOURCE = 'RulZ LLM Suite'
/** Sessions listed at most (newest first). */
export const MAX_OVERVIEW_SESSIONS = 300

export function kindOf(re: Pick<ReSession, 'practice' | 'clone'>): ReKind {
  if (re.practice) return 'practice'
  if (re.clone) return re.clone.scope === 'full' ? 'clone-full' : 'clone-pieces'
  return 'analyse'
}

export function buildOverview(input: {
  chats: { id: string; title: string; updatedAt: number; re: ReSession }[]
  traces: ReTraceInfo[]
  practice: RePracticeInfo
  recording: boolean
}): ReOverview {
  const byId = new Map(input.traces.map((t) => [t.conversationId, t]))
  const sessions: ReSessionSummary[] = []
  for (const c of input.chats) {
    const t = byId.get(c.id)
    const p = c.re.practice
    sessions.push({
      id: c.id,
      title: c.title,
      source: SUITE_SOURCE,
      kind: kindOf(c.re),
      target: c.re.target,
      goal: c.re.goal,
      outcome: c.re.outcome,
      ...(p ? { practice: { taskId: p.taskId, family: p.family, tier: p.tier, answer: p.answer, expected: p.expected } } : {}),
      responses: t?.responses ?? 0,
      toolRounds: t?.toolRounds ?? 0,
      updatedAt: Math.max(c.updatedAt, t?.updatedAt ?? 0)
    })
  }
  for (const t of input.traces) {
    const a = t.api
    if (!a) continue
    sessions.push({
      id: t.conversationId,
      title: a.title,
      source: a.client,
      kind: a.kind ?? (a.practice ? 'practice' : 'analyse'),
      target: a.target,
      goal: a.goal,
      outcome: a.outcome,
      ...(a.practice ? { practice: a.practice } : {}),
      responses: t.responses,
      toolRounds: t.toolRounds,
      updatedAt: t.updatedAt
    })
  }
  sessions.sort((x, y) => y.updatedAt - x.updatedAt)

  const stats = new Map<string, { attempts: number; solved: number }>()
  for (const s of sessions) {
    if (!s.practice) continue
    const st = stats.get(s.practice.taskId) ?? { attempts: 0, solved: 0 }
    st.attempts++
    if (s.outcome === 'solved') st.solved++
    stats.set(s.practice.taskId, st)
  }
  return {
    object: 're.overview',
    recording: input.recording,
    sessions: sessions.slice(0, MAX_OVERVIEW_SESSIONS),
    practice: {
      kitDir: input.practice.kitDir,
      ...(input.practice.error ? { error: input.practice.error } : {}),
      tasks: input.practice.tasks.map((t) => ({ ...t, ...(stats.get(t.id) ?? { attempts: 0, solved: 0 }) }))
    },
    goals: RE_GOALS,
    clonePieces: CLONE_PIECES,
    cloneLanguages: CLONE_LANGUAGES
  }
}
