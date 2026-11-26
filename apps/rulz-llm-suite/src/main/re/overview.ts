// What other apps (RulZ Coding Agent and its phone app) see of RE mode: every session with its
// grade, the practice tasks with how often each was solved, and the prompt templates.
import { keyOfModel, modelKey, practiceOf, type PracticeSources } from '@shared/practice'
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
  chats: { id: string; title: string; updatedAt: number; re: ReSession; modelId?: string | null }[]
  traces: ReTraceInfo[]
  practice: RePracticeInfo
  recording: boolean
  /**
   * Count practice for one model only: its library id, or else the text the client sent (a name). The counts then come
   * from the same records as the Reverse Engineering page, so they also include what batch runs scored and what
   * other apps recorded; `summaries`, `runs` and `models` are what those are read from.
   */
  model?: string | null
  summaries?: PracticeSources['summaries']
  runs?: PracticeSources['runs']
  models?: PracticeSources['models']
}): ReOverview {
  const byId = new Map(input.traces.map((t) => [t.conversationId, t]))
  const library = input.models ?? []
  // A batch run's chat counts for the run's model, as on the Reverse Engineering page.
  const runModel = new Map<string, string>()
  for (const run of input.runs ?? []) {
    const key = run.modelName ? modelKey(run.modelName) : run.modelId ? keyOfModel(run.modelId, library) : ''
    if (key) for (const i of run.items) if (i.conversationId) runModel.set(i.conversationId, key)
  }
  const sessions: ReSessionSummary[] = []
  for (const c of input.chats) {
    const t = byId.get(c.id)
    const p = c.re.practice
    const own = runModel.get(c.id) ?? (c.modelId ? keyOfModel(c.modelId, library) : '')
    const keys = own ? [own] : (t?.models ?? []).map(modelKey)
    sessions.push({
      id: c.id,
      title: c.title,
      source: SUITE_SOURCE,
      kind: kindOf(c.re),
      target: c.re.target,
      goal: c.re.goal,
      outcome: c.re.outcome,
      ...(p ? { practice: { taskId: p.taskId, family: p.family, tier: p.tier, answer: p.answer, expected: p.expected } } : {}),
      models: [...new Set(keys)],
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
      models: [...new Set(t.models.map(modelKey))],
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
  // One model: its attempts, solved and failed per task, read from the records the Reverse Engineering page counts.
  const asked = input.model?.trim()
  const perModel = new Map<string, { attempts: number; solved: number; failed: number }>()
  let modelKeyUsed: string | undefined
  if (asked) {
    const sources: PracticeSources = { summaries: input.summaries ?? [], traces: input.traces, runs: input.runs ?? [], models: library }
    for (const r of practiceOf(asked, sources)) {
      const st = perModel.get(r.taskId) ?? { attempts: 0, solved: 0, failed: 0 }
      st.attempts++
      if (r.outcome === 'solved') st.solved++
      else if (r.outcome === 'failed') st.failed++
      perModel.set(r.taskId, st)
    }
    modelKeyUsed = keyOfModel(asked, library)
  }
  return {
    object: 're.overview',
    recording: input.recording,
    sessions: sessions.slice(0, MAX_OVERVIEW_SESSIONS),
    practice: {
      kitDir: input.practice.kitDir,
      ...(input.practice.error ? { error: input.practice.error } : {}),
      ...(input.practice.warning ? { warning: input.practice.warning } : {}),
      sets: input.practice.sets,
      ...(modelKeyUsed !== undefined ? { model: modelKeyUsed } : {}),
      tasks: input.practice.tasks.map((t) => ({
        ...t,
        ...(modelKeyUsed !== undefined ? (perModel.get(t.id) ?? { attempts: 0, solved: 0, failed: 0 }) : (stats.get(t.id) ?? { attempts: 0, solved: 0 }))
      }))
    },
    goals: RE_GOALS,
    clonePieces: CLONE_PIECES,
    cloneLanguages: CLONE_LANGUAGES
  }
}
