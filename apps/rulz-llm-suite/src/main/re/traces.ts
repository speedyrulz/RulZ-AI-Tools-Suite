// Training data for RE sessions: every finished model round is recorded as llama-server saw it,
// and graded sessions are exported as chat samples for fine-tuning.
import { promises as fs } from 'node:fs'
import { join } from 'node:path'
import type { ApiReSession, ReExportOptions, ReExportResult, ReKind, ReSession, ReTraceInfo, WireMessage } from '@shared/types'
import type { RecordedRound } from '../chat/service'
import { writeTextAtomic } from '../util/jsonFile'

const ID_RE = /^[\w-]{1,64}$/
const EMAIL_RE = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g

/** A recorded response line (the request is kept separately, only the latest one). */
export interface ResponseLine {
  requestId: string
  at: number
  content: string
  reasoning: string
  toolCalls: RecordedRound['response']['toolCalls']
  finishReason?: string
}

/** The latest request of a session; it holds the whole history the final answer was written from. */
export interface RequestFile {
  requestId: string
  at: number
  model: string
  messages: WireMessage[]
  tools: unknown[]
  /** Any round of the session was trimmed to fit the context. */
  trimmedEver: boolean
}

interface ToolDef {
  type: 'function'
  function: { name: string; description?: string; parameters?: unknown }
}

const toolName = (t: unknown): string => (t as ToolDef)?.function?.name ?? ''

/** A conversation's files; info() is cached against their sizes and dates. */
const TRACE_FILES = ['request.json', 'responses.jsonl', 'tools.json', 'session.json'] as const
/** Conversations whose recording state is kept in memory; each holds its tool definitions. */
const STATE_LIMIT = 64

/** What record() needs from the rounds before, kept so it doesn't re-read request.json every round. */
interface RecordState {
  trimmedEver: boolean
  /** Messages in the previous request; null when there was none. */
  messages: number | null
  tools: Record<string, unknown>
}

/**
 * Per conversation: `request.json` (the latest request, overwritten each round), `responses.jsonl`
 * (every reply, appended) and `tools.json` (every tool definition offered during the session, so
 * a sample still describes tools that were unloaded before the end).
 */
export class TraceStore {
  /** Writes per conversation, chained so rounds land in order. */
  private queues = new Map<string, Promise<void>>()
  private state = new Map<string, RecordState>()
  /**
   * info() results by the stats of the files they came from. The RE page lists every session again
   * whenever a chat is saved or an API round is recorded, and request.json can be megabytes.
   */
  private infoCache = new Map<string, { sig: string; info: ReTraceInfo }>()
  /** Chats deleted in this run: a round still finishing in one must not record it again. */
  private gone = new Set<string>()
  /**
   * Writes done per conversation, part of the info cache key: a session.json rewritten at the same
   * size within one file-time tick ("solved" → "failed") would otherwise keep the old entry.
   */
  private writes = new Map<string, number>()

  constructor(private readonly dir: string) {}

  private folder(id: string): string {
    if (!ID_RE.test(id)) throw new Error(`Invalid conversation id: ${id}`)
    return join(this.dir, id)
  }

  /** Runs `task` after the conversation's earlier writes; a failure doesn't block later ones. */
  private enqueue<T>(id: string, task: () => Promise<T>): Promise<T> {
    const next = (this.queues.get(id) ?? Promise.resolve()).then(task).finally(() => this.writes.set(id, (this.writes.get(id) ?? 0) + 1))
    const tail = next.then(
      () => undefined,
      () => undefined
    )
    this.queues.set(id, tail)
    void tail.then(() => {
      if (this.queues.get(id) === tail) this.queues.delete(id)
    })
    return next
  }

  /**
   * Records a finished round. With `shrinkIsTrim` (API sessions, where the other app fits its own
   * context), a request with fewer messages than the one before counts as trimmed.
   */
  record(conversationId: string, round: RecordedRound, opts: { shrinkIsTrim?: boolean } = {}): Promise<void> {
    const dir = this.folder(conversationId)
    return this.enqueue(conversationId, async () => {
      if (this.gone.has(conversationId)) return
      await fs.mkdir(dir, { recursive: true })
      let st = this.state.get(conversationId)
      if (!st) {
        const old = await readJsonFile<RequestFile | null>(join(dir, 'request.json'), null)
        st = {
          trimmedEver: !!old?.trimmedEver,
          messages: Array.isArray(old?.messages) ? old.messages.length : null,
          tools: await readJsonFile<Record<string, unknown>>(join(dir, 'tools.json'), {})
        }
      }
      const req: RequestFile = {
        requestId: round.requestId,
        at: round.at,
        model: round.model,
        messages: round.messages,
        tools: round.tools,
        trimmedEver:
          round.trimmed || st.trimmedEver || (!!opts.shrinkIsTrim && st.messages !== null && round.messages.length < st.messages)
      }
      const tools = { ...st.tools }
      let toolsChanged = false
      for (const t of round.tools) {
        const n = toolName(t)
        if (n && JSON.stringify(tools[n]) !== JSON.stringify(t)) {
          tools[n] = t
          toolsChanged = true
        }
      }
      const line: ResponseLine = { requestId: round.requestId, at: round.at, ...round.response }
      // The reply goes first: a request.json whose reply is missing would make a broken sample.
      await fs.appendFile(join(dir, 'responses.jsonl'), JSON.stringify(line) + '\n', 'utf8')
      await writeTextAtomic(join(dir, 'request.json'), JSON.stringify(req))
      if (toolsChanged) await writeTextAtomic(join(dir, 'tools.json'), JSON.stringify(tools))
      // Most recently recorded last; the oldest go past STATE_LIMIT (a miss just reads the files again).
      this.state.delete(conversationId)
      this.state.set(conversationId, { trimmedEver: req.trimmedEver, messages: round.messages.length, tools })
      for (const old of this.state.keys()) {
        if (this.state.size <= STATE_LIMIT) break
        this.state.delete(old)
      }
    }).catch((err) => {
      // Read the files again next round rather than trust a state the failed write may not match.
      this.state.delete(conversationId)
      console.error(`Could not record RE data for ${conversationId}:`, err)
    })
  }

  /** The description of a session another app ran through the API server, if it has one. */
  apiSession(id: string): Promise<ApiReSession | null> {
    return readJsonFile<ApiReSession | null>(join(this.folder(id), 'session.json'), null)
  }

  /** Creates or updates an API session's description; queued with its rounds. */
  updateApiSession(id: string, patch: Partial<ApiReSession>): Promise<ApiReSession> {
    const dir = this.folder(id)
    return this.enqueue(id, async () => {
      await fs.mkdir(dir, { recursive: true })
      const old = await readJsonFile<ApiReSession | null>(join(dir, 'session.json'), null)
      const defined = Object.fromEntries(Object.entries(patch).filter(([, v]) => v !== undefined)) as Partial<ApiReSession>
      const result: ApiReSession = { client: '', title: '', target: '', goal: '', outcome: 'unrated', createdAt: Date.now(), ...old, ...defined }
      await writeTextAtomic(join(dir, 'session.json'), JSON.stringify(result))
      return result
    })
  }

  /**
   * Gives an API session a description if it has none yet. Queued like updateApiSession, so a
   * description the app sends at the same moment is never overwritten by these defaults.
   */
  ensureApiSession(id: string, defaults: Partial<ApiReSession>): Promise<void> {
    const dir = this.folder(id)
    return this.enqueue(id, async () => {
      if (await readJsonFile<ApiReSession | null>(join(dir, 'session.json'), null)) return
      await fs.mkdir(dir, { recursive: true })
      const fresh: ApiReSession = { client: '', title: '', target: '', goal: '', outcome: 'unrated', createdAt: Date.now(), ...defaults }
      await writeTextAtomic(join(dir, 'session.json'), JSON.stringify(fresh))
    })
  }

  async info(conversationId: string): Promise<ReTraceInfo | null> {
    const dir = this.folder(conversationId)
    // Taken before the stat: a write finishing after it bumps the count, so its entry misses next time.
    const writes = this.writes.get(conversationId) ?? 0
    const stats = await Promise.all(TRACE_FILES.map((f) => fs.stat(join(dir, f)).catch(() => null)))
    const sig = `${writes}|` + stats.map((s) => (s ? `${s.size}:${s.mtimeMs}` : '-')).join('|')
    const hit = this.infoCache.get(conversationId)
    if (hit?.sig === sig) return { ...hit.info }
    const lines = await this.responses(conversationId)
    if (!lines) {
      this.infoCache.delete(conversationId)
      return null
    }
    const req = await readJsonFile<RequestFile | null>(join(dir, 'request.json'), null)
    const api = await this.apiSession(conversationId)
    const info: ReTraceInfo = {
      ...(api ? { api } : {}),
      conversationId,
      responses: lines.length,
      toolRounds: lines.filter((l) => l.toolCalls?.length).length,
      trimmed: !!req?.trimmedEver,
      bytes: stats.slice(0, 3).reduce((sum, s) => sum + (s?.size ?? 0), 0),
      updatedAt: req?.at ?? lines.at(-1)?.at ?? 0
    }
    // Files read after the stat can only be newer than it, so a stale entry is just read again.
    this.infoCache.set(conversationId, { sig, info })
    return { ...info }
  }

  async list(): Promise<ReTraceInfo[]> {
    const names = await fs.readdir(this.dir).catch(() => [] as string[])
    const out: ReTraceInfo[] = []
    for (const n of names) {
      if (!ID_RE.test(n)) continue
      const i = await this.info(n).catch(() => null)
      if (i) out.push(i)
    }
    return out
  }

  async responses(conversationId: string): Promise<ResponseLine[] | null> {
    const text = await fs.readFile(join(this.folder(conversationId), 'responses.jsonl'), 'utf8').catch(() => null)
    if (text === null) return null
    return text
      .split('\n')
      .filter((l) => l.trim())
      .flatMap((l) => {
        try {
          return [JSON.parse(l) as ResponseLine]
        } catch {
          return []
        }
      })
  }

  async load(conversationId: string): Promise<{ request: RequestFile; responses: ResponseLine[]; tools: Record<string, unknown> } | null> {
    const dir = this.folder(conversationId)
    await this.queues.get(conversationId)
    const request = await readJsonFile<RequestFile | null>(join(dir, 'request.json'), null)
    const responses = await this.responses(conversationId)
    if (!request || !responses) return null
    return { request, responses, tools: await readJsonFile(join(dir, 'tools.json'), {}) }
  }

  /**
   * Deletes a conversation's data, queued after its pending writes; `remove` (e.g. to the Recycle
   * Bin) is tried first, and a failed one falls back to deleting the folder. With `forever` (the
   * chat itself was deleted), rounds that still arrive are not recorded. Without it (only the trace
   * was deleted), a session that carries on is recorded again.
   */
  async delete(conversationId: string, remove?: (dir: string) => Promise<void>, opts: { forever?: boolean } = {}): Promise<void> {
    const dir = this.folder(conversationId)
    if (opts.forever) this.gone.add(conversationId)
    return this.enqueue(conversationId, async () => {
      this.state.delete(conversationId)
      this.infoCache.delete(conversationId)
      if (remove) {
        try {
          await remove(dir)
          return
        } catch {
          // e.g. no Recycle Bin on that drive, or nothing was recorded
        }
      }
      await fs.rm(dir, { recursive: true, force: true })
    })
  }
}

async function readJsonFile<T>(path: string, fallback: T): Promise<T> {
  const text = await fs.readFile(path, 'utf8').catch(() => null)
  if (text === null) return fallback
  try {
    return JSON.parse(text) as T
  } catch {
    return fallback
  }
}

// ---------- Samples ----------

/** OpenAI / Hugging Face chat sample, the format the RE training kit writes (train.jsonl). */
export interface TrainingSample {
  messages: (WireMessage & { reasoning_content?: string })[]
  tools: unknown[]
  meta: Record<string, unknown>
}

/**
 * Builds a sample from a session's latest request and its reply. Thinking the app did not send
 * back (older turns) is filled in from the recorded replies, matched by tool-call id or by text.
 * Returns the reason when the session can't make a sample.
 */
export function buildSample(
  request: RequestFile,
  responses: ResponseLine[],
  toolsSeen: Record<string, unknown>,
  meta: Record<string, unknown>
): TrainingSample | string {
  const final = responses.at(-1)
  if (!final || final.requestId !== request.requestId) return 'the latest request has no recorded reply'
  if (final.toolCalls?.length) return 'the session ended on a tool call, without an answer'
  if (!final.content.trim()) return 'the final reply is empty'
  if (final.finishReason === 'length') return 'the final reply was cut off'

  const byCall = new Map<string, string>()
  const byText = new Map<string, string>()
  for (const r of responses) {
    if (!r.reasoning) continue
    for (const c of r.toolCalls ?? []) byCall.set(c.id, r.reasoning)
    if (!r.toolCalls?.length && r.content.trim()) byText.set(r.content.trim(), r.reasoning)
  }

  const messages: TrainingSample['messages'] = request.messages.map((m) => {
    if (m.role !== 'assistant' || m.reasoning_content) return m
    const id = m.tool_calls?.[0]?.id
    const text = typeof m.content === 'string' ? m.content.trim() : ''
    const reasoning = (id && byCall.get(id)) || (text && byText.get(text)) || ''
    return reasoning ? { ...m, reasoning_content: reasoning } : m
  })
  messages.push({ role: 'assistant', content: final.content, ...(final.reasoning ? { reasoning_content: final.reasoning } : {}) })

  // The tools offered at the end, plus any the session called that were unloaded before then.
  const tools = [...request.tools]
  const have = new Set(tools.map(toolName))
  const called = new Set(messages.flatMap((m) => m.tool_calls?.map((c) => c.function.name) ?? []))
  for (const n of called) {
    if (!have.has(n) && toolsSeen[n]) {
      tools.push(toolsSeen[n])
      have.add(n)
    }
  }
  return { messages, tools, meta: { ...meta, model: request.model } }
}

/** Replaces email addresses, as the kit does. */
export function scrub(text: string): string {
  return text.replace(EMAIL_RE, 'user@example.com')
}

/**
 * A sample as one JSONL line, with email addresses replaced in its strings. Scrubbing the finished
 * line instead would take the letter of an escape like "\nuser@host" into the match and leave an
 * invalid "\u" escape behind.
 */
export function sampleLine(sample: TrainingSample): string {
  return JSON.stringify(sample, (_key, value: unknown) => (typeof value === 'string' ? scrub(value) : value))
}

/** A session to export: a chat with its RE details, or an API session (`source` names the app). */
export interface ExportSession {
  id: string
  re?: ReSession
  source?: string
  /** What kind of session another app ran (chats here derive it from `re.clone`). */
  kind?: ReKind
}

/** Writes graded RE sessions as train.jsonl (plus a manifest) in `opts.outDir`. */
export async function exportDataset(
  traces: TraceStore,
  sessions: ExportSession[],
  opts: ReExportOptions
): Promise<ReExportResult> {
  const dropped: Record<string, number> = {}
  const drop = (why: string): void => {
    dropped[why] = (dropped[why] ?? 0) + 1
  }
  const outcomes = new Set(opts.outcomes)
  type Candidate = { sample: TrainingSample; rounds: number; task: string | null }
  const candidates: Candidate[] = []
  for (const c of sessions) {
    const re = c.re
    if (!re) continue
    if (!outcomes.has(re.outcome)) {
      drop(`outcome ${re.outcome}`)
      continue
    }
    const data = await traces.load(c.id)
    if (!data) {
      drop('nothing recorded')
      continue
    }
    if (opts.skipTrimmed && data.request.trimmedEver) {
      drop('context was trimmed')
      continue
    }
    const meta: Record<string, unknown> = {
      conversation: c.id,
      ...(c.source ? { source: c.source } : {}),
      outcome: re.outcome,
      graded: re.autoGraded ? 'auto' : 'user',
      ...(re.practice ? { task: re.practice.taskId, family: re.practice.family, tier: re.practice.tier } : {}),
      ...(re.clone
        ? { kind: re.clone.scope === 'full' ? 'clone-full' : 'clone-pieces', ...(re.clone.language ? { language: re.clone.language } : {}) }
        : c.kind === 'clone-full' || c.kind === 'clone-pieces'
          ? { kind: c.kind }
          : {})
    }
    const sample = buildSample(data.request, data.responses, data.tools, meta)
    if (typeof sample === 'string') {
      drop(sample)
      continue
    }
    const rounds = data.responses.filter((r) => r.toolCalls?.length).length
    candidates.push({ sample, rounds, task: re.practice?.taskId ?? null })
  }

  // Per practice task, the sessions with the fewest tool rounds (the kit's --max-per-task).
  const perTask = new Map<string, number>()
  candidates.sort((a, b) => a.rounds - b.rounds)
  const kept: TrainingSample[] = []
  for (const cand of candidates) {
    if (cand.task && opts.maxPerTask > 0) {
      const n = perTask.get(cand.task) ?? 0
      if (n >= opts.maxPerTask) {
        drop('more sessions than the per-task limit')
        continue
      }
      perTask.set(cand.task, n + 1)
    }
    kept.push(cand.sample)
  }

  await fs.mkdir(opts.outDir, { recursive: true })
  const trainPath = join(opts.outDir, 'train.jsonl')
  const manifestPath = join(opts.outDir, 'manifest.json')
  await fs.writeFile(trainPath, kept.map(sampleLine).join('\n') + (kept.length ? '\n' : ''), 'utf8')
  await fs.writeFile(
    manifestPath,
    JSON.stringify(
      {
        createdAt: new Date().toISOString(),
        format: 'OpenAI/Hugging Face chat (messages, tools, meta); assistant thinking in reasoning_content',
        samples: kept.length,
        outcomes: opts.outcomes,
        skipTrimmed: opts.skipTrimmed,
        maxPerTask: opts.maxPerTask,
        dropped
      },
      null,
      2
    ),
    'utf8'
  )
  return { outDir: opts.outDir, kept: kept.length, dropped, files: [trainPath, manifestPath] }
}
