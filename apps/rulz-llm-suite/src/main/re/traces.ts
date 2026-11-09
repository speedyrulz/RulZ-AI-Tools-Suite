// Training data for RE sessions: every finished model round is recorded as llama-server saw it,
// and graded sessions are exported as chat samples for fine-tuning.
import { promises as fs } from 'node:fs'
import { join } from 'node:path'
import type { ApiReSession, ReExportOptions, ReExportResult, ReSession, ReTraceInfo, WireMessage } from '@shared/types'
import type { RecordedRound } from '../chat/service'

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

/**
 * Per conversation: `request.json` (the latest request, overwritten each round), `responses.jsonl`
 * (every reply, appended) and `tools.json` (every tool definition offered during the session, so
 * a sample still describes tools that were unloaded before the end).
 */
export class TraceStore {
  /** Writes per conversation, chained so rounds land in order. */
  private queues = new Map<string, Promise<void>>()

  constructor(private readonly dir: string) {}

  private folder(id: string): string {
    if (!ID_RE.test(id)) throw new Error(`Invalid conversation id: ${id}`)
    return join(this.dir, id)
  }

  /**
   * Records a finished round. With `shrinkIsTrim` (API sessions, where the other app fits its own
   * context), a request with fewer messages than the one before counts as trimmed.
   */
  record(conversationId: string, round: RecordedRound, opts: { shrinkIsTrim?: boolean } = {}): Promise<void> {
    const dir = this.folder(conversationId)
    const prev = this.queues.get(conversationId) ?? Promise.resolve()
    const next = prev
      .then(async () => {
        await fs.mkdir(dir, { recursive: true })
        const old = await readJsonFile<RequestFile | null>(join(dir, 'request.json'), null)
        const req: RequestFile = {
          requestId: round.requestId,
          at: round.at,
          model: round.model,
          messages: round.messages,
          tools: round.tools,
          trimmedEver: round.trimmed || !!old?.trimmedEver || (!!opts.shrinkIsTrim && !!old && round.messages.length < old.messages.length)
        }
        const tools = await readJsonFile<Record<string, unknown>>(join(dir, 'tools.json'), {})
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
        await writeFileAtomic(join(dir, 'request.json'), JSON.stringify(req))
        if (toolsChanged) await writeFileAtomic(join(dir, 'tools.json'), JSON.stringify(tools))
      })
      .catch((err) => console.error(`Could not record RE data for ${conversationId}:`, err))
    this.queues.set(conversationId, next)
    return next
  }

  /** The description of a session another app ran through the API server, if it has one. */
  apiSession(id: string): Promise<ApiReSession | null> {
    return readJsonFile<ApiReSession | null>(join(this.folder(id), 'session.json'), null)
  }

  /** Creates or updates an API session's description; queued with its rounds. */
  updateApiSession(id: string, patch: Partial<ApiReSession>): Promise<ApiReSession> {
    const dir = this.folder(id)
    const prev = this.queues.get(id) ?? Promise.resolve()
    let result!: ApiReSession
    const next = prev.then(async () => {
      await fs.mkdir(dir, { recursive: true })
      const old = await readJsonFile<ApiReSession | null>(join(dir, 'session.json'), null)
      const defined = Object.fromEntries(Object.entries(patch).filter(([, v]) => v !== undefined)) as Partial<ApiReSession>
      result = { client: '', title: '', target: '', goal: '', outcome: 'unrated', createdAt: Date.now(), ...old, ...defined }
      await writeFileAtomic(join(dir, 'session.json'), JSON.stringify(result))
    })
    this.queues.set(
      id,
      next.catch(() => undefined)
    )
    return next.then(() => result)
  }

  async info(conversationId: string): Promise<ReTraceInfo | null> {
    const dir = this.folder(conversationId)
    const lines = await this.responses(conversationId)
    if (!lines) return null
    const req = await readJsonFile<RequestFile | null>(join(dir, 'request.json'), null)
    let bytes = 0
    for (const f of ['request.json', 'responses.jsonl', 'tools.json']) bytes += (await fs.stat(join(dir, f)).catch(() => null))?.size ?? 0
    const api = await this.apiSession(conversationId)
    return {
      ...(api ? { api } : {}),
      conversationId,
      responses: lines.length,
      toolRounds: lines.filter((l) => l.toolCalls?.length).length,
      trimmed: !!req?.trimmedEver,
      bytes,
      updatedAt: req?.at ?? lines.at(-1)?.at ?? 0
    }
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

  async delete(conversationId: string): Promise<void> {
    await this.queues.get(conversationId)
    await fs.rm(this.folder(conversationId), { recursive: true, force: true })
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

async function writeFileAtomic(path: string, text: string): Promise<void> {
  const tmp = `${path}.${process.pid}.tmp`
  await fs.writeFile(tmp, text, 'utf8')
  await fs.rename(tmp, path)
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
export function scrub(line: string): string {
  return line.replace(EMAIL_RE, 'user@example.com')
}

/** A session to export: a chat with its RE details, or an API session (`source` names the app). */
export interface ExportSession {
  id: string
  re?: ReSession
  source?: string
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
      ...(re.practice ? { task: re.practice.taskId, family: re.practice.family, tier: re.practice.tier } : {})
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
  await fs.writeFile(trainPath, kept.map((s) => scrub(JSON.stringify(s))).join('\n') + (kept.length ? '\n' : ''), 'utf8')
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
