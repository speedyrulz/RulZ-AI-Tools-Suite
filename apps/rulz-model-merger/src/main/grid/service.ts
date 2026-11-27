// Sample grids: render a merge at several values of one setting (columns) for 1 to 4 prompts (rows) with the user's own
// ComfyUI workflow as the template. Each cell is one ComfyUI prompt (merge + the template's sampler); no model file is
// saved. Cells run one at a time, column by column, so ComfyUI can keep the loaded models between prompts.
//
// Only this app's own prompts are touched: cancelling deletes this grid's pending prompt by id and interrupts by id
// (ComfyUI interrupts only when that id is the running one). The queue is never cleared and models are never freed.
import { EventEmitter } from 'node:events'
import { mkdir, readdir, readFile, rename, stat, writeFile } from 'node:fs/promises'
import { basename, join } from 'node:path'
import type { ServerCapabilities } from '@shared/types'
import {
  GRID_MAX_PROMPTS,
  GRID_MAX_VALUES,
  GRID_MIN_VALUES,
  GRID_PROMPT_MAX,
  recipeForValue,
  type GridCell,
  type GridJob,
  type GridRecord,
  type GridRequest,
  type GridTemplateInfo
} from '@shared/grid'
import type { MergeRecipe } from '@shared/merge'
import { parseCapabilities } from '../comfy/parse'
import type { HistoryEntry, HistoryImage } from '../comfy/client'
import { blockNodeOptions } from '../merge/inspect'
import { completeBlocks, validateRecipe } from '../merge/graph'
import type { MergeClient, MergeConnection } from '../merge/runner'
import { buildGridPrompt, checkTemplateForRecipe, inspectTemplate } from './template'

/** The part of ComfyClient the grid uses: the runner's, plus reading an image. */
export interface GridClient extends MergeClient {
  viewImage(img: HistoryImage): Promise<Buffer>
}

export interface GridServiceDeps {
  client: (url: string) => GridClient
  getConnection: () => Promise<MergeConnection>
  appVersion: string
  /** Folder holding one sub folder per grid. */
  gridsDir: string
  /** Open dialog for a template .json; null when cancelled. */
  pickFile: () => Promise<string | null>
  /** Save dialog; null when cancelled. */
  pickSavePath: (suggestedName: string) => Promise<string | null>
  trashItem: (path: string) => Promise<void>
  openPath: (path: string) => Promise<void>
  now?: () => number
  newId?: () => string
  /** Template functions (default: the real ones). */
  template?: {
    inspectTemplate: typeof inspectTemplate
    checkTemplateForRecipe: typeof checkTemplateForRecipe
    buildGridPrompt: typeof buildGridPrompt
  }
  /** Wait between looks at ComfyUI's history while a cell runs. Default 3000 ms. */
  pollMs?: number
  /** Wait between looks for the history entry after success (up to 8 looks). Default 250 ms. */
  historyRetryMs?: number
  objectInfoTtlMs?: number
  /** Minimum milliseconds between progress events. Default 200. */
  progressIntervalMs?: number
}

export const GRID_TEMPLATE_MAX_BYTES = 5 * 1024 * 1024
export const GRID_SAVE_MAX_BYTES = 64 * 1024 * 1024
const IMAGE_NAME = /^c\d+-r\d+\.(png|jpe?g|webp)$/
const ID_PATTERN = /^[A-Za-z0-9_-]+$/
const MAX_SEED = Number.MAX_SAFE_INTEGER

const MIME: Record<string, string> = { png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp' }

class ConnectionLost extends Error {}

interface Ctx {
  job: GridJob
  client: GridClient
  dir: string
  template: Record<string, unknown>
  info: GridTemplateInfo
  serverVersion: string | null
  blockInputs: string[] | null
  ws: { close(): void } | null
  promptId: string | null
  cancelled: boolean
  /** Outcomes of prompts seen on the websocket, by prompt id. */
  outcomes: Map<string, { ok: boolean; message: string }>
  wake: (() => void) | null
  lastProgressAt: number
  writing: Promise<void>
  finishing: boolean
  finishPromise: Promise<void> | null
  /** A cancel has begun; the run pauses before its next cell. */
  cancelRequested: boolean
  cancelling: Promise<void> | null
  /** The queuePrompt call in flight (never rejects), or the last one. */
  submitting: Promise<void> | null
}

function recOf(v: unknown): Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {}
}

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))

function round4(n: number): number {
  return Math.round(n * 10000) / 10000
}

/** Cells that never finished become cancelled (a grid stopped by the user or by the app closing). */
function cancelUnfinished(job: GridJob): void {
  for (const c of job.cells) if (c.status === 'pending' || c.status === 'running') c.status = 'cancelled'
}

export class GridService extends EventEmitter {
  private jobs: Ctx[] = []
  private starting = false
  private infoCache = new Map<string, { at: number; data: Record<string, unknown> }>()
  private seq = 0

  constructor(private readonly d: GridServiceDeps) {
    super()
  }

  private now(): number {
    return (this.d.now ?? Date.now)()
  }
  private newId(): string {
    return (this.d.newId ?? (() => `grid-${this.now()}-${++this.seq}`))()
  }
  private tpl(): NonNullable<GridServiceDeps['template']> {
    return this.d.template ?? { inspectTemplate, checkTemplateForRecipe, buildGridPrompt }
  }

  private emitJob(ctx: Ctx): void {
    this.emit('job', structuredClone(ctx.job))
  }

  private async objectInfo(url: string, client: GridClient): Promise<Record<string, unknown>> {
    const ttl = this.d.objectInfoTtlMs ?? 60_000
    const hit = this.infoCache.get(url)
    if (hit && this.now() - hit.at < ttl) return hit.data
    const data = await client.objectInfo()
    this.infoCache.set(url, { at: this.now(), data })
    return data
  }

  // ---------------------------------------------------------------- template

  async loadTemplate(path?: string): Promise<GridTemplateInfo | null> {
    const p = path ?? (await this.d.pickFile())
    if (!p) return null
    const { json, error } = await this.readTemplateFile(p)
    if (error !== null) {
      return {
        path: p,
        name: basename(p),
        error,
        warnings: [],
        modelLoader: null,
        promptInput: null,
        seedInputs: [],
        imageOutputs: [],
        defaultPrompt: null
      }
    }
    return this.tpl().inspectTemplate(json, p)
  }

  private async readTemplateFile(p: string): Promise<{ json: unknown; text: string; error: string | null }> {
    try {
      const st = await stat(p)
      if (!st.isFile()) return { json: null, text: '', error: 'That is not a file.' }
      if (st.size > GRID_TEMPLATE_MAX_BYTES) return { json: null, text: '', error: 'The template file is larger than 5 MB.' }
      const text = (await readFile(p, 'utf8')).replace(/^﻿/, '')
      try {
        return { json: JSON.parse(text), text, error: null }
      } catch {
        return { json: null, text: '', error: 'The file is not valid JSON.' }
      }
    } catch (e) {
      return { json: null, text: '', error: `The template could not be read: ${(e as Error).message}` }
    }
  }

  // ---------------------------------------------------------------- start

  private validateRequest(req: GridRequest): string[] {
    const problems: string[] = []
    const { axis, recipe } = req
    const n = Array.isArray(axis?.values) ? axis.values.length : 0
    if (n < GRID_MIN_VALUES || n > GRID_MAX_VALUES) problems.push(`Use ${GRID_MIN_VALUES} to ${GRID_MAX_VALUES} values.`)
    else if (!axis.values.every((v) => typeof v === 'number' && Number.isFinite(v))) problems.push('Every value must be a number.')
    else {
      const [lo, hi, what] =
        axis.kind === 'ratio'
          ? [0, 1, 'The ratio']
          : axis.kind === 'multiplier'
            ? [0, 2, 'The multiplier']
            : [-10, 10, 'A LoRA strength']
      if (axis.kind !== 'ratio' && axis.kind !== 'multiplier' && axis.kind !== 'loraStrength') problems.push('Unknown setting to vary.')
      else if (axis.values.some((v) => v < lo || v > hi)) problems.push(`${what} must be between ${lo} and ${hi}.`)
    }
    if (axis?.kind === 'ratio' && recipe.mode !== 'weighted' && recipe.mode !== 'blocks') {
      problems.push('The ratio can only vary in weighted or blocks mode.')
    }
    if (axis?.kind === 'multiplier' && recipe.mode !== 'addDifference') problems.push('The multiplier can only vary in add difference mode.')
    if (axis?.kind === 'loraStrength') {
      const i = axis.loraIndex
      if (i === null || !Number.isInteger(i) || i < 0 || i >= (recipe.loras ?? []).length) problems.push('Choose which LoRA to vary.')
    }
    if (!Array.isArray(req.prompts) || req.prompts.length < 1 || req.prompts.length > GRID_MAX_PROMPTS) {
      problems.push(`Use 1 to ${GRID_MAX_PROMPTS} prompts.`)
    } else {
      if (req.prompts.some((p) => typeof p !== 'string' || p.trim() === '')) problems.push('Every prompt needs text.')
      if (req.prompts.some((p) => typeof p === 'string' && p.length > GRID_PROMPT_MAX)) problems.push(`A prompt can have at most ${GRID_PROMPT_MAX} characters.`)
    }
    if (!Number.isInteger(req.seed) || req.seed < 0 || req.seed > MAX_SEED) problems.push('The seed must be a whole number from 0 to 9007199254740991.')
    return problems
  }

  async start(req: GridRequest): Promise<GridJob> {
    if (this.starting || this.jobs.some((c) => c.job.status === 'running')) {
      throw new Error('A grid is already running. Wait for it to finish or cancel it first.')
    }
    this.starting = true
    try {
      return await this.startInner(req)
    } finally {
      this.starting = false
    }
  }

  private async startInner(requested: GridRequest): Promise<GridJob> {
    const req: GridRequest = structuredClone(requested)
    const early = this.validateRequest(req)
    if (early.length) throw new Error(early.join(' '))
    req.axis.values = req.axis.values.map(round4)
    if (typeof req.templatePath !== 'string' || req.templatePath === '') throw new Error('Choose a template workflow first.')

    const { json, text: templateText, error: readError } = await this.readTemplateFile(req.templatePath)
    if (readError !== null) throw new Error(readError)
    const t = this.tpl()
    const info = t.inspectTemplate(json, req.templatePath)
    if (info.error) throw new Error(info.error)
    const fit = t.checkTemplateForRecipe(info, req.recipe)
    if (fit) throw new Error(fit)

    const conn = await this.d.getConnection()
    if (!conn.url || !conn.reachable) throw new Error('No ComfyUI server is reachable. Start the own server or check the connection.')
    if (!conn.own && !conn.sharedQueueAccepted) {
      throw new Error('This server was not started by the app. Accept its shared queue first (Server page) before queueing a grid.')
    }
    const url = conn.url
    const client = this.d.client(url)
    const objectInfo = await this.objectInfo(url, client)
    const caps: ServerCapabilities = parseCapabilities(objectInfo)
    const blockInputs =
      req.recipe.mode === 'blocks' && req.recipe.blockNode ? (blockNodeOptions(objectInfo, null).find((o) => o.node === req.recipe.blockNode)?.inputs ?? null) : null
    // Block inputs without their own value take the ratio; when every one has a value the ratio axis changes nothing.
    if (req.axis.kind === 'ratio' && req.recipe.mode === 'blocks' && blockInputs && blockInputs.length > 0 && blockInputs.every((k) => k in (req.recipe.blocks ?? {}))) {
      throw new Error('Every block has its own value on the Merge page, so changing the ratio changes nothing. Clear some block values or sweep a LoRA strength.')
    }
    // A grid saves no model file, so the output name does not matter.
    const checked: MergeRecipe = { ...req.recipe, outputName: req.recipe.outputName?.trim() ? req.recipe.outputName : 'grid' }
    // Grids build no save node, so a missing ModelSave / CheckpointSave node is no problem.
    const problems = validateRecipe(checked, caps, blockInputs).filter((p) => !/^The server has no (ModelSave|CheckpointSave) node\.?$/.test(p))
    if (problems.length) throw new Error(problems.join(' '))

    const id = this.newId()
    if (!ID_PATTERN.test(id)) throw new Error('The grid id is not valid.')
    const dir = join(this.d.gridsDir, id)
    await mkdir(dir, { recursive: true })
    // The same text that was parsed and checked above.
    await writeFile(join(dir, 'template.json'), templateText, 'utf8')

    const cells: GridCell[] = []
    for (let row = 0; row < req.prompts.length; row++) {
      for (let col = 0; col < req.axis.values.length; col++) {
        cells.push({ col, row, status: 'pending', file: null, error: null, seconds: null })
      }
    }
    const job: GridJob = {
      id,
      request: req,
      templateName: info.name || basename(req.templatePath),
      status: 'running',
      serverUrl: url,
      own: conn.own,
      createdAt: this.now(),
      finishedAt: null,
      cells,
      progress: null,
      error: null
    }
    const ctx: Ctx = {
      job,
      client,
      dir,
      template: recOf(json),
      info,
      serverVersion: conn.version,
      blockInputs,
      ws: null,
      promptId: null,
      cancelled: false,
      outcomes: new Map(),
      wake: null,
      lastProgressAt: 0,
      writing: Promise.resolve(),
      finishing: false,
      finishPromise: null,
      cancelRequested: false,
      cancelling: null,
      submitting: null
    }
    this.jobs.push(ctx)
    await this.persist(ctx)
    this.emitJob(ctx)
    const snapshot = structuredClone(job)
    void this.run(ctx)
    return snapshot
  }

  // ---------------------------------------------------------------- running

  private persist(ctx: Ctx, job: GridJob = ctx.job): Promise<void> {
    const record: GridRecord = { app: 'RulZ Model Merger', appVersion: this.d.appVersion, serverVersion: ctx.serverVersion, job: structuredClone(job) }
    const text = `${JSON.stringify(record, null, 2)}\n`
    const file = join(ctx.dir, 'grid.json')
    const tmp = `${file}.tmp`
    const next = ctx.writing.then(async () => {
      await writeFile(tmp, text, 'utf8')
      await rename(tmp, file)
    })
    ctx.writing = next.catch(() => undefined)
    return next
  }

  private async safePersist(ctx: Ctx, job: GridJob = ctx.job): Promise<void> {
    try {
      await this.persist(ctx, job)
    } catch {
      /* the folder may be gone; the run goes on */
    }
  }

  private wakeUp(ctx: Ctx): void {
    const w = ctx.wake
    ctx.wake = null
    w?.()
  }

  private openSocket(ctx: Ctx): void {
    const clientId = `rulz-model-merger-${ctx.job.id}`
    ctx.ws = ctx.client.openEvents(clientId, {
      onExecuting: (p) => {
        if (!ctx.promptId || p.promptId !== ctx.promptId || !p.node) return
        const cell = this.runningCell(ctx)
        if (cell && cell.status === 'pending') cell.status = 'running'
        this.setProgress(ctx, { node: this.classOf(ctx, p.node), value: 0, max: 0 }, true)
      },
      onProgress: (p) => {
        if (!ctx.promptId || p.promptId !== ctx.promptId) return
        const node = (p.node ? this.classOf(ctx, p.node) : null) ?? ctx.job.progress?.node ?? null
        this.setProgress(ctx, { node, value: p.value, max: p.max }, false)
      },
      onSuccess: (id) => {
        ctx.outcomes.set(id, { ok: true, message: '' })
        this.wakeUp(ctx)
      },
      onError: (p) => {
        ctx.outcomes.set(p.promptId, { ok: false, message: p.message })
        this.wakeUp(ctx)
      },
      onClose: () => this.wakeUp(ctx)
    })
  }

  private classOf(ctx: Ctx, nodeId: string): string | null {
    const c = recOf(this.lastPrompt.get(ctx))[nodeId]
    const t = recOf(c).class_type
    return typeof t === 'string' ? t : null
  }
  /** The prompt of the running cell, for naming the node in progress events. */
  private lastPrompt = new WeakMap<Ctx, Record<string, unknown>>()

  private runningCell(ctx: Ctx): GridCell | undefined {
    return ctx.job.cells.find((c) => c.status === 'running') ?? ctx.job.cells.find((c) => c.status === 'pending' && ctx.promptId !== null)
  }

  private setProgress(ctx: Ctx, progress: GridJob['progress'], force: boolean): void {
    if (ctx.cancelled || ctx.job.status !== 'running') return
    ctx.job.progress = progress
    const t = this.now()
    if (!force && t - ctx.lastProgressAt < (this.d.progressIntervalMs ?? 200)) return
    ctx.lastProgressAt = t
    this.emitJob(ctx)
  }

  /** Cells in run order: column by column, all prompts of a value first. */
  private order(job: GridJob): GridCell[] {
    return [...job.cells].sort((a, b) => a.col - b.col || a.row - b.row)
  }

  private async run(ctx: Ctx): Promise<void> {
    const job = ctx.job
    try {
      this.openSocket(ctx)
      for (const cell of this.order(job)) {
        // A cancel in progress holds the next cell back; if it fails the run goes on, if it works the run ends.
        while (ctx.cancelRequested && !ctx.cancelled) await sleep(5)
        if (ctx.cancelled) return
        await this.runCell(ctx, cell)
        if (ctx.cancelled) return
        await this.safePersist(ctx)
        this.emitJob(ctx)
      }
      await this.finish(ctx, 'done', null)
    } catch (e) {
      if (ctx.cancelled) return
      await this.finish(ctx, 'error', (e as Error).message)
    } finally {
      ctx.ws?.close()
      ctx.ws = null
    }
  }

  /** Writes the final state to grid.json first, then shows it, so a reader of the folder never sees an older state. */
  private async finish(ctx: Ctx, status: 'done' | 'error' | 'cancelled', error: string | null): Promise<void> {
    if (ctx.finishPromise) return ctx.finishPromise
    if (ctx.job.status !== 'running') return
    ctx.finishing = true
    ctx.finishPromise = this.finishInner(ctx, status, error)
    return ctx.finishPromise
  }

  private async finishInner(ctx: Ctx, status: 'done' | 'error' | 'cancelled', error: string | null): Promise<void> {
    const job = ctx.job
    const next = structuredClone(job)
    if (status !== 'done') cancelUnfinished(next)
    next.status = status === 'done' && next.cells.every((c) => c.status === 'error') ? 'error' : status
    next.error = next.status === 'error' && error === null ? 'Every cell failed.' : error
    next.finishedAt = this.now()
    next.progress = null
    await this.safePersist(ctx, next)
    Object.assign(job, next)
    this.emitJob(ctx)
  }

  private async runCell(ctx: Ctx, cell: GridCell): Promise<void> {
    const job = ctx.job
    const started = this.now()
    ctx.promptId = null
    job.progress = null
    cell.status = 'running'
    this.emitJob(ctx)
    const fail = (message: string): void => {
      cell.status = 'error'
      cell.error = message
    }
    const axis = job.request.axis
    let prompt: Record<string, unknown>
    try {
      let recipe = recipeForValue(job.request.recipe, axis, axis.values[cell.col])
      if (recipe.mode === 'blocks' && ctx.blockInputs) recipe = completeBlocks(recipe, ctx.blockInputs)
      prompt = this.tpl().buildGridPrompt(ctx.template, ctx.info, recipe, job.request.prompts[cell.row], job.request.seed)
    } catch (e) {
      fail((e as Error).message)
      return
    }
    this.lastPrompt.set(ctx, prompt)
    if (ctx.cancelled) return
    let promptId: string
    try {
      const clientId = `rulz-model-merger-${job.id}`
      // The id is stored the moment the answer arrives, so a cancel that began meanwhile can still stop this prompt.
      const submit = ctx.client.queuePrompt(prompt, clientId).then((r) => {
        ctx.promptId = r.promptId
        return r.promptId
      })
      ctx.submitting = submit.then(
        () => undefined,
        () => undefined
      )
      promptId = await submit
    } catch (e) {
      const status = (e as { status?: unknown }).status
      // A rejection by ComfyUI (HTTP status) is this cell's problem; no answer at all means the server is gone.
      if (typeof status === 'number') {
        fail((e as Error).message)
        return
      }
      throw new ConnectionLost(`Lost the connection to ComfyUI: ${(e as Error).message}`)
    }
    ctx.promptId = promptId
    if (ctx.cancelled) return
    const outcome = await this.waitForPrompt(ctx, promptId)
    if (ctx.cancelled) return
    if (!outcome.ok) {
      fail(outcome.message)
      return
    }
    // ComfyUI sends execution_success before it writes the history entry, so the entry may appear a moment later.
    const outputs = new Set(ctx.info.imageOutputs)
    let found: { nodeId: string; image: HistoryImage } | undefined
    for (let attempt = 0; attempt < 8; attempt++) {
      if (attempt > 0) {
        await sleep(this.d.historyRetryMs ?? 250)
        if (ctx.cancelled) return
      }
      try {
        const history = await ctx.client.history(promptId)
        found = history?.images.find((i) => outputs.has(i.nodeId))
      } catch (e) {
        throw new ConnectionLost(`Lost the connection to ComfyUI: ${(e as Error).message}`)
      }
      if (found) break
    }
    if (!found) {
      fail('ComfyUI finished without returning an image.')
      return
    }
    let bytes: Buffer
    try {
      bytes = await ctx.client.viewImage(found.image)
    } catch (e) {
      const status = (e as { status?: unknown }).status
      if (typeof status === 'number') {
        fail(`The image could not be fetched: ${(e as Error).message}`)
        return
      }
      throw new ConnectionLost(`Lost the connection to ComfyUI: ${(e as Error).message}`)
    }
    const ext = /\.(png|jpe?g|webp)$/i.exec(found.image.filename)?.[1].toLowerCase() ?? 'png'
    const file = `c${cell.col}-r${cell.row}.${ext}`
    try {
      await writeFile(join(ctx.dir, file), bytes)
    } catch (e) {
      fail(`The image could not be saved: ${(e as Error).message}`)
      return
    }
    cell.status = 'done'
    cell.file = file
    cell.error = null
    cell.seconds = Math.max(0, Math.round(((this.now() - started) / 1000) * 10) / 10)
    job.progress = null
  }

  /** Waits for this prompt's end: websocket events first, with history looked at every pollMs as a safety net. */
  private async waitForPrompt(ctx: Ctx, promptId: string): Promise<{ ok: boolean; message: string }> {
    const pollMs = this.d.pollMs ?? 3000
    let failures = 0
    let missing = 0
    for (;;) {
      const seen = ctx.outcomes.get(promptId)
      if (seen) return seen
      if (ctx.cancelled) return { ok: false, message: 'Cancelled' }
      let timedOut = false
      await Promise.race([
        new Promise<void>((r) => {
          ctx.wake = r
        }),
        sleep(pollMs).then(() => {
          timedOut = true
        })
      ])
      ctx.wake = null
      if (ctx.outcomes.has(promptId) || ctx.cancelled || !timedOut) continue
      let h: HistoryEntry | null
      try {
        h = await ctx.client.history(promptId)
        failures = 0
      } catch (e) {
        if (++failures >= 3) throw new ConnectionLost(`Lost the connection to ComfyUI: ${(e as Error).message}`)
        continue
      }
      if (!h) {
        try {
          const q = await ctx.client.queue()
          const known = q.running.includes(promptId) || q.pending.includes(promptId)
          missing = known ? 0 : missing + 1
        } catch {
          continue
        }
        if (missing >= 3) return { ok: false, message: 'ComfyUI no longer knows this prompt; it may have been cleared or the server restarted.' }
        continue
      }
      missing = 0
      if (h.error) return { ok: false, message: h.error }
      if (h.completed) return { ok: true, message: '' }
    }
  }

  // ---------------------------------------------------------------- cancel

  async cancel(id: string): Promise<GridJob> {
    const ctx = this.jobs.find((c) => c.job.id === id)
    if (!ctx) throw new Error('That grid is not running in this session.')
    const job = ctx.job
    if (job.status !== 'running') return structuredClone(job)
    if (ctx.finishPromise) {
      await ctx.finishPromise
      return structuredClone(job)
    }
    if (!ctx.cancelling) {
      ctx.cancelling = this.cancelInner(ctx).finally(() => {
        ctx.cancelling = null
      })
    }
    await ctx.cancelling
    return structuredClone(job)
  }

  /**
   * The run keeps going (it only pauses before queueing the next cell) until the server has been told: `cancelRequested`
   * holds the loop, `cancelled` ends it. A cancel that did not reach the server throws and the run carries on.
   */
  private async cancelInner(ctx: Ctx): Promise<void> {
    ctx.cancelRequested = true
    try {
      // A prompt that is still being submitted has no id yet: wait for the answer, then cancel that id.
      if (ctx.submitting) await ctx.submitting
      const promptId = ctx.promptId
      if (promptId && !ctx.outcomes.has(promptId)) {
        // Only this grid's prompt id, never the whole queue, never /free. Both calls: the prompt may have started between
        // a look at the queue and the delete; /interrupt with an id only stops that prompt when it is the running one.
        // On a server the app did not start, interrupt is never called (it is for the app's own server only): a prompt that
        // is already running there is left to finish and its result is ignored.
        const errors: Error[] = []
        try {
          await ctx.client.deletePending([promptId])
        } catch (e) {
          errors.push(e as Error)
        }
        if (ctx.job.own) {
          try {
            await ctx.client.interrupt(promptId)
          } catch (e) {
            errors.push(e as Error)
          }
        }
        if (errors.length === (ctx.job.own ? 2 : 1)) {
          // Nothing worked: fine when the prompt is no longer waiting (gone, or on a shared server already running),
          // otherwise the caller hears about it.
          let settled = false
          try {
            const q = await ctx.client.queue()
            settled = !q.pending.includes(promptId) && (ctx.job.own ? !q.running.includes(promptId) : true)
          } catch {
            settled = false
          }
          if (!settled) throw errors[0]
        }
      }
    } catch (e) {
      ctx.cancelRequested = false
      throw e
    }
    ctx.cancelled = true
    this.wakeUp(ctx)
    await this.finish(ctx, 'cancelled', null)
    ctx.ws?.close()
    ctx.ws = null
  }

  // ---------------------------------------------------------------- reading

  /** Grids of this session and earlier ones, newest first. A grid found on disk as running was cut off by closing the app. */
  async list(): Promise<GridJob[]> {
    const byId = new Map<string, GridJob>()
    for (const c of this.jobs) byId.set(c.job.id, structuredClone(c.job))
    let names: string[] = []
    try {
      names = await readdir(this.d.gridsDir)
    } catch {
      names = []
    }
    for (const name of names) {
      if (byId.has(name) || !ID_PATTERN.test(name)) continue
      const job = await this.readRecord(name)
      if (job) byId.set(job.id, job)
    }
    return [...byId.values()].sort((a, b) => b.createdAt - a.createdAt)
  }

  private async readRecord(folder: string): Promise<GridJob | null> {
    try {
      const raw = JSON.parse(await readFile(join(this.d.gridsDir, folder, 'grid.json'), 'utf8')) as unknown
      const job = recOf(recOf(raw).job) as unknown as GridJob
      if (typeof job.id !== 'string' || job.id !== folder || !Array.isArray(job.cells) || typeof job.status !== 'string') return null
      if (job.status === 'running') {
        job.status = 'cancelled'
        cancelUnfinished(job)
        job.progress = null
        if (job.finishedAt === null) job.finishedAt = job.createdAt
      }
      return job
    } catch {
      return null
    }
  }

  private async knownFolder(id: string): Promise<string> {
    if (typeof id !== 'string' || !ID_PATTERN.test(id)) throw new Error('That grid is not known.')
    const dir = join(this.d.gridsDir, id)
    if (this.jobs.some((c) => c.job.id === id)) return dir
    try {
      await stat(join(dir, 'grid.json'))
    } catch {
      throw new Error('That grid is not known.')
    }
    return dir
  }

  async image(id: string, file: string): Promise<string> {
    if (typeof file !== 'string' || !IMAGE_NAME.test(file)) throw new Error('That is not a grid image name.')
    const dir = await this.knownFolder(id)
    const bytes = await readFile(join(dir, file))
    const ext = file.slice(file.lastIndexOf('.') + 1).toLowerCase()
    return `data:${MIME[ext] ?? 'image/png'};base64,${bytes.toString('base64')}`
  }

  async saveImage(dataUrl: string, suggestedName: string): Promise<string | null> {
    const prefix = 'data:image/png;base64,'
    if (typeof dataUrl !== 'string' || !dataUrl.startsWith(prefix)) throw new Error('Only PNG images can be saved.')
    const b64 = dataUrl.slice(prefix.length)
    if (!/^[A-Za-z0-9+/]*={0,2}$/.test(b64)) throw new Error('The image data is not valid.')
    if (Math.floor((b64.length * 3) / 4) > GRID_SAVE_MAX_BYTES) throw new Error('The image is larger than 64 MB.')
    const stem =
      String(suggestedName ?? '')
        .replace(/\.png$/i, '')
        .replace(/[^A-Za-z0-9 _.-]/g, '_')
        .replace(/^\.+/, '')
        .trim()
        .slice(0, 100) || 'grid'
    const path = await this.d.pickSavePath(`${stem}.png`)
    if (!path) return null
    await writeFile(path, Buffer.from(b64, 'base64'))
    return path
  }

  async openFolder(id: string): Promise<void> {
    const dir = await this.knownFolder(id)
    await this.d.openPath(dir)
  }

  async delete(id: string): Promise<void> {
    const dir = await this.knownFolder(id)
    if (this.jobs.some((c) => c.job.id === id && c.job.status === 'running')) throw new Error('This grid is still running. Cancel it first.')
    await this.d.trashItem(dir)
    this.jobs = this.jobs.filter((c) => c.job.id !== id)
  }
}
