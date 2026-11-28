// Runs merge jobs on the connected ComfyUI server: inspect (read headers, check), start (queue + track + verify),
// cancel (only this app's prompt), and move the saved file into the models folder.
//
// Only one merge of this app is queued or running at a time. Cancelling touches only this app's own prompt id: a pending
// prompt is deleted by id, a running one is interrupted by id (ComfyUI interrupts only when that id is the running one).
// The queue is never cleared and models are never freed.
import { EventEmitter } from 'node:events'
import { writeFile } from 'node:fs/promises'
import { posix } from 'node:path'
import type { OwnServerSettings, ServerCapabilities } from '@shared/types'
import type {
  MergeCheck,
  MergeInspectRequest,
  MergeJob,
  MergeOutput,
  MergeRecipe,
  LoraFileInfo,
  MergeVerify,
  ModelFileInfo,
  RecipeRecord
} from '@shared/merge'
import type { ComfyEventHandlers, HistoryEntry } from '../comfy/client'
import { parseCapabilities } from '../comfy/parse'
import type { WslRunner } from '../server/wsl'
import { readSafetensorsHeader, type SafetensorsHeader } from '../models/safetensors'
import { buildMergeGraph, completeBlocks, validateRecipe, type BuiltGraph } from './graph'
import { blockNodeOptions, checkMerge, describeLoraFile, describeModelFile, isTextEncoderLoraKey, normalizeKeys } from './inspect'
import {
  freeMemoryBytes,
  listOutputCandidates,
  moveNoOverwrite,
  resolveModelsDir,
  sourcePath,
  subfolderOf,
  toUncPath,
  type ModelKind,
  type OutputCandidate
} from './wslFiles'

/** The part of ComfyClient the runner uses. */
export interface MergeClient {
  objectInfo(): Promise<Record<string, unknown>>
  queuePrompt(prompt: Record<string, unknown>, clientId: string, extraData?: Record<string, unknown>): Promise<{ promptId: string; number: number }>
  history(promptId: string): Promise<HistoryEntry | null>
  queue(): Promise<{ running: string[]; pending: string[] }>
  deletePending(promptIds: string[]): Promise<void>
  interrupt(promptId?: string): Promise<void>
  openEvents(clientId: string, handlers: ComfyEventHandlers): { close(): void }
}

export interface MergeConnection {
  url: string | null
  reachable: boolean
  own: boolean
  sharedQueueAccepted: boolean
  version: string | null
}

export interface MergeRunnerDeps {
  client: (url: string) => MergeClient
  wsl: WslRunner
  getConnection: () => Promise<MergeConnection>
  /** The running own server's launch record (frozen distro and absolute base directory), or null. */
  getOwnRecord: () => { distro: string; baseDir: string } | null
  ownSettings: () => OwnServerSettings
  /** Subscribes to the own server's log text; returns the unsubscribe function. Used to find LoRA weights ComfyUI skipped. */
  onServerLog?: (listener: (text: string) => void) => () => void
  appVersion: string
  now?: () => number
  newId?: () => string
  /** Test hooks. */
  toUnc?: (distro: string, absWslPath: string) => string
  writeText?: (path: string, text: string) => Promise<void>
  pollMs?: number
  /** Wait between looks for the saved file after success. Default 500 ms, 6 looks. */
  findRetryMs?: number
  objectInfoTtlMs?: number
}

/**
 * Keys ComfyUI reports as skipped while applying a LoRA: lines "lora key not loaded: <key>" or "NOT LOADED <key>".
 * Windows line endings are accepted; keys are returned once each, in order of appearance.
 */
export function parseLoraNotLoaded(text: string): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const line of text.split(/\r?\n/)) {
    const m = /lora key not loaded:\s*(\S+)/i.exec(line) ?? /\bNOT LOADED\s+(\S+)/.exec(line)
    if (!m || seen.has(m[1])) continue
    seen.add(m[1])
    out.push(m[1])
  }
  return out
}

export const URL_MODE_MESSAGE = 'Files of a server the app did not start cannot be read; ComfyUI checks them when the merge runs.'

interface Ctx {
  job: MergeJob
  client: MergeClient
  ws: { close(): void } | null
  timer: ReturnType<typeof setInterval> | null
  graph: BuiltGraph
  record: { distro: string; baseDir: string } | null
  stem: string
  subfolder: string
  before: Set<string>
  aHeader: SafetensorsHeader | null
  recipeRecord: RecipeRecord
  finalizing: boolean
  /** Events seen before the prompt id was known (prompt id kept), replayed once it is. */
  early: Array<{ promptId: string; run: () => void }>
  /** Polls in a row where history had no entry and the queue did not list the prompt. */
  missing: number
  /** Own-server log collection for a job with LoRAs (null when not collecting). */
  logUnsub: (() => void) | null
  /** The unfinished last log line. */
  logPartial: string
  notLoaded: Set<string>
}

const TERMINAL = new Set(['done', 'error', 'cancelled'])

function recOf(v: unknown): Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {}
}

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))

export class MergeRunner extends EventEmitter {
  private list: Ctx[] = []
  private starting = false
  private moving = new Set<string>()
  private infoCache = new Map<string, { at: number; data: Record<string, unknown> }>()
  /** Bumped by clearObjectInfoCache: a request that was already under way then does not refill the cache. */
  private infoGen = 0
  private seq = 0

  constructor(private readonly d: MergeRunnerDeps) {
    super()
  }

  private now(): number {
    return (this.d.now ?? Date.now)()
  }
  private newId(): string {
    return (this.d.newId ?? (() => `merge-${this.now()}-${++this.seq}`))()
  }
  private unc(distro: string, p: string): string {
    return (this.d.toUnc ?? toUncPath)(distro, p)
  }

  /** Forget the cached node lists, for example after the app's own server was restarted (it may have other nodes now). */
  clearObjectInfoCache(): void {
    this.infoCache.clear()
    this.infoGen++
  }

  private async objectInfo(url: string, client: MergeClient): Promise<Record<string, unknown>> {
    const ttl = this.d.objectInfoTtlMs ?? 60_000
    const hit = this.infoCache.get(url)
    if (hit && this.now() - hit.at < ttl) return hit.data
    const gen = this.infoGen
    const data = await client.objectInfo()
    if (gen === this.infoGen) this.infoCache.set(url, { at: this.now(), data })
    return data
  }

  private emitJob(ctx: Ctx): void {
    this.emit('job', structuredClone(ctx.job))
  }

  /** Newest first, this session only. */
  jobs(): MergeJob[] {
    return [...this.list].reverse().map((c) => structuredClone(c.job))
  }

  private find(id: string): Ctx {
    const c = this.list.find((x) => x.job.id === id)
    if (!c) throw new Error('That merge is not known (merges are only kept until the app closes).')
    return c
  }

  private async readSource(modelsAbs: string, distro: string, kind: ModelKind, name: string): Promise<{ info: ModelFileInfo; header: SafetensorsHeader | null }> {
    try {
      const header = await readSafetensorsHeader(this.unc(distro, sourcePath(modelsAbs, kind, name)))
      return { info: describeModelFile(name, header), header }
    } catch (e) {
      return { info: describeModelFile(name, null, (e as Error).message), header: null }
    }
  }

  private async readLora(modelsAbs: string, distro: string, name: string, aHeader: SafetensorsHeader | null): Promise<{ info: LoraFileInfo; header: SafetensorsHeader | null }> {
    try {
      const header = await readSafetensorsHeader(this.unc(distro, sourcePath(modelsAbs, 'lora', name)))
      return { info: describeLoraFile(name, header, aHeader, null), header }
    } catch (e) {
      return { info: describeLoraFile(name, null, null, (e as Error).message), header: null }
    }
  }

  async inspect(req: MergeInspectRequest): Promise<MergeCheck> {
    const conn = await this.d.getConnection()
    if (!conn.url || !conn.reachable) throw new Error('No ComfyUI server is connected.')
    const client = this.d.client(conn.url)
    const objectInfo = await this.objectInfo(conn.url, client)
    const record = conn.own ? this.d.getOwnRecord() : null
    const bake = req.mode === 'bake'
    const loraNames = req.loras ?? []
    const names: Array<[string | null, 'a' | 'b' | 'c']> = [
      [req.a || null, 'a'],
      [bake ? null : req.b || null, 'b'],
      [req.mode === 'addDifference' ? req.c || null : null, 'c']
    ]
    const sides: Record<'a' | 'b' | 'c', { info: ModelFileInfo; header: SafetensorsHeader | null } | null> = { a: null, b: null, c: null }
    let ramFreeBytes: number | null = null
    let loras: LoraFileInfo[]
    if (record) {
      const modelsAbs = await resolveModelsDir(this.d.wsl, record.distro, this.d.ownSettings())
      const [read, ram] = await Promise.all([
        Promise.all(names.map(async ([n, k]) => (n ? ([k, await this.readSource(modelsAbs, record.distro, req.kind, n)] as const) : null))),
        freeMemoryBytes(this.d.wsl, record.distro)
      ])
      for (const r of read) if (r) sides[r[0]] = r[1]
      ramFreeBytes = ram
      loras = (await Promise.all(loraNames.map((n) => this.readLora(modelsAbs, record.distro, n, sides.a?.header ?? null)))).map((r) => r.info)
    } else {
      for (const [n, k] of names) if (n) sides[k] = { info: describeModelFile(n, null, URL_MODE_MESSAGE), header: null }
      loras = loraNames.map((n) => describeLoraFile(n, null, null, URL_MODE_MESSAGE))
    }
    return checkMerge({ kind: req.kind, mode: req.mode, a: sides.a, b: sides.b, c: sides.c, objectInfo, ramFreeBytes, loras })
  }

  async start(requested: MergeRecipe): Promise<MergeJob> {
    if (this.starting || this.list.some((c) => !TERMINAL.has(c.job.status))) {
      throw new Error('A merge is already queued or running. Wait for it to finish or cancel it first.')
    }
    this.starting = true
    try {
      return await this.startInner(requested)
    } finally {
      this.starting = false
    }
  }

  private async startInner(requested: MergeRecipe): Promise<MergeJob> {
    let recipe = requested
    const conn = await this.d.getConnection()
    if (!conn.url || !conn.reachable) throw new Error('No ComfyUI server is reachable. Start the own server or check the connection.')
    if (!conn.own && !conn.sharedQueueAccepted) {
      throw new Error('This server was not started by the app. Accept its shared queue first (Server page) before queueing a merge.')
    }
    const url = conn.url
    const client = this.d.client(url)
    const objectInfo = await this.objectInfo(url, client)
    const caps: ServerCapabilities = parseCapabilities(objectInfo)
    const blockInputs =
      recipe.mode === 'blocks' && recipe.blockNode ? (blockNodeOptions(objectInfo, null).find((o) => o.node === recipe.blockNode)?.inputs ?? null) : null
    const problems = validateRecipe(recipe, caps, blockInputs)
    if (problems.length) throw new Error(problems.join(' '))
    // ComfyUI rejects a block node with a missing FLOAT input: fill the ones the recipe left out with its ratio.
    if (recipe.mode === 'blocks' && blockInputs) recipe = completeBlocks(recipe, blockInputs)
    const jobId = this.newId()
    const graph = buildMergeGraph(recipe, { jobId })

    const slash = graph.filenamePrefix.lastIndexOf('/')
    const stem = slash >= 0 ? graph.filenamePrefix.slice(slash + 1) : graph.filenamePrefix
    const subfolder = slash >= 0 ? graph.filenamePrefix.slice(0, slash) : graph.outputSubfolder

    const record = conn.own ? this.d.getOwnRecord() : null
    const loraList = recipe.loras ?? []
    const hashes: RecipeRecord['sourceHeaderSha256'] = { a: null, b: null, c: null, loras: loraList.map(() => null) }
    let loraInfos: LoraFileInfo[] = loraList.map((l) => describeLoraFile(l.name, null, null, URL_MODE_MESSAGE))
    let aHeader: SafetensorsHeader | null = null
    const sides: Record<'a' | 'b' | 'c', { info: ModelFileInfo; header: SafetensorsHeader | null } | null> = { a: null, b: null, c: null }
    const before = new Set<string>()
    if (record) {
      const modelsAbs = await resolveModelsDir(this.d.wsl, record.distro, this.d.ownSettings())
      const names: Array<[string | null, 'a' | 'b' | 'c']> = [
        [recipe.a, 'a'],
        [recipe.mode === 'bake' ? null : recipe.b, 'b'],
        [recipe.mode === 'addDifference' ? recipe.c : null, 'c']
      ]
      await Promise.all(
        names.map(async ([n, k]) => {
          if (!n) return
          const r = await this.readSource(modelsAbs, record.distro, recipe.kind, n)
          sides[k] = r
          if (r.header) hashes[k] = r.header.headerSha256
          if (k === 'a') aHeader = r.header
        })
      )
      const loraReads = await Promise.all(loraList.map((l) => this.readLora(modelsAbs, record.distro, l.name, aHeader)))
      loraInfos = loraReads.map((r) => r.info)
      loraReads.forEach((r, i) => {
        if (r.header) hashes.loras[i] = r.header.headerSha256
      })
      const check = checkMerge({ kind: recipe.kind, mode: recipe.mode, a: sides.a, b: sides.b, c: sides.c, objectInfo, ramFreeBytes: null, loras: loraInfos })
      if (check.problems.length) throw new Error(check.problems.join(' '))
      for (const c of await listOutputCandidates(this.d.wsl, record.distro, `${record.baseDir}/output`, subfolder, stem)) before.add(c.name)
    }

    const recipeRecord: RecipeRecord = {
      jobId,
      app: 'RulZ Model Merger',
      appVersion: this.d.appVersion,
      createdAt: new Date(this.now()).toISOString(),
      serverVersion: conn.version,
      recipe: structuredClone(recipe),
      sourceHeaderSha256: hashes
    }
    const job: MergeJob = {
      id: jobId,
      recipe: structuredClone(recipe),
      status: 'queued',
      serverUrl: url,
      own: conn.own,
      promptId: null,
      createdAt: this.now(),
      finishedAt: null,
      progress: null,
      output: null,
      error: null
    }
    const ctx: Ctx = { job, client, ws: null, timer: null, graph, record, stem, subfolder, before, aHeader, recipeRecord, finalizing: false, early: [], missing: 0, logUnsub: null, logPartial: '', notLoaded: new Set() }

    const clientId = `rulz-model-merger-${job.id}`
    // The socket opens before the prompt is queued, so no event is missed.
    ctx.ws = client.openEvents(clientId, this.handlers(ctx))
    // ComfyUI logs the LoRA weights it could not map; only the app's own server's log is visible here.
    if (record && loraList.length && this.d.onServerLog) ctx.logUnsub = this.d.onServerLog((text) => this.feedLog(ctx, text))
    let queued: { promptId: string; number: number }
    try {
      queued = await client.queuePrompt(graph.prompt, clientId, { extra_pnginfo: { modelmerger_recipe: recipeRecord } })
    } catch (e) {
      this.stop(ctx)
      throw e
    }
    job.promptId = queued.promptId
    this.list.push(ctx)
    this.emitJob(ctx)
    const early = ctx.early.splice(0)
    for (const e of early) if (e.promptId === queued.promptId) e.run()
    ctx.timer = setInterval(() => void this.poll(ctx), this.d.pollMs ?? 5000)
    return structuredClone(job)
  }

  private handlers(ctx: Ctx): ComfyEventHandlers {
    // Events that arrive before queuePrompt has answered are kept (with their prompt id) and replayed by start().
    const route = (id: string | null, run: () => void): void => {
      if (!id) return
      if (ctx.job.promptId === null) {
        if (ctx.early.length < 500) ctx.early.push({ promptId: id, run })
        return
      }
      if (id === ctx.job.promptId) run()
    }
    const markRunning = (): void => {
      if (ctx.job.status === 'queued') ctx.job.status = 'running'
    }
    const classOf = (nodeId: string | null): string | null => {
      if (!nodeId) return null
      const n = ctx.graph.prompt[nodeId]
      const c = recOf(n).class_type
      return typeof c === 'string' ? c : null
    }
    return {
      onExecuting: (p) =>
        route(p.promptId, () => {
          if (TERMINAL.has(ctx.job.status) || ctx.job.status === 'verifying' || !p.node) return
          markRunning()
          ctx.job.progress = { node: classOf(p.node), value: 0, max: 0 }
          this.emitJob(ctx)
        }),
      onProgress: (p) =>
        route(p.promptId, () => {
          if (TERMINAL.has(ctx.job.status) || ctx.job.status === 'verifying') return
          markRunning()
          const node = (p.node ? classOf(p.node) : null) ?? ctx.job.progress?.node ?? null
          ctx.job.progress = { node, value: p.value, max: p.max }
          this.emitJob(ctx)
        }),
      onSuccess: (id) => route(id, () => void this.finalize(ctx)),
      onError: (p) => route(p.promptId, () => this.fail(ctx, p.message))
    }
  }

  private async poll(ctx: Ctx): Promise<void> {
    if (TERMINAL.has(ctx.job.status) || ctx.finalizing || !ctx.job.promptId) return
    let h: HistoryEntry | null
    try {
      h = await ctx.client.history(ctx.job.promptId)
    } catch {
      return
    }
    if (TERMINAL.has(ctx.job.status) || ctx.finalizing) return
    if (!h) {
      let known = true
      try {
        const q = await ctx.client.queue()
        known = q.running.includes(ctx.job.promptId) || q.pending.includes(ctx.job.promptId)
      } catch {
        return
      }
      if (TERMINAL.has(ctx.job.status) || ctx.finalizing) return
      ctx.missing = known ? 0 : ctx.missing + 1
      if (ctx.missing >= 3) this.fail(ctx, 'ComfyUI no longer knows this merge; it may have been cleared or the server restarted.')
      return
    }
    ctx.missing = 0
    if (h.error) this.fail(ctx, h.error)
    else if (h.completed) await this.finalize(ctx)
  }

  private stop(ctx: Ctx): void {
    if (ctx.timer) clearInterval(ctx.timer)
    ctx.timer = null
    ctx.ws?.close()
    ctx.ws = null
    ctx.logUnsub?.()
    ctx.logUnsub = null
  }

  /** Server log text arrives in chunks that may split lines: complete lines are parsed, the rest waits for more. */
  private feedLog(ctx: Ctx, text: string): void {
    const all = ctx.logPartial + text
    const cut = all.lastIndexOf('\n')
    if (cut < 0) {
      ctx.logPartial = all.length > 20_000 ? all.slice(-20_000) : all
      return
    }
    ctx.logPartial = all.slice(cut + 1)
    for (const k of parseLoraNotLoaded(all.slice(0, cut))) ctx.notLoaded.add(k)
  }

  private fail(ctx: Ctx, message: string): void {
    if (TERMINAL.has(ctx.job.status)) return
    ctx.job.status = 'error'
    ctx.job.error = message
    ctx.job.finishedAt = this.now()
    this.stop(ctx)
    this.emitJob(ctx)
  }

  private async finalize(ctx: Ctx): Promise<void> {
    if (ctx.finalizing || TERMINAL.has(ctx.job.status)) return
    ctx.finalizing = true
    const job = ctx.job
    try {
      if (!ctx.record) {
        job.output = {
          relPath: `${ctx.subfolder}/${ctx.stem}_*.safetensors`,
          wslPath: null,
          sizeBytes: null,
          sidecarPath: null,
          verify: null,
          movedTo: null
        }
        job.status = 'done'
        job.finishedAt = this.now()
        job.progress = null
        this.stop(ctx)
        this.emitJob(ctx)
        return
      }
      job.status = 'verifying'
      this.emitJob(ctx)
      const found = await this.findNewOutput(ctx, ctx.record)
      if (!found) {
        this.fail(ctx, 'ComfyUI finished, but the saved file could not be found or is not this merge’s file.')
        return
      }
      const { file, header } = found
      const output: MergeOutput = {
        relPath: `${ctx.subfolder}/${file.name}`,
        wslPath: file.path,
        sizeBytes: file.sizeBytes,
        sidecarPath: null,
        verify: null,
        movedTo: null
      }
      output.verify = this.verify(ctx, header)
      const side = `${file.path}.recipe.json`
      try {
        // Exclusive create: an existing file (or a symlink in its place) is never truncated or followed.
        const write = this.d.writeText ?? ((p: string, t: string) => writeFile(p, t, { encoding: 'utf8', flag: 'wx' }))
        await write(this.unc(ctx.record.distro, side), `${JSON.stringify(ctx.recipeRecord, null, 2)}\n`)
        output.sidecarPath = side
      } catch (e) {
        output.sidecarPath = null
        const exists = (e as NodeJS.ErrnoException).code === 'EEXIST'
        output.verify.message += exists
          ? ' A recipe file with that name already exists; it was left as it is.'
          : ' The recipe file could not be written.'
      }
      if (TERMINAL.has(job.status)) return
      job.output = output
      job.status = 'done'
      job.finishedAt = this.now()
      job.progress = null
      this.stop(ctx)
      this.emitJob(ctx)
    } catch (e) {
      this.fail(ctx, (e as Error).message)
    }
  }

  /** Newest first; the first new file whose header carries this job's recipe record. */
  private async findNewOutput(
    ctx: Ctx,
    record: { distro: string; baseDir: string }
  ): Promise<{ file: OutputCandidate; header: SafetensorsHeader } | null> {
    for (let i = 0; i < 6; i++) {
      const all = await listOutputCandidates(this.d.wsl, record.distro, `${record.baseDir}/output`, ctx.subfolder, ctx.stem)
      const fresh = all.filter((c) => !ctx.before.has(c.name)).sort((x, y) => y.mtime - x.mtime || (y.name > x.name ? 1 : -1))
      for (const file of fresh) {
        try {
          const header = await readSafetensorsHeader(this.unc(record.distro, file.path))
          const parsed = JSON.parse(header.metadata.modelmerger_recipe ?? 'null') as { jobId?: unknown } | null
          if (parsed && parsed.jobId === ctx.job.id) return { file, header }
        } catch {
          /* still being written, or not ours */
        }
      }
      if (i < 5) await sleep(this.d.findRetryMs ?? 500)
    }
    return null
  }

  private verify(ctx: Ctx, header: SafetensorsHeader): MergeVerify {
    const v = this.verifyKeys(ctx, header)
    const loras = ctx.job.recipe.loras ?? []
    // Only the app's own server's log is seen; without LoRAs or without it nothing is known about skipped weights.
    if (!loras.length || !ctx.record) return v
    if (!ctx.logUnsub) return v
    for (const k of parseLoraNotLoaded(ctx.logPartial)) ctx.notLoaded.add(k)
    const diffusion = ctx.job.recipe.kind === 'diffusion'
    const lost: string[] = []
    let te = 0
    for (const k of ctx.notLoaded) {
      if (diffusion && isTextEncoderLoraKey(k)) te++
      else lost.push(k)
    }
    v.loraKeysNotLoaded = lost.length
    v.loraKeysNotLoadedSample = lost.slice(0, 5)
    v.loraTextEncoderKeysSkipped = te
    if (lost.length > 0) {
      v.ok = false
      v.message += ` ${lost.length} LoRA weights were not applied (for example ${lost.slice(0, 3).join(', ')}); the LoRA may be for another model.`
    }
    if (te > 0) v.message += ` ${te} text-encoder weights left out (a diffusion-model file has no text encoder).`
    if (lost.length === 0 && te === 0) v.message += ' LoRAs applied: all weights.'
    return v
  }

  private verifyKeys(ctx: Ctx, header: SafetensorsHeader): MergeVerify {
    const recipeInHeader = Object.prototype.hasOwnProperty.call(header.metadata, 'modelmerger_recipe')
    const tensorCount = Object.keys(header.tensors).length
    if (!ctx.aHeader) {
      return {
        ok: true,
        tensorCount,
        missingKeys: 0,
        extraKeys: 0,
        recipeInHeader,
        loraKeysNotLoaded: null,
        loraKeysNotLoadedSample: [],
        loraTextEncoderKeysSkipped: 0,
        message: `Saved ${tensorCount} tensors. Model A's header could not be read, so the keys were not compared with it.`
      }
    }
    const outKeys = normalizeKeys(header).keys
    const aKeys = normalizeKeys(ctx.aHeader).keys
    // Buffers (such as rope.bands) are not saved by ComfyUI and are rebuilt on load; only weights, biases and
    // quantization data count as missing.
    const isReal = (k: string): boolean => /\.(weight|bias)$/.test(k) || k.includes('scale') || k.includes('comfy_quant')
    let missing = 0
    const buffers: string[] = []
    for (const k of aKeys.keys()) {
      if (outKeys.has(k)) continue
      if (isReal(k)) missing++
      else buffers.push(k)
    }
    let extra = 0
    for (const k of outKeys.keys()) if (!aKeys.has(k)) extra++
    const ok = missing === 0 && extra === 0
    let message = ok
      ? `File checked: ${outKeys.size} layers, ${recipeInHeader ? 'recipe stored' : 'recipe not stored in the file'}.`
      : `Compared with model A: ${missing} keys missing, ${extra} keys extra.`
    if (buffers.length) {
      const names = buffers.slice(0, 3).join(', ') + (buffers.length > 3 ? ', ...' : '')
      message += ` ${buffers.length} buffer${buffers.length === 1 ? '' : 's'} not saved (${names}); ComfyUI rebuilds ${buffers.length === 1 ? 'it' : 'them'} when loading.`
    }
    return { ok, tensorCount, missingKeys: missing, extraKeys: extra, recipeInHeader, loraKeysNotLoaded: null, loraKeysNotLoadedSample: [], loraTextEncoderKeysSkipped: 0, message }
  }

  async cancel(id: string): Promise<MergeJob> {
    const ctx = this.find(id)
    const job = ctx.job
    if (TERMINAL.has(job.status)) return structuredClone(job)
    const promptId = job.promptId
    if (promptId) {
      // Never the whole queue, never /free: only this app's prompt id. Do both, because the prompt may have started
      // between a look at the queue and the delete; /interrupt with a prompt id only stops that prompt when it runs.
      const errors: Error[] = []
      let deleteError: Error | null = null
      try {
        await ctx.client.deletePending([promptId])
      } catch (e) {
        deleteError = e as Error
        errors.push(deleteError)
      }
      try {
        await ctx.client.interrupt(promptId)
      } catch (e) {
        errors.push(e as Error)
      }
      if (errors.length === 2) throw errors[0]
      if (deleteError) {
        // Interrupt with an id does nothing for a prompt that is still waiting: it would stay queued and run later
        // untracked. Only go on when the queue shows it is gone.
        let waiting: boolean
        try {
          waiting = (await ctx.client.queue()).pending.includes(promptId)
        } catch (e) {
          throw new Error(`The merge could not be cancelled: ${deleteError.message} (and the queue could not be read: ${(e as Error).message}).`)
        }
        if (waiting) throw new Error(`The merge could not be cancelled: ${deleteError.message} It is still waiting in the server queue.`)
      }
    }
    if (!TERMINAL.has(job.status)) {
      job.status = 'cancelled'
      job.finishedAt = this.now()
      job.progress = null
      this.stop(ctx)
      this.emitJob(ctx)
    }
    return structuredClone(job)
  }

  async moveToModels(id: string): Promise<MergeJob> {
    const ctx = this.find(id)
    const job = ctx.job
    const out = job.output
    if (!ctx.record || !job.own) throw new Error('Only files saved by the app’s own server can be moved into its models folder.')
    if (job.status !== 'done' || !out || !out.wslPath) throw new Error('The merge has not finished with a saved file yet.')
    if (out.movedTo) throw new Error(`This file was already moved to ${out.movedTo}.`)
    if (this.moving.has(id)) throw new Error('This file is being moved already.')
    this.moving.add(id)
    try {
      const modelsAbs = await resolveModelsDir(this.d.wsl, ctx.record.distro, this.d.ownSettings())
      const dir = posix.join(modelsAbs, subfolderOf(job.recipe.kind), 'merged')
      const r = await moveNoOverwrite(this.d.wsl, ctx.record.distro, out.wslPath, dir)
      out.movedTo = r.path
      if (r.sidecar) out.sidecarPath = r.sidecar
      this.emitJob(ctx)
      return structuredClone(job)
    } finally {
      this.moving.delete(id)
    }
  }
}
