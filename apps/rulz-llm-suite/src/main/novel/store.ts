// The Writing mode's project store: every file operation of the novel feature, with the data safety rules of
// docs/writing-mode-design.md. No Electron imports: the app passes in what it needs (paths, trashItem, emit).
import { AsyncLocalStorage } from 'node:async_hooks'
import { promises as fs } from 'node:fs'
import { basename, isAbsolute, join, resolve } from 'node:path'
import { parseEntry, formatEntry } from '@shared/frontMatter'
import {
  NOVEL_FORMAT,
  type BibleEntry,
  type BibleKind,
  type NovelChange,
  type NovelChapter,
  type NovelHistoryItem,
  type NovelOutline,
  type NovelProject,
  type NovelProjectState,
  type NovelProjectSummary,
  type NovelSaveRequest,
  type NovelSaveResult,
  type NovelScene,
  type NovelStatePatchResult,
  type NovelTextFile,
  type NovelTrashItem,
  type OutlineOp,
  type ProjectPatch,
  type RecordLine,
  type RecordsExportOptions,
  type RecordsExportResult,
  type RecordsInfo
} from '@shared/novel'
import { scenePositions } from '@shared/novelState'
import { defaultProject } from '@shared/novelPrompts'
import { applyOutlineOps, emptyOutline, validSceneFile, findChapter, findScene, newId, removeChapter, removeScene, reinsert, sceneFileName, slugify } from '@shared/novelOutline'
import { wordCount } from '@shared/novelText'
import type { SamplingParams } from '@shared/types'
import { USER_PATH, assertAppPath, assertNoLinks, cleanRelPath, decodeBytes, drainFolder, encodeText, isInside, NEW_FILE_META, readBytes, realInside, serialized, sha256, writeBytesAtomic } from './files'
import { listHistory, takeSnapshot } from './history'
import { deleteJournal, discardJournal, readJournal, writeJournal } from './journal'
import { listTrash, moveToTrash, readManifest, removeTrashFolder, restoreFiles, returnToTrash, trashRoot, undoTrash, type TrashManifest, type TrashOutline } from './trash'
import { ProjectWatcher } from './watch'
import { LineSink, MAX_RECORD_BYTES, appendRecordText, bump, exportRequest, isCardAction, localDate, localStamp, parseRecordBytes, readRawLines, RecordJoin, retryLock, validateRecordLine } from './records'
import { applyStateOps, emptyChecks, emptyDoc, parseChecksBytes, parseStateBytes, serializeChecks, serializeDoc, stateFileOf, type ChecksDoc, type StateDoc } from './state'

export interface StoreOptions {
  /** The writingRoot setting ('' = default). */
  rootSetting: () => string
  /** The user's Documents folder. */
  documents: () => string
  /** shell.trashItem. */
  trashItem: (path: string) => Promise<void>
  /**
   * Sends a novel:changed event to every listener (the window and the phones). `origin` is the owner whose request
   * caused the change ('window', 'phone:<id>'), or undefined when it came from outside the app or from the app itself:
   * the window leaves out its own origin, since its own saves come back in the reply.
   */
  emit: (change: NovelChange, origin?: string) => void
  /** Turn the folder watcher off (tests). */
  watch?: boolean
  /** The app's version, written to the training-data manifest (app.getVersion()). */
  appVersion?: () => string
  /** The clock (tests): names the day's records file and the export folder. */
  now?: () => Date
  /**
   * The start sampling of a NEW project: the model's own Writing sampling, or null for the built-in fiction preset.
   * Only createProject asks; projects already on disk keep what they saved.
   */
  newProjectSampling?: () => SamplingParams | null
}

/** The owner of the window's open projects; each phone connection is 'phone:<clientId>'. */
export const WINDOW_OWNER = 'window'

const originStore = new AsyncLocalStorage<string | undefined>()
/** Runs `fn` as the request of `owner`: changes it makes are announced with that origin. */
export function runAs<T>(owner: string | undefined, fn: () => T): T {
  return originStore.run(owner, fn)
}

/** Refusal while a project folder is being moved to the Recycle Bin. */
const DELETING_MESSAGE = 'This project is being deleted.'

interface OpenProject {
  folder: string
  /** Who has the project open (see WINDOW_OWNER); the project is dropped when the last owner closes it. */
  owners: Set<string>
  project: NovelProject
  outline: NovelOutline
  blocked: string | null
  /** The project folder is being moved to the Recycle Bin: no new file writes are accepted. */
  deleting?: boolean
  /** Why project.json / outline.json cannot be used, or null. */
  jsonErr: { 'project.json': string | null; 'outline.json': string | null }
  /** Hash of each file as last read or written by the app (lower-case relative path → sha256, '' = absent). */
  hashes: Map<string, string>
  watcher: ProjectWatcher | null
  lastSnap: Map<string, number>
  sessionSnapped: Set<string>
  chain: Promise<unknown>
  entryCache: Map<string, { key: string; entry: BibleEntry }>
  wordCache: Map<string, { key: string; words: number }>
  /** state.json as last read or written (stage 3). */
  stateDoc: StateDoc
  stateProblem: string | null
  stateCanReset: boolean
  /** state.json was present at some read in this session (so missing later is a problem, not an empty state). */
  stateExisted: boolean
  /** .rulz/checks.json as last read or written. */
  checks: ChecksDoc
  checksProblem: string | null
  /** Record appends that are running (stage 4); close and delete wait for them. */
  records: Set<Promise<unknown>>
  /** Set while the records folder is being moved to the Recycle Bin: new appends wait for it. */
  recordHold: Promise<void> | null
  /** The project is closing or being deleted: no more appends. */
  recordsOff: boolean
}

const RESERVED = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i
const SNAPSHOT_GAP_MS = 10 * 60_000
/** Outline and project files change in bursts (typing in a field); one snapshot per burst is enough. */
const JSON_SNAPSHOT_GAP_MS = 10_000

const lc = (s: string): string => s.toLowerCase()
const projectKey = (folder: string): string => (process.platform === 'win32' ? lc(resolve(folder)) : resolve(folder))
const errText = (err: unknown): string => (err instanceof Error ? err.message : String(err))
const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)

/** Folder name for a project title: Windows-safe, not empty, not a device name. */
export function folderNameFor(title: string): string {
  let name = title
    .replace(/[<>:"/\\|?*\u0000-\u001f]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 60)
    .trim()
    .replace(/[. ]+$/, '')
  if (!name || RESERVED.test(name.split('.')[0])) name = name ? `${name} novel` : 'Untitled novel'
  return name
}

function parseJsonBytes(bytes: Buffer, what: string): unknown {
  try {
    return JSON.parse(bytes.toString('utf8').replace(/^﻿/, ''))
  } catch (err) {
    throw new Error(`${what} could not be read: ${errText(err)}`)
  }
}

/** Fills missing fields with defaults and keeps unknown keys; throws Error with a message when the file is unusable. */
function normalizeProject(raw: unknown, folderName: string): NovelProject {
  if (!isObject(raw)) throw new Error('project.json could not be read: it is not a JSON object.')
  const def = defaultProject(folderName)
  const sampling = isObject(raw.sampling) ? { ...def.sampling, ...raw.sampling } : def.sampling
  const budget = isObject(raw.budget) ? { ...def.budget, ...raw.budget } : def.budget
  return {
    ...(raw as object),
    format: typeof raw.format === 'number' ? raw.format : NOVEL_FORMAT,
    id: typeof raw.id === 'string' && raw.id ? raw.id : def.id,
    title: typeof raw.title === 'string' && raw.title ? raw.title : folderName,
    createdAt: typeof raw.createdAt === 'string' ? raw.createdAt : def.createdAt,
    updatedAt: typeof raw.updatedAt === 'string' ? raw.updatedAt : def.updatedAt,
    modelId: typeof raw.modelId === 'string' ? raw.modelId : null,
    sampling,
    adapters: Array.isArray(raw.adapters) ? (raw.adapters as NovelProject['adapters']) : null,
    budget,
    povTense: typeof raw.povTense === 'string' ? raw.povTense : ''
  } as NovelProject
}

function normalizeOutline(raw: unknown): NovelOutline {
  if (!isObject(raw) || !Array.isArray(raw.chapters)) throw new Error('outline.json could not be read: it has no list of chapters.')
  const chapters = raw.chapters.map((c, ci): NovelChapter => {
    if (!isObject(c) || typeof c.id !== 'string' || !c.id) throw new Error(`outline.json could not be read: chapter ${ci + 1} has no id.`)
    if (!Array.isArray(c.scenes)) throw new Error(`outline.json could not be read: chapter ${ci + 1} has no list of scenes.`)
    const scenes = c.scenes.map((s, si): NovelScene => {
      if (!isObject(s) || typeof s.id !== 'string' || !s.id) throw new Error(`outline.json could not be read: scene ${si + 1} of chapter ${ci + 1} has no id.`)
      if (!validSceneFile(s.file)) throw new Error(`outline.json could not be read: scene "${String(s.title ?? s.id)}" has no valid file name.`)
      const status = ['idea', 'beats', 'draft', 'revised'].includes(s.status as string) ? s.status : 'idea'
      return {
        ...(s as object),
        id: s.id,
        title: typeof s.title === 'string' ? s.title : 'Untitled scene',
        file: s.file,
        beats: typeof s.beats === 'string' ? s.beats : '',
        summary: typeof s.summary === 'string' ? s.summary : '',
        summaryOf: typeof s.summaryOf === 'string' ? s.summaryOf : '',
        status,
        pov: typeof s.pov === 'string' ? s.pov : null,
        present: Array.isArray(s.present) ? s.present.filter((x): x is string => typeof x === 'string') : [],
        sticky: Array.isArray(s.sticky) ? s.sticky.filter((x): x is string => typeof x === 'string') : [],
        targetWords: typeof s.targetWords === 'number' && s.targetWords > 0 ? s.targetWords : null
      } as NovelScene
    })
    return {
      ...(c as object),
      id: c.id,
      title: typeof c.title === 'string' ? c.title : 'Untitled chapter',
      act: typeof c.act === 'string' ? c.act : '',
      summary: typeof c.summary === 'string' ? c.summary : '',
      summaryOf: typeof c.summaryOf === 'string' ? c.summaryOf : '',
      scenes
    } as NovelChapter
  })
  return { ...(raw as object), format: typeof raw.format === 'number' ? raw.format : NOVEL_FORMAT, chapters } as NovelOutline
}

/** A trash manifest's outline fragment is untrusted: run it through the outline reader's rules. */
function cleanTrashOutline(raw: unknown): TrashOutline {
  try {
    if (!isObject(raw) || typeof raw.index !== 'number' || !Number.isFinite(raw.index)) throw new Error('bad position')
    if (raw.kind === 'scene') {
      if (typeof raw.chapterId !== 'string') throw new Error('bad chapter')
      const scene = normalizeOutline({ chapters: [{ id: 'x', scenes: [raw.scene] }] }).chapters[0].scenes[0]
      return { kind: 'scene', chapterId: raw.chapterId, index: raw.index, scene }
    }
    if (raw.kind === 'chapter') {
      const chapter = normalizeOutline({ chapters: [raw.chapter] }).chapters[0]
      return { kind: 'chapter', index: raw.index, chapter }
    }
    throw new Error('bad kind')
  } catch (err) {
    throw new Error(`The trash item is damaged (${errText(err).replace('outline.json could not be read: ', '')}).`)
  }
}

const json = (value: unknown): Buffer => Buffer.from(`${JSON.stringify(value, null, 2)}\n`, 'utf8')

export class NovelStore {
  private projects = new Map<string, OpenProject>()

  constructor(private opts: StoreOptions) {}

  // ---------------------------------------------------------------------------------------------
  // Root and project list
  // ---------------------------------------------------------------------------------------------

  /** The projects folder, created on first use. */
  async root(): Promise<string> {
    const setting = this.opts.rootSetting().trim()
    const dir = setting && isAbsolute(setting) ? setting : join(this.opts.documents(), 'RulZ Novels')
    await fs.mkdir(dir, { recursive: true })
    return dir
  }

  async listProjects(): Promise<NovelProjectSummary[]> {
    const root = await this.root()
    const out: NovelProjectSummary[] = []
    for (const e of await fs.readdir(root, { withFileTypes: true }).catch(() => [])) {
      if (!e.isDirectory() || e.name.startsWith('.')) continue
      const folder = join(root, e.name)
      const pj = await readBytes(folder, join(folder, 'project.json')).catch(() => null)
      if (!pj) continue
      const mtime = (await fs.stat(join(folder, 'project.json')).catch(() => null))?.mtime.toISOString() ?? ''
      try {
        const project = normalizeProject(parseJsonBytes(pj, 'project.json'), e.name)
        out.push({ id: project.id, title: project.title, folder, updatedAt: project.updatedAt, words: await this.countWords(folder) })
      } catch {
        // Broken project.json: still listed so the user can open it and see why it is blocked.
        out.push({ id: '', title: e.name, folder, updatedAt: mtime, words: 0 })
      }
    }
    return out.sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : a.updatedAt > b.updatedAt ? -1 : 0))
  }

  /** Words in a closed project's scenes; 0 when the outline cannot be read. */
  private async countWords(folder: string): Promise<number> {
    try {
      const bytes = await readBytes(folder, join(folder, 'outline.json'))
      if (!bytes) return 0
      const outline = normalizeOutline(parseJsonBytes(bytes, 'outline.json'))
      let n = 0
      for (const c of outline.chapters) {
        for (const s of c.scenes) {
          const b = await readBytes(folder, join(folder, 'manuscript', s.file)).catch(() => null)
          if (b) n += wordCount(decodeBytes(b).text)
        }
      }
      return n
    } catch {
      return 0
    }
  }

  async createProject(title: string, owner: string = WINDOW_OWNER): Promise<NovelProjectState> {
    const clean = typeof title === 'string' ? title.trim() : ''
    if (!clean) throw new Error('The project needs a title.')
    const root = await this.root()
    const base = folderNameFor(clean)
    let folder = ''
    for (let n = 1; ; n++) {
      const candidate = join(root, n === 1 ? base : `${base} ${n}`)
      try {
        await fs.mkdir(candidate)
        folder = candidate
        break
      } catch (err) {
        if ((err as NodeJS.ErrnoException).code !== 'EEXIST') throw err
        if (n > 500) throw new Error('Could not find a free folder name for the project.')
      }
    }
    const tmp = join(folder, '.rulz', 'tmp')
    for (const d of ['bible', 'manuscript', '.rulz']) await fs.mkdir(join(folder, d), { recursive: true })
    const project = defaultProject(clean)
    try {
      const own = this.opts.newProjectSampling?.()
      if (own) project.sampling = structuredClone(own)
    } catch (err) {
      // A start sampling that cannot be read must not stop a project from being created.
      console.warn('Could not read the start sampling for the new project:', err)
    }
    const now = new Date().toISOString()
    if (!project.id) project.id = newId('p')
    project.createdAt ||= now
    project.updatedAt ||= now
    const sceneId = newId('s')
    const file = sceneFileName(0, 'Scene 1', sceneId, [])
    const outline: NovelOutline = {
      format: NOVEL_FORMAT,
      chapters: [
        {
          id: newId('c'),
          title: 'Chapter 1',
          act: '',
          summary: '',
          summaryOf: '',
          scenes: [{ id: sceneId, title: 'Scene 1', file, beats: '', summary: '', summaryOf: '', status: 'idea', pov: null, present: [], sticky: [], targetWords: null }]
        }
      ]
    }
    await writeBytesAtomic(folder, join(folder, 'project.json'), tmp, json(project))
    await writeBytesAtomic(folder, join(folder, 'outline.json'), tmp, json(outline))
    await writeBytesAtomic(folder, join(folder, 'style.md'), tmp, Buffer.alloc(0))
    await writeBytesAtomic(folder, join(folder, 'manuscript', file), tmp, Buffer.alloc(0))
    return this.openProject(folder, owner)
  }

  // ---------------------------------------------------------------------------------------------
  // Open / close
  // ---------------------------------------------------------------------------------------------

  hasOpenProject(): boolean {
    return this.projects.size > 0
  }

  /** Tells every listener about a change, tagged with who caused it. A failing listener never fails the change. */
  private announce(change: NovelChange): void {
    try {
      this.opts.emit(change, originStore.getStore())
    } catch (err) {
      console.warn('Writing: change listener failed:', err)
    }
  }

  /** The new outline/project/bible state, announced and returned. */
  private async publish(p: OpenProject): Promise<NovelProjectState> {
    const state = await this.state(p)
    this.announce({ kind: 'state', folder: p.folder, state })
    return state
  }

  /** Text-file hash of a user file, for announcing a restored file. */
  private async hashOf(p: OpenProject, rel: string): Promise<string> {
    const bytes = await readBytes(p.folder, join(p.folder, ...rel.split('/'))).catch(() => null)
    return bytes ? sha256(bytes) : ''
  }

  private get(folder: string): OpenProject {
    const p = typeof folder === 'string' ? this.projects.get(projectKey(folder)) : undefined
    if (!p) throw new Error('That project is not open.')
    return p
  }

  /** The state of an open project (for phone writing runs). */
  async projectState(folder: string): Promise<NovelProjectState> {
    return this.state(this.get(folder))
  }

  async openProject(folder: string, owner: string = WINDOW_OWNER): Promise<NovelProjectState> {
    if (typeof folder !== 'string' || !isAbsolute(folder)) throw new Error('The project folder must be a full path.')
    const key = projectKey(folder)
    const open = this.projects.get(key)
    if (open) {
      open.owners.add(owner)
      // A file that changed outside while the watcher's delay was still running is found here: announce it.
      const changed = await this.exclusive(open, () => this.loadJson(open, false))
      if (changed) await originStore.run(undefined, () => this.publish(open))
      return this.state(open)
    }
    const st = await fs.stat(folder).catch(() => null)
    if (!st?.isDirectory()) throw new Error('That folder does not exist.')
    if (!(await readBytes(folder, join(folder, 'project.json')))) throw new Error('That folder is not a RulZ novel project (there is no project.json in it).')
    const p: OpenProject = {
      folder: resolve(folder),
      owners: new Set(),
      project: defaultProject(basename(folder)),
      outline: emptyOutline(),
      blocked: null,
      jsonErr: { 'project.json': null, 'outline.json': null },
      hashes: new Map(),
      watcher: null,
      lastSnap: new Map(),
      sessionSnapped: new Set(),
      chain: Promise.resolve(),
      entryCache: new Map(),
      wordCache: new Map(),
      stateDoc: emptyDoc(),
      stateProblem: null,
      stateCanReset: false,
      stateExisted: false,
      checks: emptyChecks(),
      checksProblem: null,
      records: new Set(),
      recordHold: null,
      recordsOff: false
    }
    await this.loadJson(p, false)
    // Another owner may have opened the same folder while this one was reading it.
    const raced = this.projects.get(key)
    if (raced) {
      raced.owners.add(owner)
      return this.state(raced)
    }
    p.owners.add(owner)
    this.projects.set(key, p)
    if (this.opts.watch !== false) {
      // A watcher event is not anybody's request, whatever request happened to start the watcher.
      p.watcher = new ProjectWatcher(p.folder, (rel) => runAs(undefined, () => void this.onExternalChange(p.folder, rel).catch((err) => console.warn('Writing: watcher event failed:', err))))
      p.watcher.start()
    }
    return this.state(p)
  }

  /** Removes `owner` from the project; the watcher stops and the project is dropped only when nobody else has it open. */
  async closeProject(folder: string, owner: string = WINDOW_OWNER): Promise<void> {
    const key = projectKey(folder)
    const p = this.projects.get(key)
    if (!p) return
    p.owners.delete(owner)
    if (p.owners.size > 0) return
    p.watcher?.stop()
    this.projects.delete(key)
    p.recordsOff = true
    // Appends admitted before this point finish (in order, behind each other); none is admitted after it.
    // A Delete records that is running finishes first.
    while (p.recordHold) await p.recordHold
    await Promise.allSettled([...p.records])
    await p.chain.catch(() => undefined)
  }

  /** Closes every project `owner` holds (a phone connection that ended). */
  async releaseOwner(owner: string): Promise<void> {
    for (const p of [...this.projects.values()]) {
      if (p.owners.has(owner)) await this.closeProject(p.folder, owner)
    }
  }

  /** Who has a project open (tests and diagnostics). */
  ownersOf(folder: string): string[] {
    return [...(this.projects.get(projectKey(folder))?.owners ?? [])]
  }

  dispose(): void {
    for (const p of this.projects.values()) p.watcher?.stop()
    this.projects.clear()
  }

  private exclusive<T>(p: OpenProject, fn: () => Promise<T>): Promise<T> {
    // A project being moved to the Recycle Bin takes no new work: it would recreate folders at the old path.
    if (p.deleting) return Promise.reject(new Error(DELETING_MESSAGE))
    const next = p.chain.catch(() => undefined).then(fn)
    p.chain = next
    return next
  }

  // ---------------------------------------------------------------------------------------------
  // project.json / outline.json (main owns them; a broken file blocks the project, never resets it)
  // ---------------------------------------------------------------------------------------------

  /**
   * Reads project.json and outline.json when they differ from what the app last saw. Returns true when the
   * in-memory state changed. A file that does not parse sets `blocked` and keeps the last good values.
   */
  private async loadJson(p: OpenProject, announce: boolean): Promise<boolean> {
    const files = [
      { rel: 'project.json', what: 'project.json' },
      { rel: 'outline.json', what: 'outline.json' }
    ] as const
    let changed = false
    for (const f of files) {
      const bytes = await readBytes(p.folder, join(p.folder, f.rel))
      const hash = bytes ? sha256(bytes) : ''
      const known = p.hashes.get(f.rel)
      if (known !== undefined && known === hash) continue
      p.hashes.set(f.rel, hash)
      // Changed outside since the app last saw it: the next write must snapshot this version first (rule 5).
      if (known !== undefined) p.lastSnap.delete(f.rel)
      changed = true
      try {
        if (!bytes) throw new Error(`${f.what} is missing.`)
        const parsed = parseJsonBytes(bytes, f.what)
        if (f.rel === 'project.json') p.project = normalizeProject(parsed, basename(p.folder))
        else p.outline = normalizeOutline(parsed)
        p.jsonErr[f.rel] = null
      } catch (err) {
        p.jsonErr[f.rel] = errText(err)
      }
    }
    p.blocked = p.jsonErr['project.json'] ?? p.jsonErr['outline.json']
    if (await this.loadStateFile(p)) changed = true
    if (await this.loadChecksFile(p)) changed = true
    if (changed && announce) this.announce({ kind: 'state', folder: p.folder, state: await this.state(p) })
    return changed
  }

  /** Reads state.json when it differs from what the app last saw. A damaged file sets a problem; it is never replaced. */
  private async loadStateFile(p: OpenProject): Promise<boolean> {
    const target = join(p.folder, 'state.json')
    let bytes: Buffer | null
    try {
      // A link out of the project must not be read (assertAppPath only checks paths under .rulz).
      await this.guard(p, target)
      bytes = await readBytes(p.folder, target)
    } catch (err) {
      // Still locked after the retries, or a link out of the project: say so, and look again next time.
      const msg = `state.json could not be read: ${errText(err)}`
      const changed = p.stateProblem !== msg
      p.stateDoc = emptyDoc()
      p.stateProblem = msg
      p.stateCanReset = false
      p.hashes.delete('state.json')
      p.lastSnap.delete('state.json')
      // readBytes turns a missing file into null, not an error, so the file is there: a later disappearance is a problem.
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') p.stateExisted = true
      return changed
    }
    const hash = bytes ? sha256(bytes) : ''
    const known = p.hashes.get('state.json')
    if (known !== undefined && known === hash) return false
    p.hashes.set('state.json', hash)
    // A change made outside: the next write must snapshot this version first.
    if (known !== undefined) p.lastSnap.delete('state.json')
    const parsed = parseStateBytes(bytes, p.stateExisted)
    if (bytes) p.stateExisted = true
    p.stateDoc = parsed.doc
    p.stateProblem = parsed.problem
    p.stateCanReset = parsed.canReset
    return true
  }

  /** Reads .rulz/checks.json when it differs from what the app last saw. An unreadable file means dismissals are ignored and never written. */
  private async loadChecksFile(p: OpenProject): Promise<boolean> {
    const key = '.rulz/checks.json'
    const target = join(p.folder, '.rulz', 'checks.json')
    let bytes: Buffer | null
    try {
      await this.guard(p, target, join(p.folder, '.rulz'))
      bytes = await readBytes(p.folder, target)
    } catch (err) {
      const msg = `.rulz/checks.json could not be read: ${errText(err)}`
      const changed = p.checksProblem !== msg
      p.checks = emptyChecks()
      p.checksProblem = msg
      p.hashes.delete(key)
      return changed
    }
    const hash = bytes ? sha256(bytes) : ''
    const known = p.hashes.get(key)
    if (known !== undefined && known === hash) return false
    p.hashes.set(key, hash)
    const parsed = parseChecksBytes(bytes)
    p.checks = parsed.doc
    p.checksProblem = parsed.problem
    return true
  }

  /** Writes a JSON file after snapshotting what is on disk. The caller holds the project's exclusive lock. */
  private async writeJson(p: OpenProject, rel: 'project.json' | 'outline.json', value: unknown): Promise<void> {
    const target = join(p.folder, rel)
    const disk = await readBytes(p.folder, target)
    if (disk) await this.snapshot(p, rel, disk, JSON_SNAPSHOT_GAP_MS)
    const bytes = json(value)
    await serialized(target, () => writeBytesAtomic(p.folder, target, join(p.folder, '.rulz', 'tmp'), bytes))
    p.hashes.set(rel, sha256(bytes))
  }

  /**
   * Runs `change` on fresh outline/project data and writes the result. If a file changed on disk between the
   * refresh and the write, the whole step runs again on the new data (ops address things by id, so they apply).
   */
  private mutate<T>(p: OpenProject, fn: () => Promise<{ result: T; project?: boolean; outline?: boolean }>): Promise<T> {
    return this.exclusive(p, async () => {
      for (let attempt = 0; ; attempt++) {
        await this.loadJson(p, true)
        if (p.blocked) throw new Error(`${p.blocked} The project is read-only until that file is fixed.`)
        const { result, project, outline } = await fn()
        const stale = async (rel: string): Promise<boolean> => {
          const disk = await readBytes(p.folder, join(p.folder, rel))
          return (disk ? sha256(disk) : '') !== p.hashes.get(rel)
        }
        if ((project && (await stale('project.json'))) || (outline && (await stale('outline.json')))) {
          if (attempt >= 2) throw new Error('The project files keep changing on disk. Try again in a moment.')
          // fn mutated memory only; reload from disk and run it again.
          p.hashes.delete('project.json')
          p.hashes.delete('outline.json')
          continue
        }
        if (outline) await this.writeJson(p, 'outline.json', p.outline)
        if (project) await this.writeJson(p, 'project.json', p.project)
        return result
      }
    })
  }

  async projectPatch(folder: string, patch: ProjectPatch): Promise<NovelProject> {
    const p = this.get(folder)
    const saved = await this.mutate(p, async () => {
      const next: NovelProject = { ...p.project }
      if (patch.title !== undefined) {
        if (typeof patch.title !== 'string' || !patch.title.trim()) throw new Error('The project title cannot be empty.')
        next.title = patch.title.trim()
      }
      if (patch.modelId !== undefined) next.modelId = typeof patch.modelId === 'string' && patch.modelId ? patch.modelId : null
      if (patch.sampling !== undefined) next.sampling = { ...next.sampling, ...patch.sampling }
      if (patch.adapters !== undefined) next.adapters = Array.isArray(patch.adapters) ? patch.adapters : null
      if (patch.povTense !== undefined) next.povTense = String(patch.povTense)
      if (patch.recording !== undefined) next.recording = patch.recording !== false
      if (patch.budget !== undefined) {
        const b = { ...next.budget, ...patch.budget }
        const share = (v: unknown, d: number): number => (typeof v === 'number' && Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : d)
        next.budget = {
          ...next.budget,
          replyTokens: Number.isFinite(b.replyTokens) ? Math.max(16, Math.round(b.replyTokens)) : next.budget.replyTokens,
          proseShare: share(b.proseShare, next.budget.proseShare),
          bibleShare: share(b.bibleShare, next.budget.bibleShare)
        }
      }
      next.updatedAt = new Date().toISOString()
      p.project = next
      return { result: next, project: true }
    })
    await this.publish(p)
    return saved
  }

  async outlinePatch(folder: string, ops: OutlineOp[]): Promise<NovelProjectState> {
    const p = this.get(folder)
    if (!Array.isArray(ops)) throw new Error('The outline changes must be a list.')
    await this.mutate(p, async () => {
      const existing = await this.manuscriptFiles(p)
      const { outline, createdFiles } = applyOutlineOps(p.outline, ops, existing)
      // Empty scene files first (never over an existing file), so the outline never points at nothing.
      const tmp = join(p.folder, '.rulz', 'tmp')
      for (const file of createdFiles) {
        const rel = `manuscript/${file}`
        const target = join(p.folder, 'manuscript', file)
        if (await readBytes(p.folder, target)) continue
        await writeBytesAtomic(p.folder, target, tmp, Buffer.alloc(0))
        p.hashes.set(lc(rel), sha256(Buffer.alloc(0)))
      }
      p.outline = outline
      return { result: undefined, outline: true }
    })
    return this.publish(p)
  }

  private async manuscriptFiles(p: OpenProject): Promise<string[]> {
    if (!(await realInside(p.folder, join(p.folder, 'manuscript')))) return []
    const entries = await fs.readdir(join(p.folder, 'manuscript'), { withFileTypes: true }).catch(() => [])
    // Only files a scene can be: *.md, no dotfiles (a .txt or a sync client's temp file is not offered).
    return entries.filter((e) => e.isFile() && validSceneFile(e.name)).map((e) => e.name)
  }

  // ---------------------------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------------------------

  async state(p: OpenProject): Promise<NovelProjectState> {
    const bible = await this.readBible(p)
    const style = await this.loadText(p, 'style.md', false)
    const files = await this.manuscriptFiles(p)
    const fileSet = new Map(files.filter((f) => /\.md$/i.test(f)).map((f) => [lc(f), f]))
    const words: Record<string, number> = {}
    const missing: string[] = []
    const referenced = new Set<string>()
    for (const c of p.outline.chapters) {
      for (const s of c.scenes) {
        const rel = `manuscript/${s.file}`
        referenced.add(lc(s.file))
        const actual = fileSet.get(lc(s.file))
        if (!actual) missing.push(rel)
        else words[rel] = await this.wordsOf(p, `manuscript/${actual}`)
      }
    }
    const orphans: string[] = []
    for (const [key, name] of fileSet) {
      if (referenced.has(key)) continue
      orphans.push(`manuscript/${name}`)
      words[`manuscript/${name}`] = await this.wordsOf(p, `manuscript/${name}`)
    }
    return {
      folder: p.folder,
      project: p.project,
      outline: p.outline,
      bible,
      style,
      words,
      missing,
      orphans: orphans.sort(),
      blocked: p.blocked,
      state: stateFileOf(p.stateDoc, p.stateProblem, p.stateCanReset),
      dismissed: p.checks.dismissed,
      checksProblem: p.checksProblem
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Tracked state (state.json) and dismissed check findings (.rulz/checks.json)
  // ---------------------------------------------------------------------------------------------

  /**
   * Applies the valid ops to state.json and returns the rejected ones with reasons. Runs under the project chain
   * on fresh data; if the file changed on disk before the write, the ops are applied again to the new content.
   */
  async statePatch(folder: string, ops: unknown): Promise<NovelStatePatchResult> {
    const p = this.get(folder)
    if (!Array.isArray(ops)) throw new Error('The state changes must be a list.')
    if (ops.length > 1000) throw new Error('Too many state changes in one request.')
    const target = join(p.folder, 'state.json')
    const result = await this.exclusive(p, async () => {
      for (let attempt = 0; ; attempt++) {
        await this.loadJson(p, true)
        if (p.blocked) {
          const reason = `${p.blocked} The project is read-only until that file is fixed.`
          return { rejected: ops.map((_, index) => ({ index, reason })) }
        }
        const bible = await this.readBible(p)
        const ctx = { pos: scenePositions(p.outline), entryIds: new Set(bible.map((b) => b.id)) }
        const out = applyStateOps({ doc: p.stateDoc, problem: p.stateProblem, canReset: p.stateCanReset }, ops, ctx)
        if (out.applied === 0) return { rejected: out.rejected }
        const doc: StateDoc = { top: p.stateDoc.top, items: out.items }
        const bytes = json(serializeDoc(doc))
        const stale = await serialized(target, async (): Promise<boolean> => {
          if (p.blocked) throw new Error(p.blocked)
          await this.guard(p, target)
          const disk = await readBytes(p.folder, target)
          const diskHash = disk ? sha256(disk) : ''
          if (diskHash !== (p.hashes.get('state.json') ?? '')) return true
          if (disk && disk.length > 0) await this.snapshot(p, 'state.json', disk, SNAPSHOT_GAP_MS)
          if (p.blocked) throw new Error(p.blocked)
          const unchanged = async (): Promise<boolean> => {
            const now = await readBytes(p.folder, target)
            return (now ? sha256(now) : '') === diskHash
          }
          if (!(await writeBytesAtomic(p.folder, target, join(p.folder, '.rulz', 'tmp'), bytes, unchanged))) return true
          p.hashes.set('state.json', sha256(bytes))
          p.stateExisted = true
          // Parse what was written, so the indexes of unreadable items match the file again.
          const parsed = parseStateBytes(bytes, true)
          p.stateDoc = parsed.doc
          p.stateProblem = parsed.problem
          p.stateCanReset = parsed.canReset
          return false
        })
        if (stale) {
          if (attempt >= 2) throw new Error('state.json keeps changing on disk. Try again in a moment.')
          // The disk differs from what the app last saw: forget the hash so the next loop reads it and applies the ops again.
          p.hashes.delete('state.json')
          p.lastSnap.delete('state.json')
          continue
        }
        return { rejected: out.rejected }
      }
    })
    const state = await this.state(p)
    if (!result.rejected || result.rejected.length < ops.length) this.announce({ kind: 'state', folder: p.folder, state })
    return { state, rejected: result.rejected }
  }

  /** Adds or removes a dismissed finding key for a scene in .rulz/checks.json. */
  async dismissFinding(folder: string, sceneId: string, key: string, dismissed: boolean): Promise<NovelProjectState> {
    const p = this.get(folder)
    if (typeof sceneId !== 'string' || !sceneId) throw new Error('The scene is missing.')
    if (typeof key !== 'string' || !key || key.length > 64) throw new Error('That finding key is not valid.')
    if (typeof dismissed !== 'boolean') throw new Error('Dismissed must be true or false.')
    const target = join(p.folder, '.rulz', 'checks.json')
    await this.exclusive(p, async () => {
      for (let attempt = 0; ; attempt++) {
        await this.loadJson(p, true)
        if (p.blocked) throw new Error(`${p.blocked} The project is read-only until that file is fixed.`)
        if (p.checksProblem) throw new Error(`${p.checksProblem} Dismissals are not saved until that file is fixed.`)
        if (!findScene(p.outline, sceneId)) throw new Error('That scene no longer exists.')
        const list = Object.prototype.hasOwnProperty.call(p.checks.dismissed, sceneId) ? p.checks.dismissed[sceneId] : []
        const has = list.includes(key)
        if (dismissed === has) return
        const next: Record<string, string[]> = {}
        for (const [k, v] of Object.entries(p.checks.dismissed)) Object.defineProperty(next, k, { value: v.slice(), enumerable: true, writable: true, configurable: true })
        const updated = dismissed ? [...list, key] : list.filter((x) => x !== key)
        if (updated.length) Object.defineProperty(next, sceneId, { value: updated, enumerable: true, writable: true, configurable: true })
        else delete next[sceneId]
        const bytes = json(serializeChecks({ top: p.checks.top, dismissed: next }))
        const stale = await serialized(target, async (): Promise<boolean> => {
          if (p.blocked) throw new Error(p.blocked)
          const disk = await readBytes(p.folder, target)
          const diskHash = disk ? sha256(disk) : ''
          if (diskHash !== (p.hashes.get('.rulz/checks.json') ?? '')) return true
          const unchanged = async (): Promise<boolean> => {
            const now = await readBytes(p.folder, target)
            return (now ? sha256(now) : '') === diskHash
          }
          if (!(await writeBytesAtomic(p.folder, target, join(p.folder, '.rulz', 'tmp'), bytes, unchanged))) return true
          p.hashes.set('.rulz/checks.json', sha256(bytes))
          p.checks = parseChecksBytes(bytes).doc
          return false
        })
        if (!stale) return
        if (attempt >= 2) throw new Error('.rulz/checks.json keeps changing on disk. Try again in a moment.')
        p.hashes.delete('.rulz/checks.json')
      }
    })
    return this.publish(p)
  }

  private async wordsOf(p: OpenProject, rel: string): Promise<number> {
    const path = join(p.folder, ...rel.split('/'))
    const st = await fs.stat(path).catch(() => null)
    if (!st) return 0
    const key = `${st.mtimeMs}:${st.size}`
    const hit = p.wordCache.get(lc(rel))
    if (hit?.key === key) return hit.words
    const bytes = await readBytes(p.folder, path).catch(() => null)
    const words = bytes ? wordCount(decodeBytes(bytes).text) : 0
    p.wordCache.set(lc(rel), { key, words })
    return words
  }

  private async readBible(p: OpenProject): Promise<BibleEntry[]> {
    const dir = join(p.folder, 'bible')
    if (!(await realInside(p.folder, dir))) return []
    const entries = (await fs.readdir(dir, { withFileTypes: true }).catch(() => [])).filter((e) => e.isFile() && /\.md$/i.test(e.name))
    const out: BibleEntry[] = []
    const seen = new Set<string>()
    for (const e of entries) {
      const rel = `bible/${e.name}`
      const path = join(dir, e.name)
      const st = await fs.stat(path).catch(() => null)
      if (!st) continue
      seen.add(lc(rel))
      const key = `${st.mtimeMs}:${st.size}`
      const hit = p.entryCache.get(lc(rel))
      if (hit?.key === key) {
        out.push(hit.entry)
        continue
      }
      const bytes = await readBytes(p.folder, path).catch(() => null)
      if (!bytes) continue
      const d = decodeBytes(bytes)
      const entry = parseEntry(e.name.replace(/\.md$/i, ''), d.text)
      if (d.readOnly) entry.problems.push('This file is not valid UTF-8, so it is shown read-only.')
      // The baseline for external-change detection; the app has not read this file any other way yet.
      if (!p.hashes.has(lc(rel))) p.hashes.set(lc(rel), d.hash)
      p.entryCache.set(lc(rel), { key, entry })
      out.push(entry)
    }
    for (const k of [...p.entryCache.keys()]) if (!seen.has(k)) p.entryCache.delete(k)
    return out.sort((a, b) => a.name.localeCompare(b.name))
  }

  // ---------------------------------------------------------------------------------------------
  // User-text files
  // ---------------------------------------------------------------------------------------------

  /** Refuses a path that leaves the project through a symlink or junction (links that stay inside are fine). */
  private async guard(p: OpenProject, ...paths: string[]): Promise<void> {
    for (const path of paths) {
      if (!(await realInside(p.folder, path))) throw new Error('That path leads outside the project through a link, so it is not used.')
    }
  }

  /** Runs `fn` holding the per-file keys of `rels` (sorted, so lock order is always the same). */
  private withFileLocks<T>(p: OpenProject, rels: string[], fn: () => Promise<T>): Promise<T> {
    const keys = [...new Set(rels.map((r) => join(p.folder, ...r.split('/'))))].sort()
    const run = (i: number): Promise<T> => (i >= keys.length ? fn() : serialized(keys[i], () => run(i + 1)))
    return run(0)
  }

  private userPath(folder: string, rel: unknown): string {
    const clean = cleanRelPath(folder, rel)
    if (!USER_PATH.test(clean)) throw new Error(`"${clean}" is not a file the Writing page edits.`)
    return clean
  }

  /**
   * A read found a file different from the hash the app knew: it changed outside, and the watcher's event (delayed)
   * will find the hash already updated and say nothing. Do what the watcher would: snapshot before the next
   * overwrite, and tell every listener (not as anybody's request).
   */
  private heardLate(p: OpenProject, key: string, rel: string, hash: string, known: string): void {
    p.lastSnap.delete(key)
    originStore.run(undefined, () => {
      if (hash) this.announce({ kind: 'text', folder: p.folder, path: rel, hash })
      else if (known) this.announce({ kind: 'deleted', folder: p.folder, path: rel })
      void this.publish(p).catch(() => undefined)
    })
  }

  /** Reads a user-text file. `remember` false only records a baseline hash when there is none (state builds). */
  private async loadText(p: OpenProject, rel: string, remember: boolean): Promise<NovelTextFile> {
    const path = join(p.folder, ...rel.split('/'))
    await this.guard(p, path, join(p.folder, '.rulz'))
    const bytes = await readBytes(p.folder, path)
    const key = lc(rel)
    if (!bytes) {
      const known = p.hashes.get(key)
      if (remember) p.hashes.set(key, '')
      else if (!p.hashes.has(key)) p.hashes.set(key, '')
      if (remember && known) this.heardLate(p, key, rel, '', known)
      return { path: rel, text: '', hash: '', meta: { ...NEW_FILE_META }, readOnly: false, journal: await this.journalFor(p, rel, '') }
    }
    const d = decodeBytes(bytes)
    const known = p.hashes.get(key)
    if (remember) p.hashes.set(key, d.hash)
    else if (!p.hashes.has(key)) p.hashes.set(key, d.hash)
    if (remember && known !== undefined && known !== d.hash) this.heardLate(p, key, rel, d.hash, known)
    // First look at a file in this session: keep what was there before anything can change it.
    if (remember && !p.sessionSnapped.has(key) && (await this.snapshot(p, rel, bytes))) p.sessionSnapped.add(key)
    return { path: rel, text: d.text, hash: d.hash, meta: d.meta, readOnly: d.readOnly, journal: await this.journalFor(p, rel, d.text) }
  }

  /**
   * The journal if its text differs from the file's. The file's modification time is not used: sync clients
   * and editors touch it, so an older-looking journal can still hold the only copy of an edit. An identical
   * journal is deleted. The renderer compares its baseHash with the file's hash.
   */
  private async journalFor(p: OpenProject, rel: string, diskText: string): Promise<NovelTextFile['journal']> {
    const j = await readJournal(p.folder, rel)
    if (!j) return null
    if (j.text.replace(/\r\n/g, '\n') === diskText) {
      // A blocked project is never written to, so the identical journal just stays.
      if (!p.blocked && !p.deleting) await deleteJournal(p.folder, rel)
      return null
    }
    return j
  }

  async readFile(folder: string, rel: string): Promise<NovelTextFile> {
    const p = this.get(folder)
    const clean = this.userPath(p.folder, rel)
    // A read can snapshot or delete a journal, which would write into a folder being moved away.
    if (p.deleting) throw new Error(DELETING_MESSAGE)
    return serialized(join(p.folder, clean), () => this.loadText(p, clean, true))
  }

  /** Takes a snapshot; `minGapMs` skips it if one was taken for this path that recently. `strict` throws on failure. False when no snapshot exists (blocked project, failure). */
  private async snapshot(p: OpenProject, rel: string, bytes: Buffer, minGapMs = 0, strict = false): Promise<boolean> {
    // A blocked project is never written to, not even under .rulz\; nor is one being deleted.
    if (p.blocked || p.deleting) return false
    const key = lc(rel)
    if (minGapMs && Date.now() - (p.lastSnap.get(key) ?? 0) < minGapMs) return true
    try {
      await takeSnapshot(p.folder, rel, bytes)
      p.lastSnap.set(key, Date.now())
      return true
    } catch (err) {
      if (strict) throw err
      console.warn(`Writing: could not snapshot ${rel}:`, err)
      return false
    }
  }

  async saveFile(req: NovelSaveRequest): Promise<NovelSaveResult> {
    const p = this.get(req.folder)
    const rel = this.userPath(p.folder, req.path)
    if (typeof req.text !== 'string') throw new Error('The text to save is missing.')
    if (p.blocked) return { ok: false, reason: 'blocked' }
    if (p.deleting) return { ok: false, reason: 'error', error: 'This project is being deleted, so nothing was saved.' }
    const target = join(p.folder, ...rel.split('/'))
    // The desktop editor's crash-recovery journal is the desktop's: a save made by a phone leaves it alone (the
    // recovery offers a journal whose text differs from the disk).
    const origin = originStore.getStore()
    const keepJournal = origin !== undefined && origin !== WINDOW_OWNER
    let wrote = false
    const result = await serialized(target, async (): Promise<NovelSaveResult> => {
      try {
        // The project can have become blocked while this save waited in the queue.
        if (p.blocked) return { ok: false, reason: 'blocked' }
        await this.guard(p, target, join(p.folder, '.rulz'))
        const disk = await readBytes(p.folder, target)
        const d = disk ? decodeBytes(disk) : null
        if (d?.readOnly) return { ok: false, reason: 'readOnly' }
        const diskHash = d?.hash ?? ''
        if (diskHash !== (req.baseHash ?? '') && !req.force) {
          return { ok: false, reason: 'conflict', diskHash, diskText: d?.text ?? '' }
        }
        const key = lc(rel)
        if (!disk) {
          // A file that is gone is only recreated if it still belongs to the project: a late save must not
          // bring back a scene or entry that was just deleted.
          const inOutline = p.outline.chapters.some((c) => c.scenes.some((s) => `manuscript/${lc(s.file)}` === key))
          if (key.startsWith('manuscript/') && !inOutline) return { ok: false, reason: 'error', error: 'This scene is no longer in the outline, so it was not saved.' }
          if (key.startsWith('bible/') && req.force) return { ok: false, reason: 'error', error: 'This entry no longer exists, so it was not saved.' }
        }
        p.entryCache.delete(key)
        p.wordCache.delete(key)
        // An unedited document is never written (it would only change timestamps and wake sync clients). The
        // comparison is of text, so a file with mixed line endings is left alone too.
        if (d && disk && req.text.replace(/\r\n/g, '\n') === d.text) {
          const before = p.hashes.get(key)
          p.hashes.set(key, diskHash)
          if (!keepJournal) await deleteJournal(p.folder, rel)
          // Nothing was written, but if this is the first the app hears of the disk version, others should too.
          if (before !== diskHash) this.announce({ kind: 'text', folder: p.folder, path: rel, hash: diskHash })
          return { ok: true, hash: diskHash }
        }
        const bytes = encodeText(req.text, d?.meta ?? NEW_FILE_META)
        if (disk) {
          if (req.snapshot || req.force) await this.snapshot(p, rel, disk, 0, true)
          else await this.snapshot(p, rel, disk, SNAPSHOT_GAP_MS)
        }
        if (p.blocked) return { ok: false, reason: 'blocked' }
        const unchanged = async (): Promise<boolean> => {
          const now = await readBytes(p.folder, target)
          return (now ? sha256(now) : '') === diskHash
        }
        if (!(await writeBytesAtomic(p.folder, target, join(p.folder, '.rulz', 'tmp'), bytes, unchanged))) {
          const now = await readBytes(p.folder, target)
          const nd = now ? decodeBytes(now) : null
          return { ok: false, reason: 'conflict', diskHash: nd?.hash ?? '', diskText: nd?.text ?? '' }
        }
        const hash = sha256(bytes)
        p.hashes.set(key, hash)
        if (!keepJournal) await deleteJournal(p.folder, rel)
        // The watcher's event for this write is ignored (the hash is known now), so announce it here.
        this.announce({ kind: 'text', folder: p.folder, path: rel, hash })
        wrote = true
        return { ok: true, hash }
      } catch (err) {
        return { ok: false, reason: 'error', error: errText(err) }
      }
    })
    // The window gets its own saves back in the reply; everyone else also needs what follows from the text
    // (word counts, a bible entry's name, aliases and triggers), which the suppressed watcher event would have sent.
    if (wrote && keepJournal && this.projects.get(projectKey(p.folder)) === p) {
      await this.publish(p).catch((err) => console.warn('Writing: state after save failed:', err))
    }
    return result
  }

  async journal(folder: string, rel: string, text: string, baseHash: string): Promise<void> {
    const p = this.get(folder)
    const clean = this.userPath(p.folder, rel)
    if (p.blocked || p.deleting) return
    // Same key as saveFile/readFile: a late journal write cannot land after a save deleted the journal.
    await serialized(join(p.folder, ...clean.split('/')), async () => {
      if (p.blocked) return
      await this.guard(p, join(p.folder, '.rulz'))
      await writeJournal(p.folder, clean, String(text), String(baseHash ?? ''))
    })
  }

  async discardJournal(folder: string, rel: string): Promise<void> {
    const p = this.get(folder)
    const clean = this.userPath(p.folder, rel)
    if (p.blocked || p.deleting) return
    await serialized(join(p.folder, ...clean.split('/')), async () => {
      if (p.blocked || !(await readJournal(p.folder, clean))) return
      await discardJournal(p.folder, clean)
    })
  }

  // ---------------------------------------------------------------------------------------------
  // Bible
  // ---------------------------------------------------------------------------------------------

  async createEntry(folder: string, name: string, kind: BibleKind): Promise<BibleEntry> {
    const p = this.get(folder)
    const clean = typeof name === 'string' ? name.trim() : ''
    if (!clean) throw new Error('The entry needs a name.')
    const entryKind: BibleKind = (['character', 'place', 'lore', 'other'] as const).includes(kind) ? kind : 'other'
    const entry = await this.exclusive(p, async () => {
      if (p.blocked) throw new Error(p.blocked)
      const taken = new Set((await fs.readdir(join(p.folder, 'bible')).catch(() => [])).map((n) => lc(n)))
      const base = slugify(clean)
      let id = base
      for (let n = 2; taken.has(`${id}.md`); n++) id = `${base}-${n}`
      const text = formatEntry({ name: clean, kind: entryKind, aliases: [], triggers: [], priority: 50, always: false, body: '' })
      const bytes = Buffer.from(text, 'utf8')
      await writeBytesAtomic(p.folder, join(p.folder, 'bible', `${id}.md`), join(p.folder, '.rulz', 'tmp'), bytes)
      p.hashes.set(lc(`bible/${id}.md`), sha256(bytes))
      return parseEntry(id, text)
    })
    await this.publish(p)
    return entry
  }

  // ---------------------------------------------------------------------------------------------
  // Deletes and trash
  // ---------------------------------------------------------------------------------------------

  /** Moves files to the trash, then applies the outline change; puts the files back if that write fails. */
  private async trashThenWrite(p: OpenProject, rels: string[], label: string, outline: TrashOutline | undefined, apply?: () => NovelOutline): Promise<void> {
    await this.guard(p, ...rels.map((r) => join(p.folder, ...r.split('/'))), join(p.folder, '.rulz'))
    const manifest = await moveToTrash(p.folder, rels, label, outline)
    const before = p.outline
    try {
      if (apply) {
        // The outline may have been edited outside while the files moved; if so, do not write over it.
        const disk = await readBytes(p.folder, join(p.folder, 'outline.json'))
        if ((disk ? sha256(disk) : '') !== p.hashes.get('outline.json')) {
          throw new Error('outline.json changed on disk while deleting. Nothing was deleted; try again.')
        }
        p.outline = apply()
        await this.writeJson(p, 'outline.json', p.outline)
      }
    } catch (err) {
      p.outline = before
      try {
        await undoTrash(p.folder, manifest)
      } catch (back) {
        throw new Error(`${errText(err)} ${errText(back)}`)
      }
      throw err
    }
  }

  async deleteScene(folder: string, sceneId: string): Promise<NovelProjectState> {
    const p = this.get(folder)
    await this.mutate(p, async () => {
      const found = findScene(p.outline, sceneId)
      if (!found) throw new Error('That scene no longer exists.')
      const removed = removeScene(p.outline, sceneId) // throws for the last scene of a chapter
      const rels = [`manuscript/${found.scene.file}`]
      await this.withFileLocks(p, rels, () => this.trashThenWrite(p, rels, `Scene "${found.scene.title}"`, { kind: 'scene', chapterId: removed.chapterId, index: removed.index, scene: removed.scene }, () => removed.outline))
      for (const rel of rels) this.announce({ kind: 'deleted', folder: p.folder, path: rel })
      return { result: undefined }
    })
    return this.publish(p)
  }

  async deleteChapter(folder: string, chapterId: string): Promise<NovelProjectState> {
    const p = this.get(folder)
    await this.mutate(p, async () => {
      const found = findChapter(p.outline, chapterId)
      if (!found) throw new Error('That chapter no longer exists.')
      const removed = removeChapter(p.outline, chapterId)
      const n = found.chapter.scenes.length
      const label = `Chapter "${found.chapter.title}" (${n} ${n === 1 ? 'scene' : 'scenes'})`
      const rels = found.chapter.scenes.map((s) => `manuscript/${s.file}`)
      await this.withFileLocks(p, rels, () => this.trashThenWrite(p, rels, label, { kind: 'chapter', index: removed.index, chapter: removed.chapter }, () => removed.outline))
      for (const rel of rels) this.announce({ kind: 'deleted', folder: p.folder, path: rel })
      return { result: undefined }
    })
    return this.publish(p)
  }

  async deleteEntry(folder: string, entryId: string): Promise<NovelProjectState> {
    const p = this.get(folder)
    const rel = cleanRelPath(p.folder, `bible/${entryId}.md`)
    if (!USER_PATH.test(rel)) throw new Error('That entry does not exist.')
    await this.exclusive(p, async () => {
      if (p.blocked) throw new Error(p.blocked)
      if (!(await readBytes(p.folder, join(p.folder, ...rel.split('/'))))) throw new Error('That entry no longer exists.')
      await this.withFileLocks(p, [rel], async () => {
        await this.guard(p, join(p.folder, ...rel.split('/')), join(p.folder, '.rulz'))
        await moveToTrash(p.folder, [rel], `Entry "${entryId}"`)
        p.hashes.set(lc(rel), '')
        p.entryCache.delete(lc(rel))
      })
      this.announce({ kind: 'deleted', folder: p.folder, path: rel })
    })
    return this.publish(p)
  }

  async trash(folder: string): Promise<NovelTrashItem[]> {
    return listTrash(this.get(folder).folder)
  }

  async restoreTrash(folder: string, id: string): Promise<NovelProjectState> {
    const p = this.get(folder)
    const restored: string[] = []
    await this.mutate(p, async () => {
      const manifest: TrashManifest = await readManifest(p.folder, id)
      let next: NovelOutline | null = null
      // Check the outline side first (the manifest is untrusted JSON), so nothing is moved when it cannot be put back.
      if (manifest.outline) next = reinsert(p.outline, cleanTrashOutline(manifest.outline))
      const lockRels = manifest.files.flatMap((f) => {
        try {
          return typeof f.from === 'string' ? [cleanRelPath(p.folder, f.from, '.rulz/unsaved/')] : []
        } catch {
          return []
        }
      })
      await this.withFileLocks(p, lockRels, async () => {
        await this.guard(p, join(p.folder, '.rulz'), ...lockRels.map((r) => join(p.folder, ...r.split('/'))))
        await restoreFiles(p.folder, manifest)
        restored.push(...lockRels.filter((r) => USER_PATH.test(r)))
        for (const f of manifest.files) p.hashes.delete(lc(f.from))
        if (next) {
          const before = p.outline
          try {
            p.outline = next
            await this.writeJson(p, 'outline.json', p.outline)
          } catch (err) {
            p.outline = before
            // The outline does not list the files, so they go back into the trash item rather than sit as orphans.
            try {
              await returnToTrash(p.folder, manifest)
            } catch (back) {
              throw new Error(`${errText(err)} ${errText(back)}`)
            }
            throw err
          }
        }
      })
      await removeTrashFolder(p.folder, id)
      return { result: undefined }
    })
    for (const rel of restored) {
      const hash = await this.hashOf(p, rel)
      if (hash) this.announce({ kind: 'text', folder: p.folder, path: rel, hash })
    }
    return this.publish(p)
  }

  async emptyTrash(folder: string): Promise<void> {
    const p = this.get(folder)
    if (p.blocked) throw new Error(`${p.blocked} The project is read-only until that file is fixed.`)
    // On the project chain, like the other trash moves, so it cannot split a chapter's files mid-restore.
    await this.exclusive(p, async () => {
      const dir = trashRoot(p.folder)
      await assertAppPath(p.folder, dir)
      await assertNoLinks(dir)
      if (!(await fs.stat(dir).catch(() => null))) return
      await this.opts.trashItem(dir)
    })
  }

  /** Moves the whole project folder to the Recycle Bin; on failure nothing is deleted and the project stays open. */
  async deleteProject(folder: string): Promise<void> {
    if (typeof folder !== 'string' || !isAbsolute(folder)) throw new Error('The project folder must be a full path.')
    const dir = resolve(folder)
    const root = resolve(await this.root())
    if (projectKey(dir) === projectKey(root) || (!isInside(root, dir) && !this.projects.has(projectKey(dir)))) {
      throw new Error('Only a project folder can be deleted here.')
    }
    if (!(await readBytes(dir, join(dir, 'project.json')).catch(() => null))) throw new Error('That folder is not a RulZ novel project.')
    const open = this.projects.get(projectKey(dir))
    // The watcher holds the folder open; Windows can refuse to move a folder that is being watched.
    open?.watcher?.stop()
    try {
      if (open) {
        // New saves, journals and snapshots are refused from here on; the ones already queued are waited for below.
        open.deleting = true
        open.recordsOff = true
        while (open.recordHold) await open.recordHold
        await Promise.allSettled([...open.records])
      }
      await open?.chain.catch(() => undefined)
      // Saves in flight (scene saves, journals, snapshots) run on per-path queues, not the chain, and recreate
      // missing folders: wait for them before the folder is moved.
      await drainFolder(dir)
      await this.opts.trashItem(dir)
    } catch (err) {
      if (open) {
        open.recordsOff = false
        open.deleting = false
      }
      open?.watcher?.start()
      throw new Error(`Could not move the project folder to the Recycle Bin: ${errText(err)} Nothing was deleted.`)
    }
    this.projects.delete(projectKey(dir))
  }

  // ---------------------------------------------------------------------------------------------
  // Records (stage 4): recording of Writing AI actions, and the training-data export
  // ---------------------------------------------------------------------------------------------

  /** True while appends are allowed: the project is the open one, not blocked, not closing, recording not off. */
  private recordsLive(p: OpenProject): boolean {
    return this.projects.get(projectKey(p.folder)) === p && !p.blocked && !p.recordsOff && p.project.recording !== false
  }

  /** For an append already admitted (tracked): the project may close meanwhile, but it is never written when blocked or recording is off. */
  private recordsAllowed(p: OpenProject): boolean {
    return !p.blocked && p.project.recording !== false
  }

  /** Runs `fn` as a tracked append: it waits while the records are being deleted, and close / delete wait for it. */
  private async trackRecord<T>(p: OpenProject, fn: () => Promise<T>, refused: T): Promise<T> {
    while (p.recordHold) await p.recordHold
    // The project may have been closed or deleted while this waited: then it is not admitted.
    if (!this.recordsLive(p)) return refused
    // No await between the check above and the add below: deleteRecords sets the hold and then looks at the set.
    const run = fn()
    p.records.add(run)
    try {
      return await run
    } finally {
      p.records.delete(run)
    }
  }

  /** Appends one line to today's records file. False when it was not written; never throws for I/O. */
  async record(folder: string, line: RecordLine): Promise<boolean> {
    try {
      const p = typeof folder === 'string' ? this.projects.get(projectKey(folder)) : undefined
      if (!p || !this.recordsLive(p)) return false
      const clean = validateRecordLine(line)
      if (!clean) return false
      const text = JSON.stringify(clean)
      if (Buffer.byteLength(text, 'utf8') > MAX_RECORD_BYTES) return false
      const dir = join(p.folder, '.rulz', 'records')
      const target = join(dir, `${localDate((this.opts.now ?? (() => new Date()))())}.jsonl`)
      return await this.trackRecord(p, () =>
        serialized(target, async (): Promise<boolean> => {
          try {
            await assertAppPath(p.folder, target)
            await this.guard(p, dir, target)
            // Right before the folder is created. An admitted append still finishes when the project is closing:
            // close and delete await it, and appends that arrive after the close are refused before they are admitted.
            if (!this.recordsAllowed(p)) return false
            await retryLock(() => fs.mkdir(dir, { recursive: true }))
            await assertAppPath(p.folder, target)
            await this.guard(p, dir, target)
            if (!this.recordsAllowed(p)) return false
            await appendRecordText(target, text)
            return true
          } catch (err) {
            console.warn('Writing: could not record a line:', err)
            return false
          }
        }),
        false
      )
    } catch (err) {
      console.warn('Writing: could not record a line:', err)
      return false
    }
  }

  /** The .jsonl files of a project's records folder, sorted by name. */
  private async recordFiles(p: OpenProject): Promise<{ name: string; path: string; size: number }[]> {
    const dir = join(p.folder, '.rulz', 'records')
    await assertAppPath(p.folder, dir)
    await this.guard(p, dir)
    const out: { name: string; path: string; size: number }[] = []
    const missing = (err: unknown): boolean => (err as NodeJS.ErrnoException)?.code === 'ENOENT'
    const entries = await retryLock(() => fs.readdir(dir, { withFileTypes: true })).catch((err) => {
      if (missing(err)) return []
      throw err
    })
    for (const e of entries) {
      if (!e.isFile() || !/\.jsonl$/i.test(e.name)) continue
      const path = join(dir, e.name)
      await assertAppPath(p.folder, path)
      const st = await retryLock(() => fs.lstat(path)).catch((err) => {
        if (missing(err)) return null
        throw err
      })
      if (st?.isFile()) out.push({ name: e.name, path, size: st.size })
    }
    return out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))
  }

  async recordsInfo(folder: string): Promise<RecordsInfo> {
    const p = this.get(folder)
    const files = await this.recordFiles(p)
    return { files: files.length, bytes: files.reduce((n, f) => n + f.size, 0) }
  }

  /** Moves .rulz/records to the Recycle Bin. Waits for running appends and holds new ones off meanwhile. */
  async deleteRecords(folder: string): Promise<void> {
    const p = this.get(folder)
    if (p.blocked) throw new Error(`${p.blocked} The project is read-only until that file is fixed.`)
    while (p.recordHold) await p.recordHold
    let release: () => void = () => undefined
    p.recordHold = new Promise<void>((r) => (release = r))
    try {
      await Promise.allSettled([...p.records])
      const dir = join(p.folder, '.rulz', 'records')
      await assertAppPath(p.folder, dir)
      await assertNoLinks(dir)
      if (!(await fs.stat(dir).catch(() => null))) return
      await this.opts.trashItem(dir)
    } finally {
      p.recordHold = null
      release()
    }
  }

  /**
   * Writes train.jsonl, pairs.jsonl and manifest.json into a new subfolder of `outDir`. Two passes over the records
   * files, line by line: pass 1 keeps the outcome / viewed / final lines per id, pass 2 joins each request and
   * writes its rows. Bad lines are counted, never fatal.
   */
  async exportRecords(folder: string, outDir: string, opts: RecordsExportOptions): Promise<RecordsExportResult> {
    const p = this.get(folder)
    if (typeof outDir !== 'string' || !isAbsolute(outDir)) throw new Error('Choose a folder to export to.')
    const options: RecordsExportOptions = { editPairs: opts?.editPairs === true }
    const outReal = await fs.realpath(outDir).catch(() => null)
    if (!outReal || !(await fs.stat(outReal)).isDirectory()) throw new Error('That export folder does not exist.')
    const projectReal = await fs.realpath(p.folder).catch(() => resolve(p.folder))
    if (isInside(projectReal, outReal) || isInside(p.folder, resolve(outDir))) throw new Error('Choose a folder outside the project: the export cannot go inside it.')
    const files = await this.recordFiles(p)

    const now = (this.opts.now ?? (() => new Date()))()
    const base = `${slugify(p.project.title)}-${localStamp(now)}`
    let sub = ''
    for (let n = 1; ; n++) {
      const candidate = join(outReal, n === 1 ? base : `${base}-${n}`)
      try {
        await fs.mkdir(candidate)
        sub = candidate
        break
      } catch (err) {
        if ((err as NodeJS.ErrnoException).code !== 'EEXIST') throw err
        if (n > 500) throw new Error('Could not find a free folder name for the export.')
      }
    }

    const dropped: Record<string, number> = {}
    const joined = new RecordJoin()
    const tag = Math.random().toString(16).slice(2, 10)
    const train = new LineSink(join(sub, 'train.jsonl'), join(sub, `.train.jsonl.${tag}.tmp`))
    const pairs = new LineSink(join(sub, 'pairs.jsonl'), join(sub, `.pairs.jsonl.${tag}.tmp`))
    const mf = new LineSink(join(sub, 'manifest.json'), join(sub, `.manifest.json.${tag}.tmp`))
    const actions: Record<string, { train: number; pairs: number }> = {}
    try {
      // Pass 1: outcome / viewed / final lines per id, and every bad line counted once.
      for (const f of files) {
        await assertAppPath(p.folder, f.path)
        for await (const raw of readRawLines(f.path)) {
          const r = parseRecordBytes(raw)
          if (r.ok === 'blank') continue
          if (!r.ok) bump(dropped, r.reason)
          else joined.add(r.line)
        }
      }
      // Pass 2: request lines, one at a time.
      await train.open()
      await pairs.open()
      const seen = new Set<string>()
      for (const f of files) {
        await assertAppPath(p.folder, f.path)
        for await (const raw of readRawLines(f.path)) {
          const r = parseRecordBytes(raw)
          if (r.ok !== true || r.line.type !== 'request') continue
          const req = r.line
          if (seen.has(req.id)) {
            bump(dropped, 'duplicate')
            continue
          }
          seen.add(req.id)
          if (isCardAction(req.action)) {
            bump(dropped, 'cards')
            continue
          }
          const row = exportRequest(req, joined.get(req.id), options)
          if (row.drop) bump(dropped, row.drop)
          const a = (actions[req.action] ??= { train: 0, pairs: 0 })
          if (row.train) {
            await train.write(JSON.stringify(row.train))
            a.train++
          }
          for (const pr of row.pairs) {
            await pairs.write(JSON.stringify(pr))
            a.pairs++
          }
        }
      }
      const orphans = joined.ids().filter((id) => !seen.has(id)).length
      if (orphans) bump(dropped, 'noRequest', orphans)
      for (const k of Object.keys(actions)) if (!actions[k].train && !actions[k].pairs) delete actions[k]
      await train.commit()
      await pairs.commit()
      const manifest = {
        format: 1,
        app: this.opts.appVersion?.() ?? 'unknown',
        created: now.toISOString(),
        project: p.project.title,
        dateRange: { from: files[0]?.name.replace(/\.jsonl$/i, '') ?? null, to: files.length ? files[files.length - 1].name.replace(/\.jsonl$/i, '') : null },
        files: { 'train.jsonl': train.count, 'pairs.jsonl': pairs.count },
        actions,
        dropped,
        options
      }
      await mf.open()
      await mf.write(JSON.stringify(manifest, null, 2))
      await mf.commit()
    } catch (err) {
      await Promise.all([train.abort(), pairs.abort(), mf.abort()])
      // Leave nothing half-written behind; the subfolder is ours and new.
      for (const n of ['train.jsonl', 'pairs.jsonl', 'manifest.json']) await fs.rm(join(sub, n), { force: true }).catch(() => undefined)
      await fs.rmdir(sub).catch(() => undefined)
      throw new Error(`The export failed: ${errText(err)}`)
    }
    return { outDir: sub, train: train.count, pairs: pairs.count, dropped }
  }

  // ---------------------------------------------------------------------------------------------
  // History
  // ---------------------------------------------------------------------------------------------

  private historyPath(p: OpenProject, rel: unknown): string {
    const clean = cleanRelPath(p.folder, rel)
    if (/^(project|outline|state)\.json$/i.test(clean)) return clean.toLowerCase()
    if (!USER_PATH.test(clean)) throw new Error(`There is no history for "${clean}".`)
    return clean
  }

  async history(folder: string, rel: string): Promise<NovelHistoryItem[]> {
    const p = this.get(folder)
    const clean = this.historyPath(p, rel)
    await this.guard(p, join(p.folder, '.rulz', 'history'))
    return listHistory(p.folder, clean)
  }

  async readSnapshot(folder: string, snapshot: string): Promise<string> {
    const p = this.get(folder)
    const clean = cleanRelPath(p.folder, snapshot, '.rulz/history/')
    if (!clean.toLowerCase().startsWith('.rulz/history/')) throw new Error('That is not a history snapshot.')
    await this.guard(p, join(p.folder, ...clean.split('/')))
    const bytes = await readBytes(p.folder, join(p.folder, ...clean.split('/')))
    if (!bytes) throw new Error('That snapshot no longer exists.')
    return decodeBytes(bytes).text
  }

  // ---------------------------------------------------------------------------------------------
  // Changes made outside the app (called by the watcher; public so tests can call it directly)
  // ---------------------------------------------------------------------------------------------

  async onExternalChange(folder: string, relIn: string): Promise<void> {
    const p = this.projects.get(projectKey(folder))
    if (!p) return
    const rel = relIn.replace(/\\/g, '/')
    const key = lc(rel)
    if (key === 'project.json' || key === 'outline.json' || key === 'state.json') {
      await this.exclusive(p, () => this.loadJson(p, true))
      return
    }
    if (!USER_PATH.test(rel)) return
    const path = join(p.folder, ...rel.split('/'))
    // Same per-path key as saveFile/readFile/journal, so events and saves never interleave.
    await serialized(path, async () => {
      p.entryCache.delete(key)
      p.wordCache.delete(key)
      const st = await fs.stat(path).catch(() => null)
      if (st?.isDirectory()) return
      // A read that fails (still locked after the retries) says nothing: only a missing file is "deleted".
      let bytes: Buffer | null
      try {
        bytes = await readBytes(p.folder, path)
      } catch {
        return
      }
      const hash = bytes ? sha256(bytes) : ''
      const known = p.hashes.get(key)
      if (known === hash) return
      p.hashes.set(key, hash)
      // The next save must snapshot this outside version before it overwrites it (rule 5).
      p.lastSnap.delete(key)
      if (hash) this.announce({ kind: 'text', folder: p.folder, path: rel, hash })
      else if (known) this.announce({ kind: 'deleted', folder: p.folder, path: rel })
      // The outline tree (words, missing, orphans) and the bible list follow the files.
      this.announce({ kind: 'state', folder: p.folder, state: await this.state(p) })
    })
  }
}

