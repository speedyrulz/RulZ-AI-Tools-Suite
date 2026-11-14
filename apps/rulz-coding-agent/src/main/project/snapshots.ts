// Restore points for "Undo last task". With git installed, each project gets its own repository in
// the app's data folder (never a .git inside the project, so the owner's own repositories are
// untouched). Without git, files the agent changes are copied aside before each change.
import { execFile } from 'node:child_process'
import { promises as fs, type Dirent } from 'node:fs'
import { dirname, join } from 'node:path'
import type { ChangedFile, TaskRecord } from '@shared/types'
import { SKIP_DIRS } from '../agent/fileTools'
import { toRel } from '../agent/pathGuard'

export type Snapshot = NonNullable<TaskRecord['snapshot']>

/** Never put into restore points: installed packages, build output, caches. */
export const SNAPSHOT_EXCLUDES = [
  'node_modules/',
  'dist/',
  'out/',
  '.venv/',
  'venv/',
  '__pycache__/',
  '*.pyc',
  '.next/',
  '.cache/',
  '.parcel-cache/',
  'coverage/',
  '.pytest_cache/',
  '.mypy_cache/',
  '*.log',
  'Thumbs.db',
  '.DS_Store',
  // Big build folders their tools rebuild. Restore points don't follow the project's own
  // .gitignore (it often lists files that matter, like .env), so these are named here.
  'target/',
  '.gradle/',
  '.tox/',
  '.nox/',
  '.turbo/',
  '.svelte-kit/',
  '.nuxt/',
  '.angular/',
  '.terraform/'
]

/** Folder names the project walk doesn't go into: the excluded folders above. */
const EXCLUDED_DIRS = new Set([...SKIP_DIRS, ...SNAPSHOT_EXCLUDES.filter((p) => p.endsWith('/')).map((p) => p.slice(0, -1).toLowerCase())])

const splitZ = (out: string | Buffer): string[] => String(out).split('\0').filter(Boolean)

/**
 * Attributes for every file, ahead of the project's own .gitattributes: files are stored and put
 * back byte for byte (no line-ending changes, no LFS or other filters).
 */
const PLAIN_ATTRIBUTES = '* -text -eol -filter -ident -working-tree-encoding\n'

/** Files bigger than this (videos, models…) are left out of restore points. */
const BIG_FILE_BYTES = 50 * 1024 * 1024

export interface Snapshotter {
  readonly kind: 'git' | 'copy'
  begin(taskId: string, label: string): Promise<Snapshot>
  /** Called before the agent changes a file (only the copy fallback needs it). */
  beforeChange(snapshot: Snapshot, rel: string): Promise<void>
  /**
   * Files changed since the snapshot, or null when only the agent's own changes are known.
   * `rescan` (default true) first looks for new very large files to leave out, a walk over the project.
   */
  changedSince(snapshot: Snapshot, opts?: { rescan?: boolean }): Promise<ChangedFile[] | null>
  /** A file's content at the snapshot: null when it didn't exist, undefined when unknown. */
  fileBefore(snapshot: Snapshot, rel: string): Promise<Buffer | null | undefined>
  restore(snapshot: Snapshot, label: string): Promise<void>
}

let gitChecked: Promise<boolean> | null = null

export function gitInstalled(): Promise<boolean> {
  gitChecked ??= new Promise((resolve) => {
    execFile('git', ['--version'], { windowsHide: true }, (err) => resolve(!err))
  })
  return gitChecked
}

export class GitSnapshots implements Snapshotter {
  readonly kind = 'git' as const
  private ready: Promise<void> | null = null

  constructor(
    private readonly gitDir: string,
    private readonly workTree: string
  ) {}

  private git(args: string[], opts: { buffer?: boolean; input?: string } = {}): Promise<string | Buffer> {
    const full = [
      '--git-dir',
      this.gitDir,
      '--work-tree',
      this.workTree,
      '-c',
      'core.autocrlf=false',
      '-c',
      'core.safecrlf=false',
      '-c',
      'core.quotepath=false',
      '-c',
      'core.longpaths=true',
      '-c',
      'user.name=RulZ Coding Agent',
      '-c',
      'user.email=agent@rulz.local',
      '-c',
      'commit.gpgsign=false',
      ...args
    ]
    return new Promise((resolve, reject) => {
      const child = execFile(
        'git',
        full,
        { cwd: this.workTree, windowsHide: true, maxBuffer: 256 * 1024 * 1024, encoding: opts.buffer ? 'buffer' : 'utf8' },
        (err, stdout, stderr) => {
          if (err) reject(new Error(`git ${args[0]} failed: ${String(stderr || err.message).trim()}`))
          else resolve(stdout)
        }
      )
      if (opts.input !== undefined) child.stdin?.end(opts.input)
    })
  }

  private get specPath(): string {
    return join(this.gitDir, 'info', 'rulz-add-pathspec')
  }

  private ensure(): Promise<void> {
    this.ready ??= (async () => {
      const exists = await fs.stat(join(this.gitDir, 'HEAD')).then(() => true, () => false)
      if (!exists) {
        await fs.mkdir(dirname(this.gitDir), { recursive: true })
        await new Promise<void>((resolve, reject) =>
          execFile('git', ['init', '--quiet', '--bare', this.gitDir], { windowsHide: true }, (err) => (err ? reject(err) : resolve()))
        )
        await this.git(['config', 'core.bare', 'false'])
        await this.git(['config', 'gc.auto', '0'])
      }
    })().catch((e) => {
      this.ready = null
      throw e
    })
    return this.ready
  }

  /**
   * Walks the project and writes our own exclude list (with the very large files it found), the
   * attributes that keep files byte for byte, and the pathspecs `git add` uses (the whole project
   * apart from folders with a repository of their own, which it returns).
   */
  private async writeExcludes(): Promise<{ nested: string[] }> {
    const { big, nested } = await scanProject(this.workTree)
    const lines = ['# Managed by RulZ Coding Agent', ...SNAPSHOT_EXCLUDES, ...big.map((p) => `/${p.replace(/([*?[\]!#])/g, '\\$1')}`)]
    const spec = ['.', ...nested.map((p) => `:(exclude,literal)${p}`)]
    await fs.mkdir(join(this.gitDir, 'info'), { recursive: true })
    await fs.writeFile(join(this.gitDir, 'info', 'exclude'), lines.join('\n') + '\n', 'utf8')
    await fs.writeFile(join(this.gitDir, 'info', 'attributes'), PLAIN_ATTRIBUTES, 'utf8')
    await fs.writeFile(this.specPath, spec.join('\n') + '\n', 'utf8')
    return { nested }
  }

  private async commitAll(message: string): Promise<string> {
    const { nested } = await this.writeExcludes()
    // Everything git takes by itself: changed and deleted files, and new ones neither the
    // project's .gitignore nor our list leaves out. (Folders with a repository of their own are
    // left out here: git refuses one without commits and stores one with commits as a bare link.)
    const add = ['add', '--all', `--pathspec-from-file=${this.specPath}`]
    await this.git([...add.slice(0, 2), '--ignore-errors', ...add.slice(2)]).catch(() => this.git(add))
    // Then the rest, file by file: new files only the project's .gitignore leaves out, and the
    // files of those folders (their own .git is never touched).
    const inNested = (await Promise.all(nested.map((rel) => filesUnder(this.workTree, rel)))).flatMap((f) => f ?? [])
    const tracked = nested.length ? splitZ(await this.git(['ls-files', '-z', '--', ...nested.map((rel) => `:(literal)${rel}`)])) : []
    const paths = [...new Set([...(await this.ignoredKept()), ...inNested, ...tracked])]
    if (paths.length) await this.git(['update-index', '--add', '--remove', '-z', '--stdin'], { input: paths.join('\0') + '\0' })
    await this.git(['commit', '--quiet', '--allow-empty', '--no-verify', '-m', message])
    return String(await this.git(['rev-parse', 'HEAD'])).trim()
  }

  /**
   * New files the project's own .gitignore leaves out but restore points keep, because that list
   * often names files that matter (.env, a data folder). Only our own list counts for leaving
   * things out, except that a whole ignored folder that is big (build output such as bin/ or
   * Library/) stays out, as before.
   */
  private async ignoredKept(): Promise<string[]> {
    const ignored = splitZ(await this.git(['ls-files', '--others', '--ignored', '--exclude-standard', '--directory', '-z']))
    const big = await this.bigFiles()
    const out: string[] = []
    for (const entry of ignored) {
      const parts = entry.replace(/\/$/, '').split('/')
      // Left out by our own list (git doesn't say which list ignored it).
      if (parts.slice(0, entry.endsWith('/') ? undefined : -1).some((p) => EXCLUDED_DIRS.has(p.toLowerCase()))) continue
      if (!entry.endsWith('/')) {
        if (!big.has(entry) && !EXCLUDED_FILES.some((re) => re.test(parts[parts.length - 1]))) out.push(entry)
        continue
      }
      const files = await filesUnder(this.workTree, entry.slice(0, -1), { maxFiles: KEPT_DIR_MAX_FILES, maxBytes: KEPT_DIR_MAX_BYTES })
      if (files) out.push(...files)
    }
    return out
  }

  /** The very large files the last walk found (written into the exclude list). */
  private async bigFiles(): Promise<Set<string>> {
    const text = await fs.readFile(join(this.gitDir, 'info', 'exclude'), 'utf8').catch(() => '')
    return new Set(
      text
        .split('\n')
        .filter((l) => l.startsWith('/'))
        .map((l) => l.slice(1).replace(/\\([*?[\]!#])/g, '$1'))
    )
  }

  async begin(_taskId: string, label: string): Promise<Snapshot> {
    await this.ensure()
    return { kind: 'git', commit: await this.commitAll(`Before: ${label}`) }
  }

  async beforeChange(): Promise<void> {}

  async changedSince(snapshot: Snapshot, opts: { rescan?: boolean } = {}): Promise<ChangedFile[]> {
    if (snapshot.kind !== 'git') return []
    await this.ensure()
    // The exclude list was written when the restore point was made; refreshing it walks the project.
    if (opts.rescan ?? true) await this.writeExcludes()
    const out: ChangedFile[] = []
    const diff = String(await this.git(['diff', '--name-status', '--no-renames', '-z', snapshot.commit])).split('\0')
    for (let i = 0; i + 1 < diff.length; i += 2) {
      const status = diff[i]
      const path = diff[i + 1]
      if (!path) continue
      out.push({ path, change: status.startsWith('D') ? 'deleted' : status.startsWith('A') ? 'created' : 'changed' })
    }
    const created = splitZ(await this.git(['ls-files', '--others', '--exclude-standard', '-z']))
    for (const path of created) {
      // A new folder with a repository of its own is listed as just the folder: list its files.
      if (path.endsWith('/')) for (const p of (await filesUnder(this.workTree, path.slice(0, -1))) ?? []) out.push({ path: p, change: 'created' })
      else out.push({ path, change: 'created' })
    }
    for (const path of await this.ignoredKept()) out.push({ path, change: 'created' })
    return out.sort((a, b) => a.path.localeCompare(b.path))
  }

  async fileBefore(snapshot: Snapshot, rel: string): Promise<Buffer | null> {
    if (snapshot.kind !== 'git') return null
    await this.ensure()
    try {
      return (await this.git(['show', `${snapshot.commit}:${rel}`], { buffer: true })) as Buffer
    } catch {
      return null
    }
  }

  async restore(snapshot: Snapshot, label: string): Promise<void> {
    if (snapshot.kind !== 'git') throw new Error('This restore point was not made with git.')
    await this.ensure()
    // Keep the current state too, so an undo can itself be recovered by hand.
    const current = await this.commitAll(`Before undo: ${label}`)
    await this.git(['update-ref', `refs/rulz/undone/${Date.now()}`, current])
    await this.git(['reset', '--quiet', '--hard', snapshot.commit])
  }
}

/**
 * Walks the project (not the excluded folders): files too big for restore points, and folders
 * other than the project itself that hold a git repository of their own (a .git folder, or a .git
 * file as in submodules and worktrees).
 */
async function scanProject(root: string): Promise<{ big: string[]; nested: string[] }> {
  const big: string[] = []
  const nested: string[] = []
  let level = [root]
  let scanned = 0
  while (level.length && scanned < 50_000) {
    const next: string[] = []
    for (const dir of level) {
      let entries: Dirent[]
      try {
        entries = await fs.readdir(dir, { withFileTypes: true })
      } catch {
        continue
      }
      const filesHere: string[] = []
      for (const e of entries) {
        scanned++
        const abs = join(dir, e.name)
        if (e.name.toLowerCase() === '.git') {
          if (dir !== root) nested.push(toRel(root, dir))
        } else if (e.isDirectory()) {
          if (!EXCLUDED_DIRS.has(e.name.toLowerCase()) && !e.isSymbolicLink()) next.push(abs)
        } else if (e.isFile()) filesHere.push(abs)
      }
      const sizes = await sizesOf(filesHere)
      filesHere.forEach((abs, i) => {
        if (sizes[i] > BIG_FILE_BYTES) big.push(toRel(root, abs))
      })
    }
    level = next
  }
  return { big, nested }
}

/** File sizes (0 for one that can't be read), looked at a batch at a time: a folder can hold tens of thousands. */
async function sizesOf(paths: string[]): Promise<number[]> {
  const out: number[] = []
  for (let i = 0; i < paths.length; i += 256) {
    out.push(...(await Promise.all(paths.slice(i, i + 256).map((abs) => fs.stat(abs).then((s) => s.size, () => 0)))))
  }
  return out
}

/** Excluded file names (`*.log`, `Thumbs.db`, …) as tests. */
const EXCLUDED_FILES = SNAPSHOT_EXCLUDES.filter((p) => !p.endsWith('/')).map(
  (p) => new RegExp(`^${p.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*')}$`, 'i')
)

/** A folder the project's .gitignore leaves out as a whole is kept in restore points only up to this size. */
const KEPT_DIR_MAX_FILES = 1000
const KEPT_DIR_MAX_BYTES = 100 * 1024 * 1024

/**
 * Every file under a project folder (relative to the project), leaving out .git, the excluded
 * folders and files, and very large files. Null when the folder is over the given limits.
 */
async function filesUnder(root: string, rel: string, limits: { maxFiles?: number; maxBytes?: number } = {}): Promise<string[] | null> {
  const out: string[] = []
  let bytes = 0
  const over = (): boolean => out.length > (limits.maxFiles ?? Infinity) || bytes > (limits.maxBytes ?? Infinity)
  const walk = async (dir: string): Promise<void> => {
    let entries: Dirent[]
    try {
      entries = await fs.readdir(dir, { withFileTypes: true })
    } catch {
      return
    }
    const files: string[] = []
    const dirs: string[] = []
    for (const e of entries) {
      const abs = join(dir, e.name)
      if (e.isDirectory()) {
        if (!EXCLUDED_DIRS.has(e.name.toLowerCase()) && !e.isSymbolicLink()) dirs.push(abs)
      } else if (e.isFile() && !EXCLUDED_FILES.some((re) => re.test(e.name))) files.push(abs)
    }
    const sizes = await sizesOf(files)
    files.forEach((abs, i) => {
      if (sizes[i] > BIG_FILE_BYTES) return
      out.push(toRel(root, abs))
      bytes += sizes[i]
    })
    for (const d of dirs) {
      if (over()) return
      await walk(d)
    }
  }
  await walk(join(root, rel))
  return over() ? null : out
}

interface CopyManifest {
  /** Relative path → whether it existed before the task. */
  files: Record<string, 'existed' | 'absent'>
}

/** Fallback without git: copies each file aside the first time the agent changes it in a task. */
export class CopySnapshots implements Snapshotter {
  readonly kind = 'copy' as const

  constructor(
    private readonly backupRoot: string,
    private readonly workTree: string,
    private readonly trash: (abs: string) => Promise<void>
  ) {}

  private manifestPath = (dir: string): string => join(dir, 'manifest.json')

  private async manifest(dir: string): Promise<CopyManifest> {
    try {
      return JSON.parse(await fs.readFile(this.manifestPath(dir), 'utf8')) as CopyManifest
    } catch {
      return { files: {} }
    }
  }

  async begin(taskId: string): Promise<Snapshot> {
    const dir = join(this.backupRoot, taskId)
    await fs.mkdir(join(dir, 'files'), { recursive: true })
    await fs.writeFile(this.manifestPath(dir), JSON.stringify({ files: {} }), 'utf8')
    return { kind: 'copy', dir }
  }

  async beforeChange(snapshot: Snapshot, rel: string): Promise<void> {
    if (snapshot.kind !== 'copy' || !rel) return
    const m = await this.manifest(snapshot.dir)
    if (m.files[rel]) return
    const src = join(this.workTree, rel)
    const exists = await fs.stat(src).then(() => true, () => false)
    if (exists) await fs.cp(src, join(snapshot.dir, 'files', rel), { recursive: true })
    m.files[rel] = exists ? 'existed' : 'absent'
    await fs.writeFile(this.manifestPath(snapshot.dir), JSON.stringify(m), 'utf8')
  }

  async changedSince(snapshot: Snapshot): Promise<ChangedFile[] | null> {
    if (snapshot.kind !== 'copy') return null
    const m = await this.manifest(snapshot.dir)
    const out: ChangedFile[] = []
    for (const [path, before] of Object.entries(m.files)) {
      const now = await fs.stat(join(this.workTree, path)).then(() => true, () => false)
      if (before === 'absent' && now) out.push({ path, change: 'created' })
      else if (before === 'existed') out.push({ path, change: now ? 'changed' : 'deleted' })
    }
    return out.sort((a, b) => a.path.localeCompare(b.path))
  }

  async fileBefore(snapshot: Snapshot, rel: string): Promise<Buffer | null | undefined> {
    if (snapshot.kind !== 'copy') return undefined
    const m = await this.manifest(snapshot.dir)
    if (!m.files[rel]) return undefined
    if (m.files[rel] === 'absent') return null
    return fs.readFile(join(snapshot.dir, 'files', rel)).catch(() => undefined)
  }

  async restore(snapshot: Snapshot): Promise<void> {
    if (snapshot.kind !== 'copy') throw new Error('This restore point was not made by copying files.')
    const m = await this.manifest(snapshot.dir)
    for (const [rel, before] of Object.entries(m.files)) {
      const target = join(this.workTree, rel)
      const exists = await fs.stat(target).then(() => true, () => false)
      if (before === 'absent') {
        if (exists) await this.trash(target)
      } else {
        if (exists) await this.trash(target)
        await fs.mkdir(dirname(target), { recursive: true })
        await fs.cp(join(snapshot.dir, 'files', rel), target, { recursive: true })
      }
    }
  }
}
