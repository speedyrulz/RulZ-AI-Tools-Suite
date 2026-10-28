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
  '.DS_Store'
]

/** Files bigger than this (videos, models…) are left out of restore points. */
const BIG_FILE_BYTES = 50 * 1024 * 1024

export interface Snapshotter {
  readonly kind: 'git' | 'copy'
  begin(taskId: string, label: string): Promise<Snapshot>
  /** Called before the agent changes a file (only the copy fallback needs it). */
  beforeChange(snapshot: Snapshot, rel: string): Promise<void>
  /** Files changed since the snapshot, or null when only the agent's own changes are known. */
  changedSince(snapshot: Snapshot): Promise<ChangedFile[] | null>
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

  private git(args: string[], opts: { buffer?: boolean } = {}): Promise<string | Buffer> {
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
      execFile(
        'git',
        full,
        { cwd: this.workTree, windowsHide: true, maxBuffer: 256 * 1024 * 1024, encoding: opts.buffer ? 'buffer' : 'utf8' },
        (err, stdout, stderr) => {
          if (err) reject(new Error(`git ${args[0]} failed: ${String(stderr || err.message).trim()}`))
          else resolve(stdout)
        }
      )
    })
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

  /** Writes the exclude list, adding any very large files found in the project. */
  private async writeExcludes(): Promise<void> {
    const big = await findBigFiles(this.workTree)
    const lines = ['# Managed by RulZ Coding Agent', ...SNAPSHOT_EXCLUDES, ...big.map((p) => `/${p.replace(/([*?[\]!#])/g, '\\$1')}`)]
    await fs.mkdir(join(this.gitDir, 'info'), { recursive: true })
    await fs.writeFile(join(this.gitDir, 'info', 'exclude'), lines.join('\n') + '\n', 'utf8')
  }

  private async commitAll(message: string): Promise<string> {
    await this.writeExcludes()
    await this.git(['add', '--all', '--ignore-errors', '.']).catch(() => this.git(['add', '--all', '.']))
    await this.git(['commit', '--quiet', '--allow-empty', '--no-verify', '-m', message])
    return String(await this.git(['rev-parse', 'HEAD'])).trim()
  }

  async begin(_taskId: string, label: string): Promise<Snapshot> {
    await this.ensure()
    return { kind: 'git', commit: await this.commitAll(`Before: ${label}`) }
  }

  async beforeChange(): Promise<void> {}

  async changedSince(snapshot: Snapshot): Promise<ChangedFile[]> {
    if (snapshot.kind !== 'git') return []
    await this.ensure()
    await this.writeExcludes()
    const out: ChangedFile[] = []
    const diff = String(await this.git(['diff', '--name-status', '--no-renames', '-z', snapshot.commit])).split('\0')
    for (let i = 0; i + 1 < diff.length; i += 2) {
      const status = diff[i]
      const path = diff[i + 1]
      if (!path) continue
      out.push({ path, change: status.startsWith('D') ? 'deleted' : status.startsWith('A') ? 'created' : 'changed' })
    }
    const untracked = String(await this.git(['ls-files', '--others', '--exclude-standard', '-z'])).split('\0').filter(Boolean)
    for (const path of untracked) out.push({ path, change: 'created' })
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

async function findBigFiles(root: string): Promise<string[]> {
  const big: string[] = []
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
      for (const e of entries) {
        scanned++
        const abs = join(dir, e.name)
        if (e.isDirectory()) {
          if (!SKIP_DIRS.has(e.name.toLowerCase()) && !e.isSymbolicLink()) next.push(abs)
        } else if (e.isFile()) {
          const size = await fs.stat(abs).then((s) => s.size, () => 0)
          if (size > BIG_FILE_BYTES) big.push(toRel(root, abs))
        }
      }
    }
    level = next
  }
  return big
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
