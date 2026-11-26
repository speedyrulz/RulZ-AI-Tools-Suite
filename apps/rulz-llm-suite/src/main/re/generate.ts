// Generating a new practice task set with the RE training kit's own generator
// (`python -m kit.generate --out <kit>/sets/<id> ...`, run from the kit folder).
import { execFile, spawn, type ChildProcess } from 'node:child_process'
import { randomBytes, randomInt } from 'node:crypto'
import { promises as fs } from 'node:fs'
import { isAbsolute, join, relative, resolve } from 'node:path'
import type { ReGenerateProgress, ReGenerateRequest } from '@shared/types'
import type { PythonCommand } from './python'
import { ORIGINAL_SET, SET_ID_RE } from './practice'

export const MAX_SEED = 2 ** 31 - 1
const FAMILY_RE = /^[a-z0-9][a-z0-9_]{0,59}$/
/** Lines of generator output kept for an error message. */
const TAIL_LINES = 40

export interface CheckedGenerate {
  label: string
  perFamily: number
  families: string[]
  seed: number
}

/** Checks a window's request against the families the generator lists. Throws a sentence for the first problem. */
export function validateGenerate(req: ReGenerateRequest, known: string[], randomSeed: () => number = () => randomInt(0, MAX_SEED + 1)): CheckedGenerate {
  const label = typeof req?.label === 'string' ? req.label.trim() : ''
  if (label.length < 1 || label.length > 80) throw new Error('The label needs 1 to 80 characters.')
  if (!Number.isInteger(req.perFamily) || req.perFamily < 1 || req.perFamily > 50) throw new Error('Tasks per family must be a whole number from 1 to 50.')
  if (!Array.isArray(req.families) || !req.families.length) throw new Error('Pick at least one family.')
  const families = [...new Set(req.families)]
  for (const f of families) {
    if (typeof f !== 'string' || !FAMILY_RE.test(f) || !known.includes(f)) throw new Error(`The kit has no family called ${String(f).slice(0, 40)}.`)
  }
  let seed: number
  if (req.seed === undefined || req.seed === null) seed = randomSeed()
  else if (!Number.isInteger(req.seed) || req.seed < 0 || req.seed > MAX_SEED) throw new Error(`The seed must be a whole number from 0 to ${MAX_SEED}.`)
  else seed = req.seed
  return { label, perFamily: req.perFamily, families, seed }
}

/** Lowercase letters, digits and dashes of a label, at most 30 characters; 'set' when nothing is left. */
export function slugOf(label: string): string {
  const slug = label
    .normalize('NFKD')
    .replace(/\p{M}/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+/, '')
    .slice(0, 30)
    .replace(/-+$/, '')
  return slug || 'set'
}

const randomSuffix = (): string => {
  const n = randomBytes(4).readUInt32BE(0)
  return n.toString(36).padStart(4, '0').slice(-4)
}

/**
 * <slug>-<4 random base36 characters>. The dash keeps it from ever matching the old s<digits> ids, and it is
 * retried while `taken` says the id exists.
 */
export function deriveSetId(label: string, taken: (id: string) => boolean = () => false, suffix: () => string = randomSuffix): string {
  const slug = slugOf(label)
  for (let i = 0; i < 20; i++) {
    const id = `${slug}-${suffix()}`
    if (SET_ID_RE.test(id) && !/^s\d+$/.test(id) && id !== ORIGINAL_SET && !taken(id)) return id
  }
  throw new Error('Could not find a free set id; try another label.')
}

/**
 * The folder of a generated set, only when `id` is a valid set id and the folder lies strictly inside <kit>/sets.
 * Everything that deletes or moves a set goes through this.
 */
export function setFolder(kitDir: string, id: string): string {
  if (typeof id !== 'string' || !SET_ID_RE.test(id) || id === ORIGINAL_SET) throw new Error('Only sets generated under sets/ can be removed.')
  const root = resolve(kitDir, 'sets')
  const dir = resolve(root, id)
  const rel = relative(root, dir)
  if (!rel || rel.startsWith('..') || isAbsolute(rel) || /[/\\]/.test(rel)) throw new Error('That set is not inside the kit’s sets folder.')
  return dir
}

/** The generator's command line (after the Python command). */
export function generatorArgs(id: string, out: string, c: CheckedGenerate): string[] {
  return ['-u', '-m', 'kit.generate', '--out', out, '--seed', String(c.seed), '--per-family', String(c.perFamily), '--families', c.families.join(','), '--label', c.label, '--id-prefix', id]
}

/** Reads `TOTAL n` and `<task id> [tier] <exe>` lines; returns the new progress, or null for any other line. */
export function parseProgressLine(line: string, state: { done: number; total: number }): ReGenerateProgress | null {
  const t = /^TOTAL (\d+)\s*$/.exec(line)
  if (t) {
    state.total = Number(t[1])
    return { done: state.done, total: state.total, line }
  }
  if (/^\S+ \[[^\]]*\] \S/.test(line)) {
    state.done++
    return { done: state.done, total: Math.max(state.total, state.done), line }
  }
  return null
}

/** Stops a process and everything it started (the generator starts the C compiler). */
function killTree(child: ChildProcess): void {
  if (child.pid === undefined) return
  if (process.platform === 'win32') {
    execFile('taskkill', ['/pid', String(child.pid), '/T', '/F'], { windowsHide: true }, () => undefined)
  } else {
    try {
      child.kill('SIGKILL')
    } catch {
      // already gone
    }
  }
}

export interface GeneratorDeps {
  findPython: () => Promise<PythonCommand | null>
  /** Tests run a stand-in script instead of Python. */
  buildCommand?: (py: PythonCommand, args: string[]) => { command: string; args: string[] }
}

/** Runs the kit's generator, one job at a time. */
export class SetGenerator {
  private child: ChildProcess | null = null
  private cancelled = false
  private busy = false

  constructor(private readonly deps: GeneratorDeps) {}

  get running(): boolean {
    return this.busy
  }

  private async command(args: string[]): Promise<{ command: string; args: string[] }> {
    const py = await this.deps.findPython()
    if (!py) throw new Error('Python 3.8 or newer was not found. The kit’s generator needs it (and Visual Studio Build Tools to compile the tasks).')
    return this.deps.buildCommand ? this.deps.buildCommand(py, args) : { command: py.command, args: [...py.args, ...args] }
  }

  private async requireKit(kitDir: string): Promise<void> {
    const ok = await fs.stat(join(kitDir, 'kit')).then((s) => s.isDirectory(), () => false)
    if (!ok) throw new Error(`${kitDir} has no kit folder, so it is not an RE training kit.`)
  }

  /** The family names the generator offers. Needs no compiler. */
  async listFamilies(kitDir: string): Promise<string[]> {
    await this.requireKit(kitDir)
    const cmd = await this.command(['-m', 'kit.generate', '--list-families'])
    const out = await new Promise<string>((done, fail) => {
      execFile(cmd.command, cmd.args, { cwd: kitDir, windowsHide: true, timeout: 20_000, env: { ...process.env, PYTHONUTF8: '1' } }, (err, stdout, stderr) => {
        if (err) fail(new Error(`Could not list the kit's families: ${(String(stderr) || String(stdout) || err.message).trim().split(/\r?\n/).slice(-8).join('\n')}`))
        else done(String(stdout))
      })
    })
    let list: unknown
    try {
      list = JSON.parse(out)
    } catch {
      throw new Error('The kit’s generator did not answer with a list of families.')
    }
    if (!Array.isArray(list) || !list.every((f) => typeof f === 'string' && FAMILY_RE.test(f))) throw new Error('The kit’s generator did not answer with a list of families.')
    return list as string[]
  }

  /** Removes a partial set folder, only the exact folder and only inside <kit>/sets. */
  async removePartial(kitDir: string, id: string): Promise<void> {
    const dir = setFolder(kitDir, id)
    await fs.rm(dir, { recursive: true, force: true, maxRetries: 3, retryDelay: 200 }).catch(() => undefined)
  }

  /** Generates a set. Resolves with its id; rejects with a sentence on failure or cancel (the partial folder is gone by then). */
  async generate(kitDir: string, req: ReGenerateRequest, onProgress: (p: ReGenerateProgress) => void): Promise<{ setId: string }> {
    if (this.busy) throw new Error('A set is already being generated.')
    this.busy = true
    this.cancelled = false
    let id = ''
    try {
      await this.requireKit(kitDir)
      const checked = validateGenerate(req, await this.listFamilies(kitDir))
      const existing = new Set(
        await fs.readdir(join(kitDir, 'sets')).then((n) => n, () => [] as string[])
      )
      await fs.mkdir(join(kitDir, 'sets'), { recursive: true })
      // Reserve the folder with an exclusive mkdir: only a folder this job created is ever cleaned up below.
      let out = ''
      for (let attempt = 0; !id; attempt++) {
        const candidate = deriveSetId(checked.label, (i) => existing.has(i))
        const dir = setFolder(kitDir, candidate)
        try {
          await fs.mkdir(dir)
          id = candidate
          out = dir
        } catch (e) {
          if ((e as NodeJS.ErrnoException).code !== 'EEXIST' || attempt >= 20) throw e
          existing.add(candidate)
        }
      }
      const cmd = await this.command(generatorArgs(id, out, checked))
      if (this.cancelled) throw new Error('Generation cancelled.')
      const tail: string[] = []
      const state = { done: 0, total: 0 }
      onProgress({ done: 0, total: 0, line: '' })
      const code = await new Promise<number>((done, fail) => {
        const child = spawn(cmd.command, cmd.args, {
          cwd: kitDir,
          windowsHide: true,
          stdio: ['ignore', 'pipe', 'pipe'],
          env: { ...process.env, PYTHONUTF8: '1', PYTHONUNBUFFERED: '1' }
        })
        this.child = child
        let rest = ''
        const keep = (line: string): void => {
          tail.push(line)
          if (tail.length > TAIL_LINES) tail.shift()
        }
        const onOut = (chunk: Buffer): void => {
          rest += chunk.toString('utf8')
          const lines = rest.split(/\r?\n/)
          rest = lines.pop() ?? ''
          for (const l of lines) {
            keep(l)
            const p = parseProgressLine(l, state)
            if (p) onProgress(p)
          }
        }
        child.stdout.on('data', onOut)
        child.stderr.on('data', (c: Buffer) => {
          for (const l of c.toString('utf8').split(/\r?\n/)) if (l) keep(l)
        })
        child.on('error', fail)
        child.on('close', (c) => {
          if (rest) {
            keep(rest)
            const p = parseProgressLine(rest, state)
            if (p) onProgress(p)
          }
          done(c ?? -1)
        })
      })
      if (this.cancelled) throw new Error('Generation cancelled.')
      if (code !== 0) throw new Error(`The generator failed (exit code ${code}). It needs Python 3 and Visual Studio Build Tools (the C compiler).\n${tail.join('\n')}`.trim())
      const finished = await fs.stat(join(out, 'set.json')).then((s) => s.isFile(), () => false)
      if (!finished) throw new Error(`The generator stopped without finishing the set.\n${tail.join('\n')}`.trim())
      return { setId: id }
    } catch (err) {
      if (id) await this.removePartial(kitDir, id)
      throw err
    } finally {
      this.child = null
      this.busy = false
    }
  }

  /** Kills the running generator and its children. The generate() call then rejects and removes the partial folder. */
  cancel(): void {
    if (!this.busy) return
    this.cancelled = true
    if (this.child) killTree(this.child)
  }
}
