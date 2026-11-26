// Practice tasks from the RE training kit (C:\coding\Trainers\re-training-kit): targets with a
// known answer, graded automatically the way the kit's verify.py grades them.
import { execFile } from 'node:child_process'
import { promises as fs } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, dirname, isAbsolute, join, relative, resolve } from 'node:path'
import type { RePracticeInfo, RePracticeTask } from '@shared/types'

export type TaskCheck =
  | { type: 'run'; args: string[]; expect_stdout: string; expect_code: number }
  | { type: 'config'; expect_stdout: string; expect_code: number }
  | { type: 'exact'; value: string; normalize?: 'lower' }
  | { type: 'keyword'; accept: string; reject: string[] }
  | { type: 'env'; expect_stdout: string; expect_code: number }
  | { type: 'file'; expect_stdout: string; expect_code: number }

export interface KitTask {
  id: string
  family: string
  tier: string
  binary: string
  prompt: string
  answer: string
  check: TaskCheck
  /** Set by readKitTasks: the set (task folder) the task comes from. */
  set?: string
  /** Set by readKitTasks: that set's folder (holds tasks.jsonl and bin/<task id>/). */
  setDir?: string
}

/** One folder of generated tasks. */
export interface KitSet {
  id: string
  label: string
  dir: string
  count: number
  created?: string
  families: string[]
  seed?: number
}

export const ORIGINAL_SET = 'original'
/** Folder names (and so set ids) under <kit>/sets/. */
export const SET_ID_RE = /^[a-z0-9][a-z0-9_-]{0,39}$/

const TASK_ID_RE = /^[\w-]{1,80}$/

function parseTasks(text: string): KitTask[] {
  return text
    .split(/\r?\n/)
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as KitTask)
    .filter((t) => TASK_ID_RE.test(t.id) && t.binary && basename(t.binary) === t.binary)
}

interface LoadedSet {
  set: KitSet
  tasks: KitTask[]
}

async function readSetMeta(dir: string): Promise<Record<string, unknown> | null> {
  try {
    const m: unknown = JSON.parse(await fs.readFile(join(dir, 'set.json'), 'utf8'))
    return m && typeof m === 'object' && !Array.isArray(m) ? (m as Record<string, unknown>) : null
  } catch {
    return null
  }
}

async function loadSet(id: string, dir: string, fallbackLabel: string): Promise<LoadedSet> {
  const tasks = parseTasks(await fs.readFile(join(dir, 'tasks.jsonl'), 'utf8')).map((t) => ({ ...t, set: id, setDir: dir }))
  const meta = await readSetMeta(dir)
  const label = typeof meta?.label === 'string' && meta.label.trim() ? meta.label.trim() : fallbackLabel
  const families =
    Array.isArray(meta?.families) && meta.families.every((f) => typeof f === 'string')
      ? (meta.families as string[])
      : [...new Set(tasks.map((t) => t.family))]
  const seed = typeof meta?.seed === 'number' && Number.isFinite(meta.seed) ? meta.seed : undefined
  const created = typeof meta?.created === 'string' ? meta.created : undefined
  return { set: { id, label, dir, count: tasks.length, families, ...(created ? { created } : {}), ...(seed !== undefined ? { seed } : {}) }, tasks }
}

/**
 * Every set of the kit: the original (<kit>/build) and each finished folder under <kit>/sets/ (one with set.json and
 * tasks.jsonl; anything else is unfinished or broken and is skipped). `errors` has what stopped a set from loading.
 */
async function readSets(kitDir: string): Promise<{ sets: LoadedSet[]; errors: Error[] }> {
  const sets: LoadedSet[] = []
  const errors: Error[] = []
  const tryLoad = async (id: string, dir: string, label: string): Promise<void> => {
    try {
      sets.push(await loadSet(id, dir, label))
    } catch (e) {
      errors.push(e as Error)
    }
  }
  await tryLoad(ORIGINAL_SET, join(kitDir, 'build'), 'Original')
  let names: string[] = []
  try {
    names = (await fs.readdir(join(kitDir, 'sets'), { withFileTypes: true })).filter((d) => d.isDirectory()).map((d) => d.name).sort()
  } catch {
    // no sets folder
  }
  for (const name of names) {
    if (!SET_ID_RE.test(name) || name === ORIGINAL_SET) continue
    const dir = join(kitDir, 'sets', name)
    const [meta, tasks] = await Promise.all([
      fs.stat(join(dir, 'set.json')).then((s) => s.isFile(), () => false),
      fs.stat(join(dir, 'tasks.jsonl')).then((s) => s.isFile(), () => false)
    ])
    if (meta && tasks) await tryLoad(name, dir, name)
  }
  return { sets, errors }
}

/** The kit's task sets, original first. A missing or unreadable original is left out. */
export async function listSets(kitDir: string): Promise<KitSet[]> {
  return (await readSets(kitDir)).sets.map((s) => s.set)
}

/** Tasks of every set (the first set with an id wins), each annotated with its set and set folder. */
export async function readKitTasks(kitDir: string): Promise<KitTask[]> {
  const { sets, errors } = await readSets(kitDir)
  if (!sets.length && errors.length) throw errors[0]
  const seen = new Set<string>()
  const out: KitTask[] = []
  for (const s of sets) {
    for (const t of s.tasks) {
      if (seen.has(t.id)) continue
      seen.add(t.id)
      out.push(t)
    }
  }
  return out
}

/** Tasks for the page: no answers or checks, so they never reach the renderer (or the model). */
export async function listPractice(kitDir: string): Promise<RePracticeInfo> {
  try {
    const { sets, errors } = await readSets(kitDir)
    if (!sets.length) throw errors[0] ?? Object.assign(new Error('no sets'), { code: 'ENOENT' })
    const seen = new Set<string>()
    const dupes = new Set<string>()
    const tasks: RePracticeTask[] = []
    for (const s of sets) {
      for (const t of s.tasks) {
        if (seen.has(t.id)) {
          dupes.add(t.id)
          continue
        }
        seen.add(t.id)
        tasks.push({ id: t.id, family: t.family, tier: t.tier, binary: t.binary, prompt: t.prompt, set: s.set.id })
      }
    }
    const shown = [...dupes].slice(0, 3).join(', ')
    return {
      kitDir,
      tasks,
      sets: sets.map(({ set }) => ({
        id: set.id,
        label: set.label,
        count: set.count,
        families: set.families,
        ...(set.created ? { created: set.created } : {}),
        ...(set.seed !== undefined ? { seed: set.seed } : {})
      })),
      ...(dupes.size
        ? { warning: `${dupes.size} task id${dupes.size === 1 ? ' appears' : 's appear'} in more than one set (${shown}${dupes.size > 3 ? ', ...' : ''}); the first set's copy is used.` }
        : {})
    }
  } catch (err) {
    const e = err as NodeJS.ErrnoException
    return {
      kitDir,
      tasks: [],
      sets: [],
      error:
        e.code === 'ENOENT'
          ? `No practice tasks in ${kitDir} yet. Generate a set with New set… (or run python -m kit.generate --per-family 30 --seed 1 there).`
          : e.message
    }
  }
}

/**
 * Copies a task's binary (and its PDB, on tiers that have one) into a fresh work folder, so the
 * model sees only what the kit gives it, never the generator's source. Returns the target path
 * and the prompt with {exe} filled in.
 */
export async function preparePractice(kitDir: string, workRoot: string, taskId: string): Promise<{ target: string; prompt: string; task: KitTask }> {
  const task = (await readKitTasks(kitDir)).find((t) => t.id === taskId)
  if (!task) throw new Error(`Practice task ${taskId} is not in ${kitDir}.`)
  const work = join(workRoot, task.id, String(Date.now()))
  await copyTaskBin(kitDir, task, work)
  const target = join(work, task.binary)
  return { target, prompt: task.prompt.replaceAll('{exe}', target), task }
}

/** Copies the files of a task's bin folder into `work` (created when missing). */
async function copyTaskBin(kitDir: string, task: KitTask, work: string): Promise<void> {
  const binDir = join(task.setDir ?? join(kitDir, 'build'), 'bin', task.id)
  await fs.mkdir(work, { recursive: true })
  for (const name of await fs.readdir(binDir)) {
    const src = join(binDir, name)
    if ((await fs.stat(src)).isFile()) await fs.copyFile(src, join(work, name))
  }
}

/** True when `workRoot` and every folder from it down to `dir` that exists is a real folder, not a link. */
async function noLinksBelow(workRoot: string, dir: string): Promise<boolean> {
  const root = resolve(workRoot)
  const parts = relative(root, resolve(dir)).split(/[/\\]/).filter(Boolean)
  let at = root
  for (const p of ['', ...parts]) {
    at = p ? join(at, p) : at
    const st = await fs.lstat(at).catch(() => null)
    if (!st) return true // the rest does not exist yet and is made as real folders
    if (!st.isDirectory()) return false
  }
  return true
}

/** How long a practice work folder is kept. */
export const PRACTICE_KEEP_MS = 30 * 24 * 60 * 60 * 1000

/**
 * Deletes practice work folders (<workRoot>/<task>/<13-digit timestamp>) older than `maxAgeMs`, then a task
 * folder left empty. Anything else (files, links, other names) is left alone; errors per folder are ignored.
 */
export async function prunePracticeFolders(workRoot: string, now = Date.now(), maxAgeMs = PRACTICE_KEEP_MS): Promise<void> {
  const cutoff = now - maxAgeMs
  let tasks: string[]
  try {
    // A linked root (junction or symlink) is never followed: what it points at is not ours.
    if (!(await fs.lstat(workRoot)).isDirectory()) return
    tasks = await fs.readdir(workRoot)
  } catch {
    return
  }
  for (const task of tasks) {
    const taskDir = join(workRoot, task)
    try {
      if (!(await fs.lstat(taskDir)).isDirectory()) continue
      let removedAny = false
      for (const name of await fs.readdir(taskDir)) {
        if (!/^\d{13}$/.test(name) || Number(name) >= cutoff) continue
        const dir = join(taskDir, name)
        try {
          if (!(await fs.lstat(dir)).isDirectory()) continue
          await fs.rm(dir, { recursive: true, force: true })
          removedAny = true
        } catch {
          // locked by antivirus or already gone
        }
      }
      // Only a task folder this sweep emptied goes; any other empty folder is left as it is.
      if (removedAny && !(await fs.readdir(taskDir)).length) await fs.rmdir(taskDir)
    } catch {
      // not ours to worry about
    }
  }
}

const norm = (s: string): string => s.toLowerCase().replace(/[^a-z0-9]/g, '')

function run(exe: string, args: string[], cwd?: string, env?: NodeJS.ProcessEnv): Promise<{ code: number; stdout: string }> {
  return new Promise((done) => {
    execFile(exe, args, { cwd, env, timeout: 20_000, windowsHide: true }, (err, stdout) => {
      const code = err ? (typeof (err as { code?: unknown }).code === 'number' ? ((err as { code: number }).code) : -1) : 0
      done({ code, stdout: String(stdout ?? '') })
    })
  })
}

/** env answers: NAME=value, split at the first '='. Null when the name is not an environment variable name or the value is empty. */
export function parseEnvAnswer(answer: string): { name: string; value: string } | null {
  const i = answer.indexOf('=')
  if (i < 0) return null
  const name = answer.slice(0, i).trim()
  const value = answer.slice(i + 1).trim()
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) && value && !value.includes('\0') ? { name, value } : null
}

/** Variables every check run gets from this process (when they exist); an answer may not set them. */
export const ENV_BASE_NAMES = ['SystemRoot', 'windir', 'SystemDrive', 'TEMP', 'TMP', 'PATH']
const ENV_FORBIDDEN = /^(PATH|PATHEXT|COMSPEC|COR_|COMPLUS_|DOTNET_|NODE_|PYTHON|LD_|DYLD_|_NT_|APPDATA|LOCALAPPDATA|USERPROFILE|HOMEDRIVE|HOMEPATH|PROGRAMDATA|PROGRAMFILES|WINDIR|SYSTEMROOT|SYSTEMDRIVE|TEMP|TMP|JAVA|_JAVA|JDK_|CLASSPATH|PERL|RUBY|GIT_)/i

/**
 * The environment an env check runs the program with: a minimal one built from `base` (the base names that exist,
 * matched case-insensitively) plus NAME=value. Null (never run) when NAME is one of those or could steer a runtime or loader.
 */
export function envForCheck(name: string, value: string, base: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv | null {
  if (ENV_BASE_NAMES.some((b) => b.toLowerCase() === name.toLowerCase()) || ENV_FORBIDDEN.test(name)) return null
  const env: NodeJS.ProcessEnv = {}
  for (const b of ENV_BASE_NAMES) {
    const key = Object.keys(base).find((k) => k.toLowerCase() === b.toLowerCase())
    if (key !== undefined && base[key] !== undefined) env[key] = base[key]
  }
  env[name] = value
  return env
}

/** Largest file a file answer may describe. */
export const MAX_FILE_ANSWER_BYTES = 4096

/** file answers: hex bytes (0x prefixes, spaces and commas allowed). Null when empty, odd, not hex or over 4096 bytes. */
export function parseHexAnswer(answer: string): Buffer | null {
  const hex = answer
    .split(/[\s,]+/)
    .map((t) => t.replace(/^0x/i, ''))
    .join('')
  if (!hex || hex.length % 2 !== 0 || !/^[0-9a-fA-F]+$/.test(hex)) return null
  const bytes = Buffer.from(hex, 'hex')
  return bytes.length > MAX_FILE_ANSWER_BYTES ? null : bytes
}

/** Port of the kit's verify.check: does `answer` solve `task`? `exe` is the copy in the work folder. */
export async function checkAnswer(task: KitTask, answer: string | null, exe: string): Promise<boolean> {
  if (!answer) return false
  const c = task.check
  switch (c.type) {
    case 'run': {
      const r = await run(exe, c.args.map((a) => a.replaceAll('{answer}', answer)))
      return r.code === c.expect_code && r.stdout.includes(c.expect_stdout)
    }
    case 'config': {
      const i = answer.indexOf(';')
      if (i < 0) return false
      const fname = answer.slice(0, i).trim()
      const setting = answer.slice(i + 1).trim()
      if (!setting.includes('=') || !fname || basename(fname) !== fname) return false
      const tmp = await fs.mkdtemp(join(tmpdir(), 'rulz-re-'))
      try {
        await fs.writeFile(join(tmp, fname), setting + '\n', 'utf8')
        const r = await run(exe, [], tmp)
        return r.code === c.expect_code && r.stdout.includes(c.expect_stdout)
      } finally {
        await fs.rm(tmp, { recursive: true, force: true }).catch(() => undefined)
      }
    }
    case 'env': {
      const e = parseEnvAnswer(answer)
      if (!e) return false
      const env = envForCheck(e.name, e.value)
      if (!env) return false
      const r = await run(exe, [], undefined, env)
      return r.code === c.expect_code && r.stdout.includes(c.expect_stdout)
    }
    case 'file': {
      const bytes = parseHexAnswer(answer)
      if (!bytes) return false
      const tmp = await fs.mkdtemp(join(tmpdir(), 'rulz-re-'))
      try {
        const path = join(tmp, 'input.bin')
        await fs.writeFile(path, bytes)
        const r = await run(exe, [path])
        return r.code === c.expect_code && r.stdout.includes(c.expect_stdout)
      } finally {
        await fs.rm(tmp, { recursive: true, force: true }).catch(() => undefined)
      }
    }
    case 'exact':
      return c.normalize === 'lower' ? answer.toLowerCase() === c.value.toLowerCase() : answer === c.value
    case 'keyword': {
      const a = norm(answer)
      return a.includes(norm(c.accept)) && !c.reject.some((r) => a.includes(norm(r)))
    }
    default:
      throw new Error(`Unknown check type ${(c as { type: string }).type}`)
  }
}

/**
 * Grades a practice session's answer. `target` must be a work-folder copy made by
 * preparePractice, so grading only ever runs the kit's own task binaries.
 */
export async function gradePractice(
  kitDir: string,
  workRoot: string,
  taskId: string,
  target: string,
  answer: string | null
): Promise<{ passed: boolean; expected: string }> {
  const task = (await readKitTasks(kitDir)).find((t) => t.id === taskId)
  if (!task) throw new Error(`Practice task ${taskId} is not in ${kitDir}.`)
  const exe = resolve(target)
  const rel = relative(resolve(workRoot, task.id), dirname(exe))
  // One folder (the timestamp) below <workRoot>/<task>, and the task's own binary name.
  if (!rel || isAbsolute(rel) || rel.includes('..') || /[/\\]/.test(rel) || basename(exe) !== task.binary) {
    throw new Error('This practice session’s target is not a work-folder copy of the task binary.')
  }
  // An old chat whose work folder was pruned can still be graded: the copy is made again.
  // Only a folder this app names (a 13-digit time) is made again, and never through a link: a junction there
  // would put the copy, and the run that checks it, somewhere else.
  if (!(await fs.stat(exe).then((s) => s.isFile(), () => false)) && /^\d{13}$/.test(rel) && (await noLinksBelow(workRoot, dirname(exe)))) {
    await copyTaskBin(kitDir, task, dirname(exe))
  }
  return { passed: await checkAnswer(task, answer, exe), expected: task.answer }
}
