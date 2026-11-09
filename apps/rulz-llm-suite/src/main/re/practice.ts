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

export interface KitTask {
  id: string
  family: string
  tier: string
  binary: string
  prompt: string
  answer: string
  check: TaskCheck
}

const TASK_ID_RE = /^[\w-]{1,80}$/

export async function readKitTasks(kitDir: string): Promise<KitTask[]> {
  const text = await fs.readFile(join(kitDir, 'build', 'tasks.jsonl'), 'utf8')
  return text
    .split(/\r?\n/)
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as KitTask)
    .filter((t) => TASK_ID_RE.test(t.id) && t.binary && basename(t.binary) === t.binary)
}

/** Tasks for the page: no answers or checks, so they never reach the renderer (or the model). */
export async function listPractice(kitDir: string): Promise<RePracticeInfo> {
  try {
    const tasks = await readKitTasks(kitDir)
    return {
      kitDir,
      tasks: tasks.map((t): RePracticeTask => ({ id: t.id, family: t.family, tier: t.tier, binary: t.binary, prompt: t.prompt }))
    }
  } catch (err) {
    const e = err as NodeJS.ErrnoException
    return {
      kitDir,
      tasks: [],
      error:
        e.code === 'ENOENT'
          ? `No build/tasks.jsonl in ${kitDir}. Generate tasks there first: python -m kit.generate --per-family 30 --seed 1`
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
  const binDir = join(kitDir, 'build', 'bin', task.id)
  const work = join(workRoot, task.id, String(Date.now()))
  await fs.mkdir(work, { recursive: true })
  for (const name of await fs.readdir(binDir)) {
    const src = join(binDir, name)
    if ((await fs.stat(src)).isFile()) await fs.copyFile(src, join(work, name))
  }
  const target = join(work, task.binary)
  return { target, prompt: task.prompt.replaceAll('{exe}', target), task }
}

const norm = (s: string): string => s.toLowerCase().replace(/[^a-z0-9]/g, '')

function run(exe: string, args: string[], cwd?: string): Promise<{ code: number; stdout: string }> {
  return new Promise((done) => {
    execFile(exe, args, { cwd, timeout: 20_000, windowsHide: true }, (err, stdout) => {
      const code = err ? (typeof (err as { code?: unknown }).code === 'number' ? ((err as { code: number }).code) : -1) : 0
      done({ code, stdout: String(stdout ?? '') })
    })
  })
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
  return { passed: await checkAnswer(task, answer, exe), expected: task.answer }
}
