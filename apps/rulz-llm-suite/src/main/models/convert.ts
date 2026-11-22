// Converts a Hugging Face PEFT LoRA folder to a GGUF adapter with llama.cpp's convert_lora_to_gguf.py.
import { spawn, type ChildProcess } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import { existsSync, promises as fs } from 'node:fs'
import { dirname, join } from 'node:path'
import type { AdapterConvertRequest, AdapterConvertResult, AdapterConverterStatus } from '@shared/types'
import { killTree } from '../engine/process'
import type { PythonCommand } from '../re/python'

const SCRIPT_NAME = 'convert_lora_to_gguf.py'
/** How many folders above llama-server.exe are searched for the script. */
const MAX_LEVELS = 6
const LOG_TAIL = 40

/** Finds the converter in the llama.cpp checkout the engine's llama-server.exe was built from. */
export function findConverterScript(exePath: string, exists: (p: string) => boolean = existsSync): string | null {
  let dir = dirname(exePath)
  for (let level = 0; level <= MAX_LEVELS; level++) {
    const candidate = join(dir, SCRIPT_NAME)
    if (exists(candidate)) return candidate
    const up = dirname(dir)
    if (up === dir) break
    dir = up
  }
  return null
}

/** A plain file name ending in .gguf: no folders, no '..', none of the characters Windows forbids. Empty when nothing is left. */
export function sanitizeOutName(name: string): string {
  const base = name
    .trim()
    .split(/[\\/]/)
    .pop()!
    // eslint-disable-next-line no-control-regex
    .replace(/[<>:"|?*{},\u0000-\u001f]/g, '')
    .replace(/\.{2,}/g, '.')
    .replace(/[. ]+$/, '')
    .trim()
  const stem = base.replace(/\.gguf$/i, '').replace(/^\.+/, '')
  if (!stem || /^(con|prn|aux|nul|com\d|lpt\d)$/i.test(stem)) return ''
  return `${stem}.gguf`
}

/** Arguments after the Python command's own: the script, then its options. */
export function buildConvertArgs(script: string, req: AdapterConvertRequest, outPath: string): string[] {
  const args = [script, req.peftDir, '--outfile', outPath, '--outtype', req.outType]
  if (req.baseDir.trim()) args.push('--base', req.baseDir)
  else if (req.baseModelId.trim()) args.push('--base-model-id', req.baseModelId.trim())
  return args
}

export interface ConverterDeps {
  /** The llama-server.exe the engine uses; throws when there is none. */
  resolveExe: () => Promise<string>
  getModelsDir: () => string
  findPython: () => Promise<PythonCommand | null>
  /** One output line, as it comes. */
  onLog: (line: string) => void
  /** After a successful conversion, so the library picks the new file up. */
  onConverted: () => void
  /** Starts the process; injectable for tests. */
  spawnProcess?: (command: string, args: string[], cwd: string) => ChildProcess
  /** Finds the script from the exe; injectable for tests. */
  findScript?: (exePath: string) => string | null
}

const fail = (error: string, logTail: string[] = []): AdapterConvertResult => ({ ok: false, error, logTail })

/** Cuts a stream into lines; flush() gives the last, unterminated one. */
function lineSplitter(push: (line: string) => void): { feed: (chunk: Buffer | string) => void; flush: () => void } {
  let pending = ''
  return {
    feed: (chunk) => {
      pending += String(chunk)
      const parts = pending.split(/\r\n|\n|\r/)
      pending = parts.pop() ?? ''
      for (const l of parts) if (l.trim()) push(l)
    },
    flush: () => {
      if (pending.trim()) push(pending)
      pending = ''
    }
  }
}

export class AdapterConverter {
  private current: { proc: ChildProcess; cancelled: boolean } | null = null
  /** Set from the check for a running conversion until it ends, so two calls can't both pass it. */
  private busy = false
  /** A cancel that came before the process existed. */
  private cancelPending = false

  constructor(private readonly deps: ConverterDeps) {}

  async status(): Promise<AdapterConverterStatus> {
    let script: string | null = null
    let exeError = ''
    try {
      script = (this.deps.findScript ?? findConverterScript)(await this.deps.resolveExe())
      if (!script) exeError = `${SCRIPT_NAME} was not found in the llama.cpp folder next to the llama-server the app uses.`
    } catch {
      exeError = 'llama-server was not found, so the converter next to it cannot be located. Set the engine path in Settings.'
    }
    const py = await this.deps.findPython()
    const python = py ? [py.command, ...py.args].join(' ') : null
    const problems = [exeError, py ? '' : 'Python 3.8 or newer was not found.'].filter(Boolean)
    return { available: !!script && !!py, python, script, ...(problems.length ? { error: problems.join(' ') } : {}) }
  }

  cancel(): void {
    if (!this.current) {
      if (this.busy) this.cancelPending = true
      return
    }
    this.current.cancelled = true
    killTree(this.current.proc.pid)
  }

  async convert(req: AdapterConvertRequest): Promise<AdapterConvertResult> {
    if (this.busy) return fail('Another conversion is still running.')
    this.busy = true
    this.cancelPending = false
    try {
      return await this.run(req)
    } finally {
      this.busy = false
      this.cancelPending = false
      this.current = null
    }
  }

  private async run(req: AdapterConvertRequest): Promise<AdapterConvertResult> {
    const py = await this.deps.findPython()
    const status = await this.status()
    if (!status.available || !status.script || !py) return fail(status.error ?? 'The converter is not available.')

    if (!req.peftDir || !existsSync(join(req.peftDir, 'adapter_config.json'))) {
      return fail('The adapter folder has no adapter_config.json.')
    }
    if (req.baseDir.trim() && !existsSync(join(req.baseDir, 'config.json'))) return fail('The base model folder has no config.json.')
    const outName = sanitizeOutName(req.outName)
    if (!outName) return fail('Give the result a file name.')
    const modelsDir = this.deps.getModelsDir()
    if (!modelsDir || !existsSync(modelsDir)) return fail('The models folder does not exist.')
    const outPath = join(modelsDir, outName)
    // Refusing instead of overwriting also means a partial file left by a failure is always ours to delete.
    if (existsSync(outPath)) return fail(`${outName} already exists in the models folder; choose another name.`)

    // Written under a name the library watcher ignores, then linked into place, so a file that appears
    // under the final name in the meantime is never overwritten or deleted.
    const tmpPath = join(modelsDir, `${outName}.${randomBytes(4).toString('hex')}.part`)
    const args = [...py.args, ...buildConvertArgs(status.script, req, tmpPath)]
    const tail: string[] = []
    const push = (line: string): void => {
      tail.push(line)
      if (tail.length > LOG_TAIL) tail.shift()
      this.deps.onLog(line)
    }
    push(`$ ${py.command} ${args.join(' ')}`)
    if (this.cancelPending) return fail('The conversion was cancelled.', tail)

    const cwd = dirname(status.script)
    let proc: ChildProcess
    try {
      proc = this.deps.spawnProcess
        ? this.deps.spawnProcess(py.command, args, cwd)
        : spawn(py.command, args, {
            cwd,
            // UTF-8 so non-ASCII file names and messages survive Windows' code page.
            env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' },
            windowsHide: true,
            stdio: ['ignore', 'pipe', 'pipe']
          })
    } catch (err) {
      return fail(`Could not start Python: ${(err as Error).message}`, tail)
    }
    const run = { proc, cancelled: false }
    this.current = run

    const out = lineSplitter(push)
    const err = lineSplitter(push)
    proc.stdout?.setEncoding?.('utf8')
    proc.stderr?.setEncoding?.('utf8')
    proc.stdout?.on('data', out.feed)
    proc.stderr?.on('data', err.feed)

    const code = await new Promise<number | null | Error>((resolve) => {
      proc.once('error', (e) => resolve(e))
      proc.once('close', (c) => resolve(c))
    })
    out.flush()
    err.flush()

    const written = existsSync(tmpPath)
    if (code === 0 && written && !run.cancelled) {
      try {
        await fs.link(tmpPath, outPath)
      } catch (e) {
        await fs.rm(tmpPath, { force: true }).catch(() => undefined)
        const taken = (e as NodeJS.ErrnoException).code === 'EEXIST'
        return fail(taken ? `${outName} appeared in the models folder during the conversion; choose another name.` : `Could not move the result into place: ${(e as Error).message}`, tail)
      }
      await fs.rm(tmpPath, { force: true }).catch(() => undefined)
      this.deps.onConverted()
      return { ok: true, outPath, logTail: tail }
    }
    // Only our own temporary file is deleted, never the final name.
    if (written) await fs.rm(tmpPath, { force: true }).catch(() => undefined)
    if (run.cancelled) return fail('The conversion was cancelled.', tail)
    if (code instanceof Error) return fail(`Could not start Python: ${code.message}`, tail)
    if (code === 0) return fail('The converter finished without writing the file.', tail)
    return fail(`The converter stopped with exit code ${code}. The last output lines show why (a missing Python package is the usual cause).`, tail)
  }
}
