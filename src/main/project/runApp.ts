// "Run my app": works out how to start the project and runs it with its output in a panel.
import type { ChildProcess } from 'node:child_process'
import { existsSync, promises as fs } from 'node:fs'
import { join } from 'node:path'
import type { RunEvent, RunState } from '@shared/types'
import { cleanOutput, killTree, startShell } from '../agent/commands'

/** A plain web page: opened in the browser rather than run. */
export const OPEN_PREFIX = 'open:'

/** Guesses the start command from the project's files. */
export async function detectRunCommand(root: string): Promise<string | null> {
  try {
    const pkg = JSON.parse(await fs.readFile(join(root, 'package.json'), 'utf8')) as { scripts?: Record<string, string> }
    const script = pkg.scripts?.start ? 'npm start' : pkg.scripts?.dev ? 'npm run dev' : null
    if (script) return existsSync(join(root, 'node_modules')) ? script : `npm install; if ($LASTEXITCODE -eq 0) { ${script} }`
  } catch {
    // no package.json
  }
  for (const entry of ['main.py', 'app.py', 'gui.py', 'run.py']) {
    if (!existsSync(join(root, entry))) continue
    const venv = ['.venv', 'venv'].find((v) => existsSync(join(root, v, 'Scripts', 'python.exe')))
    return venv ? `& .\\${venv}\\Scripts\\python.exe ${entry}` : `python ${entry}`
  }
  for (const page of ['index.html', 'public/index.html', 'src/index.html']) {
    if (existsSync(join(root, page))) return `${OPEN_PREFIX}${page}`
  }
  return null
}

const LOCAL_URL = /https?:\/\/(?:localhost|127\.0\.0\.1|\[::1\]|0\.0\.0\.0):\d+[^\s'"<>)]*/i

export function findLocalUrl(text: string): string | undefined {
  const m = text.match(LOCAL_URL)
  return m ? m[0].replace('0.0.0.0', 'localhost').replace(/[.,;]+$/, '') : undefined
}

const MAX_OUTPUT = 100_000

export class AppRunner {
  private child: ChildProcess | null = null
  private state: RunState = { running: false, command: null, source: 'none', output: '' }

  constructor(
    private readonly root: string,
    private readonly emit: (e: RunEvent) => void,
    private readonly openFile: (abs: string) => Promise<void>
  ) {}

  get(): RunState {
    return { ...this.state }
  }

  private set(patch: Partial<RunState>): void {
    this.state = { ...this.state, ...patch }
    this.emit({ type: 'state', projectPath: this.root, state: this.get() })
  }

  async start(saved: string | null): Promise<RunState> {
    if (this.child) return this.get()
    const detected = saved ? null : await detectRunCommand(this.root)
    const command = saved ?? detected
    const source = saved ? 'saved' : detected ? 'detected' : 'none'
    if (!command) {
      this.set({ command: null, source, running: false, unknown: true })
      return this.get()
    }
    if (command.startsWith(OPEN_PREFIX)) {
      const page = command.slice(OPEN_PREFIX.length)
      await this.openFile(join(this.root, page))
      this.set({ command, source, running: false, opened: page, output: '', exitCode: undefined, url: undefined, unknown: false })
      return this.get()
    }
    this.set({ command, source, running: true, output: '', exitCode: undefined, url: undefined, opened: undefined, unknown: false, stoppedByOwner: false })
    const child = startShell(command, this.root)
    this.child = child
    const decoder = new TextDecoder()
    const take = (buf: Buffer): void => {
      const chunk = cleanOutput(decoder.decode(buf, { stream: true }))
      if (!chunk) return
      let output = this.state.output + chunk
      if (output.length > MAX_OUTPUT) output = output.slice(-MAX_OUTPUT)
      this.state.output = output
      this.emit({ type: 'output', projectPath: this.root, chunk })
      if (!this.state.url) {
        const url = findLocalUrl(chunk)
        if (url) this.set({ url })
      }
    }
    child.stdout?.on('data', take)
    child.stderr?.on('data', take)
    child.on('error', (e) => {
      this.state.output += `\n${e.message}`
    })
    child.on('close', (code) => {
      if (this.child === child) this.child = null
      this.set({ running: false, exitCode: code })
    })
    return this.get()
  }

  async stop(): Promise<RunState> {
    if (this.child) {
      this.state.stoppedByOwner = true
      await killTree(this.child)
    }
    return this.get()
  }

  forgetCommand(): void {
    this.set({ command: null, source: 'none', unknown: false })
  }
}
