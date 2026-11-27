// Audit fixes for restore points, project switching and shutdown (B4, B5, B8 helper, B9, B18, B19, B22).
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync } from 'node:fs'
import { promises as fs } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { DEFAULT_SETTINGS } from '@shared/defaults'
import type { AgentEvent, AppSettings, TaskRecord } from '@shared/types'
import { ProjectSession, type SessionDeps } from '../src/main/agent/session'
import { ProjectSwitcher, TransitionQueue } from '../src/main/project/projectSwitch'
import { CopySnapshots, type Snapshotter } from '../src/main/project/snapshots'
import { emptyProject, ProjectStore } from '../src/main/project/store'
import { RulzClient } from '../src/main/rulz/client'
import { ToolServers } from '../src/main/mcp/toolServers'
import { startMockRulz } from './helpers/mockRulz'

// A base folder with a space in its name, like the real ones (C:\Users\Some One\...).
const base = mkdtempSync(join(tmpdir(), 'rulz audit lifecycle '))
afterAll(() => rmSync(base, { recursive: true, force: true }))
let counter = 0

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))
const gate = (): { promise: Promise<void>; open: () => void } => {
  let open!: () => void
  const promise = new Promise<void>((r) => (open = r))
  return { promise, open }
}

interface Rig {
  session: ProjectSession
  root: string
  store: ProjectStore
  trash: (abs: string) => Promise<void>
  roots: { kind: string; root: unknown }[]
  events: AgentEvent[]
}

/** A session over a real folder with a copy-based restore point store, and a snapshotter that records what it was asked. */
function rig(over: Partial<SessionDeps> = {}, settings: Partial<AppSettings> = {}): Rig {
  const n = ++counter
  const root = join(base, `project ${n}`)
  mkdirSync(root, { recursive: true })
  const dataDir = join(base, `data ${n}`)
  const store = new ProjectStore(dataDir)
  const trash = async (abs: string): Promise<void> => {
    mkdirSync(join(dataDir, 'trash'), { recursive: true })
    await fs.rename(abs, join(dataDir, 'trash', `${Date.now()}-${basename(abs)}`))
  }
  const roots: Rig['roots'] = []
  const events: AgentEvent[] = []
  const full: AppSettings = { ...DEFAULT_SETTINGS, firstRunDone: true, model: 'Qwen3-Coder-30B-A3B-Instruct-Q5_K_M', ...settings }
  const session = new ProjectSession(emptyProject(root), {
    client: new RulzClient(() => ({ url: 'http://127.0.0.1:1', apiKey: '' })),
    settings: () => full,
    store,
    emit: (e) => events.push(e),
    activity: () => undefined,
    models: () => [{ id: full.model, loaded: true, loadedContext: 65536 }],
    refreshModels: () => undefined,
    snapshotter: async (kind, r) => {
      roots.push({ kind, root: r })
      return new CopySnapshots(store.backupDir(r), r, trash)
    },
    trash,
    ...over
  })
  return { session, root, store, trash, roots, events }
}

/** A finished task that changed `rel` from `before` to `after`, with a real restore point. */
async function finishedTask(r: Rig, id: string, rel: string, before: string, after: string, request = `request ${id}`): Promise<TaskRecord> {
  const cs = new CopySnapshots(r.store.backupDir(r.root), r.root, r.trash)
  writeFileSync(join(r.root, rel), before)
  const snapshot = await cs.begin(id)
  await cs.beforeChange(snapshot, rel)
  writeFileSync(join(r.root, rel), after)
  const task: TaskRecord = { id, title: request, startedAt: Date.now(), endedAt: Date.now(), outcome: 'done', steps: [], files: [{ path: rel, change: 'changed' }], snapshot }
  r.session.data.tasks.push(task)
  r.session.data.timeline.push({ id: `item-${id}`, kind: 'user', text: request, taskId: id })
  r.session.data.messages.push({ role: 'user', content: request, task: id })
  return task
}

describe('B4: Undo, Redo and diff use the calling session\'s own root', () => {
  it('asks the snapshotter for the session\'s root, even when stopping the commands takes a while', async () => {
    const r = rig()
    await finishedTask(r, 't1', 'a.txt', 'old', 'new')
    const g = gate()
    r.session.background.stopAll = () => g.promise
    const undoing = r.session.undo()
    await sleep(30)
    expect(r.roots).toEqual([]) // still waiting for the commands to stop
    g.open()
    await undoing
    expect(r.roots).toEqual([{ kind: 'copy', root: r.root }])
    expect(readFileSync(join(r.root, 'a.txt'), 'utf8')).toBe('old')
  })

  it('refuses to close while an undo is in progress, and says why', async () => {
    const r = rig()
    await finishedTask(r, 't1', 'a.txt', 'old', 'new')
    const g = gate()
    r.session.background.stopAll = () => g.promise
    const undoing = r.session.undo()
    await sleep(10)
    expect(r.session.busyWith).toBe('the undo')
    const switcher = new ProjectSwitcher<ProjectSession, { stop(): Promise<void> }, string>({
      current: () => ({ session: r.session, runner: null }),
      release: () => undefined,
      guard: () => undefined,
      dropUploads: () => undefined,
      already: () => null,
      create: async () => 'x'
    })
    await expect(switcher.close()).rejects.toThrow('Wait for the undo to finish first.')
    g.open()
    await undoing
    expect(r.session.busyWith).toBeNull()
  })

  it('stops the app inside the reservation: a second operation meanwhile is refused', async () => {
    const r = rig()
    await finishedTask(r, 't1', 'a.txt', 'old', 'new')
    const g = gate()
    const undoing = r.session.undo(undefined, () => g.promise)
    expect(r.session.busyWith).toBe('the undo')
    expect(() => r.session.start('another task')).toThrow('Wait for the undo to finish first.')
    await expect(r.session.undo()).rejects.toThrow('Wait for the undo to finish first.')
    g.open()
    await undoing
  })

  it('refuses undo, redo, diff and start on a closed session, and never touches the snapshotter', async () => {
    const r = rig()
    await finishedTask(r, 't1', 'a.txt', 'old', 'new')
    await r.session.close()
    expect(r.session.isClosed).toBe(true)
    await expect(r.session.undo()).rejects.toThrow(/closed/)
    await expect(r.session.redo('t1')).rejects.toThrow(/closed/)
    await expect(r.session.diff('t1', 'a.txt')).rejects.toThrow(/closed/)
    expect(() => r.session.start('hello')).toThrow(/closed/)
    expect(r.roots).toEqual([])
    expect(readFileSync(join(r.root, 'a.txt'), 'utf8')).toBe('new')
  })

  it('a copy restore started for project A while B is open writes only into A', async () => {
    const a = rig()
    const b = rig()
    await finishedTask(a, 'ta', 'same.txt', 'A before', 'A after')
    writeFileSync(join(b.root, 'same.txt'), 'B untouched')
    // The one shared snapshotter of the app (as index.ts has): it must build from the root it is given.
    const shared = async (_kind: 'git' | 'copy' | 'auto', root: string): Promise<Snapshotter> => new CopySnapshots(a.store.backupDir(root), root, a.trash)
    const s = new ProjectSession(a.session.data, { ...(a.session as unknown as { deps: SessionDeps }).deps, snapshotter: shared })
    const g = gate()
    s.background.stopAll = () => g.promise
    const undoing = s.undo()
    // The project is "switched" meanwhile: B is the open project now. A's undo must still go to A.
    await sleep(20)
    g.open()
    await undoing
    expect(readFileSync(join(a.root, 'same.txt'), 'utf8')).toBe('A before')
    expect(readFileSync(join(b.root, 'same.txt'), 'utf8')).toBe('B untouched')
  })
})

describe('B22: Redo keeps the request when it cannot start', () => {
  it('checks the start before undoing or dropping anything', async () => {
    const r = rig()
    const task = await finishedTask(r, 't1', 'a.txt', 'old', 'new', 'make a page')
    const pdf = { id: '0123456789abcdef', name: 'a.pdf', pages: 1, bytes: 10 }
    const item = r.session.data.timeline[0]
    if (item.kind === 'user') item.attachments = [pdf]
    ;(r.session as unknown as { removingPdfs: Set<string> }).removingPdfs.add(pdf.id)
    await expect(r.session.redo('t1')).rejects.toThrow(/being removed/)
    // Nothing was undone and the request is still there.
    expect(readFileSync(join(r.root, 'a.txt'), 'utf8')).toBe('new')
    expect(r.session.data.tasks).toHaveLength(1)
    expect(task.undone).toBeUndefined()
    expect(r.session.data.timeline.some((i) => i.kind === 'user' && i.text === 'make a page')).toBe(true)
    expect(r.session.data.messages.some((m) => m.content === 'make a page')).toBe(true)
    expect(r.session.busyWith).toBeNull()
  })

  it('keeps the reservation until the new task has started', async () => {
    const r = rig()
    await finishedTask(r, 't1', 'a.txt', 'old', 'new', 'make a page')
    let busyAtStart: string | null | undefined
    r.events.length = 0
    const session = r.session
    const emit = (session as unknown as { deps: SessionDeps }).deps
    const orig = emit.emit
    emit.emit = (e) => {
      if (e.type === 'task-start') busyAtStart = session.busyWith
      orig(e)
    }
    await session.redo('t1')
    expect(busyAtStart).toBe('the redo')
    expect(readFileSync(join(r.root, 'a.txt'), 'utf8')).toBe('old')
    await session.close()
  })
})

describe('B19: a failing listener in the end of a task does not leave it running', () => {
  it('running is false after the task ended even if emit throws', async () => {
    const mock = await startMockRulz({ turns: [{ content: 'hello', delay: 2000 }] })
    try {
      const r = rig({
        client: new RulzClient(() => ({ url: mock.url, apiKey: '' })),
        emit: () => undefined
      })
      const deps = (r.session as unknown as { deps: SessionDeps }).deps
      deps.emit = (e) => {
        r.events.push(e)
        if (e.type === 'item' && e.item.kind === 'notice' && e.item.text.startsWith('Not sent')) throw new Error('listener failed')
      }
      r.session.start('say hello')
      await sleep(300)
      r.session.addNote('one more thing')
      r.session.stop()
      const deadline = Date.now() + 8000
      while (r.session.isRunning && Date.now() < deadline) await sleep(25)
      expect(r.session.isRunning).toBe(false)
      expect(() => r.session.start('again')).not.toThrow(/already running/)
      await sleep(50)
      r.session.stop()
      await r.session.close()
    } finally {
      await mock.close()
    }
  }, 30_000)
})

describe('B18: Stop does not wait for the practice grading', () => {
  it('a stopped task does not wait for the grade request', async () => {
    const mock = await startMockRulz({ turns: [{ content: 'ANSWER: 1' }] })
    try {
      const never = gate()
      const client = new RulzClient(() => ({ url: mock.url, apiKey: '' }))
      client.gradePractice = () => never.promise as never
      const r = rig({ client })
      const done = new Promise<void>((resolve) => {
        ;(r.session as unknown as { deps: SessionDeps }).deps.emit = (e) => {
          r.events.push(e)
          if (e.type === 'task-end') resolve()
        }
      })
      r.session.start('practice', { re: { kind: 'practice', target: 't', practice: { taskId: 'p1', family: 'f', tier: 1 } } as never })
      const deadline = Date.now() + 10_000
      // Wait for the task to reach the grading step (its reply is in), then press Stop.
      while (Date.now() < deadline && !r.session.data.messages.some((m) => m.role === 'assistant')) await sleep(25)
      await sleep(100)
      const t0 = Date.now()
      r.session.stop()
      await Promise.race([done, sleep(5000)])
      expect(Date.now() - t0).toBeLessThan(3000)
      expect(r.session.isRunning).toBe(false)
      await r.session.close()
    } finally {
      await mock.close()
    }
  }, 30_000)
})

describe('B5: project switches run one at a time', () => {
  interface Fake {
    root: string
    isRunning: boolean
    busyWith: string | null
    closes: number
    marked: boolean
    markClosed(): void
    close(): Promise<void>
  }
  const fake = (root: string): Fake => {
    const f: Fake = {
      root,
      isRunning: false,
      busyWith: null,
      closes: 0,
      marked: false,
      markClosed: () => {
        f.marked = true
      },
      close: async () => {
        f.closes++
        await sleep(20)
      }
    }
    return f
  }
  function build(over: { create?: (abs: string) => Promise<string>; cleanupWaitMs?: number; runnerStop?: () => Promise<void> } = {}) {
    const state: { session: Fake | null; runner: { stop(): Promise<void> } | null } = { session: null, runner: null }
    const all: Fake[] = []
    const switcher = new ProjectSwitcher<Fake, { stop(): Promise<void> }, string>({
      current: () => state,
      release: () => {
        state.session = null
        state.runner = null
      },
      guard: () => undefined,
      dropUploads: () => undefined,
      already: (abs) => (state.session?.root === abs ? `state of ${abs}` : null),
      create:
        over.create ??
        (async (abs) => {
          await sleep(30)
          const s = fake(abs)
          all.push(s)
          state.session = s
          state.runner = { stop: over.runnerStop ?? (async () => undefined) }
          return `state of ${abs}`
        }),
      cleanupWaitMs: over.cleanupWaitMs
    })
    return { switcher, state, all }
  }

  it('two opens at once leave exactly one live session and close the other', async () => {
    const { switcher, state, all } = build()
    const [a, b] = await Promise.all([switcher.open('A'), switcher.open('B')])
    expect([a, b]).toEqual(['state of A', 'state of B'])
    expect(all.map((s) => [s.root, s.closes])).toEqual([
      ['A', 1],
      ['B', 0]
    ])
    expect(state.session?.root).toBe('B')
  })

  it('opening the same project twice at once creates one session', async () => {
    const { switcher, all } = build()
    const [a, b] = await Promise.all([switcher.open('A'), switcher.open('A')])
    expect(a).toBe('state of A')
    expect(b).toBe('state of A')
    expect(all).toHaveLength(1)
    expect(all[0].closes).toBe(0)
  })

  it('a rejected transition does not block the next one', async () => {
    let fail = true
    const { switcher, state } = build({
      create: async (abs) => {
        if (fail) {
          fail = false
          throw new Error('cannot load')
        }
        state.session = fake(abs)
        return `state of ${abs}`
      }
    })
    const first = switcher.open('A')
    const second = switcher.open('B')
    await expect(first).rejects.toThrow('cannot load')
    await expect(second).resolves.toBe('state of B')
    expect(state.session?.root).toBe('B')
  })

  it('refuses to close while a task runs', async () => {
    const { switcher, state } = build()
    await switcher.open('A')
    state.session!.isRunning = true
    await expect(switcher.open('B')).rejects.toThrow('A task is still running. Press Stop first.')
    state.session!.isRunning = false
    await expect(switcher.open('B')).resolves.toBe('state of B')
  })

  it('does not open a replacement while the old app cannot be stopped', async () => {
    const { switcher, all } = build({ runnerStop: () => new Promise(() => undefined), cleanupWaitMs: 50 })
    await switcher.open('A')
    await expect(switcher.open('B')).rejects.toThrow(/did not stop in time/)
    expect(all.map((s) => s.root)).toEqual(['A'])
  })

  it('opens nothing while the last cleanup that ran out of time still runs, and opens once it is done', async () => {
    let finishStop: () => void = () => undefined
    const { switcher, all, state } = build({ runnerStop: () => new Promise<void>((r) => (finishStop = r)), cleanupWaitMs: 50 })
    await switcher.open('A')
    await expect(switcher.open('B')).rejects.toThrow(/did not stop in time/)
    // Refuses everything at once, though its app has not stopped yet; quitting would wait for the cleanup.
    expect(all[0].marked).toBe(true)
    let cleaned = false
    void switcher.cleanupDone().then(() => (cleaned = true))
    // The same project again too: its old session may still be saving.
    await expect(switcher.open('A')).rejects.toThrow(/still closing/)
    expect(all.map((s) => s.root)).toEqual(['A'])
    expect(cleaned).toBe(false)
    finishStop()
    await expect(switcher.open('B')).resolves.toBe('state of B')
    expect(cleaned).toBe(true)
    expect(all.map((s) => [s.root, s.closes])).toEqual([
      ['A', 1],
      ['B', 0]
    ])
    expect(state.session?.root).toBe('B')
  })

  it('the queue runs in order and recovers from a failure', async () => {
    const q = new TransitionQueue()
    const order: string[] = []
    const a = q.run(async () => {
      await sleep(30)
      order.push('a')
      throw new Error('a failed')
    })
    const b = q.run(async () => {
      order.push('b')
      return 2
    })
    await expect(a).rejects.toThrow('a failed')
    await expect(b).resolves.toBe(2)
    expect(order).toEqual(['a', 'b'])
  })
})

describe('B9: a tool server that connects after dispose is closed at once', () => {
  it('shuts the late client (and a stdio server process) down', async () => {
    const dir = mkdtempSync(join(base, 'mcp '))
    const config = join(dir, 'mcp.json')
    const marker = join(dir, 'stdin closed.txt')
    // A server that is slow to answer the handshake, and leaves a marker when its input is closed.
    const script = join(dir, 'slow server.mjs')
    writeFileSync(
      script,
      [
        "import { createInterface } from 'node:readline'",
        "import { writeFileSync } from 'node:fs'",
        'const rl = createInterface({ input: process.stdin })',
        'process.stdin.on("close", () => { writeFileSync(process.argv[2], "closed"); process.exit(0) })',
        'rl.on("line", (line) => {',
        '  let msg; try { msg = JSON.parse(line) } catch { return }',
        '  if (msg.method === "initialize") setTimeout(() => process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: msg.id, result: { protocolVersion: msg.params.protocolVersion, capabilities: { tools: {} }, serverInfo: { name: "slow", version: "1" } } }) + "\\n"), 400)',
        '})'
      ].join(String.fromCharCode(10))
    )
    writeFileSync(config, JSON.stringify({ mcpServers: { slow: { command: process.execPath, args: [script, marker] } } }))
    const servers = new ToolServers(config)
    const starting = servers.start()
    await sleep(100)
    await servers.dispose()
    await starting
    const deadline = Date.now() + 5000
    while (!existsSync(marker) && Date.now() < deadline) await sleep(50)
    // The late connection was shut down, not kept: its process saw its input close.
    expect(existsSync(marker)).toBe(true)
    expect(servers.tools()).toEqual([])
  }, 20_000)
})
