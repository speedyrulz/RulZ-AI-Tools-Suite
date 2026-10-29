import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { promises as fs } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { DEFAULT_SETTINGS } from '@shared/defaults'
import type { AgentEvent, AppSettings, PendingPrompt, PromptAnswer, TaskRecord } from '@shared/types'
import type { PageTarget } from '../src/main/agent/pageCheck'
import { ProjectSession, type SessionDeps } from '../src/main/agent/session'
import { CopySnapshots, GitSnapshots, gitInstalled } from '../src/main/project/snapshots'
import { emptyProject, ProjectStore } from '../src/main/project/store'
import { RulzClient } from '../src/main/rulz/client'
import { startMockRulz, type MockOptions, type MockTurn } from './helpers/mockRulz'
import { SHARE_SHOWN, SHARE_TASK } from '../src/main/agent/prompt'
import { ToolServers } from '../src/main/mcp/toolServers'
import { fakeToolConfig } from './helpers/fakeTools'

const base = mkdtempSync(join(tmpdir(), 'rulz-agent-session-'))
afterAll(() => rmSync(base, { recursive: true, force: true }))
let counter = 0

interface Harness {
  session: ProjectSession
  root: string
  events: AgentEvent[]
  prompts: PendingPrompt[]
  requests: Awaited<ReturnType<typeof startMockRulz>>['requests']
  run: (text: string, opts?: { shown?: string; share?: boolean }) => Promise<TaskRecord>
  close: () => Promise<void>
}

async function harness(
  turns: MockTurn[],
  opts: {
    settings?: Partial<AppSettings>
    answer?: (p: PendingPrompt) => PromptAnswer | null
    nCtx?: number
    useGit?: boolean
    checkPage?: SessionDeps['checkPage']
    timings?: MockOptions['timings']
    toolServers?: SessionDeps['toolServers']
    /** Sees every event as it happens (to act mid-task, like the owner would). */
    onEvent?: (e: AgentEvent, session: ProjectSession) => void
  } = {}
): Promise<Harness> {
  const mock = await startMockRulz({ turns, nCtx: opts.nCtx, timings: opts.timings })
  const root = join(base, `project-${++counter}`)
  mkdirSync(root, { recursive: true })
  const dataDir = join(base, `data-${counter}`)
  const store = new ProjectStore(dataDir)
  const settings: AppSettings = { ...DEFAULT_SETTINGS, firstRunDone: true, model: 'Qwen3-Coder-30B-A3B-Instruct-Q5_K_M', ...opts.settings }
  const events: AgentEvent[] = []
  const prompts: PendingPrompt[] = []
  const trash = async (abs: string): Promise<void> => {
    const bin = join(dataDir, 'trash', `${Date.now()}-${basename(abs)}`)
    mkdirSync(join(dataDir, 'trash'), { recursive: true })
    await fs.rename(abs, bin)
  }
  const useGit = opts.useGit ?? (await gitInstalled())
  let session!: ProjectSession
  const ended: ((t: TaskRecord) => void)[] = []
  session = new ProjectSession(emptyProject(root), {
    client: new RulzClient(() => ({ url: mock.url, apiKey: '' })),
    settings: () => settings,
    store,
    emit: (e) => {
      events.push(e)
      opts.onEvent?.(e, session)
      if (e.type === 'prompt' && e.prompt) {
        prompts.push(e.prompt)
        const ans = opts.answer?.(e.prompt) ?? (e.prompt.kind === 'approval' ? { kind: 'approval', decision: 'allow' } : null)
        if (ans) setTimeout(() => session.answer(e.prompt!.id, ans), 10)
      }
      if (e.type === 'task-end') ended.shift()?.(e.task)
    },
    activity: () => undefined,
    models: () => [{ id: settings.model, loaded: true, loadedContext: 65536 }],
    refreshModels: () => undefined,
    snapshotter: async (kind) =>
      (kind === 'git' || (kind === 'auto' && useGit)) ? new GitSnapshots(store.snapshotDir(root), root) : new CopySnapshots(store.backupDir(root), root, trash),
    trash,
    checkPage: opts.checkPage,
    toolServers: opts.toolServers
  })
  return {
    session,
    root,
    events,
    prompts,
    requests: mock.requests,
    run: (text, startOpts) =>
      new Promise((resolve) => {
        ended.push(resolve)
        session.start(text, startOpts)
      }),
    close: async () => {
      await session.close()
      await mock.close()
    }
  }
}

const call = (name: string, args: unknown) => ({ name, args })

describe('the agent loop', () => {
  it('builds, asks before commands, finishes and can be undone', async () => {
    const h = await harness([
      { content: "I'll create the page.", toolCalls: [call('write_file', { path: 'index.html', content: '<h1>Hi</h1>\n', summary: 'Creating the web page' })] },
      { content: "Now I'll check it.", toolCalls: [call('run_command', { command: 'Get-Content index.html', summary: 'Checking the page' })] },
      { content: 'Done! **How to try it:** click Run my app.' }
    ])
    try {
      writeFileSync(join(h.root, 'keep.txt'), 'original')
      const task = await h.run('Make a web page')
      expect(task.outcome).toBe('done')
      expect(readFileSync(join(h.root, 'index.html'), 'utf8')).toBe('<h1>Hi</h1>\n')
      // Balanced: no question for the file, one for the command.
      expect(h.prompts.map((p) => p.kind)).toEqual(['approval'])
      expect(h.prompts[0]).toMatchObject({ action: 'Checking the page', detail: 'Get-Content index.html' })
      expect(task.steps.map((s) => [s.title, s.status])).toEqual([
        ['Creating the web page', 'done'],
        ['Checking the page', 'done']
      ])
      expect(task.steps[1].output).toContain('<h1>Hi</h1>')
      expect(task.files).toEqual([{ path: 'index.html', change: 'created' }])
      // Tool results went back to the model, paired with their calls.
      const last = h.requests[2].body.messages!
      const results = last.filter((m) => m.role === 'tool')
      expect(results).toHaveLength(2)
      expect(results[1].content).toMatch(/^Exit code: 0/)
      expect(last[0].role).toBe('system')
      expect(h.requests[0].body.tools).toHaveLength(13)
      expect(h.events.some((e) => e.type === 'delta')).toBe(true)

      await h.session.undo()
      expect(existsSync(join(h.root, 'index.html'))).toBe(false)
      expect(readFileSync(join(h.root, 'keep.txt'), 'utf8')).toBe('original')
      expect(h.session.data.notes[0]).toMatch(/Undo last task/)
      expect(h.session.canUndo()).toBeUndefined()
    } finally {
      await h.close()
    }
  }, 60_000)

  it("keeps the model's thinking with its reply, out of the answer and the history", async () => {
    const h = await harness([
      { reasoning: 'The owner wants a page. I should write index.html.', toolCalls: [call('write_file', { path: 'index.html', content: 'hi\n', summary: 'Creating the page' })] },
      { content: '<think>Check it worked.</think>Your page is ready.' }
    ])
    try {
      await h.run('Make a page')
      const replies = h.session.data.timeline.filter((i) => i.kind === 'assistant')
      expect(replies).toMatchObject([
        { text: '', reasoning: 'The owner wants a page. I should write index.html.' },
        { text: 'Your page is ready.', reasoning: 'Check it worked.' }
      ])
      const streamed = h.events.flatMap((e) => (e.type === 'reasoning' ? [e.text] : [])).join('')
      expect(streamed).toBe('The owner wants a page. I should write index.html.')
      // The thinking is never sent back to the model.
      expect(JSON.stringify(h.requests[1].body.messages)).not.toContain('I should write')
    } finally {
      await h.close()
    }
  }, 60_000)

  it('respects "Don\'t allow" and explains it to the model', async () => {
    const h = await harness(
      [{ toolCalls: [call('run_command', { command: 'npm install', summary: 'Installing' })] }, { content: 'OK, skipped.' }],
      { answer: () => ({ kind: 'approval', decision: 'deny' }) }
    )
    try {
      const task = await h.run('Install things')
      expect(task.outcome).toBe('done')
      expect(task.steps[0]).toMatchObject({ status: 'skipped', note: "You didn't allow this." })
      const tool = h.requests[1].body.messages!.find((m) => m.role === 'tool')!
      expect(tool.content).toMatch(/did not allow/)
    } finally {
      await h.close()
    }
  }, 30_000)

  it('refuses dangerous commands without asking', async () => {
    const h = await harness([{ toolCalls: [call('run_command', { command: 'shutdown /r /t 0', summary: 'Restarting' })] }, { content: 'I cannot do that.' }])
    try {
      const task = await h.run('Restart my PC')
      expect(h.prompts).toHaveLength(0)
      expect(task.steps[0].status).toBe('refused')
      const tool = h.requests[1].body.messages!.find((m) => m.role === 'tool')!
      expect(tool.content).toMatch(/^Refused: It would shut down/)
    } finally {
      await h.close()
    }
  }, 30_000)

  it('asks before file changes in Careful mode, and "Allow for this task" stops the questions', async () => {
    const h = await harness(
      [
        { toolCalls: [call('write_file', { path: 'a.txt', content: 'a' })] },
        { toolCalls: [call('write_file', { path: 'b.txt', content: 'b' })] },
        { content: 'Done.' }
      ],
      { settings: { approval: 'careful' }, answer: () => ({ kind: 'approval', decision: 'allow-task' }) }
    )
    try {
      await h.run('Make two files')
      expect(h.prompts).toHaveLength(1)
      expect(existsSync(join(h.root, 'b.txt'))).toBe(true)
    } finally {
      await h.close()
    }
  }, 30_000)

  it('pauses when going round in circles, and stops when asked', async () => {
    const same = { toolCalls: [call('read_file', { path: 'missing.txt' })] }
    const h = await harness([same, same, same, { content: 'never reached' }], { answer: (p) => (p.kind === 'stuck' ? { kind: 'stuck', decision: 'stop' } : null) })
    try {
      const task = await h.run('Read it')
      expect(h.prompts.map((p) => p.kind)).toEqual(['stuck'])
      expect(h.prompts[0]).toMatchObject({ reason: 'repeating' })
      expect(task.outcome).toBe('stopped')
      expect(h.requests).toHaveLength(3)
    } finally {
      await h.close()
    }
  }, 30_000)

  it('passes the owner\'s new idea on when they choose "Try another way"', async () => {
    const same = { toolCalls: [call('read_file', { path: 'missing.txt' })] }
    const h = await harness([same, same, same, { content: 'Trying the other way.' }], {
      answer: (p) => (p.kind === 'stuck' ? { kind: 'stuck', decision: 'change', text: 'Create the file first' } : null)
    })
    try {
      const task = await h.run('Read it')
      expect(task.outcome).toBe('done')
      const msgs = h.requests[3].body.messages!
      expect(msgs[msgs.length - 1]).toMatchObject({ role: 'user' })
      expect(msgs[msgs.length - 1].content).toMatch(/Create the file first/)
    } finally {
      await h.close()
    }
  }, 30_000)

  it('pauses at the step limit', async () => {
    const turns = Array.from({ length: 6 }, (_, i) => ({ toolCalls: [call('list_files', { depth: (i % 4) + 1 })] }))
    const h = await harness(turns, { settings: { maxSteps: 5 }, answer: (p) => (p.kind === 'stuck' ? { kind: 'stuck', decision: 'stop' } : null) })
    try {
      const task = await h.run('Look around')
      expect(h.prompts[0]).toMatchObject({ kind: 'stuck', reason: 'limit' })
      expect(task.outcome).toBe('stopped')
      expect(task.steps).toHaveLength(5)
    } finally {
      await h.close()
    }
  }, 30_000)

  it('stops a running command and keeps the conversation valid', async () => {
    const h = await harness([{ toolCalls: [call('run_command', { command: 'Start-Sleep -Seconds 60', summary: 'Waiting' })] }], { settings: { approval: 'handsoff' } })
    try {
      const done = h.run('Wait a minute')
      await new Promise((r) => setTimeout(r, 2000))
      const started = Date.now()
      h.session.stop()
      const task = await done
      expect(Date.now() - started).toBeLessThan(10_000)
      expect(task.outcome).toBe('stopped')
      const msgs = h.session.data.messages
      expect(msgs[msgs.length - 1]).toMatchObject({ role: 'tool' })
    } finally {
      await h.close()
    }
  }, 30_000)

  it('stops while the model is still writing', async () => {
    const h = await harness([{ content: 'slow', delay: 5000 }])
    try {
      const done = h.run('Hello')
      await new Promise((r) => setTimeout(r, 300))
      h.session.stop()
      expect((await done).outcome).toBe('stopped')
    } finally {
      await h.close()
    }
  }, 30_000)

  it('asks the owner a question and passes the answer back', async () => {
    const h = await harness([{ toolCalls: [call('ask_user', { question: 'Blue or red?' })] }, { content: 'Red it is.' }], {
      answer: (p) => (p.kind === 'question' ? { kind: 'question', text: 'Red please' } : null)
    })
    try {
      await h.run('Pick a colour')
      const tool = h.requests[1].body.messages!.find((m) => m.role === 'tool')!
      expect(tool.content).toBe('The owner answered: Red please')
      expect(h.session.data.timeline.map((i) => i.kind === 'notice' ? 'notice' : `${i.kind}:${i.text}`)).toEqual([
        'user:Pick a colour',
        'assistant:Blue or red?',
        'user:Red please',
        'assistant:Red it is.'
      ])
    } finally {
      await h.close()
    }
  }, 30_000)

  it('summarises older work when the context gets full', async () => {
    const bigOutput = (n: number) => ({ toolCalls: [call('run_command', { command: `1..${n} | ForEach-Object { 'output line number ' + $_ }` })] })
    const h = await harness([bigOutput(400), bigOutput(401), bigOutput(402), { content: 'first done' }, { content: 'second done' }], {
      nCtx: 4000,
      settings: { approval: 'handsoff' }
    })
    try {
      await h.run('First task')
      await h.run('Second task')
      expect(h.requests.some((r) => r.body.stream === false)).toBe(true)
      expect(h.session.data.summary).toMatch(/Summary/)
      const last = h.requests[h.requests.length - 1].body.messages!
      expect(last[0].content).toMatch(/Earlier work in this project/)
      expect(last.some((m) => m.role === 'user' && m.content.includes('Second task'))).toBe(true)
    } finally {
      await h.close()
    }
  }, 60_000)

  it('checks pages in the project and reports their errors to the model', async () => {
    const opened: PageTarget[] = []
    const h = await harness(
      [
        { content: "I'll look at the page.", toolCalls: [call('check_page', { page: 'index.html', summary: 'Looking at your web page' })] },
        { toolCalls: [call('check_page', { page: 'https://example.com' })] },
        { content: 'Found the problem.' }
      ],
      {
        checkPage: async (target) => {
          opened.push(target)
          return {
            url: target.url,
            title: 'Hi',
            messages: [{ level: 'error', message: 'Uncaught ReferenceError: todos is not defined', source: target.url, line: 3 }],
            failedRequests: [],
            text: 'Hi',
            controls: [],
            visuals: 0
          }
        }
      }
    )
    try {
      writeFileSync(join(h.root, 'index.html'), '<h1>Hi</h1>')
      const task = await h.run('Why is my page broken?')
      // Read-only: no approval needed, and the internet is refused before anything opens.
      expect(h.prompts).toEqual([])
      expect(opened).toHaveLength(1)
      expect(opened[0].url).toMatch(/^file:.*index\.html$/)
      expect(task.steps.map((s) => [s.title, s.status, s.detail])).toEqual([
        ['Looking at your web page', 'error', 'index.html'],
        ['Looking at the page', 'error', 'https://example.com']
      ])
      const results = h.requests[2].body.messages!.filter((m) => m.role === 'tool')
      expect(results[0].content).toContain('todos is not defined (index.html:3)')
      expect(results[1].content).toMatch(/on the internet/)
    } finally {
      await h.close()
    }
  }, 30_000)

  it('shows the short "Fix this" request while the model gets the details', async () => {
    const h = await harness([{ content: 'Fixed.' }])
    try {
      await h.run('I clicked Run my app and it crashed.\n\nWhat it showed: Error: boom', { shown: 'Please fix the error my app showed when I ran it.' })
      expect(h.session.data.tasks[0].title).toBe('Please fix the error my app showed when I ran it.')
      expect(h.session.data.timeline[0]).toMatchObject({ kind: 'user', text: 'Please fix the error my app showed when I ran it.' })
      const user = h.requests[0].body.messages!.find((m) => m.role === 'user')!
      expect(user.content).toContain('Error: boom')
    } finally {
      await h.close()
    }
  }, 30_000)

  it('undoes with copied files when git is not available', async () => {
    const h = await harness(
      [{ toolCalls: [call('write_file', { path: 'new.txt', content: 'new' }), call('edit_file', { path: 'old.txt', old_text: 'one', new_text: 'two' })] }, { content: 'ok' }],
      { useGit: false }
    )
    try {
      writeFileSync(join(h.root, 'old.txt'), 'one')
      const task = await h.run('Change things')
      expect(task.snapshot?.kind).toBe('copy')
      expect(task.files).toEqual([
        { path: 'new.txt', change: 'created' },
        { path: 'old.txt', change: 'changed' }
      ])
      await h.session.undo()
      expect(existsSync(join(h.root, 'new.txt'))).toBe(false)
      expect(readFileSync(join(h.root, 'old.txt'), 'utf8')).toBe('one')
    } finally {
      await h.close()
    }
  }, 30_000)

  for (const useGit of [true, false]) {
    it(`goes back to before an earlier task from History (${useGit ? 'git' : 'copied files'})`, async () => {
      if (useGit && !(await gitInstalled())) return
      const h = await harness(
        [
          { toolCalls: [call('write_file', { path: 'a.txt', content: 'a' }), call('edit_file', { path: 'old.txt', old_text: 'one', new_text: 'two' })] },
          { content: 'ok' },
          { toolCalls: [call('write_file', { path: 'b.txt', content: 'b' }), call('edit_file', { path: 'old.txt', old_text: 'two', new_text: 'three' })] },
          { content: 'ok' },
          { toolCalls: [call('write_file', { path: 'c.txt', content: 'c' })] },
          { content: 'ok' }
        ],
        { useGit }
      )
      try {
        writeFileSync(join(h.root, 'old.txt'), 'one')
        const first = await h.run('Task one')
        const second = await h.run('Task two')
        const third = await h.run('Task three')
        // Undo the newest one first, then go back past the other two.
        await h.session.undo()
        expect(existsSync(join(h.root, 'c.txt'))).toBe(false)
        await expect(h.session.undo(third.id)).rejects.toThrow(/already undone/)

        await h.session.undo(second.id)
        expect(existsSync(join(h.root, 'b.txt'))).toBe(false)
        expect(readFileSync(join(h.root, 'old.txt'), 'utf8')).toBe('two')
        expect(readFileSync(join(h.root, 'a.txt'), 'utf8')).toBe('a')
        expect(h.session.canUndo()?.id).toBe(first.id)

        expect(h.session.data.tasks.map((t) => [t.title, !!t.undone])).toEqual([
          ['Task one', false],
          ['Task two', true],
          ['Task three', true]
        ])
        expect(h.session.data.notes.at(-1)).toMatch(/used History to go back.*before the task "Task two"/)
        expect(h.session.data.timeline.at(-1)).toMatchObject({ kind: 'notice', text: expect.stringMatching(/^Undone: "Task two"\. /) })
      } finally {
        await h.close()
      }
    }, 60_000)
  }

  it("fills the Memory meter from the server's exact counts, or estimates without them", async () => {
    const h = await harness([{ content: 'Hello!' }], { timings: () => ({ cache_n: 4000, prompt_n: 1000, predicted_n: 50 }) })
    try {
      await h.run('Say hello')
      expect(h.session.data.context).toEqual({ used: 5050, size: 65536, exact: true })
      expect(h.events.filter((e) => e.type === 'context')).toHaveLength(1)
      expect(h.session.state('x', 0).context).toEqual({ used: 5050, size: 65536, exact: true })
    } finally {
      await h.close()
    }
    const e = await harness([{ content: 'Hello!' }])
    try {
      await e.run('Say hello')
      expect(e.session.data.context).toMatchObject({ size: 65536, exact: false })
      expect(e.session.data.context!.used).toBeGreaterThan(1000) // system prompt and tools
    } finally {
      await e.close()
    }
  }, 30_000)

  it('uses more of the context when the estimate turns out too high', async () => {
    // The server counts half the estimated tokens, so the next request may hold about twice as much.
    let estimate = 0
    const h = await harness(
      [
        { toolCalls: [call('read_file', { path: 'big1.txt' })] },
        { toolCalls: [call('read_file', { path: 'big2.txt' })] },
        { toolCalls: [call('read_file', { path: 'big3.txt' })] },
        { content: 'Done.' }
      ],
      {
        nCtx: 32768,
        timings: (req) => {
          estimate = Math.ceil(JSON.stringify(req.body.messages).length / 3.2)
          return { cache_n: 0, prompt_n: Math.ceil(estimate / 2), predicted_n: 10 }
        }
      }
    )
    try {
      for (const n of [1, 2, 3]) writeFileSync(join(h.root, `big${n}.txt`), `file ${n} content\n`.repeat(1600))
      await h.run('Read the three files')
      const last = h.requests[h.requests.length - 1].body.messages!.filter((m) => m.role === 'tool')
      // Without the correction the older files would have been shortened to fit 80% of 32K.
      expect(last.length).toBe(3)
      expect(last.every((m) => !m.content.includes('omitted to save space'))).toBe(true)
    } finally {
      await h.close()
    }
  }, 30_000)

  it('warns once when the model has a small context', async () => {
    const h = await harness([{ content: 'One.' }, { content: 'Two.' }], { nCtx: 16384 })
    try {
      await h.run('First')
      await h.run('Second')
      const warnings = h.session.data.timeline.filter((i) => i.kind === 'notice' && /small working memory \(16K tokens\)/.test(i.text))
      expect(warnings).toHaveLength(1)
      expect(warnings[0]).toMatchObject({ tone: 'warn' })
    } finally {
      await h.close()
    }
  }, 30_000)

  it('sends the owner\'s instructions and project notes with every request, and saves the notes', async () => {
    const h = await harness([{ content: 'OK.' }], { settings: { instructions: 'Explain things briefly.' } })
    try {
      await h.session.setMemory('  Use blue, not purple.  ')
      await h.run('Make a page')
      const system = h.requests[0].body.messages![0].content
      expect(system).toContain('Explain things briefly.')
      expect(system).toContain('Use blue, not purple.')
      expect(h.session.state('x', 0).memory).toBe('Use blue, not purple.')
      const store = new ProjectStore(join(base, `data-${counter}`))
      expect((await store.load(h.root)).memory).toBe('Use blue, not purple.')
    } finally {
      await h.close()
    }
  }, 30_000)

  it('"Share my app" points the owner at the packaged file', async () => {
    const h = await harness([
      { toolCalls: [call('write_file', { path: 'dist/site.zip', content: 'pretend zip' })] },
      { content: 'Packaged it.\nSHARE: dist/site.zip' },
      { content: 'Packaged it.\nSHARE: dist/missing.zip' }
    ])
    try {
      const task = await h.run(SHARE_TASK, { shown: SHARE_SHOWN, share: true })
      expect(task.title).toBe(SHARE_SHOWN)
      expect(h.requests[0].body.messages!.find((m) => m.role === 'user')!.content).toContain('electron-builder')
      const ok = h.session.data.timeline.at(-1)
      expect(ok).toMatchObject({ kind: 'notice', tone: 'ok', reveal: join(h.root, 'dist', 'site.zip') })
      expect((ok as { text: string }).text).toMatch(/Your app is packaged: \*\*dist\/site\.zip\*\* \(1 KB\)/)

      await h.run(SHARE_TASK, { shown: SHARE_SHOWN, share: true })
      expect(h.session.data.timeline.at(-1)).toMatchObject({ kind: 'notice', tone: 'warn', text: expect.stringMatching(/couldn't find a packaged file/) })
    } finally {
      await h.close()
    }
  }, 30_000)

  describe('tools from tool servers (like REA)', () => {
    it('finds, loads and runs them, asking once per task for ordinary tools and every time for destructive ones', async () => {
      const servers = new ToolServers(fakeToolConfig(base))
      const h = await harness(
        [
          { content: "I'll look for a tool that counts letters.", toolCalls: [call('find_tools', { query: 'letters notes' })] },
          { toolCalls: [call('load_tools', { names: ['fake__count_letters', 'nope__missing'] })] },
          { toolCalls: [call('fake__count_letters', { text: 'hello' }), call('fake__count_letters', { text: 'hi' })] },
          { toolCalls: [call('fake__wipe_notes', {})] },
          { toolCalls: [call('fake__wipe_notes', {})] },
          { content: 'Counted.' }
        ],
        {
          toolServers: servers,
          answer: (p) => (p.kind === 'approval' ? { kind: 'approval', decision: p.action.includes('count') ? 'allow-task' : 'allow' } : null)
        }
      )
      try {
        const task = await h.run('Count the letters in hello')
        expect(task.outcome).toBe('done')
        // The model sees find/load from the start, and a loaded tool only after loading it.
        const names = (i: number) => (h.requests[i].body.tools as { function: { name: string } }[]).map((t) => t.function.name)
        expect(names(0)).toContain('find_tools')
        expect(names(0)).toHaveLength(15)
        expect(names(2)).toContain('fake__count_letters')
        expect(names(2)).not.toContain('fake__wipe_notes')
        const system = h.requests[0].body.messages![0].content
        expect(system).toContain('## Extra tools from RulZ LLM Suite (FAKE)')

        const results = h.requests[5].body.messages!.filter((m) => m.role === 'tool').map((m) => m.content)
        expect(results[0]).toMatch(/fake__count_letters: Count the letters in a text\./)
        expect(results[0]).toMatch(/fake__wipe_notes: .*\(can change things: the owner is asked separately\)/)
        expect(results[1]).toMatch(/Loaded: fake__count_letters[\s\S]*Not found: nope__missing/)
        expect(results.slice(2, 4)).toEqual(['Letters: 5', 'Letters: 2'])
        expect(results.slice(4)).toEqual(['Notes wiped.', 'Notes wiped.'])

        // "Allow for this task" on an ordinary tool covers the server's ordinary tools, not its destructive
        // ones; those were allowed just once each here, so each call asked.
        expect(h.prompts.map((p) => (p.kind === 'approval' ? p.action : p.kind))).toEqual([
          'Using FAKE: count letters',
          'Using FAKE: wipe notes',
          'Using FAKE: wipe notes'
        ])
        expect(h.prompts[1]).toMatchObject({ risk: expect.stringMatching(/can change things or run programs/), detail: expect.stringContaining('fake__wipe_notes') })
        expect(task.steps.map((s) => s.title)).toEqual([
          'Looking for tools: "letters notes"',
          'Getting tools ready',
          'Using FAKE: count letters',
          'Using FAKE: count letters',
          'Using FAKE: wipe notes',
          'Using FAKE: wipe notes'
        ])
      } finally {
        await h.close()
        await servers.dispose()
      }
    }, 60_000)

    it('unloads the oldest tools when loaded tools would take too much room', async () => {
      const servers = new ToolServers(fakeToolConfig(base))
      const h = await harness(
        [
          { toolCalls: [call('load_tools', { names: ['huge_tool'] })] },
          { toolCalls: [call('load_tools', { names: ['count_letters'] })] },
          { content: 'OK.' }
        ],
        { toolServers: servers, nCtx: 32768 }
      )
      try {
        await h.run('Load things')
        const results = h.requests[2].body.messages!.filter((m) => m.role === 'tool').map((m) => m.content)
        expect(results[0]).toMatch(/^Loaded: fake__huge_tool/)
        expect(results[1]).toMatch(/Loaded: fake__count_letters[\s\S]*Unloaded to make room: fake__huge_tool/)
        const names = (h.requests[2].body.tools as { function: { name: string } }[]).map((t) => t.function.name)
        expect(names).toContain('fake__count_letters')
        expect(names).not.toContain('fake__huge_tool')
      } finally {
        await h.close()
        await servers.dispose()
      }
    }, 60_000)

    it('runs a tool called by its full name without find or load, and respects "Don\'t allow"', async () => {
      const servers = new ToolServers(fakeToolConfig(base))
      const h = await harness(
        [{ toolCalls: [call('fake__count_letters', { text: 'abc' })] }, { toolCalls: [call('fake__count_letters', { text: 'abc' })] }, { content: 'OK.' }],
        { toolServers: servers, answer: (p) => (p.kind === 'approval' ? { kind: 'approval', decision: h.prompts.length > 1 ? 'deny' : 'allow' } : null) }
      )
      try {
        await h.run('Count abc')
        const results = h.requests[2].body.messages!.filter((m) => m.role === 'tool').map((m) => m.content)
        expect(results[0]).toBe('Letters: 3')
        expect(results[1]).toMatch(/The owner did not allow this \(fake__count_letters\)/)
      } finally {
        await h.close()
        await servers.dispose()
      }
    }, 60_000)

    it('offers no extra tools when no tool servers are set up', async () => {
      const empty = join(base, 'empty-mcp.json')
      writeFileSync(empty, '{"mcpServers":{}}')
      const h = await harness([{ toolCalls: [call('find_tools', { query: 'x' })] }, { content: 'OK.' }], { toolServers: new ToolServers(empty) })
      try {
        await h.run('Look for tools')
        expect(h.requests[0].body.tools).toHaveLength(13)
        expect(h.requests[0].body.messages![0].content).not.toContain('find_tools')
        expect(h.requests[1].body.messages!.filter((m) => m.role === 'tool')[0].content).toMatch(/There is no tool named "find_tools"/)
      } finally {
        await h.close()
      }
    }, 30_000)
  })

  it('going back to an early task undoes every later one', async () => {
    const h = await harness(
      [
        { toolCalls: [call('write_file', { path: 'a.txt', content: 'a' })] },
        { content: 'ok' },
        { toolCalls: [call('write_file', { path: 'b.txt', content: 'b' })] },
        { content: 'ok' },
        { toolCalls: [call('write_file', { path: 'c.txt', content: 'c' })] },
        { content: 'ok' }
      ],
      { useGit: false }
    )
    try {
      const first = await h.run('Task one')
      await h.run('Task two')
      await h.run('Task three')
      await h.session.undo(first.id)
      for (const f of ['a.txt', 'b.txt', 'c.txt']) expect(existsSync(join(h.root, f)), f).toBe(false)
      expect(h.session.data.tasks.every((t) => t.undone)).toBe(true)
      expect(h.session.data.notes.at(-1)).toContain('the later tasks "Task two", "Task three" were undone too')
      expect(h.session.data.timeline.at(-1)).toMatchObject({ text: expect.stringContaining('"Task one" and 2 later tasks') })
      expect(h.session.canUndo()).toBeUndefined()
    } finally {
      await h.close()
    }
  }, 60_000)
})

describe('notes while a task runs', () => {
  const userItems = (h: Harness): unknown[] =>
    h.session.data.timeline.flatMap((i) => (i.kind === 'user' ? [{ text: i.text, note: i.note }] : []))

  it('adds a note before the next step, after the running tool finishes', async () => {
    let sent = false
    const h = await harness(
      [
        { content: 'Looking.', toolCalls: [call('run_command', { command: "Start-Sleep -Milliseconds 800; 'finished'", summary: 'Waiting a bit' })] },
        { content: 'Using blue.' }
      ],
      {
        onEvent: (e, session) => {
          if (e.type === 'step' && e.step.status === 'running' && !sent) {
            sent = true
            // Even "now" must not kill a running command; it only cuts the model's replies.
            session.addNote('Make it blue', true)
          }
        }
      }
    )
    try {
      const task = await h.run('Make a button')
      expect(task.outcome).toBe('done')
      expect(task.steps[0]).toMatchObject({ status: 'done' })
      expect(task.steps[0].output).toContain('finished')
      expect(h.requests).toHaveLength(2)
      const msgs = h.requests[1].body.messages!
      expect(msgs.at(-2)?.role).toBe('tool')
      expect(msgs.at(-1)).toMatchObject({ role: 'user' })
      expect(msgs.at(-1)!.content).toMatch(/added a note while you were working[\s\S]*Make it blue$/)
      expect(userItems(h)).toEqual([
        { text: 'Make a button', note: undefined },
        { text: 'Make it blue', note: 'queued' }
      ])
      expect(h.events.filter((e) => e.type === 'notes').map((e) => (e as { notes: unknown[] }).notes.length)).toEqual([1, 0])
    } finally {
      await h.close()
    }
  }, 30_000)

  it('cuts the model short and hands it its thinking with the note', async () => {
    let sent = false
    const h = await harness([{ reasoning: 'The answer is probably 7, let me check', pause: 5000, content: 'never sent' }, { content: 'It is 42.' }], {
      onEvent: (e, session) => {
        if (e.type === 'reasoning' && !sent) {
          sent = true
          setTimeout(() => session.addNote('It is 42, I already know', true), 100)
        }
      }
    })
    try {
      const started = Date.now()
      const task = await h.run('What is the answer?')
      expect(Date.now() - started).toBeLessThan(4000)
      expect(task.outcome).toBe('done')
      const msgs = h.requests[1].body.messages!
      expect(msgs.at(-2)).toMatchObject({ role: 'assistant', content: '(The owner stopped me before I had finished thinking.)' })
      const note = msgs.at(-1)!
      expect(note.role).toBe('user')
      expect(note.content).toContain('> The answer is probably 7, let me check')
      expect(note.content).toMatch(/It is 42, I already know$/)
      // The cut reply stays in the conversation with its thinking, followed by the note.
      const tl = h.session.data.timeline.filter((i) => i.kind !== 'notice')
      expect(tl[1]).toMatchObject({ kind: 'assistant', reasoning: 'The answer is probably 7, let me check' })
      expect(tl[2]).toMatchObject({ kind: 'user', text: 'It is 42, I already know', note: 'cut' })
      expect(tl[3]).toMatchObject({ kind: 'assistant', text: 'It is 42.' })
    } finally {
      await h.close()
    }
  }, 30_000)

  it('keeps going when a note arrives during the final reply', async () => {
    let sent = false
    const h = await harness([{ reasoning: 'Wrapping up', pause: 300, content: 'All done.' }, { content: 'Footer added.' }], {
      onEvent: (e, session) => {
        if (e.type === 'reasoning' && !sent) {
          sent = true
          session.addNote('Also add a footer', false)
        }
      }
    })
    try {
      const task = await h.run('Make a page')
      expect(task.outcome).toBe('done')
      expect(h.requests).toHaveLength(2)
      expect(h.requests[1].body.messages!.at(-2)).toMatchObject({ role: 'assistant', content: 'All done.' })
      expect(h.requests[1].body.messages!.at(-1)!.content).toMatch(/Also add a footer$/)
    } finally {
      await h.close()
    }
  }, 30_000)

  it('refuses notes when no task runs, and drops cancelled ones', async () => {
    let id = ''
    const h = await harness(
      [{ content: 'One.', toolCalls: [call('run_command', { command: "Start-Sleep -Milliseconds 500; 'ok'", summary: 'Waiting' })] }, { content: 'Done.' }],
      {
        onEvent: (e, session) => {
          if (e.type === 'step' && e.step.status === 'running' && !id) {
            session.addNote('never mind', false)
            id = session.state('p', 0).notes[0].id
            session.cancelNote(id)
          }
        }
      }
    )
    try {
      expect(() => h.session.addNote('hi')).toThrow(/No task is running/)
      await h.run('Go')
      expect(id).not.toBe('')
      expect(h.requests[1].body.messages!.some((m) => m.content?.includes('never mind'))).toBe(false)
    } finally {
      await h.close()
    }
  }, 30_000)
})
