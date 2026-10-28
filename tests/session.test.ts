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
import { startMockRulz, type MockTurn } from './helpers/mockRulz'

const base = mkdtempSync(join(tmpdir(), 'rulz-agent-session-'))
afterAll(() => rmSync(base, { recursive: true, force: true }))
let counter = 0

interface Harness {
  session: ProjectSession
  root: string
  events: AgentEvent[]
  prompts: PendingPrompt[]
  requests: Awaited<ReturnType<typeof startMockRulz>>['requests']
  run: (text: string, opts?: { shown?: string }) => Promise<TaskRecord>
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
  } = {}
): Promise<Harness> {
  const mock = await startMockRulz({ turns, nCtx: opts.nCtx })
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
    checkPage: opts.checkPage
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
})
