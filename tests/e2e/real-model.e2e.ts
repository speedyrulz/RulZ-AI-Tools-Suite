// Real-model check (manual): needs RulZ LLM Suite serving a model. Run with
//   $env:RULZ_REAL = '1'; npx vitest run --config vitest.e2e.config.ts real-model
// It builds a to-do app in a fresh folder, runs it, adds dark mode, then undoes that, and writes
// a report to test-results/real-model.md. Approvals are allowed automatically (and recorded).
import { appendFileSync, existsSync, mkdirSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { _electron as electron, type ElectronApplication, type Page } from 'playwright-core'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import type { RulzApi } from '@shared/ipc'
import type { ProjectState } from '@shared/types'

const appRoot = resolve(__dirname, '../..')
const results = join(appRoot, 'test-results')
const report = join(results, 'real-model.md')
const base = process.env.RULZ_REAL_DIR ?? mkdtempSync(join(tmpdir(), 'rulz-agent-real-'))
const userData = join(base, 'userData')
const projectsDir = join(base, 'projects')
const projectPath = join(projectsDir, 'Todo App')

let app: ElectronApplication
let page: Page

const log = (text: string): void => appendFileSync(report, `${text}\n`)
const shot = (name: string) => page.screenshot({ path: join(results, `real-${name}.png`) })

async function state(): Promise<ProjectState> {
  return page.evaluate((p) => (globalThis as unknown as { rulz: RulzApi }).rulz.invoke('projects:open', p), projectPath)
}

/** Sends a task and handles approvals, questions and pauses until it ends. */
async function runTask(text: string, label: string): Promise<void> {
  const started = Date.now()
  log(`\n## ${label}: "${text}"\n`)
  const box = page.getByRole('textbox', { name: 'What should I build or change?' })
  await box.fill(text)
  await box.press('Enter')
  await page.getByRole('button', { name: 'Stop', exact: true }).waitFor()
  let keepGoing = 0
  for (;;) {
    if (Date.now() - started > 40 * 60_000) throw new Error('Task took more than 40 minutes')
    const approval = page.getByRole('heading', { name: 'Is this OK?' })
    if (await approval.isVisible()) {
      const dialog = page.getByRole('dialog')
      await dialog.getByText('Show details').click()
      const detail = (await dialog.locator('pre').innerText().catch(() => '')).split('\n')[0]
      const action = await dialog.locator('p.font-medium').first().innerText()
      log(`- Asked: **${action}** — \`${detail.slice(0, 160)}\` → Allow`)
      await dialog.getByRole('button', { name: 'Allow', exact: true }).click()
      continue
    }
    if (await page.getByRole('button', { name: 'Keep going' }).isVisible()) {
      const msg = await page.getByRole('button', { name: 'Keep going' }).locator('xpath=../..').locator('p').first().innerText()
      log(`- Paused: ${msg} → ${keepGoing < 2 ? 'Keep going' : 'Stop'}`)
      await page.getByRole('button', { name: keepGoing++ < 2 ? 'Keep going' : 'Stop', exact: true }).click()
      continue
    }
    if (await page.getByText('I asked you a question above.').isVisible()) {
      log('- Question asked → "You decide; keep it simple."')
      await box.fill('You decide; keep it simple.')
      await box.press('Enter')
      continue
    }
    if (!(await page.getByRole('button', { name: 'Stop', exact: true }).isVisible())) break
    await page.waitForTimeout(1000)
  }
  const st = await state()
  const task = st.tasks[st.tasks.length - 1]
  log(`\nOutcome: **${task.outcome}** in ${Math.round((Date.now() - started) / 1000)} s, ${task.steps.length} steps.\n`)
  log('Steps:')
  for (const s of task.steps) log(`- [${s.status}] ${s.title}${s.detail ? ` — \`${s.detail.split('\n')[0].slice(0, 120)}\`` : ''}${s.note ? ` (${s.note})` : ''}`)
  log(`\nFiles: ${task.files.map((f) => `${f.path} (${f.change})`).join(', ') || 'none'}`)
  const said = st.timeline.filter((i) => i.kind !== 'user' && ('taskId' in i ? i.taskId === task.id : false))
  log('\nAgent messages:')
  for (const i of said) log(`> ${i.text.replace(/\n/g, '\n> ')}\n`)
}

beforeAll(async () => {
  mkdirSync(results, { recursive: true })
  mkdirSync(userData, { recursive: true })
  mkdirSync(projectsDir, { recursive: true })
  writeFileSync(join(userData, 'settings.json'), JSON.stringify({ projectsDir }))
  writeFileSync(report, `# Real-model check\n\nStarted ${new Date().toISOString()}\nProject: ${projectPath}\n`)
  const electronPath = (await import('electron')).default as unknown as string
  const env: NodeJS.ProcessEnv = { ...process.env, RULZ_AGENT_USER_DATA: userData, ELECTRON_RENDERER_URL: '' }
  delete env.RULZ_AGENT_CANDIDATES
  app = await electron.launch({ executablePath: electronPath, args: [appRoot], env: env as Record<string, string> })
  page = await app.firstWindow()
  page.setDefaultTimeout(60_000)
})

afterAll(async () => {
  await app?.close().catch(() => undefined)
})

describe.runIf(process.env.RULZ_REAL === '1')('real model', () => {
  it('builds a to-do app, runs it, adds dark mode and undoes it', async () => {
    await page.getByText('Connected to RulZ LLM Suite.').waitFor()
    await page.getByRole('button', { name: 'Next' }).click()
    const model = await page.getByRole('combobox').inputValue()
    log(`Model: ${model}`)
    await page.getByRole('button', { name: 'Next' }).click()
    await page.getByRole('button', { name: 'Next' }).click()
    await page.getByRole('textbox').fill('Todo App')
    await page.getByRole('button', { name: 'Create project' }).click()
    await page.getByText('What should I build in “Todo App”?').waitFor()

    await runTask('make a simple to-do list desktop app', 'Task 1')
    await shot('1-todo')

    log('\n## Run my app\n')
    await page.getByRole('button', { name: 'Run my app' }).click()
    await page.waitForTimeout(25_000)
    const runState = await page.evaluate((p) => (globalThis as unknown as { rulz: RulzApi }).rulz.invoke('run:state', p), projectPath)
    log(`Command: \`${runState.command}\` (${runState.source}); running after 25 s: ${runState.running}; exit code: ${runState.exitCode ?? '-'}`)
    log('```\n' + runState.output.slice(-1500) + '\n```')
    await shot('2-run')
    if (runState.running) await page.getByRole('button', { name: 'Stop my app' }).click()

    await runTask('add a dark mode toggle', 'Task 2')
    await shot('3-dark-mode')

    log('\n## Undo last task\n')
    const before = await state()
    const changed = before.tasks[before.tasks.length - 1].files
    await page.getByRole('button', { name: 'Undo last task' }).click()
    await page.getByRole('dialog').getByRole('button', { name: 'Undo' }).click()
    await page.getByText(/^Undone:/).waitFor()
    const created = changed.filter((f) => f.change === 'created')
    const stillThere = created.filter((f) => existsSync(join(projectPath, f.path)))
    log(`Undo restored the project. Files created by task 2: ${created.length}; still present after undo: ${stillThere.length}.`)
    await shot('4-undone')
    log(`\nFinished ${new Date().toISOString()}`)
    expect(stillThere).toHaveLength(0)
  }, 3 * 60 * 60_000)
})
