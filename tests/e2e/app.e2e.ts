// Drives the built app with Playwright against the mock RulZ server.
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { _electron as electron, type ElectronApplication, type Page } from 'playwright-core'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { WebSocket } from 'ws'
import { fakeToolConfig } from '../helpers/fakeTools'
import { startMockRulz, type MockTurn } from '../helpers/mockRulz'

const appRoot = resolve(__dirname, '../..')
const base = mkdtempSync(join(tmpdir(), 'rulz-agent-e2e-'))
const userData = join(base, 'userData')
const projectsDir = join(base, 'projects')
const shots = join(appRoot, 'test-results')

let app: ElectronApplication
let page: Page
let mock: Awaited<ReturnType<typeof startMockRulz>>

const call = (name: string, args: unknown) => ({ name, args })
const PHONE_PORT = 7431

beforeAll(async () => {
  mkdirSync(userData, { recursive: true })
  mkdirSync(projectsDir, { recursive: true })
  mkdirSync(shots, { recursive: true })
  writeFileSync(join(userData, 'settings.json'), JSON.stringify({ projectsDir, remotePort: PHONE_PORT }))
  // The share task is answered by what it asks, not by position: the "slow" task below may be
  // stopped before its request reaches the server, which would shift every later turn.
  const shareTurns = {
    start: { content: "I'll pack the page into a zip.", toolCalls: [call('write_file', { path: 'dist/e2e-app.zip', content: 'pretend zip', summary: 'Packing your app' })] },
    done: { content: 'Done: copy **e2e-app.zip** to their PC and unzip it.\n\nSHARE: dist/e2e-app.zip' }
  }
  let next = 0
  const script: MockTurn[] = [
    {
      content: "I'll create a tiny app.",
      toolCalls: [
        call('write_file', { path: 'package.json', content: '{\n  "name": "e2e-app",\n  "scripts": { "start": "node app.js" }\n}\n', summary: 'Setting up the project' }),
        call('write_file', { path: 'app.js', content: "console.log('Hello from the app')\n", summary: 'Creating the app' })
      ]
    },
    { content: "Now I'll check that it runs.", toolCalls: [call('run_command', { command: 'node app.js', summary: 'Starting your app' })] },
    { content: 'Your app is ready.\n\n**How to try it:** click **Run my app**.' },
    { content: "I'll look at the shop page.", toolCalls: [call('check_page', { page: 'shop.html', summary: 'Looking at your shop page' })] },
    { content: 'Fixed it.' },
    { content: 'Thinking very slowly…', delay: 15_000 }
  ]
  mock = await startMockRulz({
    turns: (req) => {
      const messages = req.body.messages ?? []
      if (messages.some((m) => m.role === 'user' && m.content.includes('Package this app'))) {
        return messages.at(-1)?.role === 'tool' ? shareTurns.done : shareTurns.start
      }
      return script[next++] ?? { content: 'All done.' }
    },
    timings: () => ({ cache_n: 3000, prompt_n: 1000, predicted_n: 96 })
  })
  const electronPath = (await import('electron')).default as unknown as string
  app = await electron.launch({
    executablePath: electronPath,
    args: [appRoot],
    env: {
      ...process.env,
      RULZ_AGENT_USER_DATA: userData,
      RULZ_AGENT_CANDIDATES: mock.url,
      RULZ_AGENT_MCP_CONFIG: fakeToolConfig(base),
      RULZ_AGENT_REMOTE_HOST: '127.0.0.1',
      ELECTRON_RENDERER_URL: ''
    }
  })
  page = await app.firstWindow()
})

afterAll(async () => {
  await app?.close().catch(() => undefined)
  await mock?.close()
  rmSync(base, { recursive: true, force: true, maxRetries: 10, retryDelay: 300 })
})

describe('RulZ Coding Agent', () => {
  it('opens to the first-run wizard and creates a project', async () => {
    await page.getByText('Welcome to RulZ Coding Agent').waitFor()
    await page.getByText('Connected to RulZ LLM Suite.').waitFor()
    await page.screenshot({ path: join(shots, '1-wizard.png') })
    await page.getByRole('button', { name: 'Next' }).click()
    await page.getByText('Which AI model should build your apps?').waitFor()
    await page.getByRole('button', { name: 'Next' }).click()
    await page.getByText('When should I ask you first?').waitFor()
    await page.getByRole('button', { name: 'Next' }).click()
    await page.getByRole('textbox').fill('E2E App')
    await page.getByRole('button', { name: 'Create project' }).click()
    await page.getByText('What should I build in “E2E App”?').waitFor()
    expect(existsSync(join(projectsDir, 'E2E App'))).toBe(true)
  })

  it('builds with an approval, shows steps and files, and runs the app', async () => {
    const box = page.getByRole('textbox', { name: 'What should I build or change?' })
    await box.fill('Make a tiny app')
    await box.press('Enter')
    await page.getByRole('heading', { name: 'Is this OK?' }).waitFor()
    await page.screenshot({ path: join(shots, '2-approval.png') })
    await page.getByRole('button', { name: 'Allow', exact: true }).click()
    await page.getByText('Your app is ready.').waitFor()
    await expect.poll(() => page.getByText('Starting your app').count()).toBeGreaterThan(0)
    await page.getByRole('button', { name: /app\.js/ }).waitFor()
    await page.screenshot({ path: join(shots, '3-done.png') })

    await page.getByRole('button', { name: /app\.js/ }).click()
    await page.getByText("console.log('Hello from the app')").waitFor()
    await page.keyboard.press('Escape')

    await page.getByRole('button', { name: 'Run my app' }).click()
    await page.getByText('Hello from the app').last().waitFor()
    await page.screenshot({ path: join(shots, '4-run.png') })
    expect(existsSync(join(projectsDir, 'E2E App', 'app.js'))).toBe(true)
  })

  it('fixes a crash with "Fix this" and checks the page in a hidden window', async () => {
    const dir = join(projectsDir, 'E2E App')
    writeFileSync(join(dir, 'app.js'), "throw new Error('boom from the app')\n")
    writeFileSync(
      join(dir, 'shop.html'),
      '<!doctype html><title>Shop</title><h1>Fruit shop</h1><button>Buy apples</button>' +
        "<script>document.querySelector('#missing').addEventListener('click', () => {})</script>"
    )
    await page.getByRole('button', { name: 'Run my app' }).click()
    await page.getByText(/Your app stopped with an error/).waitFor()
    await page.screenshot({ path: join(shots, '4b-crash.png') })
    await page.getByRole('button', { name: 'Fix this' }).click()
    await page.getByText('Please fix the error my app showed when I ran it.').first().waitFor()
    await page.getByText('Fixed it.').waitFor({ timeout: 30_000 })

    const fixRequest = mock.requests.find((r) => r.body.messages?.some((m) => m.role === 'user' && m.content.includes('boom from the app')))
    expect(fixRequest).toBeTruthy()
    const last = mock.requests[mock.requests.length - 1].body.messages!
    const result = last.filter((m) => m.role === 'tool').pop()!.content
    expect(result).toContain('Title: "Shop"')
    expect(result).toContain("Cannot read properties of null (reading 'addEventListener')")
    expect(result).toContain('button "Buy apples"')
    expect(result).toContain('Fruit shop')
    await page.screenshot({ path: join(shots, '4c-fixed.png') })
  })

  it('stops while the model is generating', async () => {
    const box = page.getByRole('textbox', { name: 'What should I build or change?' })
    await box.fill('Do something slow')
    await box.press('Enter')
    await page.getByRole('button', { name: 'Stop', exact: true }).click()
    await page.getByText(/^Stopped\./).waitFor({ timeout: 10_000 })
  })

  it('undoes the last task, then goes back past two tasks from History', async () => {
    // The stopped task made no changes; undo it with the button.
    await page.getByRole('button', { name: 'Undo last task' }).click()
    await page.getByRole('dialog').getByRole('button', { name: 'Undo' }).click()
    await page.getByText(/^Undone: "Do something slow"/).waitFor()

    await page.getByRole('button', { name: 'History' }).click()
    const rows = page.getByTestId('history-row')
    await expect.poll(() => rows.allInnerTexts()).toEqual([
      expect.stringMatching(/^Do something slow[\s\S]*Undone/),
      expect.stringContaining('Please fix the error my app showed when I ran it.'),
      expect.stringMatching(/^Make a tiny app[\s\S]*files? changed/)
    ])
    await page.screenshot({ path: join(shots, '5-history.png') })
    await rows.nth(2).getByRole('button', { name: 'Go back to before this' }).click()
    const confirm = page.getByRole('dialog')
    await confirm.getByText('Everything these 2 tasks added or changed will be removed').waitFor()
    await confirm.getByRole('listitem').filter({ hasText: 'Please fix the error' }).waitFor()
    await confirm.getByRole('button', { name: 'Go back' }).click()
    await page.getByText(/^Undone: "Make a tiny app" and 1 later task\./).waitFor()

    expect(existsSync(join(projectsDir, 'E2E App', 'app.js'))).toBe(false)
    expect(existsSync(join(projectsDir, 'E2E App', 'shop.html'))).toBe(false)
    expect(existsSync(join(projectsDir, 'E2E App', 'package.json'))).toBe(false)
    await page.getByRole('button', { name: 'History' }).click()
    await expect.poll(() => rows.getByRole('button', { name: 'Go back to before this' }).count()).toBe(0)
    await page.screenshot({ path: join(shots, '5-undone.png') })
    await page.keyboard.press('Escape')
  })

  it('shows the Memory meter, keeps notes and instructions, and packages the app with "Share my app"', async () => {
    // 3000 cached + 1000 new prompt tokens + 96 written, of a 64K context.
    await page.getByRole('meter', { name: 'Memory 6% full' }).waitFor()

    await page.getByRole('button', { name: 'Notes' }).click()
    await page.getByRole('textbox', { name: 'Things to always remember' }).fill('Use blue, not purple.')
    await page.getByRole('dialog').getByRole('button', { name: 'Save' }).click()
    await expect.poll(() => page.getByRole('dialog').count()).toBe(0)

    await page.getByRole('button', { name: 'Settings' }).click()
    const instructions = page.getByRole('textbox', { name: 'Instructions for every project' })
    await instructions.fill('Explain things briefly.')
    await page.getByText('Show my built-in instructions').click() // also saves the box (it saves when you leave it)
    await page.getByText(/You are RulZ Coding Agent/).waitFor()
    // Tool servers from RulZ LLM Suite's mcp.json (here the fake one) start on demand.
    const server = page.getByTestId('tool-server')
    await expect.poll(() => server.innerText()).toMatch(/fake — starts when I first need it/)
    await page.getByRole('button', { name: 'Start now' }).click()
    await expect.poll(() => server.innerText(), { timeout: 20_000 }).toMatch(/fake — running, 3 tools/)
    await page.screenshot({ path: join(shots, '6-settings.png') })
    await page.keyboard.press('Escape')

    await page.getByRole('button', { name: 'Share my app' }).click()
    await page.getByRole('dialog').getByRole('button', { name: 'Package it' }).click()
    await page.getByText('Your app is packaged:').waitFor({ timeout: 30_000 })
    await page.getByRole('button', { name: 'Show in folder' }).waitFor()
    await page.screenshot({ path: join(shots, '6-shared.png') })
    expect(existsSync(join(projectsDir, 'E2E App', 'dist', 'e2e-app.zip'))).toBe(true)

    const request = mock.requests.find((r) => r.body.messages?.some((m) => m.role === 'user' && m.content.includes('Package this app')))!
    const system = request.body.messages![0].content
    expect(system).toContain('Use blue, not purple.')
    expect(system).toContain('Explain things briefly.')
  })

  it('lets the phone app connect, and follows what it opens and closes', async () => {
    await page.getByRole('button', { name: 'Settings' }).click()
    await page.getByRole('switch', { name: 'Let my phone connect' }).click()
    await page.getByRole('img', { name: 'QR code for pairing the phone app' }).waitFor()
    await page.screenshot({ path: join(shots, '7-phone-access.png') })
    const { remoteToken } = JSON.parse(readFileSync(join(userData, 'settings.json'), 'utf8')) as { remoteToken: string }
    expect(remoteToken.length).toBeGreaterThan(20)

    // A stand-in for the phone app.
    const ws = new WebSocket(`ws://127.0.0.1:${PHONE_PORT}/rulz`)
    const replies = new Map<string, (m: { ok: boolean; value: unknown; error?: string }) => void>()
    let welcomed!: () => void
    const welcome = new Promise<void>((r) => (welcomed = r))
    ws.on('message', (d) => {
      const m = JSON.parse(d.toString())
      if (m.t === 'welcome') welcomed()
      if (m.t === 'result') replies.get(m.id)?.(m)
    })
    await new Promise((r) => ws.once('open', r))
    ws.send(JSON.stringify({ t: 'hello', token: remoteToken, device: 'Test Phone' }))
    await welcome
    let n = 0
    const phone = (channel: string, ...args: unknown[]): Promise<{ ok: boolean; value: unknown; error?: string }> => {
      const id = String(n++)
      ws.send(JSON.stringify({ t: 'invoke', id, channel, args }))
      return new Promise((r) => replies.set(id, r))
    }
    await page.getByText('Test Phone connected').first().waitFor()
    await page.getByRole('dialog').getByText('Test Phone').first().waitFor()
    await page.keyboard.press('Escape')

    expect(await phone('settings:get')).toMatchObject({ ok: true, value: { remoteToken: '' } })
    expect((await phone('projects:create', 'Phone Made', null)).ok).toBe(true)
    await page.getByText('What should I build in “Phone Made”?').waitFor()
    expect((await phone('projects:close')).ok).toBe(true)
    await page.getByRole('heading', { name: 'Your projects' }).waitFor()
    expect((await phone('projects:open', join(projectsDir, 'E2E App'))).ok).toBe(true)
    await page.getByText('Your app is packaged:').waitFor()
    ws.close()
  })

  it('switches to the light theme', async () => {
    await page.getByRole('button', { name: 'Switch to light mode' }).click()
    await expect.poll(() => page.locator('html.dark').count()).toBe(0)
    await page.screenshot({ path: join(shots, '6-light.png') })
    await page.getByRole('button', { name: 'Switch to dark mode' }).click()
  })
})
