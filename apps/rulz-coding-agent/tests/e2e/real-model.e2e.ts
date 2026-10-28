// Real-model check (manual): needs RulZ LLM Suite serving a model. Run with
//   $env:RULZ_REAL = '1'; npx vitest run --config vitest.e2e.config.ts real-model
// It builds a to-do app in a fresh folder, runs it, adds dark mode, then undoes that, and writes
// a report to test-results/real-model.md. Approvals are allowed automatically (and recorded).
import { spawn, spawnSync } from 'node:child_process'
import { appendFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { _electron as electron, type ElectronApplication, type Page } from 'playwright-core'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import type { RulzApi } from '@shared/ipc'
import type { ProjectState, RunState, TaskRecord } from '@shared/types'

const appRoot = resolve(__dirname, '../..')
const results = join(appRoot, 'test-results')
const report = join(results, 'real-model.md')
const base = process.env.RULZ_REAL_DIR ?? mkdtempSync(join(tmpdir(), 'rulz-agent-real-'))
const userData = join(base, 'userData')
const projectsDir = join(base, 'projects')
let projectPath = join(projectsDir, 'Todo App')

let app: ElectronApplication
let page: Page

const log = (text: string): void => appendFileSync(report, `${text}\n`)
const shot = (name: string) => page.screenshot({ path: join(results, `real-${name}.png`) })

async function state(): Promise<ProjectState> {
  return page.evaluate((p) => (globalThis as unknown as { rulz: RulzApi }).rulz.invoke('projects:open', p), projectPath)
}

/** Sends a task and handles approvals, questions and pauses until it ends. */
async function runTask(text: string, label: string): Promise<TaskRecord> {
  log(`\n## ${label}: "${text}"\n`)
  const box = page.getByRole('textbox', { name: 'What should I build or change?' })
  await box.fill(text)
  await box.press('Enter')
  return followTask()
}

/** Follows the running task to its end, answering like an owner who says yes to everything. */
async function followTask(): Promise<TaskRecord> {
  const started = Date.now()
  const box = page.getByRole('textbox', { name: 'What should I build or change?' })
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
    // The Stop button is hidden from the page while a dialog is open, so ask the app itself.
    if (!(await state()).running) break
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
  return task
}

const runState = (): Promise<RunState> =>
  page.evaluate((p) => (globalThis as unknown as { rulz: RulzApi }).rulz.invoke('run:state', p), projectPath)

/** Goes through the first-run wizard if it is showing. */
async function finishWizard(): Promise<void> {
  const wizard = page.getByText('Connected to RulZ LLM Suite.')
  const elsewhere = page.getByRole('button', { name: 'Back to your projects' }).or(page.getByPlaceholder('Project name, e.g. Video Upscaler'))
  await wizard.or(elsewhere).first().waitFor()
  if (!(await wizard.isVisible())) return
  await page.getByRole('button', { name: 'Next' }).click()
  log(`Model: ${await page.getByRole('combobox').inputValue()}`)
  await page.getByRole('button', { name: 'Next' }).click()
  await page.getByRole('button', { name: 'Next' }).click()
  await page.getByRole('textbox').fill('Scratch')
  await page.getByRole('button', { name: 'Create project' }).click()
  await page.getByText('What should I build in “Scratch”?').waitFor()
}

/** Creates a project from the home page and fills it with starting files. */
async function newProject(name: string, files: Record<string, string> = {}): Promise<void> {
  await finishWizard()
  const back = page.getByRole('button', { name: 'Back to your projects' })
  if (await back.isVisible()) await back.click()
  await page.getByPlaceholder('Project name, e.g. Video Upscaler').fill(name)
  await page.getByRole('button', { name: 'Create', exact: true }).click()
  await page.getByText(`What should I build in “${name}”?`).waitFor()
  projectPath = join(projectsDir, name)
  for (const [rel, text] of Object.entries(files)) {
    mkdirSync(dirname(join(projectPath, rel)), { recursive: true })
    writeFileSync(join(projectPath, rel), text)
  }
}

/** Opens a page in a hidden window of the app, clicks the first button matching `button` and says whether the page changed. */
async function clickInPage(file: string, button: RegExp): Promise<{ errors: string[]; result: string }> {
  return app.evaluate(
    async ({ BrowserWindow }, { file, source }) => {
      const win = new BrowserWindow({ show: false, webPreferences: { partition: 'real-check', sandbox: true } })
      const errors: string[] = []
      win.webContents.on('console-message', (e) => {
        if (e.level === 'error') errors.push(e.message)
      })
      await win.loadFile(file).catch(() => undefined)
      await new Promise((r) => setTimeout(r, 800))
      const result = (await win.webContents.executeJavaScript(`(() => {
        const re = new RegExp(${JSON.stringify(source)}, 'i')
        const b = [...document.querySelectorAll('button')].find((x) => re.test(x.innerText))
        if (!b) return 'no such button'
        const before = document.body.innerText
        b.click()
        return document.body.innerText !== before ? 'page changed' : 'nothing changed'
      })()`)) as string
      await new Promise((r) => setTimeout(r, 300))
      win.destroy()
      return { errors, result }
    },
    { file, source: button.source }
  )
}

const checkPageSteps = (task: TaskRecord): number => task.steps.filter((s) => s.tool === 'check_page').length

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

  it('checks web pages and fixes a crash with "Fix this"', async () => {
    log('\n# Checking pages and "Fix this"\n')

    // 1. A new web page: does the model look at it with check_page on its own?
    await newProject('Tip Calculator')
    const tip = await runTask('make a tip calculator web page', 'Page 1 (new web page)')
    log(`\ncheck_page used ${checkPageSteps(tip)} time(s).`)
    await shot('5-tip')

    // 2. A bug that only shows in the page: the server starts fine, but the button's script fails.
    await newProject('Shop', {
      'package.json': JSON.stringify({ name: 'shop', version: '1.0.0', scripts: { start: 'node server.js' } }, null, 2),
      'server.js': [
        "const http = require('http')",
        "const fs = require('fs')",
        "const path = require('path')",
        'const port = 4310',
        'http.createServer((req, res) => {',
        "  const file = req.url === '/' ? 'index.html' : req.url.slice(1).split('?')[0]",
        "  fs.readFile(path.join(__dirname, 'public', file), (err, data) => {",
        "    if (err) { res.writeHead(404); return res.end('Not found') }",
        "    res.writeHead(200, { 'Content-Type': file.endsWith('.js') ? 'text/javascript' : 'text/html' })",
        '    res.end(data)',
        '  })',
        '}).listen(port, () => console.log(`Shop running at http://localhost:${port}`))',
        ''
      ].join('\n'),
      'public/index.html':
        '<!doctype html>\n<html>\n<head><title>Fruit Shop</title></head>\n<body>\n  <h1>Fruit Shop</h1>\n  <p>Apples: $2</p>\n' +
        '  <button id="add-btn">Add to cart</button>\n  <p>Items in cart: <span id="cart-count">0</span></p>\n  <script src="shop.js"></script>\n</body>\n</html>\n',
      'public/shop.js':
        "let count = 0\ndocument.getElementById('add-button').addEventListener('click', () => {\n  count++\n  document.getElementById('cart-count').textContent = count\n})\n"
    })
    const shop = await runTask('The "Add to cart" button on my shop page does nothing. Please fix it.', 'Page 2 (bug only visible in the page)')
    log(`\ncheck_page used ${checkPageSteps(shop)} time(s).`)
    const clicked = await clickInPage(join(projectPath, 'public', 'index.html'), /add to cart/)
    log(`Clicking "Add to cart" afterwards: ${clicked.result}; page errors: ${clicked.errors.join(' | ') || 'none'}`)
    await shot('6-shop')

    // 3. "Fix this": the app crashes when it runs.
    await newProject('Greeter', {
      'package.json': JSON.stringify({ name: 'greeter', version: '1.0.0', scripts: { start: 'node app.js' } }, null, 2),
      'app.js': "const names = ['Ann', 'Bob', 'Cleo']\nfor (const name of names) console.log(greeting(name))\n\nfunction greet(name) {\n  return `Hello, ${name}!`\n}\n"
    })
    log('\n## Page 3 ("Fix this" after a crash)\n')
    await page.getByRole('button', { name: 'Run my app' }).click()
    await page.getByText(/Your app stopped with an error/).waitFor()
    log(`Run my app: exit code ${(await runState()).exitCode}`)
    await shot('7-crash')
    await page.getByRole('button', { name: 'Fix this' }).click()
    const fix = await followTask()
    await page.getByRole('button', { name: 'Run my app' }).click()
    await page.waitForTimeout(5000)
    const after = await runState()
    log(`\nRun my app after the fix: exit code ${after.exitCode ?? '-'}; output:\n\`\`\`\n${after.output.slice(-500)}\n\`\`\``)
    await shot('8-fixed')
    log(`\nFinished ${new Date().toISOString()}`)

    expect(tip.outcome).toBe('done')
    expect(shop.outcome).toBe('done')
    expect(clicked).toEqual({ errors: [], result: 'page changed' })
    expect(fix.outcome).toBe('done')
    expect(after.output).toContain('Hello, Ann!')
  }, 3 * 60 * 60_000)

  it('follows notes and instructions, shows the Memory meter, and packages apps with "Share my app"', async () => {
    log('\n# Notes, instructions, Memory meter and "Share my app"\n')
    await finishWizard()
    await page.evaluate(
      () => (globalThis as unknown as { rulz: RulzApi }).rulz.invoke('settings:update', { instructions: 'End your final summary with the line "Have a sunny day!"' })
    )

    // 1. A web page, with a project note the model has to follow.
    await newProject('Counter')
    await page.getByRole('button', { name: 'Notes' }).click()
    await page.getByRole('textbox', { name: 'Things to always remember' }).fill('The app is called "Sunny Counter". Use that name as the page title and heading.')
    await page.getByRole('dialog').getByRole('button', { name: 'Save' }).click()
    const counter = await runTask('make a simple counter web page with + and - buttons', 'Share 1 (web page, with a note)')
    const html = existsSync(join(projectPath, 'index.html')) ? readFileSync(join(projectPath, 'index.html'), 'utf8') : ''
    const st = await state()
    const lastSaid = [...st.timeline].reverse().find((i) => i.kind === 'assistant')?.text ?? ''
    log(`\nNote followed (index.html mentions "Sunny Counter"): ${html.includes('Sunny Counter')}`)
    log(`Instruction followed (last message ends with "Have a sunny day!"): ${/Have a sunny day!\s*$/.test(lastSaid)}`)
    log(`Memory meter: ${JSON.stringify(st.context)}`)
    const meter = await page.getByRole('meter').getAttribute('aria-label')
    log(`Meter shows: ${meter}`)

    log('\n## Share my app (web page)\n')
    await page.getByRole('button', { name: 'Share my app' }).click()
    await page.getByRole('dialog').getByRole('button', { name: 'Package it' }).click()
    const webShare = await followTask()
    const webFile = await sharedFile()
    log(`\nPackaged file: ${webFile ?? 'none'}`)
    const zipNames = webFile?.endsWith('.zip') ? readFileSync(webFile).toString('latin1') : ''
    log(`Zip contains index.html: ${zipNames.includes('index.html')}`)
    await shot('9-share-web')

    // 2. A Python desktop app made into one .exe.
    await newProject('Hello Window', {
      'main.py':
        'import tkinter as tk\n\nroot = tk.Tk()\nroot.title("Hello Window")\ntk.Label(root, text="Hello from Python!", font=("Segoe UI", 24)).pack(padx=40, pady=40)\nroot.mainloop()\n'
    })
    await page.evaluate((p) => (globalThis as unknown as { rulz: RulzApi }).rulz.invoke('run:setCommand', p, 'python main.py'), projectPath)
    log('\n## Share my app (Python window app)\n')
    await page.getByRole('button', { name: 'Share my app' }).click()
    await page.getByRole('dialog').getByRole('button', { name: 'Package it' }).click()
    const pyShare = await followTask()
    const exe = await sharedFile()
    const exeSize = exe ? statSync(exe).size : 0
    log(`\nPackaged file: ${exe ?? 'none'} (${(exeSize / 1024 / 1024).toFixed(1)} MB)`)
    let stillRunning = false
    if (exe?.endsWith('.exe')) {
      const child = spawn(exe, [], { stdio: 'ignore' })
      await page.waitForTimeout(8000)
      stillRunning = child.exitCode === null
      // The whole tree: a one-file .exe starts a second copy of itself that shows the window.
      spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { stdio: 'ignore' })
    }
    log(`The .exe was still running (its window open) after 8 s: ${stillRunning}`)
    await shot('10-share-python')
    log(`\nFinished ${new Date().toISOString()}`)

    expect(counter.outcome).toBe('done')
    expect(st.context?.exact).toBe(true)
    expect(webShare.outcome).toBe('done')
    expect(zipNames).toContain('index.html')
    expect(pyShare.outcome).toBe('done')
    expect(exeSize).toBeGreaterThan(1024 * 1024)
    expect(stillRunning).toBe(true)
  }, 3 * 60 * 60_000)
})

describe.runIf(process.env.RULZ_REAL === '1' && process.env.RULZ_REAL_REA === '1')('real model with REA', () => {
  it('uses REA from RulZ LLM Suite to study an app, then rebuilds its features', async () => {
    log('\n# REA from RulZ LLM Suite\n')
    // The app to study lives outside the project and only has minified code.
    const target = join(base, 'Unit Converter (to study)')
    mkdirSync(target, { recursive: true })
    writeFileSync(
      join(target, 'index.html'),
      '<!doctype html><html><head><meta charset="utf-8"><title>Unit Converter</title></head><body><div id="app"></div><script src="app.min.js"></script></body></html>\n'
    )
    writeFileSync(
      join(target, 'app.min.js'),
      '(()=>{const u={km:{mi:v=>v*.621371},mi:{km:v=>v/.621371},c:{f:v=>v*9/5+32},f:{c:v=>(v-32)*5/9}},n={km:"Kilometres",mi:"Miles",c:"Celsius",f:"Fahrenheit"},' +
        'h=[],a=document.getElementById("app");a.innerHTML=\'<h1>Unit Converter</h1><input id="v" type="number" value="1"><select id="p">' +
        '<option value="km:mi">Kilometres to Miles</option><option value="mi:km">Miles to Kilometres</option><option value="c:f">Celsius to Fahrenheit</option>' +
        '<option value="f:c">Fahrenheit to Celsius</option></select><button id="go">Convert</button><p id="r"></p><h2>Last 5 conversions</h2><ol id="l"></ol>\';' +
        'const r=document.getElementById("r"),l=document.getElementById("l");document.getElementById("go").onclick=()=>{const v=parseFloat(document.getElementById("v").value);' +
        'if(isNaN(v)){r.textContent="Please enter a number";return}const[f,t]=document.getElementById("p").value.split(":"),o=Math.round(u[f][t](v)*100)/100,' +
        's=`${v} ${n[f]} = ${o} ${n[t]}`;r.textContent=s;h.unshift(s);h.length>5&&h.pop();l.innerHTML=h.map(x=>`<li>${x}</li>`).join("")}})();\n'
    )
    await newProject('Converter Rebuild')
    const task = await runTask(
      `Use REA to study the app in "${target}" and find out exactly what it does: every feature, the conversions and their formulas, ` +
        'and the rules (like rounding and the history). Then build the same app with the same features as a web page in this project.',
      'REA 1 (study an app, then rebuild it)'
    )
    const reaSteps = task.steps.filter((s) => s.tool.startsWith('rea__'))
    const okRea = reaSteps.filter((s) => s.status === 'done')
    log(`\nREA steps: ${reaSteps.length} (${okRea.length} worked): ${[...new Set(reaSteps.map((s) => s.tool))].join(', ') || 'none'}`)
    const html = ['index.html', 'script.js', 'app.js', 'main.js']
      .map((f) => join(projectPath, f))
      .filter((f) => existsSync(f))
      .map((f) => readFileSync(f, 'utf8'))
      .join('\n')
    const found = {
      milesFactor: /0\.621371|1\.609/.test(html),
      fahrenheit: /9\s*\/\s*5|1\.8/.test(html) && /32/.test(html),
      lastFive: /\b5\b/.test(html) && /histor|last/i.test(html),
      rounding: /Math\.round|toFixed\(2\)/.test(html)
    }
    log(`Rebuilt features found in the code: ${JSON.stringify(found)}`)
    await shot('11-rea')
    log(`\nFinished ${new Date().toISOString()}`)

    expect(task.outcome).toBe('done')
    expect(okRea.length).toBeGreaterThan(0)
    expect(found).toEqual({ milesFactor: true, fahrenheit: true, lastFive: true, rounding: true })
  }, 3 * 60 * 60_000)
})

/** The file the last "Your app is packaged" notice points at, if any. */
async function sharedFile(): Promise<string | null> {
  const notice = [...(await state()).timeline].reverse().find((i) => i.kind === 'notice' && i.reveal)
  return notice?.kind === 'notice' && notice.reveal && existsSync(notice.reveal) ? notice.reveal : null
}
