// RulZ Coding Agent screenshots, against the app's own stand-in for RulZ LLM Suite (tests/helpers/mockRulz.ts):
// scripted model turns, so no model runs. Node 22.18+ (it runs this .ts file directly).
//
//   cd apps/rulz-coding-agent && npm ci && npx electron-vite build
//   node tools/screenshots/capture_coding_agent.ts <apps/rulz-coding-agent> <out folder> [electron.exe]
//
// The app gets a fresh settings folder and projects folder under the out folder's "work" subfolder (deleted at the
// start), so the real %APPDATA% settings and projects are never read. The phone-access switch listens on 127.0.0.1.
import { mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const appRoot = resolve(process.argv[2])
const out = resolve(process.argv[3])
const req = createRequire(join(appRoot, 'package.json'))
const { _electron: electron } = req('playwright-core')
const electronPath: string = process.argv[4] || req('electron')
const { startMockRulz } = await import(pathToFileURL(join(appRoot, 'tests/helpers/mockRulz.ts')).href)

const work = join(out, 'work')
rmSync(work, { recursive: true, force: true })
const userData = join(work, 'userData')
const projectsDir = join(work, 'projects')
mkdirSync(userData, { recursive: true })
mkdirSync(projectsDir, { recursive: true })
writeFileSync(join(userData, 'settings.json'), JSON.stringify({ projectsDir, remotePort: 7432 }))
const mcpConfig = join(work, 'mcp.json') // no extra tools, instead of the ones set up in RulZ LLM Suite
writeFileSync(mcpConfig, JSON.stringify({ mcpServers: {} }))

const call = (name: string, args: unknown) => ({ name, args })
const SERVER = `const http = require('http')
const fs = require('fs')
const path = require('path')

const types = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript' }
http.createServer((req, res) => {
  const file = path.join(__dirname, 'public', req.url === '/' ? 'index.html' : req.url)
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end('Not found') }
    res.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'text/plain' })
    res.end(data)
  })
}).listen(3000, () => console.log('To-do list running at http://localhost:3000'))
`
const HTML = `<!doctype html>
<html>
<head><meta charset="utf-8"><title>To-do list</title><link rel="stylesheet" href="style.css"></head>
<body>
  <h1>My to-do list</h1>
  <form id="add"><input id="task" placeholder="Something to do"><button>Add</button></form>
  <ul id="list"></ul>
  <script src="app.js"></script>
</body>
</html>
`
const CSS = `body { font-family: system-ui, sans-serif; max-width: 480px; margin: 40px auto; }
button { background: #7c3aed; color: white; border: 0; border-radius: 6px; padding: 6px 14px; }
li.done { text-decoration: line-through; color: #888; }
`
const APP = `const list = document.getElementById('list')
document.getElementById('add').addEventListener('submit', (e) => {
  e.preventDefault()
  const input = document.getElementById('task')
  if (!input.value.trim()) return
  const li = document.createElement('li')
  li.textContent = input.value
  li.addEventListener('click', () => li.classList.toggle('done'))
  list.appendChild(li)
  input.value = ''
})
`
const turns = [
  { content: "I'll build a simple to-do list that runs in your browser: a small web server, a page, some styling and the list logic.",
    toolCalls: [
      call('write_file', { path: 'package.json', content: '{\n  "name": "todo-list",\n  "version": "1.0.0",\n  "scripts": { "start": "node server.js" }\n}\n', summary: 'Setting up the project' }),
      call('write_file', { path: 'server.js', content: SERVER, summary: 'Creating the small web server' }),
      call('write_file', { path: 'public/index.html', content: HTML, summary: 'Creating the page' }),
      call('write_file', { path: 'public/app.js', content: APP, summary: 'Adding the to-do list logic' }),
      call('write_file', { path: 'public/style.css', content: CSS, summary: 'Adding the styling' })
    ] },
  { content: 'Files are in place. Now I will install what the app needs.',
    toolCalls: [call('run_command', { command: 'npm install', summary: 'Installing the pieces your app needs' })] },
  { content: 'Your to-do list is ready.\n\n**How to try it:** click **Run my app**, then open http://localhost:3000 in your browser. Type a task and press **Add**; click a task to tick it off.' },
  { content: "I'll make the buttons blue.",
    toolCalls: [call('edit_file', { path: 'public/style.css', old_text: 'background: #7c3aed;', new_text: 'background: #2563eb;', summary: 'Making the buttons blue' })] },
  { content: 'Done: the buttons are blue now. Reload the page to see it.' }
]
let next = 0
const mock = await startMockRulz({
  loadConfig: true,
  models: [{ id: 'Qwen3-Coder-30B-A3B-Instruct-Q4_K_M', state: 'loaded', loaded_context_length: 65536 }],
  turns: () => turns[next++] ?? { content: 'All done.' },
  timings: () => ({ cache_n: 9000, prompt_n: 1200, predicted_n: 240 })
})

const app = await electron.launch({
  executablePath: electronPath,
  // Keep painting when another window (e.g. a command's console) covers this one, or screenshots time out.
  args: [appRoot, '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding', '--force-device-scale-factor=1'],
  env: { ...process.env, RULZ_AGENT_USER_DATA: userData, RULZ_AGENT_CANDIDATES: mock.url, RULZ_AGENT_MCP_CONFIG: mcpConfig, RULZ_AGENT_REMOTE_HOST: '127.0.0.1', ELECTRON_RENDERER_URL: '' }
})
const page = await app.firstWindow()
await app.evaluate(({ BrowserWindow }) => {
  const w = BrowserWindow.getAllWindows()[0]
  w.unmaximize()
  w.setContentSize(1600, 960)
  w.center()
})
const shot = async (name: string) => {
  await page.waitForTimeout(600)
  await page.screenshot({ path: join(out, name) })
  console.log('saved', name)
}

try {
  await page.getByText('Welcome to RulZ Coding Agent').waitFor()
  await page.getByText('Connected to RulZ LLM Suite.').waitFor()
  await shot('first-run-wizard.png')
  await page.getByRole('button', { name: 'Next' }).click()
  await page.getByText('Which AI model should build your apps?').waitFor()
  await page.getByRole('button', { name: 'Next' }).click()
  await page.getByText('When should I ask you first?').waitFor()
  await page.getByRole('button', { name: 'Next' }).click()
  await page.getByRole('textbox').fill('To-do list')
  await page.getByRole('button', { name: 'Create project' }).click()

  const box = page.getByRole('textbox', { name: 'What should I build or change?' })
  await box.fill('Make a simple to-do list app that runs in my browser. I want to add tasks and tick them off.')
  await box.press('Enter')
  await page.getByRole('heading', { name: 'Is this OK?' }).waitFor()
  await shot('approval.png')
  await page.getByRole('button', { name: 'Allow', exact: true }).click()
  await page.getByText('Your to-do list is ready.').waitFor({ timeout: 60_000 })
  await shot('conversation-dark.png')

  await page.getByRole('button', { name: 'Run my app' }).click()
  await page.getByText('To-do list running at http://localhost:3000').last().waitFor({ timeout: 30_000 })
  await shot('run-my-app.png')
  await page.keyboard.press('Escape')

  await box.fill('Make the buttons blue.')
  await box.press('Enter')
  await page.getByText('Done: the buttons are blue now.').waitFor({ timeout: 30_000 })
  await page.getByRole('button', { name: /style\.css/ }).last().click()
  await page.getByText('#2563eb').first().waitFor()
  await shot('file-diff.png')
  await page.keyboard.press('Escape')

  await page.getByRole('button', { name: 'Switch to light mode' }).click()
  await page.waitForTimeout(500)
  await shot('conversation-light.png')
  await page.getByRole('button', { name: 'Switch to dark mode' }).click()

  await page.getByRole('button', { name: 'Settings' }).click()
  await page.getByRole('switch', { name: 'Let my phone connect' }).click()
  const qr = page.getByRole('img', { name: 'QR code for pairing the phone app' })
  await qr.waitFor()
  await qr.evaluate((el) => el.scrollIntoView({ block: 'end' }))
  await shot('phone-access.png')
  await page.keyboard.press('Escape')
} finally {
  await app.close().catch(() => undefined)
  await mock.close()
}
