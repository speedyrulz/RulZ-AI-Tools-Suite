import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { afterAll, describe, expect, it } from 'vitest'
import { outputLooksLikeError } from '@shared/runOutput'
import { formatPageReport, pageHasProblems, pageTarget, type PageReport } from '../src/main/agent/pageCheck'
import { FIX_RUN_SHOWN, fixRunTask } from '../src/main/agent/prompt'

const base = mkdtempSync(join(tmpdir(), 'rulz-agent-page-'))
const root = join(base, 'My App')
mkdirSync(join(root, 'public'), { recursive: true })
writeFileSync(join(root, 'index.html'), '<h1>Hi</h1>')
writeFileSync(join(root, 'public', 'about.html'), '<h1>About</h1>')
writeFileSync(join(base, 'secret.html'), 'not part of the project')
afterAll(() => rmSync(base, { recursive: true, force: true }))

describe('which page check_page opens', () => {
  it('opens local web addresses and keeps the page on the same server', async () => {
    expect(await pageTarget(root, 'http://localhost:5173')).toEqual({ url: 'http://localhost:5173/', allowedPrefix: 'http://localhost:5173/' })
    expect(await pageTarget(root, 'http://127.0.0.1:3000/todo?x=1')).toMatchObject({ url: 'http://127.0.0.1:3000/todo?x=1' })
    expect((await pageTarget(root, 'http://0.0.0.0:8080/')).url).toBe('http://localhost:8080/')
  })

  it('opens HTML files in the project', async () => {
    const t = await pageTarget(root, 'public/about.html')
    expect(t.url).toBe(pathToFileURL(join(root, 'public', 'about.html')).href)
    expect(t.allowedPrefix).toBe(`${pathToFileURL(root).href}/`)
    expect((await pageTarget(root, pathToFileURL(join(root, 'index.html')).href)).url).toBe(pathToFileURL(join(root, 'index.html')).href)
  })

  it('refuses the internet, files outside the project and missing files', async () => {
    await expect(pageTarget(root, 'https://example.com')).rejects.toThrow(/on the internet/)
    await expect(pageTarget(root, 'http://192.168.1.5:3000')).rejects.toThrow(/on the internet/)
    await expect(pageTarget(root, '../secret.html')).rejects.toThrow(/outside the project/)
    await expect(pageTarget(root, pathToFileURL(join(base, 'secret.html')).href)).rejects.toThrow(/outside the project/)
    await expect(pageTarget(root, 'missing.html')).rejects.toThrow(/not a file in the project/)
    await expect(pageTarget(root, '')).rejects.toThrow(/Give the page/)
  })
})

describe('page report', () => {
  const report = (patch: Partial<PageReport>): PageReport => ({
    url: 'http://localhost:5173/',
    title: 'To-do',
    status: 200,
    messages: [],
    failedRequests: [],
    text: 'My tasks\nBuy milk',
    controls: ['field (text) "New task"', 'button "Add"'],
    visuals: 0,
    ...patch
  })

  it('describes a working page', () => {
    const r = report({})
    const text = formatPageReport(r)
    expect(text).toContain('Opened http://localhost:5173/ (status 200). Title: "To-do"')
    expect(text).toContain('No errors in the page.')
    expect(text).toContain('Buy milk')
    expect(text).toContain('button "Add"')
    expect(pageHasProblems(r)).toBe(false)
  })

  it('lists script errors with their file and line, and failed files', () => {
    const r = report({
      messages: [
        { level: 'error', message: "Uncaught TypeError: Cannot read properties of null (reading 'addEventListener')", source: 'http://localhost:5173/src/app.js?t=1', line: 12 },
        { level: 'warning', message: 'Deprecated thing' }
      ],
      failedRequests: [{ url: 'http://localhost:5173/style.css', problem: 'status 404' }]
    })
    const text = formatPageReport(r)
    expect(text).toContain('Errors in the page (1):')
    expect(text).toContain("reading 'addEventListener') (app.js:12)")
    expect(text).toContain('Warnings (1):')
    expect(text).toContain('http://localhost:5173/style.css: status 404')
    expect(pageHasProblems(r)).toBe(true)
  })

  it('points out an empty page and a server that is not running', () => {
    expect(formatPageReport(report({ text: '  ', controls: [] }))).toContain('The page looks empty')
    const down = report({ loadError: 'ERR_CONNECTION_REFUSED (http://localhost:5173/)' })
    expect(formatPageReport(down)).toMatch(/did not load[\s\S]*Start the server first/)
    expect(pageHasProblems(down)).toBe(true)
  })
})

describe('"Fix this"', () => {
  it('recognises error output', () => {
    for (const out of [
      "TypeError: Cannot read properties of undefined (reading 'map')",
      'Traceback (most recent call last):\n  File "main.py", line 3',
      "Error: Cannot find module 'express'",
      'npm error code ENOENT',
      'Uncaught ReferenceError: x is not defined',
      'ModuleNotFoundError: No module named tkinter'
    ]) {
      expect(outputLooksLikeError(out), out).toBe(true)
    }
    for (const out of ['Server running at http://localhost:3000', 'Compiled with 0 errors', 'VITE v5 ready in 300 ms', 'Error handling is set up']) {
      expect(outputLooksLikeError(out), out).toBe(false)
    }
  })

  it('gives the agent the command, how it ended and the end of the output', () => {
    const text = fixRunTask({ command: 'npm start', exitCode: 1, running: false, output: `${'x'.repeat(10_000)}\nError: boom` })
    expect(text).toContain('stopped with an error (exit code 1)')
    expect(text).toContain('runs this command in the project folder: npm start')
    expect(text).toContain('Error: boom')
    expect(text).toContain('earlier output omitted')
    expect(text.length).toBeLessThan(6500)
    expect(fixRunTask({ command: 'npm start', running: true, output: 'Uncaught x' })).toContain('while it was running (I have stopped it)')
    expect(FIX_RUN_SHOWN).not.toContain('```')
  })
})
