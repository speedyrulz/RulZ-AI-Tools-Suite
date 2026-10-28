import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { expandPath, findFiles, formatSize, listDirectory, nameMatcher } from '../src/main/files/fileTools'
import { McpManager } from '../src/main/mcp/manager'

const root = mkdtempSync(join(tmpdir(), 'rulz-files-'))
mkdirSync(join(root, 'App', 'resources', 'qml'), { recursive: true })
writeFileSync(join(root, 'App', 'Some App.exe'), 'x'.repeat(2048))
writeFileSync(join(root, 'App', 'engine.dll'), 'y')
writeFileSync(join(root, 'App', 'resources', 'qml', 'Main.qml'), 'z')
afterAll(() => rmSync(root, { recursive: true, force: true }))

describe('file tool helpers', () => {
  it('matches wildcards and plain text, ignoring case', () => {
    expect(nameMatcher('*.EXE')('Some App.exe')).toBe(true)
    expect(nameMatcher('*.exe')('engine.dll')).toBe(false)
    expect(nameMatcher('app')('Some App.exe')).toBe(true)
    expect(nameMatcher('m?in.qml')('Main.qml')).toBe(true)
  })
  it('expands variables and insists on absolute paths', () => {
    expect(expandPath('%Base%\\x', { BASE: 'C:\\data' })).toBe('C:\\data\\x')
    expect(() => expandPath('relative\\path', {})).toThrow(/absolute/)
    expect(() => expandPath('', {})).toThrow()
  })
  it('formats sizes', () => {
    expect(formatSize(512)).toBe('512 B')
    expect(formatSize(2048)).toBe('2.0 KB')
  })
})

describe('list_directory', () => {
  it('lists folders first, with sizes', async () => {
    const r = await listDirectory({ path: join(root, 'App') })
    expect(r.isError).toBe(false)
    const lines = r.text.split('\n').slice(1)
    expect(lines[0]).toBe('resources\\')
    expect(lines).toContain('Some App.exe  (2.0 KB)')
  })
  it('goes deeper on request', async () => {
    const r = await listDirectory({ path: join(root, 'App'), depth: 3 })
    expect(r.text).toContain('    Main.qml')
  })
  it('explains missing folders', async () => {
    const r = await listDirectory({ path: join(root, 'nope') })
    expect(r).toEqual({ text: expect.stringMatching(/does not exist/), isError: true })
  })
})

describe('find_files', () => {
  it('finds files by pattern with paths relative to the root', async () => {
    const r = await findFiles({ root, pattern: '*.exe' })
    expect(r.text).toContain(join('App', 'Some App.exe'))
    expect(r.text).not.toContain('engine.dll')
  })
  it('respects depth and limit', async () => {
    expect((await findFiles({ root, pattern: '*.qml', max_depth: 2 })).text).toMatch(/No names matching/)
    expect((await findFiles({ root, pattern: '*', limit: 2 })).text).toMatch(/limit of 2 results/)
  })
})

describe('built-in server in the MCP manager', () => {
  it('is always listed and callable', async () => {
    const mgr = new McpManager(join(root, 'mcp.json'), () => undefined)
    await mgr.init()
    expect(mgr.status()[0]).toMatchObject({ name: 'rulz', transport: 'builtin', state: 'connected' })
    expect(mgr.availableTools(new Set()).map((t) => t.qualifiedName)).toEqual(['rulz__list_directory', 'rulz__find_files'])
    const r = await mgr.callTool({ name: 'rulz__find_files', arguments: JSON.stringify({ root, pattern: 'main' }) })
    expect(r.text).toContain('Main.qml')
    expect(mgr.availableTools(new Set(['rulz__find_files'])).map((t) => t.name)).toEqual(['list_directory'])
  })
})
