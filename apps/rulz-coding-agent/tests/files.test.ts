import { mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { deletePath, editFile, findFiles, globMatcher, listFiles, readFile, searchText, writeFile } from '../src/main/agent/fileTools'
import { isInside, PathError, resolveInProject } from '../src/main/agent/pathGuard'

let base: string
let root: string
let outside: string

beforeAll(() => {
  base = mkdtempSync(join(tmpdir(), 'rulz-agent-files-'))
  root = join(base, 'project')
  outside = join(base, 'elsewhere')
  mkdirSync(join(root, 'src'), { recursive: true })
  mkdirSync(join(root, 'node_modules', 'pkg'), { recursive: true })
  mkdirSync(outside)
  writeFileSync(join(root, 'src', 'main.js'), 'const a = 1\nconst b = 2\nconsole.log(a + b)\n')
  writeFileSync(join(root, 'node_modules', 'pkg', 'index.js'), 'secret needle')
  writeFileSync(join(root, 'notes.txt'), 'needle in a haystack\n')
  writeFileSync(join(root, 'image.bin'), Buffer.from([0, 1, 2, 3, 0]))
  writeFileSync(join(outside, 'private.txt'), 'keep out')
  // A junction (no admin rights needed on Windows) that leads out of the project.
  symlinkSync(outside, join(root, 'escape'), 'junction')
})
afterAll(() => rmSync(base, { recursive: true, force: true }))

describe('path guard', () => {
  it('accepts paths inside the project', async () => {
    expect((await resolveInProject(root, 'src/main.js')).rel).toBe('src/main.js')
    expect((await resolveInProject(root, 'src\\new\\file.txt')).rel).toBe('src/new/file.txt')
    expect((await resolveInProject(root, '')).rel).toBe('')
    expect((await resolveInProject(root, join(root, 'notes.txt'))).rel).toBe('notes.txt')
  })

  it('rejects "..", absolute paths elsewhere and links that lead out', async () => {
    await expect(resolveInProject(root, '../elsewhere/private.txt')).rejects.toBeInstanceOf(PathError)
    await expect(resolveInProject(root, 'src/../../elsewhere')).rejects.toThrow(/outside the project/)
    await expect(resolveInProject(root, join(outside, 'private.txt'))).rejects.toThrow(/outside the project/)
    await expect(resolveInProject(root, 'C:\\Windows\\win.ini')).rejects.toThrow(/outside the project/)
    await expect(resolveInProject(root, 'escape/private.txt')).rejects.toThrow(/through a link/)
    await expect(resolveInProject(root, 'escape/new.txt', { write: true })).rejects.toThrow(/through a link/)
  })

  it('protects .git folders and the project folder itself from writes', async () => {
    await expect(resolveInProject(root, '.git/config', { write: true })).rejects.toThrow(/\.git/)
    await expect(resolveInProject(root, '.', { write: true })).rejects.toThrow(/not the project folder/)
  })

  it('compares paths without regard to case on Windows', () => {
    expect(isInside('C:\\Coding\\App', 'c:\\coding\\app\\x')).toBe(process.platform === 'win32')
    expect(isInside('C:\\coding\\app', 'C:\\coding\\apple')).toBe(false)
  })
})

describe('reading tools', () => {
  it('lists files without opening node_modules', async () => {
    const r = await listFiles(root, {})
    expect(r.isError).toBe(false)
    expect(r.text).toMatch(/node_modules\/ {2}\(generated/)
    expect(r.text).toMatch(/main\.js/)
    expect(r.text).not.toMatch(/pkg/)
  })

  it('finds files by wildcard and by text', async () => {
    expect((await findFiles(root, { pattern: '*.js' })).text).toBe('src/main.js')
    expect((await findFiles(root, { pattern: 'note' })).text).toBe('notes.txt')
    expect((await findFiles(root, { pattern: '' })).isError).toBe(true)
  })

  it('searches text with file and line, skipping installed packages and binaries', async () => {
    const r = await searchText(root, { query: 'NEEDLE' })
    expect(r.text).toBe('notes.txt:1: needle in a haystack')
    expect((await searchText(root, { query: 'console', glob: '*.js' })).text).toBe('src/main.js:3: console.log(a + b)')
    expect((await searchText(root, { query: 'console', glob: '*.py' })).text).toMatch(/not found/)
  })

  it('reads with line numbers and ranges, and refuses binaries', async () => {
    const r = await readFile(root, { path: 'src/main.js' })
    expect(r.text).toContain('1| const a = 1')
    expect(r.text).toContain('3| console.log(a + b)')
    const part = await readFile(root, { path: 'src/main.js', start_line: 2, end_line: 2 })
    expect(part.text).toMatch(/lines 2-2 of 3/)
    expect(part.text).not.toContain('const a')
    expect((await readFile(root, { path: 'image.bin' })).text).toMatch(/not a text file/)
    expect((await readFile(root, { path: 'missing.js' })).text).toMatch(/does not exist/)
    expect((await readFile(root, { path: 'src' })).text).toMatch(/is a folder/)
  })

  it('matches globs', () => {
    expect(globMatcher('*.js')('src/a.js')).toBe(true)
    expect(globMatcher('src/**/*.ts')('src/x/y.ts')).toBe(true)
    expect(globMatcher('src/**/*.ts')('src/y.ts')).toBe(true)
    expect(globMatcher('*.{js,ts}')('a.ts')).toBe(true)
    expect(globMatcher('*.{js,ts}')('a.py')).toBe(false)
  })
})

describe('write_file and edit_file', () => {
  it('creates folders as needed and reports created or updated', async () => {
    const r = await writeFile(root, { path: 'app/ui/window.js', content: 'one\ntwo\n' })
    expect(r).toEqual({ text: 'Created app/ui/window.js (2 lines).', isError: false, changed: [{ path: 'app/ui/window.js', change: 'created' }] })
    expect((await writeFile(root, { path: 'app/ui/window.js', content: 'three' })).changed?.[0].change).toBe('changed')
    expect(readFileSync(join(root, 'app/ui/window.js'), 'utf8')).toBe('three')
  })

  it('accepts a JSON file given as an object', async () => {
    await writeFile(root, { path: 'package.json', content: { name: 'x', scripts: { start: 'electron .' } } })
    expect(JSON.parse(readFileSync(join(root, 'package.json'), 'utf8')).scripts.start).toBe('electron .')
  })

  it('refuses to write outside the project', async () => {
    const r = await writeFile(root, { path: '../elsewhere/evil.txt', content: 'x' })
    expect(r.isError).toBe(true)
    expect(r.text).toMatch(/outside the project/)
  })

  it('replaces one exact snippet', async () => {
    writeFileSync(join(root, 'edit.js'), 'let x = 1\nlet y = 2\n')
    const r = await editFile(root, { path: 'edit.js', old_text: 'let y = 2', new_text: 'let y = 3' })
    expect(r.isError).toBe(false)
    expect(r.text).toBe('Edited edit.js at line 2.')
    expect(readFileSync(join(root, 'edit.js'), 'utf8')).toBe('let x = 1\nlet y = 3\n')
  })

  it('explains a missing snippet', async () => {
    writeFileSync(join(root, 'edit2.js'), 'function go() {\n    return 1\n}\n')
    const r = await editFile(root, { path: 'edit2.js', old_text: 'function go() {\n  return 1\n}', new_text: '' })
    expect(r.isError).toBe(true)
    expect(r.text).toMatch(/not found/)
    expect(r.text).toMatch(/first line appears at line 1/)
  })

  it('explains a snippet that is not unique', async () => {
    writeFileSync(join(root, 'edit3.js'), 'a()\nb()\na()\n')
    const r = await editFile(root, { path: 'edit3.js', old_text: 'a()', new_text: 'c()' })
    expect(r.isError).toBe(true)
    expect(r.text).toMatch(/appears 2 times .*lines 1, 3/)
    expect(readFileSync(join(root, 'edit3.js'), 'utf8')).toBe('a()\nb()\na()\n')
  })

  it('keeps Windows line endings', async () => {
    writeFileSync(join(root, 'crlf.txt'), 'one\r\ntwo\r\nthree\r\n')
    await editFile(root, { path: 'crlf.txt', old_text: 'one\ntwo', new_text: 'uno\ndos' })
    expect(readFileSync(join(root, 'crlf.txt'), 'utf8')).toBe('uno\r\ndos\r\nthree\r\n')
  })

  it('asks for old_text and points to write_file', async () => {
    expect((await editFile(root, { path: 'edit.js', new_text: 'x' })).text).toMatch(/write_file/)
  })
})

describe('delete_path', () => {
  it('uses the given trash function, never deleting directly', async () => {
    writeFileSync(join(root, 'old.txt'), 'bye')
    const trashed: string[] = []
    const r = await deletePath(root, { path: 'old.txt' }, async (abs) => {
      trashed.push(abs)
    })
    expect(r.isError).toBe(false)
    expect(trashed).toEqual([join(root, 'old.txt')])
    expect(r.changed).toEqual([{ path: 'old.txt', change: 'deleted' }])
    expect((await deletePath(root, { path: '..' }, async () => undefined)).isError).toBe(true)
  })
})
