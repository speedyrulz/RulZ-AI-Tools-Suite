import { mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { cloneFirstMessage, clonePrompt } from '@shared/re'
import type { ReClone } from '@shared/types'
import { runFileTool } from '../src/main/files/fileTools'

const base = mkdtempSync(join(tmpdir(), 'rulz-clone-'))
const app = join(base, 'App')
const out = join(base, 'out')
const outside = join(base, 'elsewhere')
mkdirSync(app, { recursive: true })
mkdirSync(out, { recursive: true })
mkdirSync(outside, { recursive: true })
writeFileSync(join(app, 'tool.exe'), 'MZ\0\0binary')
writeFileSync(join(app, 'settings.ini'), '[main]\nmode=fast\n')
afterAll(() => rmSync(base, { recursive: true, force: true }))

const scope = { writeRoot: out, target: join(app, 'tool.exe') }
const call = (name: string, args: Record<string, unknown>) => runFileTool(name, args, undefined, scope)

describe('clone file tools', () => {
  it('write and read files in the output folder, relative or absolute', async () => {
    expect((await call('write_file', { path: 'src/main.c', content: 'int main(void) {\n  return 0;\n}\n' })).isError).toBe(false)
    expect((await call('write_file', { path: join(out, 'src', 'main.c'), content: '// end\n', append: true })).text).toMatch(/^Added/)
    expect(readFileSync(join(out, 'src', 'main.c'), 'utf8')).toBe('int main(void) {\n  return 0;\n}\n// end\n')
    const r = await call('read_file', { path: 'src/main.c', offset: 2, limit: 1 })
    expect(r.text).toContain('lines 2-2 of 5')
    expect(r.text).toContain('    2    return 0;')
  })

  it('read the target folder but never write outside the output folder', async () => {
    expect((await call('read_file', { path: join(app, 'settings.ini') })).text).toContain('mode=fast')
    expect((await call('read_file', { path: join(app, 'tool.exe') })).text).toMatch(/binary file/)
    for (const path of ['../elsewhere/x.txt', join(outside, 'x.txt'), join(app, 'evil.dll'), 'C:\\Windows\\x.txt']) {
      const r = await call('write_file', { path, content: 'x' })
      expect(r.isError, path).toBe(true)
    }
    expect(existsSync(join(outside, 'x.txt'))).toBe(false)
    expect((await call('read_file', { path: join(outside, 'secret.txt') })).isError).toBe(true)
  })

  it('do not follow a link out of the output folder', async () => {
    try {
      symlinkSync(outside, join(out, 'link'), 'junction')
    } catch {
      return // links need rights this machine may not grant
    }
    const r = await call('write_file', { path: 'link/escape.txt', content: 'x' })
    expect(r.isError).toBe(true)
    expect(existsSync(join(outside, 'escape.txt'))).toBe(false)
  })

  it('never overwrite the target, even when the output folder holds it', async () => {
    const r = await runFileTool('write_file', { path: 'tool.exe', content: 'x' }, undefined, { writeRoot: app, target: join(app, 'tool.exe') })
    expect(r.isError).toBe(true)
    expect(readFileSync(join(app, 'tool.exe'), 'utf8')).toBe('MZ\0\0binary')
  })

  it('only work in a clone session', async () => {
    expect((await runFileTool('write_file', { path: 'a.txt', content: 'x' })).text).toMatch(/only works in a clone session/)
    expect((await runFileTool('read_file', { path: 'a.txt' })).isError).toBe(true)
  })
})

describe('clone prompts', () => {
  const pieces: ReClone = { scope: 'pieces', details: 'The save-file format.', language: 'Rust', outputDir: 'D:\\clones\\tool' }
  it('names the pieces, language and folder in the first message', () => {
    expect(cloneFirstMessage(pieces, 'C:\\App\\tool.exe')).toBe('Clone these pieces of C:\\App\\tool.exe:\n\nThe save-file format.\n\nWrite the new code into D:\\clones\\tool, in Rust.')
    expect(cloneFirstMessage({ ...pieces, scope: 'full', details: '', language: '' }, 'C:\\App\\tool.exe')).toBe('Fully clone C:\\App\\tool.exe.\n\nWrite the new code into D:\\clones\\tool.')
  })
  it('tells the model where and how to write, without an ANSWER line', () => {
    const p = clonePrompt(pieces)
    expect(p).toContain('Write it in Rust')
    expect(p).toContain('D:\\clones\\tool')
    expect(p).toContain('rulz__write_file')
    expect(p).toContain('do not write an ANSWER line')
    expect(clonePrompt({ ...pieces, scope: 'full' })).toContain('the whole program')
  })
})
