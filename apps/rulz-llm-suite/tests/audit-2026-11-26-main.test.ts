// Regression tests for the 2026-11-26 audit of the main process (B1-B9, B12; A12 is a one-line guard in index.ts).
import type { ChildProcess } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, afterEach, describe, expect, it, vi } from 'vitest'
import type { LoadedInstance } from '@shared/types'

// No Recycle Bin in tests: the conversation store falls back to deleting the file.
vi.mock('electron', () => ({ shell: { trashItem: () => Promise.reject(new Error('no shell')) } }))
// The real killTree would run taskkill on a made-up pid; this one records the call and stops a real child when there is one.
const killCalls: { pid: unknown; sync: unknown }[] = []
vi.mock('../src/main/engine/process', () => ({
  killTree: (pid: number | undefined, sync?: boolean) => {
    killCalls.push({ pid, sync })
    if (pid && pid !== 4242 && pid !== 1) {
      try {
        process.kill(pid)
      } catch {
        // gone
      }
    }
  }
}))
const jsonFileCalls: string[] = []
vi.mock('../src/main/util/jsonFile', async (importOriginal) => {
  const real = await importOriginal<typeof import('../src/main/util/jsonFile')>()
  return {
    ...real,
    writeTextAtomic: (path: string, text: string) => {
      jsonFileCalls.push(path)
      return real.writeTextAtomic(path, text)
    }
  }
})

const { AdapterConverter } = await import('../src/main/models/convert')
const { SetGenerator } = await import('../src/main/re/generate')
const { gradePractice, preparePractice, prunePracticeFolders } = await import('../src/main/re/practice')
const { nameMatcher } = await import('../src/main/files/fileTools')
const { McpManager } = await import('../src/main/mcp/manager')
const { SettingsStore } = await import('../src/main/settings/store')
const { ConversationStore } = await import('../src/main/chat/conversations')
const { ChatService, localModelInUse } = await import('../src/main/chat/service')

const root = mkdtempSync(join(tmpdir(), 'rulz-audit-main-'))
afterAll(() => rmSync(root, { recursive: true, force: true }))
const fresh = (name: string): string => mkdtempSync(join(root, `${name}-`))
const tick = (ms = 10): Promise<void> => new Promise((r) => setTimeout(r, ms))
async function until(cond: () => boolean, ms = 5000): Promise<void> {
  const end = Date.now() + ms
  while (!cond()) {
    if (Date.now() > end) throw new Error('timed out waiting')
    await tick(10)
  }
}

describe('B1 quitting during a conversion or a generation', () => {
  it('AdapterConverter.disposeSync kills the tree synchronously and deletes only its own .part file', async () => {
    killCalls.length = 0
    const dir = fresh('conv')
    const modelsDir = join(dir, 'models')
    mkdirSync(modelsDir)
    const peft = join(dir, 'peft')
    mkdirSync(peft)
    writeFileSync(join(peft, 'adapter_config.json'), '{}')
    writeFileSync(join(modelsDir, 'other.gguf.deadbeef.part'), 'someone else')
    writeFileSync(join(modelsDir, 'keep.gguf'), 'model')
    let tmp = ''
    const proc = Object.assign(new EventEmitter(), { stdout: new EventEmitter(), stderr: new EventEmitter(), pid: 4242 })
    const conv = new AdapterConverter({
      resolveExe: async () => join(dir, 'llama-server.exe'),
      getModelsDir: () => modelsDir,
      findPython: async () => ({ command: 'python', args: [], version: '3.12.0' }),
      findScript: () => join(dir, 'convert_lora_to_gguf.py'),
      onLog: () => undefined,
      onConverted: () => undefined,
      spawnProcess: (_c, args) => {
        tmp = args[args.indexOf('--outfile') + 1]
        writeFileSync(tmp, 'half a conversion')
        return proc as unknown as ChildProcess
      }
    })
    const done = conv.convert({ peftDir: peft, baseDir: '', baseModelId: 'org/m', outType: 'f16', outName: 'style' })
    await until(() => !!tmp)
    expect(existsSync(tmp)).toBe(true)
    conv.disposeSync()
    expect(killCalls).toEqual([{ pid: 4242, sync: true }])
    expect(existsSync(tmp)).toBe(false)
    expect(readdirSync(modelsDir).sort()).toEqual(['keep.gguf', 'other.gguf.deadbeef.part'])
    // The process ends the way a killed one does; the call still settles as cancelled.
    proc.emit('close', 1)
    expect(await done).toMatchObject({ ok: false, error: expect.stringMatching(/cancelled/) })
  })

  it('AdapterConverter.disposeSync does nothing when nothing runs', () => {
    killCalls.length = 0
    const conv = new AdapterConverter({
      resolveExe: async () => 'x',
      getModelsDir: () => root,
      findPython: async () => null,
      onLog: () => undefined,
      onConverted: () => undefined
    })
    expect(() => conv.disposeSync()).not.toThrow()
    expect(killCalls).toEqual([])
  })

  it('SetGenerator.disposeSync kills the generator and removes the folder it reserved, not the other sets', async () => {
    killCalls.length = 0
    const kit = fresh('gen')
    mkdirSync(join(kit, 'kit'))
    mkdirSync(join(kit, 'sets', 'other-0000'), { recursive: true })
    writeFileSync(join(kit, 'sets', 'other-0000', 'set.json'), '{}')
    const script = join(kit, 'fake_generate.cjs')
    writeFileSync(
      script,
      `const fs = require('fs'), path = require('path')
const a = process.argv.slice(2)
if (a.includes('--list-families')) { console.log(JSON.stringify(['keygen'])); process.exit(0) }
const out = a[a.indexOf('--out') + 1]
fs.mkdirSync(path.join(out, 'bin'), { recursive: true })
fs.writeFileSync(path.join(out, 'partial.txt'), 'x')
console.log('TOTAL 5')
setInterval(() => {}, 1000)
`
    )
    const gen = new SetGenerator({
      findPython: async () => ({ command: 'fake', args: [], version: '3.12.0' }),
      buildCommand: (_py, args) => ({ command: process.execPath, args: [script, ...args.slice(args.indexOf('kit.generate') + 1)] })
    })
    const result = gen.generate(kit, { label: 'Quit test', perFamily: 1, families: ['keygen'] }, () => undefined)
    const settled = result.then(
      () => 'ok',
      (e: Error) => e.message
    )
    let created = ''
    await until(() => {
      created = readdirSync(join(kit, 'sets')).find((n) => n.startsWith('quit-test-')) ?? ''
      return !!created && existsSync(join(kit, 'sets', created, 'partial.txt'))
    })
    gen.disposeSync()
    expect(killCalls.length).toBe(1)
    expect(killCalls[0].sync).toBe(true)
    expect(existsSync(join(kit, 'sets', created))).toBe(false)
    expect(existsSync(join(kit, 'sets', 'other-0000', 'set.json'))).toBe(true)
    expect(await settled).toMatch(/cancelled/)
    expect(gen.running).toBe(false)
  })
})

describe('B2 wildcard names', () => {
  const m = (p: string, n: string): boolean => nameMatcher(p)(n)
  it('keeps the old semantics', () => {
    expect(m('', 'anything')).toBe(true)
    expect(m('main', 'Main.qml')).toBe(true) // plain text: substring, any case
    expect(m('main', 'domain.txt')).toBe(true)
    expect(m('*.qml', 'Main.QML')).toBe(true)
    expect(m('*.qml', 'Main.qml.bak')).toBe(false) // whole name
    expect(m('a?c', 'abc')).toBe(true)
    expect(m('a?c', 'ac')).toBe(false)
    expect(m('a?c', 'abbc')).toBe(false)
    expect(m('a*c', 'ac')).toBe(true) // * matches an empty run
    expect(m('a*c', 'a-long-middle-c')).toBe(true)
    expect(m('a*c', 'a-long-middle-d')).toBe(false)
    expect(m('*', '')).toBe(true)
    expect(m('?*', '')).toBe(false)
    expect(m('**a**', 'bab')).toBe(true)
    expect(m('file?.t*t', 'file2.txt')).toBe(true)
    expect(m('file?.t*t', 'file10.txt')).toBe(false)
    expect(m('a.b*', 'a.bcd')).toBe(true) // regex characters are literal
    expect(m('a.b*', 'axbcd')).toBe(false)
    expect(m('(x)*[y]', '(x)zz[y]')).toBe(true)
    expect(m('a+b?', 'a+bc')).toBe(true)
    expect(m('a+b?', 'aab c')).toBe(false)
    expect(m('*ab*ab', 'xabyab')).toBe(true) // needs backtracking
    expect(m('*ab*ab', 'xabyabz')).toBe(false)
    expect(m('*aab', 'aaab')).toBe(true)
  })
  it('does not blow up on many stars against a long name', () => {
    const name = 'a'.repeat(150)
    const t0 = performance.now()
    expect(m('*a*a*a*a*b', name)).toBe(false)
    expect(m('*a*a*a*a*a*a*a*a*a*a*b', name)).toBe(false)
    expect(m('*a*a*a*a*a', name)).toBe(true)
    expect(performance.now() - t0).toBeLessThan(50)
  })
})

describe('B3 benchmark start while a model is in use', () => {
  const inst = (id: string, extra: Partial<LoadedInstance> = {}): LoadedInstance => ({ instanceId: id, ...extra }) as unknown as LoadedInstance
  const manager = (list: LoadedInstance[], users: Record<string, number>): Parameters<typeof localModelInUse>[0] => ({
    list: () => list,
    usersOf: (id: string) => users[id] ?? 0
  })
  it('is true only for a local instance with users', () => {
    expect(localModelInUse(manager([inst('a')], {}))).toBe(false)
    expect(localModelInUse(manager([inst('a')], { a: 1 }))).toBe(true)
    // A rental or an OpenRouter model being in use does not stop a benchmark on this PC's models.
    expect(localModelInUse(manager([inst('r', { host: 'h' } as never), inst('c', { cloud: true } as never)], { r: 2, c: 1 }))).toBe(false)
    expect(localModelInUse(manager([inst('r', { host: 'h' } as never), inst('a')], { r: 2, a: 1 }))).toBe(true)
  })
})

describe('B4 practice work folders', () => {
  const stamp = (daysAgo: number, now: number): string => String(now - daysAgo * 86_400_000).padStart(13, '0')

  it('removes only 13-digit folders older than 30 days, and then an empty task folder', async () => {
    const work = fresh('prune')
    const now = Date.UTC(2026, 10, 26)
    const old = stamp(40, now)
    const young = stamp(5, now)
    const edge = stamp(29, now)
    const mk = (...parts: string[]): string => {
      const p = join(work, ...parts)
      mkdirSync(p, { recursive: true })
      writeFileSync(join(p, 'prog.exe'), 'x')
      return p
    }
    mk('t-old-only', old)
    mk('t-mixed', old)
    mk('t-mixed', young)
    mk('t-mixed', edge)
    mk('t-mixed', '1700000000')
    mk('t-mixed', 'notes')
    mk('t-mixed', '12345678901234')
    mkdirSync(join(work, 't-mixed', 'x'.repeat(0) + 'empty'), { recursive: true })
    writeFileSync(join(work, 't-mixed', stamp(90, now)), 'a file with a timestamp name')
    writeFileSync(join(work, 'loose-file.txt'), 'x')
    mkdirSync(join(work, 't-already-empty'))
    // A link inside a task folder, named like an old stamp, pointing at a folder outside: its target must survive.
    const outside = fresh('outside')
    writeFileSync(join(outside, 'precious.txt'), 'keep')
    let linked = false
    try {
      symlinkSync(outside, join(work, 't-mixed', stamp(60, now)), 'junction')
      linked = true
    } catch {
      // no permission to create links here
    }
    await prunePracticeFolders(work, now)
    expect(existsSync(join(work, 't-old-only'))).toBe(false)
    // An empty folder the sweep did not empty is left alone.
    expect(existsSync(join(work, 't-already-empty'))).toBe(true)
    expect(readdirSync(join(work, 't-mixed')).sort()).toEqual(
      [young, edge, '1700000000', 'notes', '12345678901234', 'empty', stamp(90, now), ...(linked ? [stamp(60, now)] : [])].sort()
    )
    expect(readFileSync(join(outside, 'precious.txt'), 'utf8')).toBe('keep')
    expect(existsSync(join(work, 'loose-file.txt'))).toBe(true)
  })

  it('never follows a work folder that is itself a link', async () => {
    const real = fresh('real-practice')
    const old = stamp(40, Date.UTC(2026, 10, 26))
    mkdirSync(join(real, 't1', old), { recursive: true })
    writeFileSync(join(real, 't1', old, 'prog.exe'), 'keep')
    const linkRoot = join(fresh('link-parent'), 're-practice')
    try {
      symlinkSync(real, linkRoot, 'junction')
    } catch {
      return // no permission to create links here
    }
    await prunePracticeFolders(linkRoot, Date.UTC(2026, 10, 26))
    expect(readFileSync(join(real, 't1', old, 'prog.exe'), 'utf8')).toBe('keep')
  })

  it('does nothing for a missing work folder', async () => {
    await expect(prunePracticeFolders(join(root, 'no-such-folder'))).resolves.toBeUndefined()
  })

  it('grading copies the binary again when the work folder was pruned', async () => {
    const kit = fresh('kit')
    const work = fresh('work')
    const taskRow = { id: 's1-x-000', family: 'x', tier: 'easy', binary: 'prog.exe', prompt: 'Which? {exe}', answer: 'crc32', check: { type: 'exact', value: 'crc32' } }
    mkdirSync(join(kit, 'build', 'bin', taskRow.id), { recursive: true })
    writeFileSync(join(kit, 'build', 'tasks.jsonl'), JSON.stringify(taskRow) + '\n')
    writeFileSync(join(kit, 'build', 'bin', taskRow.id, 'prog.exe'), 'binary')
    writeFileSync(join(kit, 'build', 'bin', taskRow.id, 'prog.pdb'), 'symbols')
    const { target } = await preparePractice(kit, work, taskRow.id)
    expect(existsSync(target)).toBe(true)
    rmSync(join(target, '..'), { recursive: true, force: true })
    expect(existsSync(target)).toBe(false)
    const graded = await gradePractice(kit, work, taskRow.id, target, 'crc32')
    expect(graded).toEqual({ passed: true, expected: 'crc32' })
    expect(readFileSync(target, 'utf8')).toBe('binary')
    expect(existsSync(join(target, '..', 'prog.pdb'))).toBe(true)
    // A target that is not a work-folder copy is still refused, and nothing is created for it.
    await expect(gradePractice(kit, work, taskRow.id, join(work, 'elsewhere', 'prog.exe'), 'crc32')).rejects.toThrow(/not a work-folder copy/)
    expect(existsSync(join(work, 'elsewhere'))).toBe(false)
    // A missing folder that is not a 13-digit time is not made again.
    await gradePractice(kit, work, taskRow.id, join(work, taskRow.id, 'notes', 'prog.exe'), 'crc32')
    expect(existsSync(join(work, taskRow.id, 'notes'))).toBe(false)
    // Nor is one reached through a link: the copy would land outside the work folder.
    const outside = fresh('grade-outside')
    let made = false
    try {
      symlinkSync(outside, join(work, taskRow.id + '-l'), 'junction')
      made = true
    } catch {
      // no permission to create links here
    }
    if (made) {
      const linkTask = { ...taskRow, id: taskRow.id + '-l' }
      mkdirSync(join(kit, 'build', 'bin', linkTask.id), { recursive: true })
      writeFileSync(join(kit, 'build', 'bin', linkTask.id, 'prog.exe'), 'binary')
      writeFileSync(join(kit, 'build', 'tasks.jsonl'), JSON.stringify(taskRow) + '\n' + JSON.stringify(linkTask) + '\n')
      await gradePractice(kit, work, linkTask.id, join(work, linkTask.id, '1764115200000', 'prog.exe'), 'crc32').catch(() => undefined)
      expect(readdirSync(outside)).toEqual([])
    }
  })
})

describe('B5 mcp.json is written atomically', () => {
  it('saveConfig goes through writeTextAtomic and leaves no temporary file', async () => {
    const dir = fresh('mcp')
    const file = join(dir, 'mcp.json')
    const mgr = new McpManager(file, () => undefined)
    await mgr.init()
    jsonFileCalls.length = 0
    const text = JSON.stringify({ mcpServers: {} }, null, 2)
    await mgr.saveConfig(text)
    expect(jsonFileCalls).toEqual([file])
    expect(readFileSync(file, 'utf8')).toBe(text)
    expect(readdirSync(dir)).toEqual(['mcp.json'])
    await mgr.dispose()
  })

  it('two saves at once both succeed and the last one is on disk', async () => {
    const dir = fresh('mcp-twice')
    const file = join(dir, 'mcp.json')
    const mgr = new McpManager(file, () => undefined)
    await mgr.init()
    const a = JSON.stringify({ mcpServers: {} }, null, 2)
    const b = JSON.stringify({ mcpServers: {} }, null, 4)
    await Promise.all([mgr.saveConfig(a), mgr.saveConfig(b)])
    expect(readFileSync(file, 'utf8')).toBe(b)
    expect(readdirSync(dir)).toEqual(['mcp.json'])
    await mgr.dispose()
  })
})

describe('B6 settings files of the wrong shape', () => {
  it('starts on defaults and keeps the odd files aside', async () => {
    const dir = fresh('settings-shape')
    writeFileSync(join(dir, 'settings.json'), '[]')
    writeFileSync(join(dir, 'model-defaults.json'), 'null')
    writeFileSync(join(dir, 'presets.json'), '{"a":1}')
    writeFileSync(join(dir, 'model-mode-defaults.json'), '5')
    const store = new SettingsStore(dir)
    await store.init()
    expect(store.get()).toBeTruthy()
    expect(store.listPresets()).toEqual([])
    expect(store.allModeDefaults()).toEqual({})
    const names = readdirSync(dir)
    for (const f of ['settings.json', 'model-defaults.json', 'presets.json', 'model-mode-defaults.json']) {
      expect(names.some((n) => n.startsWith(`${f}.corrupt-`)), f).toBe(true)
      expect(names).not.toContain(f)
    }
    // Saving works again afterwards.
    await store.update({})
    expect(names.length).toBe(4)
    expect(readdirSync(dir)).toContain('settings.json')
  })

  it('leaves well-formed files alone', async () => {
    const dir = fresh('settings-ok')
    writeFileSync(join(dir, 'presets.json'), '[]')
    writeFileSync(join(dir, 'model-defaults.json'), '{}')
    await new SettingsStore(dir).init()
    expect(readdirSync(dir).sort()).toEqual(['model-defaults.json', 'presets.json'])
  })
})

describe('B12 settings update keeps memory in step with the file', () => {
  it('a save that throws leaves the old values', async () => {
    const dir = fresh('settings-fail')
    // A folder where settings.json should be: it cannot be read as a file, so saving is refused.
    mkdirSync(join(dir, 'settings.json'))
    const store = new SettingsStore(dir)
    await store.init()
    const before = store.get().theme
    const other = before === 'dark' ? 'light' : 'dark'
    await expect(store.update({ theme: other })).rejects.toThrow()
    expect(store.get().theme).toBe(before)
  })

  it('updates that run at the same time both land', async () => {
    const dir = fresh('settings-race')
    const store = new SettingsStore(dir)
    await store.init()
    await Promise.all([store.update({ theme: 'dark' }), store.update({ unloadPreviousOnLoad: !store.get().unloadPreviousOnLoad })])
    const written = JSON.parse(readFileSync(join(dir, 'settings.json'), 'utf8'))
    expect(written.theme).toBe('dark')
    expect(written.unloadPreviousOnLoad).toBe(store.get().unloadPreviousOnLoad)
  })
})

describe('B7 conversation list', () => {
  const chat = (id: string): string => JSON.stringify({ id, title: id, messages: [], updatedAt: 1 })
  const seed = (n: number): string => {
    const dir = join(fresh('convs'), 'conversations')
    mkdirSync(dir, { recursive: true })
    for (let i = 0; i < n; i++) writeFileSync(join(dir, `c${i}.json`), chat(`c${i}`))
    return dir
  }

  it('callers that arrive while the first load runs get the whole list', async () => {
    const store = new ConversationStore(seed(12))
    const [a, b, c] = await Promise.all([store.list(), store.list(), store.list()])
    expect(a.length).toBe(12)
    expect(b.length).toBe(12)
    expect(c.length).toBe(12)
  })

  it('a load that fails is not kept: the next call reads again', async () => {
    const base = fresh('convs-fail')
    const dir = join(base, 'conversations')
    writeFileSync(dir, 'a file where the folder should be') // mkdir fails
    const store = new ConversationStore(dir)
    await expect(store.list()).rejects.toThrow()
    rmSync(dir)
    mkdirSync(dir)
    writeFileSync(join(dir, 'c1.json'), chat('c1'))
    expect((await store.list()).map((s) => s.id)).toEqual(['c1'])
    // And a save after that still lands in the same list.
    await store.save({ id: 'c2', title: 't', messages: [], updatedAt: 2 } as never)
    expect((await store.list()).map((s) => s.id).sort()).toEqual(['c1', 'c2'])
  })
})

describe('B8 tool-argument quirk lookup', () => {
  afterEach(() => vi.unstubAllGlobals())
  const instance = { instanceId: 'i1' } as unknown as LoadedInstance
  const svc = (): { stringifiesArgs: (i: LoadedInstance) => Promise<boolean> } =>
    new ChatService({ baseUrl: () => 'http://127.0.0.1:1' } as never, () => undefined) as never

  it('asks again after a failed /props lookup, and remembers a good answer', async () => {
    const s = svc()
    let calls = 0
    vi.stubGlobal('fetch', async () => {
      calls++
      if (calls === 1) throw new TypeError('fetch failed')
      if (calls === 2) return { ok: false, json: async () => ({}) }
      return { ok: true, json: async () => ({ chat_template: 'no quirk here' }) }
    })
    expect(await s.stringifiesArgs(instance)).toBe(false)
    expect(await s.stringifiesArgs(instance)).toBe(false)
    expect(calls).toBe(2)
    expect(await s.stringifiesArgs(instance)).toBe(false)
    expect(calls).toBe(3)
    // The third lookup worked, so a fourth round does not fetch again.
    await s.stringifiesArgs(instance)
    expect(calls).toBe(3)
  })
})

describe('B9 conversion result on a volume without hard links', () => {
  it('falls back to a rename when the link fails', async () => {
    const dir = fresh('conv-link')
    const modelsDir = join(dir, 'models')
    mkdirSync(modelsDir)
    const peft = join(dir, 'peft')
    mkdirSync(peft)
    writeFileSync(join(peft, 'adapter_config.json'), '{}')
    const fsp = (await import('node:fs')).promises
    const spy = vi.spyOn(fsp, 'link').mockRejectedValue(Object.assign(new Error('not permitted'), { code: 'EPERM' }))
    let converted = 0
    const proc = Object.assign(new EventEmitter(), { stdout: new EventEmitter(), stderr: new EventEmitter(), pid: 1 })
    const conv = new AdapterConverter({
      resolveExe: async () => join(dir, 'llama-server.exe'),
      getModelsDir: () => modelsDir,
      findPython: async () => ({ command: 'python', args: [], version: '3.12.0' }),
      findScript: () => join(dir, 'convert_lora_to_gguf.py'),
      onLog: () => undefined,
      onConverted: () => void converted++,
      spawnProcess: (_c, args) => {
        const out = args[args.indexOf('--outfile') + 1]
        setTimeout(() => {
          writeFileSync(out, 'gguf')
          proc.emit('close', 0)
        }, 5)
        return proc as unknown as ChildProcess
      }
    })
    try {
      const r = await conv.convert({ peftDir: peft, baseDir: '', baseModelId: 'org/m', outType: 'f16', outName: 'style' })
      expect(r).toMatchObject({ ok: true, outPath: join(modelsDir, 'style.gguf') })
      expect(readFileSync(join(modelsDir, 'style.gguf'), 'utf8')).toBe('gguf')
      expect(readdirSync(modelsDir)).toEqual(['style.gguf'])
      expect(converted).toBe(1)
    } finally {
      spy.mockRestore()
    }
  })

  it('keeps the message and the other file when the name was taken meanwhile (EEXIST)', async () => {
    const dir = fresh('conv-exist')
    const modelsDir = join(dir, 'models')
    mkdirSync(modelsDir)
    const peft = join(dir, 'peft')
    mkdirSync(peft)
    writeFileSync(join(peft, 'adapter_config.json'), '{}')
    const proc = Object.assign(new EventEmitter(), { stdout: new EventEmitter(), stderr: new EventEmitter(), pid: 1 })
    const conv = new AdapterConverter({
      resolveExe: async () => join(dir, 'llama-server.exe'),
      getModelsDir: () => modelsDir,
      findPython: async () => ({ command: 'python', args: [], version: '3.12.0' }),
      findScript: () => join(dir, 'convert_lora_to_gguf.py'),
      onLog: () => undefined,
      onConverted: () => undefined,
      spawnProcess: (_c, args) => {
        const out = args[args.indexOf('--outfile') + 1]
        setTimeout(() => {
          writeFileSync(out, 'ours')
          writeFileSync(join(modelsDir, 'style.gguf'), 'theirs')
          proc.emit('close', 0)
        }, 5)
        return proc as unknown as ChildProcess
      }
    })
    const r = await conv.convert({ peftDir: peft, baseDir: '', baseModelId: 'org/m', outType: 'f16', outName: 'style' })
    expect(r).toMatchObject({ ok: false, error: expect.stringMatching(/appeared in the models folder/) })
    expect(readFileSync(join(modelsDir, 'style.gguf'), 'utf8')).toBe('theirs')
    expect(readdirSync(modelsDir)).toEqual(['style.gguf'])
  })
})
