import { mkdtemp, mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import {
  deriveSetId,
  generatorArgs,
  MAX_SEED,
  parseProgressLine,
  SetGenerator,
  setFolder,
  slugOf,
  validateGenerate
} from '../src/main/re/generate'
import {
  checkAnswer,
  envForCheck,
  gradePractice,
  listPractice,
  listSets,
  parseEnvAnswer,
  parseHexAnswer,
  preparePractice,
  readKitTasks,
  type KitTask
} from '../src/main/re/practice'
import { buildOverview } from '../src/main/re/overview'

const task = (id: string, extra: Partial<KitTask> = {}): KitTask => ({
  id,
  family: 'hash_id',
  tier: 'easy',
  binary: 'prog.exe',
  prompt: 'Which hash does {exe} use?',
  answer: 'crc32',
  check: { type: 'exact', value: 'crc32' },
  ...extra
})

async function makeSet(dir: string, tasks: KitTask[], meta?: Record<string, unknown>): Promise<void> {
  await mkdir(dir, { recursive: true })
  await writeFile(join(dir, 'tasks.jsonl'), tasks.map((t) => JSON.stringify(t)).join('\n') + '\n')
  for (const t of tasks) {
    await mkdir(join(dir, 'bin', t.id), { recursive: true })
    await writeFile(join(dir, 'bin', t.id, 'prog.exe'), `exe of ${t.id}`)
  }
  if (meta) await writeFile(join(dir, 'set.json'), JSON.stringify(meta))
}

describe('task sets', () => {
  let kit: string
  let work: string
  beforeEach(async () => {
    kit = await mkdtemp(join(tmpdir(), 'rulz-sets-'))
    work = await mkdtemp(join(tmpdir(), 'rulz-setswork-'))
  })
  afterEach(async () => {
    await rm(kit, { recursive: true, force: true })
    await rm(work, { recursive: true, force: true })
  })

  it('lists the original and finished sets, and skips folders without set.json or tasks.jsonl', async () => {
    await makeSet(join(kit, 'build'), [task('s1-hash_id-000')])
    await makeSet(join(kit, 'sets', 'round-2-ab12'), [task('round-2-ab12-hash_id-000'), task('round-2-ab12-keygen-000', { family: 'keygen' })], {
      id: 'round-2-ab12',
      label: 'Round 2',
      seed: 7,
      per_family: 1,
      families: ['hash_id', 'keygen'],
      created: '2026-11-26T10:00:00Z',
      count: 2
    })
    await makeSet(join(kit, 'sets', 'unfinished-0000'), [task('unfinished-0000-hash_id-000')]) // no set.json
    await mkdir(join(kit, 'sets', 'nometa-1111'), { recursive: true })
    await writeFile(join(kit, 'sets', 'nometa-1111', 'set.json'), '{}') // no tasks.jsonl
    await makeSet(join(kit, 'sets', 'Bad Name'), [task('x-hash_id-000')], { label: 'bad' }) // not a valid id
    const sets = await listSets(kit)
    expect(sets.map((s) => s.id)).toEqual(['original', 'round-2-ab12'])
    expect(sets[0]).toMatchObject({ label: 'Original', count: 1, families: ['hash_id'] })
    expect(sets[1]).toMatchObject({ label: 'Round 2', count: 2, seed: 7, created: '2026-11-26T10:00:00Z', families: ['hash_id', 'keygen'] })
    const info = await listPractice(kit)
    expect(info.sets.map((s) => s.id)).toEqual(['original', 'round-2-ab12'])
    expect(info.tasks.map((t) => `${t.set}:${t.id}`)).toEqual(['original:s1-hash_id-000', 'round-2-ab12:round-2-ab12-hash_id-000', 'round-2-ab12:round-2-ab12-keygen-000'])
    expect(info.warning).toBeUndefined()
    expect(JSON.stringify(info)).not.toContain('crc32')
  })

  it('reads the original label from its set.json when it has one', async () => {
    await makeSet(join(kit, 'build'), [task('a-hash_id-000')], { label: 'First kit set' })
    expect((await listSets(kit))[0].label).toBe('First kit set')
  })

  it('works with sets only (no original build) and reports the old error when there is nothing', async () => {
    await makeSet(join(kit, 'sets', 'only-0000'), [task('only-0000-hash_id-000')], { label: 'Only' })
    expect((await listPractice(kit)).sets.map((s) => s.id)).toEqual(['only-0000'])
    const empty = await mkdtemp(join(tmpdir(), 'rulz-empty-'))
    try {
      const info = await listPractice(empty)
      expect(info.error).toMatch(/kit\.generate/)
      expect(info.sets).toEqual([])
    } finally {
      await rm(empty, { recursive: true, force: true })
    }
  })

  it('finds a task in any set and prepares and grades it from that set’s bin folder', async () => {
    await makeSet(join(kit, 'build'), [task('s1-hash_id-000')])
    await makeSet(join(kit, 'sets', 'new-0000'), [task('new-0000-hash_id-000')], { label: 'New' })
    const all = await readKitTasks(kit)
    expect(all.find((t) => t.id === 'new-0000-hash_id-000')).toMatchObject({ set: 'new-0000', setDir: join(kit, 'sets', 'new-0000') })
    const p = await preparePractice(kit, work, 'new-0000-hash_id-000')
    expect(await readFile(p.target, 'utf8')).toBe('exe of new-0000-hash_id-000')
    expect((await gradePractice(kit, work, 'new-0000-hash_id-000', p.target, 'crc32')).passed).toBe(true)
    const q = await preparePractice(kit, work, 's1-hash_id-000')
    expect(await readFile(q.target, 'utf8')).toBe('exe of s1-hash_id-000')
    await expect(preparePractice(kit, work, 'nope-hash_id-000')).rejects.toThrow(/not in/)
  })

  it('uses the first set’s copy of a duplicated task id and warns', async () => {
    await makeSet(join(kit, 'build'), [task('dup-hash_id-000')])
    await makeSet(join(kit, 'sets', 'other-0000'), [task('dup-hash_id-000', { answer: 'fnv' })], { label: 'Other' })
    const info = await listPractice(kit)
    expect(info.tasks).toHaveLength(1)
    expect(info.tasks[0].set).toBe('original')
    expect(info.warning).toMatch(/1 task id appears in more than one set \(dup-hash_id-000\)/)
    const p = await preparePractice(kit, work, 'dup-hash_id-000')
    expect(await readFile(p.target, 'utf8')).toBe('exe of dup-hash_id-000')
  })

  it('passes the sets through the API overview', async () => {
    await makeSet(join(kit, 'build'), [task('s1-hash_id-000')])
    const practice = await listPractice(kit)
    const o = buildOverview({ chats: [], traces: [], practice, recording: false })
    expect(o.practice.sets.map((s) => s.id)).toEqual(['original'])
    expect(o.practice.tasks[0]).toMatchObject({ id: 's1-hash_id-000', set: 'original', attempts: 0 })
  })
})

describe('env and file checks', () => {
  it('parses env answers like the kit', () => {
    expect(parseEnvAnswer('RULZ_KEY=abc=def')).toEqual({ name: 'RULZ_KEY', value: 'abc=def' })
    expect(parseEnvAnswer('_x1=v')).toEqual({ name: '_x1', value: 'v' })
    expect(parseEnvAnswer('A =v')).toEqual({ name: 'A', value: 'v' }) // the kit strips the name too
    expect(parseEnvAnswer('A=  v w  ')).toEqual({ name: 'A', value: 'v w' })
    for (const bad of ['', 'NOEQUALS', 'A=', 'A=   ', 'A=a\0b', '=v', '1A=v', 'A B=v', 'A-B=v']) expect(parseEnvAnswer(bad), bad).toBeNull()
  })

  it('parses hex answers: prefixes, spaces, commas, case; refuses empty, odd, non-hex and over 4096 bytes', () => {
    expect([...parseHexAnswer('0x4D 0X5a, 0x90')!]).toEqual([0x4d, 0x5a, 0x90])
    expect([...parseHexAnswer('4d5A9000')!]).toEqual([0x4d, 0x5a, 0x90, 0x00])
    for (const bad of ['', '  ', '0x', 'abc', 'zz', '4d 5', '0x4g']) expect(parseHexAnswer(bad), bad).toBeNull()
    expect(parseHexAnswer('ab'.repeat(4096))).toHaveLength(4096)
    expect(parseHexAnswer('ab'.repeat(4097))).toBeNull()
  })

  it('refuses an unparsable answer without running anything', async () => {
    const env: KitTask = task('e', { check: { type: 'env', expect_stdout: 'x', expect_code: 0 } })
    const file: KitTask = task('f', { check: { type: 'file', expect_stdout: 'x', expect_code: 0 } })
    expect(await checkAnswer(env, 'no equals sign', 'does-not-exist')).toBe(false)
    expect(await checkAnswer(file, 'xyz', 'does-not-exist')).toBe(false)
  })

  // node is the stand-in program: a file check hands it the answer's bytes as its script.
  it('file check writes the hex bytes to a file and runs the program on it', async () => {
    const hex = Buffer.from('console.log("FILEOK"); process.exit(5)').toString('hex')
    const t: KitTask = task('f', { check: { type: 'file', expect_stdout: 'FILEOK', expect_code: 5 } })
    expect(await checkAnswer(t, hex, process.execPath)).toBe(true)
    expect(await checkAnswer({ ...t, check: { type: 'file', expect_stdout: 'FILEOK', expect_code: 0 } }, hex, process.execPath)).toBe(false)
    expect(await checkAnswer({ ...t, check: { type: 'file', expect_stdout: 'OTHER', expect_code: 5 } }, hex, process.execPath)).toBe(false)
  })

  // The env check's positive run cannot be faked with node (no way to give it a script), so only the pure rule and the
  // refusals are tested; a refused name never runs the program.
  it('env check never runs the program for a forbidden variable name', async () => {
    const t: KitTask = task('e', { check: { type: 'env', expect_stdout: 'x', expect_code: 0 } })
    expect(await checkAnswer(t, 'NODE_OPTIONS=--require=x.js', process.execPath)).toBe(false)
    expect(await checkAnswer(t, 'PATH=x', process.execPath)).toBe(false)
  })

  it('builds a minimal environment for env checks and rejects names that steer the program', () => {
    const base = { SystemRoot: 'C:\\Windows', Path: 'C:\\bin', TEMP: 'C:\\t', SECRET_TOKEN: 'nope', NODE_OPTIONS: '--x' }
    for (const bad of ['PATH', 'path', 'Path', 'COR_PROFILER', 'TEMP', 'temp', 'TMP', 'NODE_OPTIONS', 'PYTHONPATH', 'LD_PRELOAD', 'DOTNET_ROOT', 'ComSpec', 'USERPROFILE', 'windir', 'SystemRoot', 'JAVA_TOOL_OPTIONS', '_JAVA_OPTIONS', 'GIT_DIR']) {
      expect(envForCheck(bad, 'x', base), bad).toBeNull()
    }
    const env = envForCheck('RZ_DIAG_MODE', '7f3a', base)!
    expect(env).toEqual({ SystemRoot: 'C:\\Windows', Path: 'C:\\bin', TEMP: 'C:\\t', RZ_DIAG_MODE: '7f3a' })
    expect(envForCheck('RZ_DIAG_MODE', '1', {})).toEqual({ RZ_DIAG_MODE: '1' })
  })
})

describe('generating a set', () => {
  const known = ['keygen', 'password', 'hash_id']

  it('validates the request', () => {
    const ok = validateGenerate({ label: '  Round 2  ', perFamily: 5, families: ['keygen', 'keygen', 'hash_id'], seed: 42 }, known)
    expect(ok).toEqual({ label: 'Round 2', perFamily: 5, families: ['keygen', 'hash_id'], seed: 42 })
    expect(validateGenerate({ label: 'x', perFamily: 1, families: ['keygen'] }, known, () => 99).seed).toBe(99)
    expect(validateGenerate({ label: 'x', perFamily: 50, families: ['keygen'], seed: 0 }, known).seed).toBe(0)
    expect(validateGenerate({ label: 'x', perFamily: 1, families: ['keygen'], seed: MAX_SEED }, known).seed).toBe(MAX_SEED)
    const bad = (r: Partial<Parameters<typeof validateGenerate>[0]>): void =>
      expect(() => validateGenerate({ label: 'x', perFamily: 5, families: ['keygen'], ...r }, known)).toThrow()
    bad({ label: '   ' })
    bad({ label: 'a'.repeat(81) })
    bad({ perFamily: 0 })
    bad({ perFamily: 51 })
    bad({ perFamily: 1.5 })
    bad({ families: [] })
    bad({ families: ['keygen', 'nope'] })
    bad({ families: ['keygen;rm'] })
    bad({ seed: -1 })
    bad({ seed: MAX_SEED + 1 })
    bad({ seed: 1.5 })
    expect(() => validateGenerate({ label: 'a'.repeat(80), perFamily: 5, families: ['keygen'] }, known)).not.toThrow()
  })

  it('derives set ids from the label: slug, dash, 4 base36 characters', () => {
    expect(slugOf('Round 2: Hard!')).toBe('round-2-hard')
    expect(slugOf('  --Ünï--  ')).toBe('uni')
    expect(slugOf('???')).toBe('set')
    expect(slugOf('a'.repeat(50))).toHaveLength(30)
    expect(slugOf('a'.repeat(29) + ' b')).toBe('a'.repeat(29))
    expect(deriveSetId('Round 2', () => false, () => 'k3z9')).toBe('round-2-k3z9')
    for (let i = 0; i < 50; i++) {
      const id = deriveSetId('S1', () => false)
      expect(id).toMatch(/^[a-z0-9][a-z0-9_-]{0,39}$/)
      expect(id).toMatch(/^s1-[0-9a-z]{4}$/)
      expect(id).not.toMatch(/^s\d+$/)
    }
    // never an id that exists
    const seq = ['aaaa', 'aaaa', 'bbbb']
    expect(deriveSetId('x', (id) => id === 'x-aaaa', () => seq.shift()!)).toBe('x-bbbb')
    expect(() => deriveSetId('x', () => true)).toThrow(/free set id/)
  })

  it('builds the generator command line', () => {
    const args = generatorArgs('r-1234', 'C:\\kit\\sets\\r-1234', { label: 'My "set"', perFamily: 3, families: ['keygen', 'hash_id'], seed: 5 })
    expect(args).toEqual(['-u', '-m', 'kit.generate', '--out', 'C:\\kit\\sets\\r-1234', '--seed', '5', '--per-family', '3', '--families', 'keygen,hash_id', '--label', 'My "set"', '--id-prefix', 'r-1234'])
    expect(args).not.toContain('-I')
  })

  it('reads progress lines', () => {
    const st = { done: 0, total: 0 }
    expect(parseProgressLine('TOTAL 55', st)).toEqual({ done: 0, total: 55, line: 'TOTAL 55' })
    expect(parseProgressLine('r-1234-keygen-000 [easy] keygen_easy.exe', st)).toMatchObject({ done: 1, total: 55 })
    expect(parseProgressLine('r-1234-keygen-001 [medium] a.exe', st)).toMatchObject({ done: 2, total: 55 })
    expect(parseProgressLine('compiling...', st)).toBeNull()
    expect(st.done).toBe(2)
  })

  it('only gives folders strictly inside <kit>/sets for removal and cleanup', () => {
    const kit = join(tmpdir(), 'somekit')
    expect(setFolder(kit, 'round-2-ab12')).toBe(join(kit, 'sets', 'round-2-ab12'))
    for (const bad of ['original', '', '.', '..', '../build', 'a/b', 'a\\b', '..\\..', 'Upper', 'x'.repeat(41), 'sets']) {
      if (bad === 'sets') expect(setFolder(kit, bad)).toBe(join(kit, 'sets', 'sets'))
      else expect(() => setFolder(kit, bad), bad).toThrow()
    }
  })
})

// A stand-in for `python -m kit.generate`, run by node.
const FAKE = `
const fs = require('fs'), path = require('path')
const a = process.argv.slice(2)
const get = (k) => a[a.indexOf(k) + 1]
if (a.includes('--list-families')) { console.log(JSON.stringify(['keygen', 'hash_id'])); process.exit(0) }
const out = get('--out'), label = get('--label'), prefix = get('--id-prefix'), fams = get('--families').split(',')
fs.mkdirSync(path.join(out, 'bin'), { recursive: true })
console.log('TOTAL ' + fams.length * Number(get('--per-family')))
if (label === 'fail') { console.error('error: cl.exe not found'); process.exit(3) }
const lines = []
let n = 0
for (const f of fams) for (let i = 0; i < Number(get('--per-family')); i++) {
  const id = prefix + '-' + f + '-' + String(i).padStart(3, '0')
  fs.mkdirSync(path.join(out, 'bin', id), { recursive: true })
  fs.writeFileSync(path.join(out, 'bin', id, 'prog.exe'), 'x')
  lines.push(JSON.stringify({ id, family: f, tier: 'easy', binary: 'prog.exe', prompt: 'p', answer: 'a', check: { type: 'exact', value: 'a' } }))
  console.log(id + ' [easy] prog.exe')
  n++
  if (label === 'hang' && n === 1) setInterval(() => {}, 1000)
}
if (label === 'hang') { /* never finishes */ } else {
  fs.writeFileSync(path.join(out, 'tasks.jsonl'), lines.join('\\n') + '\\n')
  fs.writeFileSync(path.join(out, 'set.json'), JSON.stringify({ id: prefix, label, seed: Number(get('--seed')), per_family: Number(get('--per-family')), families: fams, created: new Date().toISOString(), count: n }))
}
`

describe('SetGenerator', () => {
  let kit: string
  let script: string
  const gen = (): SetGenerator =>
    new SetGenerator({
      findPython: async () => ({ command: 'fake', args: [], version: '3.12.0' }),
      buildCommand: (_py, args) => ({ command: process.execPath, args: [script, ...args.slice(args.indexOf('kit.generate') + 1)] })
    })
  const noPython = (): SetGenerator => new SetGenerator({ findPython: async () => null })

  beforeEach(async () => {
    kit = await mkdtemp(join(tmpdir(), 'rulz-gen-'))
    await mkdir(join(kit, 'kit'), { recursive: true })
    script = join(kit, 'fake_generate.cjs')
    await writeFile(script, FAKE)
  })
  afterEach(async () => {
    await rm(kit, { recursive: true, force: true })
  })

  it('lists families', async () => {
    expect(await gen().listFamilies(kit)).toEqual(['keygen', 'hash_id'])
    await expect(noPython().listFamilies(kit)).rejects.toThrow(/Python/)
    await expect(gen().listFamilies(join(kit, 'missing'))).rejects.toThrow(/no kit folder/)
  })

  it('generates a set, streams progress, and the set then lists', async () => {
    const seen: [number, number][] = []
    const { setId } = await gen().generate(kit, { label: 'Round 2', perFamily: 2, families: ['keygen', 'hash_id'], seed: 9 }, (p) => seen.push([p.done, p.total]))
    expect(setId).toMatch(/^round-2-[0-9a-z]{4}$/)
    expect(seen.at(-1)).toEqual([4, 4])
    expect(seen).toContainEqual([0, 4])
    const sets = await listSets(kit)
    expect(sets).toHaveLength(1)
    expect(sets[0]).toMatchObject({ id: setId, label: 'Round 2', count: 4, seed: 9 })
    expect((await readKitTasks(kit)).map((t) => t.id)).toContain(`${setId}-keygen-000`)
  })

  it('removes the partial folder and reports the output when the generator fails', async () => {
    await expect(gen().generate(kit, { label: 'fail', perFamily: 1, families: ['keygen'] }, () => undefined)).rejects.toThrow(/exit code 3[\s\S]*cl\.exe not found/)
    expect(await readdir(join(kit, 'sets'))).toEqual([])
  })

  it('refuses a bad request before running anything, and a second job while one runs', async () => {
    await expect(gen().generate(kit, { label: '', perFamily: 1, families: ['keygen'] }, () => undefined)).rejects.toThrow(/label/)
    await expect(gen().generate(kit, { label: 'x', perFamily: 1, families: ['nope'] }, () => undefined)).rejects.toThrow(/no family/)
    await expect(stat(join(kit, 'sets'))).rejects.toThrow()
    const g = gen()
    const first = g.generate(kit, { label: 'hang', perFamily: 1, families: ['keygen'] }, (p) => p.done === 1 && g.cancel())
    await expect(g.generate(kit, { label: 'x', perFamily: 1, families: ['keygen'] }, () => undefined)).rejects.toThrow(/already/)
    await expect(first).rejects.toThrow(/cancelled/)
  }, 30_000)

  it('cancel stops the process and removes the partial set', async () => {
    const g = gen()
    await expect(g.generate(kit, { label: 'hang', perFamily: 1, families: ['keygen'] }, (p) => p.done === 1 && g.cancel())).rejects.toThrow(/cancelled/)
    expect(g.running).toBe(false)
    expect(await readdir(join(kit, 'sets'))).toEqual([])
    // the generator is free again
    await expect(g.generate(kit, { label: 'again', perFamily: 1, families: ['keygen'] }, () => undefined)).resolves.toMatchObject({ setId: expect.stringMatching(/^again-/) })
  }, 30_000)

  it('removePartial only touches the exact set folder inside sets/', async () => {
    await mkdir(join(kit, 'sets', 'a-0000'), { recursive: true })
    await mkdir(join(kit, 'sets', 'b-0000'), { recursive: true })
    await mkdir(join(kit, 'build'), { recursive: true })
    const g = gen()
    await g.removePartial(kit, 'a-0000')
    expect(await readdir(join(kit, 'sets'))).toEqual(['b-0000'])
    await expect(g.removePartial(kit, '..')).rejects.toThrow()
    await expect(g.removePartial(kit, 'original')).rejects.toThrow()
    expect((await stat(join(kit, 'build'))).isDirectory()).toBe(true)
  })
})
