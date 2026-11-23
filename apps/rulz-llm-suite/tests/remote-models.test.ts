// src/main/remote/models.ts and remoteArgs.ts: definitions, file lists, download addresses and server arguments.
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { DEFAULT_LOAD_CONFIG } from '@shared/defaults'
import type { RemoteModelDef } from '@shared/remote'
import type { AdapterEntry, LoadConfig, ModelEntry } from '@shared/types'
import { expandShards, fileList, hfUrl, makeSlug, RemoteModelStore, toModelEntry, validateSource } from '../src/main/remote/models'
import { buildRemoteArgs, extraArgsProblem } from '../src/main/remote/remoteArgs'
import { FILE_NAME_RE } from '../src/main/remote/ssh'

const tmp = (): string => join(mkdtempSync(join(tmpdir(), 'rulz-rm-')), 'remote-models.json')
const hf = (over: Record<string, unknown> = {}): { kind: 'hf'; repo: string; bucket: boolean; revision: string; file: string; mmproj: string } => ({
  kind: 'hf',
  repo: 'unsloth/Qwen3-GGUF',
  bucket: false,
  revision: 'main',
  file: 'Qwen3-Q4_K_M.gguf',
  mmproj: '',
  ...over
})

describe('validateSource', () => {
  it('accepts a library model and a Hugging Face file', () => {
    expect(validateSource({ kind: 'local', modelId: ' sub/model.gguf ' })).toEqual({ kind: 'local', modelId: 'sub/model.gguf' })
    expect(validateSource(hf({ file: '/Q4/a.gguf', revision: '' }))).toEqual(hf({ file: 'Q4/a.gguf' }))
  })

  it('refuses what cannot work, in sentences', () => {
    expect(() => validateSource({ kind: 'local', modelId: '' })).toThrow(/library/)
    expect(() => validateSource({ kind: 'local', modelId: 'vast/other' })).toThrow(/another remote model/)
    expect(() => validateSource(null)).toThrow()
    expect(() => validateSource({ kind: 'ftp' })).toThrow()
    for (const repo of ['', 'noslash', 'a/b/c', '../x', 'a/..', 'a b/c', "o/n'x", '/x']) expect(() => validateSource(hf({ repo })), repo).toThrow(/owner\/name/)
    for (const file of ['', 'a.txt', '../a.gguf', 'a/../b.gguf', 'a//b.gguf', 'a\\b.gguf', 'a/./b.gguf', 'x\n.gguf']) expect(() => validateSource(hf({ file })), file).toThrow()
    expect(() => validateSource(hf({ mmproj: '../mm.gguf' }))).toThrow()
    expect(() => validateSource(hf({ revision: '../main' }))).toThrow(/revision/)
    expect(() => validateSource(hf({ revision: 'a b' }))).toThrow(/revision/)
  })
})

describe('slugs', () => {
  it('makes a folder name from the name and keeps it unique', () => {
    expect(makeSlug('Qwen3 35B (Q4_K_M)!', new Set())).toBe('qwen3-35b-q4_k_m')
    expect(makeSlug('  ', new Set())).toBe('model')
    expect(makeSlug('###', new Set())).toBe('model')
    expect(makeSlug('x', new Set(['vast/x']))).toBe('x-2')
    expect(makeSlug('x', new Set(['vast/x', 'vast/x-2']))).toBe('x-3')
    expect(makeSlug('A'.repeat(200), new Set())).toHaveLength(52)
    expect(makeSlug('über model', new Set())).toBe('ber-model')
  })

  it('always passes the slug rule used on the instance', () => {
    for (const n of ['a', '..', '-x', '.hidden', 'ünï', 'a'.repeat(80), 'Model: Q4/K', '漢字']) expect(makeSlug(n, new Set())).toMatch(/^[a-z0-9][a-z0-9._-]{0,59}$/)
  })
})

describe('RemoteModelStore', () => {
  it('saves, lists, updates and deletes, and keeps it on disk', async () => {
    const path = tmp()
    const s = new RemoteModelStore(path, () => 1000)
    await s.load()
    expect(s.list()).toEqual([])
    const a = await s.save({ name: 'My Model', source: { kind: 'local', modelId: 'm/a.gguf' } })
    expect(a).toEqual({ id: 'vast/my-model', name: 'My Model', source: { kind: 'local', modelId: 'm/a.gguf' }, createdAt: 1000 })
    const b = await s.save({ name: 'My Model', source: hf() })
    expect(b.id).toBe('vast/my-model-2')
    const renamed = await s.save({ id: a.id, name: 'Renamed', source: { kind: 'local', modelId: 'm/b.gguf' } })
    expect(renamed.id).toBe(a.id)
    expect(renamed.createdAt).toBe(1000)
    expect(s.get(a.id)?.name).toBe('Renamed')
    const again = new RemoteModelStore(path)
    await again.load()
    expect(again.list().map((d) => d.id)).toEqual(['vast/my-model', 'vast/my-model-2'])
    expect(await s.delete(a.id)).toBe(true)
    expect(await s.delete(a.id)).toBe(false)
    const third = new RemoteModelStore(path)
    await third.load()
    expect(third.list().map((d) => d.id)).toEqual(['vast/my-model-2'])
  })

  it('returns copies, so callers cannot change what is saved', async () => {
    const s = new RemoteModelStore(tmp())
    const a = await s.save({ name: 'x', source: hf() })
    a.name = 'changed'
    ;(s.list()[0].source as { file: string }).file = 'other.gguf'
    expect(s.get(a.id)?.name).toBe('x')
    expect((s.get(a.id)?.source as { file: string }).file).toBe('Qwen3-Q4_K_M.gguf')
  })

  it('refuses a bad definition and an id that does not fit', async () => {
    const s = new RemoteModelStore(tmp())
    await expect(s.save({ name: '', source: hf() })).rejects.toThrow(/name/)
    await expect(s.save({ name: 'x'.repeat(200), source: hf() })).rejects.toThrow(/too long/)
    await expect(s.save({ name: 'x', source: hf({ file: '../a.gguf' }) })).rejects.toThrow()
    await expect(s.save({ id: 'vast/../x', name: 'x', source: hf() })).rejects.toThrow(/id/)
    await expect(s.save({ id: 'library/a', name: 'x', source: hf() })).rejects.toThrow(/id/)
    await expect(s.save({ id: 'vast/Upper', name: 'x', source: hf() })).rejects.toThrow(/id/)
    expect(s.list()).toEqual([])
  })

  it('drops entries on load that do not validate, and survives a broken file', async () => {
    const path = tmp()
    writeFileSync(
      path,
      JSON.stringify({
        version: 1,
        models: [
          { id: 'vast/ok', name: 'ok', source: { kind: 'local', modelId: 'a.gguf' }, createdAt: 5 },
          { id: 'vast/bad', name: 'bad', source: { kind: 'hf', repo: '../x', file: 'a.gguf' }, createdAt: 5 },
          { id: '../evil', name: 'evil', source: { kind: 'local', modelId: 'a.gguf' }, createdAt: 5 }
        ]
      })
    )
    const s = new RemoteModelStore(path)
    await s.load()
    expect(s.list().map((d) => d.id)).toEqual(['vast/ok'])
    const broken = tmp()
    writeFileSync(broken, '{not json')
    const b = new RemoteModelStore(broken)
    await b.load()
    expect(b.list()).toEqual([])
    await b.save({ name: 'after', source: hf() })
    expect(JSON.parse(readFileSync(broken, 'utf8')).models).toHaveLength(1)
  })
})

const localModel: ModelEntry = {
  id: 'fam/m-00001-of-00002.gguf',
  name: 'M',
  folder: 'fam',
  path: 'C:\\LLM\\fam\\m-00001-of-00002.gguf',
  files: ['C:\\LLM\\fam\\m-00001-of-00002.gguf', 'C:\\LLM\\fam\\m-00002-of-00002.gguf'],
  sizeBytes: 3000,
  mmproj: { path: 'C:\\LLM\\fam\\mmproj-M-F16.gguf', sizeBytes: 500 },
  draft: { path: 'C:\\LLM\\fam\\m-draft.gguf', sizeBytes: 200, kind: 'eagle3' },
  draftSidecar: { path: 'C:\\LLM\\fam\\m-draft.gguf', sizeBytes: 200, kind: 'eagle3' },
  info: { architecture: 'qwen3', hasChatTemplate: true },
  modifiedAt: 99
}
const sizes: Record<string, number> = {
  'C:\\LLM\\fam\\m-00001-of-00002.gguf': 1500,
  'C:\\LLM\\fam\\m-00002-of-00002.gguf': 1500,
  'C:\\LLM\\adapters\\style.gguf': 40,
  'C:\\LLM\\other\\style.gguf': 41
}
const localDef: RemoteModelDef = { id: 'vast/m', name: 'M on vast', source: { kind: 'local', modelId: localModel.id }, createdAt: 7 }
const deps = {
  getModel: (id: string): ModelEntry | undefined => (id === localModel.id ? localModel : undefined),
  localSize: (p: string): number | null => sizes[p] ?? null
}

describe('toModelEntry', () => {
  it('copies information, sizes and sidecars from a library model and marks it remote in the vast.ai folder', () => {
    const e = toModelEntry(localDef, deps.getModel)
    expect(e).toMatchObject({ id: 'vast/m', name: 'M on vast', folder: 'vast.ai', remote: true, sizeBytes: 3000, files: localModel.files, info: { architecture: 'qwen3', hasChatTemplate: true }, modifiedAt: 99 })
    expect(e.mmproj).toEqual(localModel.mmproj)
    expect(e.draft).toEqual(localModel.draft)
    expect(e.error).toBeUndefined()
    e.files.push('x')
    expect(localModel.files).toHaveLength(2)
  })

  it('says clearly when the library model is gone', () => {
    const e = toModelEntry(localDef, () => undefined)
    expect(e.remote).toBe(true)
    expect(e.info).toBeNull()
    expect(e.error).toContain('no longer in the library')
  })

  it('has no information for a Hugging Face model', () => {
    const e = toModelEntry({ id: 'vast/h', name: 'H', source: hf(), createdAt: 3 }, deps.getModel)
    expect(e).toMatchObject({ id: 'vast/h', folder: 'vast.ai', remote: true, info: null, sizeBytes: 0, files: [] })
    expect(e.error).toBeUndefined()
  })
})

describe('Hugging Face addresses', () => {
  it('builds the address of a model repo file with encoded path segments', () => {
    expect(hfUrl(hf(), 'Qwen3-Q4_K_M.gguf')).toBe('https://huggingface.co/unsloth/Qwen3-GGUF/resolve/main/Qwen3-Q4_K_M.gguf')
    expect(hfUrl(hf(), 'Q4 K/a b#c?.gguf')).toBe('https://huggingface.co/unsloth/Qwen3-GGUF/resolve/main/Q4%20K/a%20b%23c%3F.gguf')
    expect(hfUrl(hf({ revision: 'refs/pr/1' }), 'a.gguf')).toBe('https://huggingface.co/unsloth/Qwen3-GGUF/resolve/refs%2Fpr%2F1/a.gguf')
  })

  it('builds the address of a bucket file', () => {
    expect(hfUrl(hf({ repo: 'me/store', bucket: true }), 'models/a b.gguf')).toBe('https://huggingface.co/buckets/me/store/resolve/models/a%20b.gguf')
  })

  it('expands a split model from any of its parts', () => {
    expect(expandShards('Q4/m-00002-of-00003.gguf')).toEqual(['Q4/m-00001-of-00003.gguf', 'Q4/m-00002-of-00003.gguf', 'Q4/m-00003-of-00003.gguf'])
    expect(expandShards('m.gguf')).toEqual(['m.gguf'])
    expect(expandShards('m-00001-of-00000.gguf')).toEqual(['m-00001-of-00000.gguf'])
    expect(expandShards('m-00001-of-99999.gguf')).toEqual(['m-00001-of-99999.gguf'])
  })
})

describe('fileList', () => {
  it('lists a Hugging Face model with its parts and projector as downloads with safe names', () => {
    const files = fileList({ id: 'vast/h', name: 'H', source: hf({ file: 'Q4/m-00001-of-00002.gguf', mmproj: 'mmproj/mm-F16.gguf' }), createdAt: 1 }, deps)
    expect(files.map((f) => [f.role, f.name])).toEqual([
      ['main', 'm-00001-of-00002.gguf'],
      ['main', 'm-00002-of-00002.gguf'],
      ['mmproj', 'mm-F16.gguf']
    ])
    expect(files[1].url).toBe('https://huggingface.co/unsloth/Qwen3-GGUF/resolve/main/Q4/m-00002-of-00002.gguf')
    expect(files.every((f) => f.localPath === null && f.size === null)).toBe(true)
    for (const f of files) expect(FILE_NAME_RE.test(f.name)).toBe(true)
  })

  it('renames a projector that would collide with the main file', () => {
    const files = fileList({ id: 'vast/h', name: 'H', source: hf({ file: 'a/m.gguf', mmproj: 'b/m.gguf' }), createdAt: 1 }, deps)
    expect(files.map((f) => f.name)).toEqual(['m.gguf', 'mmproj-m.gguf'])
  })

  it('lists a library model with its parts, projector and draft as uploads, with sizes', () => {
    const files = fileList(localDef, deps)
    expect(files.map((f) => [f.role, f.name, f.size, f.localPath])).toEqual([
      ['main', 'm-00001-of-00002.gguf', 1500, 'C:\\LLM\\fam\\m-00001-of-00002.gguf'],
      ['main', 'm-00002-of-00002.gguf', 1500, 'C:\\LLM\\fam\\m-00002-of-00002.gguf'],
      ['mmproj', 'mmproj-M-F16.gguf', 500, 'C:\\LLM\\fam\\mmproj-M-F16.gguf'],
      ['draft', 'm-draft.gguf', 200, 'C:\\LLM\\fam\\m-draft.gguf']
    ])
    expect(files.every((f) => f.url === null)).toBe(true)
  })

  it('adds the adapters of the load config and leaves a draft out when it is off', () => {
    const adapters: Record<string, AdapterEntry> = {
      'adapters/style.gguf': { id: 'adapters/style.gguf', name: 'style', path: 'C:\\LLM\\adapters\\style.gguf', sizeBytes: 40, architecture: 'qwen3', info: null, modifiedAt: 1 },
      'other/style.gguf': { id: 'other/style.gguf', name: 'style', path: 'C:\\LLM\\other\\style.gguf', sizeBytes: 41, architecture: 'qwen3', info: null, modifiedAt: 1 }
    }
    const config: LoadConfig = { ...DEFAULT_LOAD_CONFIG, useDraft: false, adapters: [{ id: 'adapters/style.gguf', scale: 1 }, { id: 'other/style.gguf', scale: 0.5 }, { id: 'gone.gguf', scale: 1 }] }
    const files = fileList(localDef, { ...deps, getAdapter: (id) => adapters[id], config })
    expect(files.map((f) => [f.role, f.name])).toEqual([
      ['main', 'm-00001-of-00002.gguf'],
      ['main', 'm-00002-of-00002.gguf'],
      ['mmproj', 'mmproj-M-F16.gguf'],
      ['adapter', 'style.gguf'],
      ['adapter', 'adapter-style.gguf']
    ])
    expect(new Set(files.map((f) => f.name)).size).toBe(files.length)
  })

  it('uses a draft file chosen in the config and not the built-in one', () => {
    const custom = fileList(localDef, { ...deps, config: { ...DEFAULT_LOAD_CONFIG, draftFile: 'C:\\LLM\\drafts\\other-draft.gguf' }, localSize: () => 9 })
    expect(custom.find((f) => f.role === 'draft')).toMatchObject({ name: 'other-draft.gguf', localPath: 'C:\\LLM\\drafts\\other-draft.gguf', size: 9 })
    const builtin = fileList(localDef, { ...deps, config: { ...DEFAULT_LOAD_CONFIG, draftFile: 'builtin' } })
    expect(builtin.some((f) => f.role === 'draft')).toBe(false)
  })

  it('names files so they pass the file name rule whatever the local name', () => {
    const odd: ModelEntry = { ...localModel, id: 'odd', files: ["C:\\LLM\\it's a (model) é.gguf"], path: "C:\\LLM\\it's a (model) é.gguf", mmproj: undefined, draft: undefined, draftSidecar: undefined }
    const files = fileList({ id: 'vast/o', name: 'o', source: { kind: 'local', modelId: 'odd' }, createdAt: 1 }, { getModel: () => odd, localSize: () => 1 })
    expect(files[0].name).toMatch(FILE_NAME_RE)
  })

  it('explains a library model that is gone', () => {
    expect(() => fileList(localDef, { ...deps, getModel: () => undefined })).toThrow(/no longer in the library/)
  })
})

describe('buildRemoteArgs', () => {
  const remote = toModelEntry(localDef, deps.getModel)
  const paths = { main: '/workspace/rulz-llm/models/m/m-00001-of-00002.gguf', mmproj: '/workspace/rulz-llm/models/m/mmproj-M-F16.gguf', draft: '/workspace/rulz-llm/models/m/m-draft.gguf', adapters: ['/workspace/rulz-llm/models/m/style.gguf'] }
  const cfg = (over: Partial<LoadConfig> = {}): LoadConfig => ({ ...DEFAULT_LOAD_CONFIG, ...over })
  const build = (config: LoadConfig, extra: Partial<Parameters<typeof buildRemoteArgs>[0]> = {}): string[] =>
    buildRemoteArgs({ model: remote, config, port: '@RZ_PORT@', alias: 'qwen-remote', paths, ...extra })
  const flag = (args: string[], name: string): string | undefined => args[args.indexOf(name) + 1]

  it('uses paths on the instance, listens on 127.0.0.1 and leaves the port to the placeholder', () => {
    const a = build(cfg())
    expect(flag(a, '--model')).toBe(paths.main)
    expect(flag(a, '--host')).toBe('127.0.0.1')
    expect(flag(a, '--port')).toBe('@RZ_PORT@')
    expect(flag(a, '--alias')).toBe('qwen-remote')
    expect(a.join(' ')).not.toContain('C:\\')
  })

  it('takes the projector, draft and adapters from their remote paths', () => {
    const a = build(cfg({ useMmproj: true, useDraft: true }), { draft: localModel.draft })
    expect(flag(a, '--mmproj')).toBe(paths.mmproj)
    expect(flag(a, '--spec-draft-model')).toBe(paths.draft)
    expect(flag(a, '--spec-type')).toBe('draft-eagle3')
    const lora = buildRemoteArgs({ model: remote, config: cfg(), port: 1234, alias: 'x', paths })
    expect(flag(lora, '--lora')).toBe(paths.adapters[0])
    expect(flag(lora, '--port')).toBe('1234')
  })

  it('uses the model file for a built-in draft', () => {
    const a = build(cfg({ useDraft: true }), { draft: { path: 'C:\\x.gguf', sizeBytes: 0, kind: 'mtp', builtIn: true } })
    expect(a).toContain('draft-mtp')
    expect(a).not.toContain('--spec-draft-model')
  })

  it('drops the thread count, the memory lock and the main GPU index; keeps a custom split', () => {
    const a = build(cfg({ threads: 12, useMlock: true, splitMode: 'single', mainGpu: 1 }))
    expect(a).not.toContain('--threads')
    expect(a).not.toContain('--mlock')
    expect(a).not.toContain('--main-gpu')
    expect(flag(a, '--split-mode')).toBe('none')
    const b = build(cfg({ splitMode: 'custom', tensorSplit: '3, 1' }))
    expect(flag(b, '--split-mode')).toBe('layer')
    expect(flag(b, '--tensor-split')).toBe('3,1')
  })

  it('keeps the settings that mean the same there', () => {
    const a = build(cfg({ contextLength: 8192, flashAttention: 'on', parallel: 4, extraArgs: '--no-warmup' }))
    expect(flag(a, '--ctx-size')).toBe('8192')
    expect(flag(a, '--flash-attn')).toBe('on')
    expect(flag(a, '--parallel')).toBe('4')
    expect(a[a.length - 1]).toBe('--no-warmup')
  })

  it('refuses a local chat template file with a sentence', () => {
    expect(() => build(cfg({ chatTemplateFile: 'C:\\t\\tpl.jinja' }))).toThrow(/chat template file/)
  })

  it('refuses extra arguments that name this PC', () => {
    for (const extra of ['--host 0.0.0.0', '--port 9000', '--port=9000', '-m x.gguf', '--model=/x', '--alias z', '--path C:\\web', '--lora C:\\a.gguf', '--chat-template-file D:/t.jinja', '--x=\\\\server\\share\\f', '--grammar-file ~/g.gbnf', '--spec-draft-model "C:\\My Files\\d.gguf"']) {
      expect(extraArgsProblem(extra), extra).toBeTruthy()
      expect(() => build(cfg({ extraArgs: extra })), extra).toThrow()
    }
    for (const fine of ['', '--no-warmup', '--rope-freq-base 1000000', '--override-kv tokenizer.ggml.add_bos_token=bool:false', '--grammar-file /workspace/g.gbnf']) expect(extraArgsProblem(fine), fine).toBeUndefined()
  })

  it('refuses a projector or draft that is not on the instance', () => {
    expect(() => build(cfg({ useMmproj: true }), { paths: { ...paths, mmproj: undefined } })).toThrow(/projector/)
    expect(() => build(cfg({ useDraft: true }), { draft: localModel.draft, paths: { ...paths, draft: undefined } })).toThrow(/draft/)
  })
})

describe('a Hugging Face projector', () => {
  it('is on the model entry with an unknown size, and not in a model without one', () => {
    const withMm = toModelEntry({ id: 'vast/h', name: 'H', source: hf({ mmproj: 'mm/mmproj-F16.gguf' }), createdAt: 1 }, deps.getModel)
    expect(withMm.mmproj).toEqual({ path: 'hf:unsloth/Qwen3-GGUF/mm/mmproj-F16.gguf', sizeBytes: 0 })
    expect(toModelEntry({ id: 'vast/h', name: 'H', source: hf(), createdAt: 1 }, deps.getModel).mmproj).toBeUndefined()
  })

  it('makes buildRemoteArgs pass --mmproj with the file on the instance when the projector is on', () => {
    const e = toModelEntry({ id: 'vast/h', name: 'H', source: hf({ mmproj: 'mm/mmproj-F16.gguf' }), createdAt: 1 }, deps.getModel)
    const paths = { main: '/w/m/a.gguf', mmproj: '/w/m/mmproj-F16.gguf', adapters: [] }
    const on = buildRemoteArgs({ model: e, config: { ...DEFAULT_LOAD_CONFIG, useMmproj: true }, port: 1, alias: 'a', paths })
    expect(on[on.indexOf('--mmproj') + 1]).toBe('/w/m/mmproj-F16.gguf')
    expect(buildRemoteArgs({ model: e, config: { ...DEFAULT_LOAD_CONFIG, useMmproj: false }, port: 1, alias: 'a', paths })).not.toContain('--mmproj')
  })
})
