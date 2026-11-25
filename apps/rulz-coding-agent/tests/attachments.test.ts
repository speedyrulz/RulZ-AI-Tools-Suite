import { afterEach, describe, expect, it, vi } from 'vitest'
import { promises as fsp } from 'node:fs'
import { mkdir, mkdtemp, readdir, readFile, rm, stat, truncate, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { AttachmentStore } from '../src/main/agent/attachments'
import { projectKey } from '../src/main/project/store'
import type { PdfReader } from '../src/main/pdf/reader'

const CHUNK = 524288
const dirs: string[] = []
afterEach(async () => {
  vi.restoreAllMocks()
  await Promise.all(dirs.splice(0).map((d) => rm(d, { recursive: true, force: true })))
})
async function setup(options: ConstructorParameters<typeof AttachmentStore>[2] = {}) {
  const dir = await mkdtemp(join(tmpdir(), 'attachments-')); dirs.push(dir)
  const reader: PdfReader = { run: async () => ({ pageCount: 3, title: '', pages: [] }) }
  return { dir, store: new AttachmentStore(dir, reader, options) }
}
const folderOf = (dir: string, project: string): string => join(dir, 'attachments', projectKey(project))
const ls = (path: string): Promise<string[]> => readdir(path).catch(() => [] as string[])
/** A fake PDF of the given size: starts with the signature, the rest is a repeating pattern. */
function fakePdf(size: number): Buffer {
  const b = Buffer.alloc(size)
  for (let i = 0; i < size; i++) b[i] = (i * 31 + 7) & 0xff
  Buffer.from('%PDF-1.4\n').copy(b)
  return b
}
async function upload(store: AttachmentStore, project: string, name: string, data: Buffer, owner: unknown = 'owner') {
  const { uploadId } = await store.begin(project, name, data.length, owner)
  for (let i = 0, n = 0; i < data.length; i += CHUNK, n++) await store.chunk(uploadId, n, data.subarray(i, i + CHUNK).toString('base64'), owner)
  return store.end(uploadId, owner)
}
const b64 = (s: string | Buffer): string => Buffer.from(s).toString('base64')

describe('attachment storage', () => {
  it('stores a validated PDF in its project and rejects traversal', async () => {
    const { store } = await setup()
    const data = Buffer.from('%PDF- test')
    const { uploadId } = await store.begin('project-a', '../CON.pdf', data.length, 1)
    await store.chunk(uploadId, 0, data.toString('base64'), 1)
    const item = await store.end(uploadId, 1)
    expect(item.name).toBe('attachment.pdf')
    expect((await store.resolve('project-a', item.id))?.pages).toBe(3)
    expect(await store.resolve('project-b', item.id)).toBeNull()
    expect(await store.resolve('project-a', '../escape')).toBeNull()
    expect(await store.list('project-a')).toEqual([item])
    await store.remove('project-a', item.id)
    expect(await store.list('project-a')).toEqual([])
  })
  it('removes partial files on invalid chunks and enforces ownership', async () => {
    const { dir, store } = await setup()
    const { uploadId } = await store.begin('p', 'a.pdf', 5, 'owner')
    await expect(store.chunk(uploadId, 0, 'JVBERi0=', 'other')).rejects.toThrow()
    await expect(store.chunk(uploadId, 1, 'JVBERi0=', 'owner')).rejects.toThrow()
    const projects = await readdir(join(dir, 'attachments'))
    expect(await readdir(join(dir, 'attachments', projects[0]))).toEqual([])
  })
  it('checks source files and refuses invalid PDFs', async () => {
    const { dir, store } = await setup()
    const file = join(dir, 'wrong.pdf'); await writeFile(file, 'hello')
    await expect(store.attachPath('p', file)).rejects.toThrow('not a PDF')
    expect(await store.list('p')).toEqual([])
  })
})

describe('chunked uploads', () => {
  it('round-trips a PDF of several full chunks byte for byte', async () => {
    const { dir, store } = await setup()
    const data = fakePdf(CHUNK * 3 + 1234)
    const item = await upload(store, 'p', 'big.pdf', data)
    expect(item).toMatchObject({ name: 'big.pdf', pages: 3, bytes: data.length })
    const found = await store.resolve('p', item.id)
    expect(Buffer.compare(await readFile(found!.path), data)).toBe(0)
    // Nothing but the attachment is left behind.
    expect(await ls(folderOf(dir, 'p'))).toEqual([item.id])
    expect((await ls(join(folderOf(dir, 'p'), item.id))).sort()).toEqual(['attachment.json', 'big.pdf'])
  })
  it('round-trips a PDF that is exactly one or two full chunks', async () => {
    const { store } = await setup()
    for (const size of [CHUNK, CHUNK * 2]) {
      const data = fakePdf(size)
      const item = await upload(store, 'p', `x${size}.pdf`, data)
      expect(Buffer.compare(await readFile((await store.resolve('p', item.id))!.path), data)).toBe(0)
    }
  })
  it('hands the real bytes to the reader', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'attachments-')); dirs.push(dir)
    const seen: Buffer[] = []
    const reader: PdfReader = { run: async (d) => (seen.push(Buffer.from(d)), { pageCount: 7, title: 'T', pages: [] }) }
    const store = new AttachmentStore(dir, reader)
    const data = fakePdf(CHUNK + 10)
    const item = await upload(store, 'p', 'a.pdf', data)
    expect(item.pages).toBe(7)
    expect(seen).toHaveLength(1)
    expect(Buffer.compare(seen[0], data)).toBe(0)
  })
  it('refuses an announced size of 0, a fraction, a negative one and one over 50 MB, before creating anything', async () => {
    const { dir, store } = await setup()
    for (const n of [0, -1, 1.5, NaN, 50 * 1024 * 1024 + 1]) await expect(store.begin('p', 'a.pdf', n, 'o')).rejects.toThrow(/between 1 byte and 50 MB/)
    expect(await ls(folderOf(dir, 'p'))).toEqual([])
    const { uploadId } = await store.begin('p', 'a.pdf', 50 * 1024 * 1024, 'o')
    await store.cancel(uploadId, 'o')
  })

  describe('bad chunks drop the upload and leave no file', () => {
    const bad = async (label: string, size: number, run: (store: AttachmentStore, id: string) => Promise<unknown>, message: RegExp) => {
      const { dir, store } = await setup()
      const { uploadId } = await store.begin('p', 'a.pdf', size, 'o')
      expect(await ls(folderOf(dir, 'p')), label).toHaveLength(1)
      await expect(run(store, uploadId), label).rejects.toThrow(message)
      expect(await ls(folderOf(dir, 'p')), label).toEqual([])
      // The upload is gone: even a good chunk is refused now.
      await expect(store.chunk(uploadId, 0, b64('abc'), 'o'), label).rejects.toThrow('Upload not found.')
      // And the owner may start a new one.
      const again = await store.begin('p', 'a.pdf', 3, 'o')
      await store.cancel(again.uploadId, 'o')
    }
    it('a chunk out of order', async () => {
      await bad('skipped', 100, (s, id) => s.chunk(id, 1, b64('abc'), 'o'), /Invalid upload chunk/)
    })
    it('a chunk sent twice', async () => {
      await bad('replayed', 100, async (s, id) => { await s.chunk(id, 0, b64('abc'), 'o'); await s.chunk(id, 0, b64('abc'), 'o') }, /Invalid upload chunk/)
    })
    it('a chunk over 512 KB', async () => {
      await bad('too large', CHUNK * 2, (s, id) => s.chunk(id, 0, b64(Buffer.alloc(CHUNK + 1, 1)), 'o'), /Invalid upload chunk/)
    })
    it('a chunk that is not valid base64', async () => {
      await bad('chars', 100, (s, id) => s.chunk(id, 0, 'not*base64!', 'o'), /Invalid upload chunk/)
      await bad('no padding', 100, (s, id) => s.chunk(id, 0, 'JVBERi0', 'o'), /Invalid upload chunk/)
      await bad('non-canonical', 100, (s, id) => s.chunk(id, 0, 'JVBERi1=', 'o'), /Invalid upload chunk/)
      await bad('url-safe', 100, (s, id) => s.chunk(id, 0, '-_-_', 'o'), /Invalid upload chunk/)
      await bad('not text', 100, (s, id) => s.chunk(id, 0, 42 as never, 'o'), /Invalid upload chunk/)
    })
    it('an empty chunk', async () => {
      await bad('empty', 100, (s, id) => s.chunk(id, 0, '', 'o'), /Invalid upload chunk/)
    })
    it('a chunk past the announced size', async () => {
      await bad('one chunk', 5, (s, id) => s.chunk(id, 0, b64('123456'), 'o'), /Invalid upload chunk size/)
      await bad('two chunks', 5, async (s, id) => { await s.chunk(id, 0, b64('123'), 'o'); await s.chunk(id, 1, b64('456'), 'o') }, /Invalid upload chunk size/)
    })
    it('an end before all the bytes are there', async () => {
      await bad('short', 10, async (s, id) => { await s.chunk(id, 0, b64('%PDF-'), 'o'); await s.end(id, 'o') }, /incomplete/)
      await bad('nothing', 10, (s, id) => s.end(id, 'o'), /incomplete/)
    })
    it('bytes that are not a PDF at the end', async () => {
      await bad('text', 5, async (s, id) => { await s.chunk(id, 0, b64('hello'), 'o'); await s.end(id, 'o') }, /not a PDF/)
    })
  })

  it('lets a new upload take the place of the same owner\'s earlier one', async () => {
    const { dir, store } = await setup()
    const a = await store.begin('p', 'a.pdf', 10, 'one')
    await store.chunk(a.uploadId, 0, b64('%PDF-'), 'one')
    const b = await store.begin('p', 'b.pdf', 10, 'one')
    expect(await ls(folderOf(dir, 'p'))).toHaveLength(1)
    await expect(store.chunk(a.uploadId, 1, b64('12345'), 'one')).rejects.toThrow('Upload not found.')
    await store.chunk(b.uploadId, 0, b64('%PDF-12345'), 'one')
    expect((await store.end(b.uploadId, 'one')).name).toBe('b.pdf')
  })
  it('refuses a third upload in total', async () => {
    const { store } = await setup()
    const a = await store.begin('p', 'a.pdf', 10, 'one')
    const b = await store.begin('q', 'b.pdf', 10, 'two')
    await expect(store.begin('r', 'c.pdf', 10, 'three')).rejects.toThrow(/Too many/)
    // The first two are not harmed, and finishing or cancelling one frees a place.
    await store.chunk(a.uploadId, 0, b64('%PDF-12345'), 'one')
    await store.cancel(b.uploadId, 'two')
    const c = await store.begin('r', 'c.pdf', 10, 'three')
    await store.cancel(c.uploadId, 'three')
    expect((await store.end(a.uploadId, 'one')).bytes).toBe(10)
    // The owner is free again after finishing.
    const again = await store.begin('p', 'd.pdf', 10, 'one')
    await store.cancel(again.uploadId, 'one')
  })
  it('refuses a chunk, end or cancel for another owner and leaves the upload alone', async () => {
    const { dir, store } = await setup()
    const { uploadId } = await store.begin('p', 'a.pdf', 10, 'owner')
    await expect(store.chunk(uploadId, 0, b64('%PDF-12345'), 'thief')).rejects.toThrow('Upload not found.')
    await expect(store.end(uploadId, 'thief')).rejects.toThrow('Upload not found.')
    await expect(store.cancel(uploadId, 'thief')).rejects.toThrow('Upload not found.')
    await expect(store.chunk('0123456789abcdef', 0, b64('x'), 'owner')).rejects.toThrow('Upload not found.')
    expect(await ls(folderOf(dir, 'p'))).toHaveLength(1)
    await store.chunk(uploadId, 0, b64('%PDF-12345'), 'owner')
    expect((await store.end(uploadId, 'owner')).bytes).toBe(10)
  })
  it('does not accept two chunks at the same time on one upload', async () => {
    const { store } = await setup()
    const { uploadId } = await store.begin('p', 'a.pdf', 20, 'o')
    const first = store.chunk(uploadId, 0, b64('%PDF-12345'), 'o')
    const second = store.chunk(uploadId, 1, b64('1234567890'), 'o')
    await expect(second).rejects.toThrow('Upload is busy.')
    await first
    // The busy refusal did not drop the upload.
    await store.chunk(uploadId, 1, b64('1234567890'), 'o')
    expect((await store.end(uploadId, 'o')).bytes).toBe(20)
  })
  it('compares owners by identity, not by value', async () => {
    const { store } = await setup()
    const { uploadId } = await store.begin('p', 'a.pdf', 10, { id: 1 })
    await expect(store.chunk(uploadId, 0, b64('%PDF-12345'), { id: 1 })).rejects.toThrow('Upload not found.')
  })

  describe('the 2-minute timeout', () => {
    it('removes the partial file when the timer fires', async () => {
      const timers: { fn: () => void; ms: number; cleared: boolean }[] = []
      const { dir, store } = await setup({
        setTimeout: (fn, ms) => { const t = { fn, ms, cleared: false }; timers.push(t); return t as never },
        clearTimeout: (t) => { (t as never as { cleared: boolean }).cleared = true }
      })
      const { uploadId } = await store.begin('p', 'a.pdf', 10, 'o')
      expect(timers).toHaveLength(1)
      expect(timers[0].ms).toBe(120000)
      await store.chunk(uploadId, 0, b64('%PDF-'), 'o')
      // Every chunk starts the clock again.
      expect(timers).toHaveLength(2)
      expect(timers[0].cleared).toBe(true)
      expect(await ls(folderOf(dir, 'p'))).toHaveLength(1)
      timers[1].fn()
      await vi.waitFor(async () => expect(await ls(folderOf(dir, 'p'))).toEqual([]))
      await expect(store.chunk(uploadId, 1, b64('12345'), 'o')).rejects.toThrow('Upload not found.')
      // The place is free again.
      await store.cancel((await store.begin('p', 'b.pdf', 3, 'o')).uploadId, 'o')
    })
    it('refuses and removes an upload that has been quiet for 2 minutes even if the timer never ran', async () => {
      let now = 1_000_000
      const { dir, store } = await setup({ now: () => now, setTimeout: () => ({}) as never, clearTimeout: () => undefined })
      const { uploadId } = await store.begin('p', 'a.pdf', 10, 'o')
      now += 119_999
      await store.chunk(uploadId, 0, b64('%PDF-'), 'o')
      now += 120_000
      await expect(store.chunk(uploadId, 1, b64('12345'), 'o')).rejects.toThrow('Upload expired.')
      await vi.waitFor(async () => expect(await ls(folderOf(dir, 'p'))).toEqual([]))
      await expect(store.end(uploadId, 'o')).rejects.toThrow('Upload not found.')
    })
    it('does not fire after the upload has finished or been cancelled', async () => {
      const cleared: unknown[] = []
      const { store } = await setup({ setTimeout: () => ({ t: 1 }) as never, clearTimeout: (t) => { cleared.push(t) } })
      const { uploadId } = await store.begin('p', 'a.pdf', 5, 'o')
      await store.cancel(uploadId, 'o')
      expect(cleared.length).toBeGreaterThan(0)
    })
  })

  it('cancel removes the partial file and frees the owner', async () => {
    const { dir, store } = await setup()
    const { uploadId } = await store.begin('p', 'a.pdf', 10, 'o')
    await store.chunk(uploadId, 0, b64('%PDF-'), 'o')
    await store.cancel(uploadId, 'o')
    expect(await ls(folderOf(dir, 'p'))).toEqual([])
    await expect(store.cancel(uploadId, 'o')).rejects.toThrow('Upload not found.')
    await store.cancel((await store.begin('p', 'a.pdf', 10, 'o')).uploadId, 'o')
  })
  it('dropProject removes only that project\'s uploads', async () => {
    const { dir, store } = await setup()
    const a = await store.begin('proj-a', 'a.pdf', 10, 'one')
    const b = await store.begin('proj-b', 'b.pdf', 10, 'two')
    await store.dropProject('proj-a')
    expect(await ls(folderOf(dir, 'proj-a'))).toEqual([])
    expect(await ls(folderOf(dir, 'proj-b'))).toHaveLength(1)
    await expect(store.chunk(a.uploadId, 0, b64('%PDF-12345'), 'one')).rejects.toThrow('Upload not found.')
    await store.chunk(b.uploadId, 0, b64('%PDF-12345'), 'two')
    expect((await store.end(b.uploadId, 'two')).bytes).toBe(10)
    await store.dropProject('never-used')
  })
  it('keeps stored attachments when their project is dropped', async () => {
    const { store } = await setup()
    const item = await upload(store, 'p', 'a.pdf', fakePdf(100))
    await store.dropProject('p')
    expect(await store.list('p')).toEqual([item])
  })
})

describe('attachPath', () => {
  it('copies a PDF from a path and leaves the source untouched', async () => {
    const { dir, store } = await setup()
    const data = fakePdf(CHUNK + 5000)
    const src = join(dir, 'source folder', 'My Report.pdf')
    await mkdir(join(dir, 'source folder'))
    await writeFile(src, data)
    const before = await stat(src)
    const item = await store.attachPath('p', src)
    expect(item).toMatchObject({ name: 'My Report.pdf', bytes: data.length, pages: 3 })
    expect(Buffer.compare(await readFile((await store.resolve('p', item.id))!.path), data)).toBe(0)
    expect(Buffer.compare(await readFile(src), data)).toBe(0)
    const after = await stat(src)
    expect(after.mtimeMs).toBe(before.mtimeMs)
    expect(after.size).toBe(before.size)
    expect(await ls(folderOf(dir, 'p'))).toEqual([item.id])
  })
  it('refuses a file over 50 MB before reading it', async () => {
    const { dir, store } = await setup()
    const big = join(dir, 'big.pdf')
    await writeFile(big, '%PDF-')
    await truncate(big, 50 * 1024 * 1024 + 1)
    const read = vi.spyOn(fsp, 'readFile')
    await expect(store.attachPath('p', big)).rejects.toThrow(/up to 50 MB/)
    expect(read).not.toHaveBeenCalled()
    expect(await ls(folderOf(dir, 'p'))).toEqual([])
    expect((await stat(big)).size).toBe(50 * 1024 * 1024 + 1)
  })
  it('refuses an empty file, a folder, a missing path and a network path', async () => {
    const { dir, store } = await setup()
    const empty = join(dir, 'empty.pdf')
    await writeFile(empty, '')
    await expect(store.attachPath('p', empty)).rejects.toThrow(/regular PDF file/)
    await mkdir(join(dir, 'folder.pdf'))
    await expect(store.attachPath('p', join(dir, 'folder.pdf'))).rejects.toThrow(/regular PDF file/)
    await expect(store.attachPath('p', join(dir, 'missing.pdf'))).rejects.toThrow()
    await expect(store.attachPath('p', '\\\\host\\share\\a.pdf')).rejects.toThrow(/network or device path/)
    await expect(store.attachPath('p', '//host/share/a.pdf')).rejects.toThrow(/network or device path/)
    await expect(store.attachPath('p', '\\\\?\\C:\\a.pdf')).rejects.toThrow(/network or device path/)
    await expect(store.attachPath('p', `${join(dir, 'empty.pdf')}:stream`)).rejects.toThrow(/data stream/)
    expect(await ls(folderOf(dir, 'p'))).toEqual([])
    expect(await store.list('p')).toEqual([])
  })
  it('does not hold an upload place after a refusal', async () => {
    const { dir, store } = await setup()
    const bad = join(dir, 'bad.pdf')
    await writeFile(bad, 'plain text')
    for (let i = 0; i < 4; i++) await expect(store.attachPath('p', bad)).rejects.toThrow(/not a PDF/)
    expect((await store.attachPath('p', await (async () => { const g = join(dir, 'good.pdf'); await writeFile(g, fakePdf(50)); return g })())).bytes).toBe(50)
    expect(await ls(folderOf(dir, 'p'))).toHaveLength(1)
  })
})

describe('attachment names', () => {
  const RESERVED = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$/i
  const inputs = [
    'CON', 'con.pdf', 'NUL.txt', 'com1.PDF', 'LPT9', 'aux.x.pdf', 'prn .pdf', 'a\u0000b\u001f.pdf', 'tab\there.pdf', 'x'.repeat(300), `${'x'.repeat(300)}.pdf`,
    'noext', '..\\..\\evil.pdf', '../../evil', 'C:\\a\\b\\report.pdf', '/etc/passwd', '   ', '...', '..', '.', '.pdf', '.PDF', 'name. .pdf', 'a:b*c?"d<e>f|g.pdf',
    ' trailing dot.', `${'a'.repeat(115)} b.pdf`, `${'a'.repeat(114)}. .pdf`, '\u00e9'.repeat(200), 'report.PDF', 'a.pdf.pdf', 'dir/', 'dir\\', '', 'COM0', 'CONIN$'
  ]
  it('always end up as a safe .pdf of at most 120 characters', async () => {
    const { dir, store } = await setup()
    const seen: string[] = []
    for (const name of inputs) {
      const item = await upload(store, 'p', name, fakePdf(20))
      seen.push(item.name)
      const label = JSON.stringify(name)
      expect(item.name, label).toMatch(/\.pdf$/)
      expect(item.name.length, label).toBeLessThanOrEqual(120)
      expect(item.name, label).not.toMatch(/[\x00-\x1f\x7f<>:"/\\|?*]/)
      const base = item.name.slice(0, -4)
      expect(base, label).not.toMatch(RESERVED)
      expect(base.length, label).toBeGreaterThan(0)
      expect(base, label).not.toMatch(/[. ]$/)
      expect(base, label).not.toBe('')
      // It is stored inside its own attachment folder and is the only file there besides the metadata.
      expect((await ls(join(folderOf(dir, 'p'), item.id))).sort(), label).toEqual(['attachment.json', item.name].sort())
      expect((await store.resolve('p', item.id))?.name, label).toBe(item.name)
    }
    // Nothing was written outside the project's folder.
    expect(await ls(join(dir, 'attachments'))).toEqual([projectKey('p')])
    expect(await ls(dir)).toEqual(['attachments'])
    expect((await store.list('p')).length).toBe(inputs.length)
    const at = (n: string) => seen[inputs.indexOf(n)]
    expect(at('CON')).toBe('attachment.pdf')
    expect(at('con.pdf')).toBe('attachment.pdf')
    expect(at('NUL.txt')).toBe('attachment.pdf')
    expect(at('com1.PDF')).toBe('attachment.pdf')
    expect(at('..\\..\\evil.pdf')).toBe('evil.pdf')
    expect(at('../../evil')).toBe('evil.pdf')
    expect(at('C:\\a\\b\\report.pdf')).toBe('report.pdf')
    expect(at('/etc/passwd')).toBe('passwd.pdf')
    expect(at('noext')).toBe('noext.pdf')
    expect(at('report.PDF')).toBe('report.pdf')
    expect(at('x'.repeat(300))).toBe(`${'x'.repeat(116)}.pdf`)
    expect(at('..')).toBe('attachment.pdf')
    expect(at('')).toBe('attachment.pdf')
  })
})

describe('resolve, list and remove', () => {
  const tamper = async (dir: string, project: string, id: string, meta: string | object, extra?: { file?: string; data?: Buffer }) => {
    const folder = join(folderOf(dir, project), id)
    if (extra?.file) await writeFile(join(folder, extra.file), extra.data ?? Buffer.from('x'))
    await writeFile(join(folder, 'attachment.json'), typeof meta === 'string' ? meta : JSON.stringify(meta))
  }

  it('gives null for ids that are not 16 lowercase hex characters, and deletes nothing', async () => {
    const { dir, store } = await setup()
    const item = await upload(store, 'p', 'a.pdf', fakePdf(50))
    const canary = join(dir, 'canary.txt')
    await writeFile(canary, 'keep')
    const bad = ['..', '.', '', item.id.toUpperCase(), item.id.slice(0, 15), `${item.id}0`, `../${item.id}`, `${item.id}/..`, `..\\${item.id}`, join(dir, 'attachments'), `${item.id}\u0000`, 'zzzzzzzzzzzzzzzz', projectKey('p')]
    for (const id of bad) {
      expect(await store.resolve('p', id), JSON.stringify(id)).toBeNull()
      await store.remove('p', id)
    }
    expect(await store.list('p')).toEqual([item])
    expect(await readFile(canary, 'utf8')).toBe('keep')
    expect(await ls(join(dir, 'attachments'))).toEqual([projectKey('p')])
    expect(await store.resolve('p', item.id)).not.toBeNull()
  })
  it('does not find or remove another project\'s attachment', async () => {
    const { store } = await setup()
    const item = await upload(store, 'proj-a', 'a.pdf', fakePdf(50))
    expect(await store.resolve('proj-b', item.id)).toBeNull()
    expect(await store.list('proj-b')).toEqual([])
    await store.remove('proj-b', item.id)
    expect(await store.list('proj-a')).toEqual([item])
    await store.remove('proj-a', item.id)
    expect(await store.list('proj-a')).toEqual([])
    // Removing something that is already gone is fine.
    await store.remove('proj-a', item.id)
  })
  it('reads a project path with another spelling as the same project', async () => {
    const { store } = await setup()
    const item = await upload(store, 'C:\\Work\\App', 'a.pdf', fakePdf(50))
    expect(await store.resolve('c:\\work\\app', item.id)).not.toBeNull()
  })
  it('ignores a tampered attachment.json', async () => {
    const { dir, store } = await setup()
    const good = async () => { const i = await upload(store, 'p', 'a.pdf', fakePdf(50)); return i }
    const canary = join(dir, 'canary.pdf')
    await writeFile(canary, fakePdf(50))
    const cases: [string, (i: { id: string; name: string; pages: number; bytes: number }) => string | object][] = [
      ['wrong id', (i) => ({ ...i, id: '0123456789abcdef' })],
      ['name with path parts', (i) => ({ ...i, name: '../../canary.pdf' })],
      ['name with a backslash path', (i) => ({ ...i, name: '..\\..\\canary.pdf' })],
      ['absolute name', (i) => ({ ...i, name: canary })],
      ['name that is not a safe name', (i) => ({ ...i, name: 'CON.pdf' })],
      ['name without .pdf', (i) => ({ ...i, name: 'a.txt' })],
      ['name that is a number', (i) => ({ ...i, name: 5 })],
      ['bad JSON', () => '{ not json'],
      ['empty file', () => ''],
      ['JSON null', () => 'null'],
      ['JSON array', () => '[]'],
      ['pages 0', (i) => ({ ...i, pages: 0 })],
      ['pages a fraction', (i) => ({ ...i, pages: 1.5 })],
      ['pages as text', (i) => ({ ...i, pages: '3' })],
      ['bytes wrong', (i) => ({ ...i, bytes: i.bytes + 1 })],
      ['bytes 0', (i) => ({ ...i, bytes: 0 })],
      ['bytes over the limit', (i) => ({ ...i, bytes: 60 * 1024 * 1024 })],
      ['too large to be real', (i) => ({ ...i, junk: 'x'.repeat(5000) })]
    ]
    for (const [label, make] of cases) {
      const item = await good()
      await tamper(dir, 'p', item.id, make(item))
      expect(await store.resolve('p', item.id), label).toBeNull()
      expect((await store.list('p')).find((x) => x.id === item.id), label).toBeUndefined()
      await store.remove('p', item.id)
      expect(await ls(join(folderOf(dir, 'p'), item.id)), label).toEqual([])
      expect(Buffer.compare(await readFile(canary), fakePdf(50)), label).toBe(0)
    }
    expect(await store.list('p')).toEqual([])
  })
  it('ignores an attachment whose file is missing or changed', async () => {
    const { dir, store } = await setup()
    const a = await upload(store, 'p', 'a.pdf', fakePdf(50))
    await rm(join(folderOf(dir, 'p'), a.id, 'a.pdf'))
    expect(await store.resolve('p', a.id)).toBeNull()
    const b = await upload(store, 'p', 'b.pdf', fakePdf(50))
    await writeFile(join(folderOf(dir, 'p'), b.id, 'b.pdf'), fakePdf(51))
    expect(await store.resolve('p', b.id)).toBeNull()
    const c = await upload(store, 'p', 'c.pdf', fakePdf(50))
    await rm(join(folderOf(dir, 'p'), c.id, 'attachment.json'))
    expect(await store.resolve('p', c.id)).toBeNull()
    // A folder named like an id, without metadata, or a plain file named like one.
    await mkdir(join(folderOf(dir, 'p'), '0123456789abcdef'))
    await writeFile(join(folderOf(dir, 'p'), 'fedcba9876543210'), 'x')
    expect(await store.list('p')).toEqual([])
    expect(await store.resolve('p', 'fedcba9876543210')).toBeNull()
  })
  it('never lists a leftover .part file from a crash as an attachment', async () => {
    const { dir, store } = await setup()
    const item = await upload(store, 'p', 'a.pdf', fakePdf(50))
    const left = join(folderOf(dir, 'p'), '.0123456789abcdef.part')
    await writeFile(left, fakePdf(50))
    await mkdir(join(folderOf(dir, 'p'), '.fedcba9876543210.part'))
    expect(await store.list('p')).toEqual([item])
    expect(await store.resolve('p', '.0123456789abcdef.part')).toBeNull()
    expect(await store.resolve('p', '0123456789abcdef')).toBeNull()
    await store.remove('p', '.0123456789abcdef.part')
    expect(await readFile(left)).toHaveLength(50)
  })
  it('lists nothing for a project that has no attachments', async () => {
    const { store } = await setup()
    expect(await store.list('nothing')).toEqual([])
  })
})
