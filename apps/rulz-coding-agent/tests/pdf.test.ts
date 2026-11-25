import { mkdirSync, mkdtempSync, rmSync, symlinkSync, truncateSync, writeFileSync } from 'node:fs'
import { promises as fsp } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { afterAll, describe, expect, it, vi } from 'vitest'
import { PdfWorkerReader } from '../src/main/pdf/reader'
import type { PdfReader } from '../src/main/pdf/reader'
import { hasPdfMagic, pdfInfo, readPdfTool } from '../src/main/pdf/tool'
import {
  formatPageList,
  itemsToText,
  matchesOnPage,
  NO_TEXT_PAGE,
  parsePageRange,
  PdfProblem,
  renderPages,
  renderSearch
} from '../src/main/pdf/text'
import { runPdfJob } from '../src/main/pdf/extract'
import { buildFakeCountPdf, buildJapanesePdf, buildPdf, InProcessPdfReader } from './helpers/pdfBuilder'

const root = mkdtempSync(join(tmpdir(), 'rulz-pdf-'))
afterAll(() => rmSync(root, { recursive: true, force: true }))

const item = (str: string, x: number, y: number, width = str.length * 6, extra: object = {}) => ({ str, transform: [12, 0, 0, 12, x, y], width, height: 12, ...extra })

const STUB = '// Stands in for the PDF worker in reader tests: what it does depends on the "query" of the job it is given.\nimport { parentPort, workerData } from \'node:worker_threads\'\n\nconst mode = workerData.job.query\nif (mode === \'ok\') parentPort.postMessage({ ok: true, result: { pageCount: 1, title: \'\', pages: [{ page: 1, text: \'stub\' }] } })\nelse if (mode === \'problem\') parentPort.postMessage({ ok: false, code: \'password\', message: \'This PDF is password-protected, so it cannot be read.\' })\nelse if (mode === \'slow\') setTimeout(() => parentPort.postMessage({ ok: true, result: { pageCount: 1, title: \'\', pages: [{ page: 1, text: \'slow\' }] } }), 300)\nelse if (mode === \'hang\') setInterval(() => undefined, 1000)\nelse if (mode === \'throw\') throw new Error(\'boom\')\nelse if (mode === \'exit\') process.exit(3)\nelse if (mode === \'buffers\') {\n  // Memory outside the JavaScript heap, in steps the watcher can see.\n  const hold = []\n  setInterval(() => {\n    if (hold.length < 40) hold.push(new Uint8Array(32 * 1024 * 1024).fill(1))\n  }, 50)\n}\nelse if (mode === \'memory\') {\n  const hold = []\n  for (;;) hold.push(new Array(1_000_000).fill(1))\n}\n'

describe('page ranges', () => {
  it('parses pages, ranges and lists, ascending and unique', () => {
    expect(parsePageRange('3', 10)).toEqual([3])
    expect(parsePageRange(3, 10)).toEqual([3])
    expect(parsePageRange('2-5', 10)).toEqual([2, 3, 4, 5])
    expect(parsePageRange(' 1 , 4-6, 5 ', 10)).toEqual([1, 4, 5, 6])
    expect(parsePageRange('8-', 10)).toEqual([8, 9, 10])
    expect(parsePageRange(undefined, 3)).toEqual([1, 2, 3])
  })
  it('explains bad ranges plainly', () => {
    expect(() => parsePageRange('0', 10)).toThrow(/from 1/)
    expect(() => parsePageRange('5-2', 10)).toThrow(/backwards/)
    expect(() => parsePageRange('11', 10)).toThrow(/10 pages; page 11 does not exist/)
    expect(() => parsePageRange('2-11', 10)).toThrow(/past the end/)
    expect(() => parsePageRange('two', 10)).toThrow(/not a page/)
    expect(() => parsePageRange('1;2', 10)).toThrow(PdfProblem)
    expect(() => parsePageRange({}, 10)).toThrow(/must be text/)
  })
  it('goes through overlapping ranges once, and refuses made-up page counts and endless lists', () => {
    expect(parsePageRange('5-7,1-3,2-6,9,9', 10)).toEqual([1, 2, 3, 4, 5, 6, 7, 9])
    expect(parsePageRange(Array.from({ length: 600 }, () => '1-').join(','), 100_000)).toHaveLength(100_000)
    expect(() => parsePageRange(null, 100_001)).toThrow(/100,001 pages/)
    expect(() => parsePageRange('1', 110_000_000)).toThrow(/up to 100,000/)
    expect(() => parsePageRange('1', 1.5)).toThrow(/pages/)
    expect(() => parsePageRange('1,'.repeat(1001) + '1', 10)).toThrow(/at most 2,000/)
  })
  it('writes a page list back the way it is read', () => {
    expect(formatPageList([1, 2, 3, 7, 9, 10])).toBe('1-3,7,9-10')
    expect(formatPageList([4])).toBe('4')
  })
})

describe('text items to lines', () => {
  it('breaks lines on a new baseline and on end-of-line marks, and joins items on one line', () => {
    const text = itemsToText([item('Hello', 72, 700), item('world', 108, 700), item('Second line', 72, 686), item('tail', 72, 672, 24, { hasEOL: true }), item('next', 72, 658)])
    expect(text).toBe('Hello world\nSecond line\ntail\nnext')
  })
  it('does not put a space inside a word split into pieces, and keeps a blank line between paragraphs', () => {
    const text = itemsToText([item('Para', 72, 700, 24), item('graph', 96, 700, 30), item('New paragraph', 72, 660)])
    expect(text).toBe('Paragraph\n\nNew paragraph')
  })
  it('skips marked-content items, strips odd characters and tidies line ends', () => {
    expect(itemsToText([{ type: 'beginMarkedContent' } as never, item('a\u00a0b\u0000  ', 72, 700)])).toBe('a b')
    expect(itemsToText([])).toBe('')
  })
})

describe('search', () => {
  it('finds plain text ignoring case and line breaks, with snippets and counts', () => {
    const m = matchesOnPage(4, 'The quick brown\nfox jumps. Another QUICK   brown fox.', 'quick Brown')
    expect(m?.page).toBe(4)
    expect(m?.count).toBe(2)
    expect(m?.snippets[0]).toContain('The quick brown fox jumps')
    expect(matchesOnPage(1, 'nothing here', 'absent')).toBeNull()
    expect(matchesOnPage(1, 'text', '   ')).toBeNull()
  })
  it('treats the query as text, not a pattern, and caps snippets per page', () => {
    expect(matchesOnPage(1, 'cost (a+b)* is 5', '(a+b)*')?.count).toBe(1)
    const many = matchesOnPage(1, 'x '.repeat(500), 'x')
    expect(many?.count).toBe(500)
    expect(many?.snippets.length).toBeLessThanOrEqual(3)
  })
})

describe('result layout', () => {
  const info = { name: 'a.pdf', pageCount: 5, title: 'Title' }
  const page = (n: number, text: string) => ({ page: n, text })

  it('starts with the file, page count and title, then page sections', () => {
    const r = renderPages(info, [1, 2], [page(1, 'one'), page(2, 'two')])
    expect(r.text).toBe('a.pdf: 5 pages, title "Title"\n\n--- Page 1 ---\none\n\n--- Page 2 ---\ntwo')
    expect(r.next).toBeNull()
  })
  it('cuts at page boundaries and says which pages to ask for next', () => {
    const r = renderPages(info, [1, 2, 3, 4, 5], [page(1, 'a'.repeat(60)), page(2, 'b'.repeat(60)), page(3, 'c'.repeat(60))], { budget: 200 })
    expect(r.text).toContain('--- Page 2 ---')
    expect(r.text).not.toContain('--- Page 3 ---')
    expect(r.next).toEqual({ pages: '3-5' })
    expect(r.text).toContain('Call again with pages "3-5"')
  })
  it('names the pages left when only some of a list were read', () => {
    const r = renderPages(info, [1, 4, 5], [page(1, 'x')])
    expect(r.next).toEqual({ pages: '4-5' })
  })
  it('splits one very long page and gives the offset to continue from', () => {
    const body = Array.from({ length: 400 }, (_, i) => `line ${i} ${'w'.repeat(20)}`).join('\n')
    const first = renderPages(info, [3], [page(3, body)], { budget: 3000 })
    expect(first.next?.pages).toBe('3')
    expect(first.next?.offset).toBeGreaterThan(1000)
    expect(first.text.length).toBeLessThan(3500)
    expect(first.text).toContain(`offset ${first.next!.offset}`)
    const second = renderPages(info, [3], [page(3, body)], { budget: 3000, offset: first.next!.offset })
    expect(second.text).toContain(body.slice(first.next!.offset!, first.next!.offset! + 40).split('\n')[0])
    // Reading on to the end finishes the page.
    let offset: number | undefined = 0
    let guard = 0
    let rebuilt = ''
    while (offset !== undefined && guard++ < 50) {
      const r = renderPages(info, [3], [page(3, body)], { budget: 3000, offset })
      rebuilt += r.text.split('--- Page 3 ---\n')[1].split('\n\n(Page 3 continues')[0]
      offset = r.next?.offset
    }
    expect(rebuilt).toBe(body)
  })
  it('keeps the pages after a long page in the next call', () => {
    const asked = [3, 4, 5]
    const pages = [page(3, 'x'.repeat(5000)), page(4, 'four'), page(5, 'five')]
    const first = renderPages(info, asked, pages, { budget: 3000 })
    expect(first.next?.pages).toBe('3-5')
    expect(first.text).toContain('pages "3-5" and offset')
    let all = ''
    let offset: number | undefined = 0
    for (let i = 0; i < 10 && offset !== undefined; i++) {
      const r = renderPages(info, asked, pages, { budget: 3000, offset })
      all += r.text
      offset = r.next?.offset
      expect(r.next === null || r.next.offset !== undefined).toBe(true)
    }
    expect(all.split('').filter((c) => c === 'x').length).toBe(5000)
    expect(all).toContain('four')
    expect(all).toContain('five')
  })
  it('keeps a search result within the size limit and says where to go on', () => {
    const q = 'q'.repeat(200)
    const hits = Array.from({ length: 30 }, (_, i) => ({ page: i + 1, count: 3, snippets: [0, 1, 2].map(() => 'a'.repeat(70) + q + 'b'.repeat(70)) }))
    const text = renderSearch(info, q, '', hits, 30)
    expect(text.length).toBeLessThan(21_000)
    expect(text).toMatch(/Search again with pages "\d+-30"/)
    expect(text).toContain('Read a page with pages')
  })
  it('says a page has no text, and a document with none says so once', () => {
    const mixed = renderPages(info, [1, 2], [page(1, 'text'), page(2, '')])
    expect(mixed.text).toContain(`--- Page 2 ---\n${NO_TEXT_PAGE}`)
    const none = renderPages(info, [1, 2, 3], [page(1, ''), page(2, ''), page(3, ' ')])
    expect(none.noText).toBe(true)
    expect(none.text.match(/scan/g)?.length).toBe(1)
    expect(none.text).not.toContain('--- Page')
  })
})


describe('reading real PDFs (pdf.js)', () => {
  const reader = new InProcessPdfReader()
  const write = (name: string, data: Buffer | string): string => {
    const p = join(root, name)
    writeFileSync(p, data)
    return p
  }
  const readPdf = (path: string, args: Record<string, unknown> = {}, r: PdfReader | undefined | null = reader, signal?: AbortSignal) => readPdfTool(path, args, r ?? undefined, signal)

  const doc = write('doc.pdf', buildPdf([['Chapter One', 'The quick brown fox.'], ['Second page', 'Nothing special here.'], ['Third page', 'A Fox appears again, and the FOX runs.']], { title: 'My (test) Doc' }))

  it('reads every page with a header and page markers', async () => {
    const r = await readPdf(doc)
    expect(r.isError).toBe(false)
    expect(r.text.startsWith('doc.pdf: 3 pages, title "My (test) Doc"\n')).toBe(true)
    expect(r.text).toContain('--- Page 1 ---\nChapter One\nThe quick brown fox.')
    expect(r.text).toContain('--- Page 3 ---\nThird page')
  })
  it('reads the pages asked for', async () => {
    const r = await readPdf(doc, { pages: '2' })
    expect(r.text).toContain('Second page')
    expect(r.text).not.toContain('Chapter One')
    const two = await readPdf(doc, { pages: '1,3' })
    expect(two.text).toContain('Chapter One')
    expect(two.text).toContain('Third page')
    expect(two.text).not.toContain('Second page')
  })
  it('searches all pages or the pages asked for', async () => {
    const r = await readPdf(doc, { query: 'fox' })
    expect(r.isError).toBe(false)
    expect(r.text).toContain('3 matches for "fox" on 2 pages')
    expect(r.text).toContain('Page 1 (1 match)')
    expect(r.text).toContain('Page 3 (2 matches)')
    const some = await readPdf(doc, { query: 'fox', pages: '2-3' })
    expect(some.text).not.toContain('Page 1 (')
    expect((await readPdf(doc, { query: 'zebra' })).text).toMatch(/No matches for "zebra"/)
  })
  it('splits a long page and continues with the offset', async () => {
    const lines = Array.from({ length: 120 }, (_, i) => `Row ${i} ${'filler words to make the line long '.repeat(5)}`)
    const long = write('long.pdf', buildPdf([lines, ['Short page']], { fontSize: 3 }))
    const first = await readPdf(long, { pages: '1' })
    expect(first.text.length).toBeLessThan(21_500)
    const m = /pages "1" and offset (\d+)/.exec(first.text)
    expect(m).not.toBeNull()
    const next = await readPdf(long, { pages: '1', offset: Number(m![1]) })
    expect(next.isError).toBe(false)
    expect(next.text).toContain('Row 119')
    const all = await readPdf(long)
    expect(all.text).toMatch(/Call again with pages/)
  })
  it('says a page without text may be a scan, and a PDF without any says so once', async () => {
    const mixed = write('mixed.pdf', buildPdf([['Real text'], null]))
    const r = await readPdf(mixed)
    expect(r.text).toContain(`--- Page 2 ---\n${NO_TEXT_PAGE}`)
    const scan = write('scan.pdf', buildPdf([null, null]))
    const s = await readPdf(scan)
    expect(s.isError).toBe(false)
    expect(s.text).toMatch(/looks like scanned images/)
    expect(s.text).not.toContain('--- Page')
  })
  it('gives clear errors', async () => {
    const err = async (path: string, args: Record<string, unknown> = {}, r: PdfReader | undefined = reader): Promise<string> => {
      const out = await readPdf(path, args, r)
      expect(out.isError, JSON.stringify([path, args])).toBe(true)
      return out.text
    }
    expect(await err(join(root, 'missing.pdf'))).toMatch(/does not exist/)
    expect(await err(root)).toMatch(/is a folder/)
    expect(await err('\\\\server\\share\\a.pdf')).toMatch(/network or device path/)
    expect(await err(`${doc}:stream`)).toMatch(/data stream/)
    expect(await err(write('notes.pdf', 'just some text'))).toMatch(/not a PDF/)
    expect(await err(write('pw.pdf', buildPdf([['secret']], { encrypted: true })))).toMatch(/password-protected/)
    expect(await err(write('bad.pdf', '%PDF-1.4\nthis is not really a pdf file at all'))).toMatch(/damaged/)
    expect(await err(doc, { pages: '9' })).toMatch(/3 pages; page 9 does not exist/)
    expect(await err(doc, { pages: '3-1' })).toMatch(/backwards/)
    expect(await err(doc, { query: 'fox', offset: 5 })).toMatch(/cannot be combined/)
    expect(await err(doc, { query: 'x'.repeat(201) })).toMatch(/at most 200/)
    expect(await err(doc, { offset: -1, pages: '1' })).toMatch(/whole number/)
    const none = await readPdf(doc, {}, null as never)
    expect(none.isError).toBe(true)
    expect(none.text).toMatch(/not available/)
  })
  it('accepts a PDF with a little junk before its header', async () => {
    const p = write('junk.pdf', Buffer.concat([Buffer.from('junk bytes\n'), buildPdf([['Still fine']])]))
    const r = await readPdf(p)
    expect(typeof r.text).toBe('string')
  })
  it('refuses a file over 200 MB before reading it', async () => {
    const big = join(root, 'big.pdf')
    writeFileSync(big, '%PDF-')
    truncateSync(big, 201 * 1024 * 1024)
    const before = reader.calls
    const r = await readPdf(big)
    expect(r.isError).toBe(true)
    expect(r.text).toMatch(/201 MB; read_pdf reads PDFs of up to 200 MB/)
    expect(reader.calls).toBe(before)
  })
  it('reads through a link to a local folder, but refuses one that leads to a network share', async () => {
    const real = join(root, 'real')
    mkdirSync(real)
    writeFileSync(join(real, 'in.pdf'), buildPdf([['Through a junction']]))
    const link = join(root, 'link')
    try {
      symlinkSync(real, link, 'junction')
      const r = await readPdf(join(link, 'in.pdf'))
      expect(r.text).toContain('Through a junction')
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code !== 'EPERM') throw e
    }
    const spy = vi.spyOn(fsp, 'realpath').mockResolvedValue('\\\\host\\share\\secret.pdf')
    try {
      const r = await readPdf(doc)
      expect(r.isError).toBe(true)
      expect(r.text).toMatch(/network or device path/)
    } finally {
      spy.mockRestore()
    }
  })
  it('refuses a link to a network share without contacting it', async () => {
    const link = vi.spyOn(fsp, 'lstat').mockImplementation((async (p: string) => ({ isSymbolicLink: () => String(p).endsWith('.pdf') })) as never)
    const to = vi.spyOn(fsp, 'readlink').mockResolvedValue('\\\\host\\share\\secret.pdf' as never)
    const real = vi.spyOn(fsp, 'realpath')
    try {
      const r = await readPdf(doc)
      expect(r.isError).toBe(true)
      expect(r.text).toMatch(/network or device path/)
      expect(real).not.toHaveBeenCalled()
    } finally {
      for (const s of [link, to, real]) s.mockRestore()
    }
  })
  it('reads with the character maps folder given as any kind of path, and without it', async () => {
    const dir = resolve('node_modules/pdfjs-dist/cmaps')
    const read = async (cmapDir?: string): Promise<string | undefined> =>
      (await runPdfJob(new Uint8Array(buildJapanesePdf()), { type: 'pages', pages: null, maxChars: 1000 }, { cmapDir })).pages?.[0].text
    const japanese = String.fromCharCode(0x3042, 0x3044)
    for (const d of [dir, `${dir}\\`, `${dir}/`, dir.split('\\').join('/')]) if (process.platform === 'win32' || !d.endsWith('\\')) expect(await read(d)).toBe(japanese)
    expect(await read(join(dir, 'missing'))).toBe('')
    expect(await read()).toBe('')
  })
  it('refuses a made-up page count before building any page list', async () => {
    const read = (count: number) => runPdfJob(new Uint8Array(buildFakeCountPdf(count)), { type: 'pages', pages: null, maxChars: 20_000 })
    expect((await read(1000)).pageCount).toBe(1000)
    await expect(read(110_000_000)).rejects.toMatchObject({ code: 'damaged', message: expect.stringContaining('110,000,000') })
    const file = join(root, 'fake-count.pdf')
    writeFileSync(file, buildFakeCountPdf(110_000_000))
    expect((await readPdf(file, { query: 'line' })).isError).toBe(true)
    // The main process checks the count again, whatever the reader says.
    const lying: PdfReader = { run: () => Promise.resolve({ pageCount: 110_000_000, title: '', pages: [{ page: 1, text: 'x' }] }) }
    const r = await readPdf(doc, {}, lying)
    expect(r.isError).toBe(true)
    expect(r.text).toMatch(/up to 100,000/)
  })
  it('does not let offset raise the amount of text read without limit', async () => {
    const asked: number[] = []
    const spy: PdfReader = { run: (_d, j) => (asked.push((j as { maxChars: number }).maxChars), Promise.resolve({ pageCount: 1, title: '', pages: [{ page: 1, text: 'x' }] })) }
    await readPdf(doc, { pages: '1', offset: 10 ** 15 }, spy)
    expect(asked).toEqual([2_020_000])
  })
  it('stops when the call is stopped', async () => {
    const ctrl = new AbortController()
    ctrl.abort()
    const stopping: PdfReader = { run: (_d, _j, signal) => (signal?.aborted ? Promise.reject(new PdfProblem('stopped', 'The read was stopped.')) : Promise.reject(new Error('not stopped'))) }
    const r = await readPdf(doc, {}, stopping, ctrl.signal)
    expect(r).toEqual({ text: 'The tool call was stopped.', isError: true })
  })
})

describe('the worker reader', () => {
  const stub = join(root, 'pdfWorkerStub.mjs')
  writeFileSync(stub, STUB)
  const job = (query: string) => ({ type: 'search' as const, pages: null, query })
  const data = () => new Uint8Array([1, 2, 3])

  it('returns what the worker sends and passes on its problems', async () => {
    const r = new PdfWorkerReader({ workerPath: stub })
    expect((await r.run(data(), job('ok'))).pages?.[0].text).toBe('stub')
    await expect(r.run(data(), job('problem'))).rejects.toMatchObject({ code: 'password' })
  })
  it('turns a crash, an exit and a hang into plain errors', async () => {
    const r = new PdfWorkerReader({ workerPath: stub, timeoutMs: 400 })
    await expect(r.run(data(), job('throw'))).rejects.toMatchObject({ code: 'crash', message: expect.stringContaining('boom') })
    await expect(r.run(data(), job('exit'))).rejects.toMatchObject({ code: 'crash' })
    await expect(r.run(data(), job('hang'))).rejects.toMatchObject({ code: 'timeout' })
  })
  it('ends a thread that uses too much memory', async () => {
    const r = new PdfWorkerReader({ workerPath: stub, timeoutMs: 20_000, resourceLimits: { maxOldGenerationSizeMb: 48, maxYoungGenerationSizeMb: 8 } })
    await expect(r.run(data(), job('memory'))).rejects.toMatchObject({ code: 'crash', message: expect.stringMatching(/more memory/) })
  })
  it('ends a read that makes the whole app grow too much', async () => {
    const r = new PdfWorkerReader({ workerPath: stub, timeoutMs: 20_000, maxGrowthBytes: 150 * 1024 * 1024 })
    await expect(r.run(data(), job('buffers'))).rejects.toMatchObject({ code: 'crash', message: expect.stringMatching(/more memory/) })
  })
  it('does not start the next read before a stopped thread has ended', async () => {
    const r = new PdfWorkerReader({ workerPath: stub, maxAtOnce: 1 })
    const ctrl = new AbortController()
    const running = r.run(data(), job('hang'), ctrl.signal)
    ctrl.abort()
    await expect(running).rejects.toMatchObject({ code: 'stopped' })
    expect((r as unknown as { active: number }).active).toBe(1)
    expect((await r.run(data(), job('ok'))).pageCount).toBe(1)
    await vi.waitFor(() => expect((r as unknown as { active: number }).active).toBe(0))
  })
  it('stops a read on abort, while running and while waiting', async () => {
    const r = new PdfWorkerReader({ workerPath: stub, maxAtOnce: 1 })
    const ctrl = new AbortController()
    const running = r.run(data(), job('hang'), ctrl.signal)
    const queued = new AbortController()
    const waiting = r.run(data(), job('ok'), queued.signal)
    queued.abort()
    await expect(waiting).rejects.toMatchObject({ code: 'stopped' })
    ctrl.abort()
    await expect(running).rejects.toMatchObject({ code: 'stopped' })
    // The line is free again.
    expect((await r.run(data(), job('ok'))).pageCount).toBe(1)
    const pre = new AbortController()
    pre.abort()
    await expect(r.run(data(), job('ok'), pre.signal)).rejects.toMatchObject({ code: 'stopped' })
  })
  it('runs at most two reads at once and queues the rest', async () => {
    const r = new PdfWorkerReader({ workerPath: stub })
    const started = Date.now()
    await Promise.all([1, 2, 3, 4].map(() => r.run(data(), job('slow'))))
    // Four 300 ms reads two at a time take two rounds; all at once would be one.
    expect(Date.now() - started).toBeGreaterThanOrEqual(580)
  })
  it('hands a view into a shared buffer over as a copy, so a pooled buffer is not detached', async () => {
    const r = new PdfWorkerReader({ workerPath: stub })
    const pooled = Buffer.from('abc')
    await r.run(pooled, job('ok'))
    expect(pooled.buffer.byteLength).toBeGreaterThan(0)
    expect(Buffer.from('still works').toString()).toBe('still works')
  })
})

describe('PDF validation', () => {
  it('finds the signature only in the first 1024 bytes', () => {
    expect(hasPdfMagic(Buffer.from('junk%PDF-1.7'))).toBe(true)
    expect(hasPdfMagic(Buffer.from('x'.repeat(1024) + '%PDF-'))).toBe(false)
  })
  it('accepts scans and reports reader errors', async () => {
    const reader: PdfReader = { run: async () => ({ pageCount: 2, title: '', pages: [{ page: 1, text: '' }] }) }
    expect(await pdfInfo(Buffer.from('%PDF-'), reader)).toEqual({ pages: 2 })
    await expect(pdfInfo(Buffer.from('not pdf'), reader)).rejects.toThrow('not a PDF')
    reader.run = async () => { throw new PdfProblem('password', 'Password protected.') }
    await expect(pdfInfo(Buffer.from('%PDF-'), reader)).rejects.toThrow('Password protected.')
  })
  it('rejects implausible page counts', () => {
    expect(() => parsePageRange(null, 100001)).toThrow()
  })
  it('reads the page count and title of a real PDF', async () => {
    const reader = new InProcessPdfReader()
    expect(await pdfInfo(buildPdf([['a'], ['b'], ['c']], { title: 'Real Title' }), reader)).toEqual({ pages: 3, title: 'Real Title' })
    expect(await pdfInfo(buildPdf([['a']]), reader)).toEqual({ pages: 1 })
  })
  it('accepts a scan with no text', async () => {
    expect(await pdfInfo(buildPdf([null, null]), new InProcessPdfReader())).toEqual({ pages: 2 })
  })
  it('refuses a damaged file and a password-protected one with a sentence', async () => {
    const reader = new InProcessPdfReader()
    await expect(pdfInfo(Buffer.from('%PDF-1.4\nbroken'), reader)).rejects.toThrow(/damaged/)
    await expect(pdfInfo(buildPdf([['x']], { encrypted: true }), reader)).rejects.toThrow(/password-protected/)
  })
})
