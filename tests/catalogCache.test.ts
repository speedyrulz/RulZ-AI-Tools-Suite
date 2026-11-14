import { existsSync } from 'node:fs'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { CatalogCache, crlfToLf, gitBlobSha, inside } from '../src/main/catalogCache'
import { clearEtags, getFetch, setFetch } from '../src/main/github'
import { withAssetVersion } from '../src/shared/readme'

const original = getFetch()
let dir: string
let bundled: string
let cacheDir: string

/** What "GitHub" holds: catalog paths and contents. */
let remote: Record<string, string>
/** Contents raw.githubusercontent.com hands out instead (a stale CDN copy, say). */
let rawOverride: Record<string, string>
let calls: string[]
/** The commit main points at. */
let head: string

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'rulz-catalog-'))
  bundled = join(dir, 'bundled')
  cacheDir = join(dir, 'cache')
  remote = {}
  rawOverride = {}
  calls = []
  head = 'c1'
  clearEtags()
  setFetch(async (url) => {
    calls.push(url)
    if (url.includes('/commits?sha=main')) return new Response(JSON.stringify([{ sha: head, commit: { message: 'm', committer: null } }]))
    if (url.startsWith('https://api.github.com/')) {
      expect(url).toContain(`/git/trees/${head}%3Acatalog?recursive=1`)
      const tree = Object.entries(remote).map(([path, text]) => ({
        path,
        type: 'blob',
        sha: gitBlobSha(Buffer.from(text)),
        size: Buffer.byteLength(text)
      }))
      return new Response(JSON.stringify({ truncated: false, tree: [{ path: 'llm-suite', type: 'tree', sha: 'x' }, ...tree] }))
    }
    const prefix = `https://raw.githubusercontent.com/o/r/${head}/catalog/`
    expect(url.startsWith(prefix)).toBe(true)
    // Names are sent encoded (spaces, #).
    expect(url.slice(prefix.length)).not.toMatch(/[ #]/)
    const path = decodeURIComponent(url.slice(prefix.length))
    const text = rawOverride[path] ?? remote[path]
    return text === undefined ? new Response('nope', { status: 404 }) : new Response(text)
  })
})

afterEach(async () => {
  setFetch(original)
  await rm(dir, { recursive: true, force: true })
})

async function bundle(files: Record<string, string>): Promise<void> {
  for (const [path, text] of Object.entries(files)) {
    await mkdir(dirname(join(bundled, path)), { recursive: true })
    await writeFile(join(bundled, path), text)
  }
}

function cache(version = '1.0.0'): CatalogCache {
  return new CatalogCache({ repo: 'o/r', branch: 'main', bundledDir: bundled, cacheDir, launcherVersion: version, appIds: ['llm-suite', 'hf-manager'] })
}

describe('gitBlobSha', () => {
  it("matches git's hash of a file", () => {
    // `printf 'hello\n' | git hash-object --stdin`
    expect(gitBlobSha(Buffer.from('hello\n'))).toBe('ce013625030ba8dba906f756967f9e9ca394464a')
  })

  it('turns CRLF into LF and leaves lone CRs', () => {
    expect(Buffer.from(crlfToLf(Buffer.from('a\r\nb\rc\r\n'))).toString()).toBe('a\nb\rc\n')
  })
})

describe('inside', () => {
  it('refuses paths that leave the folder', () => {
    expect(inside('C:\\x', 'llm-suite/README.md')).toBe('C:\\x\\llm-suite\\README.md')
    expect(inside('C:\\x', '../evil.png')).toBeNull()
    expect(inside('C:\\x', 'a/../../evil.png')).toBeNull()
    expect(inside('C:\\x', 'C:\\Windows\\evil.png')).toBeNull()
    expect(inside('C:\\x', '/etc/passwd')).toBeNull()
    expect(inside('C:\\x', '')).toBeNull()
  })
})

describe('CatalogCache', () => {
  it('serves bundled files until GitHub has been asked', async () => {
    await bundle({ 'llm-suite/README.md': '# old', 'llm-suite/screenshots/01.png': 'a' })
    const c = cache()
    await c.load()
    expect(c.screenshots('llm-suite')).toBeNull()
    expect(c.file('llm-suite/README.md')).toBe(join(bundled, 'llm-suite', 'README.md'))
    expect(c.readmeFromGitHub('llm-suite')).toBe(false)
  })

  it('downloads only what differs from the bundled copy, and lists new screenshots', async () => {
    await bundle({ 'llm-suite/README.md': '# same\r\n', 'llm-suite/screenshots/01.png': 'old', 'llm-suite/icon.png': 'icon' })
    remote = {
      'llm-suite/README.md': '# same\n', // CRLF on disk (Windows checkout), LF in git: the same file
      'llm-suite/screenshots/01.png': 'new',
      'llm-suite/screenshots/02.png': 'added',
      'llm-suite/icon.png': 'icon',
      'llm-suite/icon.ico': 'not fetched',
      'llm-suite/repo/docs/shot.png': 'readme image',
      'unknown-app/README.md': 'ignored'
    }
    const c = cache()
    await c.load()
    expect(await c.sync()).toEqual({ changed: 3, failed: 0 })
    const fetched = calls.filter((u) => u.startsWith('https://raw.')).map((u) => u.split('/catalog/')[1]).sort()
    expect(fetched).toEqual(['llm-suite/repo/docs/shot.png', 'llm-suite/screenshots/01.png', 'llm-suite/screenshots/02.png'])

    expect(c.file('llm-suite/README.md')).toBe(join(bundled, 'llm-suite', 'README.md'))
    expect(c.readmeFromGitHub('llm-suite')).toBe(true)
    expect(await readFile(c.file('llm-suite/screenshots/01.png')!, 'utf8')).toBe('new')
    expect(await readFile(c.file('llm-suite/repo/docs/shot.png')!, 'utf8')).toBe('readme image')
    expect(c.screenshots('llm-suite')?.map((s) => s.name)).toEqual(['01.png', '02.png'])
    expect(c.screenshots('hf-manager')).toEqual([])
    // With the setting off, the bundled file.
    expect(c.file('llm-suite/screenshots/01.png', false)).toBe(join(bundled, 'llm-suite', 'screenshots', '01.png'))
  })

  it('keeps what it fetched across launches, and asks again without downloading unchanged files', async () => {
    await bundle({ 'llm-suite/README.md': '# old' })
    remote = { 'llm-suite/README.md': '# new' }
    await cache().load()
    const first = cache()
    await first.load()
    await first.sync()

    calls = []
    const second = cache()
    await second.load()
    expect(second.checkedAt()).toBeGreaterThan(0)
    expect(await readFile(second.file('llm-suite/README.md')!, 'utf8')).toBe('# new')
    expect(await second.sync()).toEqual({ changed: 0, failed: 0 })
    // The same commit as before: only the commit is asked for.
    expect(calls).toHaveLength(1)
    expect(calls[0]).toContain('/commits?sha=main')
  })

  it('starts over after a launcher update (whose bundled files may be newer than the cache)', async () => {
    await bundle({ 'llm-suite/README.md': '# old' })
    remote = { 'llm-suite/README.md': '# new' }
    const c = cache('1.0.0')
    await c.load()
    await c.sync()
    expect(existsSync(join(cacheDir, 'llm-suite', 'README.md'))).toBe(true)

    const next = cache('1.1.0')
    await next.load()
    expect(existsSync(cacheDir)).toBe(false)
    expect(next.file('llm-suite/README.md')).toBe(join(bundled, 'llm-suite', 'README.md'))
  })

  it('refuses a download that is not the file GitHub lists, keeps showing the bundled one and tries again', async () => {
    await bundle({ 'llm-suite/screenshots/01.png': 'old', 'llm-suite/README.md': '# old' })
    remote = { 'llm-suite/screenshots/01.png': 'new', 'llm-suite/README.md': '# new' }
    rawOverride = { 'llm-suite/screenshots/01.png': 'old', 'llm-suite/README.md': 'cut sho' }
    const c = cache()
    await c.load()
    expect(await c.sync()).toEqual({ changed: 0, failed: 2 })
    expect(c.hasFailures()).toBe(true)
    expect(c.file('llm-suite/screenshots/01.png')).toBe(join(bundled, 'llm-suite', 'screenshots', '01.png'))
    expect(c.screenshots('llm-suite')?.map((s) => s.name)).toEqual(['01.png'])
    expect(c.readmeFromGitHub('llm-suite')).toBe(false)

    // Same commit, but files are missing: asked again.
    rawOverride = {}
    expect(await c.sync()).toEqual({ changed: 2, failed: 0 })
    expect(c.hasFailures()).toBe(false)
    expect(c.readmeFromGitHub('llm-suite')).toBe(true)
    expect(await readFile(c.file('llm-suite/screenshots/01.png')!, 'utf8')).toBe('new')
  })

  it('drops files removed on GitHub, and fetched copies that match the bundled file again', async () => {
    await bundle({ 'llm-suite/screenshots/01.png': 'a', 'llm-suite/screenshots/02.png': 'b' })
    remote = { 'llm-suite/screenshots/01.png': 'changed', 'llm-suite/screenshots/02.png': 'b' }
    const c = cache()
    await c.load()
    await c.sync()
    const cached = join(cacheDir, 'llm-suite', 'screenshots', '01.png')
    expect(existsSync(cached)).toBe(true)

    remote = { 'llm-suite/screenshots/01.png': 'a' } // reverted, and 02 deleted
    head = 'c2'
    expect(await c.sync()).toEqual({ changed: 2, failed: 0 })
    expect(existsSync(cached)).toBe(false)
    expect(c.screenshots('llm-suite')?.map((s) => s.name)).toEqual(['01.png'])
  })

  it('keeps the known files when GitHub cannot be reached', async () => {
    await bundle({ 'llm-suite/README.md': '# old' })
    remote = { 'llm-suite/README.md': '# new' }
    const c = cache()
    await c.load()
    await c.sync()
    setFetch(async () => {
      throw new Error('offline')
    })
    await expect(c.sync()).rejects.toThrow('offline')
    expect(await readFile(c.file('llm-suite/README.md')!, 'utf8')).toBe('# new')
  })

  it('shares one check between calls made while it runs', async () => {
    remote = { 'llm-suite/README.md': '# new' }
    const c = cache()
    await c.load()
    await Promise.all([c.sync(), c.sync()])
    expect(calls.filter((u) => u.startsWith('https://api.'))).toHaveLength(2)
  })

  it('serves only catalog paths, and fetched files only when listed', async () => {
    await bundle({ 'llm-suite/README.md': '# old' })
    remote = { 'llm-suite/README.md': '# new', 'llm-suite/screenshots/a shot #1.png': 'x' }
    const c = cache()
    await c.load()
    await c.sync()
    expect(c.screenshots('llm-suite')?.map((s) => s.name)).toEqual(['a shot #1.png'])
    expect(c.file('../manifest.json')).toBeNull()
    expect(c.file('llm-suite/../../cache/manifest.json')).toBeNull()
    expect(c.file('C:\Windows\win.ini')).toBeNull()
    // manifest.json sits in the cache folder but isn't a listed file: the bundled path (which doesn't exist) is given.
    expect(c.file('manifest.json')).toBe(join(bundled, 'manifest.json'))
  })

  it('throws on a cut-short list instead of dropping files', async () => {
    remote = { 'llm-suite/README.md': '# new' }
    const c = cache()
    await c.load()
    const real = getFetch()
    setFetch(async (url, init) =>
      url.includes('/git/trees/') ? new Response(JSON.stringify({ truncated: true, tree: [] })) : real(url, init)
    )
    await expect(c.sync()).rejects.toThrow('cut short')
  })
})

describe('withAssetVersion', () => {
  const base = 'rulz://catalog/llm-suite/repo/'
  it('adds the version of known images only', () => {
    const v = { 'docs/a b.png': 'abc' }
    expect(withAssetVersion(base + 'docs/a%20b.png', base, v)).toBe(base + 'docs/a%20b.png?v=abc')
    expect(withAssetVersion(base + 'docs/other.png', base, v)).toBe(base + 'docs/other.png')
    expect(withAssetVersion('https://x/y.png', base, v)).toBe('https://x/y.png')
    expect(withAssetVersion(base + 'docs/a%20b.png', base, undefined)).toBe(base + 'docs/a%20b.png')
  })
})
