// Tests for the fixes from the November 2026 launcher audit: archive folder names, release lookup past the shared
// pages, download sizes, per-app Python versions and the self-update source.
import { existsSync } from 'node:fs'
import { mkdir, mkdtemp, readdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { REQUIREMENT_GUIDES, requirementRefLabel, statusFor } from '@shared/requirements'
import type { RequirementStatus } from '@shared/types'
import { backupDir, flattenSingleFolder, stagingDir, swapInto } from '../src/main/archive'
import { download } from '../src/main/download'
import { branchSha, compareVersions, getFetch, headCommit, latestRelease, setFetch } from '../src/main/github'

const original = getFetch()
let dir: string
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'rulz-audit-'))
})
afterEach(async () => {
  setFetch(original)
  await rm(dir, { recursive: true, force: true })
})

describe('archive folders', () => {
  it("uses its own names for the staging and old folders, so a user's <dir>.old backup is left alone", async () => {
    const app = join(dir, 'Grid')
    expect(stagingDir(app)).toBe(`${app}.rulz-new`)
    expect(backupDir(app)).toBe(`${app}.rulz-old`)
    await mkdir(app)
    await writeFile(join(app, 'v1.txt'), '')
    await mkdir(`${app}.old`)
    await writeFile(join(`${app}.old`, 'my backup.txt'), 'keep')
    await mkdir(stagingDir(app))
    await writeFile(join(stagingDir(app), 'v2.txt'), '')
    await swapInto(stagingDir(app), app)
    expect(await readdir(app)).toEqual(['v2.txt'])
    expect(await readdir(`${app}.old`)).toEqual(['my backup.txt'])
    expect(existsSync(backupDir(app))).toBe(false)
  })

  it('flattens a zip whose single folder holds something with its own name', async () => {
    await mkdir(join(dir, 'App', 'App'), { recursive: true })
    await writeFile(join(dir, 'App', 'App', 'inner.txt'), '')
    await writeFile(join(dir, 'App', 'App.exe'), '')
    await flattenSingleFolder(dir)
    expect((await readdir(dir)).sort()).toEqual(['App', 'App.exe'])
    expect(await readdir(join(dir, 'App'))).toEqual(['inner.txt'])
  })
})

function raw(tag: string, assets: string[] = [], extra: object = {}): object {
  return {
    tag_name: tag,
    name: tag,
    published_at: null,
    prerelease: false,
    draft: false,
    html_url: '',
    assets: assets.map((name) => ({ name, size: 1, browser_download_url: `https://x/${tag}/${name}` })),
    ...extra
  }
}

describe('latestRelease past the shared pages', () => {
  it("finds an app whose newest release is older than five full pages through its tags", async () => {
    const calls: string[] = []
    const full = Array.from({ length: 100 }, (_, i) => raw(`rulz-trainer-v3.${i}.0`))
    setFetch(async (url) => {
      calls.push(url)
      const u = new URL(url)
      if (u.pathname.endsWith('/releases')) return Response.json(full)
      if (u.pathname.includes('/git/matching-refs/tags/image-organizer-v')) {
        return Response.json(['1.2.0', '1.10.0', '1.9.0', 'x'].map((v) => ({ ref: `refs/tags/image-organizer-v${v}` })))
      }
      // 1.10.0 is a draft (not visible by tag), so 1.9.0 is the newest published one.
      if (u.pathname.endsWith('/releases/tags/image-organizer-v1.10.0')) return new Response('{}', { status: 404 })
      if (u.pathname.endsWith('/releases/tags/image-organizer-v1.9.0')) return Response.json(raw('image-organizer-v1.9.0', ['a.zip']))
      return new Response('{}', { status: 404 })
    })
    const r = await latestRelease('o/deep', 'image-organizer-v')
    expect(r?.tag).toBe('image-organizer-v1.9.0')
    expect(calls.filter((c) => c.includes('/releases?'))).toHaveLength(5)
    expect(calls.findIndex((c) => c.includes('v1.10.0'))).toBeLessThan(calls.findIndex((c) => c.includes('v1.9.0')))
  })

  it('stops at a short page without asking for tags', async () => {
    const calls: string[] = []
    setFetch(async (url) => {
      calls.push(url)
      return Response.json([raw('rulz-trainer-v1.0.0')])
    })
    expect(await latestRelease('o/short', 'image-grid-v')).toBeNull()
    expect(calls).toHaveLength(1)
  })

  it('passes over a release the caller rejects (its download not uploaded yet)', async () => {
    setFetch(async () => Response.json([raw('image-grid-v1.1.0', ['SHA256SUMS']), raw('image-grid-v1.0.0', ['grid.zip'])]))
    const r = await latestRelease('o/accept', 'image-grid-v', (rel) => rel.assets.some((a) => a.name.endsWith('.zip')))
    expect(r?.tag).toBe('image-grid-v1.0.0')
  })

  it('orders versions by number', () => {
    expect(['1.2.0', '1.10.0', '1.9.3'].sort((a, b) => compareVersions(b, a))).toEqual(['1.10.0', '1.9.3', '1.2.0'])
  })
})

describe('branch heads', () => {
  it('names every app branch from one shared request', async () => {
    const calls: string[] = []
    setFetch(async (url) => {
      calls.push(url)
      return Response.json([
        { ref: 'refs/heads/app/hf-manager', object: { sha: 'aaa' } },
        { ref: 'refs/heads/app/rulz-trainer', object: { sha: 'bbb' } }
      ])
    })
    expect(await branchSha('o/heads', 'app/rulz-trainer')).toBe('bbb')
    expect(await branchSha('o/heads', 'app/hf-manager')).toBe('aaa')
    expect(await branchSha('o/heads', 'app/missing')).toBeNull()
    expect(await branchSha('o/heads', 'main')).toBeNull()
    expect(calls).toEqual(['https://api.github.com/repos/o/heads/git/matching-refs/heads/app/'])
  })

  it('asks for the newest commit as a one-commit list, without its files', async () => {
    let asked = ''
    setFetch(async (url) => {
      asked = url
      return Response.json([{ sha: 'c0ffee', commit: { message: 'Title\n\nbody', committer: { date: '2026-11-13' } } }])
    })
    expect(await headCommit('o/list', 'app/x')).toEqual({ sha: 'c0ffee', date: '2026-11-13', message: 'Title' })
    expect(asked).toBe('https://api.github.com/repos/o/list/commits?sha=app%2Fx&per_page=1')
  })
})

describe('download size', () => {
  it("rejects a file that isn't the size GitHub lists, even when the server's own length matches", async () => {
    const short = Buffer.alloc(100, 1)
    setFetch(async () => new Response(short, { status: 200, headers: { 'content-length': '100' } }))
    const dest = join(dir, 'a.zip')
    await expect(download('https://x/a.zip', dest, { expectedSize: 4096 })).rejects.toThrow(/isn't the size of the published file/)
    expect(existsSync(dest)).toBe(false)
    expect(existsSync(`${dest}.part`)).toBe(false)
  })
})

describe('per-app Python version', () => {
  const python = (version: string): RequirementStatus => ({
    id: 'python',
    label: REQUIREMENT_GUIDES.python.label,
    ok: true,
    detail: `Python ${version}`,
    version,
    canInstall: true,
    helpUrl: ''
  })

  it('counts a Python older than the app needs as missing, for winget to install a newer one', () => {
    const s = statusFor({ id: 'python', minVersion: '3.11' }, python('3.10.11'))
    expect(s?.ok).toBe(false)
    expect(s?.canInstall).toBe(true)
    expect(s?.label).toBe('Python 3.11+')
    expect(s?.detail).toBe('Python 3.10 is too old for this app')
    // Without winget it is a step to do by hand.
    expect(statusFor({ id: 'python', minVersion: '3.11' }, { ...python('3.10.11'), canInstall: false })?.canInstall).toBe(false)
  })

  it('accepts a new enough Python, and leaves apps without a minimum alone', () => {
    expect(statusFor({ id: 'python', minVersion: '3.11' }, python('3.12.4'))?.ok).toBe(true)
    expect(statusFor({ id: 'python', minVersion: '3.11' }, python('3.11.0'))?.ok).toBe(true)
    expect(statusFor({ id: 'python' }, python('3.10.0'))).toEqual(python('3.10.0'))
    expect(statusFor({ id: 'python', minVersion: '3.11' }, undefined)).toBeUndefined()
    expect(requirementRefLabel({ id: 'git' })).toBe('Git')
  })
})

describe('launcher self-update', () => {
  it("won't download an update from anywhere but the launcher's own releases", async () => {
    const { installSelfUpdate, SELF_DOWNLOADS } = await import('../src/main/selfUpdate')
    let fetched = false
    setFetch(async () => {
      fetched = true
      return new Response('')
    })
    const u = { version: '9.9.9', tag: 'v9.9.9', size: 1, url: 'https://evil.example/setup.exe', notesUrl: '' }
    await expect(installSelfUpdate(u, () => undefined, () => [])).rejects.toThrow(/isn't one of the launcher's own releases/)
    // Another app's setup from the same repository, or one from another tag, isn't the launcher either.
    const other = { ...u, url: `${SELF_DOWNLOADS}llm-suite-v0.15.0/RulZ-LLM-Suite-Setup-0.15.0.exe` }
    await expect(installSelfUpdate(other, () => undefined, () => [])).rejects.toThrow(/own releases/)
    const app = { ...u, url: `${SELF_DOWNLOADS}v9.9.9/RulZ-LLM-Suite-Setup-0.15.0.exe` }
    await expect(installSelfUpdate(app, () => undefined, () => [])).rejects.toThrow(/own releases/)
    const sums = { ...u, url: `${SELF_DOWNLOADS}v9.9.9/RulZ-AI-Tools-Suite-Setup-9.9.9.exe`, sumsUrl: 'https://evil.example/SUMS' }
    await expect(installSelfUpdate(sums, () => undefined, () => [])).rejects.toThrow(/own releases/)
    expect(fetched).toBe(false)
    expect(SELF_DOWNLOADS).toBe('https://github.com/speedyrulz/RulZ-AI-Tools-Suite/releases/download/')
  }, 30_000)
})
