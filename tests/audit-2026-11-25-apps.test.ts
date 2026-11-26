import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { InstallOptions, JobState } from '@shared/types'
import { dropSharedLists, getFetch, setFetch } from '../src/main/github'
import { Store } from '../src/main/store'

vi.mock('electron', () => ({ shell: {}, app: { getPath: () => tmpdir() } }))

// Extracting is replaced by a stand-in the tests steer; everything else in the archive module is real.
const extract = vi.hoisted(() => ({ run: null as null | ((dest: string) => Promise<void>) }))
vi.mock('../src/main/archive', async (orig) => ({
  ...(await orig<typeof import('../src/main/archive')>()),
  extractZip: async (_zip: string, dest: string) => {
    if (!extract.run) throw new Error('extractZip was not expected')
    await extract.run(dest)
  }
}))

// winget's answer is scripted; every other program is asked for real.
const winget = vi.hoisted(() => ({ answers: [] as (string | null)[] }))
vi.mock('../src/main/process', async (orig) => {
  const actual = await orig<typeof import('../src/main/process')>()
  return {
    ...actual,
    capture: (cmd: string, args: string[], ...rest: unknown[]) =>
      cmd === 'winget' ? Promise.resolve(winget.answers.shift() ?? null) : (actual.capture as (...a: unknown[]) => Promise<string | null>)(cmd, args, ...rest)
  }
})

const { AppManager, readLatestCache } = await import('../src/main/apps')
const { checkRequirements, invalidateRequirements } = await import('../src/main/requirements')

const original = getFetch()
let dir: string
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'rulz-audit-apps-'))
  dropSharedLists()
})
afterEach(async () => {
  setFetch(original)
  extract.run = null
  await rm(dir, { recursive: true, force: true })
})

async function manager(): Promise<{ m: InstanceType<typeof AppManager>; store: Store; jobs: JobState[] }> {
  const store = new Store(join(dir, 'data'))
  await store.init()
  const jobs: JobState[] = []
  const m = new AppManager({
    store,
    catalogDir: join(dir, 'catalog'),
    downloadsDir: join(dir, 'downloads'),
    iconsDir: join(dir, 'icons'),
    emitApps: () => undefined,
    emitJob: (j) => jobs.push(j),
    emitLog: () => undefined
  })
  return { m, store, jobs }
}

function ended(jobs: JobState[], appId: string): Promise<JobState> {
  return vi.waitFor(
    () => {
      const last = jobs.filter((j) => j.appId === appId).at(-1)
      if (!last || !['done', 'error', 'cancelled'].includes(last.phase)) throw new Error('still running')
      return last
    },
    { timeout: 20_000, interval: 50 }
  )
}

const options = (installDir: string): InstallOptions => ({
  dir: installDir,
  desktopShortcut: false,
  startMenuShortcut: false,
  components: [],
  launchAfter: false,
  installPrerequisites: false
})

describe('B2: Cancel during extraction is honoured', () => {
  it.runIf(process.platform === 'win32')('ends the job as cancelled and installs nothing', async () => {
    const body = Buffer.from('not really a zip')
    const release = {
      tag_name: 'image-organizer-v2.0.0',
      name: 'v2.0.0',
      published_at: '2026-11-25T00:00:00Z',
      prerelease: false,
      draft: false,
      html_url: 'https://example.test/release',
      assets: [{ name: 'RulZ-Image-Organizer-v2.0.0-win64.zip', size: body.length, browser_download_url: 'https://dl.test/organizer.zip' }]
    }
    setFetch(async (url) => {
      if (url.startsWith('https://api.github.com/')) return new Response(JSON.stringify([release]), { status: 200 })
      if (url === 'https://dl.test/organizer.zip') return new Response(body, { status: 200 })
      return new Response('', { status: 404 })
    })
    const { m, store, jobs } = await manager()
    // The user clicks Cancel while the files are being extracted, which can't be stopped half-way.
    extract.run = async (dest) => {
      await mkdir(dest, { recursive: true })
      await writeFile(join(dest, 'RulZ Image Organizer.exe'), 'new')
      m.cancel('image-organizer')
    }
    const target = join(dir, 'Organizer')
    await m.install('image-organizer', options(target))
    const job = await ended(jobs, 'image-organizer')
    expect(job.phase).toBe('cancelled')
    expect(store.getInstalled('image-organizer')).toBeNull()
    expect(existsSync(join(target, 'RulZ Image Organizer.exe'))).toBe(false)
  }, 30_000)
})

describe('B7: forced checks one after another', () => {
  it('reuses the release pages the first forced check fetched', async () => {
    const urls: string[] = []
    setFetch(async (url) => {
      urls.push(url)
      return url.includes('/releases?per_page=100') ? new Response('[]', { status: 200 }) : new Response('', { status: 404 })
    })
    const { m } = await manager()
    await m.refreshLatest(true)
    const pages = () => urls.filter((u) => u.includes('/releases?per_page=100')).length
    expect(pages()).toBe(1)
    // A click comes later than the first check's own bookkeeping.
    await new Promise((r) => setTimeout(r, 20))
    await m.refreshLatest(true)
    // The second click, seconds later, asks GitHub for no release page again.
    expect(pages()).toBe(1)
  }, 30_000)
})

describe('B4b: a damaged latest-cache.json', () => {
  it('skips entries that are not objects instead of throwing', () => {
    const cache = { entries: { x: 's', y: 5, z: { info: { version: 'v1', label: 'v1', date: null, downloadSize: null, url: null }, at: 3 } } }
    const m = readLatestCache(cache as never)
    expect([...m.keys()]).toEqual(['z'])
  })
})

describe('B6: winget answer', () => {
  it('is asked again after a failed first answer', async () => {
    winget.answers = [null, 'v1.9.0']
    invalidateRequirements()
    const first = await checkRequirements()
    expect(first.some((r) => r.canInstall)).toBe(false)
    invalidateRequirements()
    const second = await checkRequirements()
    expect(second.some((r) => r.canInstall)).toBe(true)
    // A yes is kept: no further question.
    winget.answers = [null]
    invalidateRequirements()
    expect((await checkRequirements()).some((r) => r.canInstall)).toBe(true)
  }, 60_000)
})
