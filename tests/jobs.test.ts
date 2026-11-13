import { mkdir, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { InstalledInfo, InstallOptions, JobState } from '@shared/types'
import { getFetch, setFetch } from '../src/main/github'
import { Store } from '../src/main/store'

vi.mock('electron', () => ({ shell: {}, app: { getPath: () => tmpdir() } }))

const { AppManager, parseUninstallEntries } = await import('../src/main/apps')

const original = getFetch()
let dir: string
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'rulz-jobs-'))
  // Every GitHub call fails at once: the jobs below stop at their first download.
  setFetch(async () => {
    throw new TypeError('offline')
  })
})
afterEach(async () => {
  setFetch(original)
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

function managed(appDir: string): InstalledInfo {
  return {
    dir: appDir,
    launchPath: join(appDir, 'RulZ Image Organizer.exe'),
    launchArgs: [],
    version: 'v1.0.0',
    method: 'zip',
    managed: true,
    installedAt: '2026-11-13T00:00:00.000Z',
    shortcuts: { desktop: null, startMenu: null },
    components: [],
    needsSetup: false
  }
}

/** Resolves once the app's job has ended. */
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

function options(installDir: string): InstallOptions {
  return {
    dir: installDir,
    desktopShortcut: false,
    startMenuShortcut: false,
    components: [],
    launchAfter: false,
    installPrerequisites: false
  }
}

describe.runIf(process.platform === 'win32')('jobs', () => {
  it("holds the app from an update's first check, so Uninstall or Launch can't slip in while it waits", async () => {
    const { m, store, jobs } = await manager()
    const appDir = join(dir, 'Organizer')
    await mkdir(appDir)
    await store.setInstalled('image-organizer', managed(appDir))
    // The update first asks Windows (PowerShell) what runs from the folder, which takes a moment.
    const updating = m.update('image-organizer')
    await expect(m.uninstall('image-organizer')).rejects.toThrow(/already being installed/)
    await expect(m.launch('image-organizer')).rejects.toThrow(/Wait for RulZ Image Organizer/)
    await updating
    const job = await ended(jobs, 'image-organizer')
    expect(job.action).toBe('update')
    expect(job.phase).toBe('error')
    // Free again afterwards.
    expect(store.getInstalled('image-organizer')?.dir).toBe(appDir)
  }, 30_000)

  it('refuses a second install into a folder another install is filling', async () => {
    const { m, jobs } = await manager()
    const target = join(dir, 'Shared')
    // The first install's GitHub call waits until let go, so its job holds the folder for the checks below.
    let letGo!: () => void
    const gate = new Promise<void>((r) => (letGo = r))
    setFetch(async () => {
      await gate
      throw new TypeError('offline')
    })
    await m.install('image-grid', options(target))
    await expect(m.install('image-tournament', options(target))).rejects.toThrow(/overlaps the folder of RulZ Image Grid/)
    await expect(m.install('image-tournament', options(join(target, 'inside')))).rejects.toThrow(/overlaps/)
    letGo()
    await ended(jobs, 'image-grid')
    // Once that install is over, the folder can be used.
    await m.install('image-tournament', options(target))
    await ended(jobs, 'image-tournament')
  }, 30_000)

  it("refuses to install inside a launcher-made app's folder (its uninstall would recycle it)", async () => {
    const { m, store } = await manager()
    const appDir = join(dir, 'Organizer')
    await mkdir(appDir)
    await store.setInstalled('image-organizer', managed(appDir))
    await expect(m.install('image-grid', options(join(appDir, 'Grid')))).rejects.toThrow(/overlaps the folder of RulZ Image Organizer/)
  })

  it('refuses to update or uninstall an app while another app is installed inside its folder', async () => {
    const { m, store, jobs } = await manager()
    // A setup program's update runs the old uninstaller, which empties the folder (electron-builder's RMDir /r).
    const suite = join(dir, 'LLM Suite')
    const grid = join(suite, 'Grid')
    await mkdir(grid, { recursive: true })
    await store.setInstalled('llm-suite', { ...managed(suite), launchPath: join(suite, 'RulZ LLM Suite.exe'), method: 'installer' })
    await store.setInstalled('image-grid', { ...managed(grid), launchPath: join(grid, 'RulZ Image Grid.exe') })
    await expect(m.update('llm-suite')).rejects.toThrow(/RulZ Image Grid is installed inside RulZ LLM Suite's folder/)
    await expect(m.repair('llm-suite')).rejects.toThrow(/installed inside/)
    await expect(m.uninstall('llm-suite')).rejects.toThrow(/installed inside/)
    expect(jobs.filter((j) => j.appId === 'llm-suite')).toEqual([])
  }, 30_000)

  it("doesn't count an app inside the folder whose files are already gone", async () => {
    const { m, store, jobs } = await manager()
    const organizer = join(dir, 'Organizer')
    await mkdir(organizer)
    await store.setInstalled('image-organizer', managed(organizer))
    await store.setInstalled('image-grid', { ...managed(join(organizer, 'Grid')), launchPath: join(organizer, 'Grid', 'g.exe') })
    await m.update('image-organizer')
    expect((await ended(jobs, 'image-organizer')).action).toBe('update')
  }, 30_000)
})

describe('parseUninstallEntries', () => {
  it('reads the install folder, falling back to the uninstaller', () => {
    const out = [
      'WARNING: something',
      JSON.stringify({ l: '', u: '"C:\\Users\\José\\AppData\\Local\\Programs\\rulz-llm-suite\\Uninstall RulZ LLM Suite.exe" /currentuser', v: '0.15.0' }),
      JSON.stringify({ l: 'D:\\Other', u: '', v: '' })
    ].join('\r\n')
    expect(parseUninstallEntries(out)).toEqual({ location: 'C:\\Users\\José\\AppData\\Local\\Programs\\rulz-llm-suite', version: '0.15.0' })
  })

  it('prefers InstallLocation and skips entries without any folder', () => {
    const out = [JSON.stringify({ l: '', u: '', v: '1' }), JSON.stringify({ l: 'C:\\Apps\\X', u: 'C:\\y\\u.exe', v: '' })].join('\n')
    expect(parseUninstallEntries(out)).toEqual({ location: 'C:\\Apps\\X', version: null })
  })

  it('finds nothing in an empty or failed listing', () => {
    expect(parseUninstallEntries(null)).toBeNull()
    expect(parseUninstallEntries('')).toBeNull()
  })
})
