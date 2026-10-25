import type { ChildProcess } from 'node:child_process'
import { existsSync, readdirSync } from 'node:fs'
import { copyFile, mkdir, readdir, readFile, rename, rm } from 'node:fs/promises'
import { basename, dirname, isAbsolute, join, relative } from 'node:path'
import { shell } from 'electron'
import { CATALOG, findApp } from '@shared/catalog'
import type { LogLine } from '@shared/ipc'
import type {
  AppView,
  CatalogApp,
  ExtraComponent,
  InstalledInfo,
  InstallOptions,
  JobState,
  LatestInfo,
  ReadmeDoc,
  RequirementId,
  ShortcutOptions
} from '@shared/types'
import { exists, extractZip, findFile, flattenSingleFolder, overlay, swapInto } from './archive'
import { download } from './download'
import {
  commitsAhead,
  findAsset,
  GitHubError,
  headCommit,
  isNewer,
  latestRelease,
  rawFile,
  recentReleases,
  sourceArchiveUrl,
  type Release
} from './github'
import { capture, launchDetached, runCommand } from './process'
import { ensureRequirements, requirementLabel } from './requirements'
import { createShortcut, foreignShortcutPaths, removeShortcut, shortcutPaths } from './shortcuts'
import type { Store } from './store'

const LOG_LIMIT = 3000
// electron-builder's silent setup crashed (0xC0000005) whenever it was started from a folder inside the user
// profile (Temp, the profile itself), but ran fine from C:\, C:\coding or C:\Windows.
const SETUP_CWD = process.env.SystemRoot ?? 'C:\\Windows'
const IMAGE_EXT = /\.(png|jpe?g|webp|gif)$/i

interface JobContext {
  signal: AbortSignal
  set: (patch: Partial<JobState>) => void
  log: (line: string) => void
  /** Downloads of this job, deleted when it ends. */
  files: string[]
}

interface LatestEntry {
  info: LatestInfo | null
  error: string | null
  release: Release | null
}

export interface AppManagerOptions {
  store: Store
  catalogDir: string
  downloadsDir: string
  iconsDir: string
  emitApps: (apps: AppView[]) => void
  emitJob: (job: JobState) => void
  emitLog: (line: LogLine) => void
}

function expandEnv(path: string): string {
  return path.replace(/%([^%]+)%/g, (whole, name: string) => process.env[name] ?? whole)
}

/** "Receiving objects:  40% (11/27)" → "Receiving objects:", for collapsing progress counters into one line. */
function progressKey(line: string): string | null {
  return /^(.{3,60}?:)\s+\d{1,3}% /.exec(`${line} `)?.[1] ?? null
}

/** The install can't go on until these are installed by hand. */
class MissingRequirementsError extends Error {
  constructor(readonly missing: RequirementId[]) {
    super(
      `${missing.map(requirementLabel).join(' and ')} ${missing.length > 1 ? 'are' : 'is'} needed first. ` +
        'Follow the install steps, then try again.'
    )
  }
}

function errorText(e: unknown): string {
  if (e instanceof GitHubError && e.status === 404) return "This app isn't published on GitHub yet."
  return e instanceof Error ? e.message : String(e)
}

async function isEmptyDir(dir: string): Promise<boolean> {
  const entries = await readdir(dir).catch(() => null)
  return entries === null || entries.length === 0
}

/** Finds an app installed by its own setup program in the Windows "Installed apps" list. */
async function findUninstallEntry(displayName: string): Promise<{ location: string; version: string | null } | null> {
  for (const hive of ['HKCU', 'HKLM']) {
    const root = `${hive}\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall`
    const hits = await capture('reg', ['query', root, '/s', '/f', displayName, '/d', '/e'])
    for (const key of hits?.split(/\r?\n/).filter((l) => l.startsWith('HKEY_')) ?? []) {
      const values = await capture('reg', ['query', key.trim()])
      const value = (name: string): string | null =>
        new RegExp(`^\\s*${name}\\s+REG_\\w+\\s+(.*)$`, 'im').exec(values ?? '')?.[1]?.trim() ?? null
      // electron-builder leaves InstallLocation empty here; the uninstaller sits in the install folder.
      const uninstaller = /^"?([^"]+\.exe)/i.exec(value('UninstallString') ?? '')?.[1]
      const location = value('InstallLocation') || (uninstaller ? dirname(uninstaller) : null)
      if (value('DisplayName') === displayName && location) return { location, version: value('DisplayVersion') }
    }
  }
  return null
}

/** Installs, updates, launches and removes the apps in the catalog. */
export class AppManager {
  private latest = new Map<string, LatestEntry>()
  private jobs = new Map<string, { state: JobState; abort: AbortController }>()
  private logs = new Map<string, string[]>()
  private running = new Map<string, ChildProcess>()
  private readmes = new Map<string, ReadmeDoc>()
  private screenshots = new Map<string, string[]>()

  constructor(private readonly o: AppManagerOptions) {}

  async init(): Promise<void> {
    for (const app of CATALOG) {
      const dir = join(this.o.catalogDir, app.id, 'screenshots')
      const files = existsSync(dir) ? readdirSync(dir).filter((f) => IMAGE_EXT.test(f)).sort() : []
      this.screenshots.set(app.id, files.map((f) => `rulz://catalog/${app.id}/screenshots/${encodeURIComponent(f)}`))
    }
    await this.reconcileInstalls()
  }

  // ---------- Views ----------

  views(): AppView[] {
    return CATALOG.map((app) => this.view(app))
  }

  private view(app: CatalogApp): AppView {
    const installed = this.o.store.getInstalled(app.id)
    const latest = this.latest.get(app.id)
    return {
      app,
      icon: `rulz://catalog/${app.id}/icon.png`,
      screenshots: this.screenshots.get(app.id) ?? [],
      installed,
      latest: latest?.info ?? null,
      latestError: latest?.error ?? null,
      updateAvailable: this.updateAvailable(app, installed, latest?.info ?? null),
      running: this.running.has(app.id),
      job: this.jobs.get(app.id)?.state ?? null
    }
  }

  private updateAvailable(app: CatalogApp, installed: InstalledInfo | null, latest: LatestInfo | null): boolean {
    if (!installed?.managed || !installed.version || !latest) return false
    if (app.install.kind === 'source') return latest.version !== installed.version
    return isNewer(latest.version, installed.version)
  }

  private changed(): void {
    this.o.emitApps(this.views())
  }

  defaultDir(appId: string): string {
    const app = this.app(appId)
    return join(this.o.store.getSettings().installRoot, app.folderName)
  }

  getLog(appId: string): string[] {
    return this.logs.get(appId) ?? []
  }

  private app(appId: string): CatalogApp {
    const app = findApp(appId)
    if (!app) throw new Error(`Unknown app: ${appId}`)
    return app
  }

  /** Drops records whose files are gone and picks up apps installed outside the launcher. */
  private async reconcileInstalls(): Promise<void> {
    for (const app of CATALOG) {
      const info = this.o.store.getInstalled(app.id)
      if (info && !existsSync(info.launchPath) && !info.needsSetup) await this.o.store.setInstalled(app.id, null)
      if (this.o.store.getInstalled(app.id) || this.o.store.isDetectHidden(app.id)) continue

      let launchPath: string | null = null
      let version: string | null = null
      for (const p of app.detect ?? []) {
        if (existsSync(expandEnv(p))) {
          launchPath = expandEnv(p)
          break
        }
      }
      if (!launchPath && app.install.kind === 'installer') {
        const entry = await findUninstallEntry(app.install.uninstallName)
        const exe = entry && join(entry.location, app.launch.target)
        if (exe && existsSync(exe)) {
          launchPath = exe
          version = entry.version ? `v${entry.version}` : null
        }
      }
      if (launchPath) await this.o.store.setInstalled(app.id, this.externalInfo(app, launchPath, version))
    }
  }

  private externalInfo(app: CatalogApp, launchPath: string, version: string | null): InstalledInfo {
    // A path ending in the catalog's launch target (e.g. bin\App.exe) has its install folder above that.
    const target = app.launch.target.toLowerCase()
    const matches = launchPath.toLowerCase().endsWith(`\\${target}`)
    return {
      dir: matches ? launchPath.slice(0, launchPath.length - target.length - 1) : dirname(launchPath),
      launchPath,
      launchArgs: matches ? (app.launch.args ?? []) : [],
      version,
      method: 'external',
      managed: false,
      installedAt: new Date().toISOString(),
      shortcuts: { desktop: null, startMenu: null },
      components: [],
      needsSetup: false
    }
  }

  // ---------- GitHub ----------

  async refreshLatest(): Promise<void> {
    await this.reconcileInstalls()
    await Promise.all(
      CATALOG.map(async (app) => {
        this.latest.set(app.id, await this.fetchLatest(app))
      })
    )
    this.changed()
  }

  private async fetchLatest(app: CatalogApp): Promise<LatestEntry> {
    if (!app.repo) return { info: null, error: "This app isn't published on GitHub yet.", release: null }
    try {
      if (app.install.kind === 'source') {
        // Source installs follow the branch; the label names the latest release when there is one.
        const [c, release] = await Promise.all([headCommit(app.repo, app.branch), latestRelease(app.repo)])
        const ahead = release ? await commitsAhead(app.repo, release.tag, c.sha).catch(() => null) : null
        let label = `${app.branch} @ ${c.sha.slice(0, 7)}`
        if (release && ahead === 0) label = release.tag
        else if (release && ahead) label = `${release.tag} + ${ahead} commit${ahead === 1 ? '' : 's'}`
        return {
          info: {
            version: c.sha,
            label,
            date: c.date,
            downloadSize: null,
            url: `https://github.com/${app.repo}/commits/${app.branch}`
          },
          error: null,
          release: null
        }
      }
      const release = await latestRelease(app.repo)
      if (!release) return { info: null, error: 'No release has been published yet.', release: null }
      const asset = findAsset(release, app.install.asset)
      return {
        info: {
          version: release.tag,
          label: release.tag,
          date: release.date,
          downloadSize: asset?.size ?? null,
          url: release.htmlUrl
        },
        error: asset ? null : 'The latest release has no Windows download.',
        release
      }
    } catch (e) {
      return { info: null, error: errorText(e), release: null }
    }
  }

  async readme(appId: string, live: boolean): Promise<ReadmeDoc> {
    const app = this.app(appId)
    const linkBase = app.repo ? `https://github.com/${app.repo}/blob/${app.branch}/` : null
    if (live && app.repo) {
      const cached = this.readmes.get(appId)
      if (cached) return cached
      try {
        const markdown = await rawFile(app.repo, app.branch, 'README.md')
        const doc: ReadmeDoc = {
          markdown,
          assetBase: `https://raw.githubusercontent.com/${app.repo}/${app.branch}/`,
          linkBase,
          source: 'github'
        }
        this.readmes.set(appId, doc)
        return doc
      } catch {
        // Offline or not published: the copy shipped with the launcher.
      }
    }
    const markdown = await readFile(join(this.o.catalogDir, appId, 'README.md'), 'utf8').catch(() => `# ${app.name}\n\n${app.description}`)
    return { markdown, assetBase: `rulz://catalog/${appId}/repo/`, linkBase, source: 'bundled' }
  }

  // ---------- Jobs ----------

  private startJob(appId: string, action: JobState['action'], work: (ctx: JobContext) => Promise<void>): void {
    if (this.jobs.has(appId)) throw new Error('This app is already being installed or updated.')
    const abort = new AbortController()
    const job = { state: { appId, action, phase: 'resolving', label: 'Starting…', progress: null } as JobState, abort }
    const log: string[] = []
    this.jobs.set(appId, job)
    this.logs.set(appId, log)

    const ctx: JobContext = {
      files: [],
      signal: abort.signal,
      set: (patch) => {
        job.state = { ...job.state, ...patch }
        this.o.emitJob(job.state)
      },
      log: (line) => {
        const key = progressKey(line)
        const replace = !!key && log.length > 0 && progressKey(log[log.length - 1]) === key
        if (replace) log[log.length - 1] = line
        else log.push(line)
        if (log.length > LOG_LIMIT) log.splice(0, log.length - LOG_LIMIT)
        this.o.emitLog({ appId, line, replace })
      }
    }
    this.changed()
    void work(ctx)
      .then(
        () => ctx.set({ phase: 'done', label: 'Done', progress: 1 }),
        (e) => {
          const cancelled = abort.signal.aborted
          ctx.log(cancelled ? 'Cancelled.' : `Error: ${errorText(e)}`)
          ctx.set({
            phase: cancelled ? 'cancelled' : 'error',
            label: cancelled ? 'Cancelled' : 'Failed',
            error: cancelled ? undefined : errorText(e),
            progress: null,
            hasLog: log.length > 1,
            missing: e instanceof MissingRequirementsError ? e.missing : undefined
          })
        }
      )
      .finally(() => {
        this.jobs.delete(appId)
        this.changed()
        for (const f of ctx.files) void rm(f, { force: true }).catch(() => undefined)
      })
  }

  cancel(appId: string): void {
    this.jobs.get(appId)?.abort.abort()
  }

  private async fetchFile(ctx: JobContext, url: string, name: string, size: number | null, label: string): Promise<string> {
    const file = join(this.o.downloadsDir, name)
    ctx.files.push(file)
    ctx.log(`Downloading ${url}`)
    ctx.set({ phase: 'downloading', label, progress: size ? 0 : null, bytes: 0, total: size ?? undefined })
    await download(url, file, {
      signal: ctx.signal,
      expectedSize: size,
      onProgress: (p) =>
        ctx.set({ progress: p.total ? p.bytes / p.total : null, bytes: p.bytes, total: p.total ?? undefined, speed: p.speed })
    })
    ctx.set({ speed: undefined })
    return file
  }

  private async requireRelease(app: CatalogApp, pattern: string): Promise<{ release: Release; asset: Release['assets'][number] }> {
    let entry = this.latest.get(app.id)
    if (!entry?.release) {
      entry = await this.fetchLatest(app)
      this.latest.set(app.id, entry)
    }
    if (!entry.release) throw new Error(entry.error ?? 'No release found.')
    const asset = findAsset(entry.release, pattern)
    if (!asset) throw new Error(`The ${entry.release.tag} release has no Windows download.`)
    return { release: entry.release, asset }
  }

  // ---------- Install / update ----------

  async install(appId: string, opts: InstallOptions): Promise<void> {
    const app = this.app(appId)
    if (this.o.store.getInstalled(appId)) throw new Error(`${app.name} is already installed.`)
    if (!isAbsolute(opts.dir) || !/^[a-z]:\\/i.test(opts.dir)) throw new Error('Choose a full folder path, such as D:\\Apps.')
    if (!(await isEmptyDir(opts.dir))) {
      throw new Error(`Choose an empty folder: ${opts.dir} already has files in it.`)
    }
    await this.o.store.setDetectHidden(appId, false)
    this.startJob(appId, 'install', async (ctx) => {
      try {
        await this.runInstall(ctx, app, opts, null)
      } catch (e) {
        // The folder was empty before, so anything in it now came from this attempt.
        if (!this.o.store.getInstalled(appId)) {
          await rm(opts.dir, { recursive: true, force: true }).catch(() => undefined)
          await rm(`${opts.dir}.new`, { recursive: true, force: true }).catch(() => undefined)
        }
        throw e
      }
    })
  }

  update(appId: string): void {
    const app = this.app(appId)
    const info = this.o.store.getInstalled(appId)
    if (!info?.managed) throw new Error(`${app.name} wasn't installed by the launcher, so it can't update it.`)
    if (this.running.has(appId)) throw new Error(`Close ${app.name} first, then update.`)
    const opts: InstallOptions = {
      dir: info.dir,
      desktopShortcut: !!info.shortcuts.desktop,
      startMenuShortcut: !!info.shortcuts.startMenu,
      components: [],
      launchAfter: false,
      installPrerequisites: true
    }
    this.latest.delete(appId) // always fetch the newest release for an update
    this.startJob(appId, 'update', (ctx) => this.runInstall(ctx, app, opts, info))
  }

  /** Runs the setup steps again (after a failed or cancelled setup), or reinstalls the release. */
  repair(appId: string): void {
    const app = this.app(appId)
    const info = this.o.store.getInstalled(appId)
    if (!info?.managed) throw new Error(`${app.name} wasn't installed by the launcher.`)
    if (app.install.kind !== 'source') return this.update(appId)
    this.startJob(appId, 'repair', async (ctx) => {
      await this.ensurePrerequisites(ctx, app)
      await this.runSetupSteps(ctx, app, info)
      const launchPath = await this.resolveLaunch(app, info.dir)
      await this.o.store.setInstalled(appId, { ...info, launchPath, needsSetup: false })
    })
  }

  /** Installs missing prerequisites with winget; stops the job if any are still missing. */
  private async ensurePrerequisites(ctx: JobContext, app: CatalogApp): Promise<void> {
    ctx.set({ phase: 'prerequisites', label: 'Checking prerequisites…', progress: null })
    const missing = await ensureRequirements(
      app.requirements,
      ctx.log,
      (label) => ctx.set({ label: `Installing ${label}…` }),
      ctx.signal
    )
    if (missing.length) throw new MissingRequirementsError(missing)
  }

  private async runInstall(ctx: JobContext, app: CatalogApp, opts: InstallOptions, prev: InstalledInfo | null): Promise<void> {
    const dir = opts.dir
    if (opts.installPrerequisites) await this.ensurePrerequisites(ctx, app)
    const spec = app.install
    let version: string | null = null
    let method: InstalledInfo['method']
    let launchPath: string

    if (spec.kind === 'zip') {
      ctx.set({ phase: 'resolving', label: 'Finding the latest release…' })
      const { release, asset } = await this.requireRelease(app, spec.asset)
      const zip = await this.fetchFile(ctx, asset.url, asset.name, asset.size, `Downloading ${release.tag}`)
      const staged = `${dir}.new`
      await rm(staged, { recursive: true, force: true })
      ctx.set({ phase: 'extracting', label: 'Extracting…', progress: 0 })
      await extractZip(zip, staged, (f) => ctx.set({ progress: f }))
      await flattenSingleFolder(staged)
      const rel = relative(staged, await this.resolveLaunch(app, staged))
      ctx.set({ phase: 'installing', label: 'Installing…', progress: null })
      await mkdir(dirname(dir), { recursive: true })
      await swapInto(staged, dir)
      await rm(zip, { force: true })
      launchPath = join(dir, rel)
      version = release.tag
      method = 'zip'
    } else if (spec.kind === 'installer') {
      ctx.set({ phase: 'resolving', label: 'Finding the latest release…' })
      const { release, asset } = await this.requireRelease(app, spec.asset)
      const setup = await this.fetchFile(ctx, asset.url, asset.name, asset.size, `Downloading ${release.tag}`)
      ctx.set({ phase: 'installing', label: 'Running the setup program…', progress: null })
      ctx.log(`> ${basename(setup)} /S /currentuser /D="${dir}"`)
      const code = await runCommand([setup, '/S', '/currentuser', `/D="${dir}"`], {
        cwd: SETUP_CWD,
        signal: ctx.signal,
        gui: true
      })
      if (code !== 0) throw new Error(`The setup program exited with code ${code}.`)
      launchPath = join(dir, app.launch.target)
      for (let i = 0; i < 60 && !existsSync(launchPath); i++) await new Promise((r) => setTimeout(r, 500))
      if (!existsSync(launchPath)) throw new Error(`Setup finished, but ${launchPath} wasn't created.`)
      // The setup program makes its own shortcuts; the launcher's options decide which ones exist.
      for (const lnk of foreignShortcutPaths(spec.shortcutName)) await removeShortcut(lnk)
      await rm(setup, { force: true })
      version = release.tag
      method = 'installer'
    } else {
      const fetched = await this.fetchSource(ctx, app, dir, prev)
      version = fetched.version
      method = fetched.method
      // Saved before setup, so a failed setup can be repaired instead of leaving an unknown folder behind.
      const partial: InstalledInfo = {
        dir,
        launchPath: join(dir, app.launch.target),
        launchArgs: app.launch.args ?? [],
        version,
        method,
        managed: true,
        installedAt: prev?.installedAt ?? new Date().toISOString(),
        shortcuts: prev?.shortcuts ?? { desktop: null, startMenu: null },
        components: prev?.components ?? [],
        needsSetup: true
      }
      await this.o.store.setInstalled(app.id, partial)
      await this.runSetupSteps(ctx, app, partial)
      launchPath = await this.resolveLaunch(app, dir)
    }

    const components = [...(prev?.components ?? [])]
    for (const c of app.components ?? []) {
      if (!opts.components.includes(c.id)) continue
      await this.installComponent(ctx, c, dir)
      if (!components.includes(c.id)) components.push(c.id)
    }

    const info: InstalledInfo = {
      dir,
      launchPath,
      launchArgs: app.launch.args ?? [],
      version,
      method,
      managed: true,
      installedAt: prev?.installedAt ?? new Date().toISOString(),
      shortcuts: prev?.shortcuts ?? { desktop: null, startMenu: null },
      components,
      needsSetup: false
    }
    ctx.set({ phase: 'shortcuts', label: 'Creating shortcuts…', progress: null })
    // A shortcut that can't be made isn't worth undoing the install for.
    info.shortcuts = await this.applyShortcuts(
      app,
      info,
      { desktop: opts.desktopShortcut, startMenu: opts.startMenuShortcut },
      (e) => ctx.log(`Warning: ${errorText(e)}`)
    )
    await this.o.store.setInstalled(app.id, info)
    ctx.log(`Installed ${app.name} ${version ?? ''} in ${dir}`)
    const latest = await this.fetchLatest(app)
    this.latest.set(app.id, latest)
    if (latest.info && latest.info.version === version && latest.info.label !== version) {
      await this.o.store.setInstalled(app.id, { ...info, versionLabel: latest.info.label })
    }
    if (opts.launchAfter) await this.launch(app.id).catch((e) => ctx.log(`Couldn't start it: ${errorText(e)}`))
  }

  /** Gets the repository: git clone/pull when git is available, otherwise the source archive. */
  private async fetchSource(
    ctx: JobContext,
    app: CatalogApp,
    dir: string,
    prev: InstalledInfo | null
  ): Promise<{ version: string; method: 'git' | 'archive' }> {
    if (!app.repo) throw new Error("This app isn't published on GitHub yet.")
    const hasGit = !!(await capture('git', ['--version']))
    const gitDir = existsSync(join(dir, '.git'))

    if (hasGit && (prev ? gitDir : true)) {
      if (prev) {
        ctx.set({ phase: 'downloading', label: 'Pulling the latest changes…', progress: null })
        ctx.log('> git pull --ff-only')
        const code = await runCommand(['git', 'pull', '--ff-only'], { cwd: dir, signal: ctx.signal, onLine: ctx.log })
        if (code !== 0) throw new Error('git pull failed (local changes in the folder?). See the log for details.')
      } else {
        ctx.set({ phase: 'downloading', label: 'Cloning the repository…', progress: null })
        await mkdir(dir, { recursive: true })
        const url = `https://github.com/${app.repo}.git`
        ctx.log(`> git clone --depth 1 --branch ${app.branch} ${url}`)
        const code = await runCommand(['git', 'clone', '--progress', '--depth', '1', '--branch', app.branch, url, '.'], {
          cwd: dir,
          signal: ctx.signal,
          onLine: ctx.log
        })
        if (code !== 0) throw new Error('git clone failed. See the log for details.')
      }
      const sha = await capture('git', ['-C', dir, 'rev-parse', 'HEAD'])
      return { version: sha ?? 'unknown', method: 'git' }
    }

    ctx.set({ phase: 'resolving', label: 'Finding the latest version…' })
    const commit = await headCommit(app.repo, app.branch)
    const zip = await this.fetchFile(
      ctx,
      sourceArchiveUrl(app.repo, commit.sha),
      `${app.id}-${commit.sha.slice(0, 7)}.zip`,
      null,
      `Downloading ${app.branch} @ ${commit.sha.slice(0, 7)}`
    )
    const staged = `${dir}.new`
    await rm(staged, { recursive: true, force: true })
    ctx.set({ phase: 'extracting', label: 'Extracting…', progress: 0 })
    await extractZip(zip, staged, (f) => ctx.set({ progress: f }))
    await flattenSingleFolder(staged)
    ctx.set({ phase: 'installing', label: 'Copying files…', progress: null })
    if (prev) {
      // Keep what the app made in its folder (venv, models, outputs); replace the source files.
      await overlay(staged, dir)
      await rm(staged, { recursive: true, force: true })
    } else {
      await mkdir(dirname(dir), { recursive: true })
      await rm(dir, { recursive: true, force: true }) // empty (checked before the install started)
      await rename(staged, dir)
    }
    await rm(zip, { force: true })
    return { version: commit.sha, method: 'archive' }
  }

  private async runSetupSteps(ctx: JobContext, app: CatalogApp, info: InstalledInfo): Promise<void> {
    if (app.install.kind !== 'source') return
    for (const step of app.install.setup) {
      ctx.set({ phase: 'setup', label: `${step.label}…`, progress: null })
      ctx.log(`> ${step.run.join(' ')}`)
      const code = await runCommand(step.run, { cwd: info.dir, signal: ctx.signal, onLine: ctx.log })
      if (code !== 0) throw new Error(`${step.label} failed (exit code ${code}). See the log for details.`)
    }
  }

  private async installComponent(ctx: JobContext, c: ExtraComponent, dir: string): Promise<void> {
    ctx.set({ phase: 'resolving', label: `Finding ${c.label}…`, progress: null })
    const releases = await recentReleases(c.repo, 15)
    const release = releases.find((r) => c.assets.every((p) => findAsset(r, p)))
    if (!release) throw new Error(`No recent ${c.repo} release has the ${c.label} downloads.`)
    for (const pattern of c.assets) {
      const asset = findAsset(release, pattern)!
      const zip = await this.fetchFile(ctx, asset.url, asset.name, asset.size, `Downloading ${c.label} ${release.tag}`)
      ctx.set({ phase: 'extracting', label: `Extracting ${asset.name}…`, progress: 0 })
      await extractZip(zip, join(dir, c.dest), (f) => ctx.set({ progress: f }))
      await rm(zip, { force: true })
    }
  }

  /** The program to launch: the catalog path, or the same file name found in a subfolder. */
  private async resolveLaunch(app: CatalogApp, dir: string): Promise<string> {
    const direct = join(dir, app.launch.target)
    if (await exists(direct)) return direct
    const found = await findFile(dir, basename(app.launch.target))
    if (found) return found
    throw new Error(`${app.launch.target} wasn't found in ${dir}.`)
  }

  // ---------- Shortcuts ----------

  private async shortcutIcon(app: CatalogApp, info: InstalledInfo): Promise<string | undefined> {
    if (app.launch.icon && existsSync(join(info.dir, app.launch.icon))) return join(info.dir, app.launch.icon)
    if (/\.exe$/i.test(info.launchPath) && !/pythonw?\.exe$/i.test(info.launchPath)) return info.launchPath
    // Batch files and scripts have no icon of their own; use a copy of the catalog icon.
    const src = join(this.o.catalogDir, app.id, 'icon.ico')
    if (!existsSync(src)) return undefined
    const dest = join(this.o.iconsDir, `${app.id}.ico`)
    await mkdir(this.o.iconsDir, { recursive: true })
    await copyFile(src, dest)
    return dest
  }

  private async applyShortcuts(
    app: CatalogApp,
    info: InstalledInfo,
    opts: ShortcutOptions,
    onError?: (e: unknown) => void
  ): Promise<InstalledInfo['shortcuts']> {
    const paths = shortcutPaths(app.name)
    const target = {
      target: info.launchPath,
      args: info.launchArgs,
      cwd: info.dir,
      icon: await this.shortcutIcon(app, info),
      description: app.tagline
    }
    const result: InstalledInfo['shortcuts'] = { desktop: null, startMenu: null }
    for (const where of ['desktop', 'startMenu'] as const) {
      if (opts[where]) {
        try {
          await createShortcut(paths[where], target)
          result[where] = paths[where]
        } catch (e) {
          if (!onError) throw e
          onError(e)
        }
      } else {
        await removeShortcut(info.shortcuts[where])
      }
    }
    return result
  }

  async setShortcuts(appId: string, opts: ShortcutOptions): Promise<void> {
    const app = this.app(appId)
    const info = this.o.store.getInstalled(appId)
    if (!info) throw new Error(`${app.name} isn't installed.`)
    const shortcuts = await this.applyShortcuts(app, info, opts)
    await this.o.store.setInstalled(appId, { ...info, shortcuts })
    this.changed()
  }

  // ---------- Launch / remove ----------

  async launch(appId: string): Promise<void> {
    const app = this.app(appId)
    const info = this.o.store.getInstalled(appId)
    if (!info) throw new Error(`${app.name} isn't installed.`)
    if (info.needsSetup) throw new Error(`${app.name}'s setup didn't finish. Use Repair to run it again.`)
    if (!existsSync(info.launchPath)) throw new Error(`${info.launchPath} is missing. Reinstall ${app.name}.`)
    const child = launchDetached(info.launchPath, info.launchArgs, info.dir)
    await new Promise<void>((resolve, reject) => {
      child.once('spawn', resolve)
      child.once('error', (e) => reject(new Error(`Couldn't start ${app.name}: ${e.message}`)))
    })
    this.running.set(appId, child)
    child.once('exit', () => {
      if (this.running.get(appId) === child) this.running.delete(appId)
      this.changed()
    })
    this.changed()
  }

  uninstall(appId: string): void {
    const app = this.app(appId)
    const info = this.o.store.getInstalled(appId)
    if (!info) throw new Error(`${app.name} isn't installed.`)
    if (info.managed && this.running.has(appId)) throw new Error(`Close ${app.name} first, then uninstall it.`)
    this.startJob(appId, 'uninstall', async (ctx) => {
      ctx.set({ phase: 'removing', label: 'Removing shortcuts…' })
      await removeShortcut(info.shortcuts.desktop)
      await removeShortcut(info.shortcuts.startMenu)
      if (info.managed) {
        if (app.install.kind === 'installer') {
          const uninstaller = join(info.dir, `Uninstall ${app.install.uninstallName}.exe`)
          if (existsSync(uninstaller)) {
            ctx.set({ label: 'Running the uninstaller…' })
            ctx.log(`> ${basename(uninstaller)} /S /currentuser`)
            await runCommand([uninstaller, '/S', '/currentuser'], { cwd: SETUP_CWD, signal: ctx.signal, gui: true })
            // NSIS uninstallers copy themselves to %TEMP% and return at once; wait for the files to go.
            for (let i = 0; i < 120 && existsSync(info.launchPath); i++) await new Promise((r) => setTimeout(r, 500))
          }
        }
        if (existsSync(info.dir)) {
          ctx.set({ label: 'Moving the files to the Recycle Bin…' })
          ctx.log(`Recycling ${info.dir}`)
          try {
            await shell.trashItem(info.dir)
          } catch (e) {
            throw new Error(`Couldn't remove ${info.dir}: ${errorText(e)}. Close the app and try again.`)
          }
        }
      }
      await this.o.store.setInstalled(appId, null)
      // Removing a found install only forgets it; don't find it again on the next refresh.
      if (!info.managed) await this.o.store.setDetectHidden(appId, true)
    })
  }

  /** Registers a program the user picked as this app's install. */
  async locate(appId: string, file: string): Promise<void> {
    const app = this.app(appId)
    if (!existsSync(file)) throw new Error(`${file} doesn't exist.`)
    if (!/\.(exe|bat|cmd)$/i.test(file)) throw new Error('Pick the program file (.exe or .bat).')
    await this.o.store.setDetectHidden(appId, false)
    await this.o.store.setInstalled(appId, this.externalInfo(app, file, null))
    this.changed()
  }

  async openFolder(appId: string): Promise<void> {
    const info = this.o.store.getInstalled(appId)
    if (!info) throw new Error("It isn't installed.")
    const err = await shell.openPath(info.dir)
    if (err) throw new Error(err)
  }
}
