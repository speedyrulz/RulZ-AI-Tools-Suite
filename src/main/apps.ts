import type { ChildProcess } from 'node:child_process'
import { existsSync, readdirSync } from 'node:fs'
import { copyFile, mkdir, readdir, readFile, rm, stat } from 'node:fs/promises'
import { basename, dirname, join, relative } from 'node:path'
import { shell } from 'electron'
import { CATALOG, findApp, launchEnv } from '@shared/catalog'
import { requirementRefLabel } from '@shared/requirements'
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
import {
  exists,
  extractZip,
  findFile,
  flattenSingleFolder,
  installFolder,
  overlay,
  renameRetrying,
  stagingDir,
  swapInto
} from './archive'
import { download, publishedSha256 } from './download'
import {
  branchSha,
  commitsSince,
  dropSharedLists,
  etagSnapshot,
  findAsset,
  GitHubError,
  headCommit,
  isNewer,
  latestRelease,
  loadEtags,
  rawFile,
  recentReleases,
  releaseVersion,
  sourceArchiveUrl,
  type EtagEntry,
  type Release
} from './github'
import { capture, launchDetached, powershell, programsRunningFrom, psString, runCommand } from './process'
import { ensureRequirements, requirementLabel } from './requirements'
import { createShortcut, foreignShortcutPaths, removeShortcut, shortcutPaths } from './shortcuts'
import type { Store } from './store'

const LOG_LIMIT = 3000
/** A log is cut back to LOG_LIMIT lines once it is this many longer, not on every line (each cut moves every line). */
const LOG_SLACK = 500
const LATEST_TTL_MS = 30 * 60_000
/** After a README can't be fetched from GitHub (offline, say), the bundled copy is shown at once for this long. */
const README_RETRY_MS = 5 * 60_000
/** Log lines are sent to the window in batches: a pip install can print hundreds a second. */
const LOG_FLUSH_MS = 100
/** Leftover partial downloads (kept so a retry can resume) are deleted after this long. */
const PART_MAX_AGE_MS = 7 * 24 * 3600_000
/**
 * Any other file in the downloads folder is left over (a finished download is deleted when its job ends, so only a
 * crash leaves one; launcher setups from before the updater deleted them) and is deleted after this long.
 */
const LEFTOVER_MAX_AGE_MS = 3600_000
/** How long quitting waits for cancelled jobs to stop their programs. */
const CANCEL_WAIT_MS = 15_000

/** The last GitHub answers per app, with when each was fetched. Older launchers wrote `at` + bare entries. */
interface LatestCache {
  at?: number
  entries: Record<string, LatestInfo | { info: LatestInfo; at: number }>
}

/** The asset that lists the SHA-256 of a release's other files. */
const SUMS_ASSET = '^SHA256SUMS$'
const SETUP_CWD = process.env.SystemRoot ?? 'C:\\Windows'
const IMAGE_EXT = /\.(png|jpe?g|webp|gif)$/i
// electron-builder's NSIS setups now and then crash at startup inside their System.dll plugin
// (0xC0000005, before doing anything); running them again works.
const ACCESS_VIOLATION = 0xc0000005

interface JobContext {
  appId: string
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
  /** When `info` was fetched (0 when it's unknown or the fetch failed). */
  at?: number
}

/**
 * The entry to keep after a fetch: a failed one (rate limit, offline) keeps the versions known before, with the
 * error for display, and stays due for another try.
 */
export function mergeLatest(prev: LatestEntry | undefined, next: LatestEntry): LatestEntry {
  if (next.info || !prev?.info) return next
  return { info: prev.info, error: next.error, release: prev.release, at: prev.at ?? 0 }
}

/** Reads the latest-versions cache, either format. */
export function readLatestCache(cache: LatestCache | null): Map<string, LatestEntry> {
  const out = new Map<string, LatestEntry>()
  if (!cache?.entries) return out
  for (const [id, e] of Object.entries(cache.entries)) {
    if (!e) continue
    const entry = 'info' in e ? e : { info: e, at: cache.at ?? 0 }
    if (entry.info) out.set(id, { info: entry.info, error: null, release: null, at: entry.at ?? 0 })
  }
  return out
}

export interface AppManagerOptions {
  store: Store
  catalogDir: string
  downloadsDir: string
  iconsDir: string
  emitApps: (apps: AppView[]) => void
  emitJob: (job: JobState) => void
  /** Log lines in order, batched (every 100 ms at most, and before each job update). */
  emitLog: (lines: LogLine[]) => void
}

function expandEnv(path: string): string {
  return path.replace(/%([^%]+)%/g, (whole, name: string) => process.env[name] ?? whole)
}

/** "Receiving objects:  40% (11/27)" → "Receiving objects:", for collapsing progress counters into one line. */
function progressKey(line: string): string | null {
  return /^(.{3,60}?:)\s+\d{1,3}% /.exec(`${line} `)?.[1] ?? null
}

/**
 * Waits (checking twice a second) until `done` is true or `ms` have passed. A cancelled job stops waiting at once:
 * quitting gives jobs only a few seconds to stop.
 */
async function waitUntil(done: () => boolean, ms: number, signal: AbortSignal): Promise<void> {
  for (let waited = 0; waited < ms && !done(); waited += 500) {
    if (signal.aborted) throw new Error('Cancelled')
    await new Promise((r) => setTimeout(r, 500))
  }
  if (signal.aborted) throw new Error('Cancelled')
}

/** Runs an NSIS setup or uninstaller silently, retrying the occasional startup crash. */
async function runSetupProgram(run: string[], ctx: JobContext): Promise<number> {
  let code = 0
  for (let attempt = 1; attempt <= 3; attempt++) {
    code = await runCommand(run, { cwd: SETUP_CWD, signal: ctx.signal, gui: true })
    if ((code | 0) !== (ACCESS_VIOLATION | 0)) break
    ctx.log(`The setup program crashed while starting (attempt ${attempt} of 3).`)
  }
  return code
}

/** The install can't go on until these are installed by hand. */
class MissingRequirementsError extends Error {
  /** `labels` name what is needed, in the app's terms ("Python 3.11+"). */
  constructor(
    readonly missing: RequirementId[],
    labels: string[]
  ) {
    super(
      `${labels.join(' and ')} ${missing.length > 1 ? 'are' : 'is'} needed first. ` +
        'Follow the install steps, then try again.'
    )
  }
}

function errorText(e: unknown): string {
  if (e instanceof GitHubError && e.status === 404) return "This app isn't published on GitHub yet."
  return e instanceof Error ? e.message : String(e)
}

/**
 * The first usable entry in findUninstallEntry's output: one JSON object per line, with the install location (l),
 * uninstall command (u) and version (v). Pure, for tests.
 */
export function parseUninstallEntries(out: string | null): { location: string; version: string | null } | null {
  for (const line of out?.split(/\r?\n/) ?? []) {
    let e: { l?: unknown; u?: unknown; v?: unknown }
    try {
      e = JSON.parse(line) as typeof e
    } catch {
      continue
    }
    const text = (v: unknown): string => (typeof v === 'string' ? v.trim() : '')
    // electron-builder leaves InstallLocation empty here; the uninstaller sits in the install folder.
    const uninstaller = /^"?([^"]+\.exe)/i.exec(text(e.u))?.[1]
    const location = text(e.l) || (uninstaller ? dirname(uninstaller) : null)
    if (location) return { location, version: text(e.v) || null }
  }
  return null
}

/** Finds an app installed by its own setup program in the Windows "Installed apps" list. */
async function findUninstallEntry(displayName: string): Promise<{ location: string; version: string | null } | null> {
  // One PowerShell call reads both lists, the per-user one first (it wins when both have the app). Its output is
  // UTF-8: reg.exe's would garble an install path under C:\Users\José.
  const out = await powershell(
    "Get-ItemProperty 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*', " +
      "'HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*' -ErrorAction SilentlyContinue | " +
      `Where-Object { $_.DisplayName -eq ${psString(displayName)} } | ` +
      'ForEach-Object { ConvertTo-Json -Compress @{ l = [string]$_.InstallLocation; u = [string]$_.UninstallString; v = [string]$_.DisplayVersion } }'
  )
  return parseUninstallEntries(out)
}

/** Installs, updates, launches and removes the apps in the catalog. */
export class AppManager {
  private latest = new Map<string, LatestEntry>()
  private jobs = new Map<string, { state: JobState; abort: AbortController; done: Promise<void> }>()
  private logs = new Map<string, string[]>()
  private running = new Map<string, ChildProcess>()
  private readmes = new Map<string, ReadmeDoc>()
  private screenshots = new Map<string, string[]>()
  /** Finding installs made outside the launcher (searches the registry); done once, after the window is up. */
  private detected: Promise<void> = Promise.resolve()
  private pendingLogs: LogLine[] = []
  private logTimer: ReturnType<typeof setTimeout> | null = null
  /** When each app's README last failed to come from GitHub. */
  private readmeFailed = new Map<string, number>()
  /** The GitHub check in progress, shared by calls made while it runs. */
  private refreshing: { force: boolean; done: Promise<void> } | null = null
  /** Set once quitting has begun: no new jobs. */
  private closing = false
  /** Apps between the first check of an action and its job's start (the checks wait for PowerShell, say). */
  private preparing = new Set<string>()
  /** The folders that running installs fill, by app. */
  private installDirs = new Map<string, string>()

  constructor(private readonly o: AppManagerOptions) {}

  /** Quick: everything the first window needs. The slower search for outside installs runs afterwards. */
  async init(): Promise<void> {
    for (const app of CATALOG) {
      const dir = join(this.o.catalogDir, app.id, 'screenshots')
      const files = existsSync(dir) ? readdirSync(dir).filter((f) => IMAGE_EXT.test(f)).sort() : []
      this.screenshots.set(app.id, files.map((f) => `rulz://catalog/${app.id}/screenshots/${encodeURIComponent(f)}`))
    }
    // The last GitHub answers, so cards have versions at once and repeated launches don't use up the API's
    // 60 requests an hour; with their ETags, asking again is a short answer while nothing changed (still a request).
    const cache = await this.o.store.loadCache<LatestCache | null>('latest-cache.json', null)
    for (const [id, entry] of readLatestCache(cache)) this.latest.set(id, entry)
    loadEtags(await this.o.store.loadCache<Record<string, EtagEntry> | null>('github-etags.json', null))
    void this.cleanDownloads()
  }

  /** Looks for installs made outside the launcher; the window shows them when found. */
  detectInstalls(): Promise<void> {
    this.detected = this.reconcileInstalls(true)
      .catch(() => undefined)
      .then(() => this.changed())
    return this.detected
  }

  private async cleanDownloads(): Promise<void> {
    const dir = this.o.downloadsDir
    for (const name of await readdir(dir).catch(() => [] as string[])) {
      const file = join(dir, name)
      const s = await stat(file).catch(() => null)
      if (!s?.isFile()) continue
      const maxAge = /\.part(\.json)?$/.test(name) ? PART_MAX_AGE_MS : LEFTOVER_MAX_AGE_MS
      if (Date.now() - s.mtimeMs > maxAge) await rm(file, { force: true }).catch(() => undefined)
    }
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
      // Kept rather than forgotten: a drive that isn't connected, or a folder moved for a moment, comes back.
      missing: !!installed && !installed.needsSetup && !existsSync(installed.launchPath),
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
    return (this.logs.get(appId) ?? []).slice(-LOG_LIMIT)
  }

  private app(appId: string): CatalogApp {
    const app = findApp(appId)
    if (!app) throw new Error(`Unknown app: ${appId}`)
    return app
  }

  /**
   * Picks up apps installed outside the launcher. Records whose files are gone are kept (the view marks them
   * missing) instead of being forgotten with their shortcuts left behind. `registry` also searches Windows'
   * installed-apps list, which is slow, so only the startup check does it.
   */
  private async reconcileInstalls(registry: boolean): Promise<void> {
    await Promise.all(
      CATALOG.map(async (app) => {
        // A found install whose files are gone may have moved: look for it again.
        const known = this.o.store.getInstalled(app.id)
        const lost = !!known && !known.managed && !existsSync(known.launchPath)
        if ((known && !lost) || this.o.store.isDetectHidden(app.id)) return

        let launchPath: string | null = null
        let version: string | null = null
        for (const p of app.detect ?? []) {
          if (existsSync(expandEnv(p))) {
            launchPath = expandEnv(p)
            break
          }
        }
        if (!launchPath && registry && app.install.kind === 'installer') {
          const entry = await findUninstallEntry(app.install.uninstallName)
          const exe = entry && join(entry.location, app.launch.target)
          if (exe && existsSync(exe)) {
            launchPath = exe
            version = entry.version ? `v${entry.version}` : null
          }
        }
        // An install started (or still in its checks) while this looked doesn't get replaced.
        if (launchPath && this.o.store.getInstalled(app.id) === known && !this.busy(app.id)) {
          await this.o.store.setInstalled(app.id, this.externalInfo(app, launchPath, version))
        }
      })
    )
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

  /**
   * Asks GitHub for the newest versions; unless `force`, answers from the last 30 minutes are reused. A call made while
   * a check runs shares it (the startup check and a click on Refresh), except that a forced one asks again after an
   * unforced one, which may have reused answers.
   */
  refreshLatest(force = true): Promise<void> {
    const running = this.refreshing
    if (running && (running.force || !force)) return running.done
    const done = (running?.done ?? Promise.resolve()).catch(() => undefined).then(() => this.checkLatest(force))
    const entry = { force, done }
    this.refreshing = entry
    void done
      .catch(() => undefined)
      .then(() => {
        if (this.refreshing === entry) this.refreshing = null
      })
    return done
  }

  private async checkLatest(force: boolean): Promise<void> {
    // Asked for: the release pages and branch heads the apps share are fetched again, not taken from the last minute.
    if (force) dropSharedLists()
    // Cheap: only the catalog's known program paths (the registry search ran at startup).
    await this.reconcileInstalls(false)
    const now = Date.now()
    // Each app on its own: one that failed (or isn't published) doesn't make the others' answers stale.
    const due = CATALOG.filter((a) => {
      const e = this.latest.get(a.id)
      return force || !e?.info || !e.at || now - e.at >= LATEST_TTL_MS
    })
    if (due.length) {
      await Promise.all(
        due.map(async (app) => {
          const prev = this.latest.get(app.id)
          this.latest.set(app.id, mergeLatest(prev, await this.fetchLatest(app, prev)))
        })
      )
      await this.saveLatestCache()
    }
    this.changed()
  }

  private async saveLatestCache(): Promise<void> {
    const entries: LatestCache['entries'] = {}
    for (const [id, e] of this.latest) if (e.info) entries[id] = { info: e.info, at: e.at ?? 0 }
    await this.o.store.saveCache('latest-cache.json', { entries } satisfies LatestCache).catch(() => undefined)
    await this.o.store.saveCache('github-etags.json', etagSnapshot()).catch(() => undefined)
  }

  private async fetchLatest(app: CatalogApp, prev?: LatestEntry): Promise<LatestEntry> {
    if (!app.repo) return { info: null, error: "This app isn't published on GitHub yet.", release: null }
    try {
      if (app.install.kind === 'source') {
        // Source installs follow the app's branch; the label names the latest release when there is one, plus the
        // app's commits since (counted on main, where the branch is split from).
        const repo = app.repo
        const old = prev?.info
        // One shared request names every source app's branch head; the commit itself (for its date) is only asked
        // for when the head moved.
        const head = async (): Promise<{ sha: string; date: string | null }> => {
          const sha = await branchSha(repo, app.branch).catch(() => null)
          if (sha && old?.version === sha) return { sha, date: old.date }
          return headCommit(repo, app.branch)
        }
        const [c, release] = await Promise.all([head(), latestRelease(repo, app.tagPrefix)])
        const tag = release && releaseVersion(release.tag, app.tagPrefix)
        // The count only changes with the app's branch, so an unchanged head keeps the label it had.
        const same = !!tag && old?.version === c.sha && (old.label === tag || old.label.startsWith(`${tag} + `))
        let label = `${app.branch} @ ${c.sha.slice(0, 7)}`
        if (same && old) label = old.label
        else {
          const ahead =
            release?.date ? await commitsSince(app.repo, 'main', app.path, release.date).catch(() => null) : null
          if (tag && ahead === 0) label = tag
          else if (tag && ahead) label = `${tag} + ${ahead} commit${ahead === 1 ? '' : 's'}`
        }
        return {
          info: {
            version: c.sha,
            label,
            date: c.date,
            downloadSize: null,
            url: `https://github.com/${app.repo}/commits/${app.branch}`
          },
          error: null,
          release: null,
          at: Date.now()
        }
      }
      // A release whose files are still uploading (or failed to) is passed over for the one before it.
      const pattern = app.install.asset
      const release = await latestRelease(app.repo, app.tagPrefix, (r) => !!findAsset(r, pattern))
      if (!release) return { info: null, error: 'No release with a Windows download has been published yet.', release: null }
      const version = releaseVersion(release.tag, app.tagPrefix)
      return {
        info: {
          version,
          label: version,
          date: release.date,
          downloadSize: findAsset(release, pattern)?.size ?? null,
          url: release.htmlUrl
        },
        error: null,
        release,
        at: Date.now()
      }
    } catch (e) {
      return { info: null, error: errorText(e), release: null }
    }
  }

  async readme(appId: string, live: boolean): Promise<ReadmeDoc> {
    const app = this.app(appId)
    const linkBase = app.repo ? `https://github.com/${app.repo}/blob/main/${app.path}/` : null
    if (live && app.repo) {
      const cached = this.readmes.get(appId)
      if (cached) return cached
      // A recent failure isn't retried at once: on a bad connection each page would wait for the timeout first.
      if (Date.now() - (this.readmeFailed.get(appId) ?? 0) >= README_RETRY_MS) {
        try {
          const markdown = await rawFile(app.repo, 'main', `${app.path}/README.md`)
          const doc: ReadmeDoc = {
            markdown,
            assetBase: `https://raw.githubusercontent.com/${app.repo}/main/${app.path}/`,
            linkBase,
            source: 'github'
          }
          this.readmes.set(appId, doc)
          this.readmeFailed.delete(appId)
          return doc
        } catch {
          // Offline or not published: the copy shipped with the launcher.
          this.readmeFailed.set(appId, Date.now())
        }
      }
    }
    const markdown = await readFile(join(this.o.catalogDir, appId, 'README.md'), 'utf8').catch(() => `# ${app.name}\n\n${app.description}`)
    return { markdown, assetBase: `rulz://catalog/${appId}/repo/`, linkBase, source: 'bundled' }
  }

  // ---------- Jobs ----------

  /** Sends the log lines gathered since the last batch. */
  private flushLogs(): void {
    if (this.logTimer) clearTimeout(this.logTimer)
    this.logTimer = null
    if (!this.pendingLogs.length) return
    const lines = this.pendingLogs
    this.pendingLogs = []
    this.o.emitLog(lines)
  }

  private queueLog(line: LogLine): void {
    const last = this.pendingLogs.at(-1)
    // A progress counter moving on within one batch only needs its newest value sent.
    if (line.replace && last && last.appId === line.appId) this.pendingLogs[this.pendingLogs.length - 1] = { ...line, replace: last.replace }
    else this.pendingLogs.push(line)
    this.logTimer ??= setTimeout(() => this.flushLogs(), LOG_FLUSH_MS)
  }

  private startJob(appId: string, action: JobState['action'], work: (ctx: JobContext) => Promise<void>): void {
    if (this.closing) throw new Error('The launcher is closing.')
    if (this.jobs.has(appId)) throw new Error('This app is already being installed or updated.')
    const abort = new AbortController()
    let finished!: () => void
    const done = new Promise<void>((resolve) => (finished = resolve))
    const job = { state: { appId, action, phase: 'resolving', label: 'Starting…', progress: null } as JobState, abort, done }
    const log: string[] = []
    this.jobs.set(appId, job)
    this.logs.set(appId, log)

    const ctx: JobContext = {
      appId,
      files: [],
      signal: abort.signal,
      set: (patch) => {
        job.state = { ...job.state, ...patch }
        // Lines logged before a state change arrive before it.
        this.flushLogs()
        this.o.emitJob(job.state)
      },
      log: (line) => {
        const key = progressKey(line)
        const replace = !!key && log.length > 0 && progressKey(log[log.length - 1]) === key
        if (replace) log[log.length - 1] = line
        else log.push(line)
        if (log.length > LOG_LIMIT + LOG_SLACK) log.splice(0, log.length - LOG_LIMIT)
        this.queueLog({ appId, line, replace })
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
        // The requirement checks aren't dropped here: a winget install drops them itself, and checking again after
        // every job (uninstalls too) would start six programs for nothing.
        this.changed()
        for (const f of ctx.files) void rm(f, { force: true }).catch(() => undefined)
        finished()
      })
  }

  /**
   * Marks the app busy from an action's first check until its job starts. Those checks wait (for PowerShell, say), and
   * a second click, or Uninstall clicked during an update's checks, must not pass them on the record the first saw.
   * Call the returned function once the job has started (or the checks failed).
   */
  private reserve(appId: string): () => void {
    if (this.closing) throw new Error('The launcher is closing.')
    if (this.jobs.has(appId) || this.preparing.has(appId)) throw new Error('This app is already being installed or updated.')
    this.preparing.add(appId)
    return () => this.preparing.delete(appId)
  }

  /** True while the app has a job, or one is about to start. */
  private busy(appId: string): boolean {
    return this.jobs.has(appId) || this.preparing.has(appId)
  }

  /**
   * The name of an app whose folder overlaps `dir` (the same folder, one inside it, or one around it): a running
   * install that will fill it, or a launcher-made install. Its update would replace the folder, and its uninstall would
   * recycle it, with the other app inside.
   */
  private dirInUse(appId: string, dir: string): string | null {
    const a = dir.toLowerCase().replace(/[\\/]+$/, '')
    const overlaps = (other: string): boolean => {
      const b = other.toLowerCase().replace(/[\\/]+$/, '')
      return a === b || a.startsWith(`${b}\\`) || b.startsWith(`${a}\\`)
    }
    for (const [id, other] of this.installDirs) if (id !== appId && overlaps(other)) return findApp(id)?.name ?? id
    for (const app of CATALOG) {
      const info = this.o.store.getInstalled(app.id)
      if (app.id !== appId && info?.managed && overlaps(info.dir)) return app.name
    }
    return null
  }

  /**
   * Refuses to replace or recycle an app's folder while another installed app lives inside it (an install made
   * before overlapping folders were refused): a portable app's update deletes the old folder, and an uninstall moves
   * all of it to the Recycle Bin.
   */
  private assertNothingInside(app: CatalogApp, dir: string, what: string): void {
    const root = `${dir.toLowerCase().replace(/[\\/]+$/, '')}\\`
    // An install whose folder is already gone (deleted by hand) has nothing left to lose.
    const inside = CATALOG.filter((a) => {
      const other = a.id === app.id ? undefined : this.o.store.getInstalled(a.id)
      return !!other && other.dir.toLowerCase().startsWith(root) && existsSync(other.dir)
    })
    if (!inside.length) return
    const names = inside.map((a) => a.name).join(' and ')
    const many = inside.length > 1
    throw new Error(
      `${names} ${many ? 'are' : 'is'} installed inside ${app.name}'s folder, and ${what} would take ${many ? 'them' : 'it'} ` +
        `too. Uninstall or move ${many ? 'them' : 'it'} first.`
    )
  }

  cancel(appId: string): void {
    this.jobs.get(appId)?.abort.abort()
  }

  /** Cancels every running job and waits (up to a limit) for their programs to stop. Called when quitting. */
  async cancelAll(): Promise<void> {
    // The window stays usable while this waits; an install started now would be cut off by the quit.
    this.closing = true
    const running = [...this.jobs.values()]
    for (const job of running) job.abort.abort()
    await Promise.race([
      Promise.all(running.map((j) => j.done)),
      new Promise((r) => setTimeout(r, CANCEL_WAIT_MS))
    ])
    this.flushLogs()
  }

  /** Names of the apps with a running job, or one about to start, for the quit question and the self-update. */
  activeJobNames(): string[] {
    return [...new Set([...this.jobs.keys(), ...this.preparing])].map((id) => findApp(id)?.name ?? id)
  }

  private async fetchFile(
    ctx: JobContext,
    url: string,
    name: string,
    size: number | null,
    label: string,
    sha256: string | null = null
  ): Promise<string> {
    // Named after the app too: two apps installing at once can fetch the same file (both have a llama.cpp option),
    // and must not share one .part file.
    const file = join(this.o.downloadsDir, `${ctx.appId}-${name}`)
    ctx.files.push(file)
    ctx.log(`Downloading ${url}`)
    if (sha256) ctx.log(`Expecting SHA-256 ${sha256}`)
    ctx.set({ phase: 'downloading', label, progress: size ? 0 : null, bytes: 0, total: size ?? undefined })
    await download(url, file, {
      signal: ctx.signal,
      expectedSize: size,
      sha256,
      onProgress: (p) =>
        ctx.set({ progress: p.total ? p.bytes / p.total : null, bytes: p.bytes, total: p.total ?? undefined, speed: p.speed })
    })
    if (sha256) ctx.log('Checksum verified.')
    ctx.set({ speed: undefined })
    return file
  }

  /** The SHA-256 the release publishes for `asset` (null for releases made before SHA256SUMS was added). */
  private async releaseSha256(ctx: JobContext, release: Release, asset: Release['assets'][number]): Promise<string | null> {
    const sums = findAsset(release, SUMS_ASSET)
    const hash = await publishedSha256(sums?.url, asset.name, ctx.signal)
    if (!hash) ctx.log(sums ? `Warning: SHA256SUMS doesn't list ${asset.name}; it can't be verified.` : 'This release has no checksums to verify against.')
    return hash
  }

  private async requireRelease(app: CatalogApp, pattern: string): Promise<{ release: Release; asset: Release['assets'][number] }> {
    let entry = this.latest.get(app.id)
    if (!entry?.release) {
      entry = mergeLatest(entry, await this.fetchLatest(app, entry))
      this.latest.set(app.id, entry)
    }
    if (!entry.release) throw new Error(entry.error ?? 'No release found.')
    const asset = findAsset(entry.release, pattern)
    if (!asset) throw new Error(`The ${entry.release.name} release has no Windows download.`)
    return { release: entry.release, asset }
  }

  // ---------- Install / update ----------

  async install(appId: string, options: InstallOptions): Promise<void> {
    const app = this.app(appId)
    const release = this.reserve(appId)
    try {
      await this.detected
      if (this.o.store.getInstalled(appId)) throw new Error(`${app.name} is already installed.`)
      const opts: InstallOptions = { ...options, dir: await installFolder(options.dir) }
      if (this.o.store.getInstalled(appId)) throw new Error(`${app.name} is already installed.`)
      const clash = this.dirInUse(appId, opts.dir)
      if (clash) throw new Error(`Choose another folder: ${opts.dir} overlaps the folder of ${clash}.`)
      // Claimed at once, before anything else waits: another install's check may run in the meantime.
      this.installDirs.set(appId, opts.dir)
      try {
        await this.o.store.setDetectHidden(appId, false)
        // The release seen at the last check may be half an hour old, or seen before its files were uploaded.
        this.forgetRelease(appId)
        this.startJob(appId, 'install', async (ctx) => {
          try {
            await this.runInstall(ctx, app, opts, null)
          } catch (e) {
            // The folder was empty before, so anything in it now came from this attempt.
            if (!this.o.store.getInstalled(appId)) {
              await rm(opts.dir, { recursive: true, force: true }).catch(() => undefined)
              await rm(stagingDir(opts.dir), { recursive: true, force: true }).catch(() => undefined)
            }
            throw e
          } finally {
            this.installDirs.delete(appId)
          }
        })
      } catch (e) {
        this.installDirs.delete(appId)
        throw e
      }
    } finally {
      release()
    }
  }

  /**
   * The install record an action checked, if it is still the current one after the checks' waiting: an uninstall
   * or another window may have changed it meanwhile.
   */
  private unchanged(app: CatalogApp, info: InstalledInfo): void {
    if (this.o.store.getInstalled(app.id) !== info) throw new Error(`${app.name} changed meanwhile. Try again.`)
  }

  async update(appId: string): Promise<void> {
    const app = this.app(appId)
    const release = this.reserve(appId)
    try {
      const info = this.o.store.getInstalled(appId)
      if (!info?.managed) throw new Error(`${app.name} wasn't installed by the launcher, so it can't update it.`)
      await this.assertNotRunning(app, info, 'update')
      this.unchanged(app, info)
      // A portable app's update replaces the whole folder, and a setup program's update runs the old uninstaller,
      // which empties it (electron-builder's NSIS RMDir /r); git and the source archive only add to it.
      if (app.install.kind !== 'source') this.assertNothingInside(app, info.dir, 'updating it replaces that folder and')
      this.startUpdate(app, info)
    } finally {
      release()
    }
  }

  private startUpdate(app: CatalogApp, info: InstalledInfo): void {
    const appId = app.id
    // An install whose setup never finished still owes the shortcuts and optional parts it was started with.
    const p = info.pending
    const opts: InstallOptions = {
      dir: info.dir,
      desktopShortcut: p?.desktop ?? !!info.shortcuts.desktop,
      startMenuShortcut: p?.startMenu ?? !!info.shortcuts.startMenu,
      components: p?.components ?? [],
      launchAfter: false,
      installPrerequisites: true
    }
    this.forgetRelease(appId)
    this.startJob(appId, 'update', (ctx) => this.runInstall(ctx, app, opts, info))
  }

  /** The next install or update looks up the newest release again (the version shown stays until then). */
  private forgetRelease(appId: string): void {
    const known = this.latest.get(appId)
    if (known) this.latest.set(appId, { ...known, release: null })
  }

  /** Runs the setup steps again (after a failed or cancelled setup), or reinstalls the release. */
  async repair(appId: string): Promise<void> {
    const app = this.app(appId)
    if (app.install.kind !== 'source') return this.update(appId)
    const release = this.reserve(appId)
    try {
      const info = this.o.store.getInstalled(appId)
      if (!info?.managed) throw new Error(`${app.name} wasn't installed by the launcher.`)
      await this.assertNotRunning(app, info, 'repair')
      this.unchanged(app, info)
      this.startJob(appId, 'repair', async (ctx) => {
        await this.ensurePrerequisites(ctx, app)
        await this.runSetupSteps(ctx, app, info)
        const launchPath = await this.resolveLaunch(app, info.dir)
        const p = info.pending
        if (!p) return this.o.store.setInstalled(appId, { ...info, launchPath, needsSetup: false })
        // The install whose setup this finishes was started with optional parts and shortcuts: they come now.
        const opts: InstallOptions = {
          dir: info.dir,
          desktopShortcut: p.desktop,
          startMenuShortcut: p.startMenu,
          components: p.components,
          launchAfter: false,
          installPrerequisites: false
        }
        await this.finishInstall(ctx, app, opts, info, launchPath, info.version, info.method)
      })
    } finally {
      release()
    }
  }

  /**
   * Refuses to change an app's folder while a program runs from it, also one the launcher didn't start (it forgets
   * them when it closes on launch). Setup would fail half-way: `python -m venv` can't replace a running python.exe.
   */
  private async assertNotRunning(app: CatalogApp, info: InstalledInfo, then: string): Promise<void> {
    if (this.running.has(app.id)) throw new Error(`Close ${app.name} first, then ${then}.`)
    if (!existsSync(info.dir)) return
    const programs = await programsRunningFrom(info.dir)
    if (programs.length) {
      throw new Error(`${app.name} is still running (${programs.join(', ')} from its folder). Close it first, then ${then}.`)
    }
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
    if (missing.length) {
      const labels = missing.map((id) => {
        const ref = app.requirements.find((r) => r.id === id)
        return ref ? requirementRefLabel(ref) : requirementLabel(id)
      })
      throw new MissingRequirementsError(missing, labels)
    }
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
      version = releaseVersion(release.tag, app.tagPrefix)
      const sha256 = await this.releaseSha256(ctx, release, asset)
      const zip = await this.fetchFile(ctx, asset.url, asset.name, asset.size, `Downloading ${version}`, sha256)
      const staged = stagingDir(dir)
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
      method = 'zip'
    } else if (spec.kind === 'installer') {
      ctx.set({ phase: 'resolving', label: 'Finding the latest release…' })
      const { release, asset } = await this.requireRelease(app, spec.asset)
      version = releaseVersion(release.tag, app.tagPrefix)
      const sha256 = await this.releaseSha256(ctx, release, asset)
      const setup = await this.fetchFile(ctx, asset.url, asset.name, asset.size, `Downloading ${version}`, sha256)
      ctx.set({ phase: 'installing', label: 'Running the setup program…', progress: null })
      ctx.log(`> ${basename(setup)} /S /currentuser /D="${dir}"`)
      const code = await runSetupProgram([setup, '/S', '/currentuser', `/D="${dir}"`], ctx)
      if (code !== 0) throw new Error(`The setup program exited with code ${code}.`)
      launchPath = join(dir, app.launch.target)
      await waitUntil(() => existsSync(launchPath), 30_000, ctx.signal)
      if (!existsSync(launchPath)) throw new Error(`Setup finished, but ${launchPath} wasn't created.`)
      // The setup program makes its own shortcuts; the launcher's options decide which ones exist.
      for (const lnk of foreignShortcutPaths(spec.shortcutName)) await removeShortcut(lnk)
      await rm(setup, { force: true })
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
        needsSetup: true,
        // What the rest of this install would add, for Repair to finish if the setup stops here.
        pending: { desktop: opts.desktopShortcut, startMenu: opts.startMenuShortcut, components: opts.components }
      }
      await this.o.store.setInstalled(app.id, partial)
      await this.runSetupSteps(ctx, app, partial)
      launchPath = await this.resolveLaunch(app, dir)
    }
    await this.finishInstall(ctx, app, opts, prev, launchPath, version, method)
  }

  /** What follows once the files are in place: optional parts, shortcuts, the install record, then launching it. */
  private async finishInstall(
    ctx: JobContext,
    app: CatalogApp,
    opts: InstallOptions,
    prev: InstalledInfo | null,
    launchPath: string,
    version: string | null,
    method: InstalledInfo['method']
  ): Promise<void> {
    const dir = opts.dir
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
    // A commit's label ("v1.2.0 + 3 commits") comes with the newest version's details. A release's label is its tag,
    // and the release was just looked up, so asking GitHub again would only use up requests.
    if (app.install.kind === 'source') {
      const latest = mergeLatest(this.latest.get(app.id), await this.fetchLatest(app, this.latest.get(app.id)))
      this.latest.set(app.id, latest)
      if (latest.info && latest.info.version === version && latest.info.label !== version) {
        await this.o.store.setInstalled(app.id, { ...info, versionLabel: latest.info.label })
      }
    }
    if (opts.launchAfter) await this.launch(app.id, true).catch((e) => ctx.log(`Couldn't start it: ${errorText(e)}`))
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
      const url = `https://github.com/${app.repo}.git`
      if (prev) {
        ctx.set({ phase: 'downloading', label: 'Pulling the latest changes…', progress: null })
        // Also moves clones of the app's old repository over: its branch here continues that history commit for
        // commit, so they fast-forward. Afterwards a plain `git pull` in the folder works too.
        const steps = [
          ['git', 'remote', 'set-url', 'origin', url],
          ['git', 'remote', 'set-branches', 'origin', app.branch],
          ['git', 'fetch', '--progress', 'origin'],
          ['git', 'merge', '--ff-only', `origin/${app.branch}`],
          ['git', 'branch', `--set-upstream-to=origin/${app.branch}`]
        ]
        for (const step of steps) {
          ctx.log(`> ${step.join(' ')}`)
          const code = await runCommand(step, { cwd: dir, signal: ctx.signal, onLine: ctx.log })
          if (code !== 0) throw new Error(`${step.slice(0, 2).join(' ')} failed (local changes in the folder?). See the log for details.`)
        }
      } else {
        ctx.set({ phase: 'downloading', label: 'Cloning the repository…', progress: null })
        await mkdir(dir, { recursive: true })
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
    const staged = stagingDir(dir)
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
      await renameRetrying(staged, dir)
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
    try {
      await mkdir(this.o.iconsDir, { recursive: true })
      await copyFile(src, dest)
      return dest
    } catch {
      // A shortcut without its icon still works; failing here would fail an install whose files are already in place.
      return existsSync(dest) ? dest : undefined
    }
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
    // A running job writes the shortcuts it started with when it ends, which would undo this change.
    if (this.busy(appId)) throw new Error(`Wait for ${app.name} to finish installing, updating or uninstalling.`)
    // Held while the shortcuts are made, so an uninstall can't start and then have its record written back.
    const release = this.reserve(appId)
    try {
      const shortcuts = await this.applyShortcuts(app, info, opts)
      await this.o.store.setInstalled(appId, { ...info, shortcuts })
    } finally {
      release()
    }
    this.changed()
  }

  // ---------- Launch / remove ----------

  /** Starts the app. `fromJob` is an install starting the app it has just finished. */
  async launch(appId: string, fromJob = false): Promise<void> {
    const app = this.app(appId)
    const info = this.o.store.getInstalled(appId)
    if (!info) throw new Error(`${app.name} isn't installed.`)
    // Its files may be half replaced or on their way to the Recycle Bin, and a running copy would stop the job.
    if (!fromJob && this.busy(appId)) throw new Error(`Wait for ${app.name} to finish installing, updating or uninstalling.`)
    if (info.needsSetup) throw new Error(`${app.name}'s setup didn't finish. Use Repair to run it again.`)
    if (!existsSync(info.launchPath)) throw new Error(`${info.launchPath} is missing. Reinstall ${app.name}.`)
    const child = launchDetached(
      info.launchPath,
      info.launchArgs,
      info.dir,
      launchEnv(app, this.o.store.getSettings().launchOptions?.[appId])
    )
    // Counted as running at once: an update checking in the moment before Windows reports the start must see it.
    this.running.set(appId, child)
    const forget = (): void => {
      if (this.running.get(appId) === child) this.running.delete(appId)
    }
    child.once('exit', () => {
      forget()
      this.changed()
    })
    try {
      await new Promise<void>((resolve, reject) => {
        child.once('spawn', resolve)
        child.once('error', (e) => reject(new Error(`Couldn't start ${app.name}: ${e.message}`)))
      })
    } catch (e) {
      forget()
      throw e
    }
    this.changed()
  }

  /** True while any app is being installed, updated or removed, or about to be. */
  hasActiveJobs(): boolean {
    return this.jobs.size > 0 || this.preparing.size > 0
  }

  async uninstall(appId: string): Promise<void> {
    const app = this.app(appId)
    const release = this.reserve(appId)
    try {
      const info = this.o.store.getInstalled(appId)
      if (!info) throw new Error(`${app.name} isn't installed.`)
      if (info.managed) await this.assertNotRunning(app, info, 'uninstall it')
      this.unchanged(app, info)
      // Checked last, after the waiting: an app may have been located inside the folder meanwhile.
      if (info.managed) this.assertNothingInside(app, info.dir, 'uninstalling it recycles that folder and')
      this.startUninstall(app, info)
    } finally {
      release()
    }
  }

  private startUninstall(app: CatalogApp, info: InstalledInfo): void {
    const appId = app.id
    this.startJob(appId, 'uninstall', async (ctx) => {
      ctx.set({ phase: 'removing', label: 'Removing…' })
      if (info.managed) {
        if (app.install.kind === 'installer') {
          const uninstaller = join(info.dir, `Uninstall ${app.install.uninstallName}.exe`)
          if (existsSync(uninstaller)) {
            ctx.set({ label: 'Running the uninstaller…' })
            ctx.log(`> ${basename(uninstaller)} /S /currentuser`)
            const code = await runSetupProgram([uninstaller, '/S', '/currentuser'], ctx)
            // Its leftovers are recycled below either way; the code says why Windows may still list the app.
            if (code !== 0) ctx.log(`Warning: the uninstaller exited with code ${code}.`)
            // NSIS uninstallers copy themselves to %TEMP% and return at once; wait for the files to go.
            await waitUntil(() => !existsSync(info.launchPath), 60_000, ctx.signal)
          }
        }
        if (existsSync(info.dir)) {
          if (ctx.signal.aborted) throw new Error('Cancelled')
          ctx.set({ label: 'Moving the files to the Recycle Bin…' })
          ctx.log(`Recycling ${info.dir}`)
          try {
            await shell.trashItem(info.dir)
          } catch (e) {
            // A folder on a network drive (or one Windows won't recycle) never goes; deleting it by hand is the way
            // out, and Uninstall then only removes the shortcuts and the record.
            throw new Error(
              `Couldn't move ${info.dir} to the Recycle Bin: ${errorText(e)}. Close the app and try again, or delete ` +
                'the folder yourself and click Uninstall again.'
            )
          }
        }
      }
      // Only once the files are gone: when they can't be removed, the app stays as it was, shortcuts included.
      ctx.set({ label: 'Removing shortcuts…' })
      await removeShortcut(info.shortcuts.desktop)
      await removeShortcut(info.shortcuts.startMenu)
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
    // An install that is running would replace this record when it ends.
    if (this.busy(appId)) throw new Error(`Wait for ${app.name} to finish installing.`)
    const release = this.reserve(appId)
    try {
      await this.o.store.setDetectHidden(appId, false)
      await this.o.store.setInstalled(appId, this.externalInfo(app, file, null))
    } finally {
      release()
    }
    this.changed()
  }

  async openFolder(appId: string): Promise<void> {
    const info = this.o.store.getInstalled(appId)
    if (!info) throw new Error("It isn't installed.")
    const err = await shell.openPath(info.dir)
    if (err) throw new Error(err)
  }
}
