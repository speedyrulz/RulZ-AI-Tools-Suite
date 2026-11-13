import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { appendFile, mkdir, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { app } from 'electron'
import type { SelfUpdate } from '@shared/types'
import { download, publishedSha256 } from './download'
import type { Store } from './store'
import { findAsset, isNewer, latestRelease, type Release } from './github'

const execFileAsync = promisify(execFile)

export const SELF_REPO = 'speedyrulz/RulZ-AI-Tools-Suite'
export const SETUP_ASSET = '^RulZ-AI-Tools-Suite-Setup-[\\d.]+\\.exe$'
/** Where the launcher's setup programs are downloaded from; nothing else is run as an update. */
export const SELF_DOWNLOADS = `https://github.com/${SELF_REPO}/releases/download/`

/**
 * True when the update's download is the launcher setup of its own tag (the repository's releases also hold the
 * apps' setups), and its checksum list comes from the same place. The answer may come from a cached file.
 */
function isOwnSetup(u: SelfUpdate): boolean {
  const base = `${SELF_DOWNLOADS}${u.tag}/`
  if (!u.url.startsWith(base) || (u.sumsUrl && !u.sumsUrl.startsWith(base))) return false
  return new RegExp(SETUP_ASSET, 'i').test(decodeURIComponent(u.url.slice(base.length)))
}

/** The update the last check offered: "Update now" installs this one, whatever the window sends back. */
let offered: SelfUpdate | null = null

/** The update this run offered with that tag, or null. */
export function offeredUpdate(tag: string): SelfUpdate | null {
  return offered?.tag === tag ? offered : null
}

/** The newer launcher release, or null when this is the latest (or a dev run, which has no installer to replace). */
export async function checkSelfUpdate(store: Store, force = false): Promise<SelfUpdate | null> {
  if (!app.isPackaged) return null
  // Answers from the last 30 minutes are reused (GitHub allows 60 unauthenticated requests an hour).
  const cache = await store.loadCache<{ at: number; version: string; update: SelfUpdate | null } | null>('self-check.json', null)
  if (!force && cache && cache.version === app.getVersion() && Date.now() - cache.at < 30 * 60_000) {
    offered = cache.update
    return cache.update
  }
  const update = await lookUp()
  offered = update
  await store.saveCache('self-check.json', { at: Date.now(), version: app.getVersion(), update })
  return update
}

async function lookUp(): Promise<SelfUpdate | null> {
  // A release whose setup is still uploading is passed over (and offered once it's there, not after the cache's
  // half hour of "no update"). One retry: GitHub or the network sometimes fails right as the app starts.
  const usable = (r: Release): boolean => !!findAsset(r, SETUP_ASSET)
  const release = await latestRelease(SELF_REPO, 'v', usable).catch(async () => {
    await new Promise((r) => setTimeout(r, 4000))
    return latestRelease(SELF_REPO, 'v', usable)
  })
  if (!release || !isNewer(release.tag, app.getVersion())) return null
  const asset = findAsset(release, SETUP_ASSET)
  if (!asset) return null
  return {
    version: release.tag.replace(/^v/, ''),
    tag: release.tag,
    size: asset.size,
    url: asset.url,
    notesUrl: release.htmlUrl,
    sumsUrl: findAsset(release, '^SHA256SUMS$')?.url ?? null,
    assetName: asset.name
  }
}

// Runs in PowerShell after the launcher quits: the silent setup over the current install (retrying its occasional
// startup crash, 0xC0000005), then the launcher again. It can't be the launcher's own program: the setup stops every
// running copy of it. So it waits for the launcher to be gone first (quitting can take a while when it cancels an
// install), and gives up rather than stop a launcher that is still running. The log goes next to the settings.
export const HELPER = `
param([string]$Setup, [string]$Dir, [string]$Exe, [string]$Log, [int]$LauncherPid)
function Note($m) { try { Add-Content -LiteralPath $Log -Value ((Get-Date -Format o) + ' ' + $m) } catch {} }
Note 'updater running'
if ($LauncherPid) {
  try { Wait-Process -Id $LauncherPid -Timeout 300 -ErrorAction Stop } catch {}
  if (Get-Process -Id $LauncherPid -ErrorAction SilentlyContinue) {
    Note 'the launcher is still running after 5 minutes; not updating'
    exit 1
  }
}
Start-Sleep -Seconds 1
for ($i = 1; $i -le 3; $i++) {
  Note "setup attempt $i"
  $p = Start-Process -FilePath $Setup -ArgumentList ('/S /currentuser /D="' + $Dir + '"') -WorkingDirectory $env:SystemRoot -Wait -PassThru
  Note "exit $($p.ExitCode)"
  if ($p.ExitCode -ne -1073741819) { break }
}
Remove-Item -LiteralPath $Setup -Force -ErrorAction SilentlyContinue
# Through WMI as well, so the launcher isn't a child of this script (which ends right here).
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = ('"' + $Exe + '"'); CurrentDirectory = $Dir }
Note ('relaunched, WMI result ' + $r.ReturnValue)
`

/** Why the launcher can't update now (an app install it would cut off), or null when it can. */
export function busyReason(names: string[]): string | null {
  if (!names.length) return null
  const many = names.length > 1
  return `${names.join(', ')} ${many ? 'are' : 'is'} still being installed or updated. Update the launcher when ${many ? 'they finish' : 'it finishes'}.`
}

/**
 * Downloads the installer, then hands over to the helper and quits so the files can be replaced. `activeJobs` names
 * the apps being installed: the setup would stop the launcher in the middle of them, so the update waits for them.
 */
export async function installSelfUpdate(
  u: SelfUpdate,
  onProgress: (fraction: number) => void,
  activeJobs: () => string[]
): Promise<void> {
  const refuse = (): void => {
    const reason = busyReason(activeJobs())
    if (reason) throw new Error(reason)
  }
  refuse()
  if (!isOwnSetup(u)) throw new Error("That update isn't one of the launcher's own releases.")
  const dl = join(app.getPath('temp'), 'RulZ AI Tools Suite')
  const setup = join(dl, `RulZ-AI-Tools-Suite-Setup-${u.version}.exe`)
  // Releases since SHA256SUMS was added are checked against it; older ones (and old cached answers) aren't.
  const sha256 = u.assetName ? await publishedSha256(u.sumsUrl, u.assetName) : null
  await download(u.url, setup, { expectedSize: u.size, sha256, onProgress: (p) => p.total && onProgress(p.bytes / p.total) })
  // Asked again: an install may have been started while the update downloaded.
  refuse()
  const helper = join(app.getPath('userData'), 'self-update.ps1')
  await mkdir(dirname(helper), { recursive: true })
  // A BOM makes Windows PowerShell 5.1 read the file as UTF-8 (install paths can have non-ASCII characters).
  await writeFile(helper, '﻿' + HELPER, 'utf8')
  const log = join(app.getPath('userData'), 'self-update.log')
  // The helper must outlive the launcher, but PowerShell exits when its parent does however it is started, so
  // Windows (WMI) starts it instead and it isn't a child of the launcher at all.
  const quote = (v: string): string => `"${v}"`
  const commandLine = [
    'powershell.exe -NoProfile -ExecutionPolicy Bypass -File',
    quote(helper),
    quote(setup),
    quote(dirname(process.execPath)),
    quote(process.execPath),
    quote(log),
    String(process.pid)
  ].join(' ')
  await execFileAsync(
    'powershell.exe',
    [
      '-NoProfile',
      '-Command',
      `$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='${commandLine.replace(/'/g, "''")}'}; if ($r.ReturnValue -ne 0) { exit 1 }`
    ],
    { windowsHide: true }
  ).catch((e: Error) => {
    throw new Error(`Couldn't start the updater: ${e.message}`)
  })
  await appendFile(log, `${new Date().toISOString()} asked Windows to start the updater
`).catch(() => undefined)
  app.quit()
}
