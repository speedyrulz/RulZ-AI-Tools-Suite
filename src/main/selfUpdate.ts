import { spawn } from 'node:child_process'
import { mkdir, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { app } from 'electron'
import type { SelfUpdate } from '@shared/types'
import { download } from './download'
import { findAsset, isNewer, latestRelease } from './github'

export const SELF_REPO = 'speedyrulz/RulZ-AI-Tools-Suite'
const SETUP_ASSET = '^RulZ-AI-Tools-Suite-Setup-[\d.]+\.exe$'

/** The newer launcher release, or null when this is the latest (or a dev run, which has no installer to replace). */
export async function checkSelfUpdate(): Promise<SelfUpdate | null> {
  if (!app.isPackaged) return null
  const release = await latestRelease(SELF_REPO)
  if (!release || !isNewer(release.tag, app.getVersion())) return null
  const asset = findAsset(release, SETUP_ASSET)
  if (!asset) return null
  return { version: release.tag.replace(/^v/, ''), tag: release.tag, size: asset.size, url: asset.url, notesUrl: release.htmlUrl }
}

// Runs under Electron's own node after the launcher quits: runs the silent setup (retrying its occasional startup
// crash, 0xC0000005) over the current install, then starts the launcher again.
const HELPER = `
const { spawnSync, spawn } = require('node:child_process')
const [setup, dir, exe] = process.argv.slice(2)
setTimeout(() => {
  for (let i = 0; i < 3; i++) {
    const r = spawnSync(setup, ['/S', '/currentuser', '/D="' + dir + '"'], {
      windowsVerbatimArguments: true, argv0: '"' + setup + '"', stdio: 'ignore', cwd: process.env.SystemRoot
    })
    if ((r.status | 0) !== (0xc0000005 | 0)) break
  }
  spawn(exe, [], { detached: true, stdio: 'ignore' }).unref()
}, 1500)
`

/** Downloads the installer, then hands over to the helper and quits so the files can be replaced. */
export async function installSelfUpdate(u: SelfUpdate, onProgress: (fraction: number) => void): Promise<void> {
  const dl = join(app.getPath('temp'), 'RulZ AI Tools Suite')
  const setup = join(dl, `RulZ-AI-Tools-Suite-Setup-${u.version}.exe`)
  await download(u.url, setup, { expectedSize: u.size, onProgress: (p) => p.total && onProgress(p.bytes / p.total) })
  const helper = join(app.getPath('userData'), 'self-update.cjs')
  await mkdir(dirname(helper), { recursive: true })
  await writeFile(helper, HELPER, 'utf8')
  const child = spawn(process.execPath, [helper, setup, dirname(process.execPath), process.execPath], {
    detached: true,
    stdio: 'ignore',
    env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' }
  })
  child.unref()
  app.quit()
}
