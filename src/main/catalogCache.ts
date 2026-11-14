// Newer READMEs, README images, screenshots and icons for the catalog, fetched from GitHub and kept on disk, so a
// changed screenshot doesn't need a new launcher. Only files that differ from the copy shipped with the launcher are
// downloaded; everything else is served from the bundled catalog.

import { createHash } from 'node:crypto'
import { existsSync } from 'node:fs'
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { dirname, join, resolve, sep } from 'node:path'
import { renameRetrying } from './archive'
import { folderTree, headCommit, rawBytes, type TreeFile } from './github'
import { writeJson } from './store'

const IMAGE = 'png|jpe?g|webp|gif|avif|bmp'
/** The files of an app's catalog folder that are fetched: the icon.ico for shortcuts stays the bundled one. */
const SYNCED = new RegExp(`^(README\\.md|icon\\.png|screenshots/[^/]+\\.(${IMAGE})|repo/.+\\.(${IMAGE}|svg))$`, 'i')
const SCREENSHOT = new RegExp(`^screenshots/[^/]+\\.(${IMAGE})$`, 'i')
/** Anything bigger is left out (and the bundled copy, if any, kept): a screenshot is a few MB at most. */
const MAX_FILE = 20 * 1024 * 1024
const DOWNLOADS_AT_ONCE = 4
const MANIFEST = 'manifest.json'

/** A catalog file as known here: in the cache folder, or the bundled copy. */
export interface CatalogEntry {
  sha: string
  where: 'cache' | 'bundled'
  /** GitHub has a newer copy that couldn't be fetched yet: this one is shown meanwhile, and it's tried again. */
  stale?: boolean
}

export interface CatalogManifest {
  /** The launcher version that wrote it: a newer launcher has newer bundled files, so the cache starts over. */
  launcher: string
  /** When GitHub was last asked (ms). */
  checkedAt: number
  /** The commit the files were taken from. */
  commit?: string
  /** How many files couldn't be fetched at the last check. */
  failed?: number
  /** Catalog files (path below catalog/, "llm-suite/screenshots/01-chat.png") as last seen on GitHub. */
  files: Record<string, CatalogEntry>
}

export interface CatalogCacheOptions {
  repo: string
  branch: string
  /** The catalog shipped with the launcher. */
  bundledDir: string
  /** Where fetched files and the manifest are kept. */
  cacheDir: string
  launcherVersion: string
  /** App ids the launcher knows; other folders on GitHub are ignored. */
  appIds: string[]
}

export interface SyncResult {
  /** Files now served from a newer copy, or gone. */
  changed: number
  /** Files that couldn't be fetched (tried again at the next check). */
  failed: number
}

/** git's hash of a file's contents (what the trees API lists). */
export function gitBlobSha(data: Uint8Array): string {
  return createHash('sha1').update(`blob ${data.length}\0`).update(data).digest('hex')
}

/** The bytes with every CRLF turned into LF. */
export function crlfToLf(data: Uint8Array): Uint8Array {
  if (!data.includes(13)) return data
  const out = new Uint8Array(data.length)
  let n = 0
  for (let i = 0; i < data.length; i++) {
    if (data[i] === 13 && data[i + 1] === 10) continue
    out[n++] = data[i]
  }
  return out.subarray(0, n)
}

/** Joins a catalog path onto a folder, or null when it would leave it ("..", absolute paths). */
export function inside(dir: string, rel: string): string | null {
  if (!rel || rel.includes('\0') || /^[a-z]:|^[\\/]/i.test(rel)) return null
  const file = resolve(dir, rel)
  return file.startsWith(resolve(dir) + sep) ? file : null
}

export class CatalogCache {
  private manifest: CatalogManifest | null = null
  /** git hashes of bundled files, worked out once each. */
  private bundledSha = new Map<string, string | null>()
  private syncing: Promise<SyncResult> | null = null

  constructor(private readonly o: CatalogCacheOptions) {}

  /** Reads the manifest; one written by another launcher version is dropped with its files. */
  async load(): Promise<void> {
    let m: CatalogManifest | null = null
    try {
      m = JSON.parse(await readFile(join(this.o.cacheDir, MANIFEST), 'utf8')) as CatalogManifest
    } catch {
      m = null
    }
    if (m && m.launcher === this.o.launcherVersion && m.files && typeof m.files === 'object') {
      this.manifest = m
      return
    }
    this.manifest = null
    if (m || existsSync(this.o.cacheDir)) await rm(this.o.cacheDir, { recursive: true, force: true }).catch(() => undefined)
  }

  /** When GitHub was last asked, or 0. */
  checkedAt(): number {
    return this.manifest?.checkedAt ?? 0
  }

  /** True when the last check left files to fetch (they are tried again sooner). */
  hasFailures(): boolean {
    return (this.manifest?.failed ?? 0) > 0
  }

  /**
   * The file to serve for a catalog path: the fetched copy when there is one, else the bundled file (even when GitHub
   * has a newer one that couldn't be fetched yet). Null for paths outside the catalog.
   */
  file(rel: string, useCache = true): string | null {
    const bundled = inside(this.o.bundledDir, rel)
    if (!bundled) return null
    if (useCache && this.manifest?.files[rel]?.where === 'cache') {
      const cached = inside(this.o.cacheDir, rel)
      if (cached && existsSync(cached)) return cached
    }
    return bundled
  }

  /** What is known of a catalog file, or undefined. */
  entry(rel: string): CatalogEntry | undefined {
    return this.manifest?.files[rel]
  }

  /** The known files below `prefix` ("llm-suite/repo/"), by the path after it. */
  entries(prefix: string): [string, CatalogEntry][] {
    return Object.entries(this.manifest?.files ?? {})
      .filter(([path]) => path.startsWith(prefix))
      .map(([path, e]) => [path.slice(prefix.length), e])
  }

  /** True when the app's README is the one on GitHub (fetched, or the same as the bundled one). */
  readmeFromGitHub(appId: string): boolean {
    const e = this.manifest?.files[`${appId}/README.md`]
    return !!e && !e.stale
  }

  /**
   * The app's screenshot file names (sorted) with a short hash each, for URLs that change when the picture does; null
   * when GitHub hasn't been asked yet (the bundled folder is listed instead).
   */
  screenshots(appId: string): { name: string; sha: string }[] | null {
    if (!this.manifest) return null
    return this.entries(`${appId}/`)
      .filter(([rest]) => SCREENSHOT.test(rest))
      .map(([rest, e]) => ({ name: rest.slice('screenshots/'.length), sha: e.sha }))
      .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))
  }

  /** Asks GitHub what changed and fetches it. Calls made while a check runs share it. */
  sync(): Promise<SyncResult> {
    // Cleared inside the run, before its callers hear back: one that asks again at once gets a new check.
    this.syncing ??= this.runSync().finally(() => {
      this.syncing = null
    })
    return this.syncing
  }

  private async runSync(): Promise<SyncResult> {
    // Files are taken from one commit: raw.githubusercontent.com caches branch names for minutes, commits never go stale.
    const commit = (await headCommit(this.o.repo, this.o.branch)).sha
    const before = this.manifest?.files ?? {}
    // The same commit as last time, with everything fetched: nothing to do (one request instead of two).
    if (this.manifest && this.manifest.commit === commit && !this.manifest.failed) {
      this.manifest = { ...this.manifest, checkedAt: Date.now() }
      await this.save()
      return { changed: 0, failed: 0 }
    }

    const tree = await folderTree(this.o.repo, commit, 'catalog')
    const ids = new Set(this.o.appIds)
    const wanted = new Map<string, TreeFile>()
    for (const f of tree) {
      const slash = f.path.indexOf('/')
      if (slash < 0 || !ids.has(f.path.slice(0, slash)) || !SYNCED.test(f.path.slice(slash + 1))) continue
      if (f.size > MAX_FILE || !inside(this.o.cacheDir, f.path) || !inside(this.o.bundledDir, f.path)) continue
      wanted.set(f.path, f)
    }

    const files: Record<string, CatalogEntry> = {}
    const toFetch: TreeFile[] = []
    let changed = 0
    for (const f of wanted.values()) {
      const known = before[f.path]
      if (known?.sha === f.sha && !known.stale && (known.where === 'bundled' || existsSync(inside(this.o.cacheDir, f.path)!))) {
        files[f.path] = known
      } else if ((await this.bundledHash(f.path)) === f.sha) {
        files[f.path] = { sha: f.sha, where: 'bundled' }
        if (known?.where === 'cache' || known?.stale) changed++
      } else {
        toFetch.push(f)
      }
    }

    let failed = 0
    const queue = [...toFetch]
    const worker = async (): Promise<void> => {
      for (let f = queue.shift(); f; f = queue.shift()) {
        try {
          await this.fetchFile(commit, f)
          files[f.path] = { sha: f.sha, where: 'cache' }
          changed++
        } catch {
          failed++
          // Shown as it was (an older fetched copy, or the bundled file) and tried again at the next check.
          const known = before[f.path]
          const bundled = await this.bundledHash(f.path)
          if (known) files[f.path] = { ...known, stale: true }
          else if (bundled) files[f.path] = { sha: bundled, where: 'bundled', stale: true }
        }
      }
    }
    await Promise.all(Array.from({ length: Math.min(DOWNLOADS_AT_ONCE, queue.length) }, worker))

    // Gone from GitHub (or no longer fetched): the copy here goes too.
    for (const [path, e] of Object.entries(before)) {
      if (files[path]) continue
      changed++
      if (e.where === 'cache') await this.removeCached(path)
    }
    // A file that came back as the bundled copy has no use for its fetched one.
    for (const [path, e] of Object.entries(files)) {
      if (e.where === 'bundled' && before[path]?.where === 'cache') await this.removeCached(path)
    }

    this.manifest = { launcher: this.o.launcherVersion, checkedAt: Date.now(), commit, failed, files }
    await this.save()
    return { changed, failed }
  }

  private async save(): Promise<void> {
    await mkdir(this.o.cacheDir, { recursive: true })
    await writeJson(join(this.o.cacheDir, MANIFEST), this.manifest)
  }

  private async removeCached(rel: string): Promise<void> {
    const cached = inside(this.o.cacheDir, rel)
    if (cached) await rm(cached, { force: true }).catch(() => undefined)
  }

  /** Downloads one file, checks it against git's hash and puts it in place. */
  private async fetchFile(commit: string, f: TreeFile): Promise<void> {
    const data = await rawBytes(this.o.repo, commit, `catalog/${f.path}`)
    if (gitBlobSha(data) !== f.sha) throw new Error(`${f.path}: not the file GitHub lists`)
    const dest = inside(this.o.cacheDir, f.path)!
    await mkdir(dirname(dest), { recursive: true })
    const tmp = `${dest}.${process.pid}.tmp`
    try {
      await writeFile(tmp, data)
      await renameRetrying(tmp, dest)
    } catch (e) {
      await rm(tmp, { force: true }).catch(() => undefined)
      throw e
    }
  }

  private async bundledHash(rel: string): Promise<string | null> {
    if (this.bundledSha.has(rel)) return this.bundledSha.get(rel)!
    const file = inside(this.o.bundledDir, rel)
    const data = file ? await readFile(file).catch(() => null) : null
    // Release builds check the repo out on Windows, which gives text files CRLF; git hashes them with LF.
    const sha = data ? gitBlobSha(/\.(md|svg)$/i.test(rel) ? crlfToLf(data) : data) : null
    this.bundledSha.set(rel, sha)
    return sha
  }
}
