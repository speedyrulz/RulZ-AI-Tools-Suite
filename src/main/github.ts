// GitHub REST calls (unauthenticated: 60 requests an hour, plenty for a handful of apps).

export type FetchFn = (url: string, init?: RequestInit) => Promise<Response>

let fetchImpl: FetchFn = (url, init) => fetch(url, init)

/** Electron's net.fetch follows the system proxy settings; the main process swaps it in. */
export function setFetch(fn: FetchFn): void {
  fetchImpl = fn
}

export function getFetch(): FetchFn {
  return fetchImpl
}

export interface ReleaseAsset {
  name: string
  size: number
  url: string
}

export interface Release {
  tag: string
  name: string
  date: string | null
  prerelease: boolean
  htmlUrl: string
  assets: ReleaseAsset[]
}

export interface Commit {
  sha: string
  date: string | null
  message: string
}

export class GitHubError extends Error {
  constructor(
    message: string,
    readonly status: number
  ) {
    super(message)
  }
}

interface RawRelease {
  tag_name: string
  name: string | null
  published_at: string | null
  prerelease: boolean
  draft: boolean
  html_url: string
  assets: { name: string; size: number; browser_download_url: string }[]
}

/**
 * A GitHub answer and its ETag: asking again with If-None-Match gets a short 304 when nothing changed. Without a token
 * a 304 still counts against the 60 requests an hour (checked 2026-11-13); it saves the download, not the quota.
 */
export interface EtagEntry {
  etag: string
  body: unknown
}

const ETAG_LIMIT = 200
const etags = new Map<string, EtagEntry>()

/** The remembered answers, for keeping across launches. */
export function etagSnapshot(): Record<string, EtagEntry> {
  return Object.fromEntries(etags)
}

export function loadEtags(saved: Record<string, EtagEntry> | null | undefined): void {
  for (const [path, e] of Object.entries(saved ?? {})) {
    if (e && typeof e.etag === 'string' && !etags.has(path)) etags.set(path, e)
  }
}

export function clearEtags(): void {
  etags.clear()
}

/** `slim` keeps only what the launcher reads (release lists are large), before the answer is remembered. */
async function api<T>(path: string, slim: (body: never) => T = (b) => b as T): Promise<T> {
  const known = etags.get(path)
  const res = await fetchImpl(`https://api.github.com${path}`, {
    headers: {
      Accept: 'application/vnd.github+json',
      'User-Agent': 'RulZ-AI-Tools-Suite',
      ...(known ? { 'If-None-Match': known.etag } : {})
    },
    signal: AbortSignal.timeout(15_000)
  })
  if (res.status === 304 && known) return known.body as T
  if (res.ok) {
    const body = slim((await res.json()) as never)
    const etag = res.headers.get('etag')
    if (etag) {
      etags.delete(path)
      etags.set(path, { etag, body })
      while (etags.size > ETAG_LIMIT) etags.delete(etags.keys().next().value!)
    }
    return body
  }
  if (res.status === 403 || res.status === 429) {
    const reset = Number(res.headers.get('x-ratelimit-reset'))
    const when = reset ? ` after ${new Date(reset * 1000).toLocaleTimeString()}` : ' later'
    throw new GitHubError(`GitHub's request limit was reached. Try again${when}.`, res.status)
  }
  if (res.status === 404) throw new GitHubError('Not found on GitHub.', 404)
  throw new GitHubError(`GitHub returned ${res.status} ${res.statusText}`, res.status)
}

function toRelease(r: RawRelease): Release {
  return {
    tag: r.tag_name,
    name: r.name || r.tag_name,
    date: r.published_at,
    prerelease: r.prerelease,
    htmlUrl: r.html_url,
    assets: r.assets.map((a) => ({ name: a.name, size: a.size, url: a.browser_download_url }))
  }
}

function slimReleases(list: RawRelease[]): RawRelease[] {
  return list.map((r) => ({
    tag_name: r.tag_name,
    name: r.name,
    published_at: r.published_at,
    prerelease: r.prerelease,
    draft: r.draft,
    html_url: r.html_url,
    assets: r.assets.map((a) => ({ name: a.name, size: a.size, browser_download_url: a.browser_download_url }))
  }))
}

const RELEASE_PAGE = 100
/**
 * How deep the shared release list is read: one request a page serves every app. An app whose newest release is older
 * than this many pages of other apps' releases (RulzTrainer alone publishes dozens a month) is found through its tags
 * instead, two requests of its own.
 */
const RELEASE_PAGES = 5
/** At most this many of an app's tags, newest first, are asked about when looking it up by tag. */
const TAG_TRIES = 10
const PAGE_TTL_MS = 60_000
const pageCache = new Map<string, { at: number; list: Promise<RawRelease[]> }>()

/** A page of a repository's releases, newest first. Shared for a minute, so the apps' checks cost one request. */
function releasePage(repo: string, page: number): Promise<RawRelease[]> {
  const key = `${repo}#${page}`
  const hit = pageCache.get(key)
  if (hit && Date.now() - hit.at < PAGE_TTL_MS) return hit.list
  const list = api(`/repos/${repo}/releases?per_page=${RELEASE_PAGE}&page=${page}`, slimReleases)
  pageCache.set(key, { at: Date.now(), list })
  list.catch(() => pageCache.delete(key))
  return list
}

/** True for `prefix` followed by a version: "image-organizer-v1.2.0" for "image-organizer-v", not "vnext" for "v". */
function hasPrefix(tag: string, prefix: string): boolean {
  return tag.startsWith(prefix) && /^\d/.test(tag.slice(prefix.length))
}

/**
 * The newest full release whose tag is `prefix` followed by a version ("v" for v1.2.0, "image-organizer-v" for
 * image-organizer-v1.2.0), or null when there is none. One repository holds the releases of several apps. `accept`
 * passes over releases the caller can't use (one whose files haven't been uploaded yet, say).
 */
export async function latestRelease(
  repo: string,
  prefix = 'v',
  accept: (r: Release) => boolean = () => true
): Promise<Release | null> {
  const matches = (r: RawRelease): boolean => !r.draft && !r.prerelease && hasPrefix(r.tag_name, prefix) && accept(toRelease(r))
  for (let page = 1; page <= RELEASE_PAGES; page++) {
    const list = await releasePage(repo, page)
    const hit = list.find(matches)
    if (hit) return toRelease(hit)
    // A short page is the last one: every release has been seen.
    if (list.length < RELEASE_PAGE) return null
  }
  return releaseByTags(repo, prefix, accept)
}

/** The newest usable release among the tags starting with `prefix`, asked for one tag at a time (newest first). */
async function releaseByTags(repo: string, prefix: string, accept: (r: Release) => boolean): Promise<Release | null> {
  const tags = await api(`/repos/${repo}/git/matching-refs/tags/${encodeURIComponent(prefix)}`, (refs: { ref: string }[]) =>
    refs.map((r) => r.ref.replace(/^refs\/tags\//, ''))
  )
  const newest = tags
    .filter((t) => hasPrefix(t, prefix))
    .sort((a, b) => compareVersions(b.slice(prefix.length), a.slice(prefix.length)))
  for (const tag of newest.slice(0, TAG_TRIES)) {
    // A tag without a release, or with a draft one, isn't found.
    const raw = await api(`/repos/${repo}/releases/tags/${encodeURIComponent(tag)}`, (r: RawRelease) => slimReleases([r])[0]).catch(
      (e: unknown) => {
        if (e instanceof GitHubError && e.status === 404) return null
        throw e
      }
    )
    if (raw && !raw.draft && !raw.prerelease && accept(toRelease(raw))) return toRelease(raw)
  }
  return null
}

/** "image-organizer-v1.2.0" → "v1.2.0", the form installed versions are kept in. */
export function releaseVersion(tag: string, prefix: string): string {
  return tag.startsWith(prefix) ? `v${tag.slice(prefix.length)}` : tag
}

/** How many commits on `branch` changed `path` after `since` (at most 100). */
export async function commitsSince(repo: string, branch: string, path: string, since: string): Promise<number> {
  // Only the count is kept: the commit objects are large.
  return api(
    `/repos/${repo}/commits?sha=${encodeURIComponent(branch)}&path=${encodeURIComponent(path)}` +
      `&since=${encodeURIComponent(since)}&per_page=100`,
    (list: unknown[]) => list.length
  )
}

/** Recent releases, newest first, pre-releases included. */
export async function recentReleases(repo: string, count = 10): Promise<Release[]> {
  const list = await api(`/repos/${repo}/releases?per_page=${count}`, slimReleases)
  return list.filter((r) => !r.draft).map(toRelease)
}

type RawCommit = { sha: string; commit: { message: string; committer: { date: string } | null } }

/**
 * The newest commit on `branch`. Asked for as a one-commit list: the single-commit answer also carries every changed
 * file with its diff, which for a big commit is megabytes the launcher never reads.
 */
export async function headCommit(repo: string, branch: string): Promise<Commit> {
  const c = await api(`/repos/${repo}/commits?sha=${encodeURIComponent(branch)}&per_page=1`, (list: RawCommit[]): Commit | null =>
    list[0] ? { sha: list[0].sha, date: list[0].commit.committer?.date ?? null, message: list[0].commit.message.split('\n')[0] } : null
  )
  if (!c) throw new GitHubError(`The ${branch} branch has no commits.`, 404)
  return c
}

const headsCache = new Map<string, { at: number; heads: Promise<[branch: string, sha: string][]> }>()

/** Forgets the shared release pages and branch heads, for a check the user asked for. */
export function dropSharedLists(): void {
  pageCache.clear()
  headsCache.clear()
}

/**
 * The commit `branch` points at, or null when that can't be told this way (a branch without a folder in its name, or
 * one that doesn't exist). The source apps' branches (app/rulz-trainer, app/hf-manager…) come from one request,
 * shared for a minute, where asking each branch on its own costs a request per app.
 */
export async function branchSha(repo: string, branch: string): Promise<string | null> {
  const slash = branch.lastIndexOf('/')
  if (slash < 0) return null
  const folder = branch.slice(0, slash + 1)
  const key = `${repo}#${folder}`
  let hit = headsCache.get(key)
  if (!hit || Date.now() - hit.at >= PAGE_TTL_MS) {
    const heads = api(`/repos/${repo}/git/matching-refs/heads/${folder}`, (refs: { ref: string; object: { sha: string } }[]) =>
      refs.map((r): [string, string] => [r.ref.replace(/^refs\/heads\//, ''), r.object.sha])
    )
    hit = { at: Date.now(), heads }
    headsCache.set(key, hit)
    heads.catch(() => headsCache.delete(key))
  }
  return (await hit.heads).find(([name]) => name === branch)?.[1] ?? null
}

export async function rawFile(repo: string, branch: string, path: string): Promise<string> {
  const res = await fetchImpl(`https://raw.githubusercontent.com/${repo}/${branch}/${path}`, {
    signal: AbortSignal.timeout(10_000)
  })
  if (!res.ok) throw new GitHubError(`Couldn't fetch ${path} (${res.status})`, res.status)
  return res.text()
}

export function findAsset(release: Release, pattern: string): ReleaseAsset | undefined {
  const re = new RegExp(pattern, 'i')
  return release.assets.find((a) => re.test(a.name))
}

export function sourceArchiveUrl(repo: string, ref: string): string {
  return `https://api.github.com/repos/${repo}/zipball/${encodeURIComponent(ref)}`
}

function parseVersion(v: string): number[] | null {
  const m = /^v?(\d+(?:\.\d+)*)/.exec(v.trim())
  return m ? m[1].split('.').map(Number) : null
}

/** Orders "1.10.0" after "1.9.2" (positive when `a` is newer); versions that can't be read compare as text. */
export function compareVersions(a: string, b: string): number {
  const x = parseVersion(a)
  const y = parseVersion(b)
  if (!x || !y) return a.localeCompare(b)
  for (let i = 0; i < Math.max(x.length, y.length); i++) {
    const d = (x[i] ?? 0) - (y[i] ?? 0)
    if (d) return d
  }
  return 0
}

/** True when `latest` is a newer version than `current` ("v1.2.0" style tags; anything else compares as text). */
export function isNewer(latest: string, current: string | null): boolean {
  if (!current) return false
  if (!parseVersion(latest) || !parseVersion(current)) return latest !== current
  return compareVersions(latest, current) > 0
}
