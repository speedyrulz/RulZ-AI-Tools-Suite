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

async function api<T>(path: string): Promise<T> {
  const res = await fetchImpl(`https://api.github.com${path}`, {
    headers: { Accept: 'application/vnd.github+json', 'User-Agent': 'RulZ-AI-Tools-Suite' },
    signal: AbortSignal.timeout(15_000)
  })
  if (res.ok) return (await res.json()) as T
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

const RELEASE_PAGE = 100
const RELEASE_PAGES = 5
const PAGE_TTL_MS = 60_000
const pageCache = new Map<string, { at: number; list: Promise<RawRelease[]> }>()

/** A page of a repository's releases, newest first. Shared for a minute, so the apps' checks cost one request. */
function releasePage(repo: string, page: number): Promise<RawRelease[]> {
  const key = `${repo}#${page}`
  const hit = pageCache.get(key)
  if (hit && Date.now() - hit.at < PAGE_TTL_MS) return hit.list
  const list = api<RawRelease[]>(`/repos/${repo}/releases?per_page=${RELEASE_PAGE}&page=${page}`)
  pageCache.set(key, { at: Date.now(), list })
  list.catch(() => pageCache.delete(key))
  return list
}

/**
 * The newest full release whose tag is `prefix` followed by a version ("v" for v1.2.0, "image-organizer-v" for
 * image-organizer-v1.2.0), or null when there is none. One repository holds the releases of several apps.
 */
export async function latestRelease(repo: string, prefix = 'v'): Promise<Release | null> {
  const matches = (r: RawRelease): boolean =>
    !r.draft && !r.prerelease && r.tag_name.startsWith(prefix) && /^\d/.test(r.tag_name.slice(prefix.length))
  for (let page = 1; page <= RELEASE_PAGES; page++) {
    const list = await releasePage(repo, page)
    const hit = list.find(matches)
    if (hit) return toRelease(hit)
    if (list.length < RELEASE_PAGE) break
  }
  return null
}

/** "image-organizer-v1.2.0" → "v1.2.0", the form installed versions are kept in. */
export function releaseVersion(tag: string, prefix: string): string {
  return tag.startsWith(prefix) ? `v${tag.slice(prefix.length)}` : tag
}

/** How many commits on `branch` changed `path` after `since` (at most 100). */
export async function commitsSince(repo: string, branch: string, path: string, since: string): Promise<number> {
  const list = await api<unknown[]>(
    `/repos/${repo}/commits?sha=${encodeURIComponent(branch)}&path=${encodeURIComponent(path)}` +
      `&since=${encodeURIComponent(since)}&per_page=100`
  )
  return list.length
}

/** Recent releases, newest first, pre-releases included. */
export async function recentReleases(repo: string, count = 10): Promise<Release[]> {
  const list = await api<RawRelease[]>(`/repos/${repo}/releases?per_page=${count}`)
  return list.filter((r) => !r.draft).map(toRelease)
}

export async function headCommit(repo: string, branch: string): Promise<Commit> {
  const c = await api<{ sha: string; commit: { message: string; committer: { date: string } | null } }>(
    `/repos/${repo}/commits/${encodeURIComponent(branch)}`
  )
  return { sha: c.sha, date: c.commit.committer?.date ?? null, message: c.commit.message.split('\n')[0] }
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

/** True when `latest` is a newer version than `current` ("v1.2.0" style tags; anything else compares as text). */
export function isNewer(latest: string, current: string | null): boolean {
  if (!current) return false
  const parse = (v: string): number[] | null => {
    const m = /^v?(\d+(?:\.\d+)*)/.exec(v.trim())
    return m ? m[1].split('.').map(Number) : null
  }
  const a = parse(latest)
  const b = parse(current)
  if (!a || !b) return latest !== current
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const d = (a[i] ?? 0) - (b[i] ?? 0)
    if (d) return d > 0
  }
  return false
}
