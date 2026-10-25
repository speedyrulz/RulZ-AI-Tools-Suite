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

/** The latest full release, or null when the repository has none. */
export async function latestRelease(repo: string): Promise<Release | null> {
  try {
    return toRelease(await api<RawRelease>(`/repos/${repo}/releases/latest`))
  } catch (e) {
    if (e instanceof GitHubError && e.status === 404) {
      // 404 means either "no releases" or "no such repository"; tell them apart.
      await api(`/repos/${repo}`)
      return null
    }
    throw e
  }
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

/** How many commits `head` is ahead of `base` (a tag, branch or SHA). */
export async function commitsAhead(repo: string, base: string, head: string): Promise<number> {
  const c = await api<{ ahead_by: number }>(
    `/repos/${repo}/compare/${encodeURIComponent(base)}...${encodeURIComponent(head)}`
  )
  return c.ahead_by
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
