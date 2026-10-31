import { afterEach, describe, expect, it } from 'vitest'
import {
  clearEtags,
  etagSnapshot,
  getFetch,
  headCommit,
  latestRelease,
  loadEtags,
  releaseVersion,
  setFetch
} from '../src/main/github'

const original = getFetch()
afterEach(() => setFetch(original))

function raw(tag: string, extra: Partial<{ draft: boolean; prerelease: boolean }> = {}): object {
  return { tag_name: tag, name: tag, published_at: null, prerelease: false, draft: false, html_url: '', assets: [], ...extra }
}

/** Serves `pages` (100 releases a page means "more follow") and counts the requests. */
function serve(pages: object[][]): { calls: string[] } {
  const calls: string[] = []
  setFetch(async (url) => {
    calls.push(url)
    const page = Number(new URL(url).searchParams.get('page') ?? 1)
    return new Response(JSON.stringify(pages[page - 1] ?? []), { status: 200 })
  })
  return { calls }
}

describe('latestRelease', () => {
  it("picks the newest release with the app's tag prefix", async () => {
    serve([[raw('v1.5.0'), raw('llm-suite-v0.4.0', { prerelease: true }), raw('image-organizer-v1.3.0'), raw('llm-suite-v0.3.9')]])
    expect((await latestRelease('o/r1', 'llm-suite-v'))?.tag).toBe('llm-suite-v0.3.9')
    expect((await latestRelease('o/r1', 'image-organizer-v'))?.tag).toBe('image-organizer-v1.3.0')
  })

  it("keeps the launcher's own v1.2.3 releases apart from the apps'", async () => {
    serve([[raw('rulz-trainer-v2.14.0'), raw('vnext'), raw('v1.5.0')]])
    expect((await latestRelease('o/r2'))?.tag).toBe('v1.5.0')
  })

  it('reads further pages only when needed, and shares them between apps', async () => {
    const full = Array.from({ length: 100 }, (_, i) => raw(`rulz-trainer-v2.${100 - i}.0`))
    const { calls } = serve([full, [raw('image-tournament-v1.1.0')]])
    expect((await latestRelease('o/r3', 'rulz-trainer-v'))?.tag).toBe('rulz-trainer-v2.100.0')
    expect((await latestRelease('o/r3', 'image-tournament-v'))?.tag).toBe('image-tournament-v1.1.0')
    expect(await latestRelease('o/r3', 'clip-extractor-v')).toBeNull()
    expect(calls).toHaveLength(2)
  })
})

describe('ETags', () => {
  afterEach(() => clearEtags())

  it('asks with If-None-Match and reuses the remembered answer on a 304', async () => {
    const seen: (string | undefined)[] = []
    setFetch(async (_url, init) => {
      const tag = (init?.headers as Record<string, string>)['If-None-Match']
      seen.push(tag)
      if (tag === '"abc"') return new Response(null, { status: 304 })
      const body = { sha: 'f00', commit: { message: 'Fix it\n\nmore', committer: { date: '2026-10-01' } }, files: ['big'] }
      return new Response(JSON.stringify(body), { status: 200, headers: { etag: '"abc"' } })
    })
    const first = await headCommit('o/e1', 'app/x')
    const second = await headCommit('o/e1', 'app/x')
    expect(seen).toEqual([undefined, '"abc"'])
    expect(second).toEqual(first)
    expect(first).toEqual({ sha: 'f00', date: '2026-10-01', message: 'Fix it' })
  })

  it('survives a restart through the snapshot, kept slim', async () => {
    setFetch(async () => new Response(JSON.stringify({ sha: 's', commit: { message: 'm', committer: null } }), { status: 200, headers: { etag: 'W/"1"' } }))
    await headCommit('o/e2', 'main')
    const saved = JSON.parse(JSON.stringify(etagSnapshot()))
    clearEtags()
    loadEtags(saved)
    let asked: string | undefined
    setFetch(async (_url, init) => {
      asked = (init?.headers as Record<string, string>)['If-None-Match']
      return new Response(null, { status: 304 })
    })
    expect((await headCommit('o/e2', 'main')).sha).toBe('s')
    expect(asked).toBe('W/"1"')
  })
})

describe('releaseVersion', () => {
  it('strips the app prefix to the form installs keep', () => {
    expect(releaseVersion('image-organizer-v1.2.0', 'image-organizer-v')).toBe('v1.2.0')
    expect(releaseVersion('v1.2.0', 'image-organizer-v')).toBe('v1.2.0')
  })
})
