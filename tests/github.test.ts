import { afterEach, describe, expect, it } from 'vitest'
import { getFetch, latestRelease, releaseVersion, setFetch } from '../src/main/github'

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

describe('releaseVersion', () => {
  it('strips the app prefix to the form installs keep', () => {
    expect(releaseVersion('image-organizer-v1.2.0', 'image-organizer-v')).toBe('v1.2.0')
    expect(releaseVersion('v1.2.0', 'image-organizer-v')).toBe('v1.2.0')
  })
})
