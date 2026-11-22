import { afterEach, describe, expect, it } from 'vitest'
import { getFetch, setFetch } from '../src/main/github'
import {
  forgetPhoneApp,
  latestPhoneApp,
  PHONE_APP_SOURCES,
  PHONE_OLD_REPO,
  PHONE_TAG_PREFIX,
  WRITER_TAG_PREFIX
} from '../src/main/phoneApp'
import { SUITE_REPO } from '../src/shared/catalog'

const original = getFetch()
afterEach(() => {
  setFetch(original)
  forgetPhoneApp()
})

function release(tag: string, assets: string[], extra: object = {}): object {
  return {
    tag_name: tag,
    name: tag,
    published_at: '2026-11-09T13:40:08Z',
    prerelease: false,
    draft: false,
    html_url: `https://github.com/x/releases/tag/${tag}`,
    assets: assets.map((name) => ({ name, size: 4_679_802, browser_download_url: `https://github.com/x/download/${tag}/${name}` })),
    ...extra
  }
}

/** Serves each repository's release list ('down' fails the request). Release lists are cached per repository for a
 * minute, so every test uses repository names of its own. */
function serve(lists: Record<string, object[] | 'down'>): void {
  setFetch(async (url) => {
    const repo = Object.keys(lists).find((r) => url.includes(`/repos/${r}/releases`))
    const list = repo ? lists[repo] : []
    if (list === 'down') throw new TypeError('fetch failed')
    return new Response(JSON.stringify(list), { status: 200 })
  })
}

describe('latestPhoneApp', () => {
  it("takes the suite's newest Android release that has an APK", async () => {
    serve({
      's/a': [
        release('v1.11.0', ['RulZ-AI-Tools-Suite-Setup-1.11.0.exe']),
        release('coding-agent-android-v0.7.0', ['RulZ-Agent-0.7.0.apk'], { draft: true }),
        release('coding-agent-android-v0.6.1', ['RulZ-Agent-0.6.1.apk'], { prerelease: true }),
        release('coding-agent-android-v0.6.0', ['SHA256SUMS', 'RulZ-Agent-0.6.0.apk']),
        release('coding-agent-android-v0.5.0', ['RulZ-Agent-0.5.0.apk'])
      ],
      'old/a': [release('v0.9.0', ['RulZ-Agent-0.9.0.apk'])]
    })
    const phone = await latestPhoneApp([['s/a', PHONE_TAG_PREFIX], ['old/a', 'v']])
    expect(phone).toEqual({
      version: 'v0.6.0',
      tag: 'coding-agent-android-v0.6.0',
      url: 'https://github.com/x/download/coding-agent-android-v0.6.0/RulZ-Agent-0.6.0.apk',
      size: 4_679_802,
      date: '2026-11-09T13:40:08Z',
      releaseUrl: 'https://github.com/x/releases/tag/coding-agent-android-v0.6.0'
    })
  })

  it('falls back to the old repository while the suite has no Android release', async () => {
    serve({ 's/b': [release('v1.10.2', ['setup.exe'])], 'old/b': [release('v0.5.0', ['RulZ-Agent-0.5.0.apk'])] })
    const phone = await latestPhoneApp([['s/b', PHONE_TAG_PREFIX], ['old/b', 'v']])
    expect(phone?.url).toBe('https://github.com/x/download/v0.5.0/RulZ-Agent-0.5.0.apk')
    expect(phone?.version).toBe('v0.5.0')
  })

  it('skips a release without an APK', async () => {
    serve({ 's/c': [release('coding-agent-android-v0.6.0', ['SHA256SUMS'])], 'old/c': [] })
    expect(await latestPhoneApp([['s/c', PHONE_TAG_PREFIX], ['old/c', 'v']])).toBeNull()
  })

  it('keeps the last answer when GitHub is down, and fails before there is one', async () => {
    serve({ 's/d': 'down' })
    await expect(latestPhoneApp('coding-agent-android', [['s/d', PHONE_TAG_PREFIX]])).rejects.toThrow()
    serve({ 's/e': [release('coding-agent-android-v0.6.0', ['RulZ-Agent-0.6.0.apk'])] })
    await latestPhoneApp('coding-agent-android', [['s/e', PHONE_TAG_PREFIX]])
    serve({ 's/f': 'down' })
    expect((await latestPhoneApp('coding-agent-android', [['s/f', PHONE_TAG_PREFIX]]))?.version).toBe('v0.6.0')
  })

  it('configures writer-android with the suite repo only (no old-repo fallback)', () => {
    expect(PHONE_APP_SOURCES['coding-agent-android']).toEqual([
      [SUITE_REPO, PHONE_TAG_PREFIX],
      [PHONE_OLD_REPO, 'v']
    ])
    expect(PHONE_APP_SOURCES['writer-android']).toEqual([[SUITE_REPO, WRITER_TAG_PREFIX]])
  })

  it('looks up and caches writer-android independently from coding-agent-android', async () => {
    serve({
      's/w1': [
        release('coding-agent-android-v0.6.0', ['RulZ-Agent-0.6.0.apk']),
        release('writer-android-v0.2.0', ['RulZ-Writer-0.2.0.apk'], { draft: true }),
        release('writer-android-v0.1.0', ['SHA256SUMS', 'RulZ-Writer-0.1.0.apk'])
      ]
    })
    const writer = await latestPhoneApp('writer-android', [['s/w1', WRITER_TAG_PREFIX]])
    expect(writer).toEqual({
      version: 'v0.1.0',
      tag: 'writer-android-v0.1.0',
      url: 'https://github.com/x/download/writer-android-v0.1.0/RulZ-Writer-0.1.0.apk',
      size: 4_679_802,
      date: '2026-11-09T13:40:08Z',
      releaseUrl: 'https://github.com/x/releases/tag/writer-android-v0.1.0'
    })

    // coding-agent-android has not been fetched yet, so a down network rejects for it while writer-android returns its cached release.
    serve({ 's/w2': 'down' })
    await expect(latestPhoneApp('coding-agent-android', [['s/w2', PHONE_TAG_PREFIX]])).rejects.toThrow()
    expect((await latestPhoneApp('writer-android', [['s/w2', WRITER_TAG_PREFIX]]))?.version).toBe('v0.1.0')
  })

  it('returns null for writer-android when no writer release exists yet', async () => {
    serve({
      's/w3': [
        release('v1.18.1', ['RulZ-AI-Tools-Suite-Setup-1.18.1.exe']),
        release('coding-agent-android-v0.6.0', ['RulZ-Agent-0.6.0.apk'])
      ]
    })
    expect(await latestPhoneApp('writer-android', [['s/w3', WRITER_TAG_PREFIX]])).toBeNull()
  })
})
