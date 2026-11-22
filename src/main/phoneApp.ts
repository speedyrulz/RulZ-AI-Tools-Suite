// The newest Android companion app APKs, for the launcher's download QR codes.
// - coding-agent-android (apps/rulz-coding-agent-android): tagged coding-agent-android-v<version> with a
//   RulZ-Agent-<version>.apk asset, falling back to its old repository (tagged v<version>) while the suite has none.
// - writer-android (apps/rulz-writer-android): tagged writer-android-v<version> with a RulZ-Writer-<version>.apk asset.
import { SUITE_REPO } from '@shared/catalog'
import type { PhoneApp, PhoneAppId } from '@shared/types'
import { findAsset, latestRelease, releaseVersion } from './github'

export type PhoneSource = [repo: string, prefix: string]

export const PHONE_TAG_PREFIX = 'coding-agent-android-v'
export const WRITER_TAG_PREFIX = 'writer-android-v'
export const PHONE_OLD_REPO = 'speedyrulz/rulz-coding-agent-android'
const APK = '\\.apk$'

/** Where to look for Coding Agent for Android, in order: [repository, tag prefix]. */
export const PHONE_SOURCES: PhoneSource[] = [
  [SUITE_REPO, PHONE_TAG_PREFIX],
  [PHONE_OLD_REPO, 'v']
]

/** Sources per Android app id. */
export const PHONE_APP_SOURCES: Record<PhoneAppId, PhoneSource[]> = {
  'coding-agent-android': PHONE_SOURCES,
  'writer-android': [[SUITE_REPO, WRITER_TAG_PREFIX]]
}

const last = new Map<PhoneAppId, PhoneApp>()

/**
 * The newest release that has an APK. When GitHub can't be reached, the last answer for that app from this run (null
 * before the first); the release lists are cached for a minute and asked again with an ETag, so calling this often is
 * cheap.
 */
export async function latestPhoneApp(
  idOrSources: PhoneAppId | PhoneSource[] = 'coding-agent-android',
  overrideSources?: PhoneSource[]
): Promise<PhoneApp | null> {
  const id: PhoneAppId = typeof idOrSources === 'string' ? idOrSources : 'coding-agent-android'
  const sources: PhoneSource[] = Array.isArray(idOrSources) ? idOrSources : (overrideSources ?? PHONE_APP_SOURCES[id])
  if (!sources) return null
  const prev = last.get(id) ?? null
  try {
    for (const [repo, prefix] of sources) {
      // A newer release without an APK (its build failed, or it is still uploading) leaves the one before it.
      const release = await latestRelease(repo, prefix, (r) => !!findAsset(r, APK))
      const apk = release && findAsset(release, APK)
      if (release && apk) {
        const found: PhoneApp = {
          version: releaseVersion(release.tag, prefix),
          tag: release.tag,
          url: apk.url,
          size: apk.size,
          date: release.date,
          releaseUrl: release.htmlUrl
        }
        last.set(id, found)
        return found
      }
    }
    return prev
  } catch (e) {
    if (prev) return prev
    throw e
  }
}

/** For tests. */
export function forgetPhoneApp(id?: PhoneAppId): void {
  if (id) last.delete(id)
  else last.clear()
}
