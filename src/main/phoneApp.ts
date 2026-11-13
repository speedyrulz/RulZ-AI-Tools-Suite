// The newest RulZ Coding Agent for Android APK, for the launcher's download QR code.
// Its source is apps/rulz-coding-agent-android; the Apps workflow publishes each version as a release tagged
// coding-agent-android-v<version> with a RulZ-Agent-<version>.apk asset. Releases from before the app moved into the
// suite are in its old repository, tagged v<version>; they are only used while the suite has none.
import { SUITE_REPO } from '@shared/catalog'
import type { PhoneApp } from '@shared/types'
import { findAsset, latestRelease, releaseVersion } from './github'

export const PHONE_TAG_PREFIX = 'coding-agent-android-v'
export const PHONE_OLD_REPO = 'speedyrulz/rulz-coding-agent-android'
const APK = '\\.apk$'

/** Where to look, in order: [repository, tag prefix]. */
export const PHONE_SOURCES: [repo: string, prefix: string][] = [
  [SUITE_REPO, PHONE_TAG_PREFIX],
  [PHONE_OLD_REPO, 'v']
]

let last: PhoneApp | null = null

/**
 * The newest release that has an APK. When GitHub can't be reached, the last answer from this run (null before the
 * first); the release lists are cached for a minute and asked again with an ETag, so calling this often is cheap.
 */
export async function latestPhoneApp(sources = PHONE_SOURCES): Promise<PhoneApp | null> {
  try {
    for (const [repo, prefix] of sources) {
      // A newer release without an APK (its build failed, or it is still uploading) leaves the one before it.
      const release = await latestRelease(repo, prefix, (r) => !!findAsset(r, APK))
      const apk = release && findAsset(release, APK)
      if (release && apk) {
        last = {
          version: releaseVersion(release.tag, prefix),
          tag: release.tag,
          url: apk.url,
          size: apk.size,
          date: release.date,
          releaseUrl: release.htmlUrl
        }
        return last
      }
    }
    return last
  } catch (e) {
    if (last) return last
    throw e
  }
}

/** For tests. */
export function forgetPhoneApp(): void {
  last = null
}
