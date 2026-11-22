// How the phone-access settings reach the window. The pairing code never goes to the renderer (only phone:pairing
// carries it), and the window changes phone access only through the phone:* channels, not settings:update.
import type { AppSettings } from '@shared/types'

/** Settings as the renderer sees them: no pairing code. */
export function settingsForRenderer(s: AppSettings): AppSettings {
  return { ...s, phoneToken: '' }
}

/**
 * A settings change from the renderer, without the phone-access fields it must not set: the pairing code (the
 * renderer sees it as empty and would write that blank back) and the on/off switch (phone:setEnabled does that,
 * and makes the first code). The port is allowed; the caller restarts phone access when it changes.
 */
export function settingsPatchFromRenderer(patch: Partial<AppSettings>): Partial<AppSettings> {
  const { phoneToken: _token, phoneEnabled: _enabled, ...rest } = patch
  return rest
}
