import { App } from '@capacitor/app'
import { Browser } from '@capacitor/browser'
import { Capacitor } from '@capacitor/core'
import { Preferences } from '@capacitor/preferences'

// Phone features, with plain-browser fallbacks so the app also runs in a desktop browser while developing.

export const isNative = Capacitor.isNativePlatform()

export async function loadPref<T>(key: string, fallback: T): Promise<T> {
  try {
    const raw = isNative ? (await Preferences.get({ key })).value : localStorage.getItem(key)
    return raw ? (JSON.parse(raw) as T) : fallback
  } catch {
    return fallback
  }
}

export async function savePref(key: string, value: unknown): Promise<void> {
  const raw = JSON.stringify(value)
  try {
    if (isNative) await Preferences.set({ key, value: raw })
    else localStorage.setItem(key, raw)
  } catch {
    // Nothing else to try; the value lasts until the app closes.
  }
}

export async function openInBrowser(url: string): Promise<void> {
  if (isNative) await Browser.open({ url })
  else window.open(url, '_blank', 'noopener')
}

/** Calls `fn` with each link the phone opens the app with: pairing links (QR code) and tapped notifications. */
export function onPairingLink(fn: (url: string) => void): void {
  if (!isNative) {
    const q = new URLSearchParams(location.search).get('pair')
    if (q) fn(q)
    return
  }
  void App.getLaunchUrl().then((r) => r?.url && fn(r.url))
  void App.addListener('appUrlOpen', (e) => fn(e.url))
}

/** Calls `fn` when the app comes back to the front. */
export function onResume(fn: () => void): void {
  if (isNative) void App.addListener('resume', fn)
  else document.addEventListener('visibilitychange', () => document.visibilityState === 'visible' && fn())
}

/** The Android back button. Return true when handled; otherwise the app goes to the background. */
export function onBackButton(fn: () => boolean): void {
  if (!isNative) return
  void App.addListener('backButton', () => {
    if (!fn()) void App.minimizeApp()
  })
}

/** What the phone calls itself, shown on the computer ("Pixel 9 connected"). */
export async function deviceName(): Promise<string> {
  try {
    const uaData = (navigator as Navigator & { userAgentData?: { getHighEntropyValues(h: string[]): Promise<{ model?: string }> } }).userAgentData
    const model = (await uaData?.getHighEntropyValues(['model']))?.model
    if (model) return model
  } catch {
    // fall through
  }
  const m = /Android [\d.]+; ([^;)]+?)(?: Build\/[^;)]*)?[;)]/.exec(navigator.userAgent)
  if (m && m[1] !== 'K' && m[1] !== 'wv') return m[1].trim()
  return isNative ? 'Android phone' : 'Web browser'
}
