import { Copy, ExternalLink, Smartphone } from 'lucide-react'
import QRCode from 'qrcode'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { create } from 'zustand'
import type { PhoneApp } from '@shared/types'
import { invoke } from '../lib/api'
import { cn, formatBytes, formatDate } from '../lib/format'
import { useApps } from '../stores/apps'

// A QR code with the direct download link of the newest RulZ Coding Agent for Android APK. The link comes from the
// newest release on GitHub, so a new release changes the code: it is looked up when the launcher starts, every
// 30 minutes, and whenever the app list is checked again (Check for updates).

const RECHECK_MS = 30 * 60_000
const SAVED = 'rulz.phoneApp'

interface PhoneState {
  phone: PhoneApp | null
  /** Set when GitHub couldn't be asked and nothing is known yet. */
  error: string | null
  check: () => Promise<void>
}

function saved(): PhoneApp | null {
  try {
    return JSON.parse(localStorage.getItem(SAVED) ?? 'null') as PhoneApp | null
  } catch {
    return null
  }
}

const usePhone = create<PhoneState>((set, get) => ({
  // The last known release shows straight away, also when the launcher starts offline.
  phone: saved(),
  error: null,
  check: async () => {
    try {
      const phone = await invoke('phone:latest')
      if (phone) {
        try {
          localStorage.setItem(SAVED, JSON.stringify(phone))
        } catch {
          // Only a convenience.
        }
        set({ phone, error: null })
      } else if (!get().phone) set({ error: 'No Android release found yet.' })
    } catch (e) {
      if (!get().phone) set({ error: e instanceof Error ? e.message : String(e) })
    }
  }
}))

let started = false

/** Checks once on first use, again every 30 minutes and after each app list refresh. */
function usePhoneApp(): PhoneState {
  const state = usePhone()
  const apps = useApps((s) => s.apps)
  useEffect(() => {
    if (started) return
    started = true
    setInterval(() => void usePhone.getState().check(), RECHECK_MS)
  }, [])
  useEffect(() => {
    void usePhone.getState().check()
  }, [apps])
  return state
}

function useQrSvg(text: string | undefined): string | null {
  const [svg, setSvg] = useState<string | null>(null)
  useEffect(() => {
    if (!text) return setSvg(null)
    let live = true
    QRCode.toString(text, { type: 'svg', margin: 1, errorCorrectionLevel: 'M', color: { dark: '#000000', light: '#ffffff' } })
      .then((s) => live && setSvg(s))
      .catch(() => live && setSvg(null))
    return () => {
      live = false
    }
  }, [text])
  return svg
}

export function PhoneAppQr({ variant }: { variant: 'hero' | 'card' }) {
  const { phone, error } = usePhoneApp()
  const svg = useQrSvg(phone?.url)

  const copy = (): void => {
    if (!phone) return
    navigator.clipboard.writeText(phone.url).then(
      () => toast.success('Download link copied'),
      () => toast.error("Couldn't copy the link")
    )
  }

  const code = (
    <div
      className={cn(
        'shrink-0 rounded-lg bg-white p-2',
        variant === 'hero' ? 'size-[112px]' : 'size-[148px]',
        !svg && 'flex items-center justify-center'
      )}
      title={phone ? phone.url : undefined}
    >
      {svg ? (
        <div className="size-full [&>svg]:block [&>svg]:size-full" dangerouslySetInnerHTML={{ __html: svg }} />
      ) : (
        <Smartphone className="size-8 text-neutral-400" />
      )}
    </div>
  )

  const text = (
    <div className="min-w-0">
      <div className="flex items-center gap-1.5 text-[13px] font-semibold text-text">
        <Smartphone className="size-3.5 text-accent-text" /> RulZ Coding Agent for Android
      </div>
      {phone ? (
        <>
          <p className="mt-1 text-xs text-faint">
            {phone.version} · {formatBytes(phone.size)}
            {phone.date ? ` · ${formatDate(phone.date)}` : ''}
          </p>
          <p className="mt-2 text-xs leading-relaxed text-muted">
            Scan with the phone's camera to download the APK, then open it to install (allow installs from that source).
          </p>
          <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs">
            <button className="inline-flex cursor-pointer items-center gap-1 text-link hover:underline" onClick={copy}>
              <Copy className="size-3" /> Copy link
            </button>
            <button
              className="inline-flex cursor-pointer items-center gap-1 text-link hover:underline"
              onClick={() => void invoke('shell:openExternal', phone.releaseUrl)}
            >
              <ExternalLink className="size-3" /> Release notes
            </button>
          </div>
        </>
      ) : (
        <p className="mt-1 text-xs text-faint">{error ?? 'Looking for the newest release…'}</p>
      )}
    </div>
  )

  if (variant === 'hero') {
    return (
      <div className="flex max-w-[400px] items-center gap-4 rounded-xl border border-subtle-border bg-sunken p-3 backdrop-blur">
        {code}
        {text}
      </div>
    )
  }
  return (
    <div className="flex flex-col items-start gap-3">
      {code}
      {text}
    </div>
  )
}
