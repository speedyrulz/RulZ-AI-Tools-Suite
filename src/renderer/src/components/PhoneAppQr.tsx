import { Copy, ExternalLink, QrCode, Smartphone } from 'lucide-react'
import QRCode from 'qrcode'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { create } from 'zustand'
import type { PhoneApp, PhoneAppId } from '@shared/types'
import { invoke } from '../lib/api'
import { cn, formatBytes, formatDate } from '../lib/format'
import { useApps } from '../stores/apps'
import { Button, Dialog } from './ui'

// QR codes with the direct download links of the newest RulZ Android companion APKs (RulZ Coding Agent for Android and
// RulZ Writer for Android). Each link comes from the newest release on GitHub, so a new release changes the code: it is
// looked up when the launcher starts, every 30 minutes, and whenever the app list is checked again (Check for updates).

const RECHECK_MS = 30 * 60_000
/** The main process keeps GitHub's answer for a minute, so asking again sooner changes nothing. */
const MIN_GAP_MS = 60_000
const SAVED = 'rulz.phoneApp'

const PHONE_META: Record<PhoneAppId, { name: string; buttonLabel: string; description: string }> = {
  'coding-agent-android': {
    name: 'RulZ Coding Agent for Android',
    buttonLabel: 'Coding Agent for Android',
    description: 'Follow and steer RulZ Coding Agent from your phone.'
  },
  'writer-android': {
    name: 'RulZ Writer for Android',
    buttonLabel: 'Writer for Android',
    description: "Write in RulZ LLM Suite's Writing mode from your phone."
  }
}

const PHONE_IDS = Object.keys(PHONE_META) as PhoneAppId[]

interface PhoneEntry {
  phone: PhoneApp | null
  /** Set when GitHub couldn't be asked and nothing is known yet. */
  error: string | null
}

interface PhoneStore {
  byId: Record<PhoneAppId, PhoneEntry>
  check: () => Promise<void>
}

function savedKey(id: PhoneAppId): string {
  return id === 'coding-agent-android' ? SAVED : `${SAVED}.${id}`
}

function saved(id: PhoneAppId): PhoneApp | null {
  try {
    return JSON.parse(localStorage.getItem(savedKey(id)) ?? 'null') as PhoneApp | null
  } catch {
    return null
  }
}

const usePhone = create<PhoneStore>((set, get) => ({
  // The last known release shows straight away, also when the launcher starts offline.
  byId: {
    'coding-agent-android': { phone: saved('coding-agent-android'), error: null },
    'writer-android': { phone: saved('writer-android'), error: null }
  },
  check: async () => {
    await Promise.all(
      PHONE_IDS.map(async (id) => {
        try {
          const phone = await invoke('phone:latest', id)
          if (phone) {
            try {
              localStorage.setItem(savedKey(id), JSON.stringify(phone))
            } catch {
              // Only a convenience.
            }
            set((s) => ({ byId: { ...s.byId, [id]: { phone, error: null } } }))
          } else if (!get().byId[id].phone) {
            set((s) => ({ byId: { ...s.byId, [id]: { phone: null, error: 'No Android release found yet.' } } }))
          }
        } catch (e) {
          if (!get().byId[id].phone) {
            const msg = e instanceof Error ? e.message : String(e)
            set((s) => ({ byId: { ...s.byId, [id]: { phone: null, error: msg } } }))
          }
        }
      })
    )
  }
}))

let started = false
let lastCheck = 0

function checkNow(): void {
  if (Date.now() - lastCheck < MIN_GAP_MS) return
  lastCheck = Date.now()
  void usePhone.getState().check()
}

/**
 * Starts the checks, once for the whole window however many places show the app: on first use, every 30 minutes,
 * and after the app list changes (a refresh sends a new list; so does install progress, which the one-minute gap
 * keeps from turning into a stream of lookups).
 */
function startChecks(): void {
  if (started) return
  started = true
  checkNow()
  setInterval(checkNow, RECHECK_MS)
  let apps = useApps.getState().apps
  useApps.subscribe((s) => {
    if (s.apps === apps) return
    apps = s.apps
    checkNow()
  })
}

/** The newest Android release for `appId`, as last looked up. */
function usePhoneApp(appId: PhoneAppId): PhoneEntry {
  useEffect(startChecks, [])
  return usePhone((s) => s.byId[appId])
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

/** The QR code with its explanation: `card` for the side panel of an app's page, `dialog` (larger) for the home page's dialog. */
export function PhoneAppQr({
  appId = 'coding-agent-android',
  variant
}: {
  appId?: PhoneAppId
  variant: 'card' | 'dialog'
}) {
  const { phone, error } = usePhoneApp(appId)
  const svg = useQrSvg(phone?.url)
  const meta = PHONE_META[appId]

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
        variant === 'dialog' ? 'size-[208px]' : 'size-[148px]',
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
        <Smartphone className="size-3.5 text-accent-text" /> {meta.name}
      </div>
      {phone ? (
        <>
          <p className="mt-1 text-xs text-faint">
            {phone.version} · {formatBytes(phone.size)}
            {phone.date ? ` · ${formatDate(phone.date)}` : ''}
          </p>
          <p className="mt-2 text-xs leading-relaxed text-muted">
            Scan with the phone's camera to download the APK, then open it to install (allow installs from that source).
            Needs Android 7.0 or later, and Tailscale on the phone and this computer.
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

  if (variant === 'dialog') {
    return (
      <div className="flex items-center gap-5">
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

/** The home page's button for an Android companion app: opens the QR code in a dialog, large enough to scan from a chair. */
export function PhoneAppButton({ appId = 'coding-agent-android' }: { appId?: PhoneAppId }) {
  const { phone } = usePhoneApp(appId)
  const [open, setOpen] = useState(false)
  const meta = PHONE_META[appId]
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        title={`Show the QR code that downloads ${meta.name}`}
        className="flex shrink-0 cursor-pointer items-center gap-3 rounded-xl border border-subtle-border bg-sunken py-2 pr-4 pl-2.5 text-left backdrop-blur transition-colors hover:border-accent/60"
      >
        <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-white text-black">
          <QrCode className="size-6" />
        </span>
        <span className="min-w-0">
          <span className="block text-[13px] font-semibold text-text">{meta.buttonLabel}</span>
          <span className="block text-xs text-faint">{phone ? `${phone.version} · Show QR code` : 'Show QR code'}</span>
        </span>
      </button>
      {open && (
        <Dialog
          open
          onOpenChange={(o) => !o && setOpen(false)}
          title="Get the Android app"
          description={meta.description}
          className="w-[560px]"
          footer={<Button onClick={() => setOpen(false)}>Done</Button>}
        >
          <PhoneAppQr appId={appId} variant="dialog" />
        </Dialog>
      )}
    </>
  )
}
