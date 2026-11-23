import { ArrowDown, ArrowLeft, ArrowUp, CircleAlert, LoaderCircle, Search } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { DEFAULT_AUTO_STOP, type AutoStop, type OfferFilter, type OfferSort, type RemoteOffer, type RemoteState } from '@shared/remote'
import { errorMessage, invoke } from '../../lib/api'
import { cn, formatBytes } from '../../lib/format'
import {
  cleanAutoStop,
  dlperfPerDollar,
  minVramForModel,
  money,
  offerVramGb,
  readRentPrefs,
  remoteModelSize,
  rentBlockedReason,
  rentButtonLabel,
  rentSummary,
  sortOffers,
  type RentPrefs,
  type SortDir
} from '../../lib/remote'
import { isChannelBusy, useRemote } from '../../stores/remote'
import { useModels } from '../../stores/models'
import { toast } from 'sonner'
import { Button, Dialog, Field, Input, NumberInput, Select, Switch } from '../ui'

const PREFS_KEY = 'rulz.vast.rent'

function loadPrefs(): RentPrefs {
  try {
    return readRentPrefs(localStorage.getItem(PREFS_KEY))
  } catch {
    return readRentPrefs(null)
  }
}

/**
 * Renting, in two steps: find an offer, then a dialog that states the price and what stays billed. Nothing is sent
 * until the one primary button of the second step is pressed; Enter in the filter only searches.
 */
export function RentDialog({ state, onClose }: { state: RemoteState; onClose: () => void }) {
  const run = useRemote((s) => s.run)
  const renting = useRemote((s) => isChannelBusy(s, 'remote:rent'))
  const library = useModels((s) => s.models)
  const [prefs] = useState(loadPrefs)
  const [filter, setFilter] = useState<OfferFilter>(prefs.filter)
  const [forModel, setForModel] = useState(state.models.some((m) => m.id === prefs.forModel) ? prefs.forModel : '')
  // With the filter and model they were searched for: the rent check, disk size and model must be the ones the list is for.
  const [found, setFound] = useState<{ offers: RemoteOffer[]; filter: OfferFilter; forModel: string } | null>(null)
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState('')
  const [dir, setDir] = useState<SortDir>(prefs.dir)
  const [picked, setPicked] = useState<{ offer: RemoteOffer; filter: OfferFilter } | null>(null)
  const [auto, setAuto] = useState<AutoStop>({ ...DEFAULT_AUTO_STOP, idleMinutes: prefs.idleMinutes, capHours: prefs.capHours })
  const [setupModel, setSetupModel] = useState('')
  const seq = useRef(0)

  const def = state.models.find((m) => m.id === forModel)
  const modelBytes = def ? remoteModelSize(def, library) : null
  const needVram = minVramForModel(modelBytes)
  // The model raises the minimum, never lowers what was typed.
  const effective: OfferFilter = { ...filter, minVramGb: Math.max(filter.minVramGb, needVram ?? 0) }
  const set = <K extends keyof OfferFilter>(k: K, v: OfferFilter[K]): void => setFilter((f) => ({ ...f, [k]: v }))
  const sort = { key: filter.sort, dir }
  const rows = useMemo(() => (found ? sortOffers(found.offers, filter.sort, dir) : null), [found, filter.sort, dir])
  const blocked = rentBlockedReason(state)
  // Enter in a number field commits its draft and submits in the same event; the search reads the committed filter.
  const latest = useRef(effective)
  latest.current = effective
  const lastCap = useRef(prefs.capHours)
  lastCap.current = auto.capHours ?? lastCap.current

  useEffect(() => {
    try {
      const keep: RentPrefs = { filter, dir, forModel, idleMinutes: auto.idleMinutes, capHours: lastCap.current }
      localStorage.setItem(PREFS_KEY, JSON.stringify(keep))
    } catch {
      // Storage may be blocked; the window works without it.
    }
  }, [filter, dir, forModel, auto.idleMinutes, auto.capHours])

  const search = async (by: OfferFilter = latest.current): Promise<void> => {
    const mine = ++seq.current
    setSearching(true)
    setSearchError('')
    try {
      const offers = await invoke('remote:searchOffers', by)
      if (mine === seq.current) setFound({ offers, filter: by, forModel })
    } catch (e) {
      if (mine === seq.current) setSearchError(errorMessage(e))
    } finally {
      if (mine === seq.current) setSearching(false)
    }
  }

  const sortBy = (key: OfferSort): void => {
    if (filter.sort === key) return setDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    set('sort', key)
    setDir(key === 'price' ? 'asc' : 'desc')
    // vast.ai returns only part of the matches, so the best ones by the new column have to be asked for.
    if (found) void search({ ...latest.current, sort: key })
  }

  if (picked) {
    const summary = rentSummary(picked.offer, { diskGb: picked.filter.diskGb, autoStop: cleanAutoStop(auto), credit: state.credit, modelBytes: setupModel ? (remoteModelSizeOf(state, setupModel, library) ?? null) : null })
    const sshWarn = state.ssh.onAccount === false || !state.ssh.publicKeyPath
    return (
      <Dialog
        open
        onOpenChange={(o) => !o && onClose()}
        title="Rent this instance?"
        description="Nothing is charged until you press the red button."
        className="w-[min(640px,calc(100vw-32px))]"
        footer={
          <>
            <Button variant="ghost" disabled={renting} onClick={() => setPicked(null)}>
              <ArrowLeft size={14} /> Back
            </Button>
            <Button autoFocus variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button
              type="button"
              variant="primary"
              disabled={renting || !!blocked}
              onClick={async () => {
                const ok = await run('remote:rent', {
                  offerId: picked.offer.id,
                  shownDollarsPerHour: picked.offer.dollarsPerHour,
                  filter: picked.filter,
                  autoStop: cleanAutoStop(auto),
                  modelId: setupModel
                })
                if (ok) {
                  toast.success('Rent request sent. The instance takes a minute or two to start.')
                  onClose()
                }
              }}
            >
              {renting && <LoaderCircle size={14} className="animate-spin" />} {rentButtonLabel(picked.offer.dollarsPerHour)}
            </Button>
          </>
        }
      >
        <div className="space-y-3 px-5 py-4 text-[13px] text-body">
          <ul className="space-y-1.5">
            {summary.lines.map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
          <div className="space-y-2 rounded-lg border border-border p-3">
            <Field inline label="Stop when idle" hint="Nothing used it and the GPUs were quiet">
              <Switch checked={auto.enabled} onCheckedChange={(v) => setAuto((a) => ({ ...a, enabled: v }))} />
            </Field>
            <Field inline label="Idle minutes">
              <NumberInput value={auto.idleMinutes} min={1} max={1440} onChange={(v) => setAuto((a) => ({ ...a, idleMinutes: Math.max(1, Math.round(v ?? 30)) }))} />
            </Field>
            <Field inline label="Hard cap, hours after it starts" hint="Empty = no cap">
              <NumberInput value={auto.capHours} placeholder="None" min={1} max={720} onChange={(v) => setAuto((a) => ({ ...a, capHours: v === null ? null : Math.max(1, v) }))} />
            </Field>
          </div>
          {state.models.length > 0 && (
            <Field label="Set up a model once it runs" hint="Downloads or uploads its files; the traffic is billed as above">
              <Select value={setupModel} onChange={(e) => setSetupModel(e.target.value)}>
                <option value="">None yet</option>
                {state.models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </Select>
            </Field>
          )}
          {summary.warnings.map((w, i) => (
            <Warn key={i}>{w}</Warn>
          ))}
          {sshWarn && <Warn>Your ssh key is not on your vast.ai account yet. Add it in the Account card first, or the app cannot log in to the new instance.</Warn>}
          {blocked && <Warn>{blocked}</Warn>}
        </div>
      </Dialog>
    )
  }

  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title="Rent an instance"
      description="Search the on-demand offers, then pick one. You see the price and settings before anything is rented."
      className="w-[min(1100px,calc(100vw-32px))]"
      footer={
        <Button autoFocus variant="ghost" onClick={onClose}>
          Close
        </Button>
      }
    >
      <form
        className="space-y-3 border-b border-border px-5 py-4"
        onSubmit={(e) => {
          e.preventDefault()
          setTimeout(() => void search(), 0)
        }}
      >
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Field label="For model" hint={needVram ? `needs about ${needVram} GB` : undefined}>
            <Select value={forModel} onChange={(e) => setForModel(e.target.value)} aria-label="For model">
              <option value="">Any (no model chosen)</option>
              {state.models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Min video memory, GB">
            <NumberInput className="w-full" value={effective.minVramGb} min={1} onChange={(v) => set('minVramGb', Math.max(1, Math.round(v ?? 1)))} />
          </Field>
          <Field label="GPU name" hint="empty = any">
            <Input value={filter.gpuName} placeholder="e.g. RTX 4090" onChange={(e) => set('gpuName', e.target.value)} />
          </Field>
          <Field label="Most GPUs">
            <NumberInput className="w-full" value={filter.maxGpus} min={1} max={8} onChange={(v) => set('maxGpus', Math.max(1, Math.round(v ?? 1)))} />
          </Field>
          <Field label="Max price, $/hour" hint="empty = no limit">
            <NumberInput className="w-full" value={filter.maxDollarsPerHour} placeholder="No limit" min={0} step={0.05} onChange={(v) => set('maxDollarsPerHour', v === null ? null : Math.max(0, v))} />
          </Field>
          <Field label="Disk, GB">
            <NumberInput className="w-full" value={filter.diskGb} min={10} onChange={(v) => set('diskGb', Math.max(10, Math.round(v ?? 60)))} />
          </Field>
          <Field label="Min reliability, %">
            <NumberInput className="w-full" value={Math.round(filter.minReliability * 100)} min={0} max={100} onChange={(v) => set('minReliability', Math.min(1, Math.max(0, (v ?? 0) / 100)))} />
          </Field>
          <Field label="Min download, Mbps">
            <NumberInput className="w-full" value={filter.minInetDownMbps} min={0} onChange={(v) => set('minInetDownMbps', Math.max(0, Math.round(v ?? 0)))} />
          </Field>
        </div>
        <div className="flex items-center gap-3">
          <Button type="submit" disabled={searching}>
            {searching ? <LoaderCircle size={14} className="animate-spin" /> : <Search size={14} />} Search offers
          </Button>
          <span className="text-[11px] text-faint">
            Enter searches; it never rents. vast.ai shows up to 64 offers, a different selection each time.{def && modelBytes ? ` ${def.name} is ${formatBytes(modelBytes)}.` : ''}
          </span>
        </div>
      </form>

      <div className="px-5 py-3">
        {searchError && <div className="mb-2 text-xs text-danger-text">{searchError}</div>}
        {rows === null ? (
          <p className="py-6 text-center text-sm text-muted">Search to see what is for rent.</p>
        ) : rows.length === 0 ? (
          <p className="py-6 text-center text-sm text-muted">No offer matches. Lower the video memory, reliability or download speed, or raise the price limit.</p>
        ) : (
          <table className="w-full table-fixed text-[12px]">
            <thead className="text-left text-[10.5px] tracking-wide text-faint uppercase">
              <tr>
                <th className="py-1.5 font-medium">GPU</th>
                <SortTh k="vram" sort={sort} onSort={sortBy} className="w-20">
                  VRAM
                </SortTh>
                <SortTh k="price" sort={sort} onSort={sortBy} className="w-20">
                  $/hour
                </SortTh>
                <SortTh k="value" sort={sort} onSort={sortBy} className="w-28">
                  DLPerf/$/hr
                </SortTh>
                <SortTh k="reliability" sort={sort} onSort={sortBy} className="w-20">
                  Reliable
                </SortTh>
                <SortTh k="down" sort={sort} onSort={sortBy} className="w-24">
                  Down Mbps
                </SortTh>
                <th className="w-28 py-1.5 font-medium">Traffic $/GB</th>
                <th className="w-32 py-1.5 font-medium">Location</th>
                <th className="w-16 py-1.5 font-medium">Max days</th>
                <th className="w-20" />
              </tr>
            </thead>
            <tbody>
              {rows.map((o) => (
                <tr key={o.id} className="border-t border-border/60 hover:bg-hover/40">
                  <td className="truncate py-1.5 pr-2">
                    {o.numGpus > 1 ? `${o.numGpus}× ` : ''}
                    {o.gpuName}
                  </td>
                  <td className="tabular-nums">{offerVramGb(o)} GB</td>
                  <td className="font-medium tabular-nums">{money(o.dollarsPerHour)}</td>
                  <td className="tabular-nums" title={o.dlperf === null ? undefined : `DLPerf ${o.dlperf.toFixed(1)}`}>
                    {dlperfPerDollar(o)?.toFixed(0) ?? '—'}
                  </td>
                  <td className="tabular-nums">{Math.round(o.reliability * 1000) / 10}%</td>
                  <td className="tabular-nums">{Math.round(o.inetDownMbps)}</td>
                  <td className="tabular-nums" title="Received / sent">
                    {o.inetDownPerGb === null ? '?' : money(o.inetDownPerGb)} / {o.inetUpPerGb === null ? '?' : money(o.inetUpPerGb)}
                  </td>
                  <td className="truncate pr-2" title={o.geolocation}>
                    {o.geolocation || '—'}
                  </td>
                  <td className="tabular-nums">{o.maxDays === null ? '—' : Math.floor(o.maxDays)}</td>
                  <td className="py-1 text-right">
                    <Button
                      size="sm"
                      onClick={() => {
                        if (!found) return
                        setSetupModel(found.forModel)
                        setPicked({ offer: o, filter: found.filter })
                      }}
                    >
                      Choose
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {blocked && <p className="mt-2 text-xs text-warn">{blocked}</p>}
      </div>
    </Dialog>
  )
}

function remoteModelSizeOf(state: RemoteState, id: string, library: Parameters<typeof remoteModelSize>[1]): number | null {
  const def = state.models.find((m) => m.id === id)
  return def ? remoteModelSize(def, library) : null
}

function SortTh({
  k,
  sort,
  onSort,
  className,
  children
}: {
  k: OfferSort
  sort: { key: OfferSort; dir: SortDir }
  onSort: (k: OfferSort) => void
  className?: string
  children: string
}) {
  const on = sort.key === k
  return (
    <th className={cn('py-1.5 font-medium', className)} aria-sort={on ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'}>
      <button type="button" className={cn('inline-flex items-center gap-0.5 uppercase hover:text-text focus-visible:underline', on && 'text-text')} onClick={() => onSort(k)}>
        {children}
        {on && (sort.dir === 'asc' ? <ArrowUp size={11} /> : <ArrowDown size={11} />)}
      </button>
    </th>
  )
}

function Warn({ children }: { children: string }) {
  return (
    <div className="flex items-start gap-2 rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
      <CircleAlert size={14} className="mt-px shrink-0" />
      <div>{children}</div>
    </div>
  )
}
