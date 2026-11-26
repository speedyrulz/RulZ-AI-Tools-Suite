import { LoaderCircle, Plus, Sparkles, Trash2, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import type { ReGenerateProgress, RePracticeInfo } from '@shared/types'
import { Button, Field, Input, NumberInput, Select } from '../ui'
import { errorMessage, invoke, on } from '../../lib/api'
import { ConfirmDialog } from '../remote/ConfirmDialog'

const SET_KEY = 'rulz.re.practiceSet'

/** The set chosen last time (kept in this browser only; the list falls back to the first set when it is gone). */
export function readSavedSet(): string {
  try {
    return localStorage.getItem(SET_KEY) ?? ''
  } catch {
    return ''
  }
}

export function saveSet(id: string): void {
  try {
    localStorage.setItem(SET_KEY, id)
  } catch {
    // storage blocked: the choice just is not remembered
  }
}

const randomSeed = (): number => Math.floor(Math.random() * 2 ** 31)

/** The set dropdown with New set and Remove set, the generate form and its progress. */
export function PracticeSets({
  info,
  setId,
  onSet,
  onInfo
}: {
  info: RePracticeInfo
  setId: string
  onSet: (id: string) => void
  onInfo: (info: RePracticeInfo) => void
}) {
  const [open, setOpen] = useState(false)
  const [removing, setRemoving] = useState(false)
  const [removeBusy, setRemoveBusy] = useState(false)
  const current = info.sets.find((s) => s.id === setId)

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        {info.sets.length > 0 && (
          <>
            <span className="text-xs text-muted">Set</span>
            <Select className="w-auto min-w-56" value={setId} onChange={(e) => onSet(e.target.value)} aria-label="Task set">
              {info.sets.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label} ({s.count} tasks)
                </option>
              ))}
            </Select>
          </>
        )}
        <Button onClick={() => setOpen((v) => !v)} disabled={open}>
          <Plus size={14} /> New set…
        </Button>
        {current && current.id !== 'original' && (
          <Button variant="ghost" onClick={() => setRemoving(true)}>
            <Trash2 size={14} /> Remove set
          </Button>
        )}
        {current && (
          <span className="text-[11px] text-faint">
            {current.families.length} families
            {current.seed !== undefined && ` · seed ${current.seed}`}
            {current.created && ` · ${new Date(current.created).toLocaleDateString()}`}
          </span>
        )}
      </div>
      {open && (
        <NewSetForm
          onClose={() => setOpen(false)}
          onDone={(next, id) => {
            onInfo(next)
            onSet(id)
            setOpen(false)
          }}
        />
      )}
      <ConfirmDialog
        open={removing}
        title="Remove this set?"
        lines={[
          `“${current?.label ?? ''}” and its ${current?.count ?? 0} tasks go to the Recycle Bin.`,
          'Results already recorded for these tasks stay in the practice scores.'
        ]}
        confirmLabel="Remove set"
        variant="danger"
        busy={removeBusy}
        onCancel={() => setRemoving(false)}
        onConfirm={() => {
          if (!current) return
          setRemoveBusy(true)
          void invoke('re:removeSet', current.id)
            .then((next) => {
              onInfo(next)
              onSet(next.sets[0]?.id ?? '')
              toast.success(`Moved “${current.label}” to the Recycle Bin.`)
            })
            .catch((e) => toast.error(errorMessage(e)))
            .finally(() => {
              setRemoveBusy(false)
              setRemoving(false)
            })
        }}
      />
    </div>
  )
}

function NewSetForm({ onClose, onDone }: { onClose: () => void; onDone: (info: RePracticeInfo, setId: string) => void }) {
  const [label, setLabel] = useState('')
  const [perFamily, setPerFamily] = useState<number | null>(5)
  const [seed, setSeed] = useState<number | null>(randomSeed)
  const [families, setFamilies] = useState<string[] | null>(null)
  const [picked, setPicked] = useState<Set<string>>(new Set())
  const [error, setError] = useState('')
  const [progress, setProgress] = useState<ReGenerateProgress | null>(null)
  const [running, setRunning] = useState(false)

  useEffect(() => {
    let live = true
    void invoke('re:kitFamilies')
      .then((f) => {
        if (!live) return
        setFamilies(f)
        setPicked(new Set(f))
      })
      .catch((e) => live && setError(errorMessage(e)))
    return () => {
      live = false
    }
  }, [])
  useEffect(() => on('re:generateProgress', setProgress), [])

  const valid =
    label.trim().length >= 1 &&
    label.trim().length <= 80 &&
    perFamily !== null &&
    Number.isInteger(perFamily) &&
    perFamily >= 1 &&
    perFamily <= 50 &&
    picked.size > 0 &&
    seed !== null &&
    Number.isInteger(seed) &&
    seed >= 0 &&
    seed <= 2 ** 31 - 1

  const generate = async (): Promise<void> => {
    if (!valid || perFamily === null || seed === null || !families) return
    setError('')
    setProgress(null)
    setRunning(true)
    try {
      const r = await invoke('re:generateSet', { label: label.trim(), perFamily, families: families.filter((f) => picked.has(f)), seed })
      toast.success(`Generated “${label.trim()}”: ${r.info.sets.find((s) => s.id === r.setId)?.count ?? 0} tasks.`)
      onDone(r.info, r.setId)
    } catch (e) {
      setError(errorMessage(e))
    } finally {
      setRunning(false)
    }
  }

  const pct = progress && progress.total > 0 ? Math.min(100, Math.round((progress.done / progress.total) * 100)) : 0

  return (
    <div className="space-y-3 rounded-lg border border-border bg-panel-2/50 p-3">
      <div className="text-xs text-muted">
        Compiles a fresh set of practice programs with the kit's generator, with new answers. It needs Python 3 and Visual Studio Build Tools (the C
        compiler) on this PC.
      </div>
      <div className="grid gap-3 sm:grid-cols-[1fr_auto_auto]">
        <Field label="Label">
          <Input value={label} maxLength={80} disabled={running} placeholder="e.g. Round 2" onChange={(e) => setLabel(e.target.value)} />
        </Field>
        <Field label="Tasks per family">
          <NumberInput value={perFamily} onChange={setPerFamily} min={1} max={50} step={1} disabled={running} />
        </Field>
        <Field label="Seed">
          <div className="flex items-center gap-1.5">
            <NumberInput className="w-32" value={seed} onChange={setSeed} min={0} max={2 ** 31 - 1} step={1} disabled={running} />
            <Button variant="ghost" disabled={running} onClick={() => setSeed(randomSeed())}>
              Random
            </Button>
          </div>
        </Field>
      </div>
      <Field label="Families" hint="Tiers rotate easy / medium / hard within each family.">
        {families ? (
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            {families.map((f) => (
              <label key={f} className="flex items-center gap-1.5 text-[13px]">
                <input
                  type="checkbox"
                  checked={picked.has(f)}
                  disabled={running}
                  onChange={(e) => {
                    const next = new Set(picked)
                    if (e.target.checked) next.add(f)
                    else next.delete(f)
                    setPicked(next)
                  }}
                />
                {f}
              </label>
            ))}
          </div>
        ) : (
          <div className="text-xs text-faint">{error ? 'Not available.' : 'Reading the kit…'}</div>
        )}
      </Field>
      {error && <div className="rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-xs whitespace-pre-wrap text-warn">{error}</div>}
      {running && (
        <div className="space-y-1">
          <div className="h-2 overflow-hidden rounded-full bg-subtle" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
            <div className="h-full bg-accent transition-[width]" style={{ width: `${pct}%` }} />
          </div>
          <div className="text-[11.5px] text-muted tabular-nums">
            {progress && progress.total > 0 ? `${progress.done} / ${progress.total} tasks` : 'Starting the generator…'}
          </div>
        </div>
      )}
      <div className="flex justify-end gap-2">
        {running ? (
          <Button onClick={() => void invoke('re:generateCancel').catch((e) => toast.error(errorMessage(e)))}>
            <X size={14} /> Cancel
          </Button>
        ) : (
          <Button variant="ghost" onClick={onClose}>
            Close
          </Button>
        )}
        <Button variant="primary" disabled={!valid || running || !families} onClick={() => void generate()}>
          {running ? <LoaderCircle size={14} className="animate-spin" /> : <Sparkles size={14} />} Generate
        </Button>
      </div>
    </div>
  )
}
