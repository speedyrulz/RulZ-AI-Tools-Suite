import { Cloud } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { isRemoteModelId, type BatchHost, type BatchTarget } from '@shared/remote'
import type { LoadConfig } from '@shared/types'
import {
  CONTEXT_PER_SESSION_PARALLEL,
  contextWarning,
  hostChoices,
  hostKey,
  knownReContext,
  maxSessions,
  ramHint,
  targetFor,
  targetProblems,
  type HostChoice
} from '../../lib/remote'
import { invoke } from '../../lib/api'
import { formatTokens } from '../../lib/format'
import { modesFor } from '../../lib/modes'
import { useApp } from '../../stores/app'
import { useMcp } from '../../stores/mcp'
import { useModels } from '../../stores/models'
import { useRemote } from '../../stores/remote'
import { Button, NumberInput, Switch, Tooltip } from '../ui'

export interface TargetState {
  target: BatchTarget
  /** Sentences why this target cannot start; empty = it can. */
  problems: string[]
  choices: HostChoice[]
  /** The fields to show. */
  view: ReactNode
}

const keyToHost = (k: string): BatchHost => (k === 'local' ? { kind: 'local' } : { kind: 'vast', rentalId: Number(k.slice(5)) })

/**
 * Where a batch run goes: the hosts that can run the run's model (a model on vast.ai runs on rentals only, a model
 * from this PC on this PC only), tasks at once on each, the context of each session, and stopping the rentals at the end.
 */
export function useHostChoices(modelId: string | null): HostChoice[] {
  const rentals = useRemote((s) => s.state?.rentals)
  const servers = useMcp((s) => s.servers)
  const reStdio = servers.find((s) => s.name === 'rea')?.transport === 'stdio'
  return useMemo(() => hostChoices(modelId, rentals ?? [], reStdio), [modelId, rentals, reStdio])
}

export function useTargetState(modelId: string | null, initial?: BatchTarget): TargetState {
  const rentals = useRemote((s) => s.state?.rentals)
  const setPage = useApp((s) => s.setPage)
  const choices = useHostChoices(modelId)
  // The context the model is saved to load with: the Reverse engineering load settings, else the model's default ones.
  const reLoad = useModels((s) => modesFor(s.modeDefaults, modelId)?.re?.load)
  const [saved, setSaved] = useState<{ modelId: string; config: LoadConfig; custom: boolean } | null>(null)
  const savedSeq = useRef(0)
  useEffect(() => {
    const seq = ++savedSeq.current
    if (!modelId) return
    invoke('models:getDefaults', modelId)
      .then((d) => seq === savedSeq.current && setSaved({ modelId, config: d.config, custom: d.custom }))
      .catch(() => seq === savedSeq.current && setSaved(null))
  }, [modelId])
  const known = knownReContext(reLoad, saved && saved.modelId === modelId ? saved : null)
  const [sel, setSel] = useState<string[]>(initial ? initial.hosts.map(hostKey) : [])
  const [sessions, setSessions] = useState<number | undefined>(initial?.sessionsPerHost)
  const [ctx, setCtx] = useState<number | null | undefined>(initial ? initial.contextPerSession : undefined)
  const [stopAfter, setStopAfter] = useState(initial?.stopRentalsAfter ?? false)
  const remote = isRemoteModelId(modelId)

  const valid = choices.filter((c) => c.valid)
  const validKeys = valid.map((c) => hostKey(c.host))
  const chosen = sel.filter((k) => validKeys.includes(k))
  // Nothing chosen (or what was chosen is gone): the first usable host is the default.
  const keys = chosen.length ? chosen : validKeys.slice(0, 1)
  const hosts = keys.map(keyToHost)
  const base = targetFor(hosts, { sessionsPerHost: sessions, stopRentalsAfter: stopAfter }, known?.total)
  const target: BatchTarget = { ...base, contextPerSession: ctx === undefined ? base.contextPerSession : ctx }
  const problems = targetProblems(target, choices)
  const ctxWarning = contextWarning(target)
  const most = maxSessions(Math.max(1, hosts.length))
  const hasRental = hosts.some((h) => h.kind === 'vast')
  const whys = [...new Set(choices.filter((c) => !c.valid && c.host.kind === 'vast').map((c) => c.why))]

  const toggle = (k: string, on: boolean): void => setSel(on ? [...new Set([...keys, k])] : keys.filter((x) => x !== k))

  const view = (
    <div className="space-y-2 text-[12.5px]">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
        <span className="text-muted">Where it runs</span>
        {valid.map((c) => {
          const k = hostKey(c.host)
          return (
            <label key={k} className="flex items-center gap-1.5">
              <input type="checkbox" className="size-3.5 accent-[var(--accent)]" checked={keys.includes(k)} onChange={(e) => toggle(k, e.target.checked)} />
              {c.label}
            </label>
          )
        })}
        {valid.length === 0 && <span className="text-faint">No host can run this model yet.</span>}
      </div>
      <div className="text-[11.5px] text-faint">
        {remote
          ? 'This model runs on vast.ai, so the run goes to rentals only. A run has one model; a model from this PC cannot join it.'
          : 'This model is on this PC, so the run goes here. A model on vast.ai runs on rentals only.'}
      </div>
      {remote && !(rentals ?? []).length && (
        <div className="flex items-center gap-2 text-[11.5px] text-muted">
          <Cloud size={13} /> No rented instance yet.
          <Button size="sm" variant="ghost" onClick={() => setPage('vast')}>
            Open the vast.ai page
          </Button>
        </div>
      )}
      {whys.length > 0 && (
        <ul className="space-y-0.5 text-[11.5px] text-muted">
          {whys.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
        <Tooltip content={`Tasks running side by side on each host. At most ${most} for ${hosts.length || 1} host${hosts.length === 1 ? '' : 's'}.`}>
          <span className="flex items-center gap-1.5">
            Tasks at once on each
            <NumberInput className="w-14" value={target.sessionsPerHost} min={1} max={most} onChange={(v) => setSessions(Math.min(most, Math.max(1, Math.round(v ?? 1))))} />
          </span>
        </Tooltip>
        <Tooltip
          content={`The model is loaded with this context for each task at once, so it uses more memory the more tasks run. ${
            known
              ? `Default: the ${formatTokens(known.total)} saved in this model's ${known.from === 're' ? 'Reverse engineering' : 'default'} load settings, shared out over the tasks at once.`
              : `Default ${formatTokens(CONTEXT_PER_SESSION_PARALLEL)} when more than one runs or a rental is used.`
          }`}
        >
          <span className="flex items-center gap-1.5">
            Context per session
            <NumberInput
              className="w-20"
              value={target.contextPerSession}
              placeholder="Model's"
              min={1024}
              step={1024}
              onChange={(v) => setCtx(v === null ? null : Math.max(1024, Math.round(v)))}
            />
            tokens
          </span>
        </Tooltip>
        {hasRental && (
          <Tooltip content="After the run ends, the rentals it used are stopped (with the usual countdown and a Keep running button). Their disks stay and are billed.">
            <label className="flex items-center gap-1.5">
              <Switch checked={target.stopRentalsAfter} onCheckedChange={setStopAfter} /> Stop the rentals when the run ends
            </label>
          </Tooltip>
        )}
      </div>
      {/* REA runs on this PC for every task, also when the model is on a rental. */}
      <div className="text-[11.5px] text-faint tabular-nums">{ramHint(target.sessionsPerHost * Math.max(1, hosts.length))}</div>
      {ctxWarning && <div className="text-[11.5px] text-warn">{ctxWarning}</div>}
      {problems.length > 0 && (
        <ul className="space-y-0.5 text-[11.5px] text-warn">
          {problems.map((p) => (
            <li key={p}>{p}</li>
          ))}
        </ul>
      )}
    </div>
  )

  return { target, problems, choices, view }
}
