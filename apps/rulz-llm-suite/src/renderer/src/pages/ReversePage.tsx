import { Download, FileSearch, FolderOpen, GraduationCap, LoaderCircle, MessageSquare, Play, RefreshCw, Trash2 } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import { RE_DEFAULT_MODEL_ID, RE_GOALS } from '@shared/re'
import type { ReOutcome, RePracticeInfo, ReSettings, ReTraceInfo } from '@shared/types'
import { OutcomeButtons } from '../components/chat/ReBanner'
import { Badge, Button, Field, IconButton, Input, NumberInput, Select, Switch, Textarea, Tooltip } from '../components/ui'
import { errorMessage, invoke } from '../lib/api'
import { cn, formatBytes, relativeTime } from '../lib/format'
import { useApp } from '../stores/app'
import { isChatBusy, useChat } from '../stores/chat'
import { useMcp } from '../stores/mcp'
import { useModels } from '../stores/models'

function Card({ title, icon, children, action }: { title: string; icon: ReactNode; children: ReactNode; action?: ReactNode }) {
  return (
    <section className="rounded-xl border border-border bg-panel">
      <div className="flex items-center justify-between gap-2 border-b border-border px-4 py-2.5">
        <h2 className="flex items-center gap-1.5 text-[11px] font-semibold tracking-wide text-faint uppercase">
          {icon}
          {title}
        </h2>
        {action}
      </div>
      <div className="space-y-3 p-4">{children}</div>
    </section>
  )
}

const baseName = (p: string): string => p.split(/[\\/]/).filter(Boolean).at(-1) ?? p

export function ReversePage() {
  const settings = useApp((s) => s.settings)
  const update = useApp((s) => s.updateSettings)
  if (!settings) return null
  const re = settings.re
  const setRe = <K extends keyof ReSettings>(key: K, value: ReSettings[K]): void => void update({ re: { ...re, [key]: value } })

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl space-y-4 px-6 py-5">
        <div>
          <h1 className="text-base font-semibold">Reverse Engineering</h1>
          <div className="text-xs text-muted">
            Analyse programs with a local model and the REA tools. Sessions are recorded as they run; grade them, then export the
            good ones as fine-tuning data so the model gets better at this.
          </div>
        </div>
        <ToolStatus />
        <NewSession re={re} />
        <Practice re={re} setRe={setRe} />
        <Sessions />
        <Card title="Options" icon={null}>
          <Field inline label="Record new sessions" hint="Saves exactly what the model sees and writes in each round, for training.">
            <Switch checked={re.collect} onCheckedChange={(v) => setRe('collect', v)} />
          </Field>
          <Field
            inline
            label="Run static-analysis tools without asking"
            hint="REA tools that only read the target (decompile, search, list, annotate), plus calc and the file tools. Tools that run programs, attach debuggers or reach devices still ask."
          >
            <Switch checked={re.autoApproveStatic} onCheckedChange={(v) => setRe('autoApproveStatic', v)} />
          </Field>
          <Field inline label="Most tool rounds per reply" hint="An RE answer often takes 20–60 tool calls.">
            <div className="w-24">
              <NumberInput value={re.maxToolRounds} min={1} max={500} onChange={(v) => setRe('maxToolRounds', v ?? 80)} />
            </div>
          </Field>
          <Field inline label="Tool description budget (tokens)" hint="How many tool descriptions stay loaded at once in on-demand mode.">
            <div className="w-24">
              <NumberInput value={re.toolBudgetTokens} min={500} max={200000} onChange={(v) => setRe('toolBudgetTokens', v ?? 16000)} />
            </div>
          </Field>
        </Card>
      </div>
    </div>
  )
}

function ToolStatus() {
  const servers = useMcp((s) => s.servers)
  const setPage = useApp((s) => s.setPage)
  const [calc, setCalc] = useState<{ available: boolean; python: string | null } | null>(null)
  useEffect(() => {
    void invoke('re:calcStatus').then(setCalc).catch(() => undefined)
  }, [])
  const rea = servers.find((s) => s.name === 'rea')
  const calcServer = servers.find((s) => s.name === 'calc')
  const chip = (label: string, ok: boolean, detail: string) => (
    <Tooltip content={detail}>
      <span className={cn('flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11.5px]', ok ? 'border-ok/30 text-ok' : 'border-warn/40 text-warn')}>
        <span className={cn('size-1.5 rounded-full', ok ? 'bg-ok' : 'bg-warn')} />
        {label}
      </span>
    </Tooltip>
  )
  return (
    <div className="flex flex-wrap items-center gap-2">
      {chip(
        rea ? `REA: ${rea.state === 'connected' ? `${rea.tools.length} tools` : rea.state}` : 'REA: not configured',
        rea?.state === 'connected',
        rea
          ? rea.error ?? 'The reverse-engineering MCP server from mcp.json.'
          : 'Add a server named "rea" to mcp.json on the MCP Tools page (command "npx", args ["-y", "rea-agents@<version>", "mcp"]).'
      )}
      {chip(
        calcServer ? `calc: ${calcServer.state}` : 'calc: needs Python 3',
        calcServer?.state === 'connected',
        calc?.available
          ? `Sandboxed calculator for hash rounds, XOR decoding and bit math (${calc.python}).`
          : 'The calculator runs on Python 3, which was not found on PATH. Without it the model has to do 32-bit math in its head.'
      )}
      <Button size="sm" variant="ghost" onClick={() => setPage('mcp')}>
        MCP Tools
      </Button>
    </div>
  )
}

function useModelChoice(): [string, (id: string) => void, { id: string; name: string }[]] {
  const models = useModels((s) => s.models)
  const reModel = useApp((s) => s.settings?.re.modelId ?? RE_DEFAULT_MODEL_ID)
  const update = useApp((s) => s.updateSettings)
  const re = useApp((s) => s.settings?.re)
  const list = models.filter((m) => !m.error).map((m) => ({ id: m.id, name: m.id }))
  const choose = (id: string): void => {
    if (re) void update({ re: { ...re, modelId: id } })
  }
  return [reModel, choose, list]
}

function ModelSelect() {
  const [modelId, choose, list] = useModelChoice()
  const missing = !!modelId && !list.some((m) => m.id === modelId)
  return (
    <Field label="Model" hint={missing ? 'Not in your models folder; the loaded model is used.' : 'Loaded on start if needed.'}>
      <Select value={missing ? '' : modelId} onChange={(e) => choose(e.target.value)}>
        <option value="">The loaded model</option>
        {list.map((m) => (
          <option key={m.id} value={m.id}>
            {m.name}
          </option>
        ))}
      </Select>
    </Field>
  )
}

function NewSession({ re }: { re: ReSettings }) {
  const startRe = useChat((s) => s.startRe)
  const busy = useChat((s) => isChatBusy(s))
  const [target, setTarget] = useState('')
  const [goal, setGoal] = useState(RE_GOALS[0].text)
  const [collect, setCollect] = useState(re.collect)
  useEffect(() => setCollect(re.collect), [re.collect])

  const pickFile = async (): Promise<void> => {
    const p = await invoke('dialog:pickFile', {
      title: 'Choose the program or file to analyse',
      filters: [
        { name: 'Programs and libraries', extensions: ['exe', 'dll', 'sys', 'so', 'dylib', 'elf', 'bin', 'apk', 'jar', 'dex', 'wasm', 'node'] },
        { name: 'All files', extensions: ['*'] }
      ]
    })
    if (p) setTarget(p)
  }
  const pickFolder = async (): Promise<void> => {
    const p = await invoke('dialog:pickFolder', { title: 'Choose the app folder to analyse' })
    if (p) setTarget(p)
  }

  return (
    <Card title="New session" icon={<FileSearch size={12} />}>
      <Field label="Target">
        <div
          className="flex gap-2"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            const f = e.dataTransfer.files[0]
            if (f) setTarget(window.rulz.pathForFile(f))
          }}
        >
          <Input
            className="font-mono text-xs"
            value={target}
            placeholder="C:\path\to\program.exe (or drop a file here)"
            onChange={(e) => setTarget(e.target.value)}
          />
          <Button onClick={() => void pickFile()}>File…</Button>
          <Button onClick={() => void pickFolder()}>Folder…</Button>
        </div>
      </Field>
      <Field label="What to find out" hint="{target} is replaced by the path.">
        <div className="mb-1.5 flex flex-wrap gap-1">
          {RE_GOALS.map((g) => (
            <button
              key={g.label}
              onClick={() => setGoal(g.text)}
              className={cn(
                'rounded-md border px-2 py-0.5 text-[11.5px] transition-colors',
                goal === g.text ? 'border-accent/50 bg-accent-soft text-accent' : 'border-border text-muted hover:bg-hover hover:text-text'
              )}
            >
              {g.label}
            </button>
          ))}
        </div>
        <Textarea rows={3} value={goal} onChange={(e) => setGoal(e.target.value)} />
      </Field>
      <div className="grid grid-cols-[1fr_auto] items-end gap-4">
        <ModelSelect />
        <label className="flex h-8.5 items-center gap-2 text-[13px]">
          <Switch checked={collect} onCheckedChange={setCollect} />
          Record for training
        </label>
      </div>
      <div className="flex justify-end">
        <Button
          variant="primary"
          disabled={!target.trim() || !goal.trim() || busy}
          onClick={() => void startRe({ target: target.trim(), goal, collect, modelId: re.modelId || null })}
        >
          <Play size={14} /> Start session
        </Button>
      </div>
    </Card>
  )
}

function Practice({ re, setRe }: { re: ReSettings; setRe: <K extends keyof ReSettings>(key: K, value: ReSettings[K]) => void }) {
  const summaries = useChat((s) => s.summaries)
  const startRe = useChat((s) => s.startRe)
  const busy = useChat((s) => isChatBusy(s))
  const [info, setInfo] = useState<RePracticeInfo | null>(null)
  const [family, setFamily] = useState('')
  const [starting, setStarting] = useState(false)
  const [kitDraft, setKitDraft] = useState(re.kitDir)
  useEffect(() => setKitDraft(re.kitDir), [re.kitDir])

  const refresh = useCallback(() => {
    void invoke('re:practiceList', re.kitDir)
      .then(setInfo)
      .catch((e) => setInfo({ kitDir: re.kitDir, tasks: [], error: errorMessage(e) }))
  }, [re.kitDir])
  useEffect(refresh, [refresh])

  // Per task: how many sessions, and whether one solved it.
  const byTask = useMemo(() => {
    const m = new Map<string, { attempts: number; solved: boolean }>()
    for (const s of summaries) {
      if (!s.rePracticeTask) continue
      const e = m.get(s.rePracticeTask) ?? { attempts: 0, solved: false }
      e.attempts++
      if (s.reOutcome === 'solved') e.solved = true
      m.set(s.rePracticeTask, e)
    }
    return m
  }, [summaries])

  const families = useMemo(() => {
    const out = new Map<string, { total: number; tried: number; solved: number }>()
    for (const t of info?.tasks ?? []) {
      const f = out.get(t.family) ?? { total: 0, tried: 0, solved: 0 }
      f.total++
      const r = byTask.get(t.id)
      if (r) f.tried++
      if (r?.solved) f.solved++
      out.set(t.family, f)
    }
    return [...out.entries()].sort((a, b) => a[0].localeCompare(b[0]))
  }, [info, byTask])

  const startNext = async (): Promise<void> => {
    const tasks = (info?.tasks ?? []).filter((t) => !family || t.family === family)
    // Unsolved tasks, fewest attempts first, so each task gets tried before any is retried.
    const next = tasks
      .filter((t) => !byTask.get(t.id)?.solved)
      .sort((a, b) => (byTask.get(a.id)?.attempts ?? 0) - (byTask.get(b.id)?.attempts ?? 0))[0]
    if (!next) return void toast.success('Every practice task in this selection is solved. Generate a new seed in the kit.')
    setStarting(true)
    try {
      const p = await invoke('re:practicePrepare', re.kitDir, next.id)
      await startRe({
        target: p.target,
        goal: p.prompt,
        collect: re.collect,
        modelId: re.modelId || null,
        title: `Practice: ${next.id}`,
        practice: { taskId: next.id, family: p.family, tier: p.tier }
      })
    } catch (e) {
      toast.error(errorMessage(e))
    } finally {
      setStarting(false)
    }
  }

  return (
    <Card
      title="Practice tasks"
      icon={<GraduationCap size={12} />}
      action={<IconButton size="sm" label="Reload tasks" onClick={refresh}><RefreshCw size={13} /></IconButton>}
    >
      <div className="text-xs text-muted">
        Targets from the RE training kit with known answers. Each session copies the task's binary to a fresh folder, and the
        model's final <span className="font-mono">ANSWER:</span> line is checked automatically, so only verified successes need to
        go into training.
      </div>
      <Field label="Kit folder">
        <div className="flex gap-2">
          <Input
            className="font-mono text-xs"
            value={kitDraft}
            onChange={(e) => setKitDraft(e.target.value)}
            onBlur={() => kitDraft.trim() !== re.kitDir && setRe('kitDir', kitDraft.trim())}
            onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
          />
          <Button
            onClick={async () => {
              const p = await invoke('dialog:pickFolder', { title: 'RE training kit folder', defaultPath: re.kitDir })
              if (p) setRe('kitDir', p)
            }}
          >
            <FolderOpen size={14} />
          </Button>
        </div>
      </Field>
      {info?.error ? (
        <div className="rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">{info.error}</div>
      ) : (
        <>
          <div className="grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-2">
            {families.map(([f, c]) => (
              <button
                key={f}
                onClick={() => setFamily(family === f ? '' : f)}
                className={cn(
                  'rounded-lg border px-3 py-2 text-left transition-colors',
                  family === f ? 'border-accent/60 bg-accent-soft' : 'border-border hover:bg-hover'
                )}
              >
                <div className="text-[13px] font-medium">{f}</div>
                <div className="text-[11px] text-faint tabular-nums">
                  {c.solved}/{c.total} solved · {c.tried} tried
                </div>
              </button>
            ))}
          </div>
          <div className="flex items-center justify-between gap-2">
            <div className="text-xs text-faint">
              {info ? `${info.tasks.length} tasks` : 'Reading tasks…'}
              {family && ` · only ${family}`}
            </div>
            <Button variant="primary" disabled={!info?.tasks.length || busy || starting} onClick={() => void startNext()}>
              {starting ? <LoaderCircle size={14} className="animate-spin" /> : <Play size={14} />} Start next task
            </Button>
          </div>
        </>
      )}
    </Card>
  )
}

const OUTCOME_LABEL: Record<ReOutcome, string> = { unrated: 'Not graded', solved: 'Solved', partial: 'Partial', failed: 'Failed' }

function Sessions() {
  const summaries = useChat((s) => s.summaries)
  const open = useChat((s) => s.open)
  const setPage = useApp((s) => s.setPage)
  const [traces, setTraces] = useState<Map<string, ReTraceInfo>>(new Map())
  const [filter, setFilter] = useState<ReOutcome | 'all'>('all')
  const [outcomes, setOutcomes] = useState<ReOutcome[]>(['solved'])
  const [skipTrimmed, setSkipTrimmed] = useState(true)
  const [maxPerTask, setMaxPerTask] = useState(2)
  const [exporting, setExporting] = useState(false)

  const refresh = useCallback(() => {
    void invoke('re:traces')
      .then((list) => setTraces(new Map(list.map((t) => [t.conversationId, t]))))
      .catch(() => undefined)
  }, [])
  // Summaries change as sessions run, so the recorded counts follow them.
  useEffect(refresh, [refresh, summaries])

  const sessions = summaries.filter((s) => s.reOutcome && (filter === 'all' || s.reOutcome === filter))
  const all = summaries.filter((s) => s.reOutcome)
  const count = (o: ReOutcome): number => all.filter((s) => s.reOutcome === o).length
  const exportable = all.filter((s) => outcomes.includes(s.reOutcome!) && traces.has(s.id)).length

  const doExport = async (): Promise<void> => {
    const dir = await invoke('dialog:pickFolder', { title: 'Folder for train.jsonl' })
    if (!dir) return
    setExporting(true)
    try {
      const r = await invoke('re:export', { outDir: dir, outcomes, skipTrimmed, maxPerTask })
      const dropped = Object.entries(r.dropped)
        .map(([why, n]) => `${n} × ${why}`)
        .join('\n')
      toast.success(`Exported ${r.kept} sample${r.kept === 1 ? '' : 's'} to ${r.files[0]}`, {
        description: dropped ? `Left out:\n${dropped}` : undefined,
        duration: 15000,
        action: { label: 'Show', onClick: () => void invoke('shell:showItem', r.files[0]) }
      })
    } catch (e) {
      toast.error(`Export failed: ${errorMessage(e)}`)
    } finally {
      setExporting(false)
    }
  }

  return (
    <Card
      title="Sessions and training data"
      icon={<MessageSquare size={12} />}
      action={
        <Button size="sm" variant="ghost" onClick={async () => void invoke('shell:openPath', await invoke('re:dataFolder'))}>
          <FolderOpen size={13} /> Data folder
        </Button>
      }
    >
      <div className="flex flex-wrap items-center gap-1.5 text-[11.5px]">
        {(['all', 'unrated', 'solved', 'partial', 'failed'] as const).map((o) => (
          <button
            key={o}
            onClick={() => setFilter(o)}
            className={cn('rounded-md border px-2 py-0.5', filter === o ? 'border-accent/50 bg-accent-soft text-accent' : 'border-border text-muted hover:bg-hover')}
          >
            {o === 'all' ? `All (${all.length})` : `${OUTCOME_LABEL[o]} (${count(o)})`}
          </button>
        ))}
      </div>
      {sessions.length === 0 ? (
        <div className="py-4 text-center text-xs text-faint">No sessions here yet.</div>
      ) : (
        <div className="max-h-96 overflow-y-auto rounded-lg border border-border">
          <table className="w-full text-[12px]">
            <thead className="sticky top-0 bg-panel-2 text-[10.5px] tracking-wide text-faint uppercase">
              <tr>
                <th className="px-3 py-1.5 text-left font-medium">Session</th>
                <th className="px-2 py-1.5 text-right font-medium">Recorded</th>
                <th className="px-2 py-1.5 text-left font-medium">Grade</th>
                <th className="w-16" />
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => {
                const t = traces.get(s.id)
                return (
                  <tr key={s.id} className="border-t border-border/60 hover:bg-hover/40">
                    <td className="max-w-0 px-3 py-1.5">
                      <div className="truncate">{s.title}</div>
                      <div className="truncate font-mono text-[10.5px] text-faint" title={s.reTarget}>
                        {s.reTarget ? baseName(s.reTarget) : ''} · {relativeTime(s.updatedAt)}
                      </div>
                    </td>
                    <td className="px-2 py-1.5 text-right whitespace-nowrap text-muted tabular-nums">
                      {t ? (
                        <Tooltip content={`${t.responses} model replies, ${t.toolRounds} with tool calls, ${formatBytes(t.bytes)}${t.trimmed ? '; context was trimmed' : ''}`}>
                          <span>
                            {t.toolRounds} rounds {t.trimmed && <Badge tone="warn">trimmed</Badge>}
                          </span>
                        </Tooltip>
                      ) : (
                        <span className="text-faint">—</span>
                      )}
                    </td>
                    <td className="px-2 py-1.5">
                      <OutcomeButtons id={s.id} outcome={s.reOutcome!} size="sm" />
                    </td>
                    <td className="px-2 py-1.5">
                      <div className="flex justify-end gap-0.5">
                        <IconButton
                          size="sm"
                          label="Open in chat"
                          onClick={async () => {
                            await open(s.id)
                            setPage('chat')
                          }}
                        >
                          <MessageSquare size={13} />
                        </IconButton>
                        <IconButton
                          size="sm"
                          label="Delete recorded data (keeps the chat)"
                          disabled={!t}
                          onClick={async () => {
                            await invoke('re:deleteTrace', s.id).catch((e) => toast.error(errorMessage(e)))
                            refresh()
                          }}
                        >
                          <Trash2 size={13} />
                        </IconButton>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      <div className="rounded-lg border border-border bg-panel-2/50 p-3">
        <div className="mb-2 text-[13px] font-medium">Export for fine-tuning</div>
        <div className="mb-2 text-[11.5px] text-muted">
          Writes <span className="font-mono">train.jsonl</span> in the RE training kit's format (OpenAI / Hugging Face chat:
          messages, tools, meta; thinking in <span className="font-mono">reasoning_content</span>). Each sample is the exact context
          the model worked in. Training on its own verified successes only is what makes it better.
        </div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-[12.5px]">
          <span className="flex items-center gap-2">
            Include
            {(['solved', 'partial'] as const).map((o) => (
              <label key={o} className="flex items-center gap-1">
                <input
                  type="checkbox"
                  checked={outcomes.includes(o)}
                  onChange={(e) => setOutcomes(e.target.checked ? [...outcomes, o] : outcomes.filter((x) => x !== o))}
                />
                {OUTCOME_LABEL[o]}
              </label>
            ))}
          </span>
          <label className="flex items-center gap-1.5">
            <Switch checked={skipTrimmed} onCheckedChange={setSkipTrimmed} /> Leave out trimmed sessions
          </label>
          <span className="flex items-center gap-1.5">
            Per practice task, at most
            <span className="w-16">
              <NumberInput value={maxPerTask} min={0} max={100} onChange={(v) => setMaxPerTask(v ?? 0)} />
            </span>
          </span>
          <div className="flex-1" />
          <Button variant="primary" disabled={!outcomes.length || exporting || !exportable} onClick={() => void doExport()}>
            {exporting ? <LoaderCircle size={14} className="animate-spin" /> : <Download size={14} />} Export {exportable} session
            {exportable === 1 ? '' : 's'}
          </Button>
        </div>
      </div>
    </Card>
  )
}
