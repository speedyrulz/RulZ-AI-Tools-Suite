import { Check, Square, X } from 'lucide-react'
import { useMemo, useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import type { WritingJob } from '@shared/phone'
import { checkAccept, pickAlt } from '../lib/accept'
import { cn } from '../lib/format'
import { isDirty } from '../lib/autosave'
import { isRunning, jobLine } from '../lib/jobs'
import { findScene } from '@shared/novelOutline'
import { proposalOps, resolveSceneRefs, scenePath, stickyOps } from '../lib/outlineOps'
import { onlyAdditions, trimContext, wordDiff } from '../lib/wordDiff'
import { useScene } from '../stores/editor'
import { useJobs, useProjectJobs } from '../stores/jobs'
import { useProject } from '../stores/project'
import { useUi } from '../stores/ui'
import { Badge, Button, Sheet, Spinner, Textarea } from './ui'

/** Actions whose results are handled in the Writing sheet (state, checks and interviews have their own pages). */
const SHEET_ACTIONS = ['continue', 'rewrite', 'expand', 'condense', 'dialogue', 'draftScene', 'summarizeScene', 'summarizeChapter', 'outlineChapters', 'outlineScenes']

export function inSheet(job: WritingJob): boolean {
  return SHEET_ACTIONS.includes(job.request.action)
}

/** The frame of a job card: its title line with Stop while it runs, and what to show while it runs or failed. */
export function JobShell({ job, children, footer }: { job: WritingJob; children?: ReactNode; footer?: ReactNode }) {
  const abort = useJobs((s) => s.abort)
  const dismiss = useJobs((s) => s.dismiss)
  const running = isRunning(job)
  return (
    <section className="space-y-3 rounded-2xl border border-border bg-panel p-3.5">
      <div className="flex items-center gap-2">
        {running && <Spinner className="text-accent" />}
        <div className="min-w-0 flex-1 text-[14px] font-medium">{jobLine(job)}</div>
        {running ? (
          <Button size="sm" onClick={() => void abort(job.id)}>
            <Square size={13} /> Stop
          </Button>
        ) : (
          job.status !== 'done' && (
            <Button size="sm" variant="ghost" onClick={() => void dismiss(job.id)}>
              <X size={15} /> Close
            </Button>
          )
        )}
      </div>
      {running && job.progress !== null && job.status === 'waiting' && (
        <div className="h-1.5 overflow-hidden rounded-full bg-track">
          <div className="h-full bg-accent transition-[width]" style={{ width: `${Math.round(job.progress * 100)}%` }} />
        </div>
      )}
      {running && job.partial && <div className="max-h-48 overflow-y-auto rounded-lg bg-panel-2 p-2.5 text-[14px] leading-relaxed whitespace-pre-wrap text-body">{job.partial}</div>}
      {job.status === 'error' && <p className="text-[14px] text-danger">{job.error ?? 'Something went wrong on the computer.'}</p>}
      {job.status === 'stopped' && <p className="text-[14px] text-muted">Stopped.</p>}
      {job.warnings.length > 0 && !running && (
        <ul className="space-y-1 text-[12.5px] text-warn">
          {job.warnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}
      {children}
      {footer}
    </section>
  )
}

function DiffView({ before, after }: { before: string; after: string }) {
  const parts = useMemo(() => wordDiff(before, after), [before, after])
  const added = onlyAdditions(parts)
  const shown = useMemo(() => trimContext(parts), [parts])
  return (
    <div className="max-h-72 overflow-y-auto rounded-lg bg-panel-2 p-2.5 text-[14px] leading-relaxed whitespace-pre-wrap text-body">
      {shown.map((p, i) =>
        p.kind === 'same' ? (
          <span key={i} className={cn(added && 'text-muted')}>
            {p.text}
          </span>
        ) : p.kind === 'add' ? (
          <span key={i} className="rounded-sm bg-ok/20 text-text">
            {p.text}
          </span>
        ) : (
          <span key={i} className="rounded-sm bg-danger/15 text-danger-text line-through decoration-1">
            {p.text}
          </span>
        )
      )}
    </div>
  )
}

/** A finished prose result: switch between the alternatives, see what changes, accept or reject. */
function ProseCard({ job }: { job: WritingJob }) {
  const result = job.result?.kind === 'prose' ? job.result : null
  const [index, setIndex] = useState(0)
  const [view, setView] = useState<'changes' | 'text'>('changes')
  const [busy, setBusy] = useState(false)
  const [problem, setProblem] = useState('')
  const memo = useJobs((s) => s.memos[job.id])
  const dismiss = useJobs((s) => s.dismiss)
  const sceneId = useUi((s) => s.sceneId)
  const setScene = useUi((s) => s.setScene)
  const buffer = useScene((s) => s.buffer)
  if (!result) return <JobShell job={job} />
  const alt = pickAlt(job, Math.min(index, result.alts.length - 1))
  const other = job.request.sceneId !== null && job.request.sceneId !== sceneId

  const accept = async (): Promise<void> => {
    if (!alt || !buffer) return
    const project = useProject.getState().state
    const found = project && job.request.sceneId ? findScene(project.outline, job.request.sceneId) : null
    const check = checkAccept({
      bufferFolder: buffer.folder,
      bufferPath: buffer.path,
      expectedFolder: job.folder,
      expectedPath: found ? scenePath(found.scene) : null,
      bufferText: buffer.text,
      bufferDirty: isDirty(buffer),
      bufferBaseHash: buffer.baseHash,
      job,
      memo: memo ? { text: memo.text, baseHash: memo.baseHash } : null
    })
    if (!check.ok) return setProblem(check.reason)
    setBusy(true)
    setProblem('')
    try {
      // The buffer is the text the result was written for, so it is saved on the base it already has.
      const saved = await useScene.getState().applyText(alt.text)
      if (!saved) return setProblem('It wasn’t saved. Look at the message above the text.')
      if (project && result.newSticky.length && job.request.sceneId) {
        const ops = stickyOps(project.outline, job.request.sceneId, result.newSticky)
        if (ops) await useProject.getState().patchOutline(ops)
      }
      await dismiss(job.id)
      toast.success('Added to the scene.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <JobShell job={job}>
      {result.alts.length > 1 && (
        <div className="flex gap-1.5" role="tablist" aria-label="Alternatives">
          {result.alts.map((_, i) => (
            <button
              key={i}
              role="tab"
              aria-selected={i === index}
              className={cn('h-9 min-w-9 rounded-lg border px-3 text-[14px]', i === index ? 'border-accent bg-accent-soft text-text' : 'border-border text-muted active:bg-hover')}
              onClick={() => setIndex(i)}
            >
              {i + 1}
            </button>
          ))}
        </div>
      )}
      {alt && (
        <>
          <div className="flex gap-1.5 text-[13px]">
            <button className={cn('rounded-md px-2.5 py-1', view === 'changes' ? 'bg-hover text-text' : 'text-muted')} onClick={() => setView('changes')}>
              Changes
            </button>
            <button className={cn('rounded-md px-2.5 py-1', view === 'text' ? 'bg-hover text-text' : 'text-muted')} onClick={() => setView('text')}>
              New text
            </button>
          </div>
          {view === 'text' || !memo ? (
            <div className="max-h-72 overflow-y-auto rounded-lg bg-panel-2 p-2.5 text-[14px] leading-relaxed whitespace-pre-wrap text-body">{alt.generated}</div>
          ) : (
            <DiffView before={memo.text} after={alt.text} />
          )}
        </>
      )}
      {result.newSticky.length > 0 && <p className="text-[12.5px] text-muted">Keeps {result.newSticky.length} more story bible {result.newSticky.length === 1 ? 'entry' : 'entries'} in view for this scene.</p>}
      {problem && <p className="text-[13.5px] text-danger">{problem}</p>}
      {other && <p className="text-[13.5px] text-warn">This was written for another scene.</p>}
      <div className="flex flex-wrap justify-end gap-2">
        <Button variant="ghost" disabled={busy} onClick={() => void dismiss(job.id)}>
          Reject
        </Button>
        {other ? (
          <Button variant="primary" onClick={() => job.request.sceneId && setScene(job.request.sceneId)}>
            Open that scene
          </Button>
        ) : (
          <Button variant="primary" disabled={busy || !buffer} onClick={() => void accept()}>
            {busy ? <Spinner /> : <Check size={16} />} Accept
          </Button>
        )}
      </div>
    </JobShell>
  )
}

/** A finished summary or outline proposal. */
function ProposalCard({ job }: { job: WritingJob }) {
  const result = job.result
  const dismiss = useJobs((s) => s.dismiss)
  const state = useProject((s) => s.state)
  const [busy, setBusy] = useState(false)
  const [summary, setSummary] = useState(result?.kind === 'summary' ? result.reply.summary : '')
  if (!result || !state || (result.kind !== 'summary' && result.kind !== 'chapters' && result.kind !== 'scenes')) return <JobShell job={job} />

  const accept = async (): Promise<void> => {
    setBusy(true)
    try {
      const edited = result.kind === 'summary' ? { ...result, reply: { summary } } : result
      const ops = proposalOps(edited, state.bible)
      if (!ops.length) return void (await dismiss(job.id))
      if (await useProject.getState().patchOutline(ops)) {
        await dismiss(job.id)
        toast.success('Added to the outline.')
      }
    } finally {
      setBusy(false)
    }
  }

  const name = (id: string): string => state.bible.find((e) => e.id === id)?.name ?? id
  const refs = result.kind === 'scenes' ? resolveSceneRefs(state.bible, result.reply.scenes).scenes : []
  return (
    <JobShell job={job}>
      {result.kind === 'summary' && <Textarea rows={6} aria-label="Summary" value={summary} onChange={(e) => setSummary(e.target.value)} />}
      {result.kind === 'chapters' && (
        <ol className="space-y-2">
          {result.reply.chapters.map((c, i) => (
            <li key={i} className="rounded-lg border border-border bg-panel-2 p-2.5 text-[14px]">
              <div className="font-medium">
                {i + 1}. {c.title} {c.act && <span className="text-[12px] font-normal text-faint">{c.act}</span>}
              </div>
              <div className="mt-0.5 text-[13px] text-muted">{c.summary}</div>
            </li>
          ))}
        </ol>
      )}
      {result.kind === 'scenes' && (
        <ol className="space-y-2">
          {result.reply.scenes.map((sc, i) => (
            <li key={i} className="rounded-lg border border-border bg-panel-2 p-2.5 text-[14px]">
              <div className="font-medium">
                {i + 1}. {sc.title}
              </div>
              <div className="mt-0.5 text-[13px] whitespace-pre-wrap text-muted">{sc.beats}</div>
              {(refs[i]?.pov || refs[i]?.present.length > 0) && (
                <div className="mt-1 text-[12px] text-faint">
                  {refs[i].pov && <>POV: {name(refs[i].pov!)}. </>}
                  {refs[i].present.length > 0 && <>Present: {refs[i].present.map(name).join(', ')}.</>}
                </div>
              )}
            </li>
          ))}
        </ol>
      )}
      {(result.kind === 'chapters' || result.kind === 'scenes') && result.unresolved.length > 0 && (
        <p className="text-[12.5px] text-warn">Not in the story bible: {result.unresolved.join(', ')}. They are left out.</p>
      )}
      <div className="flex flex-wrap justify-end gap-2">
        <Button variant="ghost" disabled={busy} onClick={() => void dismiss(job.id)}>
          Reject
        </Button>
        <Button variant="primary" disabled={busy || (result.kind === 'summary' && !summary.trim())} onClick={() => void accept()}>
          {busy ? <Spinner /> : <Check size={16} />} Accept
        </Button>
      </div>
    </JobShell>
  )
}

/** Any job of the Writing sheet: running, failed or finished. */
export function JobCard({ job }: { job: WritingJob }) {
  if (job.status === 'done' && job.result?.kind === 'prose') return <ProseCard key={job.id} job={job} />
  if (job.status === 'done' && (job.result?.kind === 'summary' || job.result?.kind === 'chapters' || job.result?.kind === 'scenes')) return <ProposalCard key={job.id} job={job} />
  return <JobShell job={job} />
}

/** The results of the Writing actions, in a sheet that opens by itself when something is started or ready. */
export function JobsSheet() {
  const open = useUi((s) => s.jobsOpen)
  const setOpen = useUi((s) => s.setJobsOpen)
  const folder = useProject((s) => s.state?.folder)
  const jobs = useProjectJobs(folder).filter(inSheet)
  return (
    <Sheet open={open} onOpenChange={setOpen} title="Writing" description="Results from your computer. Nothing changes in your story until you accept it.">
      <div className="space-y-3 px-4 pb-4">
        {jobs.length === 0 && <p className="py-6 text-center text-[14px] text-faint">Nothing here. Start an action from the scene.</p>}
        {jobs.map((job) => (
          <JobCard key={job.id} job={job} />
        ))}
      </div>
    </Sheet>
  )
}

/** A small badge on the editor: results waiting, or a job running. */
export function JobsPill() {
  const folder = useProject((s) => s.state?.folder)
  const jobs = useProjectJobs(folder).filter(inSheet)
  const open = useUi((s) => s.jobsOpen)
  const setOpen = useUi((s) => s.setJobsOpen)
  if (open || jobs.length === 0) return null
  const running = jobs.some(isRunning)
  return (
    <button className="flex w-full items-center gap-2 border-t border-border bg-panel px-4 py-2.5 text-left text-[14px] active:bg-hover" onClick={() => setOpen(true)}>
      {running ? <Spinner className="text-accent" /> : <Badge tone="accent">{jobs.length}</Badge>}
      <span className="min-w-0 flex-1 truncate">{running ? jobLine(jobs.find(isRunning)!) : jobs.length === 1 ? 'A result is ready' : `${jobs.length} results are ready`}</span>
      <span className="text-[13px] text-link">Show</span>
    </button>
  )
}
