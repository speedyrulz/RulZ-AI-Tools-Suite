import { Crosshair, EyeOff, ScanSearch } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'
import type { CheckFinding } from '@shared/novel'
import type { WritingJob } from '@shared/phone'
import { findingKey, locateQuote } from '@shared/novelChecks'
import { findScene } from '@shared/novelOutline'
import { isRunning } from '../lib/jobs'
import { startCheck } from '../stores/actions'
import { useScene } from '../stores/editor'
import { useJobs, useProjectJobs } from '../stores/jobs'
import { useProject } from '../stores/project'
import { useUi } from '../stores/ui'
import { JobShell } from './JobCards'
import { Badge, Button, Spinner } from './ui'

const TYPE_LABEL: Record<CheckFinding['type'], string> = {
  continuity: 'Continuity',
  character: 'Character',
  timeline: 'Timeline',
  pov: 'Point of view',
  tense: 'Tense',
  other: 'Other'
}

/** The consistency check of the scene being written: findings to tap, show in the text, or dismiss. */
export function CheckTab() {
  const state = useProject((s) => s.state)
  const sceneId = useUi((s) => s.sceneId)
  const jobs = useProjectJobs(state?.folder).filter((j) => j.request.action === 'checkScene')
  const [busy, setBusy] = useState(false)
  if (!state) return null
  const found = sceneId ? findScene(state.outline, sceneId) : null
  if (!found) return <p className="px-6 py-10 text-center text-[14px] text-faint">Choose a scene on the Write page first.</p>
  const running = jobs.some(isRunning)

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-lg space-y-4 px-4 py-4">
        <p className="text-[13.5px] text-muted">Checks “{found.scene.title}” against your story bible and the earlier story, and lists what doesn’t fit.</p>
        <Button
          variant="primary"
          disabled={running || busy}
          onClick={async () => {
            setBusy(true)
            try {
              await startCheck()
            } finally {
              setBusy(false)
            }
          }}
        >
          {running || busy ? <Spinner /> : <ScanSearch size={16} />} Check this scene
        </Button>
        {jobs.map((job) => (job.status === 'done' && job.result?.kind === 'check' ? <Findings key={job.id} job={job} /> : <JobShell key={job.id} job={job} />))}
      </div>
    </div>
  )
}

function Findings({ job }: { job: WritingJob }) {
  const result = job.result?.kind === 'check' ? job.result : null
  const dismissedKeys = useProject((s) => (result ? (s.state?.dismissed[result.sceneId] ?? []) : []))
  const dismissFinding = useProject((s) => s.dismissFinding)
  const dismiss = useJobs((s) => s.dismiss)
  const setTab = useUi((s) => s.setTab)
  const sceneId = useUi((s) => s.sceneId)
  const [showHidden, setShowHidden] = useState(false)
  if (!result) return null

  const all = result.findings.map((f) => ({ f, key: findingKey(f) }))
  const hidden = all.filter((x) => dismissedKeys.includes(x.key))
  const shown = showHidden ? all : all.filter((x) => !dismissedKeys.includes(x.key))
  const mine = result.sceneId === sceneId

  const show = (f: CheckFinding): void => {
    const b = useScene.getState().buffer
    if (!mine || !b) return void toast.error('Open that scene first.')
    const at = locateQuote(b.text, f.quote)
    if (!at) return void toast.error('I can’t find that passage. The text may have changed since the check.')
    useScene.getState().select(at.from, at.to)
    setTab('write')
  }

  return (
    <JobShell job={job}>
      {all.length === 0 && <p className="text-[14px] text-muted">Nothing found. The scene fits what the story bible and earlier scenes say.</p>}
      {all.length > 0 && shown.length === 0 && <p className="text-[14px] text-muted">All {all.length} findings are dismissed.</p>}
      <ul className="space-y-2.5">
        {shown.map(({ f, key }) => {
          const isHidden = dismissedKeys.includes(key)
          return (
            <li key={key} className="space-y-1.5 rounded-xl border border-border bg-panel-2 p-3 text-[14px]">
              <div className="flex items-center gap-2">
                <Badge tone="warn">{TYPE_LABEL[f.type] ?? f.type}</Badge>
                {isHidden && <Badge>Dismissed</Badge>}
              </div>
              {f.quote && <blockquote className="border-l-2 border-accent pl-2.5 text-body italic">{f.quote}</blockquote>}
              <p>{f.problem}</p>
              {f.conflictsWith && <p className="text-[13px] text-muted">Conflicts with: {f.conflictsWith}</p>}
              {f.suggestion && <p className="text-[13px] text-muted">Suggestion: {f.suggestion}</p>}
              <div className="flex flex-wrap gap-2 pt-1">
                <Button size="sm" onClick={() => show(f)}>
                  <Crosshair size={15} /> Show in the text
                </Button>
                <Button size="sm" variant="ghost" onClick={() => void dismissFinding(result.sceneId, key, !isHidden)}>
                  <EyeOff size={15} /> {isHidden ? 'Bring back' : 'Dismiss'}
                </Button>
              </div>
            </li>
          )
        })}
      </ul>
      <div className="flex flex-wrap items-center justify-between gap-2">
        {hidden.length > 0 ? (
          <button className="text-[13px] text-muted underline" onClick={() => setShowHidden(!showHidden)}>
            {showHidden ? 'Hide dismissed' : `Show ${hidden.length} dismissed`}
          </button>
        ) : (
          <span />
        )}
        <Button size="sm" onClick={() => void dismiss(job.id)}>
          Done
        </Button>
      </div>
    </JobShell>
  )
}
