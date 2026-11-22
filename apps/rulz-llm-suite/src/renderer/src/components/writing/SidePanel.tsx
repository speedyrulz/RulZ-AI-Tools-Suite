import { cn } from '../../lib/format'
import { useNovel, type PanelTab } from '../../stores/novel'
import { ChecksPanel, useVisibleFindings } from './ChecksPanel'
import { ContextPanel } from './ContextPanel'
import { InterviewPanel } from './InterviewPanel'
import { ProjectPanel } from './ProjectPanel'
import { ProposalCard } from './ProposalCard'
import { ScenePanel } from './ScenePanel'
import { StateCard } from './StateCard'
import { StatePanel, useStateProblemCount } from './StatePanel'

const TABS: { id: PanelTab; label: string }[] = [
  { id: 'scene', label: 'Scene' },
  { id: 'context', label: 'Context' },
  { id: 'checks', label: 'Checks' },
  { id: 'state', label: 'State' },
  { id: 'interview', label: 'Interview' },
  { id: 'project', label: 'Project' }
]

export function SidePanel() {
  const tab = useNovel((s) => s.panelTab)
  const setTab = useNovel((s) => s.setPanelTab)
  const sceneId = useNovel((s) => (s.target?.kind === 'scene' ? s.target.id : null))
  const checks = useVisibleFindings(sceneId).visible.length
  const stateProblems = useStateProblemCount()
  const labelOf = (id: PanelTab, label: string): string => (id === 'checks' && checks > 0 ? `${label} (${checks})` : id === 'state' && stateProblems > 0 ? `${label} (${stateProblems})` : label)
  return (
    <aside className="flex w-80 shrink-0 flex-col border-l border-border bg-panel">
      <div className="grid grid-cols-3 gap-1 border-b border-border p-2" role="tablist" aria-label="Side panel">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            id={`wtab-${t.id}`}
            aria-selected={tab === t.id}
            aria-controls="wpanel"
            onClick={() => setTab(t.id)}
            className={cn('h-8 rounded-md text-[13px] font-medium', tab === t.id ? 'bg-hover text-text' : 'text-muted hover:bg-hover hover:text-text')}
          >
            {labelOf(t.id, t.label)}
          </button>
        ))}
      </div>
      <div id="wpanel" role="tabpanel" aria-labelledby={`wtab-${tab}`} className="min-h-0 flex-1 overflow-y-auto">
        <ProposalCard />
        <StateCard />
        {tab === 'scene' && <ScenePanel />}
        {tab === 'context' && <ContextPanel />}
        {tab === 'checks' && <ChecksPanel />}
        {tab === 'state' && <StatePanel />}
        {tab === 'interview' && <InterviewPanel />}
        {tab === 'project' && <ProjectPanel />}
      </div>
    </aside>
  )
}
