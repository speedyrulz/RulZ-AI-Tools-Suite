import { cn } from '../../lib/format'
import { useNovel, type PanelTab } from '../../stores/novel'
import { ContextPanel } from './ContextPanel'
import { ProjectPanel } from './ProjectPanel'
import { ProposalCard } from './ProposalCard'
import { ScenePanel } from './ScenePanel'

const TABS: { id: PanelTab; label: string }[] = [
  { id: 'scene', label: 'Scene' },
  { id: 'context', label: 'Context' },
  { id: 'project', label: 'Project' }
]

export function SidePanel() {
  const tab = useNovel((s) => s.panelTab)
  const setTab = useNovel((s) => s.setPanelTab)
  return (
    <aside className="flex w-80 shrink-0 flex-col border-l border-border bg-panel">
      <div className="flex gap-1 border-b border-border p-2" role="tablist" aria-label="Side panel">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            id={`wtab-${t.id}`}
            aria-selected={tab === t.id}
            aria-controls="wpanel"
            onClick={() => setTab(t.id)}
            className={cn('h-8 flex-1 rounded-md text-[13px] font-medium', tab === t.id ? 'bg-hover text-text' : 'text-muted hover:bg-hover hover:text-text')}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div id="wpanel" role="tabpanel" aria-labelledby={`wtab-${tab}`} className="min-h-0 flex-1 overflow-y-auto">
        <ProposalCard />
        {tab === 'scene' && <ScenePanel />}
        {tab === 'context' && <ContextPanel />}
        {tab === 'project' && <ProjectPanel />}
      </div>
    </aside>
  )
}
