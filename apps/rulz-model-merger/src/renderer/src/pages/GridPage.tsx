import { LayoutGrid } from 'lucide-react'
import { useEffect } from 'react'
import { GridResultCard, EarlierGridsCard } from '../components/grid/ResultCards'
import { MergeCard, PromptsCard, TemplateCard, ValuesCard } from '../components/grid/SetupCards'
import { Button, PageTitle, Tooltip } from '../components/ui'
import { gridBlockReason, resolveAxisOption } from '../lib/grid'
import { useApp } from '../stores/app'
import { useGrid } from '../stores/grid'
import { useMerge } from '../stores/merge'

export function GridPage() {
  const status = useApp((s) => s.status)
  const recipe = useMerge((s) => s.recipe)
  const template = useGrid((s) => s.template)
  const axisKey = useGrid((s) => s.axisKey)
  const valuesText = useGrid((s) => s.valuesText)
  const prompts = useGrid((s) => s.prompts)
  const seed = useGrid((s) => s.seed)
  const grids = useGrid((s) => s.grids)
  const starting = useGrid((s) => s.starting)
  const start = useGrid((s) => s.start)

  // Earlier grids and the last workflow are read each time the page opens.
  useEffect(() => {
    void useGrid.getState().loadList()
    void useGrid.getState().reloadRemembered()
  }, [])

  const connected = !!status?.reachable && (status.own || status.sharedQueueAccepted)
  const reason = gridBlockReason({ connected, template, recipe, axis: resolveAxisOption(recipe, axisKey), valuesText, prompts, seed, grids, starting })
  const button = (
    <Button variant="primary" size="lg" disabled={reason !== null} onClick={() => void start()}>
      <LayoutGrid size={18} />
      Start grid
    </Button>
  )

  return (
    <div className="space-y-6">
      <PageTitle title="Grids">
        Renders the merge from the Merge page at several values of one setting, so you can compare them side by side. Nothing is saved as a model.
      </PageTitle>
      <TemplateCard />
      <MergeCard />
      <ValuesCard />
      <PromptsCard />
      <div className="flex items-center gap-4">
        {reason ? (
          <Tooltip content={reason}>
            <span tabIndex={0} className="inline-flex">
              {button}
            </span>
          </Tooltip>
        ) : (
          button
        )}
        {reason && <span className="text-[13px] text-muted">{reason}</span>}
      </div>
      <GridResultCard />
      <EarlierGridsCard />
    </div>
  )
}
