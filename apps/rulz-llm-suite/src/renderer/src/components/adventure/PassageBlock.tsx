import { ChevronLeft, ChevronRight, CornerUpLeft, Pencil, RotateCcw, Shuffle, Trash2 } from 'lucide-react'
import { memo } from 'react'
import { newestLeaf, type Adventure, type AdventureNode } from '@shared/adventure'
import { siblingIndex } from '../../lib/adventure'
import { cn } from '../../lib/format'
import { Markdown } from '../chat/Markdown'
import { IconButton } from '../ui'

export interface PassageActions {
  onGoHere: (id: string) => void
  onSwitch: (id: string) => void
  onTryAgain: (id: string) => void
  onNewChoices: (id: string) => void
  onEdit: (id: string) => void
  onDeleteBranch: (id: string) => void
}

const small = 'size-7'

/** One passage of the reading column, with the toolbar that shows on hover or keyboard focus. */
export const PassageBlock = memo(function PassageBlock({
  adv,
  node,
  isPosition,
  busy,
  actions
}: {
  adv: Adventure
  node: AdventureNode
  isPosition: boolean
  busy: boolean
  actions: PassageActions
}) {
  const sib = siblingIndex(adv, node.id)
  const goTo = (id: string | null): void => {
    const leaf = id ? newestLeaf(adv, id) : undefined
    if (leaf) actions.onSwitch(leaf.id)
  }
  return (
    <div className="group relative">
      <div className="absolute -top-3 right-0 z-10 flex items-center gap-0.5 rounded-lg border border-border bg-panel px-1 py-0.5 opacity-0 shadow-sm transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
        {sib.count > 1 && (
          <>
            <IconButton label="Previous path" className={small} disabled={!sib.prev} onClick={() => goTo(sib.prev)}>
              <ChevronLeft size={15} />
            </IconButton>
            <span className="text-[11px] text-muted tabular-nums" title="Other paths from the same step">
              Path {sib.index + 1} of {sib.count}
            </span>
            <IconButton label="Next path" className={small} disabled={!sib.next} onClick={() => goTo(sib.next)}>
              <ChevronRight size={15} />
            </IconButton>
          </>
        )}
        {!isPosition && (
          <IconButton label="Go back here" className={small} onClick={() => actions.onGoHere(node.id)}>
            <CornerUpLeft size={15} />
          </IconButton>
        )}
        {isPosition && node.parentId !== null && (
          <IconButton label="Try again" className={small} disabled={busy} onClick={() => actions.onTryAgain(node.id)}>
            <RotateCcw size={15} />
          </IconButton>
        )}
        {isPosition && (
          <IconButton label="New choices" className={small} disabled={busy} onClick={() => actions.onNewChoices(node.id)}>
            <Shuffle size={15} />
          </IconButton>
        )}
        <IconButton label="Edit" className={small} disabled={busy} onClick={() => actions.onEdit(node.id)}>
          <Pencil size={15} />
        </IconButton>
        {node.parentId !== null && (
          <IconButton label="Delete branch" className={small} disabled={busy} onClick={() => actions.onDeleteBranch(node.id)}>
            <Trash2 size={15} />
          </IconButton>
        )}
      </div>
      <Markdown text={node.text} className={cn('text-[15px] leading-[1.7]', !isPosition && 'text-body')} />
      {node.edited && <div className="mt-1 text-[11px] text-faint">Edited by you</div>}
    </div>
  )
})
