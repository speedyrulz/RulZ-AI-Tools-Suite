import { FolderOpen, Images, Play, RotateCcw, Trash2 } from 'lucide-react'
import { Component, type ReactNode } from 'react'
import type { HistoryEntry } from '@shared/history'
import { call } from '../../lib/api'
import { entryDate, entryKindLabel, entryOutput, entryReadable, entryStatus, entrySummary } from '../../lib/history'
import { useHistory } from '../../stores/history'
import { Badge, Button, StatusChip } from '../ui'

interface RowProps {
  entry: HistoryEntry
  onDeleteGrid: (id: string) => void
}

/** Only Remove / Delete, for an entry that cannot be read or drawn. Nothing here reads the recipe. */
function BrokenRow({ entry, onDeleteGrid }: RowProps) {
  const kind = entry?.kind
  const id = entry?.job?.id
  const canAct = (kind === 'merge' || kind === 'loraMerge' || kind === 'grid') && typeof id === 'string' && id !== ''
  return (
    <li className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-danger-border bg-danger-bg px-3 py-2.5">
      <p className="text-[13px] text-danger-text">This entry could not be read.</p>
      {canAct &&
        (kind === 'grid' ? (
          <Button size="sm" variant="ghost" onClick={() => onDeleteGrid(id)}>
            <Trash2 size={14} />
            Delete grid
          </Button>
        ) : (
          <Button size="sm" variant="ghost" onClick={() => void useHistory.getState().remove(entry)}>
            <Trash2 size={14} />
            Remove from history
          </Button>
        ))}
    </li>
  )
}

class RowBoundary extends Component<RowProps & { children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true }
  }
  componentDidUpdate(prev: RowProps): void {
    if (this.state.failed && prev.entry !== this.props.entry) this.setState({ failed: false })
  }
  render(): ReactNode {
    return this.state.failed ? <BrokenRow entry={this.props.entry} onDeleteGrid={this.props.onDeleteGrid} /> : this.props.children
  }
}

function ReadableRow({ entry, onDeleteGrid }: RowProps) {
  const st = entryStatus(entry)
  const output = entryOutput(entry)
  const running = entry.kind === 'grid' && entry.job.status === 'running'
  const { reopen, rerun, remove, openInResults } = useHistory.getState()
  return (
    <li className="space-y-2 rounded-lg border border-border bg-panel-2 px-3 py-2.5">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <Badge tone={entry.kind === 'grid' ? 'accent' : 'neutral'}>{entryKindLabel(entry)}</Badge>
        <span className="text-xs text-faint tabular-nums">{entryDate(entry)}</span>
        <StatusChip tone={st.tone} label={st.label} busy={st.busy} />
      </div>
      <p className="text-[13px] break-words">{entrySummary(entry) || 'No details saved.'}</p>
      {output && (
        <p className="text-xs break-all text-muted" title={output}>
          {output}
        </p>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" disabled={running} onClick={() => void reopen(entry)}>
          <RotateCcw size={14} />
          Reopen
        </Button>
        <Button size="sm" disabled={running} onClick={() => void rerun(entry)}>
          <Play size={14} />
          Re-run
        </Button>
        {entry.kind === 'grid' ? (
          <>
            <Button size="sm" disabled={running} onClick={() => void openInResults(entry.job.id)}>
              <Images size={14} />
              Open in Results
            </Button>
            <Button size="sm" onClick={() => void call('grid:openFolder', entry.job.id)}>
              <FolderOpen size={14} />
              Open folder
            </Button>
            <Button size="sm" variant="ghost" disabled={running} onClick={() => onDeleteGrid(entry.job.id)}>
              <Trash2 size={14} />
              Delete grid
            </Button>
          </>
        ) : (
          <Button size="sm" variant="ghost" onClick={() => void remove(entry)}>
            <Trash2 size={14} />
            Remove from history
          </Button>
        )}
      </div>
    </li>
  )
}

export function HistoryRow(props: RowProps) {
  if (!entryReadable(props.entry)) return <BrokenRow {...props} />
  return (
    <RowBoundary {...props}>
      <ReadableRow {...props} />
    </RowBoundary>
  )
}
