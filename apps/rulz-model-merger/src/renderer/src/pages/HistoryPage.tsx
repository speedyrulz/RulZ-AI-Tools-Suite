import { Trash2 } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { HistoryRow } from '../components/history/HistoryRow'
import { Button, Card, Dialog, Input, PageTitle, Segmented } from '../components/ui'
import { HISTORY_FILTERS, entryId, filterEntries } from '../lib/history'
import { useGrid } from '../stores/grid'
import { useHistory } from '../stores/history'

export function HistoryPage() {
  const entries = useHistory((s) => s.entries)
  const loaded = useHistory((s) => s.loaded)
  const filter = useHistory((s) => s.filter)
  const query = useHistory((s) => s.query)
  const setFilter = useHistory((s) => s.setFilter)
  const setQuery = useHistory((s) => s.setQuery)
  const [confirm, setConfirm] = useState<string | null>(null)

  useEffect(() => {
    void useHistory.getState().refresh()
  }, [])

  const shown = useMemo(() => filterEntries(entries, filter, query), [entries, filter, query])

  return (
    <div className="space-y-6">
      <PageTitle title="History">
        Every merge, bake, LoRA merge and grid, kept after the app closes. Reopen fills its page with the recipe; Re-run reopens it and presses its start
        button.
      </PageTitle>
      <Card>
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <Segmented label="Show" value={filter} options={HISTORY_FILTERS} onChange={setFilter} />
          <div className="w-64">
            <Input aria-label="Search names" placeholder="Search names" value={query} onChange={(e) => setQuery(e.target.value)} />
          </div>
        </div>
        {shown.length === 0 ? (
          <p className="text-[13px] text-muted">
            {!loaded ? 'Loading history.' : entries.length === 0 ? 'Nothing here yet. Finished merges and grids show up in this list.' : 'Nothing matches.'}
          </p>
        ) : (
          <ul className="space-y-2">
            {shown.map((e, i) => (
              <HistoryRow key={`${entryId(e)}#${i}`} entry={e} onDeleteGrid={setConfirm} />
            ))}
          </ul>
        )}
      </Card>
      <Dialog
        open={confirm !== null}
        onOpenChange={(o) => {
          if (!o) setConfirm(null)
        }}
        title="Delete this grid?"
        footer={
          <>
            <Button onClick={() => setConfirm(null)}>Keep it</Button>
            <Button
              variant="danger"
              onClick={() => {
                const id = confirm
                setConfirm(null)
                if (id) void useGrid.getState().remove(id).then(() => useHistory.getState().refresh())
              }}
            >
              <Trash2 size={14} />
              Delete grid
            </Button>
          </>
        }
      >
        <p className="text-[13px] text-muted">The folder of this grid, with its images, goes to the Recycle Bin. You can restore it from there.</p>
      </Dialog>
    </div>
  )
}
