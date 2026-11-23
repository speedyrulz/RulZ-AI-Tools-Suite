import { childrenOf, newestLeaf, siblingsOf, type Adventure, type AdventureNode, type AdventureSetup } from '@shared/adventure'
import { Button, Sheet } from '../ui'
import { SetupFields } from '../../pages/Adventures'

export function PassageMenuSheet({ adventure, node, position, busy, onClose, onMove, onRetry, onEdit }: {
  adventure: Adventure
  node: AdventureNode | undefined
  position: string | null
  busy: boolean
  onClose: () => void
  onMove: (id: string) => void
  onRetry: (node: AdventureNode, what: 'passage' | 'choices') => void
  onEdit: (node: AdventureNode) => void
}) {
  const siblings = node ? siblingsOf(adventure, node.id) : []
  return (
    <Sheet open={!!node} onOpenChange={(open) => !open && onClose()} title="Passage">
      <div className="space-y-2 px-4 pb-4">
        {node && (
          <>
            <Button className="w-full" onClick={() => {
              onMove(node.id)
              onClose()
            }}>Go back here</Button>
            {siblings.length > 1 && (
              <div className="flex gap-2">
                {[-1, 1].map((delta) => {
                  const sibling = siblings[siblings.findIndex((item) => item.id === node.id) + delta]
                  return (
                    <Button
                      key={delta}
                      className="flex-1"
                      disabled={!sibling}
                      onClick={() => {
                        const leaf = sibling && newestLeaf(adventure, sibling.id)
                        if (leaf) onMove(leaf.id)
                        onClose()
                      }}
                    >
                      {delta < 0 ? 'Previous path' : 'Next path'}
                    </Button>
                  )
                })}
              </div>
            )}
            {node.id === position && (
              <>
                <Button className="w-full" disabled={busy} onClick={() => onRetry(node, 'passage')}>Try again</Button>
                <Button className="w-full" disabled={busy} onClick={() => onRetry(node, 'choices')}>New choices</Button>
              </>
            )}
            <Button className="w-full" disabled={busy} onClick={() => onEdit(node)}>Edit</Button>
          </>
        )}
      </div>
    </Sheet>
  )
}

export function StateSheet({ open, node, onClose }: {
  open: boolean
  node: AdventureNode | undefined
  onClose: () => void
}) {
  const stats = node ? Object.entries(node.state.stats).map(([name, value]) => `${name}: ${value}`).join(', ') : ''
  return (
    <Sheet open={open} onOpenChange={(value) => !value && onClose()} title="State">
      <div className="space-y-4 px-4 pb-4">
        <p>Inventory: {node?.state.inventory.join(', ') || 'None'}</p>
        <p>Stats: {stats || 'None'}</p>
        <p>Flags: {node?.state.flags.join(', ') || 'None'}</p>
      </div>
    </Sheet>
  )
}

export function MapSheet({ open, adventure, path, onClose, onMove }: {
  open: boolean
  adventure: Adventure
  path: AdventureNode[]
  onClose: () => void
  onMove: (id: string) => void
}) {
  const mapNode = (node: AdventureNode, depth: number): React.ReactNode => (
    <div key={node.id}>
      <button
        className={`min-h-11 w-full rounded-lg px-3 text-left ${
          path.some((item) => item.id === node.id) ? 'bg-accent-soft text-text' : 'text-muted'
        }`}
        style={{ paddingLeft: `${depth * 16 + 12}px` }}
        onClick={() => {
          onMove(node.id)
          onClose()
        }}
      >
        {node.input.kind === 'start' ? 'Opening' : node.input.text.slice(0, 65)}
      </button>
      {childrenOf(adventure, node.id).map((child) => mapNode(child, depth + 1))}
    </div>
  )
  const root = adventure.rootId && adventure.nodes[adventure.rootId]
  return (
    <Sheet open={open} onOpenChange={(value) => !value && onClose()} title="Map">
      <div className="space-y-4 px-4 pb-4">{root && mapNode(root, 0)}</div>
    </Sheet>
  )
}

export function SettingsSheet({ setup, setSetup, models, onClose, onSave }: {
  setup: AdventureSetup
  setSetup: (setup: AdventureSetup) => void
  models: { id: string, name: string, loaded: boolean }[]
  onClose: () => void
  onSave: () => void
}) {
  return (
    <Sheet open onOpenChange={(value) => !value && onClose()} title="Settings">
      <div className="space-y-4 px-4 pb-4">
        <SetupFields setup={setup} setSetup={setSetup} models={models} />
        <Button variant="primary" className="w-full" onClick={onSave}>Save</Button>
      </div>
    </Sheet>
  )
}
