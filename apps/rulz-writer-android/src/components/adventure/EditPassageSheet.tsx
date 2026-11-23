import { ArrowDown, ArrowUp, Plus, Trash2 } from 'lucide-react'
import { useState } from 'react'
import type { AdventureEdit, AdventureNode } from '@shared/adventure'
import { Button, Field, IconButton, Input, Sheet, Textarea } from '../ui'
import { choicesForEdit, parseStatValue, type ChoiceRow, type StatRow } from './editHelpers'

let nextKey = 0
const newKey = (): string => String(++nextKey)
const lines = (value: string): string[] => value.split('\n').map((line) => line.trim()).filter(Boolean)

export function EditPassageSheet({
  adventureId,
  node,
  onClose,
  onSave
}: {
  adventureId: string
  node: AdventureNode
  onClose: () => void
  onSave: (edit: AdventureEdit) => Promise<'ok' | 'conflict' | 'error'>
}) {
  const [text, setText] = useState(node.text)
  const [choices, setChoices] = useState<ChoiceRow[]>(() => node.choices.map((choice) => ({ ...choice, key: newKey() })))
  const [inventory, setInventory] = useState(node.state.inventory.join('\n'))
  const [flags, setFlags] = useState(node.state.flags.join('\n'))
  const [stats, setStats] = useState<StatRow[]>(() => Object.entries(node.state.stats).map(([name, value]) => ({
    key: newKey(), name, value: String(value)
  })))

  const moveChoice = (index: number, offset: number): void => {
    setChoices((rows) => {
      const next = [...rows]
      const target = index + offset
      if (target < 0 || target >= next.length) return rows
      const previous = next[index]
      next[index] = next[target]
      next[target] = previous
      return next
    })
  }
  const save = (): void => {
    const stateStats = Object.fromEntries(
      stats.filter((row) => row.name.trim()).map((row) => [row.name.trim(), parseStatValue(row.value)])
    )
    const nextChoices = choicesForEdit(choices)
    const state = { inventory: lines(inventory), flags: lines(flags), stats: stateStats }
    const edit: AdventureEdit = {
      adventureId,
      nodeId: node.id,
      rev: node.rev
    }
    if (text !== node.text) edit.text = text
    if (JSON.stringify(nextChoices) !== JSON.stringify(node.choices)) edit.choices = nextChoices
    if (JSON.stringify(state) !== JSON.stringify(node.state)) edit.state = state
    void onSave(edit).then((result) => result !== 'error' && onClose())
  }

  return (
    <Sheet open onOpenChange={(open) => !open && onClose()} title="Edit passage">
      <div className="space-y-4 px-4 pb-4">
        <Field label="Passage">
          <Textarea aria-label="Passage" rows={8} value={text} onChange={(e) => setText(e.target.value)} />
        </Field>
        <Field label="Choices">
          <div className="space-y-2">
            {choices.map((choice, index) => (
              <div className="flex items-center gap-1" key={choice.key}>
                <Input
                  aria-label={`Choice ${index + 1}`}
                  value={choice.text}
                  onChange={(e) => setChoices((rows) => rows.map((row) =>
                    row.key === choice.key ? { ...row, text: e.target.value } : row
                  ))}
                />
                <IconButton label={`Move choice ${index + 1} up`} disabled={index === 0} onClick={() => moveChoice(index, -1)}>
                  <ArrowUp size={18} />
                </IconButton>
                <IconButton
                  label={`Move choice ${index + 1} down`}
                  disabled={index === choices.length - 1}
                  onClick={() => moveChoice(index, 1)}
                >
                  <ArrowDown size={18} />
                </IconButton>
                <IconButton label={`Remove choice ${index + 1}`} onClick={() => setChoices((rows) =>
                  rows.filter((row) => row.key !== choice.key)
                )}>
                  <Trash2 size={18} />
                </IconButton>
              </div>
            ))}
            <Button onClick={() => setChoices((rows) => [...rows, { key: newKey(), text: '' }])}>
              <Plus size={18} /> Add choice
            </Button>
          </div>
        </Field>
        <Field label="Inventory" hint="One per line">
          <Textarea aria-label="Inventory" value={inventory} onChange={(e) => setInventory(e.target.value)} />
        </Field>
        <Field label="Flags" hint="One per line">
          <Textarea aria-label="Flags" value={flags} onChange={(e) => setFlags(e.target.value)} />
        </Field>
        <Field label="Stats">
          <div className="space-y-2">
            {stats.map((stat, index) => (
              <div className="flex items-center gap-1" key={stat.key}>
                <Input
                  aria-label={`Stat ${index + 1} name`}
                  placeholder="Name"
                  value={stat.name}
                  onChange={(e) => setStats((rows) => rows.map((row) =>
                    row.key === stat.key ? { ...row, name: e.target.value } : row
                  ))}
                />
                <Input
                  aria-label={`Stat ${index + 1} value`}
                  placeholder="Value"
                  value={stat.value}
                  onChange={(e) => setStats((rows) => rows.map((row) =>
                    row.key === stat.key ? { ...row, value: e.target.value } : row
                  ))}
                />
                <IconButton label={`Remove stat ${index + 1}`} onClick={() => setStats((rows) =>
                  rows.filter((row) => row.key !== stat.key)
                )}>
                  <Trash2 size={18} />
                </IconButton>
              </div>
            ))}
            <Button onClick={() => setStats((rows) => [...rows, { key: newKey(), name: '', value: '' }])}>
              <Plus size={18} /> Add stat
            </Button>
          </div>
        </Field>
        <Button variant="primary" className="w-full" onClick={save}>Save</Button>
      </div>
    </Sheet>
  )
}
