import { ArrowDown, ArrowUp, Plus, X } from 'lucide-react'
import { useState } from 'react'
import { draftOfState, moveItem, sameState, stateOfDraft, type StateDraft } from '../../lib/adventure'
import { useAdventure } from '../../stores/adventure'
import { Button, Dialog, Field, IconButton, Input, Textarea } from '../ui'

interface ChoiceDraft {
  id?: string
  text: string
}

function StringList({ label, items, onChange, placeholder }: { label: string; items: string[]; onChange: (v: string[]) => void; placeholder: string }) {
  return (
    <Field label={label}>
      <div className="space-y-1.5">
        {items.map((v, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <Input aria-label={`${label} ${i + 1}`} value={v} placeholder={placeholder} onChange={(e) => onChange(items.map((x, j) => (j === i ? e.target.value : x)))} />
            <IconButton label={`Remove ${label.toLowerCase()} ${i + 1}`} onClick={() => onChange(items.filter((_, j) => j !== i))}>
              <X size={15} />
            </IconButton>
          </div>
        ))}
        <Button size="sm" variant="ghost" onClick={() => onChange([...items, ''])}>
          <Plus size={14} aria-hidden /> Add
        </Button>
      </div>
    </Field>
  )
}

/** Edit one passage: its text, its choices (ids kept) and the tracked state. Saved with the node's rev. */
export function EditDialog({ nodeId, onClose }: { nodeId: string; onClose: () => void }) {
  const node = useAdventure((s) => s.adv?.nodes[nodeId])
  // What the dialog was opened on: the rev it saves against, and the values it started from.
  const [base] = useState(() => {
    const n = useAdventure.getState().adv?.nodes[nodeId]
    return n ? { rev: n.rev, text: n.text, choices: n.choices, state: n.state } : null
  })
  const [text, setText] = useState(base?.text ?? '')
  const [choices, setChoices] = useState<ChoiceDraft[]>(() => (base?.choices ?? []).map((c) => ({ id: c.id, text: c.text })))
  const [state, setState] = useState<StateDraft>(() => draftOfState(base?.state ?? { inventory: [], stats: {}, flags: [] }))
  const [saving, setSaving] = useState(false)

  if (!base || !node) return null

  const cleanChoices = choices.map((c) => ({ ...c, text: c.text.trim() })).filter((c) => c.text)
  const choicesChanged = JSON.stringify(cleanChoices.map((c) => [c.id ?? null, c.text])) !== JSON.stringify(base.choices.map((c) => [c.id, c.text]))
  const nextState = stateOfDraft(state)
  const stateChanged = !sameState(nextState, base.state)
  const textChanged = text !== base.text
  const changed = textChanged || choicesChanged || stateChanged

  const save = async (): Promise<void> => {
    if (!changed) return onClose()
    if (!text.trim()) return
    setSaving(true)
    const result = await useAdventure.getState().edit({
      adventureId: useAdventure.getState().adv?.id ?? '',
      nodeId,
      rev: base.rev,
      ...(textChanged ? { text } : {}),
      ...(choicesChanged ? { choices: cleanChoices } : {}),
      ...(stateChanged ? { state: nextState } : {})
    })
    setSaving(false)
    // On a conflict the store has loaded the newest copy; close so it is read again from the start.
    if (result !== 'error') onClose()
  }

  return (
    <Dialog
      open
      onOpenChange={(v) => !v && !saving && onClose()}
      title="Edit passage"
      description="Your edits replace the model’s text. Changed state carries down the passages after this one."
      className="w-[min(680px,calc(100vw-32px))]"
      footer={
        <>
          <Button variant="ghost" onClick={onClose} disabled={saving}>
            Cancel
          </Button>
          <Button variant="primary" disabled={saving || !text.trim() || !changed} onClick={() => void save()}>
            {saving ? 'Saving…' : 'Save'}
          </Button>
        </>
      }
    >
      <div className="space-y-5 p-5">
        <Field label="Passage">
          <Textarea rows={10} value={text} onChange={(e) => setText(e.target.value)} className="text-[14px] leading-relaxed" />
        </Field>

        <Field label="Choices">
          <div className="space-y-1.5">
            {choices.map((c, i) => (
              <div key={c.id ?? `new-${i}`} className="flex items-center gap-1.5">
                <Input aria-label={`Choice ${i + 1}`} value={c.text} onChange={(e) => setChoices(choices.map((x, j) => (j === i ? { ...x, text: e.target.value } : x)))} />
                <IconButton label="Move up" disabled={i === 0} onClick={() => setChoices(moveItem(choices, i, i - 1))}>
                  <ArrowUp size={15} />
                </IconButton>
                <IconButton label="Move down" disabled={i === choices.length - 1} onClick={() => setChoices(moveItem(choices, i, i + 1))}>
                  <ArrowDown size={15} />
                </IconButton>
                <IconButton label={`Remove choice ${i + 1}`} onClick={() => setChoices(choices.filter((_, j) => j !== i))}>
                  <X size={15} />
                </IconButton>
              </div>
            ))}
            <Button size="sm" variant="ghost" onClick={() => setChoices([...choices, { text: '' }])}>
              <Plus size={14} aria-hidden /> Add choice
            </Button>
          </div>
        </Field>

        <div className="space-y-4 rounded-lg border border-border p-4">
          <div className="text-[11px] font-semibold tracking-wide text-faint uppercase">State after this passage</div>
          <StringList label="Inventory" items={state.inventory} placeholder="rusty key" onChange={(inventory) => setState({ ...state, inventory })} />
          <Field label="Stats">
            <div className="space-y-1.5">
              {state.stats.map((s, i) => (
                <div key={i} className="flex items-center gap-1.5">
                  <Input
                    aria-label={`Stat ${i + 1} name`}
                    value={s.name}
                    placeholder="Health"
                    onChange={(e) => setState({ ...state, stats: state.stats.map((x, j) => (j === i ? { ...x, name: e.target.value } : x)) })}
                  />
                  <Input
                    aria-label={`Stat ${i + 1} value`}
                    className="w-28"
                    value={s.value}
                    placeholder="8"
                    onChange={(e) => setState({ ...state, stats: state.stats.map((x, j) => (j === i ? { ...x, value: e.target.value } : x)) })}
                  />
                  <IconButton label={`Remove stat ${i + 1}`} onClick={() => setState({ ...state, stats: state.stats.filter((_, j) => j !== i) })}>
                    <X size={15} />
                  </IconButton>
                </div>
              ))}
              <Button size="sm" variant="ghost" onClick={() => setState({ ...state, stats: [...state.stats, { name: '', value: '' }] })}>
                <Plus size={14} aria-hidden /> Add
              </Button>
            </div>
          </Field>
          <StringList label="Flags" items={state.flags} placeholder="met the ferryman" onChange={(flags) => setState({ ...state, flags })} />
        </div>
      </div>
    </Dialog>
  )
}
