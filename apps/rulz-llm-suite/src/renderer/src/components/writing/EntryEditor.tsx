import { History as HistoryIcon, TriangleAlert } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { BibleEntry, BibleKind } from '@shared/novel'
import { formatEntry, parseEntry } from '@shared/frontMatter'
import { entryPath, useNovel } from '../../stores/novel'
import { Field, IconButton, Input, NumberInput, Select, Slider, Switch, Textarea } from '../ui'
import { BufferBanners } from './Banners'
import { CodeEditor } from './CodeEditor'
import { KIND_LABELS } from './common'
import { HistoryDialog } from './HistoryDialog'

/** "a, b" or one per line → trimmed, non-empty words. */
export function parseWords(text: string): string[] {
  return text
    .split(/[,\n]/)
    .map((w) => w.trim())
    .filter(Boolean)
}

/** A comma or one-per-line list. Keeps its own text while typing, so a trailing comma stays until the next word. */
function ListField({ value, onChange, placeholder, label }: { value: string[]; onChange: (list: string[]) => void; placeholder?: string; label: string }) {
  const [draft, setDraft] = useState(value.join(', '))
  const [seen, setSeen] = useState(value)
  if (seen !== value) {
    setSeen(value)
    if (JSON.stringify(parseWords(draft)) !== JSON.stringify(value)) setDraft(value.join(', '))
  }
  return (
    <Textarea
      rows={2}
      aria-label={label}
      placeholder={placeholder}
      value={draft}
      onChange={(e) => {
        setDraft(e.target.value)
        onChange(parseWords(e.target.value))
      }}
    />
  )
}

type Fields = Omit<BibleEntry, 'id' | 'problems'>

/** A bible entry: its fields on top and its body in a text editor. The buffer holds the whole file; every change rebuilds it with formatEntry. */
export function EntryEditor({ entryId }: { entryId: string }) {
  const path = entryPath(entryId)
  const buffer = useNovel((s) => s.buffers[path])
  const blocked = useNovel((s) => !!s.state?.blocked)
  const [historyOpen, setHistoryOpen] = useState(false)
  // The body editor keeps its own text while typing; it restarts only when the file changed from outside.
  const [generation, setGeneration] = useState(0)
  const ownVersion = useRef(buffer?.version ?? 0)
  useEffect(() => {
    if (buffer && buffer.version !== ownVersion.current) {
      ownVersion.current = buffer.version
      setGeneration((g) => g + 1)
    }
  }, [buffer?.version])
  useEffect(() => {
    ownVersion.current = useNovel.getState().buffers[path]?.version ?? 0
    setGeneration(0)
  }, [path])

  // What the body editor starts with; typing never feeds back into it (formatting could shift the cursor).
  const startBody = useMemo(
    () => parseEntry(entryId, useNovel.getState().buffers[path]?.text ?? '').body,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [generation, path, !!buffer]
  )

  if (!buffer) return <div className="p-6 text-sm text-faint">Opening…</div>
  const entry = parseEntry(entryId, buffer.text)
  const readOnly = blocked || buffer.readOnly

  const write = (patch: Partial<Fields>): void => {
    const cur = useNovel.getState().buffers[path]
    if (!cur) return
    const parsed = parseEntry(entryId, cur.text)
    const fields: Fields = { name: parsed.name, kind: parsed.kind, aliases: parsed.aliases, triggers: parsed.triggers, priority: parsed.priority, always: parsed.always, body: parsed.body }
    useNovel.getState().edit(path, formatEntry({ ...fields, ...patch }, cur.text))
    ownVersion.current = useNovel.getState().buffers[path]?.version ?? ownVersion.current
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <BufferBanners path={path} />
      <div className="space-y-3 border-b border-border bg-panel px-4 py-3">
        <div className="flex items-end gap-3">
          <div className="min-w-0 flex-1">
            <Field label="Name">
              <Input value={entry.name} disabled={readOnly} onChange={(e) => write({ name: e.target.value })} />
            </Field>
          </div>
          <div className="w-36">
            <Field label="Kind">
              <Select value={entry.kind} disabled={readOnly} onChange={(e) => write({ kind: e.target.value as BibleKind })}>
                {(Object.keys(KIND_LABELS) as BibleKind[]).map((k) => (
                  <option key={k} value={k}>
                    {KIND_LABELS[k]}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
          <IconButton label="History of this entry" onClick={() => setHistoryOpen(true)}>
            <HistoryIcon size={16} />
          </IconButton>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          <Field label="Aliases" hint="Other names; comma or one per line">
            <ListField label="Aliases" value={entry.aliases} onChange={(aliases) => write({ aliases })} placeholder="Mara, the Salt Witch" />
          </Field>
          <Field label="Triggers" hint="Extra words that pull this entry in">
            <ListField label="Triggers" value={entry.triggers} onChange={(triggers) => write({ triggers })} placeholder="witch, Venn" />
          </Field>
        </div>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
          <div className="flex min-w-60 flex-1 items-center gap-3">
            <span className="text-[13px] whitespace-nowrap">Priority</span>
            <Slider value={entry.priority} min={0} max={100} step={1} disabled={readOnly} onChange={(priority) => write({ priority: Math.round(priority) })} />
            <NumberInput className="w-16" value={entry.priority} min={0} max={100} step={1} onChange={(v) => write({ priority: Math.min(100, Math.max(0, Math.round(v ?? 50))) })} />
          </div>
          <label className="flex items-center gap-2 text-[13px]">
            <Switch checked={entry.always} disabled={readOnly} onCheckedChange={(always) => write({ always })} />
            In every request
          </label>
        </div>
        {entry.problems.length > 0 && (
          <ul className="space-y-1 rounded-lg border border-warn/30 bg-warn/10 px-3 py-2 text-xs text-warn-text" aria-label="Problems in this entry">
            {entry.problems.map((p, i) => (
              <li key={i} className="flex items-start gap-1.5">
                <TriangleAlert size={13} className="mt-px shrink-0" /> {p}
              </li>
            ))}
          </ul>
        )}
      </div>
      <div className="min-h-0 flex-1">
        <CodeEditor
          key={`${path}:${generation}`}
          path={path}
          text={startBody}
          readOnly={readOnly}
          prose
          placeholder="Appearance, history, voice notes. Markdown."
          onChange={(body) => write({ body })}
        />
      </div>
      <HistoryDialog path={path} open={historyOpen} onOpenChange={setHistoryOpen} />
    </div>
  )
}
