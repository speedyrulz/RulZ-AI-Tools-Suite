import { Mic, Send, Trash2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { appendVoiceNote } from '@shared/novelVoice'
import { cn, errorMessage } from '../lib/format'
import { entryPath } from '../lib/outlineOps'
import { link } from '../lib/remote'
import { chatKey, useInterview } from '../stores/interview'
import { useProject } from '../stores/project'
import { useUi } from '../stores/ui'
import { Button, Select, Sheet, Spinner, Textarea } from './ui'

/** Saves a note to the character's "Voice" section: read the entry file, add the note, save on its base hash. */
export async function saveVoiceNote(characterId: string, note: string): Promise<boolean> {
  const folder = useProject.getState().state?.folder
  if (!folder) return false
  const path = entryPath(characterId)
  try {
    const file = await link.invoke('novel:readFile', folder, path)
    if (file.readOnly) {
      toast.error('This entry can’t be changed from here (it isn’t plain UTF-8 text).')
      return false
    }
    const next = appendVoiceNote(file.text, note)
    if (next === file.text) {
      toast.error('There was nothing to add.')
      return false
    }
    const res = await link.invoke('novel:saveFile', { folder, path, text: next, baseHash: file.hash })
    if (res.ok) {
      toast.success('Added to the voice notes.')
      return true
    }
    toast.error(res.reason === 'conflict' ? 'The entry changed on the computer. Try again.' : res.reason === 'error' ? res.error : 'The entry can’t be saved right now.')
    return false
  } catch (e) {
    toast.error(errorMessage(e))
    return false
  }
}

/** Talk to one of your characters. Their answers use the story bible and the story so far. */
export function InterviewTab() {
  const state = useProject((s) => s.state)
  const sceneId = useUi((s) => s.sceneId)
  const characterId = useUi((s) => s.characterId)
  const setCharacter = useUi((s) => s.setCharacter)
  const chats = useInterview((s) => s.chats)
  const waiting = useInterview((s) => s.waiting)
  const [text, setText] = useState('')
  const [sending, setSending] = useState(false)
  const [note, setNote] = useState<string | null>(null)
  const end = useRef<HTMLDivElement>(null)

  const characters = state?.bible.filter((e) => e.kind === 'character') ?? []
  const current = characters.find((c) => c.id === characterId) ?? characters[0]
  const key = state && current ? chatKey(state.folder, current.id) : ''
  const turns = chats[key] ?? []
  const busy = !!waiting[key]

  useEffect(() => end.current?.scrollIntoView({ block: 'end' }), [turns.length, busy])

  if (!state) return null
  if (!current) return <p className="px-6 py-10 text-center text-[14px] text-faint">Add a character to the story bible to interview them.</p>

  const send = async (): Promise<void> => {
    const q = text.trim()
    if (!q || busy || sending) return
    setSending(true)
    try {
      if (await useInterview.getState().ask(state.folder, sceneId, current.id, q)) setText('')
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <Select className="flex-1" value={current.id} onChange={(e) => setCharacter(e.target.value)} aria-label="Character">
          {characters.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </Select>
        <Button size="sm" variant="ghost" disabled={turns.length === 0} aria-label="Clear the conversation" onClick={() => useInterview.getState().clear(state.folder, current.id)}>
          <Trash2 size={16} />
        </Button>
      </div>
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-4 py-3">
        {turns.length === 0 && <p className="py-8 text-center text-[14px] text-faint">Ask {current.name} something. It helps you find their voice.</p>}
        {turns.map((t, i) => (
          <div key={i} className={cn('flex', t.role === 'user' ? 'justify-end' : 'justify-start')}>
            <div className={cn('max-w-[88%] rounded-2xl px-3.5 py-2.5 text-[15px] leading-relaxed whitespace-pre-wrap', t.role === 'user' ? 'bg-accent-soft text-text' : t.failed ? 'border border-danger-border bg-danger-bg text-danger-text' : 'border border-border bg-panel')}>
              {t.text}
              {t.role === 'assistant' && !t.failed && (
                <div className="mt-1.5">
                  <button className="flex items-center gap-1 text-[12.5px] text-muted" onClick={() => setNote(t.text)}>
                    <Mic size={13} /> Add to voice notes
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}
        {busy && (
          <div className="flex items-center gap-2 text-[14px] text-muted">
            <Spinner className="text-accent" /> {current.name} is thinking…
          </div>
        )}
        <div ref={end} />
      </div>
      <div className="flex items-end gap-2 border-t border-border bg-panel px-3 py-2">
        <Textarea rows={2} className="flex-1" placeholder={`Ask ${current.name}…`} value={text} onChange={(e) => setText(e.target.value)} />
        <Button variant="primary" className="h-11" disabled={!text.trim() || busy || sending} onClick={() => void send()} aria-label="Send">
          {sending ? <Spinner /> : <Send size={17} />}
        </Button>
      </div>
      <VoiceSheet characterId={current.id} name={current.name} note={note} setNote={setNote} />
    </div>
  )
}

function VoiceSheet({ characterId, name, note, setNote }: { characterId: string; name: string; note: string | null; setNote: (n: string | null) => void }) {
  const [busy, setBusy] = useState(false)
  return (
    <Sheet
      open={note !== null}
      onOpenChange={(o) => !o && setNote(null)}
      title={`Add to ${name}’s voice notes`}
      description="Trim it to what shows how they talk. It becomes one line in the entry’s Voice section."
      footer={
        <>
          <Button onClick={() => setNote(null)}>Cancel</Button>
          <Button
            variant="primary"
            disabled={busy || !note?.trim()}
            onClick={async () => {
              setBusy(true)
              try {
                if (note && (await saveVoiceNote(characterId, note))) setNote(null)
              } finally {
                setBusy(false)
              }
            }}
          >
            Add
          </Button>
        </>
      }
    >
      <div className="px-4 pb-4">
        <Textarea rows={6} maxLength={500} value={note ?? ''} onChange={(e) => setNote(e.target.value)} />
        <p className="mt-1 text-right text-[12px] text-faint">{(note ?? '').length} / 500</p>
      </div>
    </Sheet>
  )
}
