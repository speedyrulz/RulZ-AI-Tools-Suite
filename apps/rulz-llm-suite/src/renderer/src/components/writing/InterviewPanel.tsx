import { Mic, Send, Square, Trash2 } from 'lucide-react'
import { useRef, useState } from 'react'
import { interviewKey, voiceNoteBlock } from '../../lib/novelRun'
import { entryPath, sceneOf, useNovel } from '../../stores/novel'
import { Button, Select, Textarea } from '../ui'
import { PanelSection } from './common'

export function InterviewPanel() {
  const state = useNovel((s) => s.state)
  const target = useNovel((s) => s.target)
  const lastSceneId = useNovel((s) => s.lastSceneId)
  const gen = useNovel((s) => s.gen)
  const runKey = useNovel((s) => s.interviewRun?.key ?? null)
  const interviews = useNovel((s) => s.interviews)
  const dropped = useNovel((s) => s.interviewDropped)
  const [picked, setPicked] = useState<string>('')
  const [question, setQuestion] = useState('')

  const sceneId = target?.kind === 'scene' ? target.id : lastSceneId
  const scene = sceneId ? sceneOf(state, sceneId)?.scene : null
  const characters = (state?.bible ?? []).filter((e) => e.kind === 'character')
  // The scene's POV is the default; the user's pick stays while it is still a character.
  const characterId = characters.some((c) => c.id === picked) ? picked : scene?.pov && characters.some((c) => c.id === scene.pov) ? scene.pov : (characters[0]?.id ?? '')
  const character = characters.find((c) => c.id === characterId)
  const key = scene && character ? interviewKey(character.id, scene.id) : ''
  const turns = key ? (interviews[key] ?? []) : []
  // The stream belongs to the character and scene it was asked in; elsewhere it shows only as busy.
  const busyInterview = !!gen && gen.action === 'interview'
  const thinking = busyInterview && runKey === key
  const busy = !!gen
  const store = useNovel.getState()
  const voiceBlock = useNovel((s) => (character ? voiceNoteBlock(s, entryPath(character.id)) : null))

  if (!state) return null

  const send = async (): Promise<void> => {
    const q = question.trim()
    if (!q || !character || busy) return
    setQuestion('')
    const ok = await store.askInterview(character.id, q)
    // A question that got no answer goes back into the box, unless something new was typed.
    if (!ok) setQuestion((cur) => (cur ? cur : q))
  }

  if (characters.length === 0) return <div className="p-4 text-xs text-faint">Add a character to the bible to interview them.</div>
  if (!scene) return <div className="p-4 text-xs text-faint">Open a scene first. The character knows what happened before it, and nothing after.</div>

  return (
    <div>
      <PanelSection
        title="Interview"
        action={
          <Button size="sm" variant="ghost" disabled={turns.length === 0 || thinking} onClick={() => store.clearInterview(key)}>
            <Trash2 size={13} /> Clear
          </Button>
        }
      >
        <Select aria-label="Character" value={characterId} onChange={(e) => setPicked(e.target.value)}>
          {characters.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </Select>
        <p className="text-[11px] text-faint">{character?.name} answers as they are before “{scene.title}”. The chat is kept for this character and scene until you close the project.</p>
        {key && (dropped[key] ?? 0) > 0 && <p className="text-[11px] text-warn-text">The earliest {dropped[key]} turns did not fit and were left out of the last request.</p>}
      </PanelSection>

      <PanelSection title="Conversation">
        {turns.length === 0 && !thinking && <p className="text-xs text-faint">Ask a question to start.</p>}
        <ul className="space-y-2">
          {turns.map((t, i) =>
            t.role === 'user' ? (
              <li key={i} className="rounded-md bg-hover px-3 py-2 text-[13px] whitespace-pre-wrap">
                {t.content}
              </li>
            ) : (
              <Answer key={i} text={t.content} name={character?.name ?? ''} block={voiceBlock} onAdd={(note) => void store.addVoiceNote(characterId, note)} />
            )
          )}
          {thinking && (
            <li className="rounded-md border border-border bg-panel-2 px-3 py-2 text-[13px] whitespace-pre-wrap text-muted" role="status">
              {gen?.text || (gen?.phase === 'waiting' && gen.progress !== null ? `Reading the prompt… ${Math.round(gen.progress * 100)}%` : 'Thinking…')}
            </li>
          )}
        </ul>
      </PanelSection>

      <PanelSection title="Ask">
        <Textarea
          rows={3}
          aria-label="Question"
          placeholder={`Ask ${character?.name ?? 'the character'} something…`}
          value={question}
          disabled={busy && !busyInterview}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
              e.preventDefault()
              void send()
            }
          }}
        />
        <div className="flex justify-end gap-2">
          {busyInterview ? (
            <Button size="sm" onClick={() => store.stop()}>
              <Square size={13} /> Stop
            </Button>
          ) : (
            <Button size="sm" variant="primary" disabled={busy || !question.trim()} title="Ctrl+Enter" onClick={() => void send()}>
              <Send size={14} /> Ask
            </Button>
          )}
        </div>
      </PanelSection>
    </div>
  )
}

function Answer({ text, name, block, onAdd }: { text: string; name: string; block: string | null; onAdd: (note: string) => void }) {
  const ref = useRef<HTMLLIElement>(null)
  // The part of the answer the user selected, if any; else all of it.
  const chosen = (): string => {
    const sel = window.getSelection()
    const part = sel && !sel.isCollapsed && ref.current && sel.anchorNode && ref.current.contains(sel.anchorNode) ? sel.toString().trim() : ''
    return part || text
  }
  return (
    <li ref={ref} className="space-y-1.5 rounded-md border border-border bg-panel-2 px-3 py-2 text-[13px]">
      <div className="text-[11px] font-medium text-faint">{name}</div>
      <div className="whitespace-pre-wrap">{text}</div>
      <div className="flex justify-end">
        <Button
          size="sm"
          variant="ghost"
          disabled={!!block}
          title={block ?? 'Add the answer, or the part you selected, to the character’s voice notes'}
          // mousedown would clear the text selection before the click reads it
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => onAdd(chosen())}
        >
          <Mic size={13} /> Add to voice notes
        </Button>
      </div>
    </li>
  )
}
