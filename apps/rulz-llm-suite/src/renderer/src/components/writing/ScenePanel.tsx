import { ListTree, Sparkles } from 'lucide-react'
import { useState } from 'react'
import type { OutlineOp, SceneStatus } from '@shared/novel'
import { sceneHashKey } from '@shared/novelText'
import { scenePath } from '../../lib/novelRun'
import { sceneOf, summaryState, useNovel } from '../../stores/novel'
import { Button, Field, NumberInput, Select } from '../ui'
import { DraftArea, DraftInput, PanelSection, STATUS_LABELS } from './common'

export function ScenePanel() {
  const target = useNovel((s) => s.target)
  const state = useNovel((s) => s.state)
  const hashes = useNovel((s) => s.hashes)
  const busy = useNovel((s) => !!s.gen || !!s.pending)
  const blocked = !!state?.blocked
  const found = target?.kind === 'scene' ? sceneOf(state, target.id) : null
  const [planCount, setPlanCount] = useState<number | null>(5)
  if (!state || !found) {
    return <div className="p-4 text-xs text-faint">Open a scene to edit its title, beats, characters and summary.</div>
  }
  const { scene, chapterId } = found
  const chapter = state.outline.chapters.find((c) => c.id === chapterId)!
  const store = useNovel.getState()
  const characters = state.bible.filter((e) => e.kind === 'character').sort((a, b) => a.name.localeCompare(b.name))
  const known = new Set(state.bible.map((e) => e.id))
  const sum = summaryState(scene, hashes[scenePath(scene)])
  const written = (state.words[scenePath(scene)] ?? 0) > 0
  const chapterHashes = chapter.scenes.map((s) => hashes[scenePath(s)])
  const chapterKnown = chapterHashes.every((h) => h !== undefined)
  const chapterSum = !chapter.summary.trim() ? 'missing' : chapterKnown && chapter.summaryOf !== sceneHashKey(chapterHashes as string[]) ? 'stale' : 'current'

  return (
    <div className={blocked ? 'pointer-events-none opacity-60' : undefined}>
      <PanelSection title="Scene">
        <Field label="Title">
          <DraftInput value={scene.title} onCommit={(v) => v.trim() && void store.setScene(scene.id, 'title', v.trim())} />
        </Field>
        <Field label="Beats" hint="The plan the model follows">
          <DraftArea value={scene.beats} rows={6} placeholder="What happens in this scene, in a few lines." onCommit={(v) => void store.setScene(scene.id, 'beats', v)} />
        </Field>
        <Field label="Point of view">
          <Select value={scene.pov ?? ''} aria-label="Point of view" onChange={(e) => void store.setScene(scene.id, 'pov', e.target.value || null)}>
            <option value="">No one in particular</option>
            {scene.pov && !known.has(scene.pov) && <option value={scene.pov}>{scene.pov} (no such entry)</option>}
            {characters.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Present" hint="Characters in the scene">
          {characters.length === 0 ? (
            <p className="text-[11px] text-faint">Add character entries to the story bible to pick them here.</p>
          ) : (
            <div className="max-h-36 space-y-1 overflow-y-auto rounded-lg border border-border bg-panel p-2">
              {characters.map((c) => (
                <label key={c.id} className="flex items-center gap-2 text-[13px]">
                  <input
                    type="checkbox"
                    checked={scene.present.includes(c.id)}
                    onChange={(e) => void store.setScene(scene.id, 'present', e.target.checked ? [...scene.present, c.id] : scene.present.filter((id) => id !== c.id))}
                  />
                  {c.name}
                </label>
              ))}
            </div>
          )}
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Target words">
            <NumberInput className="w-full" value={scene.targetWords} placeholder="None" min={0} step={100} onChange={(v) => void store.setScene(scene.id, 'targetWords', v === null || v <= 0 ? null : Math.round(v))} />
          </Field>
          <Field label="Status">
            <Select value={scene.status} aria-label="Status" onChange={(e) => void store.setScene(scene.id, 'status', e.target.value as SceneStatus)}>
              {(Object.keys(STATUS_LABELS) as SceneStatus[]).map((s) => (
                <option key={s} value={s}>
                  {STATUS_LABELS[s]}
                </option>
              ))}
            </Select>
          </Field>
        </div>
      </PanelSection>

      <PanelSection
        title="Scene summary"
        action={
          <Button size="sm" disabled={busy || !written} title={written ? 'Ask the model for a summary of this scene' : 'The scene has no text yet'} onClick={() => void store.summarizeScene(scene.id)}>
            <Sparkles size={14} /> Summarize
          </Button>
        }
      >
        <p className="text-[11px] text-faint" role="status">
          {sum === 'missing' && (written ? 'No summary yet. Until there is one, later requests use this scene’s beats.' : 'No summary yet.')}
          {sum === 'stale' && 'Out of date: the scene changed after this was written. Later requests use the beats instead.'}
          {sum === 'current' && 'Current: written from the text as it is now.'}
        </p>
        <DraftArea
          value={scene.summary}
          rows={5}
          placeholder="Summary of what happens, used as “story so far” in later requests."
          onCommit={(v) => {
            if (v === scene.summary) return
            // A summary the author wrote or edited stands for the text as it is now.
            void store.patch([
              { op: 'setScene', sceneId: scene.id, field: 'summary', value: v },
              { op: 'setScene', sceneId: scene.id, field: 'summaryOf', value: hashes[scenePath(scene)] ?? scene.summaryOf }
            ])
          }}
        />
      </PanelSection>

      <PanelSection title={`Chapter: ${chapter.title}`}>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Title">
            <DraftInput value={chapter.title} onCommit={(v) => v.trim() && void store.patch([{ op: 'setChapter', chapterId: chapter.id, field: 'title', value: v.trim() }])} />
          </Field>
          <Field label="Act">
            <DraftInput value={chapter.act} placeholder="Act I" onCommit={(v) => void store.patch([{ op: 'setChapter', chapterId: chapter.id, field: 'act', value: v.trim() }])} />
          </Field>
        </div>
        <Field label="Chapter summary" hint="Used as “story so far” for later chapters">
          <DraftArea
            value={chapter.summary}
            rows={4}
            onCommit={(v) => {
              if (v === chapter.summary) return
              const ops: OutlineOp[] = [{ op: 'setChapter', chapterId: chapter.id, field: 'summary', value: v }]
              if (chapterKnown) ops.push({ op: 'setChapter', chapterId: chapter.id, field: 'summaryOf', value: sceneHashKey(chapterHashes as string[]) })
              void store.patch(ops)
            }}
          />
        </Field>
        <p className="text-[11px] text-faint" role="status">
          {chapterSum === 'missing' && 'No chapter summary yet. Later chapters use this chapter’s scene summaries or beats instead.'}
          {chapterSum === 'stale' && 'Out of date: a scene changed after this was written.'}
          {chapterSum === 'current' && 'Current.'}
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" disabled={busy} onClick={() => void store.summarizeChapter(chapter.id)}>
            <Sparkles size={14} /> Summarize chapter
          </Button>
          <div className="flex-1" />
          <NumberInput className="w-14" value={planCount} min={1} max={20} step={1} onChange={(v) => setPlanCount(v === null ? null : Math.min(20, Math.max(1, Math.round(v))))} />
          <Button size="sm" disabled={busy || !planCount} title="Ask the model to plan scenes for this chapter" onClick={() => void store.planScenes(chapter.id, planCount ?? 5, '')}>
            <ListTree size={14} /> Plan scenes
          </Button>
        </div>
      </PanelSection>
    </div>
  )
}
