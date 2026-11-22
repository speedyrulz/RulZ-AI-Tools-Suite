import { ArrowDown, ArrowUp, ChevronRight, Ellipsis, ListPlus, Plus, Sparkles, Trash2, Undo2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import type { NovelChapter, NovelScene, SceneStatus } from '@shared/novel'
import { scenePath, SCENE_STATUSES, addChapterOps, addSceneOps, moveChapterOps, moveSceneOps, renameChapterOps, renameSceneOps, setBeatsOps, setStatusOps, statusLabel } from '../lib/outlineOps'
import { cn, formatNumber } from '../lib/format'
import { startPlanChapters, startPlanScenes, startSummarizeChapter, startSummarizeScene } from '../stores/actions'
import { useProject } from '../stores/project'
import { useUi } from '../stores/ui'
import { Badge, Button, Field, IconButton, Input, Select, Sheet, Textarea } from './ui'

type Menu = { kind: 'scene'; id: string } | { kind: 'chapter'; id: string } | { kind: 'project' } | null
type Ask =
  | { kind: 'rename-scene'; id: string; value: string }
  | { kind: 'rename-chapter'; id: string; value: string }
  | { kind: 'new-chapter'; value: string }
  | { kind: 'new-scene'; chapterId: string; value: string }
  | { kind: 'plan-chapters'; count: number; note: string }
  | { kind: 'plan-scenes'; chapterId: string; count: number; note: string }
  | { kind: 'beats'; id: string; value: string }
  | { kind: 'delete-scene'; id: string }
  | { kind: 'delete-chapter'; id: string }
  | null

const statusTone = (s: SceneStatus): 'neutral' | 'warn' | 'ok' | 'accent' => (s === 'revised' ? 'ok' : s === 'draft' ? 'accent' : s === 'beats' ? 'warn' : 'neutral')

/** The chapters and scenes of the open project: pick a scene to write, and change the outline. */
export function OutlineSheet() {
  const open = useUi((s) => s.outlineOpen)
  const setOpen = useUi((s) => s.setOutlineOpen)
  const sceneId = useUi((s) => s.sceneId)
  const setScene = useUi((s) => s.setScene)
  const state = useProject((s) => s.state)
  const patch = useProject((s) => s.patchOutline)
  const [menu, setMenu] = useState<Menu>(null)
  const [ask, setAsk] = useState<Ask>(null)
  const [trashOpen, setTrashOpen] = useState(false)
  if (!state) return null
  const { outline } = state

  const pick = (id: string): void => {
    setScene(id)
    setOpen(false)
  }
  const run = async (ops: ReturnType<typeof renameSceneOps> | null): Promise<boolean> => (ops ? patch(ops) : false)
  const sceneOf = (id: string): NovelScene | undefined => outline.chapters.flatMap((c) => c.scenes).find((s) => s.id === id)
  const chapterOf = (id: string): NovelChapter | undefined => outline.chapters.find((c) => c.id === id)

  const submit = async (): Promise<void> => {
    if (!ask) return
    const a = ask
    setAsk(null)
    if (a.kind === 'rename-scene') await run(renameSceneOps(a.id, a.value))
    else if (a.kind === 'rename-chapter') await run(renameChapterOps(a.id, a.value))
    else if (a.kind === 'new-chapter') {
      const made = addChapterOps(a.value)
      if (await patch(made.ops)) pick(made.sceneId)
    } else if (a.kind === 'new-scene') {
      const made = addSceneOps(a.chapterId, a.value)
      if (await patch(made.ops)) pick(made.sceneId)
    } else if (a.kind === 'beats') await run(setBeatsOps(a.id, a.value))
    else if (a.kind === 'plan-chapters') await startPlanChapters(a.count, a.note)
    else if (a.kind === 'plan-scenes') await startPlanScenes(a.chapterId, a.count, a.note)
    else if (a.kind === 'delete-scene') {
      if (await useProject.getState().deleteScene(a.id)) toast.success('Moved to the trash.')
    } else if (a.kind === 'delete-chapter') {
      if (await useProject.getState().deleteChapter(a.id)) toast.success('Moved to the trash.')
    }
  }

  const menuScene = menu?.kind === 'scene' ? sceneOf(menu.id) : undefined
  const menuChapter = menu?.kind === 'chapter' ? chapterOf(menu.id) : undefined

  return (
    <>
      <Sheet
        full
        open={open}
        onOpenChange={setOpen}
        title="Chapters and scenes"
        description={state.project.title}
      >
        <div className="space-y-4 px-3 pb-6">
          <div className="flex flex-wrap gap-2 px-1">
            <Button size="sm" onClick={() => setAsk({ kind: 'new-chapter', value: '' })}>
              <Plus size={15} /> Chapter
            </Button>
            <Button size="sm" onClick={() => setAsk({ kind: 'plan-chapters', count: 5, note: '' })}>
              <Sparkles size={15} /> Plan chapters
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setTrashOpen(true)}>
              <Trash2 size={15} /> Trash
            </Button>
          </div>
          {state.blocked && <p className="rounded-lg border border-danger-border bg-danger-bg p-3 text-[14px] text-danger-text">The computer couldn’t read this project’s files ({state.blocked}). It is read-only until that is fixed there.</p>}
          {outline.chapters.length === 0 && <p className="px-2 py-8 text-center text-[14px] text-faint">No chapters yet. Add one, or plan them with the AI.</p>}
          {outline.chapters.map((chapter, ci) => (
            <section key={chapter.id} className="rounded-2xl border border-border bg-panel p-1.5">
              <div className="flex items-center">
                <button className="min-w-0 flex-1 rounded-xl px-3 py-2.5 text-left active:bg-hover" onClick={() => setMenu({ kind: 'chapter', id: chapter.id })}>
                  <div className="truncate text-[16px] font-medium">
                    {ci + 1}. {chapter.title}
                  </div>
                  {(chapter.act || chapter.summary) && <div className="truncate text-[12.5px] text-faint">{[chapter.act, chapter.summary].filter(Boolean).join(' · ')}</div>}
                </button>
                <IconButton label={`Options for ${chapter.title}`} onClick={() => setMenu({ kind: 'chapter', id: chapter.id })}>
                  <Ellipsis size={18} />
                </IconButton>
              </div>
              {chapter.scenes.map((scene) => (
                <div key={scene.id} className={cn('flex items-center rounded-xl', scene.id === sceneId && 'bg-accent-soft')}>
                  <button className="flex min-w-0 flex-1 items-center gap-2 px-3 py-2.5 text-left active:bg-hover" onClick={() => pick(scene.id)}>
                    <ChevronRight size={15} className="shrink-0 text-faint" />
                    <span className="min-w-0 flex-1 truncate text-[15px]">{scene.title}</span>
                    <span className="shrink-0 text-[12px] text-faint">{formatNumber(state.words[scenePath(scene)] ?? 0)} w</span>
                    <Badge tone={statusTone(scene.status)}>{statusLabel(scene.status)}</Badge>
                  </button>
                  <IconButton label={`Options for ${scene.title}`} onClick={() => setMenu({ kind: 'scene', id: scene.id })}>
                    <Ellipsis size={18} />
                  </IconButton>
                </div>
              ))}
              <div className="px-1 pt-0.5 pb-1">
                <Button size="sm" variant="ghost" onClick={() => setAsk({ kind: 'new-scene', chapterId: chapter.id, value: '' })}>
                  <ListPlus size={15} /> Add a scene
                </Button>
              </div>
            </section>
          ))}
          {state.missing.length > 0 && <p className="px-2 text-[12.5px] text-warn">Missing on the computer: {state.missing.join(', ')}.</p>}
          {state.orphans.length > 0 && <p className="px-2 text-[12.5px] text-faint">Not in the outline: {state.orphans.join(', ')}. Add them on the computer.</p>}
        </div>
      </Sheet>

      <Sheet open={!!menuScene} onOpenChange={(o) => !o && setMenu(null)} title={menuScene?.title ?? ''} description="Scene">
        {menuScene && menu?.kind === 'scene' && (
          <div className="space-y-4 px-4 pb-5">
            <Field label="Status">
              <Select value={menuScene.status} onChange={(e) => void run(setStatusOps(menuScene.id, e.target.value as SceneStatus))}>
                {SCENE_STATUSES.map((s) => (
                  <option key={s.value} value={s.value}>
                    {s.label}
                  </option>
                ))}
              </Select>
            </Field>
            {menuScene.summary && <p className="text-[13.5px] text-muted">{menuScene.summary}</p>}
            <div className="grid grid-cols-2 gap-2">
              <Button onClick={() => (setMenu(null), setAsk({ kind: 'rename-scene', id: menuScene.id, value: menuScene.title }))}>Rename</Button>
              <Button onClick={() => (setMenu(null), setAsk({ kind: 'beats', id: menuScene.id, value: menuScene.beats }))}>Edit beats</Button>
              <Button disabled={!moveSceneOps(outline, menuScene.id, -1)} onClick={() => void run(moveSceneOps(outline, menuScene.id, -1))}>
                <ArrowUp size={16} /> Move up
              </Button>
              <Button disabled={!moveSceneOps(outline, menuScene.id, 1)} onClick={() => void run(moveSceneOps(outline, menuScene.id, 1))}>
                <ArrowDown size={16} /> Move down
              </Button>
              <Button onClick={() => (setMenu(null), void startSummarizeScene(menuScene.id))}>
                <Sparkles size={16} /> Summarize
              </Button>
              <Button variant="danger" onClick={() => (setMenu(null), setAsk({ kind: 'delete-scene', id: menuScene.id }))}>
                <Trash2 size={16} /> Delete
              </Button>
            </div>
          </div>
        )}
      </Sheet>

      <Sheet open={!!menuChapter} onOpenChange={(o) => !o && setMenu(null)} title={menuChapter?.title ?? ''} description="Chapter">
        {menuChapter && (
          <div className="space-y-4 px-4 pb-5">
            {menuChapter.summary && <p className="text-[13.5px] text-muted">{menuChapter.summary}</p>}
            <div className="grid grid-cols-2 gap-2">
              <Button onClick={() => (setMenu(null), setAsk({ kind: 'rename-chapter', id: menuChapter.id, value: menuChapter.title }))}>Rename</Button>
              <Button onClick={() => (setMenu(null), setAsk({ kind: 'new-scene', chapterId: menuChapter.id, value: '' }))}>
                <Plus size={16} /> Add a scene
              </Button>
              <Button disabled={!moveChapterOps(outline, menuChapter.id, -1)} onClick={() => void run(moveChapterOps(outline, menuChapter.id, -1))}>
                <ArrowUp size={16} /> Move up
              </Button>
              <Button disabled={!moveChapterOps(outline, menuChapter.id, 1)} onClick={() => void run(moveChapterOps(outline, menuChapter.id, 1))}>
                <ArrowDown size={16} /> Move down
              </Button>
              <Button onClick={() => (setMenu(null), void startSummarizeChapter(menuChapter.id))}>
                <Sparkles size={16} /> Summarize
              </Button>
              <Button onClick={() => (setMenu(null), setAsk({ kind: 'plan-scenes', chapterId: menuChapter.id, count: 4, note: '' }))}>
                <Sparkles size={16} /> Plan scenes
              </Button>
              <Button variant="danger" className="col-span-2" onClick={() => (setMenu(null), setAsk({ kind: 'delete-chapter', id: menuChapter.id }))}>
                <Trash2 size={16} /> Delete chapter
              </Button>
            </div>
          </div>
        )}
      </Sheet>

      <AskSheet ask={ask} setAsk={setAsk} onSubmit={() => void submit()} sceneTitle={(id) => sceneOf(id)?.title ?? 'this scene'} chapterTitle={(id) => chapterOf(id)?.title ?? 'this chapter'} sceneCount={(id) => chapterOf(id)?.scenes.length ?? 0} />
      <TrashSheet open={trashOpen} onOpenChange={setTrashOpen} />
    </>
  )
}

function AskSheet({
  ask,
  setAsk,
  onSubmit,
  sceneTitle,
  chapterTitle,
  sceneCount
}: {
  ask: Ask
  setAsk: (a: Ask) => void
  onSubmit: () => void
  sceneTitle: (id: string) => string
  chapterTitle: (id: string) => string
  sceneCount: (id: string) => number
}) {
  const title =
    ask?.kind === 'rename-scene'
      ? 'Rename scene'
      : ask?.kind === 'rename-chapter'
        ? 'Rename chapter'
        : ask?.kind === 'new-chapter'
          ? 'New chapter'
          : ask?.kind === 'new-scene'
            ? 'New scene'
            : ask?.kind === 'beats'
              ? 'Beats'
              : ask?.kind === 'plan-chapters'
                ? 'Plan chapters'
                : ask?.kind === 'plan-scenes'
                  ? `Plan scenes for “${chapterTitle(ask.chapterId)}”`
                  : ask?.kind === 'delete-scene'
                    ? `Delete “${sceneTitle(ask.id)}”?`
                    : ask?.kind === 'delete-chapter'
                      ? `Delete “${chapterTitle(ask.id)}”?`
                      : ''
  const destructive = ask?.kind === 'delete-scene' || ask?.kind === 'delete-chapter'
  return (
    <Sheet
      open={!!ask}
      onOpenChange={(o) => !o && setAsk(null)}
      title={title}
      footer={
        <>
          <Button onClick={() => setAsk(null)}>Cancel</Button>
          <Button variant={destructive ? 'danger' : 'primary'} onClick={onSubmit}>
            {destructive ? 'Move to trash' : ask?.kind === 'plan-chapters' || ask?.kind === 'plan-scenes' ? 'Ask the AI' : 'Done'}
          </Button>
        </>
      }
    >
      <div className="space-y-4 px-4 pb-4">
        {(ask?.kind === 'rename-scene' || ask?.kind === 'rename-chapter' || ask?.kind === 'new-chapter' || ask?.kind === 'new-scene') && (
          <Input autoFocus value={ask.value} onChange={(e) => setAsk({ ...ask, value: e.target.value })} onKeyDown={(e) => e.key === 'Enter' && onSubmit()} placeholder="Title" />
        )}
        {ask?.kind === 'beats' && <Textarea autoFocus rows={8} value={ask.value} onChange={(e) => setAsk({ ...ask, value: e.target.value })} placeholder="What happens in this scene" />}
        {(ask?.kind === 'plan-chapters' || ask?.kind === 'plan-scenes') && (
          <>
            <Field label={ask.kind === 'plan-chapters' ? 'How many chapters' : 'How many scenes'}>
              <Input type="number" inputMode="numeric" min={1} max={30} value={ask.count} onChange={(e) => setAsk({ ...ask, count: Math.max(1, Math.min(30, Math.round(Number(e.target.value)) || 1)) })} />
            </Field>
            <Field label="Note" hint="Optional. Say what you want, for example “end on a cliffhanger”.">
              <Textarea rows={3} value={ask.note} onChange={(e) => setAsk({ ...ask, note: e.target.value })} />
            </Field>
          </>
        )}
        {ask?.kind === 'delete-scene' && <p className="text-[15px] text-muted">The scene’s text goes to this project’s trash on the computer. You can restore it from Trash. A chapter always keeps at least one scene.</p>}
        {ask?.kind === 'delete-chapter' && <p className="text-[15px] text-muted">The chapter and its {sceneCount(ask.id)} scenes go to this project’s trash on the computer. You can restore them from Trash.</p>}
      </div>
    </Sheet>
  )
}

function TrashSheet({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const trash = useProject((s) => s.trash)
  const load = useProject((s) => s.loadTrash)
  const restore = useProject((s) => s.restore)
  useEffect(() => {
    if (open) void load()
  }, [open, load])
  return (
    <Sheet open={open} onOpenChange={onOpenChange} title="Trash" description="Deleted scenes and chapters. The computer keeps them until the trash is emptied there.">
      <div className="space-y-2 px-4 pb-5">
        {trash.length === 0 && <p className="py-6 text-center text-[14px] text-faint">The trash is empty.</p>}
        {trash.map((item) => (
          <div key={item.id} className="flex items-center gap-2 rounded-xl border border-border bg-panel-2 p-3">
            <div className="min-w-0 flex-1">
              <div className="truncate text-[15px]">{item.label}</div>
              <div className="text-[12px] text-faint">{new Date(item.deletedAt).toLocaleString()}</div>
            </div>
            <Button size="sm" onClick={() => void restore(item.id).then((ok) => ok && toast.success('Restored.'))}>
              <Undo2 size={15} /> Restore
            </Button>
          </div>
        ))}
      </div>
    </Sheet>
  )
}
