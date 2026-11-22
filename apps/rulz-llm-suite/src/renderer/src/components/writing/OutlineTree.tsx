import { AlertTriangle, ArrowDown, ArrowUp, ChevronRight, Ellipsis, FilePlus2, FolderInput, GripVertical, Pencil, Plus, ScanSearch, Trash2 } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import type { NovelChapter, NovelScene } from '@shared/novel'
import { cn, formatNumber } from '../../lib/format'
import { scenePath } from '../../lib/novelRun'
import { liveWords, useNovel } from '../../stores/novel'
import { Button, DropdownMenu, IconButton, Input, Tooltip, type MenuItem } from '../ui'
import { ConfirmDialog, NameDialog, STATUS_LABELS, StatusIcon } from './common'

const SCENE_MIME = 'application/x-rulz-scene'
const CHAPTER_MIME = 'application/x-rulz-chapter'

type Renaming = { kind: 'scene' | 'chapter'; id: string } | null
type Deleting = { kind: 'scene' | 'chapter'; id: string; label: string; detail: string } | null

export function OutlineTree() {
  const outline = useNovel((s) => s.state?.outline)
  const orphans = useNovel((s) => s.state?.orphans ?? [])
  const blocked = useNovel((s) => !!s.state?.blocked)
  const target = useNovel((s) => s.target)
  const checkRun = useNovel((s) => s.checkRun)
  const pending = useNovel((s) => s.pending)
  const checking = useNovel((s) => !!s.gen || !!s.checkRun)
  const [renaming, setRenaming] = useState<Renaming>(null)
  const [deleting, setDeleting] = useState<Deleting>(null)
  const [adding, setAdding] = useState<{ chapterId: string | null } | null>(null)
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())
  const [dropAt, setDropAt] = useState<string | null>(null)
  const store = useNovel.getState()
  if (!outline) return null

  const toggle = (id: string): void =>
    setCollapsed((c) => {
      const n = new Set(c)
      if (n.has(id)) n.delete(id)
      else n.add(id)
      return n
    })

  // Moves use the position in the destination after the item has left its old place.
  const moveSceneTo = (sceneId: string, toChapterId: string, beforeSceneId: string | null): void => {
    const dest = outline.chapters.find((c) => c.id === toChapterId)
    if (!dest || sceneId === beforeSceneId) return
    const rest = dest.scenes.filter((s) => s.id !== sceneId)
    const index = beforeSceneId ? rest.findIndex((s) => s.id === beforeSceneId) : rest.length
    void store.patch([{ op: 'moveScene', sceneId, toChapterId, index: index < 0 ? rest.length : index }])
  }
  const moveChapterTo = (chapterId: string, beforeChapterId: string | null): void => {
    if (chapterId === beforeChapterId) return
    const rest = outline.chapters.filter((c) => c.id !== chapterId)
    const index = beforeChapterId ? rest.findIndex((c) => c.id === beforeChapterId) : rest.length
    void store.patch([{ op: 'moveChapter', chapterId, index: index < 0 ? rest.length : index }])
  }
  const nudgeScene = (chapter: NovelChapter, ci: number, si: number, dir: -1 | 1): void => {
    const scene = chapter.scenes[si]
    if (dir === -1) {
      if (si > 0) void store.patch([{ op: 'moveScene', sceneId: scene.id, toChapterId: chapter.id, index: si - 1 }])
      else if (ci > 0) void store.patch([{ op: 'moveScene', sceneId: scene.id, toChapterId: outline.chapters[ci - 1].id, index: outline.chapters[ci - 1].scenes.length }])
    } else if (si < chapter.scenes.length - 1) void store.patch([{ op: 'moveScene', sceneId: scene.id, toChapterId: chapter.id, index: si + 1 }])
    else if (ci < outline.chapters.length - 1) void store.patch([{ op: 'moveScene', sceneId: scene.id, toChapterId: outline.chapters[ci + 1].id, index: 0 }])
  }

  const renameField = (value: string, commit: (v: string) => void): ReactNode => (
    <RenameInput value={value} onCommit={(v) => { commit(v); setRenaming(null) }} onCancel={() => setRenaming(null)} />
  )

  return (
    <div className="min-h-0">
      <div className="flex items-center justify-between px-3 pt-3 pb-1">
        <h2 className="text-[11px] font-semibold tracking-wide text-faint uppercase">Outline</h2>
        <IconButton size="sm" label="Add chapter" disabled={blocked} onClick={() => setAdding({ chapterId: null })}>
          <Plus size={14} />
        </IconButton>
      </div>
      <div className="space-y-0.5 px-2 pb-2" role="tree" aria-label="Outline">
        {outline.chapters.length === 0 && <div className="px-2 py-3 text-xs text-faint">No chapters yet.</div>}
        {outline.chapters.map((chapter, ci) => {
          const open = !collapsed.has(chapter.id)
          const chapterItems: (MenuItem | 'sep')[] = [
            { label: 'Rename', icon: <Pencil size={13} />, onSelect: () => setRenaming({ kind: 'chapter', id: chapter.id }) },
            { label: 'Add scene', icon: <FilePlus2 size={13} />, onSelect: () => setAdding({ chapterId: chapter.id }) },
            { label: 'Check chapter', icon: <ScanSearch size={13} />, disabled: checking || !!pending, onSelect: () => void store.checkChapter(chapter.id) },
            { label: 'Move up', icon: <ArrowUp size={13} />, disabled: ci === 0, onSelect: () => void store.patch([{ op: 'moveChapter', chapterId: chapter.id, index: ci - 1 }]) },
            { label: 'Move down', icon: <ArrowDown size={13} />, disabled: ci === outline.chapters.length - 1, onSelect: () => void store.patch([{ op: 'moveChapter', chapterId: chapter.id, index: ci + 1 }]) },
            'sep',
            {
              label: 'Delete chapter…',
              icon: <Trash2 size={13} />,
              danger: true,
              disabled: outline.chapters.length <= 1,
              onSelect: () => setDeleting({ kind: 'chapter', id: chapter.id, label: chapter.title, detail: `${chapter.scenes.length} ${chapter.scenes.length === 1 ? 'scene' : 'scenes'}` })
            }
          ]
          return (
            <div key={chapter.id} role="treeitem" aria-expanded={open}>
              <div
                draggable={!blocked && renaming?.id !== chapter.id}
                onDragStart={(e) => {
                  e.dataTransfer.setData(CHAPTER_MIME, chapter.id)
                  e.dataTransfer.effectAllowed = 'move'
                }}
                onDragOver={(e) => {
                  if (e.dataTransfer.types.includes(CHAPTER_MIME) || e.dataTransfer.types.includes(SCENE_MIME)) {
                    e.preventDefault()
                    setDropAt(`c:${chapter.id}`)
                  }
                }}
                onDragLeave={() => setDropAt((d) => (d === `c:${chapter.id}` ? null : d))}
                onDrop={(e) => {
                  e.preventDefault()
                  setDropAt(null)
                  const sid = e.dataTransfer.getData(SCENE_MIME)
                  const cid = e.dataTransfer.getData(CHAPTER_MIME)
                  if (sid) moveSceneTo(sid, chapter.id, null)
                  else if (cid) moveChapterTo(cid, chapter.id)
                }}
                className={cn('group flex items-center rounded-md', dropAt === `c:${chapter.id}` && 'bg-accent-soft ring-1 ring-accent/50')}
              >
                {renaming?.kind === 'chapter' && renaming.id === chapter.id ? (
                  <div className="flex min-w-0 flex-1 items-center gap-1 px-1.5 py-1">
                    <ChevronRight size={13} className={cn('shrink-0 text-faint', open && 'rotate-90')} />
                    {renameField(chapter.title, (v) => void store.patch([{ op: 'setChapter', chapterId: chapter.id, field: 'title', value: v }]))}
                  </div>
                ) : (
                  <button
                    className="flex min-w-0 flex-1 items-center gap-1 rounded-md px-1.5 py-1.5 text-left text-[13px] font-medium hover:bg-hover"
                    onClick={() => toggle(chapter.id)}
                    aria-label={`${open ? 'Collapse' : 'Expand'} ${chapter.title}`}
                  >
                    <ChevronRight size={13} className={cn('shrink-0 text-faint transition-transform', open && 'rotate-90')} />
                    <span className="truncate">{chapter.title}</span>
                    {chapter.act && <span className="shrink-0 text-[11px] font-normal text-faint">{chapter.act}</span>}
                  </button>
                )}
                {checkRun?.chapterId === chapter.id && (
                  <span className="flex shrink-0 items-center gap-1 pr-1 text-[11px] text-muted" role="status">
                    Checking {checkRun.index + 1} of {checkRun.sceneIds.length}
                    <button className="rounded px-1 text-[11px] text-link hover:bg-hover" onClick={() => store.stop()}>
                      Stop
                    </button>
                  </span>
                )}
                <ChapterWords chapter={chapter} />
                <DropdownMenu
                  trigger={
                    <button className="rounded p-1 text-faint opacity-0 group-hover:opacity-100 hover:bg-hover hover:text-text focus-visible:opacity-100 data-[state=open]:opacity-100" aria-label={`Options for ${chapter.title}`} disabled={blocked}>
                      <Ellipsis size={14} />
                    </button>
                  }
                  items={chapterItems}
                />
              </div>
              {open && (
                <div className="ml-3 space-y-px border-l border-border pl-1">
                  {chapter.scenes.map((scene, si) => (
                    <SceneRow
                      key={scene.id}
                      scene={scene}
                      selected={target?.kind === 'scene' && target.id === scene.id}
                      renaming={renaming?.kind === 'scene' && renaming.id === scene.id}
                      dropHere={dropAt === `s:${scene.id}`}
                      blocked={blocked}
                      onOpen={() => void store.selectScene(scene.id)}
                      onRename={(v) => {
                        void store.setScene(scene.id, 'title', v)
                        setRenaming(null)
                      }}
                      onCancelRename={() => setRenaming(null)}
                      onDragOver={() => setDropAt(`s:${scene.id}`)}
                      onDragLeave={() => setDropAt((d) => (d === `s:${scene.id}` ? null : d))}
                      onDropScene={(sid) => {
                        setDropAt(null)
                        moveSceneTo(sid, chapter.id, scene.id)
                      }}
                      menu={[
                        { label: 'Rename', icon: <Pencil size={13} />, onSelect: () => setRenaming({ kind: 'scene', id: scene.id }) },
                        { label: 'Move up', icon: <ArrowUp size={13} />, disabled: ci === 0 && si === 0, onSelect: () => nudgeScene(chapter, ci, si, -1) },
                        { label: 'Move down', icon: <ArrowDown size={13} />, disabled: ci === outline.chapters.length - 1 && si === chapter.scenes.length - 1, onSelect: () => nudgeScene(chapter, ci, si, 1) },
                        ...(outline.chapters.length > 1
                          ? [
                              'sep' as const,
                              ...outline.chapters
                                .filter((c) => c.id !== chapter.id)
                                .map((c) => ({ label: `Move to ${c.title}`, icon: <FolderInput size={13} />, onSelect: () => moveSceneTo(scene.id, c.id, null) }))
                            ]
                          : []),
                        'sep' as const,
                        {
                          label: 'Delete scene…',
                          icon: <Trash2 size={13} />,
                          danger: true,
                          disabled: chapter.scenes.length <= 1,
                          onSelect: () => setDeleting({ kind: 'scene', id: scene.id, label: scene.title, detail: 'its text file' })
                        }
                      ]}
                    />
                  ))}
                  <button className="flex w-full items-center gap-1.5 rounded-md px-2 py-1 text-left text-xs text-faint hover:bg-hover hover:text-text disabled:opacity-40" disabled={blocked} onClick={() => setAdding({ chapterId: chapter.id })}>
                    <Plus size={12} /> Add scene
                  </button>
                </div>
              )}
            </div>
          )
        })}
        {orphans.length > 0 && (
          <div className="mt-3 border-t border-border pt-2">
            <div className="px-1.5 pb-1 text-[11px] font-semibold tracking-wide text-faint uppercase">Not in outline</div>
            {orphans.map((file) => (
              <div key={file} className="flex items-center gap-1 rounded-md px-1.5 py-1 text-[13px]">
                <span className="min-w-0 flex-1 truncate font-mono text-xs text-muted" title={file}>
                  {file.replace(/^manuscript\//, '')}
                </span>
                <DropdownMenu
                  trigger={
                    <Button size="sm" variant="ghost" disabled={blocked || outline.chapters.length === 0}>
                      Add to outline
                    </Button>
                  }
                  items={outline.chapters.map((c) => ({ label: `Add to ${c.title}`, onSelect: () => void store.adoptFile(file.replace(/^manuscript\//, ''), c.id) }))}
                />
              </div>
            ))}
          </div>
        )}
      </div>
      <NameDialog
        open={!!adding}
        onOpenChange={(o) => !o && setAdding(null)}
        title={adding?.chapterId ? 'Add scene' : 'Add chapter'}
        label="Title"
        confirm="Add"
        onSubmit={(title) => {
          if (adding?.chapterId) void store.addScene(adding.chapterId, title)
          else void store.addChapter(title)
        }}
      />
      <ConfirmDialog
        open={!!deleting}
        onOpenChange={(o) => !o && setDeleting(null)}
        title={deleting?.kind === 'chapter' ? 'Delete this chapter?' : 'Delete this scene?'}
        body={
          <>
            <p>
              <strong>{deleting?.label}</strong> ({deleting?.detail}) moves to the project trash. You can bring it back from Trash in the project menu.
            </p>
          </>
        }
        confirm="Delete"
        onConfirm={() => {
          if (!deleting) return
          if (deleting.kind === 'scene') void store.deleteScene(deleting.id)
          else void store.deleteChapter(deleting.id)
        }}
      />
    </div>
  )
}

function ChapterWords({ chapter }: { chapter: NovelChapter }) {
  const words = useNovel((s) => chapter.scenes.reduce((n, sc) => n + liveWords(s, scenePath(sc)), 0))
  return <span className="shrink-0 pr-1 text-[11px] text-faint tabular-nums">{formatNumber(words)}</span>
}

function RenameInput({ value, onCommit, onCancel }: { value: string; onCommit: (v: string) => void; onCancel: () => void }) {
  const [v, setV] = useState(value)
  return (
    <Input
      autoFocus
      className="h-7 flex-1"
      aria-label="New title"
      value={v}
      onClick={(e) => e.stopPropagation()}
      onChange={(e) => setV(e.target.value)}
      onBlur={() => (v.trim() && v.trim() !== value ? onCommit(v.trim()) : onCancel())}
      onKeyDown={(e) => {
        e.stopPropagation()
        if (e.key === 'Enter') (v.trim() && v.trim() !== value ? onCommit(v.trim()) : onCancel())
        else if (e.key === 'Escape') onCancel()
      }}
    />
  )
}

function SceneRow({
  scene,
  selected,
  renaming,
  dropHere,
  blocked,
  menu,
  onOpen,
  onRename,
  onCancelRename,
  onDragOver,
  onDragLeave,
  onDropScene
}: {
  scene: NovelScene
  selected: boolean
  renaming: boolean
  dropHere: boolean
  blocked: boolean
  menu: (MenuItem | 'sep')[]
  onOpen: () => void
  onRename: (v: string) => void
  onCancelRename: () => void
  onDragOver: () => void
  onDragLeave: () => void
  onDropScene: (sceneId: string) => void
}) {
  const path = scenePath(scene)
  const words = useNovel((s) => liveWords(s, path))
  const missing = useNovel((s) => s.state?.missing.includes(path) ?? false)
  const dirty = useNovel((s) => !!s.buffers[path]?.dirty)
  return (
    <div
      role="treeitem"
      aria-selected={selected}
      draggable={!blocked && !renaming}
      onDragStart={(e) => {
        e.dataTransfer.setData(SCENE_MIME, scene.id)
        e.dataTransfer.effectAllowed = 'move'
      }}
      onDragOver={(e) => {
        if (e.dataTransfer.types.includes(SCENE_MIME)) {
          e.preventDefault()
          e.stopPropagation()
          onDragOver()
        }
      }}
      onDragLeave={onDragLeave}
      onDrop={(e) => {
        const sid = e.dataTransfer.getData(SCENE_MIME)
        if (sid) {
          e.preventDefault()
          e.stopPropagation()
          onDropScene(sid)
        }
      }}
      className={cn('group flex items-center rounded-md', selected ? 'bg-hover text-text' : 'text-body hover:bg-hover', dropHere && 'ring-1 ring-accent/60')}
    >
      <GripVertical size={12} className="shrink-0 cursor-grab text-faint opacity-0 group-hover:opacity-60" aria-hidden />
      {renaming ? (
        <div className="flex min-w-0 flex-1 items-center gap-1.5 py-1 pr-1">
          <StatusIcon status={scene.status} className="shrink-0" />
          <RenameInput value={scene.title} onCommit={onRename} onCancel={onCancelRename} />
        </div>
      ) : (
        <button className="flex min-w-0 flex-1 items-center gap-1.5 py-1.5 pr-1 text-left text-[13px]" onClick={onOpen}>
          <Tooltip content={STATUS_LABELS[scene.status]}>
            <span className="shrink-0" aria-label={`Status: ${STATUS_LABELS[scene.status]}`}>
              <StatusIcon status={scene.status} />
            </span>
          </Tooltip>
          <span className="min-w-0 flex-1 truncate">{scene.title}</span>
          {missing && (
            <span className="flex shrink-0 items-center gap-1 text-[11px] text-danger-text">
              <AlertTriangle size={12} /> File missing
            </span>
          )}
          {dirty && <span className="shrink-0 text-[11px] text-faint">edited</span>}
        </button>
      )}
      <span className="shrink-0 pr-1 text-[11px] text-faint tabular-nums">{formatNumber(words)}</span>
      <DropdownMenu
        trigger={
          <button className="rounded p-1 text-faint opacity-0 group-hover:opacity-100 hover:bg-hover hover:text-text focus-visible:opacity-100 data-[state=open]:opacity-100" aria-label={`Options for ${scene.title}`} disabled={blocked}>
            <Ellipsis size={14} />
          </button>
        }
        items={menu}
      />
    </div>
  )
}
