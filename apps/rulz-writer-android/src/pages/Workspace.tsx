import { ArrowLeft, BookMarked, ListTree, MessagesSquare, PenLine, ScanSearch, Settings as SettingsIcon, SquareStack } from 'lucide-react'
import { useEffect } from 'react'
import { BibleTab } from '../components/BibleTab'
import { CheckTab } from '../components/CheckTab'
import { ConflictSheet, SaveBadge } from '../components/EditorParts'
import { InterviewTab } from '../components/InterviewTab'
import { JobsSheet } from '../components/JobCards'
import { OutlineSheet } from '../components/OutlineSheet'
import { StateTab } from '../components/StateTab'
import { WriteTab } from '../components/WriteTab'
import { IconButton, TopBar } from '../components/ui'
import { cn } from '../lib/format'
import { loadPref, savePref } from '../lib/native'
import { flatScenes, scenePath } from '../lib/outlineOps'
import { useScene } from '../stores/editor'
import { useJobs } from '../stores/jobs'
import { useProject } from '../stores/project'
import { useUi, type Tab } from '../stores/ui'

const TABS: { id: Tab; label: string; icon: typeof PenLine }[] = [
  { id: 'write', label: 'Write', icon: PenLine },
  { id: 'bible', label: 'Bible', icon: BookMarked },
  { id: 'state', label: 'State', icon: SquareStack },
  { id: 'check', label: 'Check', icon: ScanSearch },
  { id: 'interview', label: 'Interview', icon: MessagesSquare }
]

/**
 * Leaves the project: unsaved text is sent first, then the computer is told the project is closed. Resolves false
 * (and stays) while a conflict on the open scene waits for an answer.
 */
export async function leaveProject(): Promise<boolean> {
  if (!(await useScene.getState().leave())) return false
  await useProject.getState().close()
  useUi.getState().setScene(null)
  return true
}

export function Workspace() {
  const state = useProject((s) => s.state)
  const tab = useUi((s) => s.tab)
  const setTab = useUi((s) => s.setTab)
  const sceneId = useUi((s) => s.sceneId)
  const setScene = useUi((s) => s.setScene)
  const go = useUi((s) => s.go)
  const setOutlineOpen = useUi((s) => s.setOutlineOpen)
  const folder = state?.folder
  const outline = state?.outline

  // The results the computer still holds for this project (from before the app was closed or the phone slept).
  useEffect(() => {
    if (folder) void useJobs.getState().refresh(folder)
  }, [folder])

  // Choose the scene: the one written last, else the first. A scene that is gone (deleted, or removed on the
  // computer) is replaced by the first one.
  useEffect(() => {
    if (!folder || !outline) return
    const all = flatScenes(outline)
    if (sceneId && all.some((s) => s.scene.id === sceneId)) return
    let cancelled = false
    void loadPref<Record<string, string>>('lastScenes', {}).then((map) => {
      if (cancelled) return
      const wanted = map[folder]
      setScene(all.find((s) => s.scene.id === wanted)?.scene.id ?? all[0]?.scene.id ?? null)
    })
    return () => {
      cancelled = true
    }
  }, [folder, outline, sceneId, setScene])

  // Open the scene's file when the scene changes.
  const found = outline && sceneId ? flatScenes(outline).find((s) => s.scene.id === sceneId) : undefined
  const path = found ? scenePath(found.scene) : null
  useEffect(() => {
    if (!folder || !path) return
    void useScene.getState().open(folder, path).then((ok) => {
      if (ok) return void loadPref<Record<string, string>>('lastScenes', {}).then((map) => savePref('lastScenes', { ...map, [folder]: sceneId }))
      // Refused because the open scene has a conflict to settle: the screen goes back to that scene.
      const b = useScene.getState().buffer
      if (b?.conflict && b.path !== path && outline) {
        const back = flatScenes(outline).find((s) => scenePath(s.scene) === b.path)
        if (back) setScene(back.scene.id)
      }
    })
    // sceneId is what picks path; it is only read to remember the choice.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [folder, path])

  if (!state) return null

  return (
    <div className="flex h-full flex-col">
      <TopBar>
        <IconButton
          label="Back to projects"
          onClick={async () => {
            if (await leaveProject()) go('projects')
          }}
        >
          <ArrowLeft size={21} />
        </IconButton>
        <button className="min-w-0 flex-1 rounded-lg px-1 py-1 text-left active:bg-hover" onClick={() => setOutlineOpen(true)}>
          <div className="truncate font-[family-name:var(--font-display)] text-[16px] font-semibold">{state.project.title}</div>
          <div className="truncate text-[12px] text-faint">{found ? `${found.chapterTitle} · ${found.scene.title}` : 'Choose a scene'}</div>
        </button>
        <SaveBadge store={useScene} />
        <IconButton label="Chapters and scenes" onClick={() => setOutlineOpen(true)}>
          <ListTree size={20} />
        </IconButton>
        <IconButton label="Settings" onClick={() => go('settings')}>
          <SettingsIcon size={20} />
        </IconButton>
      </TopBar>
      <div className="px-safe min-h-0 flex-1">
        {tab === 'write' && <WriteTab />}
        {tab === 'bible' && <BibleTab />}
        {tab === 'state' && <StateTab />}
        {tab === 'check' && <CheckTab />}
        {tab === 'interview' && <InterviewTab />}
      </div>
      <nav className="pb-safe px-safe shrink-0 border-t border-border bg-panel">
        <div className="grid grid-cols-5">
          {TABS.map((t) => (
            <button key={t.id} className={cn('flex h-14 flex-col items-center justify-center gap-0.5 text-[11.5px]', tab === t.id ? 'text-accent-text' : 'text-muted active:bg-hover')} onClick={() => setTab(t.id)}>
              <t.icon size={20} />
              {t.label}
            </button>
          ))}
        </div>
      </nav>
      <OutlineSheet />
      <JobsSheet />
      <ConflictSheet store={useScene} />
    </div>
  )
}
