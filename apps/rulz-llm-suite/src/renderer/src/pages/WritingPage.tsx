import { Feather, Scale } from 'lucide-react'
import { useEffect, useState } from 'react'
import { WbenchView } from '../components/wbench/WbenchView'
import { Button } from '../components/ui'
import { BlockedBanner } from '../components/writing/Banners'
import { EntryEditor } from '../components/writing/EntryEditor'
import { LeftColumn, Welcome } from '../components/writing/LeftColumn'
import { SceneEditor } from '../components/writing/SceneEditor'
import { SidePanel } from '../components/writing/SidePanel'
import { StatusBar } from '../components/writing/StatusBar'
import { lastProjectFolder, useNovel } from '../stores/novel'

let triedReopen = false
let benchWasOpen = false

export function WritingPage() {
  const folder = useNovel((s) => s.folder)
  const target = useNovel((s) => s.target)
  const panelOpen = useNovel((s) => s.panelOpen)
  const opening = useNovel((s) => s.opening)
  // The benchmark view stays up while you visit another page and come back.
  const [benchOpen, setBenchOpenState] = useState(benchWasOpen)
  const setBenchOpen = (v: boolean): void => {
    benchWasOpen = v
    setBenchOpenState(v)
  }

  useEffect(() => {
    const s = useNovel.getState()
    void s.refreshProjects()
    // Come back to the project that was open last, once per run of the app.
    if (!s.folder && !triedReopen) {
      triedReopen = true
      const last = lastProjectFolder()
      if (last) void s.openProject(last)
    }
  }, [])

  if (!folder) {
    if (opening) return <div className="flex h-full items-center justify-center text-sm text-faint">Opening the project…</div>
    if (benchOpen) return <WbenchView onBack={() => setBenchOpen(false)} />
    return (
      <div className="relative h-full">
        <Welcome />
        <Button size="sm" className="absolute top-4 right-6" onClick={() => setBenchOpen(true)}>
          <Scale size={14} aria-hidden /> Benchmark
        </Button>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col">
      <BlockedBanner />
      <div className="flex min-h-0 flex-1">
        <LeftColumn />
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="min-h-0 flex-1">
            {target?.kind === 'scene' && <SceneEditor key={target.id} sceneId={target.id} />}
            {target?.kind === 'entry' && <EntryEditor key={target.id} entryId={target.id} />}
            {!target && (
              <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-sm text-faint">
                <Feather size={28} aria-hidden />
                Pick a scene in the outline, or a bible entry, to start.
              </div>
            )}
          </div>
          <StatusBar />
        </div>
        {panelOpen && <SidePanel />}
      </div>
    </div>
  )
}
