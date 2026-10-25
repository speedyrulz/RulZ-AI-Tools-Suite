import { FolderOpen } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { AppInfo } from '@shared/ipc'
import { invoke } from '../lib/api'
import { act, useApps } from '../stores/apps'
import { RequirementList } from './Requirements'
import { Button, Dialog, Switch } from './ui'

export function SettingsDialog({ onClose }: { onClose: () => void }) {
  const settings = useApps((s) => s.settings)
  const update = useApps((s) => s.updateSettings)
  const [info, setInfo] = useState<AppInfo | null>(null)
  const [root, setRoot] = useState(settings?.installRoot ?? '')

  useEffect(() => void invoke('app:info').then(setInfo), [])
  if (!settings) return null

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()} title="Settings" className="w-[600px]" footer={<Button onClick={onClose}>Done</Button>}>
      <div className="space-y-6">
        <section>
          <label className="mb-1.5 block text-xs font-medium tracking-wide text-faint uppercase">Default install folder</label>
          <div className="flex gap-2">
            <input
              value={root}
              onChange={(e) => setRoot(e.target.value)}
              onBlur={() => root.trim() && root !== settings.installRoot && void act(() => update({ installRoot: root.trim() }))}
              spellCheck={false}
              className="h-9 min-w-0 flex-1 rounded-lg border border-border bg-panel-2 px-3 text-[13px] outline-none focus:border-accent"
            />
            <Button
              onClick={async () => {
                const dir = await invoke('dialog:pickFolder', { title: 'Default install folder', defaultPath: root })
                if (dir) {
                  setRoot(dir)
                  await act(() => update({ installRoot: dir }))
                }
              }}
            >
              <FolderOpen className="size-4" /> Browse
            </Button>
          </div>
          <p className="mt-1.5 text-xs text-faint">Each app gets its own folder inside it. You can still pick another place when installing.</p>
        </section>

        <section className="space-y-3">
          <Toggle
            label="Check for updates when the launcher starts"
            checked={settings.checkUpdatesOnStart}
            onChange={(v) => void act(() => update({ checkUpdatesOnStart: v }))}
          />
          <Toggle
            label="Show the latest README from GitHub"
            hint="Off: show the copy that came with the launcher"
            checked={settings.liveReadme}
            onChange={(v) => void act(() => update({ liveReadme: v }))}
          />
        </section>

        <section>
          <div className="mb-2 text-xs font-medium tracking-wide text-faint uppercase">Tools on this PC</div>
          <RequirementList refs={[{ id: 'python' }, { id: 'git' }, { id: 'ffmpeg' }, { id: 'nvidia' }, { id: 'dotnet-fx' }]} />
        </section>

        {info && (
          <p className="border-t border-border pt-3 text-xs text-faint">
            RulZ AI Tools Suite {info.version} · data in {info.userData}
          </p>
        )}
      </div>
    </Dialog>
  )
}

function Toggle({ label, hint, checked, onChange }: { label: string; hint?: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div>
        <div className="text-sm">{label}</div>
        {hint && <div className="text-xs text-faint">{hint}</div>}
      </div>
      <Switch checked={checked} onChange={onChange} />
    </div>
  )
}
