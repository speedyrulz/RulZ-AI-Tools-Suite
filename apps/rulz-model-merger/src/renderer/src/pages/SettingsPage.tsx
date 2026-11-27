import { Moon, Sun } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Card, PageTitle, Segmented } from '../components/ui'
import { call } from '../lib/api'
import { useApp } from '../stores/app'

export function SettingsPage() {
  const theme = useApp((s) => s.settings?.theme ?? 'dark')
  const setTheme = useApp((s) => s.setTheme)
  const [version, setVersion] = useState<string | null>(null)

  useEffect(() => {
    void call('app:version').then((v) => setVersion(v ?? null))
  }, [])

  return (
    <div>
      <PageTitle title="Settings" />
      <div className="max-w-2xl space-y-4">
        <Card title="Appearance" description="Dark is the default. You can also switch with Ctrl+Shift+L.">
          <Segmented
            label="Theme"
            value={theme}
            onChange={(t) => void setTheme(t)}
            options={[
              { value: 'dark', label: 'Dark', icon: <Moon size={14} /> },
              { value: 'light', label: 'Light', icon: <Sun size={14} /> }
            ]}
          />
        </Card>
        <Card title="About">
          <p className="text-[13px] text-body">
            RulZ Model Merger merges checkpoints and LoRAs through ComfyUI and compares the mixes side by side.
          </p>
          <p className="mt-2 text-xs text-faint">Version {version ?? 'unknown'}</p>
        </Card>
      </div>
    </div>
  )
}
