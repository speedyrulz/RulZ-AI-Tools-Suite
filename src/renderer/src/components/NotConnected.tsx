import { KeyRound, LoaderCircle, PlugZap, RefreshCw, Settings } from 'lucide-react'
import { useState } from 'react'
import { useApp } from '../stores/app'
import { LogoWordmark } from './Logo'
import { Button, Input } from './ui'

/** What to do when RulZ LLM Suite can't be reached, in plain words. */
export function ConnectionHelp({ compact }: { compact?: boolean }) {
  const connection = useApp((s) => s.connection)
  const reconnect = useApp((s) => s.reconnect)
  const updateSettings = useApp((s) => s.updateSettings)
  const setSettingsOpen = useApp((s) => s.setSettingsOpen)
  const [key, setKey] = useState('')
  const checking = connection.status === 'checking'

  if (connection.status === 'needs-key') {
    return (
      <div className="space-y-3">
        <p className="flex items-center gap-2 text-[15px] font-medium">
          <KeyRound size={18} className="text-accent" /> RulZ LLM Suite needs its API key
        </p>
        <p className="text-sm text-muted">
          Its server is protected with a key. Copy the key from the <b>Developer</b> page in RulZ LLM Suite and paste it here.
        </p>
        <div className="flex gap-2">
          <Input type="password" placeholder="API key" value={key} onChange={(e) => setKey(e.target.value)} />
          <Button
            variant="primary"
            disabled={!key.trim()}
            onClick={async () => {
              await updateSettings({ apiKey: key.trim() })
              await reconnect()
            }}
          >
            Connect
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <p className="flex items-center gap-2 text-[15px] font-medium">
        <PlugZap size={18} className="text-accent" />
        {checking ? 'Looking for RulZ LLM Suite…' : "I can't find RulZ LLM Suite"}
      </p>
      {connection.status === 'other-server' ? (
        <p className="text-sm text-muted">
          Something else is answering at {connection.otherUrl} (probably LM Studio), but not RulZ LLM Suite. In RulZ LLM Suite, open the{' '}
          <b>Developer</b> page, set the port to <b>1235</b> and click <b>Start server</b>.
        </p>
      ) : (
        <ol className="list-decimal space-y-1 pl-5 text-sm text-muted">
          <li>Open RulZ LLM Suite.</li>
          <li>
            Go to the <b>Developer</b> page.
          </li>
          <li>
            Click <b>Start server</b>.
          </li>
        </ol>
      )}
      <div className="flex gap-2">
        <Button variant="primary" className={compact ? '' : 'h-10 px-5 text-sm'} onClick={() => void reconnect()} disabled={checking}>
          {checking ? <LoaderCircle size={15} className="animate-spin" /> : <RefreshCw size={15} />}
          Try again
        </Button>
        <Button variant="ghost" className={compact ? '' : 'h-10'} onClick={() => setSettingsOpen(true)}>
          <Settings size={15} /> Use a different address
        </Button>
      </div>
    </div>
  )
}

export function NotConnectedScreen() {
  return (
    <div className="flex h-full items-center justify-center p-8">
      <div className="w-full max-w-md space-y-8">
        <LogoWordmark className="mx-auto h-12" />
        <div className="rounded-xl border border-border bg-panel p-6">
          <ConnectionHelp />
        </div>
      </div>
    </div>
  )
}
