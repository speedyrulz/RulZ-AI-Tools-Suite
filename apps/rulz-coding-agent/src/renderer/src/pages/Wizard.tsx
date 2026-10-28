import { ArrowLeft, ArrowRight, CircleCheck, FolderOpen } from 'lucide-react'
import { useEffect, useState } from 'react'
import { LogoWordmark } from '../components/Logo'
import { ConnectionHelp } from '../components/NotConnected'
import { ApprovalChooser, ModelSelect } from '../components/common'
import { Button, Input } from '../components/ui'
import { invoke } from '../lib/api'
import { cn } from '../lib/format'
import { activeModelId, useApp } from '../stores/app'
import { useWorkspace } from '../stores/workspace'

const STEPS = ['Connect', 'Model', 'Approvals', 'First project']

/** First run: find the server, pick a model, explain approvals, create a project. */
export function Wizard() {
  const [step, setStep] = useState(0)
  const settings = useApp((s) => s.settings)!
  const connection = useApp((s) => s.connection)
  const update = useApp((s) => s.updateSettings)
  const create = useWorkspace((s) => s.create)
  const [model, setModel] = useState('')
  const [name, setName] = useState('My First App')
  const [parent, setParent] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const connected = connection.status === 'connected'

  useEffect(() => {
    if (!model && connection.models.length) setModel(activeModelId(settings, connection))
  }, [connection, settings, model])

  const finish = async (withProject: boolean): Promise<void> => {
    setBusy(true)
    await update({ firstRunDone: true, model: model || settings.model })
    if (withProject) await create(name, parent)
    setBusy(false)
  }

  const folder = `${(parent ?? settings.projectsDir).replace(/[\\/]+$/, '')}\\${name.trim() || '…'}`

  return (
    <div className="flex h-full items-center justify-center overflow-y-auto p-8">
      <div className="w-full max-w-xl">
        <LogoWordmark className="mx-auto mb-6 h-12" />
        <div className="mb-5 flex items-center justify-center gap-2">
          {STEPS.map((s, i) => (
            <div key={s} className="flex items-center gap-2">
              <span
                className={cn(
                  'flex size-6 items-center justify-center rounded-full text-xs font-semibold',
                  i < step ? 'bg-accent text-accent-fg' : i === step ? 'border-2 border-accent text-accent' : 'border border-border text-faint'
                )}
              >
                {i + 1}
              </span>
              <span className={cn('text-xs', i === step ? 'text-text' : 'text-faint')}>{s}</span>
              {i < STEPS.length - 1 && <span className="mx-1 h-px w-6 bg-border" />}
            </div>
          ))}
        </div>

        <div className="rounded-xl border border-border bg-panel p-7">
          {step === 0 && (
            <div className="space-y-5">
              <div>
                <h1 className="text-xl font-semibold">Welcome to RulZ Coding Agent</h1>
                <p className="mt-2 text-sm text-muted">
                  Describe the app you want in plain English. An AI model running on this computer builds it for you: it writes the files, installs what
                  is needed, runs the app and fixes problems.
                </p>
              </div>
              {connected ? (
                <p className="flex items-center gap-2 rounded-lg border border-ok/30 bg-ok/10 px-3.5 py-3 text-sm">
                  <CircleCheck size={18} className="text-ok" /> Connected to RulZ LLM Suite.
                </p>
              ) : (
                <div className="rounded-lg border border-border bg-panel-2/60 p-4">
                  <ConnectionHelp compact />
                </div>
              )}
            </div>
          )}

          {step === 1 && (
            <div className="space-y-4">
              <h1 className="text-xl font-semibold">Which AI model should build your apps?</h1>
              <p className="text-sm text-muted">
                A coding model works best. Qwen3-Coder is a great choice. Models marked “ready” are already loaded; others load when you start your
                first task.
              </p>
              <ModelSelect models={connection.models} value={model} onChange={setModel} />
            </div>
          )}

          {step === 2 && (
            <div className="space-y-4">
              <h1 className="text-xl font-semibold">When should I ask you first?</h1>
              <p className="text-sm text-muted">
                Before each task I save a restore point, so <b>Undo last task</b> can always put your project back. I'll ask for your OK before steps
                you might not want, and you can change this any time in Settings.
              </p>
              <ApprovalChooser value={settings.approval} onChange={(approval) => void update({ approval })} />
            </div>
          )}

          {step === 3 && (
            <div className="space-y-4">
              <h1 className="text-xl font-semibold">Name your first project</h1>
              <p className="text-sm text-muted">Each project gets its own folder. You can make as many as you like.</p>
              <Input value={name} onChange={(e) => setName(e.target.value)} className="h-10 text-sm" autoFocus onKeyDown={(e) => e.key === 'Enter' && name.trim() && void finish(true)} />
              <div className="flex items-center justify-between gap-3 text-xs text-muted">
                <span className="truncate">Folder: {folder}</span>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={async () => {
                    const dir = await invoke('dialog:pickFolder', { title: 'Where should the project folder go?', defaultPath: parent ?? settings.projectsDir })
                    if (dir) setParent(dir)
                  }}
                >
                  <FolderOpen size={13} /> Choose folder
                </Button>
              </div>
            </div>
          )}

          <div className="mt-7 flex items-center justify-between">
            {step > 0 ? (
              <Button variant="ghost" onClick={() => setStep(step - 1)}>
                <ArrowLeft size={15} /> Back
              </Button>
            ) : (
              <span />
            )}
            {step < 3 ? (
              <Button variant="primary" className="h-10 px-5 text-sm" disabled={!connected || (step === 1 && !model)} onClick={() => setStep(step + 1)}>
                Next <ArrowRight size={15} />
              </Button>
            ) : (
              <div className="flex gap-2">
                <Button variant="ghost" className="h-10" disabled={busy} onClick={() => void finish(false)}>
                  Skip for now
                </Button>
                <Button variant="primary" className="h-10 px-5 text-sm" disabled={busy || !name.trim()} onClick={() => void finish(true)}>
                  Create project <ArrowRight size={15} />
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
