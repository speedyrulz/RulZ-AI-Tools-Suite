import { ArrowLeft, Plus } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useWbench } from '../../stores/wbench'
import { Button } from '../ui'
import { RunList } from './RunList'
import { RunView } from './RunView'
import { SetupForm } from './SetupForm'

/** The writing benchmark: the list of runs, a new run's setup, or one run (voting and scoreboard). */
export function WbenchView({ onBack }: { onBack: () => void }) {
  const init = useWbench((s) => s.init)
  const openId = useWbench((s) => s.openId)
  const entries = useWbench((s) => s.entries)
  const running = useWbench((s) => s.running)
  const [setup, setSetup] = useState(false)
  useEffect(() => void init(), [init])

  const open = entries.find((e) => e.ok && e.run.id === openId)
  if (open?.ok) {
    return (
      <div className="h-full overflow-y-auto">
        <RunView run={open.run} />
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-3xl space-y-6 px-6 py-8">
        <div className="flex items-start gap-3">
          <Button size="sm" variant="ghost" onClick={setup ? () => setSetup(false) : onBack}>
            <ArrowLeft size={14} aria-hidden /> {setup ? 'All runs' : 'Writing'}
          </Button>
        </div>
        <div>
          <h1 className="font-[family-name:var(--font-display)] text-lg font-semibold">Writing benchmark</h1>
          <p className="mt-1 text-[13px] text-muted">
            Compare models, adapters or settings on the same scene briefs. Each one writes every scene; you vote on blind pairs, and a rating per candidate comes from your votes. It uses the model while it runs, so chats and Writing actions wait.
          </p>
        </div>
        {setup ? (
          <SetupForm onCancel={() => setSetup(false)} />
        ) : (
          <>
            <div>
              <Button variant="primary" disabled={running} onClick={() => setSetup(true)}>
                <Plus size={15} aria-hidden /> New run
              </Button>
              {running && <span className="ml-3 text-[12px] text-faint">A run is in progress. Open it below.</span>}
            </div>
            <section>
              <h2 className="mb-2 text-[11px] font-semibold tracking-wide text-faint uppercase">Your runs</h2>
              <RunList />
            </section>
          </>
        )}
      </div>
    </div>
  )
}
