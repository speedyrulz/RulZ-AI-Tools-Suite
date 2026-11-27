import { Layers } from 'lucide-react'
import { useEffect } from 'react'
import { CheckPanel } from '../components/loraMerge/CheckPanel'
import { InputsCard } from '../components/loraMerge/InputsCard'
import { JobsCard } from '../components/loraMerge/JobsCard'
import { ResultCard } from '../components/loraMerge/ResultCard'
import { Button, PageTitle, Tooltip } from '../components/ui'
import { loraMergeBlockReason, namedRows } from '../lib/loraMerge'
import { useLoraMerge } from '../stores/loraMerge'

export function LoraMergePage() {
  const rows = useLoraMerge((s) => s.rows)
  const check = useLoraMerge((s) => s.check)
  const checking = useLoraMerge((s) => s.checking)
  const checkError = useLoraMerge((s) => s.checkError)
  const jobs = useLoraMerge((s) => s.jobs)
  const starting = useLoraMerge((s) => s.starting)
  const outputName = useLoraMerge((s) => s.outputName)
  const runCheck = useLoraMerge((s) => s.runCheck)
  const start = useLoraMerge((s) => s.start)

  // The LoRA list is read from disk each time the page opens; the first check follows when the form is already filled in.
  useEffect(() => {
    void useLoraMerge.getState().loadList()
    if (useLoraMerge.getState().check === null) useLoraMerge.getState().runCheck()
  }, [])

  const reason = loraMergeBlockReason({ rows, check, checking, checkError, jobs, starting, outputName })
  const button = (
    <Button variant="primary" size="lg" disabled={reason !== null} onClick={() => void start()}>
      <Layers size={18} />
      Merge and save
    </Button>
  )

  return (
    <div className="space-y-6">
      <PageTitle title="LoRA merge">Combines LoRAs into one. Runs in the app on the CPU; ComfyUI is not needed. The result goes to models/loras/merged/.</PageTitle>
      <InputsCard />
      <ResultCard />
      <CheckPanel check={check} checking={checking} checkError={checkError} ready={namedRows(rows).length >= 2} />
      <div className="flex items-center gap-4">
        {reason ? (
          <Tooltip content={reason}>
            <span tabIndex={0} className="inline-flex">
              {button}
            </span>
          </Tooltip>
        ) : (
          button
        )}
        {reason && <span className="text-[13px] text-muted">{reason}</span>}
        {checkError && !checking && (
          <Button size="sm" onClick={() => runCheck()}>
            Retry
          </Button>
        )}
      </div>
      <JobsCard />
    </div>
  )
}
