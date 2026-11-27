import { GitMerge } from 'lucide-react'
import { useEffect } from 'react'
import { CheckPanel } from '../components/merge/CheckPanel'
import { JobsCard } from '../components/merge/JobsCard'
import { LorasCard } from '../components/merge/LorasCard'
import { BakeOutputCard, MixCard, SourcesCard } from '../components/merge/RecipeCards'
import { Button, Card, PageTitle, Tooltip } from '../components/ui'
import { mergeStartReason } from '../lib/history'
import { useApp } from '../stores/app'
import { useMerge } from '../stores/merge'

export function MergePage() {
  const status = useApp((s) => s.status)
  const caps = useApp((s) => s.capabilities)
  const setPage = useApp((s) => s.setPage)
  const recipe = useMerge((s) => s.recipe)
  const check = useMerge((s) => s.check)
  const checking = useMerge((s) => s.checking)
  const checkError = useMerge((s) => s.checkError)
  const runCheck = useMerge((s) => s.runCheck)
  const jobs = useMerge((s) => s.jobs)
  const starting = useMerge((s) => s.starting)
  const start = useMerge((s) => s.start)

  // The first check after coming back to the page, when the form is already filled in.
  useEffect(() => {
    if (useMerge.getState().check === null) useMerge.getState().runCheck()
  }, [])

  const connected = !!status?.reachable && (status.own || status.sharedQueueAccepted)

  if (!connected || !caps) {
    const sharedNotAccepted = !!status?.reachable && !status.own && !status.sharedQueueAccepted
    return (
      <div>
        <PageTitle title="Merge" />
        <Card title="Connect to ComfyUI first">
          <p className="text-[13px] text-muted">
            {sharedNotAccepted
              ? 'Merges run in the queue of the connected ComfyUI. This server was not started by this app, so you need to agree to share its queue on the Server page.'
              : status?.reachable
                ? 'Loading the models of the server.'
                : "Merges are run by ComfyUI, so they need a connection. Start the app's own server or connect to a URL on the Server page."}
          </p>
          <div className="mt-4">
            <Button variant="primary" onClick={() => setPage('server')}>
              Go to the Server page
            </Button>
          </div>
        </Card>
      </div>
    )
  }

  // Why the button is off, most useful reason first (shared with History's Re-run).
  const reason = mergeStartReason({ recipe, connected, nodes: caps.nodes, jobs, check, checking, checkError, starting })

  const ready = !!recipe.a && (recipe.mode === 'bake' || !!recipe.b) && (recipe.mode !== 'addDifference' || !!recipe.c)
  const button = (
    <Button variant="primary" size="lg" disabled={reason !== null} onClick={() => void start()}>
      <GitMerge size={18} />
      Merge and save
    </Button>
  )

  return (
    <div className="space-y-6">
      <PageTitle title="Merge">
        Mix two or three models into one new file, or bake LoRAs into one. ComfyUI does the work and saves it in its output folder.
      </PageTitle>
      <SourcesCard caps={caps} />
      {recipe.mode === 'bake' ? <BakeOutputCard /> : <MixCard check={check} />}
      <LorasCard caps={caps} />
      <CheckPanel check={check} checking={checking} ready={ready} kind={recipe.kind} />
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
