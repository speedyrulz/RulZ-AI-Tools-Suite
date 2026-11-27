import { GitMerge } from 'lucide-react'
import { useEffect } from 'react'
import { CheckPanel } from '../components/merge/CheckPanel'
import { JobsCard } from '../components/merge/JobsCard'
import { MixCard, SourcesCard } from '../components/merge/RecipeCards'
import { Button, Card, PageTitle, Tooltip } from '../components/ui'
import { jobIsActive, validOutputName } from '../lib/merge'
import { useApp } from '../stores/app'
import { useMerge } from '../stores/merge'

const NODES_BY_MODE = { weighted: ['ModelMergeSimple'], addDifference: ['ModelMergeSubtract', 'ModelMergeAdd'], blocks: [] } as const

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
                : 'Merges are run by ComfyUI, so they need a connection. Start the app\'s own server or connect to a URL on the Server page.'}
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

  // Why the button is off, most useful reason first.
  const busy = jobs.some((j) => jobIsActive(j.status))
  const missingNodes = [...NODES_BY_MODE[recipe.mode]].filter((n) => caps.nodes[n] === false)
  let reason: string | null = null
  if (busy) reason = 'A merge of this app is still running. Wait for it or cancel it.'
  else if (!recipe.a || !recipe.b) reason = 'Choose A and B.'
  else if (recipe.mode === 'addDifference' && !recipe.c) reason = 'Choose C.'
  else if (recipe.mode === 'blocks' && !recipe.blockNode) reason = 'Choose a block node.'
  else if (!validOutputName(recipe.outputName)) reason = 'Give the output a valid name.'
  else if (missingNodes.length > 0) reason = `This server has no ${missingNodes.join(' or ')} node.`
  else if (checking) reason = 'The files are still being checked.'
  else if (checkError) reason = `The files could not be checked: ${checkError}. Try again.`
  else if (!check) reason = 'The files have not been checked yet.'
  else if (check.problems.length > 0) reason = 'Fix the problems in the check first.'
  else if (starting) reason = 'Starting.'

  const ready = !!recipe.a && !!recipe.b && (recipe.mode !== 'addDifference' || !!recipe.c)
  const button = (
    <Button variant="primary" size="lg" disabled={reason !== null} onClick={() => void start()}>
      <GitMerge size={18} />
      Merge and save
    </Button>
  )

  return (
    <div className="space-y-6">
      <PageTitle title="Merge">Mix two or three models into one new file. ComfyUI does the work and saves it in its output folder.</PageTitle>
      <SourcesCard caps={caps} />
      <MixCard check={check} />
      <CheckPanel check={check} checking={checking} ready={ready} />
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
