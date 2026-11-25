import { ArrowLeft, Backpack, Map, Settings, Share2, Square } from 'lucide-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { pathTo, type AdventureNode, type AdventureSetup } from '@shared/adventure'
import { ModeLine } from '../components/ModeLine'
import type { PhoneModel } from '@shared/phone'
import { EditPassageSheet } from '../components/adventure/EditPassageSheet'
import { MapSheet, PassageMenuSheet, SettingsSheet, StateSheet } from '../components/adventure/PlaySheets'
import { ActionBox, ChoiceList, JobStatus, PassageCard } from '../components/adventure/Reading'
import { Button, IconButton, Sheet, TopBar } from '../components/ui'
import { link } from '../lib/remote'
import { adventureError, running, useAdventure } from '../stores/adventure'
import { useUi } from '../stores/ui'

export function AdventurePlay() {
  const { adventure: adv, position, jobs, otherNode, starting, move, play, retry, update, edit } = useAdventure()
  const go = useUi((s) => s.go)
  const [action, setAction] = useState('')
  const [menu, setMenu] = useState<string | null>(null)
  const [sheet, setSheet] = useState<'state' | 'map' | 'settings' | 'edit' | 'choice' | null>(null)
  const [choiceId, setChoiceId] = useState('')
  const [setup, setSetup] = useState<AdventureSetup | null>(null)
  const [models, setModels] = useState<PhoneModel[]>([])
  const [editNode, setEditNode] = useState<AdventureNode | null>(null)

  useEffect(() => {
    void link.invoke('writing:models').then(setModels).catch((e) => toast.error(adventureError(e)))
  }, [])

  if (!adv) {
    return <div className="p-4"><Button onClick={() => go('adventures')}>Back to adventures</Button></div>
  }

  const path = pathTo(adv, position)
  const node = position ? adv.nodes[position] : undefined
  const active = Object.values(jobs).filter(running).sort((a, b) => b.startedAt - a.startedAt)[0]
  const busy = !!active || starting
  const selected = menu ? adv.nodes[menu] : undefined
  const closeSheet = (): void => setSheet(null)
  const retryNode = (target: AdventureNode, what: 'passage' | 'choices'): void => {
    if (what === 'choices' && target.edited &&
      !window.confirm('New choices will replace this passage’s hand-edited state. Continue?')) return
    setMenu(null)
    void retry({ adventureId: adv.id, nodeId: target.id, what })
  }
  const stop = (): void => {
    if (active) void link.invoke('adventure:abort', active.id).catch((e) => toast.error(adventureError(e)))
  }
  const exportStory = async (): Promise<void> => {
    try {
      const story = await link.invoke('adventure:exportMarkdown', adv.id, position)
      if (navigator.share) await navigator.share({ title: adv.setup.title, text: story })
      else {
        await navigator.clipboard.writeText(story)
        toast.success('Copied the story')
      }
    } catch (e) {
      if (!(e instanceof Error && e.name === 'AbortError')) toast.error(adventureError(e))
    }
  }
  const writeChoice = (id: string): void => {
    const choice = node?.choices.find((item) => item.id === id)
    if (node && choice) {
      void play({
        adventureId: adv.id,
        parentId: node.id,
        input: { kind: 'choice', choiceId: choice.id, text: choice.text }
      })
    }
    closeSheet()
  }

  return (
    <div className="flex h-full flex-col">
      <TopBar>
        <IconButton label="Back" onClick={() => {
          useAdventure.getState().close()
          go('adventures')
        }}>
          <ArrowLeft size={20} />
        </IconButton>
        <h1 className="min-w-0 flex-1 truncate text-[17px] font-semibold">{adv.setup.title}</h1>
        <IconButton label="Export" onClick={() => void exportStory()}><Share2 size={19} /></IconButton>
        {active && <IconButton label="Stop" onClick={stop}><Square size={19} /></IconButton>}
      </TopBar>
      <ModeLine mode="adventure" modelId={adv.setup.modelId} />
      <div className="px-safe pb-safe min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-lg space-y-4 px-4 py-4">
          {otherNode && (
            <button className="min-h-11 text-link" onClick={() => void move(otherNode)}>
              New passage on another device
            </button>
          )}
          {path.map((passage, index) => (
            <PassageCard key={passage.id} node={passage} next={path[index + 1]?.input} onMenu={() => setMenu(passage.id)} />
          ))}
          {active && <JobStatus job={active} onStop={stop} />}
          {node && (
            <div className="space-y-2">
              {node.ended && <p className="font-medium">The end</p>}
              {!node.ended && !node.choices.length && (
                <>
                  <p className="text-muted">No choices yet</p>
                  <Button disabled={busy} onClick={() => retryNode(node, 'choices')}>New choices</Button>
                </>
              )}
              <ChoiceList
                adventure={adv}
                node={node}
                busy={busy}
                onMove={(id) => void move(id)}
                onWrite={writeChoice}
                onMenu={(id) => {
                  setChoiceId(id)
                  setSheet('choice')
                }}
              />
              <ActionBox
                action={action}
                busy={busy}
                onChange={setAction}
                onSubmit={() => {
                  const value = action.trim()
                  setAction('')
                  void play({ adventureId: adv.id, parentId: node.id, input: { kind: 'action', text: value } })
                    .then((job) => { if (!job) setAction((current) => current || value) })
                }}
              />
            </div>
          )}
          {!adv.rootId && !node && !active && (
            <Button disabled={busy} onClick={() => void play({ adventureId: adv.id, parentId: null, input: { kind: 'start' } })}>
              Write the opening
            </Button>
          )}
        </div>
      </div>
      <div className="pb-safe flex justify-around border-t border-border bg-panel">
        <Button variant="ghost" onClick={() => setSheet('state')}><Backpack size={18} /> State</Button>
        <Button variant="ghost" onClick={() => setSheet('map')}><Map size={18} /> Map</Button>
        <Button variant="ghost" onClick={() => {
          setSetup(adv.setup)
          setSheet('settings')
        }}>
          <Settings size={18} /> Settings
        </Button>
      </div>
      <PassageMenuSheet
        adventure={adv}
        node={selected}
        position={position}
        busy={busy}
        onClose={() => setMenu(null)}
        onMove={(id) => void move(id)}
        onRetry={retryNode}
        onEdit={(target) => {
          setEditNode(target)
          setMenu(null)
          setSheet('edit')
        }}
      />
      <StateSheet open={sheet === 'state'} node={node} onClose={closeSheet} />
      <MapSheet open={sheet === 'map'} adventure={adv} path={path} onClose={closeSheet} onMove={(id) => void move(id)} />
      {sheet === 'settings' && setup && (
        <SettingsSheet
          setup={setup}
          setSetup={setSetup}
          models={models}
          onClose={closeSheet}
          onSave={() => {
            const { title, genre, tone, length, choiceCount, modelId, temperature } = setup
            void update(adv.id, { title, genre, tone, length, choiceCount, modelId, temperature })
              .then((ok) => ok && closeSheet())
          }}
        />
      )}
      {sheet === 'edit' && editNode && (
        <EditPassageSheet adventureId={adv.id} node={editNode} onClose={closeSheet} onSave={edit} />
      )}
      <Sheet open={sheet === 'choice'} onOpenChange={(open) => !open && closeSheet()} title="Choice">
        <div className="px-4 pb-4">
          <Button className="w-full" disabled={busy} onClick={() => writeChoice(choiceId)}>Write again</Button>
        </div>
      </Sheet>
    </div>
  )
}
