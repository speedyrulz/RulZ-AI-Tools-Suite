import { ArrowLeft, MoreVertical, Plus } from 'lucide-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import type { AdventureSetup, AdventureStarter } from '@shared/adventure'
import type { NovelProjectSummary } from '@shared/novel'
import { Badge, Button, Field, IconButton, Input, Select, Sheet, Switch, Textarea, TopBar } from '../components/ui'
import { relativeTime } from '../lib/format'
import { link } from '../lib/remote'
import { adventureError, useAdventure } from '../stores/adventure'
import { useUi } from '../stores/ui'

export const initialSetup: AdventureSetup = {
  title: '', genre: '', premise: '', character: '', tone: '', length: 'medium', choiceCount: 3,
  modelId: null, temperature: null
}
type Model = { id: string, name: string, loaded: boolean }

export function SetupFields({ setup, setSetup, models, full = false }: {
  setup: AdventureSetup
  setSetup: (setup: AdventureSetup) => void
  models: Model[]
  full?: boolean
}) {
  const change = (patch: Partial<AdventureSetup>): void => setSetup({ ...setup, ...patch })
  return (
    <div className="space-y-3">
      <Field label="Title">
        <Input aria-label="Title" value={setup.title} onChange={(e) => change({ title: e.target.value })} />
      </Field>
      <Field label="Genre">
        <Input aria-label="Genre" value={setup.genre} onChange={(e) => change({ genre: e.target.value })} />
      </Field>
      {full && (
        <>
          <Field label="Premise">
            <Textarea aria-label="Premise" rows={3} value={setup.premise} onChange={(e) => change({ premise: e.target.value })} />
          </Field>
          <Field label="Your character">
            <Textarea
              aria-label="Your character"
              rows={2}
              value={setup.character}
              onChange={(e) => change({ character: e.target.value })}
            />
          </Field>
        </>
      )}
      <Field label="Tone">
        <Input aria-label="Tone" value={setup.tone} onChange={(e) => change({ tone: e.target.value })} />
      </Field>
      <Field label="Passage length">
        <Select
          aria-label="Passage length"
          value={setup.length}
          onChange={(e) => change({ length: e.target.value as AdventureSetup['length'] })}
        >
          <option value="short">Short</option>
          <option value="medium">Medium</option>
          <option value="long">Long</option>
        </Select>
      </Field>
      <Field label="Choices">
        <Select aria-label="Choices" value={setup.choiceCount} onChange={(e) => change({ choiceCount: Number(e.target.value) })}>
          {[2, 3, 4].map((number) => <option key={number}>{number}</option>)}
        </Select>
      </Field>
      <Field label="Model">
        <Select aria-label="Model" value={setup.modelId ?? ''} onChange={(e) => change({ modelId: e.target.value || null })}>
          <option value="">Loaded model</option>
          {models.map((model) => <option key={model.id} value={model.id}>{model.name}{model.loaded ? ' (loaded)' : ''}</option>)}
        </Select>
      </Field>
      <Field label="Temperature" hint="Leave empty for the default.">
        <Input
          aria-label="Temperature"
          type="number"
          step="0.1"
          value={setup.temperature ?? ''}
          onChange={(e) => change({ temperature: e.target.value === '' ? null : Number(e.target.value) })}
        />
      </Field>
    </div>
  )
}

export function Adventures() {
  const { list, error, load, open, update } = useAdventure()
  const go = useUi((s) => s.go)
  const [rename, setRename] = useState<string | null>(null)
  const [title, setTitle] = useState('')
  useEffect(() => { void load() }, [load])
  return (
    <div className="flex h-full flex-col">
      <TopBar>
        <IconButton label="Back" onClick={() => go('projects')}><ArrowLeft size={20} /></IconButton>
        <h1 className="flex-1 text-[17px] font-semibold">Adventures</h1>
      </TopBar>
      <div className="px-safe pb-safe min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-lg space-y-3 px-4 py-4">
          <Button variant="primary" className="w-full" onClick={() => go('adventure-new')}>
            <Plus size={18} /> New adventure
          </Button>
          {error && <p className="text-danger">{error}</p>}
          {!error && !list.length && <p className="py-8 text-center text-faint">No adventures yet.</p>}
          {list.map((item) => (
            <div key={item.id} className="flex items-center rounded-2xl border border-border bg-panel">
              <button
                className="min-h-16 min-w-0 flex-1 p-4 text-left"
                onClick={() => void open(item.id).then((ok) => ok && go('adventure-play'))}
              >
                <span className="block font-medium">{item.title} {item.ended && <Badge>Ended</Badge>}</span>
                <span className="block text-sm text-faint">
                  {item.genre} · {item.nodeCount} passages · {relativeTime(item.updatedAt)}
                  {item.worldTitle ? ` · ${item.worldTitle}` : ''}
                </span>
              </button>
              <IconButton label={`Rename ${item.title}`} onClick={() => {
                setRename(item.id)
                setTitle(item.title)
              }}>
                <MoreVertical size={20} />
              </IconButton>
            </div>
          ))}
        </div>
      </div>
      <Sheet
        open={!!rename}
        onOpenChange={(value) => !value && setRename(null)}
        title="Rename adventure"
        footer={
          <Button
            variant="primary"
            disabled={!title.trim()}
            onClick={() => {
              if (rename) void update(rename, { title: title.trim() }).then((ok) => ok && setRename(null))
            }}
          >
            Save
          </Button>
        }
      >
        <div className="px-4 pb-4">
          <Input aria-label="Title" value={title} onChange={(e) => setTitle(e.target.value)} />
        </div>
      </Sheet>
    </div>
  )
}

export function AdventureNew() {
  const go = useUi((s) => s.go)
  const [setup, setSetup] = useState(initialSetup)
  const [starters, setStarters] = useState<AdventureStarter[]>([])
  const [models, setModels] = useState<Model[]>([])
  const [projects, setProjects] = useState<NovelProjectSummary[]>([])
  const [fromProject, setFromProject] = useState(false)
  const [folder, setFolder] = useState('')
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    void Promise.all([
      link.invoke('adventure:starters'), link.invoke('adventure:models'), link.invoke('novel:listProjects')
    ]).then(([ideas, availableModels, availableProjects]) => {
      setStarters(ideas)
      setModels(availableModels)
      setProjects(availableProjects)
    }).catch((e) => toast.error(adventureError(e)))
  }, [])
  const start = async (): Promise<void> => {
    setBusy(true)
    try {
      const adventure = await link.invoke('adventure:create', setup, fromProject ? folder : null)
      useAdventure.setState({
        adventure, position: adventure.currentId, jobs: {}, ownJobs: [], followedJobs: [], otherNode: null
      })
      go('adventure-play')
      await useAdventure.getState().play({ adventureId: adventure.id, parentId: null, input: { kind: 'start' } })
    } catch (e) {
      toast.error(adventureError(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="flex h-full flex-col">
      <TopBar>
        <IconButton label="Back" onClick={() => go('adventures')}><ArrowLeft size={20} /></IconButton>
        <h1 className="text-[17px] font-semibold">New adventure</h1>
      </TopBar>
      <div className="px-safe pb-safe min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-lg space-y-4 px-4 py-4">
          {starters.length > 0 && (
            <Field label="Starter ideas">
              <div className="flex flex-wrap gap-2">
                {starters.map((starter) => (
                  <Button
                    key={starter.id}
                    onClick={() => setSetup({
                      ...setup, title: starter.title, genre: starter.genre, premise: starter.premise,
                      character: starter.character, tone: starter.tone
                    })}
                  >
                    {starter.title}
                  </Button>
                ))}
              </div>
            </Field>
          )}
          <Field inline label="From a Writing project">
            <Switch label="From a Writing project" checked={fromProject} onCheckedChange={setFromProject} />
          </Field>
          {fromProject && (
            <Field label="Project">
              <Select aria-label="Project" value={folder} onChange={(e) => setFolder(e.target.value)}>
                <option value="">Choose a project</option>
                {projects.map((project) => <option key={project.folder} value={project.folder}>{project.title}</option>)}
              </Select>
            </Field>
          )}
          <SetupFields setup={setup} setSetup={setSetup} models={models} full />
          <Button
            variant="primary"
            className="w-full"
            disabled={busy || !setup.title.trim() || (!fromProject && !setup.premise.trim()) || (fromProject && !folder)}
            onClick={() => void start()}
          >
            Start
          </Button>
        </div>
      </div>
    </div>
  )
}
