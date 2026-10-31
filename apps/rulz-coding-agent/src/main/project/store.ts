// Recent projects and per-project history (conversation, tasks, steps), kept in the app's data folder.
import { createHash } from 'node:crypto'
import { join, resolve } from 'node:path'
import type { ContextUsage, ProjectInfo, TaskRecord, TimelineItem, WireMessage } from '@shared/types'
import { readJson, writeJsonAtomic } from '../util/jsonFile'

export interface ProjectData {
  version: 1
  path: string
  runCommand: string | null
  /** Summary of conversation that no longer fits. */
  summary: string
  /** The conversation with the model (without the system prompt). */
  messages: WireMessage[]
  /** Notes for the model, added to the next request (e.g. "the owner undid the last task"). */
  notes: string[]
  timeline: TimelineItem[]
  tasks: TaskRecord[]
  /** The owner's "Things to always remember", always in the system prompt. */
  memory: string
  /** Last measured context use, for the meter. */
  context: ContextUsage | null
}

const MAX_TIMELINE = 600
const MAX_TASKS = 60
const MAX_STEP_OUTPUT = 6000
const MAX_RECENT = 30

export const projectKey = (path: string): string => createHash('sha1').update(resolve(path).toLowerCase()).digest('hex').slice(0, 16)

export const emptyProject = (path: string): ProjectData => ({
  version: 1,
  path,
  runCommand: null,
  summary: '',
  messages: [],
  notes: [],
  timeline: [],
  tasks: [],
  memory: '',
  context: null
})

/** Trims what is saved: old timeline items, old tasks, long step output. */
export function trimForSave(data: ProjectData): ProjectData {
  return {
    ...data,
    timeline: data.timeline.slice(-MAX_TIMELINE),
    tasks: data.tasks.slice(-MAX_TASKS).map((t) => ({
      ...t,
      steps: t.steps.map((s) => (s.output && s.output.length > MAX_STEP_OUTPUT ? { ...s, output: `…${s.output.slice(-MAX_STEP_OUTPUT)}` } : s))
    }))
  }
}

export class ProjectStore {
  constructor(private readonly dir: string) {}

  private file = (path: string): string => join(this.dir, 'projects', `${projectKey(path)}.json`)
  private recentFile = (): string => join(this.dir, 'recent-projects.json')

  snapshotDir(path: string): string {
    return join(this.dir, 'snapshots', `${projectKey(path)}.git`)
  }

  backupDir(path: string): string {
    return join(this.dir, 'backups', projectKey(path))
  }

  async load(path: string): Promise<ProjectData> {
    const data = await readJson<ProjectData | null>(this.file(path), null)
    if (!data || data.version !== 1) return emptyProject(path)
    // A task can't still be running after a restart.
    for (const t of data.tasks) {
      if (t.outcome === 'running') {
        t.outcome = 'stopped'
        for (const s of t.steps) if (s.status === 'running' || s.status === 'waiting') s.status = 'skipped'
      }
    }
    return { ...emptyProject(path), ...data, path }
  }

  save(data: ProjectData): Promise<void> {
    return writeJsonAtomic(this.file(data.path), trimForSave(data), { compact: true })
  }

  async recent(): Promise<ProjectInfo[]> {
    const list = await readJson<ProjectInfo[]>(this.recentFile(), [])
    return Array.isArray(list) ? list.sort((a, b) => b.lastOpened - a.lastOpened) : []
  }

  async touch(info: ProjectInfo): Promise<ProjectInfo[]> {
    const key = resolve(info.path).toLowerCase()
    const list = [info, ...(await this.recent()).filter((p) => resolve(p.path).toLowerCase() !== key)].slice(0, MAX_RECENT)
    await writeJsonAtomic(this.recentFile(), list)
    return list
  }

  async forget(path: string): Promise<ProjectInfo[]> {
    const key = resolve(path).toLowerCase()
    const list = (await this.recent()).filter((p) => resolve(p.path).toLowerCase() !== key)
    await writeJsonAtomic(this.recentFile(), list)
    return list
  }
}
