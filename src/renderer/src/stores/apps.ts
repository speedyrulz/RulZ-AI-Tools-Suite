import { toast } from 'sonner'
import { create } from 'zustand'
import type { AppView, JobState, SelfUpdate, RequirementId, RequirementStatus, Settings, ThemeName } from '@shared/types'
import { errorMessage, invoke, on } from '../lib/api'
import { applyThemeClass, initialTheme } from '../lib/theme'

const LOG_LIMIT = 3000
const FINISHED = new Set(['done', 'error', 'cancelled'])
let initialized = false

export function isActive(job: JobState | null | undefined): job is JobState {
  return !!job && !FINISHED.has(job.phase)
}

interface AppsState {
  apps: AppView[]
  loaded: boolean
  refreshing: boolean
  /** Latest job state per app, kept after a failure until dismissed. */
  jobs: Record<string, JobState>
  logs: Record<string, string[]>
  requirements: RequirementStatus[]
  settings: Settings | null
  selectedId: string | null
  /** The prerequisite whose install guide is open. */
  guideId: RequirementId | null
  /** The theme in use; settings.theme stays unset until the first-launch choice. */
  theme: ThemeName
  /** A newer launcher release to offer, and the install progress (null before it starts, -1 on failure). */
  selfUpdate: SelfUpdate | null
  selfProgress: number | null
  /** True while Update all works through the apps one at a time. */
  updatingAll: boolean

  init: () => Promise<void>
  refresh: () => Promise<void>
  checkRequirements: (reloadPath?: boolean) => Promise<void>
  select: (id: string | null) => void
  dismissJob: (id: string) => void
  updateAll: () => Promise<void>
  updateSettings: (patch: Partial<Settings>) => Promise<void>
  setTheme: (theme: ThemeName) => Promise<void>
}

const ACTION_DONE: Record<JobState['action'], string> = {
  install: 'installed',
  update: 'updated',
  repair: 'repaired',
  uninstall: 'removed'
}

export const useApps = create<AppsState>((set, get) => ({
  apps: [],
  loaded: false,
  refreshing: false,
  jobs: {},
  logs: {},
  requirements: [],
  settings: null,
  selectedId: null,
  guideId: null,
  theme: initialTheme,
  selfUpdate: null,
  selfProgress: null,
  updatingAll: false,

  init: async () => {
    // StrictMode runs effects twice in development; subscribe only once.
    if (initialized) return
    initialized = true
    on('apps:changed', (apps) => set({ apps }))
    // A batch of lines (pip can print hundreds a second) is one store update.
    on('job:log', (lines) =>
      set((s) => {
        const logs = { ...s.logs }
        for (const { appId, line, replace } of lines) {
          const prev = logs[appId] ?? []
          const kept = replace ? prev.slice(0, -1) : prev
          logs[appId] = kept.length >= LOG_LIMIT ? [...kept.slice(kept.length - LOG_LIMIT + 1), line] : [...kept, line]
        }
        return { logs }
      })
    )
    on('job:update', (job) => {
      const prev = get().jobs[job.appId]
      // A new job starts with a fresh log.
      if (!isActive(prev) && isActive(job)) set((s) => ({ logs: { ...s.logs, [job.appId]: [] } }))
      set((s) => ({ jobs: { ...s.jobs, [job.appId]: job } }))
      if (isActive(prev) || !prev) notify(job, get().apps.find((a) => a.app.id === job.appId)?.app.name ?? job.appId)
      // Something has to be installed by hand: walk them through it.
      if (job.phase === 'error' && job.missing?.length) {
        set({ guideId: job.missing[0] })
        void get().checkRequirements()
      }
      // A prerequisite may have just been installed.
      if (job.phase === 'done') void get().checkRequirements()
      if (job.phase === 'done' || job.phase === 'cancelled') {
        setTimeout(() => {
          if (get().jobs[job.appId] === job) get().dismissJob(job.appId)
        }, 2500)
      }
    })
    const [apps, settings] = await Promise.all([invoke('apps:list'), invoke('settings:get')])
    set({ apps, settings, loaded: true })
    on('self:progress', (f) => set({ selfProgress: f }))
    if (settings.checkUpdatesOnStart) {
      void invoke('self:check').then((u) => {
        if (u && u.tag !== get().settings?.skippedUpdate) set({ selfUpdate: u })
      })
    }
    void get().checkRequirements()
  },

  refresh: async () => {
    set({ refreshing: true })
    try {
      set({ apps: await invoke('apps:refresh') })
      // A manual check also offers a newer launcher, even one that was skipped before.
      const u = await invoke('self:check', true)
      if (u) set({ selfUpdate: u })
    } catch (e) {
      toast.error(errorMessage(e))
    } finally {
      set({ refreshing: false })
    }
    void get().checkRequirements()
  },

  checkRequirements: async (reloadPath) => {
    set({ requirements: await invoke('requirements:check', reloadPath) })
  },

  select: (id) => set({ selectedId: id }),

  dismissJob: (id) =>
    set((s) => {
      const jobs = { ...s.jobs }
      delete jobs[id]
      return { jobs }
    }),

  updateAll: async () => {
    if (get().updatingAll) return
    set({ updatingAll: true })
    // Each app is tried once, even if it fails and still shows an update.
    const tried = new Set<string>()
    let updated = 0
    let failed = 0
    try {
      // One at a time: several setup programs running at once tend to get in each other's way.
      for (;;) {
        const next = get().apps.find((v) => updatable(v) && !tried.has(v.app.id) && !isActive(get().jobs[v.app.id] ?? v.job))
        if (!next) break
        const id = next.app.id
        tried.add(id)
        get().dismissJob(id)
        try {
          await invoke('apps:update', id)
        } catch (e) {
          toast.error(`${next.app.name}: ${errorMessage(e)}`)
          failed++
          continue
        }
        const phase = await waitForJob(id)
        if (phase === 'done') updated++
        else failed++
        // Cancelling one update stops the rest too.
        if (phase === 'cancelled') break
      }
    } finally {
      set({ updatingAll: false })
    }
    if (failed) toast.warning(`Updated ${updated} of ${updated + failed} apps`)
    else if (updated > 1) toast.success(`Updated ${updated} apps`)
  },

  updateSettings: async (patch) => {
    set({ settings: await invoke('settings:update', patch) })
  },

  setTheme: async (theme) => {
    applyThemeClass(theme)
    set({ theme })
    await get().updateSettings({ theme })
  }
}))

/** Installed, set up and behind its latest release. */
export function updatable(view: AppView): boolean {
  return view.updateAvailable && !!view.installed && !view.installed.needsSetup
}

/** Resolves with the final phase once the app's job ends. */
function waitForJob(id: string): Promise<JobState['phase']> {
  return new Promise((resolve) => {
    let seen = false
    const check = (s: AppsState): boolean => {
      const job = s.jobs[id]
      if (job && FINISHED.has(job.phase)) return finish(job.phase)
      if (job) seen = true
      // Finished jobs are dismissed after a moment; gone after being seen means it ended well.
      else if (seen) return finish('done')
      return false
    }
    const finish = (phase: JobState['phase']): true => {
      unsubscribe?.()
      resolve(phase)
      return true
    }
    let unsubscribe: (() => void) | undefined
    if (!check(useApps.getState())) unsubscribe = useApps.subscribe(check)
  })
}

function notify(job: JobState, name: string): void {
  if (job.phase === 'done') toast.success(`${name} ${ACTION_DONE[job.action]}`)
  else if (job.phase === 'error') toast.error(`${name}: ${job.error ?? 'failed'}`)
}

/** Runs an IPC action and shows its error as a toast. */
export async function act(fn: () => Promise<unknown>): Promise<void> {
  try {
    await fn()
  } catch (e) {
    toast.error(errorMessage(e))
  }
}
