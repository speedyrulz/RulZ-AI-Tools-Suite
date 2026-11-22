import { toast } from 'sonner'
import { create } from 'zustand'
import type { WritingJob, WritingRunRequest } from '@shared/phone'
import type { RunMemo } from '../lib/accept'
import { errorMessage } from '../lib/format'
import { isRunning, jobsOf, reduceJobs, type JobMap } from '../lib/jobs'
import { loadPref, savePref } from '../lib/native'
import { link } from '../lib/remote'

// Writing jobs run on the computer. This keeps the phone's view of them: events update it, and a reconnect or
// opening a project asks the computer for everything it still holds. For prose jobs the phone also remembers
// the text it sent, so Accept can tell whether the buffer is still what the result was written for.

type StoredMemo = RunMemo & { at: number }
const MEMO_DAYS = 7

interface JobsStore {
  jobs: JobMap
  memos: Record<string, StoredMemo>
  /** When each job last reached the phone (event or run answer), and when each was dismissed here. */
  receivedAt: Record<string, number>
  removedAt: Record<string, number>
  loaded: boolean
  init: () => Promise<void>
  /** Starts a job. Returns it, or null when the computer refused (the reason is shown). */
  run: (req: WritingRunRequest) => Promise<WritingJob | null>
  abort: (id: string) => Promise<void>
  dismiss: (id: string) => Promise<void>
  /** Asks the computer for the jobs it holds for a project (after opening it and after a reconnect). */
  refresh: (folder: string) => Promise<void>
  clear: () => void
  memoOf: (id: string) => RunMemo | null
}

export const useJobs = create<JobsStore>((set, get) => {
  const saveMemos = (): void => void savePref('runMemos', get().memos)

  return {
    jobs: {},
    memos: {},
    receivedAt: {},
    removedAt: {},
    loaded: false,

    init: async () => {
      const stored = await loadPref<Record<string, StoredMemo>>('runMemos', {})
      const cutoff = Date.now() - MEMO_DAYS * 86_400_000
      set({ memos: Object.fromEntries(Object.entries(stored).filter(([, m]) => m.at > cutoff)), loaded: true })
      link.on('writing:job', (job) => set((s) => ({ jobs: reduceJobs(s.jobs, { type: 'update', job }), receivedAt: { ...s.receivedAt, [job.id]: Date.now() } })))
    },

    run: async (req) => {
      try {
        const job = await link.invoke('writing:run', req)
        set((s) => ({
          jobs: reduceJobs(s.jobs, { type: 'update', job }),
          receivedAt: { ...s.receivedAt, [job.id]: Date.now() },
          memos: { ...s.memos, [job.id]: { text: req.text, baseHash: req.baseHash, at: Date.now() } }
        }))
        saveMemos()
        return job
      } catch (e) {
        toast.error(errorMessage(e))
        return null
      }
    },

    abort: async (id) => {
      try {
        await link.invoke('writing:abort', id)
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    dismiss: async (id) => {
      // Gone from the screen at once; the computer forgets it a moment later (or on the next listing).
      set((s) => {
        const memos = { ...s.memos }
        delete memos[id]
        return { jobs: reduceJobs(s.jobs, { type: 'remove', id }), memos, removedAt: { ...s.removedAt, [id]: Date.now() } }
      })
      saveMemos()
      try {
        await link.invoke('writing:dismiss', id)
      } catch {
        // If the computer didn't hear it, the job comes back with the next listing and can be dismissed again.
      }
    },

    refresh: async (folder) => {
      const startedAt = Date.now()
      try {
        const jobs = await link.invoke('writing:jobs', folder)
        set((s) => ({ jobs: reduceJobs(s.jobs, { type: 'load', folder, jobs, startedAt, receivedAt: s.receivedAt, removedAt: s.removedAt }) }))
      } catch {
        // The next reconnect asks again.
      }
    },

    clear: () => set({ jobs: reduceJobs(get().jobs, { type: 'clear' }) }),

    memoOf: (id) => {
      const m = get().memos[id]
      return m ? { text: m.text, baseHash: m.baseHash } : null
    }
  }
})

export function useProjectJobs(folder: string | undefined): WritingJob[] {
  const jobs = useJobs((s) => s.jobs)
  return folder ? jobsOf(jobs, folder) : []
}

export function anyJobRunning(jobs: JobMap): boolean {
  return Object.values(jobs).some(isRunning)
}
