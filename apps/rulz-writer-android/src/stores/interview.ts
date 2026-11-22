import { create } from 'zustand'
import { cardRequest } from '../lib/runs'
import { loadPref, savePref } from '../lib/native'
import { useJobs } from './jobs'

// The character interview: the conversation is kept on the phone (the computer holds only the job that
// answers the last question). A finished interview job is added to its chat as the character's answer and then
// dismissed, so an answer that arrives while the app was closed is still picked up.

export interface Turn {
  role: 'user' | 'assistant'
  text: string
  /** The answer could not be written. */
  failed?: boolean
}

const MAX_TURNS = 60
const PENDING = 'pending'
const omit = <T,>(o: Record<string, T>, key: string): Record<string, T> => {
  const next = { ...o }
  delete next[key]
  return next
}
export const chatKey = (folder: string, characterId: string): string => `${folder}\n${characterId}`

interface InterviewStore {
  chats: Record<string, Turn[]>
  /** Chat key → id of the job that is writing its next answer. */
  waiting: Record<string, string>
  init: () => Promise<void>
  /** Sends a question. Returns false when the job couldn't start (the input is kept). */
  ask: (folder: string, sceneId: string | null, characterId: string, question: string) => Promise<boolean>
  clear: (folder: string, characterId: string) => void
}

export const useInterview = create<InterviewStore>((set, get) => {
  let write: ReturnType<typeof setTimeout> | null = null
  const persist = (): void => {
    if (write) clearTimeout(write)
    write = setTimeout(() => void savePref('interviews', get().chats), 500)
  }
  const adopted = new Set<string>()

  const adopt = (): void => {
    // A wait whose job is gone (computer restarted, jobs cleared, a listing without it) can't end by itself.
    const jobs = useJobs.getState().jobs
    const stale = Object.entries(get().waiting).filter(([, id]) => id !== PENDING && !jobs[id])
    if (stale.length) set((s) => ({ waiting: Object.fromEntries(Object.entries(s.waiting).filter(([k]) => !stale.some(([sk]) => sk === k))) }))
    for (const job of Object.values(useJobs.getState().jobs)) {
      if (job.request.action !== 'interview' || adopted.has(job.id)) continue
      if (job.status !== 'done' && job.status !== 'error' && job.status !== 'stopped') continue
      adopted.add(job.id)
      const characterId = job.request.interview?.characterId
      if (!characterId) continue
      const key = chatKey(job.folder, characterId)
      const turn: Turn | null =
        job.result?.kind === 'interview'
          ? { role: 'assistant', text: job.result.reply.trim() }
          : job.status === 'error'
            ? { role: 'assistant', text: job.error ?? 'The answer could not be written.', failed: true }
            : null
      set((s) => {
        const waiting = { ...s.waiting }
        if (waiting[key] === job.id) delete waiting[key]
        const turns = s.chats[key] ?? []
        return { waiting, chats: turn ? { ...s.chats, [key]: [...turns, turn].slice(-MAX_TURNS) } : s.chats }
      })
      persist()
      void useJobs.getState().dismiss(job.id)
    }
  }

  return {
    chats: {},
    waiting: {},

    init: async () => {
      set({ chats: await loadPref<Record<string, Turn[]>>('interviews', {}) })
      useJobs.subscribe(adopt)
    },

    ask: async (folder, sceneId, characterId, question) => {
      const key = chatKey(folder, characterId)
      const turns: Turn[] = [...(get().chats[key] ?? []).filter((t) => !t.failed), { role: 'user', text: question.trim() }]
      // Marked before the request, so a second tap can't start a second job while the first is being sent.
      set((s) => ({ waiting: { ...s.waiting, [key]: PENDING } }))
      const job = await useJobs.getState().run(
        cardRequest(folder, 'interview', null, { sceneId, interview: { characterId, turns: turns.map((t) => ({ role: t.role, text: t.text })) } })
      )
      if (!job) {
        set((s) => ({ waiting: omit(s.waiting, key) }))
        return false
      }
      set((s) => ({ chats: { ...s.chats, [key]: turns.slice(-MAX_TURNS) }, waiting: { ...s.waiting, [key]: job.id } }))
      persist()
      return true
    },

    clear: (folder, characterId) => {
      const key = chatKey(folder, characterId)
      set((s) => ({ chats: omit(s.chats, key), waiting: omit(s.waiting, key) }))
      persist()
    }
  }
})
