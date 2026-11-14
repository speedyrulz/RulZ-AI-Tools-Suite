import { create } from 'zustand'
import { toast } from 'sonner'
import type {
  AgentEvent,
  AgentStatus,
  ContextUsage,
  PendingNote,
  PendingPrompt,
  ProjectInfo,
  ProjectState,
  PromptAnswer,
  RunEvent,
  RunState,
  TaskRecord,
  TimelineItem
} from '@shared/types'
import { errorMessage, invoke, on } from '../lib/api'

const MAX_LIVE_OUTPUT = 40_000
/** Streamed text and command output are applied at most this often: each update re-renders the reply's Markdown. */
const FLUSH_MS = 80

interface WorkspaceState {
  project: ProjectInfo | null
  timeline: TimelineItem[]
  tasks: TaskRecord[]
  running: boolean
  status: AgentStatus
  prompt: PendingPrompt | null
  /** Notes waiting for the running task's next step. */
  notes: PendingNote[]
  runCommand: string | null
  /** The owner's "Things to always remember". */
  memory: string
  context: ContextUsage | null
  run: RunState | null
  /** Task whose steps and files the side panel shows; null = the latest. */
  selectedTaskId: string | null

  open: (path: string) => Promise<boolean>
  create: (name: string, parentDir: string | null) => Promise<boolean>
  close: () => Promise<void>
  /**
   * Starts a task, or answers my question when `promptId` (the question shown when the owner
   * pressed Send) is given. False when it didn't go through, so the typed text can stay.
   */
  send: (text: string, promptId?: string) => Promise<boolean>
  stop: () => void
  /** A note for the running task: before its next step, or at once (cutting the model's reply short) with `interrupt`. */
  addNote: (text: string, interrupt: boolean) => Promise<boolean>
  cancelNote: (id: string) => void
  /** Cuts the model's current reply short so the waiting notes apply now. */
  interrupt: () => void
  answer: (answer: PromptAnswer) => void
  /** Undoes the last task, or with an id goes back to before that task (undoing every later one too). */
  undo: (taskId?: string) => Promise<void>
  /** Deletes a message (a request takes its replies with it). */
  deleteItem: (itemId: string) => Promise<void>
  /** Runs the latest request again, edited when `text` is given, after undoing its file changes. */
  redo: (taskId: string, text?: string) => Promise<void>
  startRun: () => Promise<void>
  stopRun: () => Promise<void>
  askHowToRun: () => Promise<void>
  /** "Fix this": stops the app if needed and asks the agent to fix the error it showed. */
  fixRun: () => Promise<void>
  /** "Share my app": asks the agent to package the app to give to other people. */
  share: () => Promise<void>
  setMemory: (text: string) => Promise<boolean>
  setRunCommand: (command: string | null) => Promise<void>
  selectTask: (id: string | null) => void
}

const updateTask = (tasks: TaskRecord[], id: string, fn: (t: TaskRecord) => TaskRecord): TaskRecord[] =>
  tasks.map((t) => (t.id === id ? fn(t) : t))

const CLOSED = {
  project: null,
  timeline: [],
  tasks: [],
  running: false,
  prompt: null,
  notes: [],
  run: null,
  memory: '',
  context: null,
  status: { phase: 'idle' }
} satisfies Partial<WorkspaceState>

export const useWorkspace = create<WorkspaceState>((set, get) => {
  /** Shows a project's state; for the project already shown, `same` keeps the selected task and the live status. */
  const apply = (state: ProjectState, same = false): void => {
    dropPending()
    set({
      project: state.project,
      timeline: state.timeline,
      tasks: state.tasks,
      running: state.running,
      prompt: state.prompt,
      notes: state.notes ?? [],
      runCommand: state.runCommand,
      memory: state.memory,
      context: state.context,
      status: same && state.running ? get().status : ({ phase: state.running ? 'working' : 'idle', text: 'Working…' } as AgentStatus),
      selectedTaskId: same ? get().selectedTaskId : null,
      // Another project's Run panel (its output, Stop, Fix this) mustn't stay up for this one.
      ...(same ? {} : { run: null })
    })
    const path = state.project.path
    void invoke('run:state', path).then(
      (run) => {
        // Another project may have been opened while this was on its way.
        if (isMine(path)) set({ run })
      },
      () => undefined
    )
  }

  const isMine = (projectPath: string): boolean => get().project?.path.toLowerCase() === projectPath.toLowerCase()

  // ---- Streamed text, thoughts and command output are batched, then applied in one update ----
  let pendingText = new Map<string, { text: string; reasoning: string }>()
  let pendingOutput = new Map<string, string>()
  let flushTimer: ReturnType<typeof setTimeout> | null = null
  const flush = (): void => {
    if (flushTimer) clearTimeout(flushTimer)
    flushTimer = null
    if (!pendingText.size && !pendingOutput.size) return
    const text = pendingText
    const output = pendingOutput
    pendingText = new Map()
    pendingOutput = new Map()
    set((s) => ({
      timeline: text.size
        ? s.timeline.map((x) => {
            const add = text.get(x.id)
            if (!add || x.kind !== 'assistant') return x
            return { ...x, text: x.text + add.text, ...(add.reasoning ? { reasoning: (x.reasoning ?? '') + add.reasoning } : {}) }
          })
        : s.timeline,
      tasks: output.size
        ? s.tasks.map((t) =>
            t.steps.some((x) => output.has(x.id))
              ? { ...t, steps: t.steps.map((x) => (output.has(x.id) ? { ...x, output: ((x.output ?? '') + output.get(x.id)).slice(-MAX_LIVE_OUTPUT) } : x)) }
              : t
          )
        : s.tasks
    }))
  }
  const later = (): void => {
    flushTimer ??= setTimeout(flush, FLUSH_MS)
  }
  /** Drops what was batched for a project that is no longer shown. */
  const dropPending = (): void => {
    if (flushTimer) clearTimeout(flushTimer)
    flushTimer = null
    pendingText = new Map()
    pendingOutput = new Map()
    // Run output isn't replaced by the project's state, so it is applied rather than dropped
    // (another project's panel is cleared right after).
    flushRun()
  }

  // "Run my app" output too: a busy dev server can send many small pieces a second.
  let pendingRun = ''
  let runTimer: ReturnType<typeof setTimeout> | null = null
  const flushRun = (): void => {
    if (runTimer) clearTimeout(runTimer)
    runTimer = null
    if (!pendingRun) return
    const chunk = pendingRun
    pendingRun = ''
    set((s) => (s.run ? { run: { ...s.run, output: (s.run.output + chunk).slice(-100_000) } } : {}))
  }

  on('agent:event', (e: AgentEvent) => {
    if (!isMine(e.projectPath)) return
    if (e.type === 'delta' || e.type === 'reasoning') {
      const p = pendingText.get(e.itemId) ?? { text: '', reasoning: '' }
      if (e.type === 'delta') p.text += e.text
      else p.reasoning += e.text
      pendingText.set(e.itemId, p)
      return later()
    }
    if (e.type === 'output') {
      pendingOutput.set(e.stepId, (pendingOutput.get(e.stepId) ?? '') + e.chunk)
      return later()
    }
    // Everything else applies after the text that came before it.
    flush()
    switch (e.type) {
      case 'task-start':
        set((s) => ({ tasks: [...s.tasks, e.task], timeline: [...s.timeline, e.item], running: true, selectedTaskId: null }))
        break
      case 'status':
        set({ status: e.status })
        break
      case 'item':
        set((s) => {
          const i = s.timeline.findIndex((x) => x.id === e.item.id)
          if (i < 0) return { timeline: [...s.timeline, e.item] }
          if (e.item.kind === 'assistant' && !e.item.text && !e.item.reasoning) return { timeline: s.timeline.filter((x) => x.id !== e.item.id) }
          const timeline = s.timeline.slice()
          timeline[i] = e.item
          return { timeline }
        })
        break
      case 'step':
        set((s) => ({
          tasks: updateTask(s.tasks, e.step.taskId, (t) => {
            const i = t.steps.findIndex((x) => x.id === e.step.id)
            const steps = t.steps.slice()
            if (i < 0) steps.push(e.step)
            else steps[i] = { ...e.step, output: e.step.output ?? steps[i].output }
            return { ...t, steps }
          })
        }))
        break
      case 'files':
        set((s) => ({ tasks: updateTask(s.tasks, e.taskId, (t) => ({ ...t, files: e.files })) }))
        break
      case 'prompt':
        set({ prompt: e.prompt })
        break
      case 'task-end':
        set((s) => ({ tasks: updateTask(s.tasks, e.task.id, () => e.task), running: false, prompt: null, notes: [], status: { phase: 'idle' } }))
        // The event carries the way to run the app (a "how do I run this?" task may have found it);
        // an older main process didn't send it, so the project is read again then.
        if (e.runCommand !== undefined) set({ runCommand: e.runCommand })
        else void invoke('projects:open', e.projectPath).then((st) => set({ runCommand: st.runCommand }), () => undefined)
        break
      case 'tasks':
        set({ tasks: e.tasks })
        break
      case 'context':
        set({ context: e.context })
        break
      case 'draft':
        set((s) => ({ tasks: updateTask(s.tasks, e.taskId, (t) => ({ ...t, draft: e.draft })) }))
        break
      case 'notes':
        set({ notes: e.notes })
        break
    }
  })

  on('run:event', (e: RunEvent) => {
    if (!isMine(e.projectPath)) return
    if (e.type === 'state') {
      // The output that came before a new state belongs to it.
      flushRun()
      set({ run: e.state })
    } else {
      pendingRun += e.chunk
      runTimer ??= setTimeout(flushRun, FLUSH_MS)
    }
  })

  // The open project can change from the phone app (or after Undo); every screen follows it.
  on('project:changed', (state) => {
    if (state) apply(state, isMine(state.project.path))
    else if (get().project) set(CLOSED)
  })

  return {
    project: null,
    timeline: [],
    tasks: [],
    running: false,
    status: { phase: 'idle' },
    prompt: null,
    notes: [],
    runCommand: null,
    memory: '',
    context: null,
    run: null,
    selectedTaskId: null,

    open: async (path) => {
      try {
        apply(await invoke('projects:open', path))
        return true
      } catch (e) {
        toast.error(errorMessage(e))
        return false
      }
    },

    create: async (name, parentDir) => {
      try {
        apply(await invoke('projects:create', name, parentDir))
        return true
      } catch (e) {
        toast.error(errorMessage(e))
        return false
      }
    },

    close: async () => {
      try {
        await invoke('projects:close')
        set(CLOSED)
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    send: async (text, promptId) => {
      const project = get().project
      if (!project || !text.trim()) return false
      const prompt = get().prompt
      if (promptId !== undefined || prompt?.kind === 'question') {
        // Only the question the owner was answering: it may have been answered on the phone, and
        // another one may be waiting now.
        if (!prompt || prompt.kind !== 'question' || prompt.id !== promptId) {
          toast.error(promptId ? 'That question was already answered (from your phone?). What you typed is still in the box.' : 'I asked you something meanwhile: read my question first.')
          return false
        }
        get().answer({ kind: 'question', text: text.trim() })
        return true
      }
      try {
        await invoke('task:start', project.path, text)
        return true
      } catch (e) {
        toast.error(errorMessage(e))
        return false
      }
    },

    stop: () => {
      const project = get().project
      if (project) void invoke('task:stop', project.path)
    },

    addNote: async (text, interrupt) => {
      const project = get().project
      if (!project || !text.trim()) return false
      try {
        await invoke('task:note', project.path, text, interrupt)
        return true
      } catch (e) {
        toast.error(errorMessage(e))
        return false
      }
    },

    cancelNote: (id) => {
      const project = get().project
      if (project) void invoke('task:cancelNote', project.path, id)
    },

    interrupt: () => {
      const project = get().project
      if (project) void invoke('task:interrupt', project.path)
    },

    answer: (answer) => {
      const { project, prompt } = get()
      if (!project || !prompt) return
      set({ prompt: null })
      void invoke('task:answer', project.path, prompt.id, answer)
    },

    deleteItem: async (itemId) => {
      const project = get().project
      if (!project) return
      try {
        apply(await invoke('task:deleteItem', project.path, itemId), true)
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    redo: async (taskId, text) => {
      const project = get().project
      if (!project) return
      try {
        apply(await invoke('task:redo', project.path, taskId, text), true)
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    undo: async (taskId) => {
      const project = get().project
      if (!project) return
      try {
        apply(await invoke('task:undo', project.path, taskId))
        toast.success('Undone. Your project is back to how it was.')
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    startRun: async () => {
      const project = get().project
      if (!project) return
      try {
        set({ run: await invoke('run:start', project.path) })
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    stopRun: async () => {
      const project = get().project
      if (project) set({ run: await invoke('run:stop', project.path) })
    },

    askHowToRun: async () => {
      const project = get().project
      if (!project) return
      try {
        await invoke('run:askAgent', project.path)
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    fixRun: async () => {
      const project = get().project
      if (!project) return
      try {
        await invoke('run:fix', project.path)
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    share: async () => {
      const project = get().project
      if (!project) return
      try {
        await invoke('run:share', project.path)
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    setMemory: async (text) => {
      const project = get().project
      if (!project) return false
      try {
        await invoke('project:setMemory', project.path, text)
        set({ memory: text.trim() })
        return true
      } catch (e) {
        toast.error(errorMessage(e))
        return false
      }
    },

    setRunCommand: async (command) => {
      const project = get().project
      if (!project) return
      set({ run: await invoke('run:setCommand', project.path, command), runCommand: command?.trim() || null })
    },

    selectTask: (selectedTaskId) => set({ selectedTaskId })
  }
})

/** The task the side panel shows. */
export function shownTask(tasks: TaskRecord[], selectedId: string | null): TaskRecord | undefined {
  return (selectedId && tasks.find((t) => t.id === selectedId)) || tasks[tasks.length - 1]
}

/** The task "Undo last task" would undo. */
export function undoableTask(tasks: TaskRecord[]): TaskRecord | undefined {
  return [...tasks].reverse().find((t) => !t.undone && t.snapshot && t.outcome !== 'running')
}

/** The tasks going back to before `task` would undo: it and every later task not undone yet. */
export function tasksUndoneBy(tasks: TaskRecord[], task: TaskRecord): TaskRecord[] {
  // By id: the list is replaced with fresh copies whenever the project changes (from the phone, say).
  const i = tasks.findIndex((t) => t.id === task.id)
  return i < 0 ? [] : tasks.slice(i).filter((t) => !t.undone && t.outcome !== 'running')
}
