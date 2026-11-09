import { create } from 'zustand'
import { toast } from 'sonner'
import type {
  AgentEvent,
  AgentStatus,
  ContextUsage,
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
import { useUi } from './ui'

// The same state and event handling as the computer's workspace, fed over the connection.

const MAX_LIVE_OUTPUT = 40_000

interface WorkspaceState {
  /** The project open on the computer right now. */
  project: ProjectInfo | null
  timeline: TimelineItem[]
  tasks: TaskRecord[]
  running: boolean
  status: AgentStatus
  prompt: PendingPrompt | null
  runCommand: string | null
  memory: string
  context: ContextUsage | null
  run: RunState | null
  /** Task whose steps and files are shown; null = the latest. */
  selectedTaskId: string | null

  /** Picks up the project open on the computer (after connecting). */
  resync: () => Promise<void>
  open: (path: string) => Promise<boolean>
  create: (name: string, parentDir: string | null) => Promise<boolean>
  /** Closes the project on the computer too. */
  close: () => Promise<void>
  send: (text: string) => Promise<boolean>
  stop: () => void
  answer: (answer: PromptAnswer) => void
  undo: (taskId?: string) => Promise<void>
  startRun: () => Promise<void>
  stopRun: () => Promise<void>
  askHowToRun: () => Promise<void>
  fixRun: () => Promise<void>
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
  run: null,
  memory: '',
  context: null,
  status: { phase: 'idle' }
} satisfies Partial<WorkspaceState>

export const useWorkspace = create<WorkspaceState>((set, get) => {
  const apply = (state: ProjectState, same = false): void => {
    set({
      project: state.project,
      timeline: state.timeline,
      tasks: state.tasks,
      running: state.running,
      prompt: state.prompt,
      runCommand: state.runCommand,
      memory: state.memory,
      context: state.context,
      status: same && state.running ? get().status : ({ phase: state.running ? 'working' : 'idle', text: 'Working…' } as AgentStatus),
      selectedTaskId: same ? get().selectedTaskId : null
    })
    void invoke('run:state', state.project.path).then((run) => set({ run }), () => undefined)
  }

  const isMine = (projectPath: string): boolean => get().project?.path.toLowerCase() === projectPath.toLowerCase()

  const closed = (): void => {
    set(CLOSED)
    if (useUi.getState().view === 'workspace') useUi.getState().go('projects')
  }

  on('agent:event', (e: AgentEvent) => {
    if (!isMine(e.projectPath)) return
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
      case 'delta':
        set((s) => ({
          timeline: s.timeline.map((x) => (x.id === e.itemId && x.kind === 'assistant' ? { ...x, text: x.text + e.text } : x))
        }))
        break
      case 'reasoning':
        set((s) => ({
          timeline: s.timeline.map((x) =>
            x.id === e.itemId && x.kind === 'assistant' ? { ...x, reasoning: (x.reasoning ?? '') + e.text } : x
          )
        }))
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
      case 'output':
        set((s) => ({
          tasks: s.tasks.map((t) =>
            t.steps.some((x) => x.id === e.stepId)
              ? { ...t, steps: t.steps.map((x) => (x.id === e.stepId ? { ...x, output: ((x.output ?? '') + e.chunk).slice(-MAX_LIVE_OUTPUT) } : x)) }
              : t
          )
        }))
        break
      case 'files':
        set((s) => ({ tasks: updateTask(s.tasks, e.taskId, (t) => ({ ...t, files: e.files })) }))
        break
      case 'prompt':
        set({ prompt: e.prompt })
        break
      case 'task-end':
        set((s) => ({ tasks: updateTask(s.tasks, e.task.id, () => e.task), running: false, prompt: null, status: { phase: 'idle' } }))
        void invoke('projects:open', e.projectPath).then((st) => set({ runCommand: st.runCommand }), () => undefined)
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
    }
  })

  on('run:event', (e: RunEvent) => {
    if (!isMine(e.projectPath)) return
    if (e.type === 'state') set({ run: e.state })
    else set((s) => (s.run ? { run: { ...s.run, output: (s.run.output + e.chunk).slice(-100_000) } } : {}))
  })

  // Opening, closing or undoing on the computer (or another phone) shows up here too.
  on('project:changed', (state) => {
    if (state) apply(state, isMine(state.project.path))
    else if (get().project) closed()
  })

  const withProject = async (fn: (path: string) => Promise<unknown>): Promise<void> => {
    const project = get().project
    if (!project) return
    try {
      await fn(project.path)
    } catch (e) {
      toast.error(errorMessage(e))
    }
  }

  return {
    ...CLOSED,
    runCommand: null,
    selectedTaskId: null,

    resync: async () => {
      try {
        const state = await invoke('projects:current')
        if (state) apply(state, isMine(state.project.path))
        else if (get().project) closed()
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    open: async (path) => {
      try {
        apply(await invoke('projects:open', path), isMine(path))
        useUi.getState().go('workspace')
        return true
      } catch (e) {
        toast.error(errorMessage(e))
        return false
      }
    },

    create: async (name, parentDir) => {
      try {
        apply(await invoke('projects:create', name, parentDir))
        useUi.getState().go('workspace')
        return true
      } catch (e) {
        toast.error(errorMessage(e))
        return false
      }
    },

    close: async () => {
      try {
        await invoke('projects:close')
        closed()
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    send: async (text) => {
      const project = get().project
      if (!project || !text.trim()) return false
      if (get().prompt?.kind === 'question') {
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

    stop: () => void withProject((p) => invoke('task:stop', p)),

    answer: (answer) => {
      const { project, prompt } = get()
      if (!project || !prompt) return
      set({ prompt: null })
      invoke('task:answer', project.path, prompt.id, answer).catch((e) => {
        // The answer didn't arrive; show the question again.
        set({ prompt })
        toast.error(errorMessage(e))
      })
    },

    undo: (taskId) =>
      withProject(async (p) => {
        apply(await invoke('task:undo', p, taskId), true)
        toast.success('Undone. Your project is back to how it was.')
      }),

    startRun: () => withProject(async (p) => set({ run: await invoke('run:start', p) })),
    stopRun: () => withProject(async (p) => set({ run: await invoke('run:stop', p) })),
    askHowToRun: () => withProject((p) => invoke('run:askAgent', p)),
    fixRun: () => withProject((p) => invoke('run:fix', p)),
    share: () => withProject((p) => invoke('run:share', p)),

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

    setRunCommand: (command) =>
      withProject(async (p) => set({ run: await invoke('run:setCommand', p, command), runCommand: command?.trim() || null })),

    selectTask: (selectedTaskId) => set({ selectedTaskId })
  }
})

/** The task whose steps and files are shown. */
export function shownTask(tasks: TaskRecord[], selectedId: string | null): TaskRecord | undefined {
  return (selectedId && tasks.find((t) => t.id === selectedId)) || tasks[tasks.length - 1]
}

/** The task "Undo last task" would undo. */
export function undoableTask(tasks: TaskRecord[]): TaskRecord | undefined {
  return [...tasks].reverse().find((t) => !t.undone && t.snapshot && t.outcome !== 'running')
}

/** The tasks going back to before `task` would undo: it and every later task not undone yet. */
export function tasksUndoneBy(tasks: TaskRecord[], task: TaskRecord): TaskRecord[] {
  return tasks.slice(tasks.indexOf(task)).filter((t) => !t.undone && t.outcome !== 'running')
}
