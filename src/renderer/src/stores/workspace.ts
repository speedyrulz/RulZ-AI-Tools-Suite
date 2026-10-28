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

const MAX_LIVE_OUTPUT = 40_000

interface WorkspaceState {
  project: ProjectInfo | null
  timeline: TimelineItem[]
  tasks: TaskRecord[]
  running: boolean
  status: AgentStatus
  prompt: PendingPrompt | null
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
  send: (text: string) => Promise<void>
  stop: () => void
  answer: (answer: PromptAnswer) => void
  /** Undoes the last task, or with an id goes back to before that task (undoing every later one too). */
  undo: (taskId?: string) => Promise<void>
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
  run: null,
  memory: '',
  context: null,
  status: { phase: 'idle' }
} satisfies Partial<WorkspaceState>

export const useWorkspace = create<WorkspaceState>((set, get) => {
  /** Shows a project's state; for the project already shown, `same` keeps the selected task and the live status. */
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
          if (e.item.kind === 'assistant' && !e.item.text) return { timeline: s.timeline.filter((x) => x.id !== e.item.id) }
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
    }
  })

  on('run:event', (e: RunEvent) => {
    if (!isMine(e.projectPath)) return
    if (e.type === 'state') set({ run: e.state })
    else set((s) => (s.run ? { run: { ...s.run, output: (s.run.output + e.chunk).slice(-100_000) } } : {}))
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

    send: async (text) => {
      const project = get().project
      if (!project || !text.trim()) return
      const prompt = get().prompt
      if (prompt?.kind === 'question') {
        get().answer({ kind: 'question', text: text.trim() })
        return
      }
      try {
        await invoke('task:start', project.path, text)
      } catch (e) {
        toast.error(errorMessage(e))
      }
    },

    stop: () => {
      const project = get().project
      if (project) void invoke('task:stop', project.path)
    },

    answer: (answer) => {
      const { project, prompt } = get()
      if (!project || !prompt) return
      set({ prompt: null })
      void invoke('task:answer', project.path, prompt.id, answer)
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
  return tasks.slice(tasks.indexOf(task)).filter((t) => !t.undone && t.outcome !== 'running')
}
