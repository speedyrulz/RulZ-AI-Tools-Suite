import { App } from '@capacitor/app'
import { registerPlugin } from '@capacitor/core'
import { LocalNotifications } from '@capacitor/local-notifications'
import { create } from 'zustand'
import { alertFor, DEFAULT_ALERT_PREFS, statusLine, type AlertAction, type AlertPrefs } from '../lib/alerts'
import { isNative, loadPref, savePref } from '../lib/native'
import { link } from '../lib/remote'
import { currentComputer, useLink } from './link'
import { useWorkspace } from './workspace'

// Phone notifications when a task finishes or the agent waits for the owner, and the background
// connection (ConnectionService in android/) that keeps them coming while the app is closed. Its
// ongoing notification shows what the agent is doing right now.
//
// On the phone, notifications are posted by KeepAlivePlugin (android/): the local-notifications
// plugin treats every notification as an exact alarm and first asks for that permission, which
// can't be done from the background. That plugin only asks for permission to show notifications,
// and shows them when the app runs in a desktop browser.

interface KeepAlive {
  start(o: { title: string; text: string }): Promise<void>
  update(o: { title: string; text: string }): Promise<{ running: boolean }>
  stop(): Promise<void>
  setBusy(o: { busy: boolean }): Promise<void>
  notify(o: { id: number; channel: string; title: string; body: string }): Promise<void>
  cancel(o: { id: number }): Promise<void>
  notificationStatus(): Promise<{ allowed: boolean; problem?: string | null }>
  batteryStatus(): Promise<{ unrestricted: boolean }>
  openBatterySettings(): Promise<void>
  openNotificationSettings(): Promise<void>
}
const KeepAlive = registerPlugin<KeepAlive>('KeepAlive')

export type Permission = 'granted' | 'denied' | 'prompt' | 'unknown'

export interface LastAlert {
  title: string
  at: number
  /** Why it didn't show, if it didn't. */
  error?: string
}

interface AlertsState {
  prefs: AlertPrefs
  permission: Permission
  /** Why notifications can't show right now (a channel turned off, …), from Android. */
  problem: string
  /** Android lets the app run without battery limits. */
  unrestricted: boolean | null
  /** Why the background connection couldn't start, if it couldn't. */
  backgroundError: string
  last: LastAlert | null
  init: () => Promise<void>
  setPrefs: (patch: Partial<AlertPrefs>) => Promise<void>
  /** Asks Android for permission to show notifications (once it says no, only its settings can change that). */
  askPermission: () => Promise<Permission>
  refreshStatus: () => Promise<void>
  test: () => Promise<void>
  openNotificationSettings: () => void
  openBatterySettings: () => void
}

let active = true
let finishedCount = 0
let started = false

const wantsAlerts = (p: AlertPrefs): boolean => p.finished || p.waiting
const message = (e: unknown): string => (e instanceof Error ? e.message : String(e))

export const useAlerts = create<AlertsState>((set, get) => {
  const show = async (actions: AlertAction[]): Promise<void> => {
    for (const a of actions) {
      try {
        if (a.kind === 'cancel') {
          if (isNative) await KeepAlive.cancel({ id: a.id })
          else await LocalNotifications.cancel({ notifications: [{ id: a.id }] })
          continue
        }
        if (a.channel === 'finished') finishedCount++
        if (isNative) await KeepAlive.notify({ id: a.id, channel: a.channel, title: a.title, body: a.body })
        else await LocalNotifications.schedule({ notifications: [{ id: a.id, title: a.title, body: a.body, isExactNotification: false }] })
        set({ last: { title: a.title, at: Date.now() } })
      } catch (e) {
        if (a.kind === 'show') set({ last: { title: a.title, at: Date.now(), error: message(e) } })
      }
    }
  }

  /** Starts or stops the background connection to match the settings and the chosen computer. */
  const syncBackground = async (): Promise<void> => {
    if (!isNative) return
    const { prefs } = get()
    try {
      if (currentComputer() && prefs.background && wantsAlerts(prefs)) {
        lastStatus = ''
        const s = currentStatus()
        await KeepAlive.start(s)
        lastStatus = JSON.stringify(s)
        await KeepAlive.setBusy({ busy: useWorkspace.getState().running })
      } else {
        await KeepAlive.stop()
      }
      set({ backgroundError: '' })
    } catch (e) {
      set({ backgroundError: message(e) })
    }
  }

  const currentStatus = (): { title: string; text: string } => {
    const ws = useWorkspace.getState()
    const state = useLink.getState().state
    return statusLine({
      computer: currentComputer()?.name ?? 'your computer',
      link: state.kind === 'connected' ? 'connected' : state.kind === 'connecting' ? 'connecting' : 'stopped',
      project: ws.project?.name,
      running: ws.running,
      task: [...ws.tasks].reverse().find((t) => t.outcome === 'running')?.title,
      doing: ws.status.phase === 'idle' ? undefined : ws.status.text,
      prompt: ws.prompt
    })
  }

  // The ongoing notification follows what the agent does, at most about once a second.
  let lastStatus = ''
  let statusTimer: ReturnType<typeof setTimeout> | null = null
  const refreshOngoing = (): void => {
    if (!isNative || statusTimer) return
    statusTimer = setTimeout(() => {
      statusTimer = null
      const s = currentStatus()
      const key = JSON.stringify(s)
      if (key === lastStatus) return
      lastStatus = key
      void KeepAlive.update(s).catch(() => undefined)
    }, 1000)
  }

  return {
    prefs: DEFAULT_ALERT_PREFS,
    permission: 'unknown',
    problem: '',
    unrestricted: null,
    backgroundError: '',
    last: null,

    init: async () => {
      if (started) return
      started = true
      set({ prefs: { ...DEFAULT_ALERT_PREFS, ...(await loadPref<Partial<AlertPrefs>>('alerts', {})) } })

      if (isNative) {
        void App.addListener('appStateChange', ({ isActive }) => {
          active = isActive
          // Starting the background connection is only allowed while the app is open.
          if (isActive) {
            void syncBackground()
            void get().refreshStatus()
          }
        })
      } else {
        document.addEventListener('visibilitychange', () => (active = document.visibilityState === 'visible'))
      }

      link.on('agent:event', (e) => {
        if (e.type === 'task-start' || e.type === 'task-end') void KeepAlive.setBusy({ busy: e.type === 'task-start' }).catch(() => undefined)
        const actions = alertFor(e, {
          prefs: get().prefs,
          projectName: useWorkspace.getState().project?.name ?? 'your project',
          timeline: useWorkspace.getState().timeline,
          finishedCount
        })
        // While the app is open the conversation and the approval sheet show it already; only clear old ones.
        void show(active ? actions.filter((a) => a.kind === 'cancel') : actions)
      })

      useWorkspace.subscribe((s, prev) => {
        if (s.running !== prev.running || s.prompt !== prev.prompt || s.status !== prev.status || s.project !== prev.project) refreshOngoing()
      })
      useLink.subscribe((s, prev) => {
        if (s.currentId !== prev.currentId) void syncBackground()
        else if (s.state !== prev.state) refreshOngoing()
      })
      // Ask once, when the app first connects to a computer.
      link.onState((s) => {
        if (s.kind !== 'connected') return
        if (active) void syncBackground()
        void loadPref<boolean>('alertsAsked', false).then(async (asked) => {
          if (asked || !wantsAlerts(get().prefs)) return
          await savePref('alertsAsked', true)
          await get().askPermission()
        })
      })
      await get().refreshStatus()
    },

    setPrefs: async (patch) => {
      const prefs = { ...get().prefs, ...patch }
      set({ prefs })
      await savePref('alerts', prefs)
      if ((patch.finished || patch.waiting) && get().permission !== 'granted') await get().askPermission()
      await syncBackground()
    },

    askPermission: async () => {
      try {
        const r = await LocalNotifications.requestPermissions()
        const permission: Permission = r.display === 'granted' ? 'granted' : r.display === 'denied' ? 'denied' : 'prompt'
        set({ permission })
        void get().refreshStatus()
        return permission
      } catch {
        return get().permission
      }
    },

    refreshStatus: async () => {
      try {
        const r = await LocalNotifications.checkPermissions()
        set({ permission: r.display === 'granted' ? 'granted' : r.display === 'denied' ? 'denied' : 'prompt' })
      } catch {
        set({ permission: 'unknown' })
      }
      if (!isNative) return
      try {
        set({ problem: (await KeepAlive.notificationStatus()).problem ?? '' })
      } catch {
        set({ problem: '' })
      }
      try {
        set({ unrestricted: (await KeepAlive.batteryStatus()).unrestricted })
      } catch {
        set({ unrestricted: null })
      }
    },

    test: async () => {
      if (get().permission !== 'granted' && (await get().askPermission()) !== 'granted') {
        set({ last: { title: 'Test notification', at: Date.now(), error: 'Android didn’t give permission to show notifications.' } })
        return
      }
      await show([{ kind: 'show', id: 99, channel: 'waiting', title: 'Notifications work', body: 'This is how RulZ Agent will tell you a task is done or needs you.' }])
    },

    openNotificationSettings: () => void KeepAlive.openNotificationSettings().catch(() => undefined),
    openBatterySettings: () => void KeepAlive.openBatterySettings().catch(() => undefined)
  }
})
