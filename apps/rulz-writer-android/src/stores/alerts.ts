import { App } from '@capacitor/app'
import { registerPlugin } from '@capacitor/core'
import { LocalNotifications } from '@capacitor/local-notifications'
import { create } from 'zustand'
import { alertForJob, DEFAULT_ALERT_PREFS, statusLine, type Alert, type AlertPrefs } from '../lib/alerts'
import { isRunning, jobLine, newlyFinished } from '../lib/jobs'
import { isNative, loadPref, savePref } from '../lib/native'
import { link } from '../lib/remote'
import { useJobs } from './jobs'
import { currentComputer, useLink } from './link'
import { useProject } from './project'

// Phone notifications when a writing job finishes, and the background connection (ConnectionService in
// android/) that keeps them coming while the app is closed. Its ongoing notification shows what is going on.
//
// On the phone, notifications are posted by KeepAlivePlugin (android/): the local-notifications plugin treats
// every notification as an exact alarm and first asks for that permission, which can't be done from the
// background. That plugin only asks for permission to show notifications, and shows them when the app runs in a
// desktop browser.

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
  error?: string
}

interface AlertsState {
  prefs: AlertPrefs
  permission: Permission
  problem: string
  unrestricted: boolean | null
  backgroundError: string
  last: LastAlert | null
  init: () => Promise<void>
  setPrefs: (patch: Partial<AlertPrefs>) => Promise<void>
  askPermission: () => Promise<Permission>
  refreshStatus: () => Promise<void>
  test: () => Promise<void>
  openNotificationSettings: () => void
  openBatterySettings: () => void
}

let active = true
let finishedCount = 0
let started = false
const message = (e: unknown): string => (e instanceof Error ? e.message : String(e))
const anyRunning = (): boolean => Object.values(useJobs.getState().jobs).some(isRunning)

export const useAlerts = create<AlertsState>((set, get) => {
  const show = async (a: Alert): Promise<void> => {
    try {
      finishedCount++
      if (isNative) await KeepAlive.notify({ id: a.id, channel: a.channel, title: a.title, body: a.body })
      else await LocalNotifications.schedule({ notifications: [{ id: a.id, title: a.title, body: a.body, isExactNotification: false }] })
      set({ last: { title: a.title, at: Date.now() } })
    } catch (e) {
      set({ last: { title: a.title, at: Date.now(), error: message(e) } })
    }
  }

  const currentStatus = (): { title: string; text: string } => {
    const state = useLink.getState().state
    const project = useProject.getState().state
    const running = project ? Object.values(useJobs.getState().jobs).find((j) => j.folder === project.folder && isRunning(j)) : undefined
    return statusLine({
      computer: currentComputer()?.name ?? 'your computer',
      link: state.kind === 'connected' ? 'connected' : state.kind === 'connecting' ? 'connecting' : 'stopped',
      project: project?.project.title,
      job: running ? jobLine(running) : undefined
    })
  }

  let lastStatus = ''
  /** Starts or stops the background connection to match the settings and the chosen computer. */
  const syncBackground = async (): Promise<void> => {
    if (!isNative) return
    const { prefs } = get()
    try {
      if (currentComputer() && prefs.background && prefs.finished) {
        const s = currentStatus()
        await KeepAlive.start(s)
        lastStatus = JSON.stringify(s)
        await KeepAlive.setBusy({ busy: link.connected && anyRunning() })
      } else {
        await KeepAlive.stop()
      }
      set({ backgroundError: '' })
    } catch (e) {
      set({ backgroundError: message(e) })
    }
  }

  // The ongoing notification follows what happens, at most about once a second.
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

      useJobs.subscribe((s, prev) => {
        // While the app is open the job card shows it already; only notify when it is in the background.
        if (!active && get().prefs.finished) {
          for (const job of newlyFinished(prev.jobs, s.jobs)) {
            const alert = alertForJob(job, useProject.getState().state?.project.title ?? 'your project', finishedCount)
            if (alert) void show(alert)
          }
        }
        const was = Object.values(prev.jobs).some(isRunning)
        if (isNative && anyRunning() !== was) void KeepAlive.setBusy({ busy: anyRunning() && link.connected }).catch(() => undefined)
        refreshOngoing()
      })
      useLink.subscribe((s, prev) => {
        if (s.currentId !== prev.currentId) void syncBackground()
        else if (s.state !== prev.state) {
          if (isNative && s.state.kind !== prev.state.kind) void KeepAlive.setBusy({ busy: s.state.kind === 'connected' && anyRunning() }).catch(() => undefined)
          refreshOngoing()
        }
      })
      // Ask once, when the app first connects to a computer.
      link.onState((s) => {
        if (s.kind !== 'connected') return
        if (active) void syncBackground()
        void loadPref<boolean>('alertsAsked', false).then(async (asked) => {
          if (asked || !get().prefs.finished) return
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
      if (patch.finished && get().permission !== 'granted') await get().askPermission()
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
      await show({ id: 99, channel: 'finished', title: 'Notifications work', body: 'This is how RulZ Writer will tell you a writing job is done.' })
    },

    openNotificationSettings: () => void KeepAlive.openNotificationSettings().catch(() => undefined),
    openBatterySettings: () => void KeepAlive.openBatterySettings().catch(() => undefined)
  }
})
