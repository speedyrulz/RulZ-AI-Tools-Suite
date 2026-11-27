// Opening and closing the project, one at a time. A double click on a recent project, or the desktop and the phone
// both opening one, must not run two switches at once (both would pass the "close the old one" step and the first new
// session would be dropped without being closed, leaving its background commands running).

/** How long closing the old project may take (stopping its app, its background commands, the last save). */
export const CLEANUP_WAIT_MS = 20_000
/** After this many times CLEANUP_WAIT_MS a cleanup that still has not ended no longer blocks opening a project. */
const LINGER_LIMIT = 6

/** Runs the given functions one after the other. A function that fails does not stop the ones queued behind it. */
export class TransitionQueue {
  private tail: Promise<unknown> = Promise.resolve()

  run<T>(fn: () => Promise<T>): Promise<T> {
    const result = this.tail.then(fn)
    this.tail = result.then(
      () => undefined,
      () => undefined
    )
    return result
  }
}

/** `promise`, or a rejection with `message` when it has not finished after `ms` (the work itself is not cancelled). */
export function withTimeout<T>(promise: Promise<T>, ms: number, message: string): Promise<T> {
  let timer: NodeJS.Timeout | undefined
  const late = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(message)), ms)
  })
  return Promise.race([promise, late]).finally(() => clearTimeout(timer))
}

export interface SwitchSession {
  isRunning: boolean
  /** What Undo, Redo or Delete is doing ("the undo"), or null. */
  busyWith: string | null
  /** Refuses every operation from now on (close() also does this, but only once the app has stopped). */
  markClosed(): void
  close(): Promise<void>
}

export interface SwitchHooks<S extends SwitchSession, R extends { stop(): Promise<unknown> }, T> {
  /** The open project's session and app runner now (null when none). */
  current: () => { session: S | null; runner: R | null }
  /** Lets go of the open project at once: nothing can use it from here on. */
  release: () => void
  /** Throws a sentence when something (a batch run) forbids switching now. */
  guard: () => void
  /** Cancels uploads under way for the session's project. */
  dropUploads: (session: S) => void
  /** The state of the project when `abs` is already the open one, else null. */
  already: (abs: string) => T | null
  /** Makes the new project the open one and returns the state of the session it created. Closes nothing. */
  create: (abs: string) => Promise<T>
  /** For tests: how long closing may take. */
  cleanupWaitMs?: number
}

/** Serialises project switches; see the file's comment. */
export class ProjectSwitcher<S extends SwitchSession, R extends { stop(): Promise<unknown> }, T> {
  private readonly queue = new TransitionQueue()

  constructor(private readonly hooks: SwitchHooks<S, R, T>) {}

  open(abs: string): Promise<T> {
    return this.queue.run(async () => {
      const same = this.hooks.already(abs)
      if (same) return same
      await this.closeNow()
      return this.hooks.create(abs)
    })
  }

  close(): Promise<void> {
    return this.queue.run(() => this.closeNow())
  }

  /** Not queued: open() calls it from inside the queue, where waiting for a queued close would never end. */
  private async closeNow(): Promise<void> {
    const h = this.hooks
    const wait = h.cleanupWaitMs ?? CLEANUP_WAIT_MS
    // An earlier switch whose cleanup ran out of time may still be saving or stopping commands: nothing new opens
    // (the same project again included) until it is done.
    if (this.lingering) await withTimeout(this.lingering, wait, 'The last project is still closing. Try again in a moment.')
    h.guard()
    const { session, runner } = h.current()
    if (session?.isRunning) throw new Error('A task is still running. Press Stop first.')
    // An Undo, Redo or Delete under way works on this project's files: closing now would let the next project's
    // session start while it still runs.
    if (session?.busyWith) throw new Error(`Wait for ${session.busyWith} to finish first.`)
    if (session) h.dropUploads(session)
    // Let go before the awaits below, so nothing (a batch run, a task) can start in the project meanwhile.
    h.release()
    session?.markClosed()
    // The app is stopped, then the session closed (also when stopping the app failed).
    const cleanup = (async () => {
      try {
        await runner?.stop()
      } finally {
        await session?.close()
      }
    })()
    const tracked: Promise<unknown> = cleanup
      .catch(() => undefined)
      .finally(() => {
        if (this.lingering === tracked) this.lingering = null
      })
    this.lingering = tracked
    // A cleanup that never ends must not block every later open until the app restarts: after a while it is let go.
    setTimeout(() => {
      if (this.lingering === tracked) this.lingering = null
    }, wait * LINGER_LIMIT).unref?.()
    // Not opening a replacement while this may still change files: a cleanup that hangs fails the switch.
    await withTimeout(cleanup, wait, "The project's app or its commands did not stop in time, so no other project was opened. Open the project again in a moment.")
  }

  /** The cleanup of the last closed project while it has not finished. */
  private lingering: Promise<unknown> | null = null

  /** Settles when the last closed project's cleanup has finished (at once when there is none): quitting waits for it. */
  cleanupDone(): Promise<unknown> {
    return this.lingering ?? Promise.resolve()
  }
}
