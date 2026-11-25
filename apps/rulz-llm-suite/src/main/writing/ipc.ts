// The phone's writing:* channels (contract: src/shared/phone.ts, WritingIpcInvoke / WritingIpcEvents).
// The phone dispatcher (src/main/phone) passes the caller's owner id first, then the invoke arguments; a job may be used by the owner that started it or by any owner that holds its folder (WritingRunsDeps.holds).
// src/main/index.ts joins this with the rest: build a WritingEvents, give its `emit` to WritingRuns, forward the
// events to the phone clients with `events.subscribe`, and answer PHONE_WRITING_CHANNELS with `writingHandlers(runs)`.
import type { WritingIpcInvoke, WritingJob, WritingRunRequest } from '@shared/phone'
import { parsePhoneModelId, parsePhoneMode, type PhoneModes } from './modeInfo'
import type { WritingRuns } from './runs'

export type WritingHandler = (ownerId: string, ...args: unknown[]) => Promise<unknown>

const str = (v: unknown, what: string): string => {
  if (typeof v !== 'string' || !v) throw new Error(`${what} is missing.`)
  return v
}

/** One handler per phone channel. Arguments are checked here; the phone is not trusted to send the right shapes. */
export function writingHandlers(runs: WritingRuns, modes?: PhoneModes): Record<keyof WritingIpcInvoke, WritingHandler> {
  const needModes = (): PhoneModes => {
    if (!modes) throw new Error('That is not available yet.')
    return modes
  }
  return {
    'writing:run': async (owner, req) => runs.run(req as WritingRunRequest, owner),
    'writing:jobs': async (owner, folder) => runs.list(str(folder, 'The project folder'), owner),
    'writing:job': async (owner, jobId) => runs.get(str(jobId, 'The job id'), owner),
    'writing:abort': async (owner, jobId) => void runs.abort(str(jobId, 'The job id'), owner),
    'writing:dismiss': async (owner, jobId) => void runs.dismiss(str(jobId, 'The job id'), owner),
    'writing:applyState': async (owner, jobId, keys) => runs.applyState(str(jobId, 'The job id'), Array.isArray(keys) ? keys.map(String) : [], owner),
    'writing:models': async () => runs.models(),
    // No project is needed: any phone may ask what applies to a mode, and ask for a reload.
    'writing:mode': async (_owner, mode, modelId) => needModes().info(parsePhoneMode(mode), parsePhoneModelId(modelId, true)),
    'writing:reloadMode': async (_owner, mode, modelId) => needModes().reloadMode(parsePhoneMode(mode), parsePhoneModelId(modelId, false) as string)
  }
}

/** Fan-out of 'writing:job' events: WritingRuns emits into it, the phone server subscribes. */
export class WritingEvents {
  private listeners = new Set<(job: WritingJob) => void>()

  /** Pass as WritingRunsDeps.emit. A listener that throws does not stop the others. */
  readonly emit = (job: WritingJob): void => {
    for (const l of [...this.listeners]) {
      try {
        l(job)
      } catch (e) {
        console.warn('Writing: a job event listener failed:', e)
      }
    }
  }

  /** Returns the function that stops listening. */
  subscribe(listener: (job: WritingJob) => void): () => void {
    this.listeners.add(listener)
    return () => void this.listeners.delete(listener)
  }
}
