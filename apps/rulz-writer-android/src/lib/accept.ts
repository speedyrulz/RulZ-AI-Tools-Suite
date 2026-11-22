import type { WritingAlt, WritingJob } from '@shared/phone'

// Accepting a prose result: the alternative's text is the whole scene as it would be after the action, made on
// the text the phone sent. It is only safe to apply when the phone's buffer is still that text.

export const STALE_MESSAGE = 'The text changed since this was written.'
export const WRONG_SCENE_MESSAGE = 'Open the scene this was written for first.'

export interface RunMemo {
  /** The buffer text the run was started with. */
  text: string
  baseHash: string
}

export interface AcceptInput {
  /** The file the editor holds, and the project and scene file the job was written for. */
  bufferFolder: string
  bufferPath: string
  expectedFolder: string
  expectedPath: string | null
  /** What the editor holds now. */
  bufferText: string
  bufferDirty: boolean
  /** The disk hash the editor is based on now. */
  bufferBaseHash: string
  job: WritingJob
  /** The text sent with the run, remembered on the phone; null when the phone no longer has it (app restarted). */
  memo: RunMemo | null
}

export type AcceptCheck = { ok: true } | { ok: false; reason: string }

/**
 * The accept check. With the remembered request text: the buffer must equal it. Without it (after a restart),
 * only a clean buffer on the same base as the job's is trusted.
 */
export function checkAccept(input: AcceptInput): AcceptCheck {
  const { job, memo } = input
  if (job.result?.kind !== 'prose') return { ok: false, reason: 'This result has no text to apply.' }
  // The buffer must be the scene file the result was written for, in the same project.
  if (!input.expectedPath || input.bufferFolder !== input.expectedFolder || input.bufferPath !== input.expectedPath) return { ok: false, reason: WRONG_SCENE_MESSAGE }
  if (memo) return input.bufferText === memo.text ? { ok: true } : { ok: false, reason: STALE_MESSAGE }
  if (!input.bufferDirty && input.bufferBaseHash === job.request.baseHash) return { ok: true }
  return { ok: false, reason: STALE_MESSAGE }
}

export function pickAlt(job: WritingJob, index: number): WritingAlt | null {
  if (job.result?.kind !== 'prose') return null
  return job.result.alts[index] ?? null
}

/** The sticky list a scene gets when a result's new sticky entries are accepted. */
export function mergeSticky(current: string[], added: string[]): string[] {
  const out = [...current]
  for (const id of added) if (!out.includes(id)) out.push(id)
  return out
}
