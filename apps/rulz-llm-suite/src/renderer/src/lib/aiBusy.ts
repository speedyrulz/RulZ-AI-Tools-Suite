// One shared answer to "is a model busy writing something?". Only one thing at a time may use the
// loaded model: a chat reply, a batch run, a hardware benchmark, a Writing action or a writing benchmark.
// The stores are read when the function is called, never at import time, so stores that import this
// file do not run into import cycles.
import { isCloudModelId } from '@shared/cloud'
import { activeBenchRun, useBench } from '../stores/bench'
import { useBatch } from '../stores/batch'
import { isChatBusy, useChat } from '../stores/chat'
import { useNovel } from '../stores/novel'
import { useWbench } from '../stores/wbench'

export type AiBusySource = 'writing' | 'wbench'

/**
 * Why a new AI job cannot start now, as a short sentence, or null when nothing is running.
 * `except` names the caller's own kind of job, which does not count against itself. `opts.local`: the caller works on a model
 * on this PC (a reload), so a chat reply from an OpenRouter model does not count: it uses nothing here.
 */
export function aiBusyReason(except?: AiBusySource, opts?: { local?: boolean }): string | null {
  const batch = useBatch.getState()
  if (batch.activeId || batch.starting) return 'A batch run is running.'
  const chat = useChat.getState()
  if (isChatBusy(chat) || chat.handoff) {
    // A reply and a summary for a new chat can be under way; both must be at OpenRouter.
    const used = [isChatBusy(chat) ? (chat.genConversation?.modelId ?? chat.autoLoadingModelId) : undefined, chat.handoff?.modelId].filter((x) => x !== undefined)
    const cloudOnly = !!opts?.local && used.length > 0 && used.every((id) => !!id && isCloudModelId(id))
    if (!cloudOnly) return 'A chat reply is running.'
  }
  const bench = useBench.getState()
  if (bench.starting || activeBenchRun(bench.runs)) return 'A hardware benchmark is running.'
  if (except !== 'writing') {
    const n = useNovel.getState() as { gen?: unknown; checkRun?: unknown; interviewRun?: unknown }
    if (n.gen || n.checkRun || n.interviewRun) return 'A Writing action is running.'
  }
  if (except !== 'wbench' && useWbench.getState().running) return 'A writing benchmark is running.'
  return null
}

/** True while anything uses the model. */
export const aiBusy = (except?: AiBusySource): boolean => aiBusyReason(except) !== null
