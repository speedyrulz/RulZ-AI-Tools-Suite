// Writing mode, renderer side. The request building, fitting and result shaping live in src/shared/novelRun.ts
// (main's WritingRuns uses them too); this file re-exports them and binds the renderer's IPC for token counts and generation.
import type { StreamEvent, WireMessage } from '@shared/types'
import {
  makeIpcCounter,
  startGeneration as startGenerationWith,
  templateOverheadWith,
  type CountFn,
  type GenBackend,
  type GenHandle,
  type GenInput
} from '@shared/novelRun'
import { invoke, on } from './api'
import { uid } from './format'

export * from '@shared/novelRun'

export const ipcCounter = makeIpcCounter((modelId, text) => invoke('chat:countText', modelId, text))

/** The chat template's own tokens around a system + user pair, counted through IPC. Cached per instance. */
export function templateOverhead(instanceId: string, modelId: string, count: CountFn): Promise<number> {
  return templateOverheadWith((m, messages) => invoke('chat:countTokens', m, messages as WireMessage[], null), instanceId, modelId, count)
}

const streams = new Map<string, (e: StreamEvent) => void>()
let subscribed = false

function ensureSubscribed(): void {
  if (subscribed) return
  subscribed = true
  on('chat:event', (e) => streams.get(e.requestId)?.(e))
}

const backend: GenBackend = {
  start: (req) => invoke('chat:start', req),
  abort: (requestId) => invoke('chat:abort', requestId),
  subscribe: (requestId, handler) => {
    ensureSubscribed()
    streams.set(requestId, handler)
    return () => void streams.delete(requestId)
  },
  newId: uid
}

export function startGeneration(input: GenInput): GenHandle {
  return startGenerationWith(input, backend)
}
