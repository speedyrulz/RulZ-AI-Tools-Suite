// Binds WritingRuns' dependencies to main's real services. Kept apart from runs.ts so the service itself has no
// import of the chat service, the store, the library or the instance manager (tests use fakes).
import type { LoadedInstance, ModelEntry, StreamEvent, WireMessage } from '@shared/types'
import type { ChatService } from '../chat/service'
import type { InstanceManager } from '../engine/instances'
import type { NovelStore } from '../novel/store'
import type { WritingChat, WritingModel, WritingModels, WritingStore } from './runs'

/**
 * Hands the chat service's stream events to whoever waits for a request id. ChatService sends every event to the
 * window through one callback; wrap that callback so it also calls `tap.push(event)`:
 *   new ChatService(instances, (e) => { sendToWindow(e); tap.push(e) }, ...)
 */
export class ChatEventTap {
  private handlers = new Map<string, (e: StreamEvent) => void>()

  push(e: StreamEvent): void {
    this.handlers.get(e.requestId)?.(e)
  }

  subscribe(requestId: string, handler: (e: StreamEvent) => void): () => void {
    this.handlers.set(requestId, handler)
    return () => {
      if (this.handlers.get(requestId) === handler) this.handlers.delete(requestId)
    }
  }
}

/**
 * `by` says whose requests these are, so a rental can tell what keeps it running: 'window' for runs the user started
 * here, 'phone' for runs a phone started (and adventure jobs on a phone's behalf).
 */
export function chatAdapter(chat: ChatService, tap: ChatEventTap, by: 'window' | 'phone' = 'window'): WritingChat {
  return {
    countText: (modelId, text) => chat.countText(modelId, text),
    countTokens: (modelId, messages: WireMessage[]) => chat.countTokens(modelId, messages, null),
    start: (req) => chat.start(req, by),
    abort: (id) => chat.abort(id),
    subscribe: (id, handler) => tap.subscribe(id, handler)
  }
}

/**
 * A sentence when the chat service has a request in flight (the desktop or the API is generating), else null.
 * WritingRuns asks only while no phone job runs, so a running request belongs to the desktop or the API.
 */
export function chatBusyReason(chat: ChatService, benchRunning?: () => boolean, gate?: { busyReason(): string | null }): () => string | null {
  return () => {
    // A job of ours that is preparing (or between two requests) has no chat request in flight; the gate knows it runs.
    const held = gate?.busyReason()
    if (held) return held
    if (chat.busy) return 'LLM Suite is writing something else on the computer.'
    if (benchRunning?.()) return 'LLM Suite is running a benchmark on the computer.'
    return null
  }
}

/** NovelStore as a run reads it. */
export function storeAdapter(store: NovelStore): WritingStore {
  return {
    hold: async (folder, owner) => void (await store.openProject(folder, owner)),
    release: (folder, owner) => store.closeProject(folder, owner),
    projectState: (folder) => store.projectState(folder),
    readFile: (folder, rel) => store.readFile(folder, rel),
    statePatch: (folder, ops) => store.statePatch(folder, ops)
  }
}

const modelOf = (i: LoadedInstance): WritingModel => ({
  modelId: i.modelId,
  instanceId: i.instanceId,
  contextLength: i.contextLength,
  ...(i.host ? { rental: { label: i.host.label, reconnecting: !!i.reconnecting } } : {})
})

/**
 * The model catalog (the library, plus the models on rentals) and instance manager as the Writing page's
 * `useModels.ensure` and `readyInstance` use them.
 */
export function modelsAdapter(library: { list(): ModelEntry[] }, instances: InstanceManager): WritingModels {
  return {
    list: () => library.list().map((m) => ({ id: m.id, name: m.name, ...(m.remote ? { remote: true } : {}) })),
    ready: (modelId) => {
      if (modelId) {
        const i = instances.readyFor(modelId)
        return i ? modelOf(i) : undefined
      }
      // No model chosen for the project: the desktop uses its active model, which is the newest loaded one here.
      const ready = instances.list().filter((i) => i.status === 'ready')
      ready.sort((a, b) => b.startedAt - a.startedAt)
      return ready[0] ? modelOf(ready[0]) : undefined
    },
    ensure: async (modelId, mode) => {
      // A run never starts anything on a rental: it uses the model where it is already loaded, or says so.
      const entry = library.list().find((m) => m.id === modelId)
      if (entry?.remote) {
        const ready = instances.readyFor(modelId)
        if (!ready) throw new Error(`Load ${entry.name} on a rental first (vast.ai page).`)
        return modelOf(ready)
      }
      return modelOf((await instances.ensure(modelId, mode)).instance)
    }
  }
}
