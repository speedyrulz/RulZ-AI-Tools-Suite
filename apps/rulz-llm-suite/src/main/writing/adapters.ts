// Binds WritingRuns' dependencies to main's real services. Kept apart from runs.ts so the service itself has no
// import of the chat service, the store, the library or the instance manager (tests use fakes).
import type { LoadedInstance, StreamEvent, WireMessage } from '@shared/types'
import type { ChatService } from '../chat/service'
import type { InstanceManager } from '../engine/instances'
import type { ModelLibrary } from '../models/library'
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

export function chatAdapter(chat: ChatService, tap: ChatEventTap): WritingChat {
  return {
    countText: (modelId, text) => chat.countText(modelId, text),
    countTokens: (modelId, messages: WireMessage[]) => chat.countTokens(modelId, messages, null),
    start: (req) => chat.start(req),
    abort: (id) => chat.abort(id),
    subscribe: (id, handler) => tap.subscribe(id, handler)
  }
}

/**
 * A sentence when the chat service has a request in flight (the desktop or the API is generating), else null.
 * WritingRuns asks only while no phone job runs, so a running request belongs to the desktop or the API.
 */
export function chatBusyReason(chat: ChatService, benchRunning?: () => boolean): () => string | null {
  return () => {
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

const modelOf = (i: LoadedInstance): WritingModel => ({ modelId: i.modelId, instanceId: i.instanceId, contextLength: i.contextLength })

/** The model library and instance manager as the Writing page's `useModels.ensure` and `readyInstance` use them. */
export function modelsAdapter(library: ModelLibrary, instances: InstanceManager): WritingModels {
  return {
    list: () => library.list().map((m) => ({ id: m.id, name: m.name })),
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
    ensure: async (modelId) => modelOf((await instances.ensure(modelId)).instance)
  }
}
