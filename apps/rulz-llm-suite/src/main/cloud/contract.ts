// What the rest of the main process sees of the OpenRouter side (docs/cloud-design.md).
import type { CloudIpcInvoke } from '@shared/cloud'
import type { ModelEntry } from '@shared/types'

/** Who asks: the window (the user at this PC), another app through the API server, or a phone. */
export type CloudCaller = 'window' | 'api' | 'phone'

export interface CloudFetchInit {
  method?: string
  /** The llama-server style JSON body. */
  body?: string
  signal?: AbortSignal
}

/** What InstanceManager needs. */
export interface CloudGateway {
  /** Throws a sentence when the model cannot be connected now (no key saved, not in the list any more). */
  check(modelId: string): void
  /** What the model reports once connected; read again whenever the user changes its limits. */
  props(modelId: string): { contextLength: number; hasVision: boolean; free: boolean }
  /**
   * Answers as a llama-server would: GET /props, POST /tokenize, /apply-template, /v1/chat/completions and
   * /v1/chat/completions/input_tokens; anything else is a 400. Failures are Responses too ({error: {message, type,
   * code: CloudErrorKind}}); it rejects only when `init.signal` is aborted. A caller other than 'window' is refused
   * (403 cloud_not_allowed) unless the model is ticked for other apps: checked here, on every request.
   */
  fetch(modelId: string, path: string, init: CloudFetchInit, by: CloudCaller): Promise<Response>
}

export interface CloudService extends CloudGateway {
  /** The user's cloud models as catalog entries (`cloud` set, `remote` not). */
  models(): ModelEntry[]
  /** Whether a key is saved. */
  hasKey(): boolean
  /** Whether other apps may use a model now (ticked, and a key is saved). */
  allowedForApps(modelId: string): boolean
  /** 'cloud:*' channels for the window. */
  handlers: { [K in keyof CloudIpcInvoke]: (...args: Parameters<CloudIpcInvoke[K]>) => Promise<ReturnType<CloudIpcInvoke[K]>> }
  load(): Promise<void>
  /** App shutdown: aborts what runs; open reservations stay on disk and are settled at the next start. */
  dispose(): void
  /** An error's text without the key. */
  scrub(err: unknown): string
}

export interface CloudServiceDeps {
  userData: string
  safeStorage: { isEncryptionAvailable(): boolean; encryptString(text: string): Buffer; decryptString(data: Buffer): string }
  onChange: (state: import('@shared/cloud').CloudState) => void
  /** The list of cloud models (or one's limits) changed. */
  onModels: () => void
  /** Ejects a connected model (removed, or the key was forgotten). */
  eject: (modelId: string) => Promise<void>
  /** Requests using a connected model right now. */
  usersOf: (modelId: string) => number
  /** Forgets the per-mode settings of a removed model. */
  forgetDefaults: (modelId: string) => Promise<void>
  appVersion: string
  /** Injectable for tests. */
  fetch?: typeof fetch
  now?: () => number
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>
}
