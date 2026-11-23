// What the parts of src/main/remote expect of each other (docs/vast-design.md). Types only.
import type { AutoStop, RemoteGpu, RemoteModelDef } from '@shared/remote'

export interface SshRoute {
  route: 'direct' | 'proxy'
  host: string
  port: number
}

/** What the ssh side needs to know about a rental; the rental manager answers. */
export interface RentalInfo {
  rentalId: number
  label: string
  origin: 'app' | 'attached'
  /** vast says it is running. */
  running: boolean
  /** Best first. Changes after a stop and start. */
  routes: SshRoute[]
  /** Engine tag to use there. */
  engineTag: string
  dollarsPerHour: number
}

export interface ProbeResult {
  gpus: RemoteGpu[]
  diskFreeBytes: number | null
  foreignProcesses: boolean
  /** Engine tags installed and checked. */
  engines: string[]
  watchdog: { armed: boolean; error?: string }
  /** Model ids whose files are complete there. */
  models: string[]
}

export interface TransferProgress {
  stage: string
  bytesDone: number
  bytesTotal: number
}

/** The ssh side (ssh.ts, scripts.ts, transfer.ts), as the rental manager uses it. Errors are sentences for the user. */
export interface RentalOps {
  /** Tries the routes, remembers the one that works, reads the instance's state. */
  probe(rentalId: number, signal?: AbortSignal): Promise<ProbeResult>
  /** Writes the limits, (re)starts the watchdog and checks that it can stop the instance. */
  armWatchdog(rentalId: number, limits: { autoStop: AutoStop; capAt: number | null; origin: 'app' | 'attached' }): Promise<{ armed: boolean; error?: string }>
  /** The `onstart` script of a rental made by the app: installs and starts the watchdog. Under 4000 characters. */
  onstartScript(limits: { autoStop: AutoStop }): string
  installEngine(rentalId: number, tag: string, onProgress: (p: TransferProgress) => void, signal: AbortSignal): Promise<void>
  /** Downloads or uploads every file of the model; resumes, checks, then marks it complete. */
  transferModel(rentalId: number, def: RemoteModelDef, onProgress: (p: TransferProgress) => void, signal: AbortSignal): Promise<void>
  removeModelFiles(rentalId: number, modelId: string): Promise<void>
  /** Bytes a setup of this model still has to fetch; null = unknown (a Hugging Face size that was not reported). */
  missingBytes(rentalId: number, def: RemoteModelDef): Promise<number | null>
  /** Refreshes keepalive once, now. */
  touch(rentalId: number): Promise<void>
  /** While true, keepalive is refreshed every 60 s over one held connection, re-made when it drops. */
  setInUse(rentalId: number, inUse: boolean): void
  /** Drops every connection to the rental (stopped, destroyed, detached, app closing). Synchronous. */
  disconnect(rentalId: number): void
  log(rentalId: number): string[]
}

/** The rental manager, as the ssh side and InstanceManager's wiring use it. */
export interface RentalDirectory {
  info(rentalId: number): RentalInfo | null
  /** Who uses a rental; called for every request, lane and setup. The returned function ends that use (once). */
  use(rentalId: number, what: string, by: 'window' | 'api' | 'phone'): () => void
  /** Adds a line to the rental's log (scrubbed of secrets by the manager). */
  note(rentalId: number, line: string): void
}

export interface Secrets {
  vastKey(): Promise<{ value: string; source: string } | null>
  hfToken(): Promise<{ value: string; source: string } | null>
  setVastKey(value: string): Promise<void>
  setHfToken(value: string): Promise<void>
  /** Replaces every known secret in `text` with ***. */
  scrub(text: string): string
}
