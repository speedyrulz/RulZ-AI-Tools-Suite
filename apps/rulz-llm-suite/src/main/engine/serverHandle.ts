// What InstanceManager needs from a running llama-server, wherever it runs (docs/vast-design.md, section 7).
import type { InstanceAdapter, LoadConfig, ModelEntry } from '@shared/types'
import type { RemoteHost } from '@shared/remote'

export interface ServerExit {
  code: number | null
  /** A sentence that explains the exit better than the code ("The connection to the instance was lost."). */
  reason?: string
}

export interface ServerHandle {
  /** Local port that answers as the llama-server. */
  readonly port: number
  /** Pid of a local process; null for a remote one. */
  readonly pid: number | null
  /** Where it runs; undefined = this PC. */
  readonly host?: RemoteHost
  /** Lines are not split; the caller feeds them to its own splitter. Output from before the call is delivered too. */
  onOutput(fn: (chunk: string) => void): void
  /** Called once, also when it has already exited. */
  onExit(fn: (exit: ServerExit) => void): void
  /** True while requests may fail because the connection is being re-made; the server itself still runs. */
  reconnecting(): boolean
  /** Ends the server and resolves once it is really gone, or rejects saying why it may still run. */
  stop(): Promise<void>
  /** App shutdown, synchronous: a local server is killed, a remote one is left running and only disconnected. */
  detachSync(): void
}

export interface RemoteLaunch {
  model: ModelEntry
  config: LoadConfig
  alias: string
  adapters: InstanceAdapter[]
  instanceId: string
  /** Preflight warnings and progress for the instance's log. */
  note: (line: string) => void
}

/** Starts llama-server on a rental. Rejects with a sentence when the rental is not running or not set up. */
export interface RemoteLauncher {
  start(rentalId: number, launch: RemoteLaunch): Promise<ServerHandle>
}
