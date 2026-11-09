import type { EventChannel, InvokeChannel, IpcEvents, IpcInvoke } from '@shared/ipc'
import { link } from './remote'

/** Typed request to RulZ Coding Agent on the computer (the same requests its own window makes). */
export function invoke<K extends InvokeChannel>(channel: K, ...args: Parameters<IpcInvoke[K]>): Promise<Awaited<ReturnType<IpcInvoke[K]>>> {
  return link.invoke(channel, ...args)
}

export function on<K extends EventChannel>(channel: K, listener: (payload: IpcEvents[K]) => void): () => void {
  return link.on(channel, listener)
}

export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}
