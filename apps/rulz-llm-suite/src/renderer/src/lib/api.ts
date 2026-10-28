import type { EventChannel, InvokeChannel, IpcEvents, IpcInvoke } from '@shared/ipc'

/** Typed IPC call; strips Electron's "Error invoking remote method" prefix from errors. */
export async function invoke<K extends InvokeChannel>(
  channel: K,
  ...args: Parameters<IpcInvoke[K]>
): Promise<Awaited<ReturnType<IpcInvoke[K]>>> {
  try {
    return await window.rulz.invoke(channel, ...args)
  } catch (e) {
    throw new Error(errorMessage(e))
  }
}

export function on<K extends EventChannel>(channel: K, listener: (payload: IpcEvents[K]) => void): () => void {
  return window.rulz.on(channel, listener)
}

export function errorMessage(e: unknown): string {
  const msg = e instanceof Error ? e.message : String(e)
  return msg.replace(/^Error invoking remote method '[^']+':\s*(Error:\s*)?/, '')
}
