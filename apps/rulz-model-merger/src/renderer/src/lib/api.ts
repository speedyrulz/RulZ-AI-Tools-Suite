import { toast } from 'sonner'
import type { InvokeChannel, IpcInvoke } from '@shared/ipc'

export function errText(e: unknown): string {
  const raw = e instanceof Error ? e.message : String(e)
  // Electron prefixes rejected invokes with "Error invoking remote method 'x': Error: ".
  return raw.replace(/^Error invoking remote method '[^']*': (Error: )?/, '')
}

type Result<K extends InvokeChannel> = Awaited<ReturnType<IpcInvoke[K]>>

/** Calls the main process; on failure shows the error in a toast and resolves to undefined. */
export async function call<K extends InvokeChannel>(channel: K, ...args: Parameters<IpcInvoke[K]>): Promise<Result<K> | undefined> {
  try {
    return await window.rulz.invoke(channel, ...args)
  } catch (e) {
    toast.error(errText(e))
    return undefined
  }
}

/** Calls the main process and lets the caller handle a rejection. */
export function invoke<K extends InvokeChannel>(channel: K, ...args: Parameters<IpcInvoke[K]>): Promise<Result<K>> {
  return window.rulz.invoke(channel, ...args)
}
