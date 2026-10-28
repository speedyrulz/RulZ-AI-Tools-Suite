import { BrowserWindow, ipcMain } from 'electron'
import type { EventChannel, InvokeChannel, IpcEvents, IpcInvoke } from '@shared/ipc'

type Handler<K extends InvokeChannel> = (
  ...args: Parameters<IpcInvoke[K]>
) => ReturnType<IpcInvoke[K]> | Promise<ReturnType<IpcInvoke[K]>>

type AnyHandler = (...args: unknown[]) => unknown

// The same handlers serve this computer's window and the phone app.
const handlers = new Map<InvokeChannel, AnyHandler>()
const listeners = new Set<(channel: EventChannel, payload: unknown) => void>()

export function handle<K extends InvokeChannel>(channel: K, fn: Handler<K>): void {
  handlers.set(channel, fn as AnyHandler)
  ipcMain.handle(channel, (_event, ...args) => fn(...(args as Parameters<IpcInvoke[K]>)))
}

/** Calls a handler for the phone app. */
export async function invokeHandler(channel: string, args: unknown[]): Promise<unknown> {
  const fn = handlers.get(channel as InvokeChannel)
  if (!fn) throw new Error(`Unknown request "${channel}".`)
  return fn(...args)
}

/** Hears every broadcast (the phone app's connection uses it). */
export function onBroadcast(listener: (channel: EventChannel, payload: unknown) => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function broadcast<K extends EventChannel>(channel: K, payload: IpcEvents[K]): void {
  for (const win of BrowserWindow.getAllWindows()) {
    if (!win.isDestroyed()) win.webContents.send(channel, payload)
  }
  for (const listener of listeners) listener(channel, payload)
}
