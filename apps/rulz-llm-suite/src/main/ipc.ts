import { BrowserWindow, ipcMain } from 'electron'
import type { EventChannel, InvokeChannel, IpcEvents, IpcInvoke } from '@shared/ipc'

type Handler<K extends InvokeChannel> = (
  ...args: Parameters<IpcInvoke[K]>
) => ReturnType<IpcInvoke[K]> | Promise<ReturnType<IpcInvoke[K]>>

export function handle<K extends InvokeChannel>(channel: K, fn: Handler<K>): void {
  ipcMain.handle(channel, (_event, ...args) => fn(...(args as Parameters<IpcInvoke[K]>)))
}

export function broadcast<K extends EventChannel>(channel: K, payload: IpcEvents[K]): void {
  for (const win of BrowserWindow.getAllWindows()) {
    if (!win.isDestroyed()) win.webContents.send(channel, payload)
  }
}
