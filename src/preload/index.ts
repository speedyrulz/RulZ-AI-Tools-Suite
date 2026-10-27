import { contextBridge, ipcRenderer } from 'electron'
import type { RulzApi } from '@shared/ipc'

const api: RulzApi = {
  invoke: (channel, ...args) => ipcRenderer.invoke(channel, ...args),
  on: (channel, listener) => {
    const wrapped = (_event: Electron.IpcRendererEvent, payload: unknown): void =>
      listener(payload as never)
    ipcRenderer.on(channel, wrapped)
    return () => {
      ipcRenderer.removeListener(channel, wrapped)
    }
  }
}

contextBridge.exposeInMainWorld('rulz', api)
