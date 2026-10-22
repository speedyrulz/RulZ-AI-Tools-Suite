import { useEffect } from 'react'
import { useApp, type Page } from '../stores/app'
import { useChat } from '../stores/chat'
import { useModels } from '../stores/models'

export const SHORTCUTS: [string, string][] = [
  ['Ctrl+L', 'Open the model picker'],
  ['Ctrl+N', 'New chat'],
  ['Ctrl+1 / 2 / 3 / 4', 'Chat / My Models / MCP Tools / Settings'],
  ['Ctrl+E', 'Eject all models'],
  ['Ctrl+B', 'Toggle the chat settings panel'],
  ['Enter / Shift+Enter', 'Send / new line'],
  ['Esc', 'Stop generating'],
  ['F12', 'Developer tools']
]

const PAGE_KEYS: Record<string, Page> = { '1': 'chat', '2': 'models', '3': 'mcp', '4': 'settings' }

export function useShortcuts(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      const app = useApp.getState()
      if (e.key === 'Escape' && useChat.getState().generation && !app.pickerOpen) {
        useChat.getState().stop()
        return
      }
      if (!e.ctrlKey || e.altKey) return
      const key = e.key.toLowerCase()
      if (key === 'l') {
        e.preventDefault()
        if (app.pickerOpen) app.closePicker()
        else app.openPicker()
      } else if (key === 'n' && !e.shiftKey) {
        e.preventDefault()
        app.setPage('chat')
        void useChat.getState().newChat()
      } else if (key === 'e') {
        e.preventDefault()
        void useModels.getState().unloadAll()
      } else if (key === 'b') {
        e.preventDefault()
        const open = app.settings?.rightPanelOpen ?? true
        void app.updateSettings({ rightPanelOpen: !open })
      } else if (key in PAGE_KEYS) {
        e.preventDefault()
        app.setPage(PAGE_KEYS[key])
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
}
