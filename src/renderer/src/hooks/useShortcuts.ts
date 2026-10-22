import { useEffect } from 'react'
import { useApp } from '../stores/app'
import { useChat } from '../stores/chat'
import { useModels } from '../stores/models'

export const SHORTCUTS: [string, string][] = [
  ['Ctrl+L', 'Open the model picker'],
  ['Ctrl+N', 'New chat'],
  ['Ctrl+1 / 2 / 3', 'Chat / My Models / Settings'],
  ['Ctrl+E', 'Eject all models'],
  ['Ctrl+B', 'Toggle the chat settings panel'],
  ['Enter / Shift+Enter', 'Send / new line'],
  ['Esc', 'Stop generating'],
  ['F12', 'Developer tools']
]

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
      } else if (key === '1' || key === '2' || key === '3') {
        e.preventDefault()
        app.setPage(key === '1' ? 'chat' : key === '2' ? 'models' : 'settings')
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
}
