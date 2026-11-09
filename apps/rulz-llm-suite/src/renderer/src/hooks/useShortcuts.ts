import { useEffect } from 'react'
import { useApp, type Page } from '../stores/app'
import { useChat } from '../stores/chat'
import { useModels } from '../stores/models'

export const SHORTCUTS: [string, string][] = [
  ['Ctrl+L', 'Open the model picker'],
  ['Ctrl+N', 'New chat'],
  ['Ctrl+1 … 6', 'Chat / My Models / MCP Tools / Developer / Settings / Reverse Engineering'],
  ['Ctrl+E', 'Eject all models'],
  ['Ctrl+B', 'Toggle the chat settings panel'],
  ['Enter / Shift+Enter', 'Send / new line'],
  ['Esc', 'Stop generating'],
  ['F12', 'Developer tools']
]

const PAGE_KEYS: Record<string, Page> = { '1': 'chat', '2': 'models', '3': 'mcp', '4': 'developer', '5': 'settings', '6': 're' }

/**
 * Esc stops the reply only when nothing else used the key: closing a dialog or menu, or
 * cancelling an edit, must not also end the reply (or, on the approval dialog, the tool turn).
 */
function escapeStopsReply(e: KeyboardEvent): boolean {
  // Radix dialogs and menus call preventDefault when Esc closes them. Tooltips do too, but
  // closing a hover tooltip shouldn't keep the reply running.
  if (e.defaultPrevented && !document.querySelector('[role="tooltip"]')) return false
  if (document.querySelector('[role="dialog"], [role="alertdialog"], [role="menu"]')) return false
  // Fields use Esc to cancel their edit; only the message box stops the reply.
  const el = e.target instanceof Element ? e.target : null
  if (el?.closest('input, textarea, select, [contenteditable="true"]') && !el.closest('[data-composer]')) return false
  return true
}

export function useShortcuts(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      const app = useApp.getState()
      if (e.key === 'Escape') {
        const chat = useChat.getState()
        // Only the reply of the chat on screen, where its Stop button is; one in the background stays.
        const shown = chat.generation && chat.generation.conversationId === chat.current?.id
        if (shown && !app.pickerOpen && escapeStopsReply(e)) chat.stop()
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
