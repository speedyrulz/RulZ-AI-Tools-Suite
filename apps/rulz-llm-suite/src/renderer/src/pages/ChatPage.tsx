import { ChatView } from '../components/chat/ChatView'
import { ConversationList } from '../components/chat/ConversationList'
import { InferencePanel } from '../components/chat/InferencePanel'
import { useApp } from '../stores/app'

export function ChatPage() {
  const panelOpen = useApp((s) => s.settings?.rightPanelOpen ?? true)
  return (
    <div className="flex h-full">
      <ConversationList />
      <div className="min-w-0 flex-1">
        <ChatView />
      </div>
      {panelOpen && <InferencePanel />}
    </div>
  )
}
