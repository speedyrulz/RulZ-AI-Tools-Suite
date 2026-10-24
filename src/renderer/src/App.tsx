import { useEffect, useState } from 'react'
import { Toaster } from 'sonner'
import { TooltipProvider } from './components/ui'
import { NavRail } from './components/shell/NavRail'
import { TopBar } from './components/shell/TopBar'
import { ModelPicker } from './components/models/ModelPicker'
import { LogDialog } from './components/models/LogDialog'
import { ChatPage } from './pages/ChatPage'
import { ModelsPage } from './pages/ModelsPage'
import { McpPage } from './pages/McpPage'
import { DeveloperPage } from './pages/DeveloperPage'
import { useMcp } from './stores/mcp'
import { useServer } from './stores/server'
import { SettingsPage } from './pages/SettingsPage'
import { ComingSoonPage } from './pages/ComingSoonPage'
import { useShortcuts } from './hooks/useShortcuts'
import { useApp } from './stores/app'
import { useChat } from './stores/chat'
import { useModels } from './stores/models'

export function App() {
  const page = useApp((s) => s.page)
  const theme = useApp((s) => s.settings?.theme)
  const [ready, setReady] = useState(false)
  useShortcuts()

  useEffect(() => {
    let cancelled = false
    void (async () => {
      await useApp.getState().init()
      await useModels.getState().init()
      await useMcp.getState().init()
      await useServer.getState().init()
      await useChat.getState().init()
      if (!cancelled) setReady(true)
    })()
    return () => {
      cancelled = true
    }
  }, [])

  if (!ready) {
    return <div className="flex h-full items-center justify-center text-sm text-faint">Starting RulZ LLM Suite…</div>
  }

  return (
    <TooltipProvider>
      <div className="flex h-full">
        <NavRail />
        <div className="flex min-w-0 flex-1 flex-col">
          <TopBar />
          <main className="min-h-0 flex-1">
            {page === 'chat' && <ChatPage />}
            {page === 'models' && <ModelsPage />}
            {page === 'mcp' && <McpPage />}
            {page === 'settings' && <SettingsPage />}
            {page === 'discover' && <ComingSoonPage title="Discover" what="Search Hugging Face for GGUF models, pick a quantization that fits your GPUs, and download it straight into your models folder." />}
            {page === 'developer' && <DeveloperPage />}
          </main>
        </div>
      </div>
      <ModelPicker />
      <LogDialog />
      <Toaster
        theme={theme === 'light' ? 'light' : theme === 'system' ? 'system' : 'dark'}
        position="bottom-right"
        richColors
        closeButton
        toastOptions={{ style: { whiteSpace: 'pre-wrap' } }}
      />
    </TooltipProvider>
  )
}
