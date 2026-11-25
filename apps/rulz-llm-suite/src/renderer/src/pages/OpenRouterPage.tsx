import { Globe } from 'lucide-react'
import { AddModels } from '../components/cloud/AddModels'
import { CloudAccountCard } from '../components/cloud/AccountCard'
import { PrivacyCard } from '../components/cloud/PrivacyCard'
import { SpendingCard } from '../components/cloud/SpendingCard'
import { YourModels } from '../components/cloud/YourModels'
import { useCloud } from '../stores/cloud'

export function OpenRouterPage() {
  const state = useCloud((s) => s.state)
  const loaded = useCloud((s) => s.loaded)

  if (!state) {
    return (
      <div className="flex h-full items-center justify-center px-6 text-center text-sm text-muted">
        {loaded ? 'The OpenRouter state could not be read. Press Ctrl+0 again later, or restart the app.' : 'Reading the OpenRouter state…'}
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-4xl space-y-5 px-6 py-8">
        <div>
          <h1 className="flex items-center gap-2 font-[family-name:var(--font-display)] text-lg font-semibold">
            <Globe size={18} /> OpenRouter
          </h1>
          <p className="mt-0.5 text-xs text-muted">
            Use models that run at OpenRouter in Chat, Writing, Adventure, Reverse Engineering and batch practice. They cost money unless they are marked Free, and
            the daily limit below stops spending. Models on this PC and on vast.ai work as before.
          </p>
        </div>

        <CloudAccountCard state={state} />
        <SpendingCard state={state} />
        <PrivacyCard state={state} />
        <YourModels models={state.models} keySaved={state.keySaved} />
        <AddModels added={state.models} />
      </div>
    </div>
  )
}
