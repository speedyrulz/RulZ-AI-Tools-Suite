import { Globe } from 'lucide-react'
import { spendView, chipText, spentTodayText } from '../../lib/cloud'
import { cn } from '../../lib/format'
import { useApp } from '../../stores/app'
import { useCloud } from '../../stores/cloud'
import { useModels } from '../../stores/models'
import { Tooltip } from '../ui'

/**
 * Top-bar chip, next to the vast.ai one: what OpenRouter models cost today. Only with a saved key and something spent,
 * or while an OpenRouter model is connected. Click opens the OpenRouter page.
 */
export function OpenRouterChip() {
  const state = useCloud((s) => s.state)
  const connected = useModels((s) => s.instances.some((i) => i.cloud))
  const setPage = useApp((s) => s.setPage)
  const text = chipText(state, connected ? [{ cloud: { provider: 'openrouter', free: false } }] : [])
  if (!state || !text) return null
  const v = spendView(state)
  return (
    <Tooltip content={`${spentTodayText(state)}. ${v.word}.`}>
      <button
        onClick={() => setPage('openrouter')}
        aria-label={`${text}. ${v.word}. Open the OpenRouter page.`}
        className={cn(
          'flex h-9 shrink-0 items-center gap-1.5 rounded-lg border px-2.5 text-xs tabular-nums transition-colors',
          v.tone === 'ok' && 'border-border bg-panel-2 text-muted hover:text-text',
          v.tone === 'warn' && 'border-warn/40 bg-warn/10 text-warn',
          v.tone === 'danger' && 'border-danger/40 bg-danger/10 text-danger-text'
        )}
      >
        <Globe size={14} />
        <span>{text}</span>
      </button>
    </Tooltip>
  )
}
