import { Check, Moon, Sun } from 'lucide-react'
import { useState } from 'react'
import { cn } from '../../lib/format'
import { applyTheme, useApp } from '../../stores/app'
import { Button, Dialog } from '../ui'

type Look = 'dark' | 'light'

/** First-run "Choose your look" prompt; shown until a theme has been saved as the user's choice. */
export function ThemeChooser() {
  const chosen = useApp((s) => s.settings?.themeChosen ?? true)
  const setTheme = useApp((s) => s.setTheme)
  const [look, setLook] = useState<Look>('dark')

  const pick = (next: Look) => {
    setLook(next)
    applyTheme(next) // live preview; saved on Continue
  }

  return (
    <Dialog
      open={!chosen}
      // Closing without choosing keeps the default look (Dark) and doesn't ask again.
      onOpenChange={(open) => !open && void setTheme('dark')}
      title="Choose your look"
      description="Pick a theme for RulZ LLM Suite."
      className="w-[min(520px,calc(100vw-32px))]"
      keepOnFocusOutside
      footer={
        <Button variant="primary" onClick={() => void setTheme(look)} autoFocus>
          Continue
        </Button>
      }
    >
      <div className="space-y-4 px-5 py-4">
        <div className="grid grid-cols-2 gap-3" role="radiogroup" aria-label="Theme">
          <LookCard look="dark" selected={look === 'dark'} onSelect={pick} />
          <LookCard look="light" selected={look === 'light'} onSelect={pick} />
        </div>
        <p className="text-xs text-muted">
          You can change this any time with the sun/moon button in the top bar, or in Settings ▸ Appearance.
        </p>
      </div>
    </Dialog>
  )
}

function LookCard({ look, selected, onSelect }: { look: Look; selected: boolean; onSelect: (look: Look) => void }) {
  const dark = look === 'dark'
  return (
    <button
      role="radio"
      aria-checked={selected}
      onClick={() => onSelect(look)}
      className={cn(
        'rounded-xl border-2 p-2.5 text-left transition-colors',
        selected ? 'border-accent bg-accent-soft/40' : 'border-border hover:border-faint'
      )}
    >
      <Preview dark={dark} />
      <div className="mt-2.5 flex items-center gap-2 px-0.5 text-[13px] font-medium">
        {dark ? <Moon size={14} /> : <Sun size={14} />}
        {dark ? 'Dark' : 'Light'}
        {selected && <Check size={15} className="ml-auto text-accent" />}
      </div>
    </button>
  )
}

/** A tiny drawing of the app in each look (fixed colours, independent of the current theme). */
function Preview({ dark }: { dark: boolean }) {
  const c = dark
    ? { bg: '#000000', panel: '#0b0b0b', line: '#3a3a3a', text: '#ededed', border: '#262626' }
    : { bg: '#ffffff', panel: '#f5f5f7', line: '#d0d0d6', text: '#111114', border: '#d9d9de' }
  return (
    <div className="flex h-24 overflow-hidden rounded-lg" style={{ background: c.bg, border: `1px solid ${c.border}` }}>
      <div className="flex w-6 flex-col items-center gap-1.5 pt-2" style={{ background: c.panel, borderRight: `1px solid ${c.border}` }}>
        <span className="size-2.5 rounded-sm" style={{ background: '#ed000d' }} />
        <span className="size-2 rounded-sm" style={{ background: c.line }} />
        <span className="size-2 rounded-sm" style={{ background: c.line }} />
      </div>
      <div className="flex flex-1 flex-col gap-1.5 p-2">
        <div className="h-2 w-2/3 rounded-full" style={{ background: c.text, opacity: 0.85 }} />
        <div className="h-1.5 w-full rounded-full" style={{ background: c.line }} />
        <div className="h-1.5 w-5/6 rounded-full" style={{ background: c.line }} />
        <div className="mt-auto flex items-center gap-1.5">
          <div className="h-4 flex-1 rounded" style={{ background: c.panel, border: `1px solid ${c.border}` }} />
          <div className="h-4 w-7 rounded" style={{ background: '#ed000d' }} />
        </div>
      </div>
    </div>
  )
}
