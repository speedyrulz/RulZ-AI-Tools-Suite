import { Check } from 'lucide-react'
import { useState } from 'react'
import type { ThemeName } from '@shared/types'
import { cn } from '../lib/format'
import { applyThemeClass, LOGOS } from '../lib/theme'
import { act, useApps } from '../stores/apps'
import { Button, Dialog } from './ui'

/** Asked once, on first launch: Dark (the default) or Light. Picking previews it live. */
export function ThemeChooser() {
  const setTheme = useApps((s) => s.setTheme)
  const [choice, setChoice] = useState<ThemeName>(useApps.getState().theme)

  const pick = (t: ThemeName): void => {
    setChoice(t)
    applyThemeClass(t)
  }
  const save = (t: ThemeName): void => void act(() => setTheme(t))

  return (
    <Dialog
      open
      onOpenChange={(o) => !o && save('dark')}
      title="Choose your look"
      description="You can change it any time with the sun/moon button in the top bar, or in Settings."
      className="w-[560px]"
      footer={
        <Button variant="primary" onClick={() => save(choice)}>
          Continue
        </Button>
      }
    >
      <div className="grid grid-cols-2 gap-4">
        <ThemeCard theme="dark" label="Dark" note="Black and red (default)" selected={choice === 'dark'} onPick={pick} />
        <ThemeCard theme="light" label="Light" note="White and red" selected={choice === 'light'} onPick={pick} />
      </div>
    </Dialog>
  )
}

export function ThemeCard({
  theme,
  label,
  note,
  selected,
  onPick
}: {
  theme: ThemeName
  label: string
  note?: string
  selected: boolean
  onPick: (t: ThemeName) => void
}) {
  const dark = theme === 'dark'
  return (
    <button
      type="button"
      onClick={() => onPick(theme)}
      className={cn(
        'relative cursor-pointer overflow-hidden rounded-xl border-2 text-left transition-colors',
        selected ? 'border-accent' : 'border-border hover:border-border-strong'
      )}
    >
      {/* A tiny picture of the launcher in that theme. */}
      <div className={cn('h-28 p-3', dark ? 'bg-[#050505]' : 'bg-white')}>
        <img src={LOGOS[theme].logo} alt="" className="h-5" draggable={false} />
        <div className="mt-3 grid grid-cols-3 gap-1.5">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className={cn('h-12 rounded-md border', dark ? 'border-[#232328] bg-[#0c0c0e]' : 'border-[#d9d9de] bg-[#f5f5f7]')}
            >
              <div className="mt-8 ml-auto mr-1.5 h-2 w-6 rounded-sm bg-[#ed000d]" />
            </div>
          ))}
        </div>
      </div>
      <div className="flex items-center justify-between border-t border-border bg-panel px-3 py-2">
        <div>
          <div className="text-sm font-semibold">{label}</div>
          {note && <div className="text-xs text-faint">{note}</div>}
        </div>
        {selected && (
          <span className="flex size-5 items-center justify-center rounded-full bg-accent text-white">
            <Check className="size-3.5" strokeWidth={3} />
          </span>
        )}
      </div>
    </button>
  )
}
