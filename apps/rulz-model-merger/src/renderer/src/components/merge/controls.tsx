import { Check, ChevronDown, Search } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { filterNames, splitName } from '../../lib/merge'
import { cn } from '../../lib/format'
import { Input, NumberInput } from '../ui'

function ModelName({ name, muted }: { name: string; muted?: boolean }) {
  const { folder, file } = splitName(name)
  return (
    <span className="flex min-w-0 items-baseline gap-2">
      <span className={cn('truncate text-[13px]', muted && 'text-muted')}>{file}</span>
      {folder && <span className="shrink truncate text-xs text-faint">{folder}</span>}
    </span>
  )
}

/** Select with a search box. The file name is prominent and its folder is muted. */
export function ModelSelect({
  value,
  options,
  onChange,
  placeholder,
  label,
  disabled
}: {
  value: string
  options: string[]
  onChange: (v: string) => void
  placeholder: string
  label: string
  disabled?: boolean
}) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const root = useRef<HTMLDivElement>(null)
  const shown = useMemo(() => filterNames(options, query), [options, query])

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent): void => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  const pick = (v: string): void => {
    onChange(v)
    setOpen(false)
    setQuery('')
  }

  return (
    <div ref={root} className="relative">
      <button
        type="button"
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={open}
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        className="flex h-9 w-full items-center justify-between gap-2 rounded-lg border border-border bg-panel px-3 text-left hover:border-border-strong focus:border-accent focus:outline-none disabled:cursor-not-allowed disabled:opacity-45"
      >
        {value ? <ModelName name={value} /> : <span className="text-[13px] text-faint">{placeholder}</span>}
        <ChevronDown size={14} className="shrink-0 text-muted" />
      </button>
      {open && (
        <div className="absolute z-30 mt-1 w-full min-w-[280px] rounded-lg border border-border bg-panel shadow-lg">
          <div className="relative border-b border-border p-2">
            <Search size={14} className="pointer-events-none absolute top-1/2 left-4 -translate-y-1/2 text-faint" />
            <Input
              autoFocus
              value={query}
              placeholder="Type to filter"
              className="pl-8"
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Escape') setOpen(false)
                if (e.key === 'Enter' && shown[0]) pick(shown[0])
              }}
            />
          </div>
          <ul role="listbox" aria-label={label} className="max-h-64 overflow-y-auto p-1">
            {shown.length === 0 && <li className="px-3 py-2 text-[13px] text-muted">{options.length === 0 ? 'The server lists no files here.' : 'No match.'}</li>}
            {shown.map((n) => (
              <li key={n} role="option" aria-selected={n === value}>
                <button
                  type="button"
                  onClick={() => pick(n)}
                  className={cn('flex h-9 w-full items-center justify-between gap-2 rounded-md px-3 text-left hover:bg-hover', n === value && 'bg-accent-soft text-accent-text')}
                >
                  <ModelName name={n} />
                  {n === value && <Check size={14} className="shrink-0" />}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

/** Slider with a number box next to it. */
export function SliderField({
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.01,
  label,
  compact
}: {
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  label: string
  compact?: boolean
}) {
  return (
    <div className="flex items-center gap-3">
      <input
        type="range"
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-5 min-w-0 flex-1 cursor-pointer accent-accent"
      />
      <NumberInput value={Number(value.toFixed(2))} onCommit={onChange} min={min} max={max} step={step} className={compact ? 'h-8 w-[72px]' : 'w-[88px]'} />
    </div>
  )
}
