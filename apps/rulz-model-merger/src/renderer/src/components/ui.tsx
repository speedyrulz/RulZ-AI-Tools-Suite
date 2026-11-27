import * as DialogPrimitive from '@radix-ui/react-dialog'
import * as SwitchPrimitive from '@radix-ui/react-switch'
import * as TooltipPrimitive from '@radix-ui/react-tooltip'
import { AlertTriangle, CheckCircle2, Circle, Loader2, X } from 'lucide-react'
import {
  forwardRef,
  useEffect,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type MutableRefObject,
  type ReactNode
} from 'react'
import { cn } from '../lib/format'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'

const VARIANTS: Record<Variant, string> = {
  primary:
    'bg-accent text-accent-fg hover:bg-accent-hover font-medium shadow-[0_6px_20px_-8px_rgba(237,0,13,0.8)] disabled:shadow-none',
  secondary: 'bg-panel-2 text-text border border-border hover:bg-hover hover:border-border-strong',
  ghost: 'hover:bg-hover text-muted hover:text-text',
  danger: 'bg-danger-bg text-danger-text border border-danger-border hover:bg-danger-hover'
}

export const Button = forwardRef<
  HTMLButtonElement,
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: 'sm' | 'md' | 'lg' }
>(function Button({ variant = 'secondary', size = 'md', className, type = 'button', ...props }, ref) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        'inline-flex items-center justify-center font-medium whitespace-nowrap transition-colors select-none disabled:cursor-not-allowed disabled:opacity-45',
        size === 'sm'
          ? 'h-8 gap-1.5 rounded-md px-3 text-[13px]'
          : size === 'lg'
            ? 'h-11 gap-2 rounded-lg px-6 text-[15px] font-semibold'
            : 'h-9 gap-2 rounded-lg px-4 text-[13px]',
        VARIANTS[variant],
        className
      )}
      {...props}
    />
  )
})

export const IconButton = forwardRef<HTMLButtonElement, ButtonHTMLAttributes<HTMLButtonElement> & { label: string }>(
  function IconButton({ label, className, children, type = 'button', ...props }, ref) {
    return (
      <Tooltip content={label}>
        <button
          ref={ref}
          type={type}
          aria-label={label}
          className={cn(
            'inline-flex size-9 items-center justify-center rounded-lg text-muted transition-colors hover:bg-hover hover:text-text disabled:cursor-not-allowed disabled:opacity-45',
            className
          )}
          {...props}
        >
          {children}
        </button>
      </Tooltip>
    )
  }
)

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input(
  { className, ...props },
  ref
) {
  return (
    <input
      ref={ref}
      className={cn(
        'h-9 w-full rounded-lg border border-border bg-panel px-3 text-[13px] placeholder:text-faint focus:border-accent focus:outline-none disabled:cursor-not-allowed disabled:opacity-45',
        className
      )}
      {...props}
    />
  )
})

/** Text field that saves after the user stops typing (default 400 ms) and on blur or unmount. */
export function DebouncedInput({
  value,
  onCommit,
  onDraft,
  flushRef,
  delay = 400,
  mono,
  ...props
}: Omit<InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange'> & {
  value: string
  onCommit: (v: string) => void | Promise<void>
  /** Called with the live text on every keystroke and when an outside value is adopted. */
  onDraft?: (v: string) => void
  /** Filled with a function that saves any pending text now and resolves when the save is done. */
  flushRef?: MutableRefObject<(() => Promise<void>) | null>
  delay?: number
  mono?: boolean
}) {
  const [draft, setDraft] = useState(value)
  const dirty = useRef(false)
  const timer = useRef<number | undefined>(undefined)
  const inflight = useRef<Promise<void>>(Promise.resolve())
  const latest = useRef({ draft, onCommit, onDraft })
  latest.current = { draft, onCommit, onDraft }

  // Adopt outside changes while the user is not typing.
  useEffect(() => {
    if (!dirty.current) {
      setDraft(value)
      latest.current.onDraft?.(value)
    }
  }, [value])

  const flush = (): void => {
    window.clearTimeout(timer.current)
    if (!dirty.current) return
    dirty.current = false
    inflight.current = Promise.resolve(latest.current.onCommit(latest.current.draft)).catch(() => undefined)
  }
  useEffect(() => {
    if (flushRef) {
      flushRef.current = async () => {
        flush()
        await inflight.current
      }
    }
    return () => {
      if (flushRef) flushRef.current = null
      flush()
    }
  }, [])

  return (
    <Input
      {...props}
      className={cn(mono && 'font-mono text-xs', props.className)}
      value={draft}
      onChange={(e) => {
        const next = e.target.value
        latest.current.draft = next
        setDraft(next)
        latest.current.onDraft?.(next)
        dirty.current = true
        window.clearTimeout(timer.current)
        timer.current = window.setTimeout(flush, delay)
      }}
      onBlur={flush}
    />
  )
}

/** Number field: typing is a draft, applied (clamped) on blur or Enter. */
export function NumberInput({
  value,
  onCommit,
  min,
  max,
  step,
  disabled,
  className
}: {
  value: number
  onCommit: (v: number) => void
  min: number
  max: number
  step?: number
  disabled?: boolean
  className?: string
}) {
  const [draft, setDraft] = useState<string | null>(null)
  // Read by the unmount cleanup, which only sees the first render's values.
  const latest = useRef({ draft, value, onCommit, min, max })
  latest.current = { draft, value, onCommit, min, max }
  const commit = (): void => {
    const { draft: d, value: v, onCommit: apply, min: lo, max: hi } = latest.current
    if (d === null) return
    latest.current.draft = null
    setDraft(null)
    const n = Number(d)
    if (d.trim() === '' || !Number.isFinite(n)) return
    const next = Math.min(hi, Math.max(lo, n))
    if (next !== v) apply(next)
  }
  useEffect(() => commit, [])
  return (
    <Input
      type="number"
      className={cn('tabular-nums', className)}
      value={draft ?? value}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') commit()
      }}
    />
  )
}

export function Switch({
  checked,
  onCheckedChange,
  disabled,
  id,
  label
}: {
  checked: boolean
  onCheckedChange: (v: boolean) => void
  disabled?: boolean
  id?: string
  label?: string
}) {
  return (
    <SwitchPrimitive.Root
      id={id}
      aria-label={label}
      checked={checked}
      onCheckedChange={onCheckedChange}
      disabled={disabled}
      className="relative h-5 w-9 shrink-0 rounded-full bg-border-strong transition-colors data-[state=checked]:bg-accent disabled:cursor-not-allowed disabled:opacity-45"
    >
      <SwitchPrimitive.Thumb className="block size-4 translate-x-0.5 rounded-full bg-white shadow transition-transform data-[state=checked]:translate-x-[18px]" />
    </SwitchPrimitive.Root>
  )
}

export function Checkbox({ checked, onChange, children }: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode }) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-[13px] select-none">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="size-[18px] shrink-0 rounded-[4px] accent-accent"
      />
      {children}
    </label>
  )
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label
}: {
  value: T
  options: { value: T; label: string; icon?: ReactNode }[]
  onChange: (v: T) => void
  label: string
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex gap-1 rounded-lg border border-border bg-panel p-1">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={cn(
            'inline-flex h-8 items-center gap-1.5 rounded-md px-4 text-[13px] font-medium transition-colors',
            value === o.value ? 'bg-panel-2 text-text shadow-[inset_0_0_0_1px_var(--border-strong)]' : 'text-muted hover:text-text'
          )}
        >
          {o.icon}
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Tooltip({ content, children, side = 'top' }: { content: ReactNode; children: ReactNode; side?: 'top' | 'right' | 'bottom' | 'left' }) {
  return (
    <TooltipPrimitive.Root delayDuration={350}>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content
          side={side}
          sideOffset={6}
          className="z-50 max-w-xs rounded-lg border border-border bg-panel px-2 py-1 text-xs text-text shadow-lg"
        >
          {content}
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  )
}

export const TooltipProvider = TooltipPrimitive.Provider

export function Dialog({
  open,
  onOpenChange,
  title,
  children,
  footer
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: ReactNode
  children?: ReactNode
  footer?: ReactNode
}) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-scrim backdrop-blur-[2px]" />
        <DialogPrimitive.Content className="fixed top-1/2 left-1/2 z-50 flex max-h-[88vh] w-[min(520px,calc(100vw-32px))] -translate-x-1/2 -translate-y-1/2 flex-col rounded-xl border border-border bg-panel focus:outline-none">
          <div className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
            <DialogPrimitive.Title className="font-[family-name:var(--font-display)] text-lg font-semibold">{title}</DialogPrimitive.Title>
            <DialogPrimitive.Close className="rounded-md p-1 text-muted hover:bg-hover hover:text-text" aria-label="Close">
              <X size={16} />
            </DialogPrimitive.Close>
          </div>
          <DialogPrimitive.Description className="sr-only">{typeof title === 'string' ? title : 'Dialog'}</DialogPrimitive.Description>
          <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">{children}</div>
          {footer && <div className="flex justify-end gap-2 border-t border-border px-5 py-3">{footer}</div>}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}

export function Badge({
  children,
  tone = 'neutral',
  className
}: {
  children: ReactNode
  tone?: 'neutral' | 'accent' | 'warn' | 'danger' | 'ok'
  className?: string
}) {
  const tones = {
    neutral: 'bg-subtle text-muted border-subtle-border',
    accent: 'bg-accent-soft text-accent-text border-accent/40',
    warn: 'bg-warn/10 text-warn-text border-warn/25',
    danger: 'bg-danger-bg text-danger-text border-danger-border',
    ok: 'bg-ok/10 text-ok border-ok/25'
  }
  return (
    <span className={cn('inline-flex h-6 items-center gap-1 rounded-full border px-2.5 text-xs font-medium whitespace-nowrap', tones[tone], className)}>
      {children}
    </span>
  )
}

export type StatusTone = 'ok' | 'warn' | 'danger' | 'idle'

/** A status always shows a word and an icon. */
export function StatusChip({ tone, label, busy }: { tone: StatusTone; label: string; busy?: boolean }) {
  const color = tone === 'ok' ? 'text-ok' : tone === 'warn' ? 'text-warn-text' : tone === 'danger' ? 'text-danger-text' : 'text-muted'
  const Icon = busy ? Loader2 : tone === 'ok' ? CheckCircle2 : tone === 'idle' ? Circle : AlertTriangle
  return (
    <span className={cn('inline-flex items-center gap-1.5 text-[13px] font-medium', color)}>
      <Icon size={14} className={cn(busy && 'animate-spin')} />
      {label}
    </span>
  )
}

export function Card({ title, description, actions, children, className }: { title?: ReactNode; description?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cn('rounded-xl border border-border bg-panel p-5', className)}>
      {(title || actions) && (
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="font-[family-name:var(--font-display)] text-[15px] font-semibold">{title}</h2>}
            {description && <p className="mt-0.5 text-[13px] text-muted">{description}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  )
}

export function Field({ label, hint, children }: { label: ReactNode; hint?: ReactNode; children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <div className="text-[13px] font-medium">{label}</div>
      {children}
      {hint && <div className="text-xs text-faint">{hint}</div>}
    </div>
  )
}

export function PageTitle({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="mb-6">
      <h1 className="font-[family-name:var(--font-display)] text-[28px] leading-tight font-bold tracking-tight">{title}</h1>
      {children && <p className="mt-1 text-sm text-muted">{children}</p>}
    </header>
  )
}
