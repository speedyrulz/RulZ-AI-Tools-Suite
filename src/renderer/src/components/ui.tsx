import * as RDialog from '@radix-ui/react-dialog'
import * as RMenu from '@radix-ui/react-dropdown-menu'
import { Check, X } from 'lucide-react'
import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { cn } from '../lib/format'

type Variant = 'primary' | 'secondary' | 'ghost' | 'outline' | 'danger'
type Size = 'sm' | 'md' | 'lg'

const VARIANTS: Record<Variant, string> = {
  primary:
    'bg-accent text-accent-fg hover:bg-accent-hover shadow-[0_6px_20px_-8px_rgb(237_0_13/0.8)] disabled:shadow-none',
  secondary: 'bg-panel-2 text-text border border-border hover:bg-hover hover:border-border-strong',
  outline: 'border border-accent/70 text-[#ff5961] hover:bg-accent-soft hover:border-accent',
  ghost: 'text-muted hover:text-text hover:bg-hover',
  danger: 'bg-[#3a0c0f] text-[#ff8a8e] border border-[#6b1418] hover:bg-[#4a1014]'
}
const SIZES: Record<Size, string> = {
  sm: 'h-8 px-3 text-[13px] gap-1.5 rounded-md',
  md: 'h-9 px-4 text-sm gap-2 rounded-lg',
  lg: 'h-11 px-6 text-[15px] gap-2 rounded-lg font-semibold'
}

export const Button = forwardRef<
  HTMLButtonElement,
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: Size }
>(({ variant = 'secondary', size = 'md', className, ...props }, ref) => (
  <button
    ref={ref}
    className={cn(
      'inline-flex shrink-0 cursor-pointer items-center justify-center font-medium whitespace-nowrap transition-colors select-none',
      'disabled:cursor-not-allowed disabled:opacity-45',
      VARIANTS[variant],
      SIZES[size],
      className
    )}
    {...props}
  />
))
Button.displayName = 'Button'

export const IconButton = forwardRef<HTMLButtonElement, ButtonHTMLAttributes<HTMLButtonElement> & { label: string }>(
  ({ label, className, ...props }, ref) => (
    <button
      ref={ref}
      title={label}
      aria-label={label}
      className={cn(
        'inline-flex size-9 shrink-0 cursor-pointer items-center justify-center rounded-lg text-muted transition-colors',
        'hover:bg-hover hover:text-text disabled:cursor-not-allowed disabled:opacity-40',
        className
      )}
      {...props}
    />
  )
)
IconButton.displayName = 'IconButton'

export function Badge({ children, tone = 'neutral', className }: { children: ReactNode; tone?: 'neutral' | 'accent' | 'ok' | 'warn'; className?: string }) {
  const tones = {
    neutral: 'bg-white/[0.06] text-muted border-white/[0.08]',
    accent: 'bg-accent-soft text-[#ff6b72] border-accent/40',
    ok: 'bg-[#4ade80]/10 text-ok border-[#4ade80]/25',
    warn: 'bg-[#fbbf24]/10 text-warn border-[#fbbf24]/25'
  }
  return (
    <span
      className={cn(
        'inline-flex h-6 items-center gap-1 rounded-full border px-2.5 text-[11.5px] font-medium whitespace-nowrap',
        tones[tone],
        className
      )}
    >
      {children}
    </span>
  )
}

export function ProgressBar({ value, className }: { value: number | null; className?: string }) {
  return (
    <div className={cn('h-1.5 overflow-hidden rounded-full bg-white/[0.08]', className)}>
      {value === null ? (
        <div className="progress-stripes h-full w-full rounded-full bg-accent/80" />
      ) : (
        <div
          className="h-full rounded-full bg-accent transition-[width] duration-200"
          style={{ width: `${Math.max(2, Math.min(100, value * 100))}%` }}
        />
      )}
    </div>
  )
}

export function Checkbox({
  checked,
  onChange,
  label,
  hint,
  disabled
}: {
  checked: boolean
  onChange: (v: boolean) => void
  label: ReactNode
  hint?: ReactNode
  disabled?: boolean
}) {
  return (
    <label className={cn('flex cursor-pointer items-start gap-3 py-1.5', disabled && 'cursor-not-allowed opacity-50')}>
      <button
        type="button"
        role="checkbox"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={cn(
          'mt-0.5 flex size-[18px] shrink-0 items-center justify-center rounded border transition-colors',
          checked ? 'border-accent bg-accent text-white' : 'border-border-strong bg-panel-2 hover:border-muted'
        )}
      >
        {checked && <Check className="size-3.5" strokeWidth={3} />}
      </button>
      <span className="min-w-0">
        <span className="block text-sm text-text">{label}</span>
        {hint && <span className="block text-xs text-faint">{hint}</span>}
      </span>
    </label>
  )
}

export function Switch({ checked, onChange, disabled }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        'relative h-5 w-9 shrink-0 cursor-pointer rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-40',
        checked ? 'bg-accent' : 'bg-border-strong'
      )}
    >
      <span
        className={cn(
          'absolute top-0.5 left-0.5 size-4 rounded-full bg-white shadow transition-transform',
          checked && 'translate-x-4'
        )}
      />
    </button>
  )
}

export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  children,
  footer,
  className
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: ReactNode
  description?: ReactNode
  children?: ReactNode
  footer?: ReactNode
  className?: string
}) {
  return (
    <RDialog.Root open={open} onOpenChange={onOpenChange}>
      <RDialog.Portal>
        <RDialog.Overlay className="fixed inset-0 z-40 bg-black/70 backdrop-blur-[2px]" />
        <RDialog.Content
          className={cn(
            'fixed top-1/2 left-1/2 z-50 flex max-h-[86vh] w-[520px] max-w-[92vw] -translate-x-1/2 -translate-y-1/2 flex-col',
            'rounded-xl border border-border bg-panel shadow-[0_30px_80px_-20px_rgb(0_0_0/0.9)]',
            className
          )}
        >
          <div className="flex items-start gap-3 border-b border-border px-5 pt-4 pb-3">
            <div className="min-w-0 flex-1">
              <RDialog.Title className="font-[family-name:var(--font-display)] text-lg font-semibold">{title}</RDialog.Title>
              {description ? (
                <RDialog.Description className="mt-0.5 text-[13px] text-muted">{description}</RDialog.Description>
              ) : (
                <RDialog.Description className="sr-only">{typeof title === 'string' ? title : 'Dialog'}</RDialog.Description>
              )}
            </div>
            <RDialog.Close asChild>
              <IconButton label="Close" className="-mt-1 -mr-2">
                <X className="size-4" />
              </IconButton>
            </RDialog.Close>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">{children}</div>
          {footer && <div className="flex justify-end gap-2 border-t border-border px-5 py-3">{footer}</div>}
        </RDialog.Content>
      </RDialog.Portal>
    </RDialog.Root>
  )
}

export const Menu = RMenu.Root
export const MenuTrigger = RMenu.Trigger

export function MenuContent({ children, align = 'end' }: { children: ReactNode; align?: 'start' | 'end' }) {
  return (
    <RMenu.Portal>
      <RMenu.Content
        align={align}
        sideOffset={6}
        className="z-50 min-w-[220px] rounded-lg border border-border bg-panel-2 p-1 shadow-[0_16px_40px_-10px_rgb(0_0_0/0.9)]"
      >
        {children}
      </RMenu.Content>
    </RMenu.Portal>
  )
}

export function MenuItem({
  children,
  onSelect,
  icon,
  danger,
  disabled
}: {
  children: ReactNode
  onSelect: () => void
  icon?: ReactNode
  danger?: boolean
  disabled?: boolean
}) {
  return (
    <RMenu.Item
      disabled={disabled}
      onSelect={onSelect}
      className={cn(
        'flex cursor-pointer items-center gap-2.5 rounded-md px-2.5 py-2 text-[13px] outline-none select-none',
        'data-[disabled]:cursor-not-allowed data-[disabled]:opacity-40 data-[highlighted]:bg-hover',
        danger ? 'text-[#ff7a7f]' : 'text-text'
      )}
    >
      <span className="flex size-4 items-center justify-center text-muted [&>svg]:size-4">{icon}</span>
      {children}
    </RMenu.Item>
  )
}

export function MenuSeparator() {
  return <RMenu.Separator className="my-1 h-px bg-border" />
}
