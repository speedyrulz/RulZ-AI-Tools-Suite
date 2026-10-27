import * as DialogPrimitive from '@radix-ui/react-dialog'
import * as Menu from '@radix-ui/react-dropdown-menu'
import * as SwitchPrimitive from '@radix-ui/react-switch'
import * as TooltipPrimitive from '@radix-ui/react-tooltip'
import { X } from 'lucide-react'
import { forwardRef, type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes, type TextareaHTMLAttributes } from 'react'
import { cn } from '../lib/format'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'

const VARIANTS: Record<Variant, string> = {
  primary: 'bg-accent text-accent-fg hover:bg-accent-hover font-medium',
  secondary: 'bg-panel-2 border border-border hover:bg-hover',
  ghost: 'hover:bg-hover text-muted hover:text-text',
  danger: 'bg-danger/90 text-white hover:bg-danger font-medium'
}

export const Button = forwardRef<
  HTMLButtonElement,
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: 'sm' | 'md' }
>(function Button({ variant = 'secondary', size = 'md', className, ...props }, ref) {
  return (
    <button
      ref={ref}
      className={cn(
        'inline-flex items-center justify-center gap-1.5 rounded-md transition-colors disabled:pointer-events-none disabled:opacity-45 select-none whitespace-nowrap',
        size === 'sm' ? 'h-7 px-2.5 text-xs' : 'h-8.5 px-3.5 text-[13px]',
        VARIANTS[variant],
        className
      )}
      {...props}
    />
  )
})

export const IconButton = forwardRef<
  HTMLButtonElement,
  ButtonHTMLAttributes<HTMLButtonElement> & { label: string; active?: boolean; size?: 'sm' | 'md' }
>(function IconButton({ label, active, size = 'md', className, children, ...props }, ref) {
  return (
    <Tooltip content={label}>
      <button
        ref={ref}
        aria-label={label}
        className={cn(
          'inline-flex items-center justify-center rounded-md text-muted transition-colors hover:bg-hover hover:text-text disabled:pointer-events-none disabled:opacity-40',
          size === 'sm' ? 'size-7' : 'size-8.5',
          active && 'bg-hover text-text',
          className
        )}
        {...props}
      >
        {children}
      </button>
    </Tooltip>
  )
})

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input(
  { className, ...props },
  ref
) {
  return (
    <input
      ref={ref}
      className={cn(
        'h-8.5 w-full rounded-md border border-border bg-panel-2 px-2.5 text-[13px] placeholder:text-faint focus:border-accent focus:outline-none',
        className
      )}
      {...props}
    />
  )
})

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(
  function Textarea({ className, ...props }, ref) {
    return (
      <textarea
        ref={ref}
        className={cn(
          'w-full resize-y rounded-md border border-border bg-panel-2 px-2.5 py-2 text-[13px] leading-relaxed placeholder:text-faint focus:border-accent focus:outline-none',
          className
        )}
        {...props}
      />
    )
  }
)

export function Select({ className, children, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={cn(
        'h-8.5 w-full rounded-md border border-border bg-panel-2 px-2 text-[13px] focus:border-accent focus:outline-none',
        className
      )}
      {...props}
    >
      {children}
    </select>
  )
}

export function Switch({
  checked,
  onCheckedChange,
  disabled,
  id
}: {
  checked: boolean
  onCheckedChange: (v: boolean) => void
  disabled?: boolean
  id?: string
}) {
  return (
    <SwitchPrimitive.Root
      id={id}
      checked={checked}
      onCheckedChange={onCheckedChange}
      disabled={disabled}
      className="relative h-5 w-9 shrink-0 rounded-full bg-track transition-colors data-[state=checked]:bg-accent disabled:opacity-40"
    >
      <SwitchPrimitive.Thumb className="block size-4 translate-x-0.5 rounded-full bg-white shadow transition-transform data-[state=checked]:translate-x-[18px]" />
    </SwitchPrimitive.Root>
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
          className="z-50 max-w-xs rounded-md border border-border bg-panel-2 px-2 py-1 text-xs text-text shadow-lg"
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
  description,
  children,
  className,
  footer,
  keepOnFocusOutside
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: ReactNode
  description?: ReactNode
  children?: ReactNode
  className?: string
  footer?: ReactNode
  /** Don't close when something else on the page takes focus (e.g. a field that autofocuses after startup). */
  keepOnFocusOutside?: boolean
}) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/30 dark:bg-black/50 backdrop-blur-[2px]" />
        <DialogPrimitive.Content
          className={cn(
            'fixed top-1/2 left-1/2 z-50 flex max-h-[88vh] w-[min(560px,calc(100vw-32px))] -translate-x-1/2 -translate-y-1/2 flex-col rounded-xl border border-border bg-panel shadow-2xl focus:outline-none',
            className
          )}
          aria-describedby={description ? undefined : undefined}
          onFocusOutside={keepOnFocusOutside ? (e) => e.preventDefault() : undefined}
        >
          <div className="flex items-start justify-between gap-4 border-b border-border px-5 py-3.5">
            <div>
              <DialogPrimitive.Title className="text-[15px] font-semibold">{title}</DialogPrimitive.Title>
              {description ? (
                <DialogPrimitive.Description className="mt-0.5 text-xs text-muted">{description}</DialogPrimitive.Description>
              ) : (
                <DialogPrimitive.Description className="sr-only">{typeof title === 'string' ? title : 'Dialog'}</DialogPrimitive.Description>
              )}
            </div>
            <DialogPrimitive.Close className="rounded-md p-1 text-muted hover:bg-hover hover:text-text" aria-label="Close">
              <X size={16} />
            </DialogPrimitive.Close>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
          {footer && <div className="flex justify-end gap-2 border-t border-border px-5 py-3">{footer}</div>}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}

export interface MenuItem {
  label: string
  icon?: ReactNode
  onSelect: () => void
  danger?: boolean
  disabled?: boolean
}

export function DropdownMenu({ trigger, items, align = 'end' }: { trigger: ReactNode; items: (MenuItem | 'sep')[]; align?: 'start' | 'end' }) {
  return (
    <Menu.Root>
      <Menu.Trigger asChild>{trigger}</Menu.Trigger>
      <Menu.Portal>
        <Menu.Content
          align={align}
          sideOffset={4}
          className="z-50 min-w-44 rounded-lg border border-border bg-panel p-1 shadow-xl"
        >
          {items.map((item, i) =>
            item === 'sep' ? (
              <Menu.Separator key={i} className="my-1 h-px bg-border" />
            ) : (
              <Menu.Item
                key={i}
                disabled={item.disabled}
                onSelect={item.onSelect}
                className={cn(
                  'flex cursor-default items-center gap-2 rounded-md px-2 py-1.5 text-[13px] outline-none data-[disabled]:opacity-40 data-[highlighted]:bg-hover',
                  item.danger && 'text-danger'
                )}
              >
                {item.icon}
                {item.label}
              </Menu.Item>
            )
          )}
        </Menu.Content>
      </Menu.Portal>
    </Menu.Root>
  )
}

export function Badge({ children, tone = 'neutral', className }: { children: ReactNode; tone?: 'neutral' | 'accent' | 'warn' | 'danger' | 'ok'; className?: string }) {
  const tones = {
    neutral: 'bg-panel-2 text-muted border-border',
    accent: 'bg-accent-soft text-accent border-accent/30',
    warn: 'bg-warn/10 text-warn border-warn/30',
    danger: 'bg-danger/10 text-danger border-danger/30',
    ok: 'bg-ok/10 text-ok border-ok/30'
  }
  return (
    <span className={cn('inline-flex items-center gap-1 rounded border px-1.5 py-px text-[11px] font-medium whitespace-nowrap', tones[tone], className)}>
      {children}
    </span>
  )
}

export function ProgressBar({ value, className }: { value: number; className?: string }) {
  return (
    <div className={cn('h-1 overflow-hidden rounded-full bg-track', className)}>
      <div className="h-full rounded-full bg-accent transition-[width] duration-200" style={{ width: `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%` }} />
    </div>
  )
}

export function Field({ label, hint, children, inline }: { label: ReactNode; hint?: ReactNode; children: ReactNode; inline?: boolean }) {
  if (inline) {
    return (
      <div className="flex items-center justify-between gap-3 py-1">
        <div className="min-w-0">
          <div className="text-[13px]">{label}</div>
          {hint && <div className="text-[11px] text-faint">{hint}</div>}
        </div>
        <div className="shrink-0">{children}</div>
      </div>
    )
  }
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[13px]">{label}</span>
        {hint && <span className="text-[11px] text-faint">{hint}</span>}
      </div>
      {children}
    </div>
  )
}

/** Number input that allows an empty value (mapped to null). */
export function NumberInput({
  value,
  onChange,
  placeholder,
  min,
  max,
  step,
  className
}: {
  value: number | null
  onChange: (v: number | null) => void
  placeholder?: string
  min?: number
  max?: number
  step?: number
  className?: string
}) {
  return (
    <Input
      type="number"
      className={cn('h-7.5 w-24 text-right tabular-nums', className)}
      value={value ?? ''}
      placeholder={placeholder}
      min={min}
      max={max}
      step={step}
      onChange={(e) => {
        const raw = e.target.value
        if (raw === '') return onChange(null)
        const n = Number(raw)
        if (Number.isFinite(n)) onChange(n)
      }}
    />
  )
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded border border-border bg-panel-2 px-1.5 py-px font-mono text-[11px] text-muted">{children}</kbd>
}
