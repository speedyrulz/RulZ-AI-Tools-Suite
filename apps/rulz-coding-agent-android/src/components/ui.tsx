import * as DialogPrimitive from '@radix-ui/react-dialog'
import * as SwitchPrimitive from '@radix-ui/react-switch'
import { ArrowLeft, X } from 'lucide-react'
import { forwardRef, type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes, type TextareaHTMLAttributes } from 'react'
import { cn } from '../lib/format'

// The computer app's controls, sized for fingers: taller buttons and fields, dialogs as sheets from the bottom.

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'

const VARIANTS: Record<Variant, string> = {
  primary: 'bg-accent text-accent-fg active:bg-accent-hover font-medium',
  secondary: 'bg-panel-2 border border-border active:bg-hover',
  ghost: 'active:bg-hover text-muted',
  danger: 'bg-danger/90 text-white active:bg-danger font-medium'
}

export const Button = forwardRef<HTMLButtonElement, ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: 'sm' | 'md' | 'lg' }>(
  function Button({ variant = 'secondary', size = 'md', className, ...props }, ref) {
    return (
      <button
        ref={ref}
        className={cn(
          'inline-flex items-center justify-center gap-1.5 rounded-lg transition-colors select-none whitespace-nowrap disabled:pointer-events-none disabled:opacity-45',
          size === 'sm' ? 'h-9 px-3 text-[13px]' : size === 'lg' ? 'h-12 px-5 text-[15px]' : 'h-11 px-4 text-sm',
          VARIANTS[variant],
          className
        )}
        {...props}
      />
    )
  }
)

export const IconButton = forwardRef<HTMLButtonElement, ButtonHTMLAttributes<HTMLButtonElement> & { label: string; active?: boolean }>(
  function IconButton({ label, active, className, children, ...props }, ref) {
    return (
      <button
        ref={ref}
        aria-label={label}
        title={label}
        className={cn(
          'inline-flex size-11 shrink-0 items-center justify-center rounded-full text-muted transition-colors active:bg-hover disabled:pointer-events-none disabled:opacity-40',
          active && 'bg-hover text-text',
          className
        )}
        {...props}
      >
        {children}
      </button>
    )
  }
)

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input({ className, ...props }, ref) {
  return (
    <input
      ref={ref}
      className={cn(
        'h-11 w-full rounded-lg border border-border bg-panel-2 px-3 text-[15px] placeholder:text-faint focus:border-accent focus:outline-none',
        className
      )}
      {...props}
    />
  )
})

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(function Textarea({ className, ...props }, ref) {
  return (
    <textarea
      ref={ref}
      className={cn(
        'w-full resize-none rounded-lg border border-border bg-panel-2 px-3 py-2.5 text-[15px] leading-relaxed placeholder:text-faint focus:border-accent focus:outline-none',
        className
      )}
      {...props}
    />
  )
})

export function Select({ className, children, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select className={cn('h-11 w-full rounded-lg border border-border bg-panel-2 px-2.5 text-[15px] focus:border-accent focus:outline-none', className)} {...props}>
      {children}
    </select>
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
      className="relative h-7 w-12 shrink-0 rounded-full bg-track transition-colors data-[state=checked]:bg-accent disabled:opacity-40"
    >
      <SwitchPrimitive.Thumb className="block size-6 translate-x-0.5 rounded-full bg-white shadow transition-transform data-[state=checked]:translate-x-[22px]" />
    </SwitchPrimitive.Root>
  )
}

/** A sheet that slides up from the bottom of the screen (the phone's version of the computer's dialogs). */
export function Sheet({
  open,
  onOpenChange,
  title,
  description,
  children,
  footer,
  full,
  dismissable = true
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: ReactNode
  description?: ReactNode
  children?: ReactNode
  footer?: ReactNode
  /** Takes the whole screen (diffs, long lists). */
  full?: boolean
  /** False keeps it open when tapping outside (questions that need an answer). */
  dismissable?: boolean
}) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/40 dark:bg-black/60" />
        <DialogPrimitive.Content
          className={cn(
            'sheet pb-safe fixed inset-x-0 bottom-0 z-50 flex flex-col border-t border-border bg-panel shadow-2xl focus:outline-none',
            full ? 'pt-safe top-0' : 'max-h-[90dvh] rounded-t-2xl'
          )}
          onPointerDownOutside={dismissable ? undefined : (e) => e.preventDefault()}
          onInteractOutside={dismissable ? undefined : (e) => e.preventDefault()}
        >
          {!full && <div className="mx-auto mt-2 h-1 w-10 shrink-0 rounded-full bg-track" />}
          <div className="flex items-start gap-2 px-4 pt-2 pb-3">
            {full && (
              <DialogPrimitive.Close className="-ml-2 inline-flex size-11 shrink-0 items-center justify-center rounded-full text-muted active:bg-hover" aria-label="Back">
                <ArrowLeft size={20} />
              </DialogPrimitive.Close>
            )}
            <div className={cn('min-w-0 flex-1', full && 'pt-2.5')}>
              <DialogPrimitive.Title className="text-[17px] font-semibold break-words">{title}</DialogPrimitive.Title>
              {description ? (
                <DialogPrimitive.Description className="mt-0.5 text-[13px] text-muted">{description}</DialogPrimitive.Description>
              ) : (
                <DialogPrimitive.Description className="sr-only">{typeof title === 'string' ? title : 'Dialog'}</DialogPrimitive.Description>
              )}
            </div>
            {!full && dismissable && (
              <DialogPrimitive.Close className="-mr-2 inline-flex size-11 shrink-0 items-center justify-center rounded-full text-muted active:bg-hover" aria-label="Close">
                <X size={18} />
              </DialogPrimitive.Close>
            )}
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">{children}</div>
          {footer && <div className="flex flex-wrap justify-end gap-2 border-t border-border px-4 py-3">{footer}</div>}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
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
    <span className={cn('inline-flex items-center gap-1 rounded border px-1.5 py-px text-[11px] font-medium whitespace-nowrap', tones[tone], className)}>{children}</span>
  )
}

export function Field({ label, hint, children, inline }: { label: ReactNode; hint?: ReactNode; children: ReactNode; inline?: boolean }) {
  if (inline) {
    return (
      <div className="flex items-center justify-between gap-3 py-1">
        <div className="min-w-0">
          <div className="text-[15px]">{label}</div>
          {hint && <div className="text-[12px] text-faint">{hint}</div>}
        </div>
        <div className="shrink-0">{children}</div>
      </div>
    )
  }
  return (
    <div className="space-y-1.5">
      <div>
        <div className="text-[15px] font-medium">{label}</div>
        {hint && <div className="text-[12px] text-faint">{hint}</div>}
      </div>
      {children}
    </div>
  )
}

/** The bar at the top of each page, under the phone's status bar. */
export function TopBar({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <header className={cn('pt-safe px-safe shrink-0 border-b border-border bg-panel', className)}>
      <div className="flex h-14 items-center gap-1 px-1.5">{children}</div>
    </header>
  )
}

export function Spinner({ className }: { className?: string }) {
  return <span className={cn('inline-block size-4 animate-spin rounded-full border-2 border-current border-t-transparent', className)} />
}
