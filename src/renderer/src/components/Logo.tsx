import logoDark from '../assets/logo-dark.png'
import logoLight from '../assets/logo-light.png'
import markDark from '../assets/mark-dark.png'
import markLight from '../assets/mark-light.png'
import { cn } from '../lib/format'

// The brand art has black strokes; the dark theme uses a copy with those strokes in white.

export function LogoMark({ className }: { className?: string }) {
  return (
    <>
      <img src={markLight} alt="RulZ" draggable={false} className={cn('dark:hidden', className)} />
      <img src={markDark} alt="RulZ" draggable={false} className={cn('hidden dark:block', className)} />
    </>
  )
}

export function LogoWordmark({ className }: { className?: string }) {
  return (
    <>
      <img src={logoLight} alt="RulZ" draggable={false} className={cn('dark:hidden', className)} />
      <img src={logoDark} alt="RulZ" draggable={false} className={cn('hidden dark:block', className)} />
    </>
  )
}
