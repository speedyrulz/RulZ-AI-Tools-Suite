import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs))
}

/** MiB to a short "x.y GiB" / "n MiB" text. */
export function formatMiB(mib: number): string {
  return mib >= 1024 ? `${(mib / 1024).toFixed(1)} GiB` : `${Math.round(mib)} MiB`
}

export function formatGiB(gib: number | null): string {
  return gib === null ? 'unknown' : `${gib.toFixed(1)} GiB`
}
