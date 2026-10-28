import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs))
}

export function formatBytes(bytes: number | null | undefined, digits = 1): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = bytes
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(i >= 3 ? digits : 0)} ${units[i]}`
}

export function formatNumber(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—'
  return n.toLocaleString('en-US')
}

export function formatCompact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n % 1_000_000 ? 1 : 0)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(n % 1000 ? 1 : 0)}K`
  return String(n)
}

/** Context sizes are powers of two, so 8192 reads as "8K" and 262144 as "256K". */
export function formatTokens(n: number): string {
  if (n >= 1024 && n % 1024 === 0) return `${n / 1024}K`
  return formatCompact(n)
}

export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return '—'
  if (ms < 1000) return `${Math.round(ms)} ms`
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(s < 10 ? 2 : 1)} s`
  const m = Math.floor(s / 60)
  return `${m}m ${Math.round(s % 60)}s`
}

export function relativeTime(ts: number): string {
  const diff = Date.now() - ts
  const min = 60_000
  if (diff < min) return 'just now'
  if (diff < 60 * min) return `${Math.floor(diff / min)}m ago`
  if (diff < 24 * 60 * min) return `${Math.floor(diff / (60 * min))}h ago`
  if (diff < 7 * 24 * 60 * min) return `${Math.floor(diff / (24 * 60 * min))}d ago`
  return new Date(ts).toLocaleDateString()
}

export function uid(): string {
  return crypto.randomUUID()
}
