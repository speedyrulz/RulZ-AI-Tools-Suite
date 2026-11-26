// Small pure helpers for the renderer fixes of the 2026-11-26 audit.

/** `n` limited to `min` and `max` when they are given. */
export function clampTo(n: number, min?: number, max?: number): number {
  let v = n
  if (min !== undefined && v < min) v = min
  if (max !== undefined && v > max) v = max
  return v
}

/** The last `cap` lines of `lines` once `line` is added. */
export function appendCapped<T>(lines: T[], line: T, cap: number): T[] {
  const next = lines.length >= cap ? lines.slice(lines.length - cap + 1) : lines.slice()
  next.push(line)
  return next
}

/** One stop string per non-empty line. */
export const textToStops = (t: string): string[] => t.split('\n').filter((x) => x.length > 0)
