// Notices when the agent goes round in circles: the same call three times in a row, or five
// failures in a row.

export const REPEAT_LIMIT = 3
export const ERROR_LIMIT = 5

export type GuardTrip = 'repeating' | 'errors'

/** Arguments in a stable form, ignoring the plain-language summary (it can vary while the call is the same). */
export function callKey(name: string, args: string): string {
  let parsed: unknown
  try {
    parsed = JSON.parse(args || '{}')
  } catch {
    return `${name}:${args}`
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return `${name}:${args}`
  const entries = Object.entries(parsed as Record<string, unknown>)
    .filter(([k]) => k !== 'summary')
    .sort(([a], [b]) => a.localeCompare(b))
  return `${name}:${JSON.stringify(entries)}`
}

/**
 * Tools that are called again on purpose to see what changed: the same call counts as a repeat only
 * when it also brought the same result (polling a build whose output keeps coming is progress).
 */
const POLLING_TOOLS = new Set(['check_background_command'])

export class LoopGuard {
  private lastKey = ''
  private repeats = 0
  private errors = 0
  /** Last error text, for the plain-language message. */
  lastError = ''

  record(name: string, args: string, isError: boolean, resultText = ''): GuardTrip | null {
    const key = callKey(name, args) + (POLLING_TOOLS.has(name) ? `\n${resultText}` : '')
    this.repeats = key === this.lastKey ? this.repeats + 1 : 1
    this.lastKey = key
    if (isError) {
      this.errors++
      this.lastError = resultText
    } else this.errors = 0
    if (this.repeats >= REPEAT_LIMIT) return 'repeating'
    if (this.errors >= ERROR_LIMIT) return 'errors'
    return null
  }

  reset(): void {
    this.lastKey = ''
    this.repeats = 0
    this.errors = 0
    this.lastError = ''
  }
}
