/**
 * What a number field's typed text means once the user finishes editing: `undefined` when there
 * is nothing to apply (unchanged, or not a number), `null` for an empty field, or the number.
 */
export function parseNumberDraft(draft: string, current: number | null): number | null | undefined {
  const raw = draft.trim()
  if (raw === '') return current === null ? undefined : null
  const n = Number(raw)
  if (!Number.isFinite(n) || n === current) return undefined
  return n
}
