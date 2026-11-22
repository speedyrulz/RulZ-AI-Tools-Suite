// A small word diff for showing what an alternative changes: text split into words and spaces, matched with a
// longest-common-subsequence table. Big inputs fall back to "everything replaced" so the phone never stalls.

export interface DiffPart {
  kind: 'same' | 'add' | 'del'
  text: string
}

const MAX_CELLS = 4_000_000

function tokens(text: string): string[] {
  return text.match(/\s+|[^\s]+/g) ?? []
}

export function wordDiff(before: string, after: string): DiffPart[] {
  if (before === after) return before ? [{ kind: 'same', text: before }] : []
  const a = tokens(before)
  const b = tokens(after)
  // Trim what both ends share first; it is most of a typical edit and keeps the table small.
  let start = 0
  while (start < a.length && start < b.length && a[start] === b[start]) start++
  let endA = a.length
  let endB = b.length
  while (endA > start && endB > start && a[endA - 1] === b[endB - 1]) {
    endA--
    endB--
  }
  const midA = a.slice(start, endA)
  const midB = b.slice(start, endB)
  const parts: DiffPart[] = []
  const push = (kind: DiffPart['kind'], text: string): void => {
    if (!text) return
    const last = parts[parts.length - 1]
    if (last && last.kind === kind) last.text += text
    else parts.push({ kind, text })
  }
  push('same', a.slice(0, start).join(''))
  if ((midA.length + 1) * (midB.length + 1) > MAX_CELLS) {
    push('del', midA.join(''))
    push('add', midB.join(''))
  } else {
    const n = midA.length
    const m = midB.length
    const table = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1))
    for (let i = n - 1; i >= 0; i--) {
      for (let j = m - 1; j >= 0; j--) {
        table[i][j] = midA[i] === midB[j] ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1])
      }
    }
    let i = 0
    let j = 0
    while (i < n && j < m) {
      if (midA[i] === midB[j]) {
        push('same', midA[i])
        i++
        j++
      } else if (table[i + 1][j] >= table[i][j + 1]) {
        push('del', midA[i++])
      } else {
        push('add', midB[j++])
      }
    }
    while (i < n) push('del', midA[i++])
    while (j < m) push('add', midB[j++])
  }
  push('same', a.slice(endA).join(''))
  return parts
}

/** True when the diff is only additions (a Continue): then the new text can be shown alone. */
export function onlyAdditions(parts: DiffPart[]): boolean {
  return parts.every((p) => p.kind !== 'del')
}

/**
 * Shortens long unchanged stretches to their ends, so a change in a long scene shows with a little context
 * around it. A cut stretch becomes a 'same' part holding the ends and an ellipsis in the middle.
 */
export function trimContext(parts: DiffPart[], keep = 120): DiffPart[] {
  return parts.map((p, i) => {
    if (p.kind !== 'same' || p.text.length <= keep * 2 + 20) return p
    const first = i === 0
    const last = i === parts.length - 1
    const head = first ? '' : p.text.slice(0, keep).trimEnd()
    const tail = last ? '' : p.text.slice(p.text.length - keep).trimStart()
    return { kind: 'same', text: `${head}${head ? ' ' : ''}…${tail ? ' ' : ''}${tail}` }
  })
}
