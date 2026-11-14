// Line diff (Myers' algorithm) for the "what changed in this file" view.

export interface DiffLine {
  kind: 'same' | 'add' | 'del'
  text: string
  /** Line number in the old / new file. */
  oldNo?: number
  newNo?: number
}

/**
 * Gives up on a precise diff (and shows the changed part as old-then-new instead) once
 * (lines in the changed part) × (edits so far) passes this. It bounds both the time and the memory
 * the edit search takes: a few tens of milliseconds and, at worst, about 20 MB.
 */
const MAX_WORK = 5_000_000

const toLines = (text: string): string[] => (text === '' ? [] : text.replace(/\r\n/g, '\n').replace(/\n$/, '').split('\n'))

export function diffLines(before: string, after: string): DiffLine[] {
  const a = toLines(before)
  const b = toLines(after)
  const n = a.length
  const m = b.length
  // A new or emptied file: nothing to search for.
  if (!n || !m) return oldThenNew(a, b, 0, n, 0, m)
  // Lines the same at the start and the end are left out of the search.
  let start = 0
  while (start < n && start < m && a[start] === b[start]) start++
  let end = 0
  while (end < n - start && end < m - start && a[n - 1 - end] === b[m - 1 - end]) end++
  const out: DiffLine[] = []
  for (let i = 0; i < start; i++) out.push({ kind: 'same', text: a[i], oldNo: i + 1, newNo: i + 1 })
  const middle = myers(a, b, start, n - end, start, m - end) ?? oldThenNew(a, b, start, n - end, start, m - end)
  for (const l of middle) out.push(l)
  for (let i = end; i > 0; i--) out.push({ kind: 'same', text: a[n - i], oldNo: n - i + 1, newNo: m - i + 1 })
  return out
}

/** Lines aFrom..aTo of the old file removed, then lines bFrom..bTo of the new one added. */
function oldThenNew(a: string[], b: string[], aFrom: number, aTo: number, bFrom: number, bTo: number): DiffLine[] {
  const out: DiffLine[] = []
  for (let i = aFrom; i < aTo; i++) out.push({ kind: 'del', text: a[i], oldNo: i + 1 })
  for (let i = bFrom; i < bTo; i++) out.push({ kind: 'add', text: b[i], newNo: i + 1 })
  return out
}

/**
 * Myers' shortest edit script between a[aFrom..aTo) and b[bFrom..bTo), or null when it would take
 * more than MAX_WORK. Only the diagonals an edit step can reach are kept for the way back (2d + 3
 * numbers at step d, not the whole array).
 */
function myers(a: string[], b: string[], aFrom: number, aTo: number, bFrom: number, bTo: number): DiffLine[] | null {
  const n = aTo - aFrom
  const m = bTo - bFrom
  // Only lines added, or only lines removed.
  if (!n || !m) return oldThenNew(a, b, aFrom, aTo, bFrom, bTo)
  const max = n + m
  const offset = max + 1
  const v = new Int32Array(2 * max + 3)
  const trace: Int32Array[] = []
  let found = false
  for (let d = 0; d <= max; d++) {
    if (max * d > MAX_WORK) return null
    // v at diagonals -d-1..d+1, all the way back needs from this step.
    trace.push(v.slice(offset - d - 1, offset + d + 2))
    for (let k = -d; k <= d; k += 2) {
      let x = k === -d || (k !== d && v[offset + k - 1] < v[offset + k + 1]) ? v[offset + k + 1] : v[offset + k - 1] + 1
      let y = x - k
      while (x < n && y < m && a[aFrom + x] === b[bFrom + y]) {
        x++
        y++
      }
      v[offset + k] = x
      if (x >= n && y >= m) {
        found = true
        break
      }
    }
    if (found) break
  }
  if (!found) return null
  // Walk back through the trace to recover the edit script.
  const out: DiffLine[] = []
  let x = n
  let y = m
  for (let d = trace.length - 1; d >= 0; d--) {
    const vd = trace[d]
    // Diagonal k is at vd[k + d + 1].
    const at = (k: number): number => vd[k + d + 1]
    const k = x - y
    const prevK = k === -d || (k !== d && at(k - 1) < at(k + 1)) ? k + 1 : k - 1
    const prevX = at(prevK)
    const prevY = prevX - prevK
    while (x > prevX && y > prevY) {
      out.push({ kind: 'same', text: a[aFrom + x - 1], oldNo: aFrom + x, newNo: bFrom + y })
      x--
      y--
    }
    if (d > 0) {
      if (x === prevX) out.push({ kind: 'add', text: b[bFrom + y - 1], newNo: bFrom + y })
      else out.push({ kind: 'del', text: a[aFrom + x - 1], oldNo: aFrom + x })
    }
    x = prevX
    y = prevY
  }
  return out.reverse()
}

/** Keeps changed lines plus `context` lines around them; long unchanged runs become a gap marker (null). */
export function withContext(lines: DiffLine[], context = 3): (DiffLine | null)[] {
  const keep = new Uint8Array(lines.length)
  lines.forEach((l, i) => {
    if (l.kind === 'same') return
    for (let j = Math.max(0, i - context); j <= Math.min(lines.length - 1, i + context); j++) keep[j] = 1
  })
  const out: (DiffLine | null)[] = []
  let gap = false
  lines.forEach((l, i) => {
    if (keep[i]) {
      out.push(l)
      gap = false
    } else if (!gap) {
      out.push(null)
      gap = true
    }
  })
  return out
}
