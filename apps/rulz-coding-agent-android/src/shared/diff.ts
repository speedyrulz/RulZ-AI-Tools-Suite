// Line diff (Myers' algorithm) for the "what changed in this file" view.

export interface DiffLine {
  kind: 'same' | 'add' | 'del'
  text: string
  /** Line number in the old / new file. */
  oldNo?: number
  newNo?: number
}

/** Gives up on precise diffs past this many edits and shows old-then-new instead. */
const MAX_EDITS = 4000

export function diffLines(before: string, after: string): DiffLine[] {
  const a = before === '' ? [] : before.replace(/\r\n/g, '\n').replace(/\n$/, '').split('\n')
  const b = after === '' ? [] : after.replace(/\r\n/g, '\n').replace(/\n$/, '').split('\n')
  const n = a.length
  const m = b.length
  const max = n + m
  const offset = max + 1
  const v = new Int32Array(2 * max + 3)
  const trace: Int32Array[] = []
  let found = false
  for (let d = 0; d <= Math.min(max, MAX_EDITS); d++) {
    trace.push(v.slice())
    for (let k = -d; k <= d; k += 2) {
      let x = k === -d || (k !== d && v[offset + k - 1] < v[offset + k + 1]) ? v[offset + k + 1] : v[offset + k - 1] + 1
      let y = x - k
      while (x < n && y < m && a[x] === b[y]) {
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
  if (!found) {
    return [
      ...a.map((text, i) => ({ kind: 'del' as const, text, oldNo: i + 1 })),
      ...b.map((text, i) => ({ kind: 'add' as const, text, newNo: i + 1 }))
    ]
  }
  // Walk back through the trace to recover the edit script.
  const out: DiffLine[] = []
  let x = n
  let y = m
  for (let d = trace.length - 1; d >= 0; d--) {
    const vd = trace[d]
    const k = x - y
    const prevK = k === -d || (k !== d && vd[offset + k - 1] < vd[offset + k + 1]) ? k + 1 : k - 1
    const prevX = vd[offset + prevK]
    const prevY = prevX - prevK
    while (x > prevX && y > prevY) {
      out.push({ kind: 'same', text: a[x - 1], oldNo: x, newNo: y })
      x--
      y--
    }
    if (d > 0) {
      if (x === prevX) out.push({ kind: 'add', text: b[y - 1], newNo: y })
      else out.push({ kind: 'del', text: a[x - 1], oldNo: x })
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
